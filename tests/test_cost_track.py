from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch


import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from db.models import Base, TenantModel, ApiKeyModel
from db.session import get_db
from gateway.auth import generate_api_key
from main import app

# ── Test database ──────────────────────────────────────────────────────────────
# Uses Postgres in CI (DATABASE_URL set), SQLite locally as fallback.
TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite+aiosqlite:///:memory:"
)
USING_POSTGRES = TEST_DATABASE_URL.startswith("postgresql")

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=NullPool if USING_POSTGRES else None,
)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def create_tables():
    if not USING_POSTGRES:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    if not USING_POSTGRES:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture()
async def client():
    """
    AsyncClient wired to test DB.
    Each request gets its own session — required for Postgres/asyncpg.
    """
    async def override_get_db():
        async with TestingSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    redis_state: dict[str, str] = {}

    async def redis_get(key: str):
        return redis_state.get(key)

    async def redis_setex(key: str, _ttl: int, value: str):
        redis_state[key] = value
        return True

    async def redis_setnx(key: str, value: float):
        if key in redis_state:
            return False
        redis_state[key] = str(value)
        return True

    async def redis_incrbyfloat(key: str, amount: float):
        current = float(redis_state.get(key, "0.0"))
        updated = current + float(amount)
        redis_state[key] = str(updated)
        return updated

    mock_redis = AsyncMock()
    mock_redis.get.side_effect = redis_get
    mock_redis.setex.side_effect = redis_setex
    mock_redis.setnx.side_effect = redis_setnx
    mock_redis.incrbyfloat.side_effect = redis_incrbyfloat

    with patch("gateway.auth.CACHE_REGISTRY", mock_redis), patch("gateway.usage.CACHE_REGISTRY", mock_redis):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


# ── Seed helpers ───────────────────────────────────────────────────────────────

async def seed_tenant(
    spend: float = 90.0,
    budget: float = 100.0,
) -> tuple[uuid.UUID, str]:
    """Insert a tenant at given spend/budget and one active API key."""
    async with TestingSessionLocal() as db:
        tenant = TenantModel(
            id=uuid.uuid4(),
            name=f"Race Tenant {uuid.uuid4().hex[:6]}",
            currency="USD",
            monthly_budget=budget,
            current_spend=spend,
            is_active=True,
        )
        db.add(tenant)

        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = ApiKeyModel(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            name="race-key",
            is_active=True,
        )
        db.add(api_key)
        await db.commit()
    return tenant.id, raw_key


def _mock_app_state():
    """Minimal fake app state so the chat handler doesn't crash."""
    from routing.router import RoutingRule

    det_result = MagicMock(is_blocked=False, content="hello", findings=[])
    sem_result = MagicMock(is_blocked=False, findings=[])

    rule_result = MagicMock()
    rule_result.rule = RoutingRule.DEFAULT   # real enum value, no .value override
    rule_result.model = MagicMock(name="gpt-4o")

    route_result = MagicMock(
        model=MagicMock(name="gpt-4o"), category="general"
    )

    app.state.semantic_guard = MagicMock(evaluate=MagicMock(return_value=sem_result))
    app.state.rule_router = MagicMock(route=MagicMock(return_value=rule_result))
    app.state.semantic_router = MagicMock(route=MagicMock(return_value=route_result))

    proxy_resp = MagicMock(
        content="ok", model_used="gpt-4o",
        input_tokens=5, output_tokens=5,
    )
    app.state.proxy = AsyncMock()
    app.state.proxy.forward = AsyncMock(return_value=proxy_resp)


class TestBudgetRaceCondition:

    async def test_already_at_limit_blocks_immediately(
        self, client: AsyncClient
    ):
        """Tenant at exactly the budget limit must be blocked on first request."""
        _mock_app_state()
        _, raw_key = await seed_tenant(spend=100.0, budget=100.0)

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429
        assert "budget" in resp.json()["detail"].lower()

    async def test_over_limit_blocks_immediately(
        self, client: AsyncClient
    ):
        """Tenant over budget must also be blocked."""
        _mock_app_state()
        _, raw_key = await seed_tenant(spend=150.0, budget=100.0)

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429

    async def test_concurrent_requests_race_condition(
        self, client: AsyncClient
    ):
        """
        Fire 10 simultaneous requests against a tenant with spend=90, budget=100.
        Budget reservation is atomic, so only one request can consume
        the remaining budget under concurrency.
        """
        _mock_app_state()
        _, raw_key = await seed_tenant(spend=90.0, budget=100.0)

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        headers = {"X-API-KEY": raw_key}

        with patch("api.chat.record_usage", new_callable=AsyncMock):
            responses = await asyncio.gather(*[
                client.post("/chat", headers=headers, json=payload)
                for _ in range(10)
            ])

        statuses = [r.status_code for r in responses]
        passed = statuses.count(200)
        blocked = statuses.count(429)

        print(f"\n  Concurrent: {passed} passed, {blocked} blocked out of 10")

        assert passed == 1
        assert blocked == 9