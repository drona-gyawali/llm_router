from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from db.models import ApiKeyModel, Base, TenantModel
from db.session import get_db
from gateway.auth import generate_api_key
from main import app

TEST_DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
USING_POSTGRES = TEST_DATABASE_URL.startswith("postgresql")

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=NullPool if USING_POSTGRES else None,
)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="module", autouse=True, loop_scope="module")
async def create_tables():
    if not USING_POSTGRES:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    if not USING_POSTGRES:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(loop_scope="module")
async def client():
    """
    AsyncClient wired to the test DB.
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
    # asyncio.Lock makes redis_eval atomic — without this, concurrent coroutines
    # all pass the `key not in redis_state` check before any writes back,
    # causing all 10 to initialise from DB spend simultaneously and all pass.
    redis_lock = asyncio.Lock()

    async def redis_get(key: str):
        return redis_state.get(key)

    async def redis_setex(key: str, _ttl: int, value: str):
        redis_state[key] = value
        return True

    async def redis_incrbyfloat(key: str, amount: float):
        current = float(redis_state.get(key, "0.0"))
        updated = current + float(amount)
        redis_state[key] = str(updated)
        return updated

    async def redis_eval(
        _script: str,
        _numkeys: int,
        key: str,
        reservation: str,  # ARGV[1]
        monthly_budget: str,  # ARGV[2]
    ):
        # The lock makes the entire check-and-increment atomic, exactly as a
        # Redis Lua script is atomic on the server.  Without it, asyncio.gather
        # interleaves coroutines at every `await`, so all 10 concurrent calls
        # race through the `key not in redis_state` guard before any write lands.
        async with redis_lock:
            if key not in redis_state:
                tenant_id_str = key.split(":")[-1]
                async with TestingSessionLocal() as db:
                    result = await db.execute(
                        select(TenantModel).where(
                            TenantModel.id == uuid.UUID(tenant_id_str)
                        )
                    )
                    tenant = result.scalar_one_or_none()
                    seed = float(tenant.current_spend) if tenant else 0.0
                redis_state[key] = str(seed)

            updated = float(redis_state[key]) + float(reservation)
            if updated > float(monthly_budget):
                return 0  # blocked

            redis_state[key] = str(updated)
            return float(reservation)  # accepted

    mock_redis = AsyncMock()
    mock_redis.get.side_effect = redis_get
    mock_redis.setex.side_effect = redis_setex
    mock_redis.incrbyfloat.side_effect = redis_incrbyfloat
    mock_redis.eval.side_effect = redis_eval

    with (
        patch("gateway.auth.CACHE_REGISTRY", mock_redis),
        patch("gateway.usage.CACHE_REGISTRY", mock_redis),
    ):
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

    sem_result = MagicMock(is_blocked=False, findings=[])

    rule_result = MagicMock()
    rule_result.rule = RoutingRule.DEFAULT
    rule_result.model = MagicMock(name="gpt-4o")

    route_result = MagicMock(model=MagicMock(name="gpt-4o"), category="general")

    app.state.semantic_guard = MagicMock(evaluate=MagicMock(return_value=sem_result))
    app.state.rule_router = MagicMock(route=MagicMock(return_value=rule_result))
    app.state.semantic_router = MagicMock(route=MagicMock(return_value=route_result))

    proxy_resp = MagicMock(
        content="ok",
        model_used="gpt-4o",
        input_tokens=5,
        output_tokens=5,
    )
    app.state.proxy = AsyncMock()
    app.state.proxy.forward = AsyncMock(return_value=proxy_resp)


@pytest.mark.asyncio(loop_scope="module")
class TestBudgetRaceCondition:

    async def test_already_at_limit_blocks_immediately(self, client: AsyncClient):
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

    async def test_over_limit_blocks_immediately(self, client: AsyncClient):
        """Tenant over budget must also be blocked."""
        _mock_app_state()
        _, raw_key = await seed_tenant(spend=150.0, budget=100.0)

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429

    async def test_concurrent_requests_race_condition(self, client: AsyncClient):
        """
        Fire 10 simultaneous requests against a tenant with spend=90, budget=100.
        redis_eval is protected by an asyncio.Lock so the check-and-increment is
        atomic — exactly like a Redis Lua script.  Only the first reservation that
        pushes spend past 100 is accepted; the other 9 get 0 back and are blocked.
        """
        _mock_app_state()
        _, raw_key = await seed_tenant(spend=90.0, budget=100.0)

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        headers = {"X-API-KEY": raw_key}

        with (
            patch("api.chat.record_usage", new_callable=AsyncMock),
            patch("api.chat.reconcile_budget_reservation", new_callable=AsyncMock),
            patch("api.chat.release_budget_reservation", new_callable=AsyncMock),
        ):
            responses = await asyncio.gather(
                *[
                    client.post("/chat", headers=headers, json=payload)
                    for _ in range(10)
                ]
            )

        statuses = [r.status_code for r in responses]
        passed = statuses.count(200)
        blocked = statuses.count(429)

        print(f"\n  Concurrent: {passed} passed, {blocked} blocked out of 10")

        assert passed == 1
        assert blocked == 9

    async def test_budget_reservation_reconciles_actual_usage(
        self, client: AsyncClient
    ):
        """
        A large max_tokens estimate should not permanently lock the tenant out.
        After the first request, reconcile releases (reservation - actual_cost)
        back to Redis so the second request can proceed.
        """
        _mock_app_state()
        tenant_id, raw_key = await seed_tenant(spend=90.0, budget=100.0)
        headers = {"X-API-KEY": raw_key}
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
        }

        budget_key = f"tenant:budget:spend:{tenant_id}"

        async def fake_reconcile(tenant_id, reservation, actual_cost):
            from gateway.usage import CACHE_REGISTRY

            await CACHE_REGISTRY.incrbyfloat(budget_key, -(reservation - actual_cost))

        with (
            patch("api.chat.record_usage", new_callable=AsyncMock),
            patch("api.chat.reconcile_budget_reservation", side_effect=fake_reconcile),
            patch("api.chat.release_budget_reservation", new_callable=AsyncMock),
        ):
            first = await client.post("/chat", headers=headers, json=payload)
            second = await client.post("/chat", headers=headers, json=payload)

        assert first.status_code == 200, first.json()
        assert second.status_code == 200, second.json()
