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

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def create_tables():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture()
async def db():
    async with TestingSessionLocal() as session:
        yield session


@pytest_asyncio.fixture()
async def client(db: AsyncSession):
    """
    AsyncClient wired to SQLite.
    Patches both get_db AND db.session.engine so no Postgres connection
    is ever attempted.
    """
    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    # Patch the module-level engine in db.session to prevent Postgres init
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True

    with patch("db.session.engine", test_engine), \
         patch("gateway.auth.CACHE_REGISTRY", mock_redis):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


# ── Seed helpers ───────────────────────────────────────────────────────────────

async def seed_tenant(
    db: AsyncSession,
    spend: float = 90.0,
    budget: float = 100.0,
) -> tuple[uuid.UUID, str]:
    """Insert a tenant at given spend/budget and one active API key."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBudgetRaceCondition:

    async def test_already_at_limit_blocks_immediately(
        self, client: AsyncClient, db: AsyncSession
    ):
        """Tenant at exactly the budget limit must be blocked on first request."""
        _, raw_key = await seed_tenant(db, spend=100.0, budget=100.0)

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429
        assert "budget" in resp.json()["detail"].lower()

    async def test_over_limit_blocks_immediately(
        self, client: AsyncClient, db: AsyncSession
    ):
        """Tenant over budget must also be blocked."""
        _, raw_key = await seed_tenant(db, spend=150.0, budget=100.0)

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429

    async def test_concurrent_requests_race_condition(
        self, client: AsyncClient, db: AsyncSession
    ):
        """
        Fire 10 simultaneous requests against a tenant with spend=90, budget=100.

        CURRENT EXPECTED RESULT (bug present):
            All 10 pass — because verify_api_key reads spend before
            record_usage writes it back, so all 10 see spend=90 < 100.

        AFTER FIX (SELECT FOR UPDATE in record_usage):
            Only 1 should pass. Flip the assert at the bottom once fixed.
        """
        _mock_app_state()
        _, raw_key = await seed_tenant(db, spend=90.0, budget=100.0)

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        headers = {"X-API-KEY": raw_key}

        with patch("gateway.usage.record_usage", new_callable=AsyncMock):
            responses = await asyncio.gather(*[
                client.post("/chat", headers=headers, json=payload)
                for _ in range(10)
            ])

        statuses = [r.status_code for r in responses]
        passed = statuses.count(200)
        blocked = statuses.count(429)

        print(f"\n  Concurrent: {passed} passed, {blocked} blocked out of 10")

        if USING_POSTGRES:
            # SELECT FOR UPDATE works on Postgres — enforce strict budget gate
            assert passed <= 1, (
                f"Race condition on Postgres: {passed}/10 requests passed "
                f"the budget gate. Expected ≤ 1."
            )
        else:
            # SQLite ignores WITH FOR UPDATE — document the known limitation
            import warnings
            if passed == 10:
                warnings.warn(
                    "KNOWN LIMITATION: SQLite ignores SELECT FOR UPDATE. "
                    "Race condition test is informational only on SQLite. "
                    "Run against Postgres (set DATABASE_URL) for strict enforcement.",
                    stacklevel=2,
                )
            assert passed > 0, "No requests passed — something else is broken"