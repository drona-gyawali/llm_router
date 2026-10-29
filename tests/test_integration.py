from __future__ import annotations

import os
import uuid
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch


import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Project imports ────────────────────────────────────────────────────────────
from db.models import Base
from db.session import get_db
from main import app

# ── Test database ──────────────────────────────────────────────────────────────
# Uses Postgres in CI (DATABASE_URL set), SQLite locally as fallback.
TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite+aiosqlite:///:memory:"
)
USING_POSTGRES = TEST_DATABASE_URL.startswith("postgresql")

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)



@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    """Create all tables once for the test session."""
    if not USING_POSTGRES:
        # Postgres uses alembic migrations; SQLite needs explicit create
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    if not USING_POSTGRES:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture()
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an AsyncClient wired to the test DB.
    Each test gets its own session to avoid asyncpg concurrency issues.
    Redis is mocked out so no real Redis is required.
    """

    async def override_get_db():
        # Fresh session per request — required for Postgres/asyncpg
        async with TestingSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True

    with patch("gateway.auth.CACHE_REGISTRY", mock_redis):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()



async def create_tenant_and_key(
    client: AsyncClient,
    name: str = "Acme Corp",
    budget: float = 100.0,
) -> tuple[dict, str]:
    """
    Creates a tenant and one API key; returns (tenant_json, raw_key).
    """
    tenant_resp = await client.post(
        "/auth/tenants",
        json={"name": name, "monthly_budget": budget, "currency": "USD"},
    )
    assert tenant_resp.status_code == 201, tenant_resp.text
    tenant = tenant_resp.json()

    key_resp = await client.post(
        f"/auth/tenants/{tenant['id']}/keys",
        json={"name": "test-key"},
    )
    assert key_resp.status_code == 201, key_resp.text
    raw_key = key_resp.json()["raw_key"]

    return tenant, raw_key

class TestCreateTenant:
    async def test_create_tenant_success(self, client: AsyncClient):
        resp = await client.post(
            "/auth/tenants",
            json={"name": "Test Org", "monthly_budget": 50.0, "currency": "EUR"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Org"
        assert data["currency"] == "EUR"
        assert data["monthly_budget"] == 50.0
        assert data["current_spend"] == 0.0
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data

    async def test_create_tenant_currency_uppercased(self, client: AsyncClient):
        resp = await client.post(
            "/auth/tenants",
            json={"name": "Lower Org", "monthly_budget": 10.0, "currency": "usd"},
        )
        assert resp.status_code == 201
        assert resp.json()["currency"] == "USD"

    async def test_create_tenant_duplicate_name_returns_409(self, client: AsyncClient):
        payload = {"name": "Duplicate Inc", "monthly_budget": 20.0, "currency": "USD"}
        await client.post("/auth/tenants", json=payload)
        resp = await client.post("/auth/tenants", json=payload)
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    async def test_create_tenant_missing_name_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/auth/tenants", json={"monthly_budget": 10.0, "currency": "USD"}
        )
        assert resp.status_code == 422

    async def test_create_tenant_zero_budget_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/auth/tenants",
            json={"name": "Zero Budget", "monthly_budget": 0.0, "currency": "USD"},
        )
        assert resp.status_code == 422

    async def test_create_tenant_negative_budget_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/auth/tenants",
            json={"name": "Negative Budget", "monthly_budget": -5.0, "currency": "USD"},
        )
        assert resp.status_code == 422


class TestGetTenant:
    async def test_get_tenant_success(self, client: AsyncClient):
        create_resp = await client.post(
            "/auth/tenants",
            json={"name": "Readable Org", "monthly_budget": 30.0, "currency": "USD"},
        )
        tenant_id = create_resp.json()["id"]

        resp = await client.get(f"/auth/tenants/{tenant_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == tenant_id

    async def test_get_tenant_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/auth/tenants/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_get_tenant_invalid_uuid_returns_422(self, client: AsyncClient):
        resp = await client.get("/auth/tenants/not-a-uuid")
        assert resp.status_code == 422


class TestCreateApiKey:
    async def test_create_key_returns_raw_key_once(self, client: AsyncClient):
        tenant, _ = await create_tenant_and_key(client, name="Key Owner")
        # The raw_key must start with the gateway prefix
        key_resp = await client.post(
            f"/auth/tenants/{tenant['id']}/keys",
            json={"name": "second-key"},
        )
        data = key_resp.json()
        assert data["raw_key"].startswith("gw_live_")
        assert data["key_prefix"] == data["raw_key"][:12]
        assert "message" in data       # "Save this key now" notice

    async def test_create_key_for_unknown_tenant_returns_404(self, client: AsyncClient):
        resp = await client.post(
            f"/auth/tenants/{uuid.uuid4()}/keys",
            json={"name": "ghost-key"},
        )
        assert resp.status_code == 404

    async def test_create_key_name_optional(self, client: AsyncClient):
        create_resp = await client.post(
            "/auth/tenants",
            json={"name": "No Name Tenant", "monthly_budget": 10.0, "currency": "USD"},
        )
        tenant_id = create_resp.json()["id"]
        resp = await client.post(f"/auth/tenants/{tenant_id}/keys", json={})
        assert resp.status_code == 201
        assert resp.json()["name"] is None


class TestListApiKeys:
    async def test_list_keys_empty_on_new_tenant(self, client: AsyncClient):
        create_resp = await client.post(
            "/auth/tenants",
            json={"name": "Empty Keys Tenant", "monthly_budget": 10.0, "currency": "USD"},
        )
        tenant_id = create_resp.json()["id"]

        resp = await client.get(f"/auth/tenants/{tenant_id}/keys")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_keys_returns_created_keys(self, client: AsyncClient):
        create_resp = await client.post(
            "/auth/tenants",
            json={"name": "Multi Key Tenant", "monthly_budget": 10.0, "currency": "USD"},
        )
        tenant_id = create_resp.json()["id"]

        for i in range(3):
            await client.post(
                f"/auth/tenants/{tenant_id}/keys", json={"name": f"key-{i}"}
            )

        resp = await client.get(f"/auth/tenants/{tenant_id}/keys")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    async def test_list_keys_does_not_expose_raw_key(self, client: AsyncClient):
        tenant, _ = await create_tenant_and_key(client, name="Safe Keys Tenant")
        resp = await client.get(f"/auth/tenants/{tenant['id']}/keys")
        for key in resp.json():
            assert "raw_key" not in key
            assert "key_hash" not in key

    async def test_list_keys_unknown_tenant_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/auth/tenants/{uuid.uuid4()}/keys")
        assert resp.status_code == 404


class TestRevokeApiKey:
    async def test_revoke_key_success(self, client: AsyncClient):
        tenant, _ = await create_tenant_and_key(client, name="Revoke Tenant")
        keys_resp = await client.get(f"/auth/tenants/{tenant['id']}/keys")
        key_id = keys_resp.json()[0]["id"]

        resp = await client.delete(f"/auth/keys/{key_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["revoked"] is True
        assert data["id"] == key_id

    async def test_revoked_key_appears_inactive_in_list(self, client: AsyncClient):
        tenant, _ = await create_tenant_and_key(client, name="Inactive Key Tenant")
        keys_resp = await client.get(f"/auth/tenants/{tenant['id']}/keys")
        key_id = keys_resp.json()[0]["id"]

        await client.delete(f"/auth/keys/{key_id}")

        keys_after = await client.get(f"/auth/tenants/{tenant['id']}/keys")
        key = next(k for k in keys_after.json() if k["id"] == key_id)
        assert key["is_active"] is False

    async def test_revoke_unknown_key_returns_404(self, client: AsyncClient):
        resp = await client.delete(f"/auth/keys/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestAuthMiddleware:
    """
    These tests hit /chat which is gated by verify_api_key.
    We mock out app_state (guardrails / router / proxy) so the test
    focuses purely on the auth layer.
    """

    def _mock_app_state(self, app):
        """Attach minimal fake app state so the chat handler doesn't crash."""
        det_result = MagicMock(is_blocked=False, content="hello", findings=[])
        sem_result = MagicMock(is_blocked=False, findings=[])
        rule_result = MagicMock()
        rule_result.rule.value = "RULE_A"
        route_result = MagicMock(model=MagicMock(name="gpt-4o"), category="general")

        app.state.semantic_guard = MagicMock(evaluate=MagicMock(return_value=sem_result))
        app.state.rule_router = MagicMock(route=MagicMock(return_value=rule_result))
        app.state.semantic_router = MagicMock(route=MagicMock(return_value=route_result))

        proxy_resp = MagicMock(
            content="Hello!", model_used="gpt-4o",
            input_tokens=5, output_tokens=10,
        )
        app.state.proxy = AsyncMock()
        app.state.proxy.forward = AsyncMock(return_value=proxy_resp)

    async def test_valid_key_is_accepted(self, client: AsyncClient):
        self._mock_app_state(app)
        _, raw_key = await create_tenant_and_key(client, name="Valid Auth Tenant")

        with patch("gateway.usage.record_usage", new_callable=AsyncMock):
            resp = await client.post(
                "/chat",
                headers={"X-API-KEY": raw_key},
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 200

    async def test_missing_key_returns_403(self, client: AsyncClient):
        resp = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code in (401, 403, 422)

    async def test_invalid_key_returns_403(self, client: AsyncClient):
        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": "gw_live_totally_fake_key_xyz"},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 403

    async def test_revoked_key_returns_403(self, client: AsyncClient):
        tenant, raw_key = await create_tenant_and_key(client, name="Revoked Auth Tenant")
        keys_resp = await client.get(f"/auth/tenants/{tenant['id']}/keys")
        key_id = keys_resp.json()[0]["id"]
        await client.delete(f"/auth/keys/{key_id}")

        resp = await client.post(
            "/chat",
            headers={"X-API-KEY": raw_key},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 403

    async def test_budget_exceeded_returns_429(self, client: AsyncClient):
        """
        Create a tenant whose current_spend already equals monthly_budget.
        We do this by patching verify_hash to return an over-budget tenant record.
        """
        from gateway.auth import CachedTenantRecord

        over_budget_record = CachedTenantRecord(
            id=uuid.uuid4(),
            is_active=True,
            monthly_budget=10.0,
            current_spend=10.0,   # exactly at limit
        )

        key_id = uuid.uuid4()
        tenant_id = over_budget_record.id

        with patch(
            "gateway.middleware.verify_hash",
            new_callable=AsyncMock,
            return_value=(key_id, True, tenant_id, over_budget_record),
        ):
            resp = await client.post(
                "/chat",
                headers={"X-API-KEY": "gw_live_any_key"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        assert resp.status_code == 429
        assert "budget" in resp.json()["detail"].lower()


class TestHealth:
    async def test_health_returns_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}