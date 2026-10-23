from __future__ import annotations

"""
Auth Router — Tenant and API Key Management
────────────────────────────────────────────
Provides REST endpoints for:
    POST /auth/tenants          → create a new tenant
    GET  /auth/tenants/{id}     → get tenant info
    POST /auth/tenants/{id}/keys → create an API key for a tenant
    GET  /auth/tenants/{id}/keys → list API keys for a tenant
    DELETE /auth/keys/{id}      → revoke an API key

These are admin endpoints. In production, protect them with an
internal admin token. For now they are open for testing.
"""

from typing import Optional
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from db.models import TenantModel, ApiKeyModel
from db.session import get_db
from gateway.auth import generate_api_key

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/auth", tags=["auth"])


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    monthly_budget: float = Field(default=100.0, gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class TenantResponse(BaseModel):
    id: uuid.UUID
    name: str
    currency: str
    monthly_budget: float
    current_spend: float
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateApiKeyRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)


class ApiKeyCreatedResponse(BaseModel):
    """
    Returned ONCE when a key is created.
    raw_key is never stored — user must save it now.
    """
    id: uuid.UUID
    name: Optional[str]
    key_prefix: str
    raw_key: str                    # ← shown ONCE, never stored
    message: str = "Save this key now. It will never be shown again."


class ApiKeyResponse(BaseModel):
    """Safe representation — no raw key."""
    id: uuid.UUID
    name: Optional[str]
    key_prefix: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class RevokeResponse(BaseModel):
    id: uuid.UUID
    revoked: bool


@router.post(
    "/tenants",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new tenant",
)
async def create_tenant(
    body: CreateTenantRequest,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """
    Create a new tenant (organisation).
    Each tenant gets their own budget and API keys.
    """
    tenant = TenantModel(
        id=uuid.uuid4(),
        name=body.name,
        currency=body.currency.upper(),
        monthly_budget=body.monthly_budget,
        current_spend=0.0,
        is_active=True,
    )
    db.add(tenant)

    try:
        await db.commit()
        await db.refresh(tenant)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant with name '{body.name}' already exists.",
        )

    logger.info("Tenant created: %s (%s)", tenant.name, tenant.id)
    return TenantResponse.model_validate(tenant)


@router.get(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant info",
)
async def get_tenant(
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    tenant = await db.get(TenantModel, tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found.",
        )
    return TenantResponse.model_validate(tenant)


# ── API Key endpoints ─────────────────────────────────────────────────────────

@router.post(
    "/tenants/{tenant_id}/keys",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an API key for a tenant",
)
async def create_api_key(
    tenant_id: uuid.UUID,
    body: CreateApiKeyRequest,
    db: AsyncSession = Depends(get_db),
) -> ApiKeyCreatedResponse:
    """
    Generate a new API key for a tenant.
    The raw key is returned ONCE and never stored.
    The caller must save it immediately.
    """
    # Verify tenant exists and is active
    tenant = await db.get(TenantModel, tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found.",
        )
    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is inactive.",
        )

    # Generate key
    raw_key, key_prefix, key_hash = generate_api_key()

    api_key = ApiKeyModel(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        key_prefix=key_prefix,
        key_hash=key_hash,
        name=body.name,
        is_active=True,
    )
    db.add(api_key)

    try:
        await db.commit()
        await db.refresh(api_key)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Key collision, please try again.",
        )

    logger.info(
        "API key created for tenant %s  prefix=%s",
        tenant_id, key_prefix,
    )

    return ApiKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=key_prefix,
        raw_key=raw_key,
    )


@router.get(
    "/tenants/{tenant_id}/keys",
    response_model=list[ApiKeyResponse],
    summary="List API keys for a tenant",
)
async def list_api_keys(
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[ApiKeyResponse]:
    """List all API keys for a tenant. Raw keys are never returned."""
    tenant = await db.get(TenantModel, tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found.",
        )

    stmt = select(ApiKeyModel).where(ApiKeyModel.tenant_id == tenant_id)
    result = await db.execute(stmt)
    keys = result.scalars().all()

    return [ApiKeyResponse.model_validate(k) for k in keys]


@router.delete(
    "/keys/{key_id}",
    response_model=RevokeResponse,
    summary="Revoke an API key",
)
async def revoke_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RevokeResponse:
    """Revoke (deactivate) an API key. Does not delete it."""
    api_key = await db.get(ApiKeyModel, key_id)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found.",
        )

    api_key.is_active = False
    await db.commit()

    logger.info("API key revoked: %s", key_id)
    return RevokeResponse(id=key_id, revoked=True)