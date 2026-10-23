## incoming req -> check in redis -> false hit the db and cache the res for 10 min ->true
## -> hit the redis(1ms)

## user enter the details we provide the key and all susbequent request must be verified..

from dataclasses import dataclass
import secrets
import hashlib
import logging
from types import SimpleNamespace
from typing import Any, Optional, Protocol, Tuple, Union
import uuid
from sqlalchemy import select
from db.models import TenantModel, ApiKeyModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from config.settings import CACHE_REGISTRY
from redis import RedisError
import json

logger = logging.getLogger("uvicorn.error")


@dataclass
class CachedTenantRecord:
    id: uuid.UUID
    is_active: bool
    monthly_budget: float
    current_spend: float


class TenantRecord(Protocol):
    """Attributes shared by SQLAlchemy and cached tenant records."""

    id: uuid.UUID
    is_active: bool
    monthly_budget: float
    current_spend: float



def generate_api_key() -> tuple[str, str, str]:
    """
    Generates a brand-new virtual API key for a tenant.
    Returns: (raw_key, key_prefix, key_hash)
    - raw_key: Given to the user ONCE. Never stored.
    - key_prefix: Stored in DB so users can see which key is which in a dashboard.
    - key_hash: Stored securely in DB for authentication checks.
    """
    raw_key = f"gw_live_{secrets.token_urlsafe(32)}"
    key_prefix = raw_key[:12]
    key_hash = hash_key(raw_key)
    logger.info("[INFO] Virtual key generated")
    return raw_key, key_prefix, key_hash


def hash_key(raw_key: str) -> str:
    """
    Takes a plaintext API key and returns its secure SHA-256 cryptographic hash.
    Used both when creating keys and when verifying incoming requests.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def parse_cached_json(
    cached_string: str,
) -> tuple[
    Optional[uuid.UUID],
    Optional[bool],
    Optional[uuid.UUID],
    Optional[CachedTenantRecord],
]:
    """
    Parse cached Redis JSON and reconstruct typed tenant data.

    Returns a tuple of None values if the cached payload is invalid.
    """
    try:
        data = json.loads(cached_string)

        if not isinstance(data, dict):
            raise ValueError("Cached payload must be a JSON object")

        api_key_id_str = data.get("api_key_id")
        tenant_id_str = data.get("tenant_id")
        api_key_is_active = data.get("is_active")
        tenant_is_active = data.get("tenant_is_active")
        monthly_budget = data.get("monthly_budget")
        current_spend = data.get("current_spend")

        # Validate required fields.
        if not all(
            isinstance(value, str) and value
            for value in (api_key_id_str, tenant_id_str)
        ):
            raise ValueError("Missing or invalid UUID fields")

        if not isinstance(api_key_is_active, bool):
            raise ValueError("Invalid API key active status")

        if not isinstance(tenant_is_active, bool):
            raise ValueError("Invalid tenant active status")

        if monthly_budget is None or current_spend is None:
            raise ValueError("Missing tenant budget or spend")

        api_key_id = uuid.UUID(api_key_id_str)
        tenant_id = uuid.UUID(tenant_id_str)

        tenant_record = CachedTenantRecord(
            id=tenant_id,
            is_active=tenant_is_active,
            monthly_budget=float(monthly_budget),
            current_spend=float(current_spend),
        )

        return (
            api_key_id,
            api_key_is_active,
            tenant_id,
            tenant_record,
        )

    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
        logger.error("[CACHE PARSE ERROR]: Invalid cached payload: %s", e)
        return None, None, None, None


async def verify_hash(
    db: AsyncSession,
    raw_key: str,
) -> tuple[
    Optional[uuid.UUID],
    Optional[bool],
    Optional[uuid.UUID],
    Optional[TenantRecord],
]:
    """
    Verify an API key using Redis first and the database as a fallback.
    """
    if not raw_key:
        logger.error("[DATA ERROR]: Raw key is empty")
        return None, None, None, None

    hashed_key = hash_key(raw_key)
    cache_key = f"auth:hash:{hashed_key}"

    try:
        cached_data = await CACHE_REGISTRY.get(cache_key)

        if cached_data:
            (
                api_key_id,
                api_key_is_active,
                api_key_tenant_id,
                tenant_record,
            ) = parse_cached_json(cached_data)

            if (
                api_key_id is not None
                and api_key_is_active is True
                and api_key_tenant_id is not None
                and tenant_record is not None
                and tenant_record.is_active
            ):
                logger.info(
                    "[INFO] %s verified via cache",
                    api_key_tenant_id,
                )

                return (
                    api_key_id,
                    api_key_is_active,
                    api_key_tenant_id,
                    tenant_record,
                )

    except RedisError:
        logger.exception("[CACHE ERROR GET]: Redis lookup failed")

    except Exception:
        logger.exception("[UNEXPECTED ERROR]: Cache lookup failed")

    stmt = (
        select(
            ApiKeyModel.id,
            ApiKeyModel.is_active,
            ApiKeyModel.tenant_id,
            TenantModel,
        )
        .join(
            TenantModel,
            ApiKeyModel.tenant_id == TenantModel.id,
        )
        .where(ApiKeyModel.key_hash == hashed_key)
        .where(ApiKeyModel.is_active.is_(True))
        .where(TenantModel.is_active.is_(True))
    )

    try:
        result = await db.execute(stmt)
        row = result.first()

    except SQLAlchemyError:
        logger.exception(
            "[DATABASE ERROR]: Failed to verify the API key hash"
        )
        return None, None, None, None

    except Exception:
        logger.exception(
            "[UNEXPECTED ERROR]: Database verification failed"
        )
        return None, None, None, None

    if row is None:
        return None, None, None, None

    (
        api_key_id,
        api_key_is_active,
        api_key_tenant_id,
        tenant_record,
    ) = row

    try:
        await CACHE_REGISTRY.setex(
            cache_key,
            300,  # TTL: 5 minutes
            json.dumps(
                {
                    "api_key_id": str(api_key_id),
                    "tenant_id": str(api_key_tenant_id),
                    "is_active": api_key_is_active,
                    "monthly_budget": tenant_record.monthly_budget,
                    "current_spend": tenant_record.current_spend,
                    "tenant_is_active": tenant_record.is_active,
                }
            ),
        )

    except RedisError:
        logger.exception("[CACHE ERROR SET]: Redis write failed")

    except Exception:
        logger.exception("[UNEXPECTED ERROR]: Cache write failed")

    return (
        api_key_id,
        api_key_is_active,
        api_key_tenant_id,
        tenant_record,
    )
