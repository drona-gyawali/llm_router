from __future__ import annotations

"""
Usage Tracker
─────────────
Calculates actual cost from LLM response token counts and writes
it back to TenantModel.current_spend in the database.
"""

import logging
import uuid

from sqlalchemy import select
from redis import RedisError

from db.models import TenantModel
from config.settings import ModelConfig
from config.settings import CACHE_REGISTRY
from db.session import AsyncSessionLocal

logger = logging.getLogger("uvicorn.error")


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: ModelConfig,
) -> float:
    """
    Calculate actual USD cost from real token counts returned by upstream.
    """
    try:
        raw_input_rate = model.pricing.input_cost_per_1k_tokens
        raw_output_rate = model.pricing.output_cost_per_1k_tokens
    except AttributeError:
        return 0.0

    if not isinstance(raw_input_rate, (int, float)) or not isinstance(raw_output_rate, (int, float)):
        return 0.0

    input_rate = float(raw_input_rate)
    output_rate = float(raw_output_rate)

    input_cost = (input_tokens / 1000) * input_rate
    output_cost = (output_tokens / 1000) * output_rate
    total = float(input_cost + output_cost)

    logger.debug(
        "cost  model=%s  in=%d($%.6f)  out=%d($%.6f)  total=$%.6f",
        model.name,
        input_tokens, input_cost,
        output_tokens, output_cost,
        total,
    )
    return total


async def record_usage(
    tenant_id: uuid.UUID,
    input_tokens: int,
    output_tokens: int,
    model: ModelConfig,
) -> None:
    cost = calculate_cost(input_tokens, output_tokens, model)
    if cost == 0.0:
        return

    async with AsyncSessionLocal() as db:
        try:
            # SELECT FOR UPDATE locks the row until this transaction commits.
            # Any concurrent call hitting the same tenant_id will wait here
            # rather than reading a stale spend value.
            result = await db.execute(
                select(TenantModel)
                .where(TenantModel.id == tenant_id)
                .with_for_update()
            )
            tenant = result.scalar_one_or_none()

            if not tenant:
                logger.error(
                    "Usage record failed — tenant %s not found.", tenant_id
                )
                return

            tenant.current_spend = round(tenant.current_spend + cost, 8)
            await db.commit()

            logger.info(
                "Usage recorded  tenant=%s  model=%s  cost=$%.6f  total=$%.4f",
                tenant_id, model.name, cost, tenant.current_spend,
            )

        except Exception:
            await db.rollback()
            logger.exception("Failed to record usage for tenant %s", tenant_id)


def estimate_request_cost(
    input_text: str,
    max_tokens: int,
    model: ModelConfig,
) -> float:
    estimated_input_tokens = max(len(input_text) // 4, 0)
    estimated_output_tokens = max(max_tokens, 0)
    return calculate_cost(estimated_input_tokens, estimated_output_tokens, model)


def _budget_key(tenant_id: uuid.UUID) -> str:
    return f"tenant:budget:spend:{tenant_id}"


async def reserve_budget(
    tenant_id: uuid.UUID,
    current_spend: float,
    monthly_budget: float,
    estimated_cost: float,
) -> float:
    remaining = max(monthly_budget - current_spend, 0.0)
    if remaining <= 0.0:
        return 0.0

    reservation = estimated_cost if estimated_cost > 0.0 else remaining
    reservation = min(reservation, remaining)
    if reservation <= 0.0:
        return 0.0

    budget_key = _budget_key(tenant_id)

    try:
        await CACHE_REGISTRY.setnx(budget_key, current_spend)
        new_spend = await CACHE_REGISTRY.incrbyfloat(budget_key, reservation)

        if float(new_spend) > monthly_budget:
            await CACHE_REGISTRY.incrbyfloat(budget_key, -reservation)
            return 0.0
        return reservation
    except RedisError:
        logger.exception("Redis budget reservation failed for tenant %s", tenant_id)
        return 0.0


async def release_budget_reservation(
    tenant_id: uuid.UUID,
    reservation: float,
) -> None:
    if reservation <= 0.0:
        return

    try:
        await CACHE_REGISTRY.incrbyfloat(_budget_key(tenant_id), -reservation)
    except RedisError:
        logger.exception("Redis budget release failed for tenant %s", tenant_id)
