from __future__ import annotations

"""
Usage Tracker
─────────────
Calculates actual cost from LLM response token counts and writes
it back to TenantModel.current_spend in the database.

Called AFTER the LLM responds — never before.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import TenantModel
from config.settings import ModelConfig
from db.session import AsyncSessionLocal

logger = logging.getLogger("uvicorn.error")


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: ModelConfig,
) -> float:
    """
    Calculate actual USD cost from real token counts returned by upstream.

    Uses pricing from ModelConfig (models.yaml) so changing pricing
    requires only a yaml edit.
    """
    input_cost = (input_tokens / 1000) * model.pricing.input_cost_per_1k_tokens
    output_cost = (output_tokens / 1000) * model.pricing.output_cost_per_1k_tokens
    total = input_cost + output_cost

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
    async with AsyncSessionLocal() as db:
        cost = calculate_cost(input_tokens, output_tokens, model)
        if cost == 0.0:
            return

        tenant = await db.get(TenantModel, tenant_id)
        if not tenant:
            logger.error("Usage record failed — tenant %s not found.", tenant_id)
            return

        tenant.current_spend = round(tenant.current_spend + cost, 8)
        try:
            await db.commit()
            logger.info(
                "Usage recorded  tenant=%s  model=%s  cost=$%.6f  total=$%.4f",
                tenant_id, model.name, cost, tenant.current_spend,
            )
        except Exception:
            await db.rollback()
            logger.exception("Failed to record usage for tenant %s", tenant_id)