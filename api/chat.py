import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from db.session import get_db
from gateway.middleware import verify_api_key
from gateway.usage import (
    record_usage,
    estimate_request_cost,
    reserve_budget,
    release_budget_reservation,
)
from guardrails.deterministic_rule import DeterministicRuleGuard
from proxy.proxy import Message, ProxyRequest
from routing.router import RoutingRule, UserTier
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("uvicorn.error")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: int = Field(default=1024, gt=0, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    content: str
    model_used: str
    category: str
    routing_rule: str
    input_tokens: int
    output_tokens: int


router = APIRouter(tags=["auth"])


@router.post(
    "/chat",
    response_model=None,
    summary="Send a message to the LLM router",
)
async def chat(
    request: Request,
    body: ChatRequest,
    auth: tuple = Depends(verify_api_key),
) -> Any:
    """
    Full pipeline:
    1. Auth already verified by verify_api_key dependency
    2. Deterministic guardrail
    3. Semantic guardrail
    4. Rule-based routing
    5. Semantic routing (if no hard rule matched)
    6. Forward to upstream LLM
    7. Record usage cost to DB
    """
    app_state = request.app.state
    api_key_id, api_key_is_active, tenant_id, tenant_record = auth

    # Build input text for guardrails and routing (use the last user message as primary signal)
    input_text = " ".join(
        m.content for m in body.messages if m.role == "user"
    )

    det_result = DeterministicRuleGuard(input_text).evaluate()

    if det_result.is_blocked:
        logger.warning(
            "Deterministic guard blocked  tenant=%s  findings=%s",
            tenant_id, det_result.findings,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Request blocked by content policy.",
                "findings": list(det_result.findings),
            },
        )

    # Use redacted content if PII was found
    safe_text = det_result.content or input_text

    sem_guard_result = app_state.semantic_guard.evaluate(safe_text)

    if sem_guard_result.is_blocked:
        logger.warning(
            "Semantic guard blocked  tenant=%s  findings=%s",
            tenant_id, sem_guard_result.findings,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Request blocked by content policy.",
                "findings": list(sem_guard_result.findings),
            },
        )

    rule_decision = app_state.rule_router.route(
        text=safe_text,
        user_tier=UserTier.FREE,
        team_daily_spend=tenant_record.current_spend,
        team_daily_budget=tenant_record.monthly_budget,
    )

    routing_rule = rule_decision.rule.value
    category = "rule_based"

    if rule_decision.rule == RoutingRule.DEFAULT:
        sem_route_result = app_state.semantic_router.route(safe_text)
        selected_model = sem_route_result.model or rule_decision.model
        category = sem_route_result.category
        routing_rule = f"SEMANTIC:{category.upper()}"
    else:
        selected_model = rule_decision.model

    if not selected_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model available for this request.",
        )

    logger.info(
        "Routing decision  tenant=%s  rule=%s  model=%s",
        tenant_id, routing_rule, selected_model.name,
    )

    proxy_req = ProxyRequest(
        messages=[Message(role=m.role, content=m.content) for m in body.messages],
        model=selected_model,
        max_tokens=body.max_tokens,
        stream=body.stream,
        temperature=body.temperature,
        system_prompt=body.system_prompt,
    )

    if body.stream:
        async def generate() -> AsyncGenerator[str, None]:
            async for chunk in app_state.proxy.stream(proxy_req):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "X-Model-Used": selected_model.name,
                "X-Routing-Rule": routing_rule,
                "X-Category": category,
            },
        )

    estimated_cost = estimate_request_cost(
        input_text=safe_text,
        max_tokens=body.max_tokens,
        model=selected_model,
    )
    reserved_amount = await reserve_budget(
        tenant_id=tenant_id,
        current_spend=tenant_record.current_spend,
        monthly_budget=tenant_record.monthly_budget,
        estimated_cost=estimated_cost,
    )
    if reserved_amount <= 0.0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly budget limit reached for organization: {tenant_record.id}.",
        )

    # Non-streaming
    try:
        response = await app_state.proxy.forward(proxy_req)
    except Exception:
        await release_budget_reservation(
            tenant_id=tenant_id,
            reservation=reserved_amount,
        )
        raise

    await record_usage(
        tenant_id=tenant_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        model=selected_model,
    )

    return ChatResponse(
        id=str(uuid.uuid4()),
        content=response.content,
        model_used=response.model_used,
        category=category,
        routing_rule=routing_rule,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}