from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from api.auth import router as auth_router
from api.chat import router as chat_router
from guardrails.sematntic_guard import SemanticGuard
from proxy.proxy import LLMProxy
from routing.router import RuleBasedRouter
from routing.semantic_router import SemanticRouter

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Starting LLM Router...")

    app.state.proxy           = LLMProxy()
    app.state.rule_router     = RuleBasedRouter()
    app.state.semantic_router = SemanticRouter()
    app.state.semantic_guard  = SemanticGuard()

    logger.info("LLM Router ready.")
    yield

    await app.state.proxy.close()
    logger.info("LLM Router shut down.")


app = FastAPI(
    title="LLM Router",
    description="Multi-provider LLM gateway with guardrails and semantic routing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)