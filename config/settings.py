import logging
import os
from pathlib import Path
from typing import Any

import redis.asyncio as redis
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"
logger = logging.getLogger("uvicorn.error")


class ApiSchema(BaseModel):
    """
    Everything the generic proxy needs to talk to a provider.
    Lives in models.yaml — zero provider-specific Python code needed.
    """

    # Request building
    endpoint_path: str
    auth_header: str | None = None
    auth_scheme: str | None = None
    auth_query_param: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    body_template: dict[str, Any] = Field(default_factory=dict)

    # Message formatting
    messages_key: str = "messages"
    messages_wrapper: str = "none"
    system_handling: str = "inline"
    role_map: dict[str, str] = Field(default_factory=dict)

    # Response parsing — dot-notation paths e.g. "choices.0.message.content"
    response_text_path: str
    response_model_path: str
    response_input_tokens_path: str
    response_output_tokens_path: str

    # Stream parsing
    stream_event_prefix: str = "data: "
    stream_done_signal: str | None = None
    stream_text_path: str = ""
    stream_type_field: str | None = None
    stream_type_value: str | None = None


class ModelCapabilities(BaseModel):
    max_context_tokens: int
    supports_streaming: bool
    supports_tool_calling: bool
    supports_json_mode: bool


class ModelPricing(BaseModel):
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


class ModelResilience(BaseModel):
    timeout_seconds: float = 30.0
    fallback_model: str | None = None


class ModelConfig(BaseModel):
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key_env_var: str
    api_schema: ApiSchema
    capabilities: ModelCapabilities
    pricing: ModelPricing
    resilience: ModelResilience = Field(default_factory=ModelResilience)


class LongContextRule(BaseModel):
    token_threshold: int
    model: str


class SimpleRule(BaseModel):
    model: str


class RulesConfig(BaseModel):
    long_context: LongContextRule
    code: SimpleRule
    enterprise: SimpleRule
    over_budget: SimpleRule
    default: SimpleRule


class RoutingConfig(BaseModel):
    rules: RulesConfig
    categories: dict[str, str]
    semantic_fallback: str


class GatewayConfig(BaseModel):
    models: list[ModelConfig]
    routing: RoutingConfig


def load_config() -> tuple[dict[str, ModelConfig], RoutingConfig]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("[FATAL] models.yaml not found at %s", CONFIG_PATH)
        raise

    gateway = GatewayConfig(**raw)
    registry: dict[str, ModelConfig] = {m.name: m for m in gateway.models}

    _validate_routing_references(gateway.routing, registry)

    logger.info(
        "Config loaded: %d models, %d routing categories",
        len(registry),
        len(gateway.routing.categories),
    )
    return registry, gateway.routing


def _validate_routing_references(
    routing: RoutingConfig,
    registry: dict[str, ModelConfig],
) -> None:
    errors: list[str] = []

    rule_refs = {
        "rules.long_context": routing.rules.long_context.model,
        "rules.code": routing.rules.code.model,
        "rules.enterprise": routing.rules.enterprise.model,
        "rules.over_budget": routing.rules.over_budget.model,
        "rules.default": routing.rules.default.model,
        "semantic_fallback": routing.semantic_fallback,
    }
    for path, model_name in rule_refs.items():
        if model_name and model_name not in registry:
            errors.append(f"  routing.{path} → '{model_name}' not in models")

    for category, model_name in routing.categories.items():
        if model_name not in registry:
            errors.append(
                f"  routing.categories.{category} → '{model_name}' not in models"
            )

    if errors:
        raise ValueError(
            "models.yaml has broken routing references:\n" + "\n".join(errors)
        )


MODEL_REGISTRY, ROUTING_CONFIG = load_config()


class EnvConfig(BaseModel):
    DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"
    WEB_PORT: str
    REDIS_HOST: str
    REDIS_PORT: int


def clean_env(value: str | None, prefix: str, default: str = "") -> str | None:
    if not value:
        return default
    # Strip spaces, remove the "VARIABLE_NAME=" prefix if present, and strip again
    return value.strip().removeprefix(f"{prefix}=").strip()


def import_env_config() -> dict[str, EnvConfig]:

    config_data = {
        "DATABASE_URL": clean_env(
            os.getenv("DATABASE_URL"),
            "DATABASE_URL",
            default="sqlite+aiosqlite:///:memory:",
        ),
        "WEB_PORT": clean_env(os.getenv("WEB_PORT"), "WEB_PORT"),
        "REDIS_HOST": clean_env(os.getenv("REDIS_HOST"), "REDIS_HOST"),
        "REDIS_PORT": clean_env(os.getenv("REDIS_PORT"), "REDIS_PORT"),
    }

    try:
        parsed_config = EnvConfig(**config_data)
        return {"env": parsed_config}
    except ValidationError as validation_error:
        logger.critical(
            "[FATAL ERROR]: Environment configuration is missing",
            validation_error.json(indent=2),
        )
        raise validation_error


CONFIG_REGISTRY = import_env_config()

# use case : CONFIG_REGISTRY["env"].DATABASE_URL


def init_redis():

    redis_client = redis.Redis(
        host=CONFIG_REGISTRY["env"].REDIS_HOST,
        port=CONFIG_REGISTRY["env"].REDIS_PORT,
        decode_responses=True,
    )

    return redis_client


CACHE_REGISTRY = init_redis()
