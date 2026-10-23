from __future__ import annotations

"""
Generic LLM Proxy Engine
─────────────────────────
One engine handles every provider.
All provider-specific behaviour comes from api_schema in models.yaml.

To add a new provider:
    1. Add model entry with api_schema in models.yaml
    2. Zero Python changes needed here

DRY principle enforced:
    Before: 9 functions (3 per provider × 3 providers)
    After:  3 generic functions that read behaviour from config
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from config.settings import MODEL_REGISTRY, ModelConfig, ApiSchema

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS: int = 1024


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ProxyRequest:
    messages: list[Message]
    model: ModelConfig
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = True
    temperature: float = 0.7
    system_prompt: str | None = None


@dataclass
class ProxyResponse:
    content: str
    model_used: str
    input_tokens: int
    output_tokens: int
    was_fallback: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _resolve_path(data: dict, path: str) -> Any:
    """
    Traverse a dot-notation path through nested dicts/lists.

    Examples:
        "content.0.text"                    → data["content"][0]["text"]
        "choices.0.message.content"         → data["choices"][0]["message"]["content"]
        "usageMetadata.promptTokenCount"    → data["usageMetadata"]["promptTokenCount"]

    Returns None if any key or index is missing.
    """
    current = data
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


_TEMPLATE_VARS = {
    "{{model_id}}",
    "{{messages}}",
    "{{max_tokens}}",
    "{{temperature}}",
    "{{stream}}",
    "{{system}}",
}

def _render_template(
    template: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    Walk body_template recursively, replacing {{var}} strings with runtime values.
    Keys whose rendered value is None are dropped from the output.

    Supports nested dicts (e.g. Google's generationConfig).
    """
    result: dict[str, Any] = {}

    for key, value in template.items():
        if isinstance(value, dict):
            # Recurse into nested dicts
            nested = _render_template(value, values)
            if nested:
                result[key] = nested

        elif isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            var_name = value[2:-2]
            rendered = values.get(var_name)
            if rendered is not None:
                result[key] = rendered

        else:
            result[key] = value

    return result


def _format_messages(
    messages: list[Message],
    system_prompt: str | None,
    schema: ApiSchema,
) -> list[dict]:
    """
    Format messages list according to the provider's api_schema.

    Handles three system prompt strategies:
        inline  → system goes inside messages array as {role: "system", content: ...}
        field   → system extracted, put in body_template.{{system}} separately
        prefix  → system prepended as first user message

    Handles two message wrapper formats:
        none    → {"role": "user", "content": "..."}
        parts   → {"role": "user", "parts": [{"text": "..."}]}
    """
    formatted: list[dict] = []

    # Handle system prompt
    if system_prompt:
        if schema.system_handling == "inline":
            system_role = schema.role_map.get("system", "system")
            msg = {"role": system_role, "content": system_prompt}
            if schema.messages_wrapper == "parts":
                msg = {"role": system_role, "parts": [{"text": system_prompt}]}
            formatted.append(msg)

        elif schema.system_handling == "prefix":
            # Prepend system content to first user message
            prefix = f"[System instructions]: {system_prompt}\n\n"
            prefixed_messages = list(messages)
            if prefixed_messages and prefixed_messages[0].role == "user":
                prefixed_messages[0] = Message(
                    role="user",
                    content=prefix + prefixed_messages[0].content,
                )
            else:
                # No user message to prepend to — add as user message
                prefixed_messages.insert(
                    0, Message(role="user", content=prefix)
                )
            messages = prefixed_messages

        # "field" → system goes into body via {{system}} template var, not here

    # Format each message
    for msg in messages:
        role = schema.role_map.get(msg.role, msg.role)

        if schema.messages_wrapper == "parts":
            formatted.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })
        else:
            formatted.append({
                "role": role,
                "content": msg.content,
            })

    return formatted



def _build_request(
    req: ProxyRequest,
    model: ModelConfig,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """
    Build (url, headers, body) for any provider using api_schema from models.yaml.
    """
    schema = model.api_schema
    api_key = _get_api_key(model)

    path = schema.endpoint_path.replace("{{model_id}}", model.model_id)
    url = model.base_url.rstrip("/") + path
    if schema.auth_query_param:
        url = f"{url}?{schema.auth_query_param}={api_key}"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    headers.update(schema.extra_headers)

    if schema.auth_header:
        key_value = f"{schema.auth_scheme} {api_key}" if schema.auth_scheme else api_key
        headers[schema.auth_header] = key_value

    formatted_messages = _format_messages(
        req.messages, req.system_prompt, schema
    )

    template_values: dict[str, Any] = {
        "model_id":    model.model_id,
        "messages":    formatted_messages,
        "max_tokens":  req.max_tokens,
        "temperature": req.temperature,
        "stream":      req.stream,
        "system":      req.system_prompt,
    }

    body = _render_template(schema.body_template, template_values)

    logger.debug(
        "Built request  provider=%s  url=%s  messages=%d",
        model.provider, url, len(formatted_messages),
    )
    return url, headers, body


def _parse_response(data: dict, model: ModelConfig) -> ProxyResponse:
    """
    Parse upstream JSON response into ProxyResponse using api_schema paths.
    Replaces _parse_openai_response + _parse_anthropic_response + _parse_google_response.
    """
    schema = model.api_schema

    content = _resolve_path(data, schema.response_text_path) or ""
    model_used = _resolve_path(data, schema.response_model_path) or model.model_id
    input_tokens = int(_resolve_path(data, schema.response_input_tokens_path) or 0)
    output_tokens = int(_resolve_path(data, schema.response_output_tokens_path) or 0)

    return ProxyResponse(
        content=str(content),
        model_used=str(model_used),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _parse_stream(
    response: httpx.Response,
    model: ModelConfig,
) -> AsyncGenerator[str, None]:
    """
    Parse SSE stream from any provider using api_schema stream config.
    Replaces _stream_openai + _stream_anthropic + _stream_google.
    """
    schema = model.api_schema
    prefix = schema.stream_event_prefix
    done_signal = schema.stream_done_signal

    async for line in response.aiter_lines():
        if not line.startswith(prefix):
            continue

        payload = line[len(prefix):]

        if done_signal and payload == done_signal:
            break

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # Optional type filtering (e.g. Anthropic content_block_delta)
        if schema.stream_type_field:
            event_type = _resolve_path(data, schema.stream_type_field)
            if event_type != schema.stream_type_value:
                continue

        text = _resolve_path(data, schema.stream_text_path)
        if text:
            yield str(text)


def _get_api_key(model: ModelConfig) -> str:
    key = os.environ.get(model.api_key_env_var)
    if not key:
        raise RuntimeError(
            f"Environment variable '{model.api_key_env_var}' is not set "
            f"for model '{model.name}'."
        )
    return key


class LLMProxy:
    """
    Generic LLM proxy. One engine, all providers.

    Provider behaviour comes entirely from api_schema in models.yaml.
    Adding a new provider requires zero changes to this class.

    Handles:
    ─────────
    - Request building    (generic, schema-driven)
    - Response parsing    (generic, schema-driven)
    - Streaming           (generic, schema-driven)
    - Timeout enforcement (from ModelConfig.resilience.timeout_seconds)
    - Automatic fallback  (from ModelConfig.resilience.fallback_model)
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=None)
        logger.info("LLMProxy initialised (generic schema-driven engine).")

    async def close(self) -> None:
        await self._client.aclose()

    async def forward(self, req: ProxyRequest) -> ProxyResponse:
        """Send request, return complete response. Auto-fallback on failure."""
        req.stream = False
        try:
            return await self._call(req, req.model)
        except Exception as exc:
            logger.warning(
                "Primary model '%s' failed: %s — trying fallback.",
                req.model.name, exc,
            )
            return await self._fallback(req, exc)

    async def _call(
        self, req: ProxyRequest, model: ModelConfig
    ) -> ProxyResponse:
        """Single upstream call using generic request builder."""
        url, headers, body = _build_request(req, model)
        timeout = model.resilience.timeout_seconds

        logger.info(
            "→ %s  model=%s  timeout=%.1fs",
            model.provider, model.model_id, timeout,
        )

        response = await self._client.post(
            url, headers=headers, json=body, timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"{model.provider} returned {response.status_code}: "
                f"{response.text[:300]}"
            )

        result = _parse_response(response.json(), model)
        result.was_fallback = (model.name != req.model.name)

        logger.info(
            "← %s  in=%d  out=%d  fallback=%s",
            result.model_used,
            result.input_tokens,
            result.output_tokens,
            result.was_fallback,
        )
        return result

    async def _fallback(
        self, req: ProxyRequest, original_exc: Exception
    ) -> ProxyResponse:
        fallback_name = req.model.resilience.fallback_model
        if not fallback_name:
            raise RuntimeError(
                f"Model '{req.model.name}' failed and has no fallback configured."
            ) from original_exc

        fallback_model = MODEL_REGISTRY.get(fallback_name)
        if not fallback_model:
            raise RuntimeError(
                f"Fallback model '{fallback_name}' not in MODEL_REGISTRY."
            ) from original_exc

        logger.warning("Falling back to '%s'.", fallback_name)
        try:
            return await self._call(req, fallback_model)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both '{req.model.name}' and fallback '{fallback_name}' failed."
            ) from fallback_exc

    async def stream(self, req: ProxyRequest) -> AsyncGenerator[str, None]:
        """
        Stream response chunks. Falls back to non-streaming on failure.
        Yields plain text chunks — caller wraps in SSE format.
        """
        req.stream = True
        url, headers, body = _build_request(req, req.model)
        timeout = req.model.resilience.timeout_seconds

        logger.info(
            "→ stream  %s  model=%s  timeout=%.1fs",
            req.model.provider, req.model.model_id, timeout,
        )

        try:
            async with self._client.stream(
                "POST", url,
                headers=headers,
                json=body,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"{req.model.provider} returned {response.status_code}"
                    )
                async for chunk in _parse_stream(response, req.model):
                    yield chunk

        except Exception as exc:
            logger.warning(
                "Streaming from '%s' failed: %s — falling back.",
                req.model.name, exc,
            )
            fallback_response = await self._fallback(req, exc)
            yield fallback_response.content


    def estimate_cost(self, text: str, model: ModelConfig) -> float:
        """Estimate input cost in USD. Uses 4 chars ≈ 1 token heuristic."""
        tokens = len(text) // 4
        return (tokens / 1000) * model.pricing.input_cost_per_1k_tokens