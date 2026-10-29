from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Final

from config.settings import MODEL_REGISTRY, ROUTING_CONFIG, ModelConfig

logger = logging.getLogger(__name__)


class RoutingRule(str, Enum):
    LONG_CONTEXT = "LONG_CONTEXT"
    CODE_DETECTED = "CODE_DETECTED"
    ENTERPRISE = "ENTERPRISE"
    OVER_BUDGET = "OVER_BUDGET"
    DEFAULT = "DEFAULT"


class UserTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


_CODE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w[\w.]+\s+import\b"),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\bconst\s+\w+\s*="),
    re.compile(r"\bclass\s+\w+[\s:(]"),
    re.compile(
        r"```(?:python|javascript|typescript|go|java|sql|bash|sh|rust|cpp|c\+\+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bSELECT\b.{0,200}\bFROM\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bpublic\s+(?:static\s+)?(?:void|int|String)\b"),
    re.compile(r"\bfunc\s+\w+\s*\("),
    re.compile(r"#!\s*/(?:usr/bin|bin)/(?:python|bash|sh)"),
]


def _estimate_tokens(text: str) -> int:
    """4 chars ≈ 1 token. Fast heuristic, not billing-accurate."""
    return len(text) // 4


def _contains_code(text: str) -> bool:
    return any(p.search(text) for p in _CODE_PATTERNS)


def _get_model(model_name: str) -> ModelConfig | None:
    """Resolve model name → ModelConfig from registry."""
    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        logger.warning("Model '%s' not found in MODEL_REGISTRY.", model_name)
    return model


@dataclass(frozen=True)
class RoutingDecision:
    id: uuid.UUID
    timestamp: datetime
    rule: RoutingRule
    model: ModelConfig | None
    estimated_tokens: int
    logs: tuple[str, ...]

    @property
    def model_name(self) -> str:
        return self.model.name if self.model else "NONE"

    @property
    def provider(self) -> str:
        return self.model.provider if self.model else "NONE"

    @property
    def routed(self) -> bool:
        return self.model is not None


def _make_decision(
    rule: RoutingRule,
    model_name: str,
    estimated_tokens: int,
    extra_logs: list[str] | None = None,
) -> RoutingDecision:
    model = _get_model(model_name)
    logs: list[str] = [
        f"Rule         : {rule.value}",
        f"Model        : {model.name if model else 'NOT FOUND'}",
        f"Provider     : {model.provider if model else 'N/A'}",
        f"Est. tokens  : {estimated_tokens:,}",
    ]
    if extra_logs:
        logs.extend(extra_logs)

    decision = RoutingDecision(
        id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        rule=rule,
        model=model,
        estimated_tokens=estimated_tokens,
        logs=tuple(logs),
    )
    logger.debug(
        "routing_decision id=%s rule=%s model=%s tokens=%d",
        decision.id,
        rule.value,
        decision.model_name,
        estimated_tokens,
    )
    return decision


class RuleBasedRouter:
    """
    Deterministic rule-based router.
    All model assignments and thresholds come from ROUTING_CONFIG (models.yaml).
    No hardcoded model names anywhere in this class.
    """

    def __init__(self) -> None:
        # Read config once at construction — if models.yaml changes,
        # restart the app to pick up changes.
        self._rules = ROUTING_CONFIG.rules
        self._token_threshold = self._rules.long_context.token_threshold

        logger.info(
            "RuleBasedRouter ready  "
            "long_context=%d tokens → %s  |  code → %s  |  "
            "enterprise → %s  |  over_budget → %s  |  default → %s",
            self._token_threshold,
            self._rules.long_context.model,
            self._rules.code.model,
            self._rules.enterprise.model,
            self._rules.over_budget.model,
            self._rules.default.model,
        )

    def _check_long_context(self, estimated_tokens: int) -> RoutingDecision | None:
        if estimated_tokens <= self._token_threshold:
            return None
        model_name = self._rules.long_context.model
        return _make_decision(
            rule=RoutingRule.LONG_CONTEXT,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            extra_logs=[
                f"Threshold    : {self._token_threshold:,} tokens",
                "Reason       : Payload exceeds standard context window.",
            ],
        )

    def _check_code(self, text: str, estimated_tokens: int) -> RoutingDecision | None:
        if not _contains_code(text):
            return None
        model_name = self._rules.code.model
        matched = [p.pattern for p in _CODE_PATTERNS if p.search(text)]
        return _make_decision(
            rule=RoutingRule.CODE_DETECTED,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            extra_logs=[
                f"Matched      : {matched[:3]}{'…' if len(matched) > 3 else ''}",
                "Reason       : Code patterns detected.",
            ],
        )

    def _check_enterprise(
        self, user_tier: UserTier, estimated_tokens: int
    ) -> RoutingDecision | None:
        if user_tier != UserTier.ENTERPRISE:
            return None
        model_name = self._rules.enterprise.model
        return _make_decision(
            rule=RoutingRule.ENTERPRISE,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            extra_logs=["Reason       : Enterprise tier → premium model."],
        )

    def _check_over_budget(
        self,
        team_daily_spend: float,
        team_daily_budget: float,
        estimated_tokens: int,
    ) -> RoutingDecision | None:
        if team_daily_spend < team_daily_budget:
            return None
        model_name = self._rules.over_budget.model
        return _make_decision(
            rule=RoutingRule.OVER_BUDGET,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            extra_logs=[
                f"Spend        : ${team_daily_spend:.4f}",
                f"Budget       : ${team_daily_budget:.4f}",
                "Reason       : Over daily budget → cheapest model.",
            ],
        )

    def _default(self, estimated_tokens: int) -> RoutingDecision:
        model_name = self._rules.default.model
        return _make_decision(
            rule=RoutingRule.DEFAULT,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            extra_logs=["Reason       : No rule matched → default model."],
        )

    def route(
        self,
        text: str,
        user_tier: UserTier = UserTier.FREE,
        team_daily_spend: float = 0.0,
        team_daily_budget: float = float("inf"),
    ) -> RoutingDecision:
        """
        Evaluate rules in priority order. First match wins.

        Priority: LONG_CONTEXT → CODE → ENTERPRISE → OVER_BUDGET → DEFAULT
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")

        estimated_tokens = _estimate_tokens(text)

        decision = (
            self._check_long_context(estimated_tokens)
            or self._check_code(text, estimated_tokens)
            or self._check_enterprise(user_tier, estimated_tokens)
            or self._check_over_budget(
                team_daily_spend, team_daily_budget, estimated_tokens
            )
            or self._default(estimated_tokens)
        )

        logger.info(
            "RuleRouter → rule=%-14s  model=%-25s  tokens=%d",
            decision.rule.value,
            decision.model_name,
            estimated_tokens,
        )
        return decision

    def estimate_cost(self, text: str, model: ModelConfig) -> float:
        """Estimate input cost in USD using pricing from ModelConfig."""
        tokens = _estimate_tokens(text)
        return (tokens / 1000) * model.pricing.input_cost_per_1k_tokens
