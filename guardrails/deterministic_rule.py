from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal

__all__ = [
    "Action",
    "DeterministicRuleGuard",
    "GuardConfig",
    "GuardResponse",
]

logger = logging.getLogger(__name__)

Action = Literal["allow", "redact", "block"]


@dataclass(frozen=True)
class GuardConfig:
    """
    Runtime configuration for DeterministicRuleGuard.

    Attributes
    ----------
    max_bytes:
        Maximum UTF-8 payload size accepted.  Payloads larger than this are
        blocked before any other rule runs.
    enable_pii:
        Run the PII-redaction guard.
    enable_sql:
        Run the SQL-injection detection guard.
    enable_prompt_injection:
        Run the prompt-injection detection guard.
    """

    max_bytes: int = 32 * 1024
    enable_pii: bool = True
    enable_sql: bool = True
    enable_prompt_injection: bool = True

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {self.max_bytes}")


_DEFAULT_CONFIG: Final[GuardConfig] = GuardConfig()


@dataclass(frozen=True)
class GuardResponse:
    """
    Immutable result returned by every guard method and by ``evaluate()``.

    Attributes
    ----------
    id:
        Unique identifier for this evaluation (useful for correlating logs).
    timestamp:
        UTC time the response was created.
    type:
        The rule category that produced this response
        (e.g. ``"PII"``, ``"SQL_INJECTION"``).  ``"PIPELINE"`` is used by
        ``evaluate()`` for its aggregated result.
    action:
        ``"allow"``  – payload is clean; ``content`` holds the original text.
        ``"redact"`` – PII was found and masked; ``content`` holds sanitized text.
        ``"block"``  – payload was rejected; ``content`` is ``None``.
    content:
        The (possibly redacted) payload, or ``None`` when the action is
        ``"block"``.  Callers should always check ``action`` before using
        ``content``.
    findings:
        Sorted tuple of matched rule names.  Empty when action is ``"allow"``.
    logs:
        Human-readable description of what matched (or didn't).
    """

    id: uuid.UUID
    timestamp: datetime
    type: str
    action: Action
    content: str | None
    findings: tuple[str, ...]
    logs: tuple[str, ...]

    @property
    def is_blocked(self) -> bool:
        return self.action == "block"

    @property
    def is_redacted(self) -> bool:
        return self.action == "redact"

    @property
    def is_clean(self) -> bool:
        return self.action == "allow"


def _make_response(
    *,
    content: str | None,
    rule_type: str,
    action: Action,
    findings: list[str],
    logs: list[str],
) -> GuardResponse:
    resp = GuardResponse(
        id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        type=rule_type,
        action=action,
        content=content,
        findings=tuple(sorted(set(findings))),
        logs=tuple(logs),
    )
    logger.debug(
        "guard_response id=%s type=%s action=%s findings=%s",
        resp.id,
        resp.type,
        resp.action,
        resp.findings,
    )
    return resp


class DeterministicRuleGuard:
    """
    Conservative, first-layer deterministic guardrail.

    Regex rules are intentionally limited in scope.  They catch common,
    well-known patterns but are **not** a complete defense against all
    variants.  Pair with a semantic (LLM-based) guard for stronger coverage.

    Parameters
    ----------
    content:
        The raw input string to evaluate.
    config:
        Optional ``GuardConfig``; falls back to module-level defaults.

    Notes
    -----
    * ``content`` is normalized to NFC unicode once at construction so that
      lookalike characters do not trivially bypass regex rules.
    * All guard methods are pure (no side-effects beyond logging); ``self``
      is effectively immutable after ``__init__``.
    * ``evaluate()`` runs the full pipeline and aggregates all findings so
      the caller always receives a complete picture.
    """

    # Compiled once at class level — shared across all instances.

    _EMAIL_RE: Final = re.compile(
        r"(?<![A-Z0-9._%+\-])"
        r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[A-Z0-9](?:[A-Z0-9\-]{0,61}[A-Z0-9])?"
        r"(?:\.[A-Z0-9](?:[A-Z0-9\-]{0,61}[A-Z0-9])?)+"
        r"(?![A-Z0-9.\-])",
        re.IGNORECASE,
    )

    _SSN_RE: Final = re.compile(
        r"(?<!\d)" r"(?!000|666|9\d\d)\d{3}" r"-(?!00)\d{2}" r"-(?!0000)\d{4}" r"(?!\d)"
    )

    # Phone: require an explicit separator or parentheses so plain digit
    # sequences (dates, IDs) are not captured.
    _PHONE_RE: Final = re.compile(
        r"(?<!\w)"
        r"(?:\+?1[\s.\-]?)?"
        r"(?:\(\d{3}\)[\s.\-]?|\d{3}[\s.\-])"
        r"\d{3}"
        r"[\s.\-]"
        r"\d{4}"
        r"(?!\w)"
    )

    _CARD_RE: Final = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")

    _MEDICAL_RE: Final = re.compile(
        r"\b("
        r"medical record number|mrn|patient id|patient identifier|"
        r"diagnosis|medical history|prescription number"
        r")\s*[:#=]\s*"
        r"([^\r\n,;]{1,120})",
        re.IGNORECASE,
    )

    _SQL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
        (
            "SQL_COMMENT_TAIL",
            re.compile(r"(?:--[ \t]|/\*.*?\*/|#\s*).*$", re.IGNORECASE | re.DOTALL),
        ),
        (
            "SQL_UNION_SELECT",
            re.compile(r"\bUNION\b\s+(?:ALL\s+)?\bSELECT\b", re.IGNORECASE),
        ),
        (
            "SQL_BOOLEAN_TAUTOLOGY",
            re.compile(
                r"""(?:['"]?\s*)\bOR\b\s+"""
                r"""(?:['"]?\w+['"]?\s*=\s*['"]?\w+['"]?"""
                r"""|['"]?\d+['"]?\s*=\s*['"]?\d+['"]?)""",
                re.IGNORECASE,
            ),
        ),
        (
            "SQL_STACKED_QUERY",
            re.compile(
                r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|"
                r"CREATE|EXEC(?:UTE)?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "SQL_DANGEROUS_STATEMENT",
            re.compile(
                r"\b(?:DROP\s+TABLE|TRUNCATE\s+TABLE|DELETE\s+FROM)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    _INJECTION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
        (
            "OVERRIDE_INSTRUCTIONS",
            re.compile(
                r"\b(?:ignore|disregard|override|forget)\b"
                r".{0,80}\b(?:previous|prior|above|system|all)\b"
                r".{0,40}\b(?:instructions?|rules?|prompts?|polic(?:y|ies))\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "REVEAL_SYSTEM_PROMPT",
            re.compile(
                r"\b(?:reveal|print|show|output|repeat|leak)\b"
                r".{0,50}\b(?:system prompt|hidden instructions|"
                r"developer message|secret instructions)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "JAILBREAK_PERSONA",
            re.compile(
                r"\b(?:DAN mode|developer mode enabled|"
                r"bypass all safety|disable safety restrictions|"
                r"ignore all safety rules)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "INSTRUCTION_EXFILTRATION",
            re.compile(
                r"\b(?:send|exfiltrate|forward|upload)\b"
                r".{0,60}\b(?:secrets?|credentials?|api\s*keys?|"
                r"system prompt|private data)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
    ]

    def __init__(
        self,
        content: str,
        config: GuardConfig = _DEFAULT_CONFIG,
    ) -> None:
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__!r}")
        # NFC normalization collapses unicode lookalikes into canonical form.
        self._payload: Final[str] = unicodedata.normalize("NFC", content)
        self._config: Final[GuardConfig] = config

    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        """Return True iff *digits* (only digit chars) passes the Luhn check."""
        if not digits.isdigit():
            raise ValueError("digits must contain only ASCII digit characters")
        n = len(digits)
        if not (13 <= n <= 19):
            return False
        total = 0
        parity = n % 2
        for i, ch in enumerate(digits):
            d = int(ch)
            if i % 2 == parity:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    @staticmethod
    def _mask_card(raw: str) -> tuple[str, bool]:
        """
        Return *(masked, was_redacted)*.

        Only masks the number if it passes the Luhn check.
        The last four digits are preserved; everything else becomes ``*``.
        """
        digits = "".join(c for c in raw if c.isdigit())
        if not DeterministicRuleGuard._luhn_valid(digits):
            return raw, False
        masked = "*" * (len(digits) - 4) + digits[-4:]
        return masked, True

    # ------------------------------------------------------------------ #
    # Individual guards (each returns a GuardResponse)                    #
    # ------------------------------------------------------------------ #

    def check_payload_size(self) -> GuardResponse:
        """Block payloads that exceed ``config.max_bytes`` bytes (UTF-8)."""
        size = len(self._payload.encode("utf-8"))
        limit = self._config.max_bytes
        if size > limit:
            return _make_response(
                content=None,
                rule_type="PAYLOAD_SIZE",
                action="block",
                findings=["PAYLOAD_TOO_LARGE"],
                logs=[f"Payload size {size} bytes exceeds limit of {limit} bytes"],
            )
        return _make_response(
            content=self._payload,
            rule_type="PAYLOAD_SIZE",
            action="allow",
            findings=[],
            logs=[f"Payload size accepted: {size} bytes"],
        )

    def check_sql_injection(self) -> GuardResponse:
        """
        Block payloads containing suspicious SQL constructs.

        Detects common patterns only; not a complete SQLi firewall.
        """
        if not self._config.enable_sql:
            return _make_response(
                content=self._payload,
                rule_type="SQL_INJECTION",
                action="allow",
                findings=[],
                logs=["SQL injection guard disabled"],
            )

        matched: list[str] = []
        for reason, pattern in self._SQL_PATTERNS:
            if pattern.search(self._payload):
                matched.append(reason)

        if matched:
            return _make_response(
                content=None,
                rule_type="SQL_INJECTION",
                action="block",
                findings=matched,
                logs=[f"Matched rules: {', '.join(matched)}"],
            )
        return _make_response(
            content=self._payload,
            rule_type="SQL_INJECTION",
            action="allow",
            findings=[],
            logs=["No SQL injection patterns matched"],
        )

    def check_prompt_injection(self) -> GuardResponse:
        """
        Block payloads containing known prompt-injection indicators.

        Not a complete defense against novel or indirect injection.
        """
        if not self._config.enable_prompt_injection:
            return _make_response(
                content=self._payload,
                rule_type="PROMPT_INJECTION",
                action="allow",
                findings=[],
                logs=["Prompt injection guard disabled"],
            )

        matched: list[str] = []
        for reason, pattern in self._INJECTION_PATTERNS:
            if pattern.search(self._payload):
                matched.append(reason)

        if matched:
            return _make_response(
                content=None,
                rule_type="PROMPT_INJECTION",
                action="block",
                findings=matched,
                logs=[f"Matched rules: {', '.join(matched)}"],
            )
        return _make_response(
            content=self._payload,
            rule_type="PROMPT_INJECTION",
            action="allow",
            findings=[],
            logs=["No prompt injection patterns matched"],
        )

    def check_pii(self) -> GuardResponse:
        """
        Redact recognizable PII from the payload and return sanitized content.

        Detects: email addresses, US SSNs, phone numbers (with separators),
        credit card numbers (Luhn-validated), and explicit medical-data labels.
        """
        if not self._config.enable_pii:
            return _make_response(
                content=self._payload,
                rule_type="PII",
                action="allow",
                findings=[],
                logs=["PII guard disabled"],
            )

        content = self._payload
        findings: list[str] = []

        # Credit cards (Luhn-validated).
        def _replace_card(m: re.Match[str]) -> str:
            masked, redacted = self._mask_card(m.group())
            if redacted:
                findings.append("CREDIT_CARD")
            return masked

        content = self._CARD_RE.sub(_replace_card, content)

        # Structured PII.
        pii_patterns: list[tuple[str, re.Pattern[str], str]] = [
            ("EMAIL", self._EMAIL_RE, "[REDACTED_EMAIL]"),
            ("US_SSN", self._SSN_RE, "[REDACTED_SSN]"),
            ("PHONE", self._PHONE_RE, "[REDACTED_PHONE]"),
        ]
        for label, pattern, replacement in pii_patterns:
            content, count = pattern.subn(replacement, content)
            if count:
                findings.append(label)

        # Medical labels.
        content, medical_count = self._MEDICAL_RE.subn(
            lambda m: f"{m.group(1)}: [REDACTED_MEDICAL_DATA]",
            content,
        )
        if medical_count:
            findings.append("MEDICAL_DATA")

        if findings:
            return _make_response(
                content=content,
                rule_type="PII",
                action="redact",
                findings=findings,
                logs=["Redacted categories: " + ", ".join(sorted(set(findings)))],
            )
        return _make_response(
            content=content,
            rule_type="PII",
            action="allow",
            findings=[],
            logs=["No PII patterns matched"],
        )

    def evaluate(self) -> GuardResponse:
        """
        Run the full guard pipeline and return an aggregated result.

        Pipeline
        --------
        1. Payload size  → block immediately if over limit
        2. SQL injection → block, but continue to collect other findings
        3. Prompt injection → block, but continue
        4. PII → redact (or allow)

        The pipeline always runs all enabled guards so the caller receives
        the complete set of ``findings`` even when the ``action`` is
        ``"block"``.  PII is redacted in the sanitized ``content``
        regardless of whether other guards fired.

        Returns
        -------
        GuardResponse
            ``action`` is the most severe result across all guards
            (block > redact > allow).  ``content`` is ``None`` on block,
            otherwise the PII-sanitized payload.
        """
        all_findings: list[str] = []
        all_logs: list[str] = []
        blocked = False

        # --- Size --------------------------------------------------------
        size_result = self.check_payload_size()
        all_findings.extend(size_result.findings)
        all_logs.extend(size_result.logs)
        if size_result.is_blocked:
            # Nothing more to do, we don't even have a usable string.
            return _make_response(
                content=None,
                rule_type="PIPELINE",
                action="block",
                findings=all_findings,
                logs=all_logs,
            )

        sql_result = self.check_sql_injection()
        all_findings.extend(sql_result.findings)
        all_logs.extend(sql_result.logs)
        if sql_result.is_blocked:
            blocked = True

        inj_result = self.check_prompt_injection()
        all_findings.extend(inj_result.findings)
        all_logs.extend(inj_result.logs)
        if inj_result.is_blocked:
            blocked = True

        pii_result = self.check_pii()
        all_findings.extend(pii_result.findings)
        all_logs.extend(pii_result.logs)

        if blocked:
            return _make_response(
                content=None,
                rule_type="PIPELINE",
                action="block",
                findings=all_findings,
                logs=all_logs,
            )

        action: Action = "redact" if pii_result.is_redacted else "allow"
        return _make_response(
            content=pii_result.content,
            rule_type="PIPELINE",
            action=action,
            findings=all_findings,
            logs=all_logs,
        )
