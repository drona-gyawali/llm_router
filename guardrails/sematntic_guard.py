from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from guardrails.deterministic_rule import _make_response, GuardResponse
import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = [
    "Action",
    "SemanticConfig",
    "GuardResponse",
    "SemanticGuard",
]

logger = logging.getLogger(__name__)

Action = Literal["allow", "redact", "block"]


@dataclass(frozen=True)
class SemanticConfig:
    """
    Runtime configuration for SemanticGuard.

    Attributes
    ----------
    references_dir:
        Directory that holds the pre-compiled .npy embedding files,
        one per category (e.g. ``sexual.npy``, ``sql_injection.npy``).
    thresholds_file:
        JSON file produced by the threshold optimizer.  Maps category name
        → ``{"threshold": float, ...}``.
    model_name:
        SentenceTransformer model used to encode incoming text.
        Must match the model used to build the reference embeddings.
    enabled_categories:
        Subset of categories to run.  ``None`` means run every category
        that has both a .npy file and a threshold entry.
    """

    references_dir: Path = Path("data/references")
    thresholds_file: Path = Path("data/thresholds.json")
    model_name: str = "all-MiniLM-L6-v2"
    enabled_categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.references_dir.is_dir():
            raise ValueError(
                f"references_dir does not exist: {self.references_dir}"
            )
        if not self.thresholds_file.is_file():
            raise ValueError(
                f"thresholds_file does not exist: {self.thresholds_file}"
            )


class SemanticGuard:
    """
    Second-layer semantic guardrail using pre-compiled embedding similarity.

    Complements DeterministicRuleGuard by catching content that regex cannot
    reliably detect — paraphrased harmful requests, novel prompt injections,
    implicit policy violations, etc.

    How it works
    ------------
    1. At construction the model, reference embeddings (.npy), and thresholds
       are loaded once and kept in memory.
    2. For each call to ``evaluate()`` the input text is encoded into a single
       384-dim vector (L2-normalised).
    3. A dot product between that vector and every reference matrix gives a
       cosine similarity score for each reference sentence (dot product ==
       cosine similarity when both sides are L2-normalised).
    4. The maximum similarity per category is compared against that category's
       optimised threshold from ``thresholds.json``.
    5. Any category whose max similarity >= threshold fires and blocks the
       request.

    Parameters
    ----------
    config:
        SemanticConfig instance.  Defaults to module-level defaults.

    Notes
    -----
    * The model and reference matrices are loaded once at construction and
      reused across calls, instantiate once and call ``evaluate()`` many times.
    * All guard methods are stateless with respect to the payload; the payload
      is passed in per-call, not stored.
    * The response contract (GuardResponse fields) is identical to
      DeterministicRuleGuard so the two guards can be chained or swapped.
    """

    _DEFAULT_CONFIG: Final[SemanticConfig] = SemanticConfig()

    def __init__(self, config: SemanticConfig = _DEFAULT_CONFIG) -> None:
        self._config = config

        with config.thresholds_file.open("r", encoding="utf-8") as f:
            raw_thresholds: dict[str, dict] = json.load(f)

        npy_files = {p.stem: p for p in config.references_dir.glob("*.npy")}

        if config.enabled_categories is not None:
            active = set(config.enabled_categories)
        else:
            active = set(npy_files) & set(raw_thresholds)

        if not active:
            raise ValueError(
                "No active categories found. Check references_dir and "
                "thresholds_file are populated and consistent."
            )

        logger.info("Loading semantic model: %s", config.model_name)
        self._model: Final[SentenceTransformer] = SentenceTransformer(
            config.model_name
        )

        # _refs: category → (N, 384) L2-normalised float32 matrix
        # _thresholds: category → float
        self._refs: Final[dict[str, np.ndarray]] = {}
        self._thresholds: Final[dict[str, float]] = {}

        for category in sorted(active):
            npy_path = npy_files[category]
            try:
                matrix = np.load(npy_path).astype(np.float32)
            except Exception as exc:
                logger.error(
                    "Failed to load reference embeddings for '%s': %s",
                    category, exc,
                )
                continue

            if matrix.ndim != 2 or matrix.shape[0] == 0:
                logger.error(
                    "Invalid embedding shape %s for '%s' — skipping",
                    matrix.shape, category,
                )
                continue

            self._refs[category] = self._normalize(matrix)
            self._thresholds[category] = float(raw_thresholds[category]["threshold"])
            logger.info(
                "Loaded category '%s'  refs=%d  threshold=%.2f",
                category,
                matrix.shape[0],
                self._thresholds[category],
            )

        if not self._refs:
            raise RuntimeError(
                "SemanticGuard initialised with no usable categories."
            )


    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so dot product == cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single string → (1, 384) L2-normalised float32 array."""
        vec = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        return self._normalize(vec)   # shape (1, 384)


    def check_category(self, text: str, category: str) -> GuardResponse:
        """
        Run the semantic check for a single *category*.

        Returns a GuardResponse with ``action="block"`` if the maximum
        cosine similarity to any reference sentence meets or exceeds the
        category threshold, ``"allow"`` otherwise.

        Parameters
        ----------
        text:
            Raw input string to evaluate.
        category:
            Category name (must be one of the loaded categories).
        """
        if category not in self._refs:
            return _make_response(
                content=text,
                rule_type=category.upper(),
                action="allow",
                findings=[],
                logs=[f"Category '{category}' not loaded — skipping"],
            )

        vec = self._encode(text)                          # (1, 384)
        ref_matrix = self._refs[category]                 # (N, 384)
        similarities = np.dot(vec, ref_matrix.T)          # (1, N)
        max_sim = float(np.max(similarities))
        threshold = self._thresholds[category]

        if max_sim >= threshold:
            return _make_response(
                content=None,
                rule_type=category.upper(),
                action="block",
                findings=[category.upper()],
                logs=[
                    f"Category '{category}' matched: "
                    f"max_similarity={max_sim:.4f} >= threshold={threshold:.2f}"
                ],
            )

        return _make_response(
            content=text,
            rule_type=category.upper(),
            action="allow",
            findings=[],
            logs=[
                f"Category '{category}' clear: "
                f"max_similarity={max_sim:.4f} < threshold={threshold:.2f}"
            ],
        )


    def evaluate(self, text: str) -> GuardResponse:
        """
        Run every loaded category against *text* and return an aggregated result.

        The input is encoded **once** and reused across all category checks
        for efficiency.

        Pipeline
        --------
        For each active category (sorted alphabetically):
            - Compute max cosine similarity vs reference matrix.
            - If >= threshold → mark blocked, record finding.

        The aggregated response mirrors DeterministicRuleGuard.evaluate():
            - ``action`` is ``"block"`` if any category fired, else ``"allow"``.
            - ``content`` is ``None`` on block, original text on allow.
            - ``findings`` is the sorted tuple of all fired category names.
            - ``logs`` contains per-category detail for every check run.

        Parameters
        ----------
        text:
            Raw input string to evaluate.

        Returns
        -------
        GuardResponse
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")

        all_findings: list[str] = []
        all_logs: list[str] = []
        blocked = False

        # Encode once, reuse for all categories.
        vec = self._encode(text)   # (1, 384)

        for category in sorted(self._refs):
            ref_matrix = self._refs[category]             # (N, 384)
            similarities = np.dot(vec, ref_matrix.T)      # (1, N)
            max_sim = float(np.max(similarities))
            threshold = self._thresholds[category]

            if max_sim >= threshold:
                all_findings.append(category.upper())
                all_logs.append(
                    f"[{category.upper()}] BLOCKED  "
                    f"max_similarity={max_sim:.4f} >= threshold={threshold:.2f}"
                )
                blocked = True
            else:
                all_logs.append(
                    f"[{category.upper()}] clear  "
                    f"max_similarity={max_sim:.4f} < threshold={threshold:.2f}"
                )

        if blocked:
            return _make_response(
                content=None,
                rule_type="PIPELINE",
                action="block",
                findings=all_findings,
                logs=all_logs,
            )

        return _make_response(
            content=text,
            rule_type="PIPELINE",
            action="allow",
            findings=[],
            logs=all_logs,
        )