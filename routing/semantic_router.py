from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import MODEL_REGISTRY, ROUTING_CONFIG, ModelConfig

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class SemanticRouterConfig:
    references_dir: Path = Path("data/routing/references")
    thresholds_file: Path = Path("data/routing/thresholds.json")
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if not self.references_dir.is_dir():
            raise ValueError(
                f"references_dir not found: {self.references_dir}\n"
                "Run build_reference.py --source data/routing/references_raw "
                "--output data/routing/references"
            )
        if not self.thresholds_file.is_file():
            raise ValueError(
                f"thresholds_file not found: {self.thresholds_file}\n"
                "Run benchmark_threshold.py --raw data/routing/raw "
                "--refs data/routing/references "
                "--output data/routing/thresholds.json"
            )


@dataclass(frozen=True)
class SemanticRoutingDecision:
    id: uuid.UUID
    timestamp: datetime
    category: str
    confidence: float
    threshold: float
    model: ModelConfig | None
    all_scores: dict[str, float]
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

    @property
    def is_fallback(self) -> bool:
        return self.category == "unknown"


class SemanticRouter:
    """
    Embedding-based semantic router.
    Category → model mapping loaded from ROUTING_CONFIG (models.yaml).
    No hardcoded model names.
    """

    _DEFAULT_CONFIG: Final[SemanticRouterConfig] = SemanticRouterConfig()

    def __init__(
        self, config: SemanticRouterConfig = _DEFAULT_CONFIG
    ) -> None:
        self._config = config

        # Read category→model mapping from ROUTING_CONFIG (models.yaml)
        # This is now fully dynamic — change yaml, restart, done.
        self._category_to_model: dict[str, str] = dict(ROUTING_CONFIG.categories)
        self._fallback_model_name: str = ROUTING_CONFIG.semantic_fallback

        # Load thresholds from benchmark output
        with config.thresholds_file.open("r", encoding="utf-8") as f:
            raw_thresholds: dict[str, dict] = json.load(f)

        # Load embedding model
        logger.info("Loading semantic router model: %s", config.model_name)
        self._model: Final[SentenceTransformer] = SentenceTransformer(
            config.model_name
        )

        # Load reference embeddings
        self._refs: dict[str, np.ndarray] = {}
        self._thresholds: dict[str, float] = {}

        npy_files = {p.stem: p for p in config.references_dir.glob("*.npy")}

        # Only load categories that exist in ALL THREE:
        #   references_dir (.npy files)
        #   thresholds_file (benchmark output)
        #   ROUTING_CONFIG.categories (models.yaml)
        active = set(npy_files) & set(raw_thresholds) & set(self._category_to_model)

        if not active:
            raise RuntimeError(
                "No active routing categories. Ensure references, thresholds, "
                "and routing.categories in models.yaml are all consistent."
            )

        for category in sorted(active):
            try:
                matrix = np.load(npy_files[category]).astype(np.float32)
            except Exception as exc:
                logger.error("Failed to load embeddings for '%s': %s", category, exc)
                continue

            if matrix.ndim != 2 or matrix.shape[0] == 0:
                logger.error("Invalid shape for '%s' — skipping", category)
                continue

            self._refs[category] = self._normalize(matrix)
            self._thresholds[category] = float(raw_thresholds[category]["threshold"])

            mapped_model = self._category_to_model[category]
            logger.info(
                "Loaded '%-12s'  refs=%-3d  threshold=%.2f  → %s",
                category,
                matrix.shape[0],
                self._thresholds[category],
                mapped_model,
            )

        if not self._refs:
            raise RuntimeError("SemanticRouter: no usable categories loaded.")


    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        return self._normalize(vec)

    def _resolve_model(self, category: str) -> ModelConfig | None:
        """
        Resolve category → model name (from ROUTING_CONFIG) → ModelConfig.
        Falls back to semantic_fallback if category not mapped.
        """
        model_name = self._category_to_model.get(category, self._fallback_model_name)
        model = MODEL_REGISTRY.get(model_name)
        if model is None:
            logger.warning(
                "Category '%s' maps to '%s' which is not in MODEL_REGISTRY.",
                category, model_name,
            )
        return model


    def route(self, text: str) -> SemanticRoutingDecision:
        """
        Classify text and return the best-matching routing decision.
        Category→model mapping comes entirely from models.yaml.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")

        vec = self._encode(text)

        # Score all categories
        all_scores: dict[str, float] = {
            cat: float(np.max(np.dot(vec, ref.T)))
            for cat, ref in self._refs.items()
        }

        # Filter to categories clearing their threshold
        passing = {
            cat: score
            for cat, score in all_scores.items()
            if score >= self._thresholds[cat]
        }

        scores_str = "  ".join(
            f"{cat}={score:.3f}{'✓' if cat in passing else ''}"
            for cat, score in sorted(all_scores.items(), key=lambda x: -x[1])
        )

        if passing:
            best_category = max(passing, key=passing.__getitem__)
            confidence = passing[best_category]
            threshold = self._thresholds[best_category]
            model = self._resolve_model(best_category)
            logs = [
                f"Category     : {best_category}",
                f"Confidence   : {confidence:.4f}",
                f"Threshold    : {threshold:.2f}",
                f"Model        : {model.name if model else 'NOT FOUND'}",
                f"Provider     : {model.provider if model else 'N/A'}",
                f"Mapping from : models.yaml routing.categories",
                f"All scores   : {scores_str}",
            ]
            logger.info(
                "SemanticRouter → category=%-12s  confidence=%.4f  model=%s",
                best_category, confidence,
                model.name if model else "NOT FOUND",
            )
        else:
            best_category = "unknown"
            confidence = max(all_scores.values()) if all_scores else 0.0
            threshold = 0.0
            model = MODEL_REGISTRY.get(self._fallback_model_name)
            logs = [
                "Category     : unknown (no category cleared threshold)",
                f"Best score   : {confidence:.4f}",
                f"Fallback     : {self._fallback_model_name} (from models.yaml)",
                f"All scores   : {scores_str}",
            ]
            logger.info(
                "SemanticRouter → no match  best_score=%.4f  fallback=%s",
                confidence, self._fallback_model_name,
            )

        return SemanticRoutingDecision(
            id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            category=best_category,
            confidence=confidence,
            threshold=threshold,
            model=model,
            all_scores=all_scores,
            logs=tuple(logs),
        )

    def explain(self, text: str) -> dict[str, float]:
        """Return similarity scores for all categories without routing."""
        vec = self._encode(text)
        scores = {
            cat: float(np.max(np.dot(vec, ref.T)))
            for cat, ref in self._refs.items()
        }
        return dict(sorted(scores.items(), key=lambda x: -x[1]))