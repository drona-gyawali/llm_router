from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_REF_DIR = Path("data/references")
DEFAULT_OUTPUT = Path("data/thresholds.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESH_MIN = 0.40
DEFAULT_THRESH_MAX = 0.95
DEFAULT_THRESH_STEP = 0.01

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep thresholds and find best F1 for any category set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing labeled evaluation *.jsonl files.",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=DEFAULT_REF_DIR,
        help="Directory containing pre-compiled reference *.npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the resulting thresholds JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name. Must match the one used in build_reference.",
    )
    parser.add_argument(
        "--thresh-min",
        type=float,
        default=DEFAULT_THRESH_MIN,
        help="Lower bound of threshold sweep.",
    )
    parser.add_argument(
        "--thresh-max",
        type=float,
        default=DEFAULT_THRESH_MAX,
        help="Upper bound of threshold sweep.",
    )
    parser.add_argument(
        "--thresh-step",
        type=float,
        default=DEFAULT_THRESH_STEP,
        help="Step size for threshold sweep.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional whitelist of category names to benchmark. "
        "If omitted, all *.jsonl files in --raw are processed.",
    )
    return parser.parse_args()


# ── Core logic (unchanged from original) ─────────────────────────────────────


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    """Load evaluation texts and binary labels from a JSONL file.

    Expected keys per line: "text" (str), "label" (0 or 1).
    """
    texts: list[str] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d – invalid JSON, skipping: %s", path, lineno, exc)
                continue

            if "text" not in data or "label" not in data:
                logger.warning(
                    "%s:%d – missing 'text' or 'label', skipping", path, lineno
                )
                continue

            text = data["text"]
            if not isinstance(text, str) or not text.strip():
                logger.warning(
                    "%s:%d – empty or non-string text, skipping", path, lineno
                )
                continue

            try:
                label = int(data["label"])
                if label not in (0, 1):
                    raise ValueError("label must be 0 or 1")
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "%s:%d – invalid label %r: %s",
                    path,
                    lineno,
                    data.get("label"),
                    exc,
                )
                continue

            texts.append(text)
            labels.append(label)

    return texts, labels


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Protects against zero vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def find_best_threshold(
    max_similarities: np.ndarray,
    true_labels: list[int],
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
) -> dict[str, float]:
    """Sweep thresholds and return the one with highest F1.
    On F1 ties, prefers the higher (more conservative) threshold.
    """
    thresholds = np.arange(thresh_min, thresh_max + 1e-9, thresh_step)
    best: dict[str, float] = {
        "threshold": 0.50,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": -1.0,
    }

    y_true = np.asarray(true_labels, dtype=int)

    for thresh in thresholds:
        y_pred = (max_similarities >= thresh).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if f1 > best["f1_score"] or (
            f1 == best["f1_score"] and thresh > best["threshold"]
        ):
            best = {
                "threshold": float(round(thresh, 4)),
                "precision": p,
                "recall": r,
                "f1_score": f1,
            }

    return best


def process_category(
    category: str,
    jsonl_path: Path,
    ref_path: Path,
    model: SentenceTransformer,
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
) -> dict[str, float] | None:
    """Run threshold sweep for one category. Returns metrics dict or None on failure."""
    logger.info("=" * 60)
    logger.info("Category: %s", category)
    logger.info("=" * 60)

    if not ref_path.is_file():
        logger.warning("Reference file missing: %s – skipping", ref_path)
        return None

    try:
        ref_embeddings = np.load(ref_path)
    except Exception as exc:
        logger.error("Failed to load reference embeddings %s: %s", ref_path, exc)
        return None

    if ref_embeddings.ndim != 2 or ref_embeddings.shape[0] == 0:
        logger.error(
            "Invalid reference embeddings shape %s – skipping", ref_embeddings.shape
        )
        return None

    ref_embeddings = normalize_embeddings(ref_embeddings.astype(np.float32))

    texts, labels = load_jsonl(jsonl_path)
    if not texts:
        logger.warning("No valid evaluation examples for '%s' – skipping", category)
        return None

    pos = sum(labels)
    neg = len(labels) - pos
    logger.info("Eval set: %d items (%d positives, %d negatives)", len(texts), pos, neg)

    try:
        eval_embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        eval_embeddings = normalize_embeddings(eval_embeddings.astype(np.float32))
    except Exception as exc:
        logger.error("Encoding failed for '%s': %s", category, exc)
        return None

    # Cosine similarity via normalized dot product
    sim_matrix = np.dot(eval_embeddings, ref_embeddings.T)
    max_sims = np.max(sim_matrix, axis=1)

    result = find_best_threshold(max_sims, labels, thresh_min, thresh_max, thresh_step)

    logger.info(
        "Best → threshold=%.2f  precision=%.4f  recall=%.4f  f1=%.4f",
        result["threshold"],
        result["precision"],
        result["recall"],
        result["f1_score"],
    )
    return result


def main() -> int:
    args = parse_args()

    # Validate directories
    if not args.refs.is_dir():
        logger.error("Reference directory not found: %s", args.refs)
        logger.error("Run build_reference.py first.")
        return 1

    if not args.raw.is_dir():
        logger.error("Raw eval directory not found: %s", args.raw)
        return 1

    # Resolve which JSONL files to benchmark
    all_jsonl = sorted(args.raw.glob("*.jsonl"))
    if not all_jsonl:
        logger.error("No JSONL files found in %s", args.raw)
        return 1

    if args.categories:
        whitelist = set(args.categories)
        jsonl_files = [f for f in all_jsonl if f.stem in whitelist]
        missing = whitelist - {f.stem for f in jsonl_files}
        if missing:
            logger.warning(
                "Requested categories not found in --raw: %s", sorted(missing)
            )
    else:
        jsonl_files = all_jsonl

    if not jsonl_files:
        logger.error("No matching category files to benchmark.")
        return 1

    # Log run config
    logger.info("Raw dir  : %s", args.raw)
    logger.info("Refs dir : %s", args.refs)
    logger.info("Output   : %s", args.output)
    logger.info("Model    : %s", args.model)
    logger.info(
        "Sweep    : %.2f → %.2f  step=%.2f",
        args.thresh_min,
        args.thresh_max,
        args.thresh_step,
    )
    if args.categories:
        logger.info("Filter   : %s", args.categories)

    # Load model
    logger.info("Loading embedding model: %s ...", args.model)
    try:
        model = SentenceTransformer(args.model)
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", args.model, exc)
        return 1

    logger.info("Found %d category file(s) to evaluate.\n", len(jsonl_files))

    # Process each category
    optimal_thresholds: dict[str, dict[str, float]] = {}

    for jsonl_path in jsonl_files:
        category = jsonl_path.stem
        ref_path = args.refs / f"{category}.npy"

        result = process_category(
            category=category,
            jsonl_path=jsonl_path,
            ref_path=ref_path,
            model=model,
            thresh_min=args.thresh_min,
            thresh_max=args.thresh_max,
            thresh_step=args.thresh_step,
        )
        if result is not None:
            optimal_thresholds[category] = result

    if not optimal_thresholds:
        logger.error("No categories were successfully processed.")
        return 1

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=4, ensure_ascii=False)

    logger.info("Success! Thresholds saved to '%s'", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
