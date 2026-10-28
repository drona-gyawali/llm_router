from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_SOURCE = Path("data/references_raw")
DEFAULT_OUTPUT = Path("data/references")
DEFAULT_MODEL  = "all-MiniLM-L6-v2"
DEFAULT_BATCH  = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reference embeddings for any category set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Directory containing *.jsonl reference files.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Directory to write .npy embedding files.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--categories", nargs="*", default=None,
        help="Optional whitelist of category names (stems). "
             "If omitted, all *.jsonl files in --source are processed.",
    )
    return parser.parse_args()


def load_positive_texts(file_path: Path) -> List[str]:
    """
    Load texts from a JSONL file.

    Accepts two formats:
      - Reference format: {"text": "..."}            (label assumed 1)
      - Eval format:      {"text": "...", "label": 1} (only label==1 loaded)

    Lines with label != 1 are silently skipped so the same file can be
    used as a reference source even if it contains negatives.
    """
    texts: List[str] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path.name}:{line_num}: invalid JSON – {exc}"
                ) from exc

            if "text" not in data:
                raise ValueError(
                    f"{file_path.name}:{line_num}: missing required key 'text'"
                )

            # Treat missing label as positive; skip explicit non-positives.
            label = data.get("label", 1)

            # Support string labels (routing) and int labels (guardrails).
            # A string label is always treated as positive (it is the category name).
            if isinstance(label, int) and label != 1:
                continue

            text = data["text"]
            if not isinstance(text, str) or not text.strip():
                logger.warning(
                    "%s:%d: empty or non-string text – skipping",
                    file_path.name, line_num,
                )
                continue

            texts.append(text.strip())

    return texts


def encode_and_save(
    model: SentenceTransformer,
    texts: Sequence[str],
    output_path: Path,
    batch_size: int = DEFAULT_BATCH,
) -> None:
    """Encode *texts* and save the numpy array to *output_path*."""
    logger.info("  Encoding %d texts…", len(texts))
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(
        "  Saved → %s  (shape=%s, dtype=%s)",
        output_path, embeddings.shape, embeddings.dtype,
    )


def main() -> None:
    args = parse_args()

    if not args.source.is_dir():
        logger.error("Source directory does not exist: %s", args.source)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Resolve which JSONL files to process.
    all_jsonl = sorted(args.source.glob("*.jsonl"))
    if not all_jsonl:
        logger.warning("No *.jsonl files found in %s", args.source)
        return

    if args.categories:
        whitelist = set(args.categories)
        jsonl_files = [f for f in all_jsonl if f.stem in whitelist]
        missing = whitelist - {f.stem for f in jsonl_files}
        if missing:
            logger.warning("Requested categories not found: %s", sorted(missing))
    else:
        jsonl_files = all_jsonl

    if not jsonl_files:
        logger.error("No matching category files to process.")
        sys.exit(1)

    logger.info(
        "Source   : %s  (%d file(s))", args.source, len(jsonl_files)
    )
    logger.info("Output   : %s", args.output)
    logger.info("Model    : %s", args.model)
    logger.info("Batch    : %d", args.batch)
    if args.categories:
        logger.info("Filter   : %s", args.categories)
    logger.info("")

    logger.info("Loading embedding model: %s", args.model)
    model = SentenceTransformer(args.model)

    processed = 0
    for file_path in jsonl_files:
        category = file_path.stem
        logger.info("─── Category: %s", category)

        try:
            texts = load_positive_texts(file_path)
        except ValueError as exc:
            logger.error("  Failed to load %s: %s", file_path.name, exc)
            continue

        if not texts:
            logger.warning("  No positive examples in %s – skipping", file_path.name)
            continue

        out_path = args.output / f"{category}.npy"
        try:
            encode_and_save(model, texts, out_path, batch_size=args.batch)
            processed += 1
        except Exception:
            logger.exception("  Failed to encode/save '%s'", category)
            continue

    logger.info("")
    logger.info(
        "Done. Successfully processed %d / %d categories.",
        processed, len(jsonl_files),
    )


if __name__ == "__main__":
    main()