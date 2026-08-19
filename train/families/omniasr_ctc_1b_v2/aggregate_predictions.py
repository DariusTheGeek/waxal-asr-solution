#!/usr/bin/env python3
"""Reconcile rank predictions and compute canonical raw and target-weighted score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import jiwer
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from scoring.asr import raw_text, score_texts, score_weighted_texts  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_errors(reference: str, hypothesis: str) -> dict[str, int | float]:
    words = jiwer.process_words(
        raw_text(reference, casefold=True), raw_text(hypothesis, casefold=True)
    )
    chars = jiwer.process_characters(
        raw_text(reference, casefold=False), raw_text(hypothesis, casefold=False)
    )
    word_ref = words.hits + words.substitutions + words.deletions
    char_ref = chars.hits + chars.substitutions + chars.deletions
    return {
        "word_errors": words.substitutions + words.deletions + words.insertions,
        "reference_words": word_ref,
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
        "reference_chars": char_ref,
        "row_wer": (words.substitutions + words.deletions + words.insertions)
        / max(1, word_ref),
        "row_cer": (chars.substitutions + chars.deletions + chars.insertions)
        / max(1, char_ref),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = pl.read_parquet(args.manifest).sort("manifest_index")
    if args.max_rows is not None:
        source = source.head(args.max_rows)
    prediction_records: list[dict[str, object]] = []
    rank_hashes = []
    for rank in range(args.world_size):
        path = args.rank_dir / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rank_hashes.append({"rank": rank, "sha256": sha256_file(path)})
        for line in path.read_text(encoding="utf-8").splitlines():
            prediction_records.append(json.loads(line))
    predictions = pl.DataFrame(prediction_records)
    if predictions.height != source.height:
        raise RuntimeError(
            f"prediction count drift: {predictions.height} != {source.height}"
        )
    if predictions["row_key"].n_unique() != predictions.height:
        raise RuntimeError("duplicate prediction row keys")
    joined = source.join(
        predictions.select("row_key", "hypothesis", "rank"),
        on="row_key",
        how="left",
        validate="1:1",
    ).sort("manifest_index")
    if joined["hypothesis"].null_count():
        raise RuntimeError("missing hypotheses")
    expected_ranks = joined["manifest_index"] % args.world_size
    if not (expected_ranks == joined["rank"]).all():
        raise RuntimeError("rank-shard assignment drift")

    references = joined["transcription_nfc"].fill_null("").to_list()
    hypotheses = joined["hypothesis"].to_list()
    errors = pl.DataFrame(
        [row_errors(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True)]
    )
    joined = pl.concat([joined, errors], how="horizontal_extend")
    raw = score_texts(references, hypotheses)
    weighted = None
    if "target_weight" in joined.columns:
        weighted = score_weighted_texts(
            references, hypotheses, joined["target_weight"].cast(pl.Float64).to_list()
        )
    metrics = {
        "schema_version": 1,
        "rows": joined.height,
        "blank_rows": joined.filter(pl.col("hypothesis").str.strip_chars() == "").height,
        "raw": raw,
        "target_weighted": weighted,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "rank_predictions": rank_hashes,
    }
    joined.write_parquet(args.output / "predictions.parquet", compression="zstd")
    joined.select("id", "hypothesis").rename(
        {"id": "ID", "hypothesis": "Target"}
    ).write_csv(args.output / "predictions.csv", quote_style="necessary")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
