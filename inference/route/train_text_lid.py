#!/usr/bin/env python3
"""Fit the train-only character n-gram language classifier (Lingala vs Shona).

Stage 0 routes each evaluation clip to a language stack by classifying its
decoded text.  This classifier is fitted on the gold training transcripts and
their language labels; at inference it is applied to the ASR hypothesis
strings produced by the tag-free bilingual model.

Usage
-----
python inference/route/train_text_lid.py \
    --train-manifest data/manifests/train.rows.parquet \
    --output artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LANGUAGES = ["lin", "sna"]
SEED = 42
EXPECTED_TRAIN_ROWS = 32_328


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large inputs stay bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_classifier() -> Pipeline:
    """Character 3-5 gram TF-IDF followed by L2 logistic regression.

    Character n-grams are used rather than word n-grams because Lingala and
    Shona differ sharply in orthographic texture (Shona's frequent 'dz', 'sv',
    'zv' clusters against Lingala's 'ng', 'mb', 'nz'), which survives ASR noise
    far better than whole-word identity does.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    lowercase=True,
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def load_training_corpus(manifest: Path) -> tuple[list[str], list[str]]:
    """Read gold transcripts and language labels, failing loudly on drift."""
    frame = pd.read_parquet(manifest).sort_values("manifest_index")
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"training corpus drift: expected {EXPECTED_TRAIN_ROWS} rows, "
            f"found {len(frame)}"
        )
    if set(frame["language"].unique()) != set(LANGUAGES):
        raise RuntimeError(f"unexpected languages: {sorted(frame['language'].unique())}")
    if int(frame["training_target"].isna().sum()):
        raise RuntimeError("training corpus contains null transcripts")
    return (
        frame["training_target"].astype(str).tolist(),
        frame["language"].astype(str).tolist(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="parquet with columns: manifest_index, language, training_target")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    texts, labels = load_training_corpus(args.train_manifest)

    classifier = build_classifier()
    classifier.fit(texts, labels)
    if classifier.named_steps["classifier"].classes_.tolist() != LANGUAGES:
        raise RuntimeError("class ordering drift; downstream indexing assumes [lin, sna]")

    model_path = args.output / "text_lid_train_only.joblib"
    joblib.dump(classifier, model_path, compress=3)

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "fit_data": "gold training transcripts only",
        "train_rows": len(texts),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "classes": LANGUAGES,
        "vocabulary_size": len(classifier.named_steps["tfidf"].vocabulary_),
        "model_sha256": sha256_file(model_path),
    }
    (args.output / "TRAIN_RECORD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(f"fitted on {len(texts)} gold transcripts "
          f"({labels.count('lin')} lin / {labels.count('sna')} sna)")
    print(f"vocabulary {record['vocabulary_size']}  ->  {model_path}")
    print(f"sha256 {record['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
