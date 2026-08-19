#!/usr/bin/env python3
"""Route evaluation clips to the Lingala or Shona stack using hypothesis text.

Stage 0 of the pipeline.  A tag-free bilingual ASR model (OmniASR CTC-1B
jointly fine-tuned on Lingala and Shona with no language conditioning)
decodes every clip; the resulting hypothesis strings are classified by the
train-only character n-gram model fitted in ``train_text_lid.py``. The
routing decision reads the audio and the training transcripts.

A clip that decodes to nothing carries no language evidence.  Rather than
guess, the classifier is allowed to answer from its own training-set prior,
and the affected IDs are recorded.  More than a few blanks means the upstream
decode is broken, so ``--max-blank`` fails the run.

Usage
-----
python inference/route/route.py \
    --hypotheses outputs/route/joint_hypotheses.csv \
    --model artifacts/text_lid_train_only.joblib \
    --output outputs/route/route.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

LANGUAGES = ["lin", "sna"]
LOW_MARGIN_WARN = 0.10

# The shipped classifier was fitted under this exact scikit-learn version.
# Unpickling an estimator under a different version is not supported and can
# change predictions silently rather than raising, so refuse outright.
FITTED_SKLEARN_VERSION = "1.5.2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hypotheses(path: Path) -> tuple[list[str], list[str]]:
    """Read an (ID, Target) or (id, hypothesis) CSV of ASR output."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for id_column, text_column in (("ID", "Target"), ("id", "hypothesis")):
            if id_column in fields and text_column in fields:
                break
        else:
            raise SystemExit(f"unrecognised hypothesis schema: {fields}")
        rows = list(reader)
    identifiers = [str(row[id_column]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit(f"duplicate ID in {path}")
    return identifiers, [str(row[text_column] or "") for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True,
                        help="tag-free bilingual ASR output over every test clip")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-blank", type=int, default=5,
                        help="tolerate this many blank hypotheses before failing")
    args = parser.parse_args()

    identifiers, texts = read_hypotheses(args.hypotheses)
    blank = [i for i, t in zip(identifiers, texts) if not t.strip()]
    if len(blank) > args.max_blank:
        # A handful of blanks is the data; many mean the upstream decode broke.
        raise SystemExit(
            f"{len(blank)} blank hypotheses exceeds --max-blank {args.max_blank}; "
            f"the routing decode looks broken. First: {blank[:5]}"
        )

    if sklearn.__version__ != FITTED_SKLEARN_VERSION:
        raise SystemExit(
            f"classifier was fitted under scikit-learn {FITTED_SKLEARN_VERSION}, "
            f"this environment has {sklearn.__version__}; predictions would not "
            f"be reproducible. Build the `fuse` environment: bash env/build_envs.sh fuse"
        )
    classifier = joblib.load(args.model)
    if classifier.classes_.tolist() != LANGUAGES:
        raise SystemExit(f"unexpected classes: {classifier.classes_.tolist()}")

    probabilities = classifier.predict_proba(texts)
    predicted = classifier.classes_[probabilities.argmax(axis=1)].tolist()
    margins = np.abs(probabilities[:, 0] - probabilities[:, 1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "language", "probability_lin", "probability_sna", "margin"],
            lineterminator="\n",
        )
        writer.writeheader()
        for identifier, language, row, margin in zip(
            identifiers, predicted, probabilities, margins
        ):
            writer.writerow({
                "ID": identifier, "language": language,
                "probability_lin": f"{row[0]:.9f}",
                "probability_sna": f"{row[1]:.9f}",
                "margin": f"{margin:.9f}",
            })

    counts = Counter(predicted)
    low = int((margins < LOW_MARGIN_WARN).sum())
    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(identifiers),
        "counts": dict(counts),
        "minimum_margin": float(margins.min()),
        "median_margin": float(np.median(margins)),
        "rows_below_margin_warn": low,
        "blank_hypotheses": blank,
        "blank_routed_by_prior": len(blank),
        "margin_warn_threshold": LOW_MARGIN_WARN,
        "hypotheses_sha256": sha256_file(args.hypotheses),
        "model_sha256": sha256_file(args.model),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )

    if blank:
        # With no characters to score, TF-IDF yields a zero vector and the
        # classifier falls back to its own training-set prior. That is the
        # honest answer for a silent clip, and it is deterministic -- but say
        # so out loud rather than letting it pass as a normal decision.
        print(f"NOTE {len(blank)} blank hypothes(es) routed by the classifier prior: "
              f"{blank[:5]}")
    print(f"routed {len(identifiers)} clips: {dict(counts)}")
    print(f"minimum margin {margins.min():.6f}  median {np.median(margins):.6f}")
    if low:
        print(f"NOTE {low} row(s) below margin {LOW_MARGIN_WARN} -- inspect before trusting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
