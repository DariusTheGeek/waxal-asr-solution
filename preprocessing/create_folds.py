#!/usr/bin/env python3
"""Split a manifest into train and validation ID lists.

A single stratified hold-out, not k-fold.  Stratification is on language, so
the validation set carries the same Lingala/Shona proportion as the training
corpus and the two lanes stay independently measurable.

Utility only: this writes ID lists from whatever manifest it is given. It is
not what produced the manifests the released models trained against — those
ship with their own fixed row sets, recorded in the `BUILD.json` beside them.

Usage
-----
python preprocessing/create_folds.py \
    --manifest data/manifests/train.rows.parquet \
    --output data/cv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 0.10
LANGUAGES = ["lin", "sna"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="parquet with columns: id, language, training_target")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest)
    missing = {"id", "language"} - set(frame.columns)
    if missing:
        raise SystemExit(f"manifest is missing required columns: {sorted(missing)}")
    if set(frame["language"].unique()) != set(LANGUAGES):
        raise SystemExit(f"unexpected languages: {sorted(frame['language'].unique())}")
    if frame["id"].duplicated().any():
        raise SystemExit("manifest contains duplicate ids")

    # Sort first so the split depends only on the seed, never on the order the
    # manifest happened to be written in.
    frame = frame.sort_values("id").reset_index(drop=True)

    train_frame, val_frame = train_test_split(
        frame,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=frame["language"],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_frame), ("val", val_frame)):
        path = args.output / f"cv_{name}_ids.csv"
        part[["id", "language"]].sort_values("id").to_csv(path, index=False)

    overlap = set(train_frame["id"]) & set(val_frame["id"])
    if overlap:
        raise SystemExit(f"train/val overlap: {len(overlap)} ids")

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "single stratified hold-out on language",
        "seed": args.seed,
        "val_size": args.val_size,
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_val": int(len(val_frame)),
        "train_by_language": train_frame["language"].value_counts().to_dict(),
        "val_by_language": val_frame["language"].value_counts().to_dict(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output / "SPLIT_RECORD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"train {len(train_frame)}  val {len(val_frame)}")
    print(f"val by language: {record['val_by_language']}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
