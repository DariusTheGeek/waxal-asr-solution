#!/usr/bin/env python3
"""Merge the fused language lanes into one submission and normalise the text.

Deliberately minimal: post-processing is Unicode NFC normalisation and
whitespace collapsing, nothing else.

Usage
-----
python inference/submit/postprocess.py --lanes outputs/fused/lin.csv outputs/fused/sna.csv \
    --output outputs/submissions/final_submission.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.text import normalize_text  # noqa: E402


def read_lane(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty lane: {path}")
    return [(str(r["ID"]), str(r["Target"] or "")) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=Path,
                        help="optional CSV whose ID column fixes the output row order")
    args = parser.parse_args()

    merged: dict[str, str] = {}
    changed = 0
    for lane in args.lanes:
        for identifier, target in read_lane(lane):
            if identifier in merged:
                raise SystemExit(f"ID {identifier} appears in more than one lane")
            normalised = normalize_text(target)
            if normalised != target:
                changed += 1
            merged[identifier] = normalised

    blank = [k for k, v in merged.items() if not v.strip()]
    if blank:
        raise SystemExit(f"{len(blank)} blank target(s), first: {blank[:3]}")

    identifiers = list(merged)
    if args.order:
        with args.order.open("r", encoding="utf-8-sig", newline="") as handle:
            wanted = [str(r["ID"]) for r in csv.DictReader(handle)]
        if set(wanted) != set(identifiers):
            raise SystemExit("--order ID set does not match the fused lanes")
        identifiers = wanted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"], lineterminator="\n")
        writer.writeheader()
        for identifier in identifiers:
            writer.writerow({"ID": identifier, "Target": merged[identifier]})

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lanes": [str(p) for p in args.lanes],
        "rows": len(identifiers),
        "rows_changed_by_normalisation": changed,
        "unicode_form": "NFC",
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(identifiers)} rows ({changed} changed by normalisation)")
    print(f"sha256 {record['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
