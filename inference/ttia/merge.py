#!/usr/bin/env python3
"""Merge sharded embedding files into one, ordered by clip id.

Shards are written independently by parallel GPU processes, so the merged file
is sorted by id to give a single deterministic ordering regardless of how the
work was divided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = [np.load(p, allow_pickle=True) for p in args.shards]
    layer_keys = sorted({k for d in loaded for k in d.files if k.startswith("layer_")})
    if not layer_keys:
        raise SystemExit("no layer arrays found in the shards")
    for d, p in zip(loaded, args.shards):
        missing = [k for k in layer_keys if k not in d.files]
        if missing:
            raise SystemExit(f"{p.name} is missing {missing}")

    ids = np.concatenate([d["ids"] for d in loaded])
    if len(set(map(str, ids))) != len(ids):
        raise SystemExit("shards overlap: duplicate clip ids after concatenation")
    order = np.argsort(ids.astype(str))

    merged = {k: np.concatenate([d[k] for d in loaded])[order] for k in layer_keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids[order], **merged)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"clips": int(len(ids)), "layers": [int(k.split("_")[1]) for k in layer_keys],
         "shards": [str(p) for p in args.shards]}, indent=2) + "\n")
    print(f"merged {len(ids)} clips x {len(layer_keys)} layers -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
