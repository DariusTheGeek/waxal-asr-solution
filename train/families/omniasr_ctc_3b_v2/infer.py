#!/usr/bin/env python3
"""Deterministic rank-sharded greedy CTC inference for a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl
import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--work-chunk", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size not in {1, 8}:
        raise RuntimeError(f"unsupported world size: {world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"visible GPU count {torch.cuda.device_count()} < world size {world_size}"
        )
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)

    frame = pl.read_parquet(args.manifest).sort("manifest_index")
    if args.max_rows is not None:
        frame = frame.head(args.max_rows)
    if frame["row_key"].n_unique() != frame.height:
        raise RuntimeError("manifest row keys are not unique")
    rank_frame = frame.filter(pl.col("manifest_index") % world_size == rank)
    audio_root = args.manifest.resolve().parent.parent / "audio"
    paths = [audio_root / value for value in rank_frame["derived_audio_relpath"]]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing derived audio: {missing[:3]}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"rank_{rank}.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    pipeline = ASRInferencePipeline(
        args.model_card,
        device=torch.device("cuda", local_rank),
        dtype=torch.bfloat16,
    )
    records = rank_frame.select(
        "row_key",
        "id",
        "manifest_index",
        "derived_audio_relpath",
        "derived_audio_sha256",
    ).to_dicts()
    completed = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for start in range(0, len(paths), args.work_chunk):
            chunk_paths = paths[start : start + args.work_chunk]
            hypotheses = pipeline.transcribe(
                [str(path) for path in chunk_paths], batch_size=args.batch_size
            )
            if len(hypotheses) != len(chunk_paths):
                raise RuntimeError("inference output count drift")
            for source, hypothesis in zip(
                records[start : start + args.work_chunk], hypotheses, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            **source,
                            "hypothesis": str(hypothesis),
                            "rank": rank,
                            "world_size": world_size,
                            "model_card": args.model_card,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            completed += len(chunk_paths)
            print(
                json.dumps(
                    {"rank": rank, "completed": completed, "total": len(paths)}
                ),
                flush=True,
            )
    terminal = {
        "rank": rank,
        "world_size": world_size,
        "rows": completed,
        "predictions_sha256": sha256_file(output_path),
    }
    (args.output / f"rank_{rank}.terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
