#!/usr/bin/env python3
"""Deterministic rank-sharded OmniASR LLM inference for a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl
import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

from runtime_assets import resolve_repo_root


EXPECTED_PARAMETERS = 4_380_578_432


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio_root(manifest: Path, root: Path) -> Path:
    """Resolve the portable manifest's frozen relative TSV audio root."""

    suffix = ".rows.parquet"
    if not manifest.name.endswith(suffix):
        raise RuntimeError(f"manifest must end in {suffix}: {manifest}")
    split = manifest.name[: -len(suffix)]
    tsv = manifest.parent / f"{split}.tsv"
    if tsv.is_symlink() or not tsv.is_file():
        raise RuntimeError(f"missing or unsafe TSV companion: {tsv}")
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].strip():
        raise RuntimeError(f"TSV companion has no audio-root header: {tsv}")
    relative = Path(lines[0].strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe TSV audio-root header: {relative}")
    audio_root = (root / relative).resolve()
    try:
        audio_root.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError("TSV audio root escapes WAXAL3") from error
    if audio_root.is_symlink() or not audio_root.is_dir():
        raise RuntimeError(f"missing or unsafe audio root: {audio_root}")
    return audio_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--language-code", required=True)
    parser.add_argument("--beam-size", type=int, choices=(1, 5), default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--work-chunk", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.work_chunk <= 0:
        raise ValueError("batch-size and work-chunk must be positive")

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

    root = resolve_repo_root()
    manifest = args.manifest.resolve()
    try:
        manifest.relative_to(root)
    except ValueError as error:
        raise RuntimeError("manifest escapes WAXAL3") from error
    frame = pl.read_parquet(manifest).sort("manifest_index")
    if args.max_rows is not None:
        frame = frame.head(args.max_rows)
    if frame["row_key"].n_unique() != frame.height:
        raise RuntimeError("manifest row keys are not unique")
    rank_frame = frame.filter(pl.col("manifest_index") % world_size == rank)
    audio_root = resolve_audio_root(manifest, root)
    paths = [audio_root / value for value in rank_frame["derived_audio_relpath"]]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing derived audio: {missing[:3]}")

    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"rank_{rank}.jsonl"
    terminal_path = args.output / f"rank_{rank}.terminal.json"
    if output_path.exists() or terminal_path.exists():
        raise FileExistsError(output_path if output_path.exists() else terminal_path)

    beam_config = Wav2Vec2LlamaBeamSearchConfig(
        nbest=args.beam_size,
        length_norm=False,
    )
    pipeline = ASRInferencePipeline(
        args.model_card,
        device=torch.device("cuda", local_rank),
        dtype=torch.bfloat16,
        beam_search_config=beam_config,
    )
    if not isinstance(pipeline.model, Wav2Vec2LlamaModel):
        raise RuntimeError("inference model is not Wav2Vec2LlamaModel")
    parameters = sum(parameter.numel() for parameter in pipeline.model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"LLM parameter-count drift: {parameters} != {EXPECTED_PARAMETERS}"
        )
    if (
        pipeline.model.lang_mapping is None
        or args.language_code.casefold() not in pipeline.model.lang_mapping
    ):
        raise RuntimeError(f"unsupported language code: {args.language_code}")

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
                [str(path) for path in chunk_paths],
                lang=[args.language_code] * len(chunk_paths),
                batch_size=args.batch_size,
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
                            "language_code": args.language_code,
                            "beam_size": args.beam_size,
                            "length_normalization": False,
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
        "schema_version": 1,
        "status": "PASS",
        "rank": rank,
        "world_size": world_size,
        "rows": completed,
        "model_card": args.model_card,
        "model_parameters": parameters,
        "language_code": args.language_code,
        "language_id": int(
            pipeline.model.lang_mapping[args.language_code.casefold()]
        ),
        "beam_size": args.beam_size,
        "length_normalization": False,
        "predictions_sha256": sha256_file(output_path),
    }
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
