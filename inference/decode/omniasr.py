#!/usr/bin/env python3
"""Decode one OmniASR model over the clips assigned to a language lane.

Runs under .venvs/omni (fairseq2 + omnilingual-asr). Work is sharded across the
visible GPUs; each shard writes its own JSONL and the parent merges them in
manifest order, so the output does not depend on how the work was divided.

Two model shapes are handled from one script because they differ only in how the
decoder is configured:

    CTC models  (ctc3b, ctc7b, joint)  greedy
    LLM models  (llm1b, llm3b)         beam search, length_norm from the config

``configs/sna/ctc7b.yaml`` names three checkpoints rather than one. Their fusion
is output-space, so each decodes independently and the three hypotheses are
combined by conservative word ROVER before the lane fusion sees them.

Usage
-----
python inference/decode/omniasr.py --config configs/lin/llm3b.yaml \
    --audio data/test_audio --route outputs/route/route.csv --lane lin \
    --output outputs/decodes/lin_llm3b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.text import normalize_text  # noqa: E402

CTC_MODELS = {"ctc3b", "ctc7b", "joint"}
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def require_omni_environment() -> None:
    expected = ROOT / ".venvs/omni/bin/python"
    if expected.exists() and Path(sys.executable).resolve() != expected.resolve():
        raise SystemExit(
            f"decode/omniasr.py must run under the omni environment.\n"
            f"  expected: {expected}\n  running:  {sys.executable}\n"
            f"  fix:      bash run_inference.sh"
        )


def render_asset_card(template: Path, weights_dir: Path, work_dir: Path) -> Path:
    """Materialise the shipped asset card with real paths and register it."""
    rendered = template.read_text(encoding="utf-8").replace(
        "@WAXAL_MODEL_DIR@", weights_dir.resolve().as_posix())
    if "@WAXAL_MODEL_DIR@" in rendered:
        raise SystemExit("asset-card placeholder survived rendering")
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "waxal.yaml"
    path.write_text(rendered, encoding="utf-8")
    os.environ["FAIRSEQ2_ASSET_DIR"] = str(work_dir)
    return path


def clip_ids_for_lane(route: Path | None, lane: str | None) -> list[str] | None:
    """Return the IDs this lane owns, or None to decode every clip."""
    if route is None or lane is None:
        return None
    with route.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [str(r["ID"]) for r in rows if r["language"] == lane]
    if not selected:
        raise SystemExit(f"route file assigns no clip to lane '{lane}'")
    return selected


def resolve_audio(audio_dir: Path, identifiers: list[str] | None) -> list[tuple[str, Path]]:
    by_stem = {}
    for path in sorted(audio_dir.rglob("*")):
        if path.suffix.lower() in AUDIO_SUFFIXES:
            by_stem.setdefault(path.stem, path)
    if identifiers is None:
        identifiers = sorted(by_stem)
    missing = [i for i in identifiers if i not in by_stem]
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) have no audio, first: {missing[:3]}")
    return [(i, by_stem[i]) for i in identifiers]


def decode_shard(rank: int, world_size: int, card_name: str, config: dict,
                 work: list[tuple[str, Path]], out_dir: Path, batch_size: int) -> None:
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    # TF32 changes the numerics of convolution and matmul on Ampere and later.
    # PyTorch leaves cuDNN's TF32 on by default, which perturbs the CTC models'
    # convolutional front end enough to change roughly a fifth of the decoded
    # words. The reference decodes ran with tf32 disabled, and training pinned
    # allow_tf32: false for the same reason.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    torch.cuda.set_device(rank)
    torch.set_num_threads(2)

    kwargs = {}
    decode = config["decode"]
    if decode["strategy"] == "beam":
        from omnilingual_asr.models.wav2vec2_llama.config import (
            Wav2Vec2LlamaBeamSearchConfig,
        )
        kwargs["beam_search_config"] = Wav2Vec2LlamaBeamSearchConfig(
            nbest=decode["beam_size"], length_norm=bool(decode.get("length_norm", False)))

    pipeline = ASRInferencePipeline(card_name, device=torch.device("cuda", rank),
                                    dtype=torch.bfloat16, **kwargs)

    shard = [item for index, item in enumerate(work) if index % world_size == rank]
    out_path = out_dir / f"rank_{rank}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        work_chunk = int(decode.get("work_chunk", 32))
        for start in range(0, len(shard), work_chunk):
            chunk = shard[start:start + work_chunk]
            # `lang` conditions the LLM decoder and is ignored by CTC models.
            # The code comes from this pipeline's own routing decision.
            language_code = decode.get("language_code")
            hypotheses = pipeline.transcribe(
                [str(p) for _, p in chunk],
                lang=[language_code] * len(chunk) if language_code else None,
                batch_size=batch_size)
            if len(hypotheses) != len(chunk):
                raise RuntimeError("inference returned a different number of hypotheses")
            for (identifier, _), hypothesis in zip(chunk, hypotheses):
                handle.write(json.dumps({"ID": identifier,
                                         "Target": normalize_text(hypothesis)},
                                        ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"rank": rank, "done": min(start + work_chunk, len(shard)),
                              "total": len(shard)}), flush=True)


def run_one_checkpoint(config: dict, checkpoint: str, audio: list[tuple[str, Path]],
                       weights_dir: Path, batch_size: int) -> dict[str, str]:
    import torch
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="waxal-decode-") as tmp:
        tmp_path = Path(tmp)
        card_template = weights_dir / config.get("asset_card", "card.yaml")
        if not card_template.is_file():
            raise SystemExit(f"missing asset card: {card_template}")
        # ctc7b ships three checkpoints; point the card at the one being decoded.
        rendered = card_template.read_text().replace(
            "@WAXAL_MODEL_DIR@/model.pt", f"@WAXAL_MODEL_DIR@/{checkpoint}")
        (tmp_path / "card_src.yaml").write_text(rendered)
        render_asset_card(tmp_path / "card_src.yaml", weights_dir, tmp_path / "assets")

        card_name = next(
            line.split(":", 1)[1].strip()
            for line in rendered.splitlines()
            if line.startswith("name:") and "tokenizer" not in line)

        world_size = max(1, min(torch.cuda.device_count(), len(audio)))
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        if world_size == 1:
            decode_shard(0, 1, card_name, config, audio, shard_dir, batch_size)
        else:
            mp.spawn(decode_shard,
                     args=(world_size, card_name, config, audio, shard_dir, batch_size),
                     nprocs=world_size, join=True)

        merged: dict[str, str] = {}
        for path in sorted(shard_dir.glob("rank_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                merged[record["ID"]] = record["Target"]

    missing = [i for i, _ in audio if i not in merged]
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) produced no hypothesis, first: {missing[:3]}")
    return merged


def main() -> int:
    require_omni_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--route", type=Path)
    parser.add_argument("--lane")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    weights_dir = args.weights or (ROOT / "weights" / config["repo"].split("/")[-1])
    if not weights_dir.is_dir():
        raise SystemExit(f"weights not found at {weights_dir}; "
                         f"run: python models/download_models.py")

    audio = resolve_audio(args.audio, clip_ids_for_lane(args.route, args.lane))
    checkpoints = config["weights"]
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]

    # Batch size comes from the config, where it is part of the decode contract.
    batch_size = int(config["decode"].get("batch_size", args.batch_size))
    surfaces = [run_one_checkpoint(config, c, audio, weights_dir, batch_size)
                for c in checkpoints]

    if len(surfaces) == 1:
        final = surfaces[0]
    else:
        combine = config["decode"].get("combine")
        if combine != "conservative_word_rover":
            raise SystemExit(f"{len(surfaces)} checkpoints need a 'combine' rule, got {combine!r}")
        from inference.fuse.fuse import conservative_word_rover
        final = {identifier: conservative_word_rover([s[identifier] for s in surfaces])
                 for identifier, _ in audio}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"], lineterminator="\n")
        writer.writeheader()
        for identifier, _ in audio:
            writer.writerow({"ID": identifier, "Target": final[identifier]})

    print(f"decoded {len(audio)} clips with {config['model']} "
          f"({len(checkpoints)} checkpoint(s)) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
