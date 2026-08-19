#!/usr/bin/env python3
"""Create one byte-identical language CTC-head initialization for all MMS arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from supervised.mms_adapter import inspect_head_overlap


ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_LANGUAGES = {"lin", "sna"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=sorted(SUPPORTED_LANGUAGES), required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
    )
    parser.add_argument(
        "--source-vocab",
        type=Path,
        default=ROOT / "models/mms-1b-all/vocab.json",
    )
    parser.add_argument(
        "--target-vocab",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--seed", type=int, default=42003)
    args = parser.parse_args()
    language = str(args.language)
    adapter = (
        args.adapter
        or ROOT / f"models/mms-1b-all/adapter.{language}.safetensors"
    ).resolve()
    target_vocab = (
        args.target_vocab
        or ROOT / f"data/derived/mms/{language}_cv002_native_adapter_v1/vocab.json"
    ).resolve()
    output = (
        args.output
        or ROOT / f"data/derived/mms/{language}_cv002_head_init_v1"
    ).resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(f".{output.name}.building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)

    source_vocab = args.source_vocab.resolve()
    report = inspect_head_overlap(
        adapter_path=adapter,
        source_vocab_path=source_vocab,
        target_vocab_path=target_vocab,
        language=language,
    )
    target_rows = int(report["target_head_rows"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(args.seed))
        head = torch.nn.Linear(1280, target_rows, bias=True, device="cpu")
    native = load_file(str(adapter), device="cpu")
    with torch.no_grad():
        for item in report["mapping"]:
            source_id = int(item["source_id"])
            target_id = int(item["target_id"])
            head.weight[target_id].copy_(native["lm_head.weight"][source_id])
            head.bias[target_id].copy_(native["lm_head.bias"][source_id])
    state = {
        "lm_head.bias": head.bias.detach().contiguous(),
        "lm_head.weight": head.weight.detach().contiguous(),
    }
    artifact = building / "head.safetensors"
    save_file(
        state,
        str(artifact),
        metadata={
            "schema_version": "1",
            "language": language,
            "seed": str(args.seed),
            "mapping_sha256": str(report["mapping_sha256"]),
        },
    )
    reloaded = load_file(str(artifact), device="cpu")
    if set(reloaded) != set(state) or any(
        not torch.equal(reloaded[name], value) for name, value in state.items()
    ):
        raise RuntimeError("head-init reload identity failed")
    record = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "seed": int(args.seed),
        "algorithm": "torch.nn.Linear.reset_parameters_then_native_overlap_copy",
        "torch_version": torch.__version__,
        "fan_in": 1280,
        "fresh_bias_bound": 1.0 / math.sqrt(1280),
        "adapter_path": str(adapter.relative_to(ROOT)),
        "adapter_sha256": sha256_file(adapter),
        "source_vocab_path": str(source_vocab.relative_to(ROOT)),
        "source_vocab_sha256": sha256_file(source_vocab),
        "target_vocab_path": str(target_vocab.relative_to(ROOT)),
        "target_vocab_sha256": sha256_file(target_vocab),
        "mapping_sha256": report["mapping_sha256"],
        "source_head_rows": report["source_head_rows"],
        "target_head_rows": target_rows,
        "mapped_head_rows": report["mapped_head_rows"],
        "fresh_head_rows": report["fresh_head_rows"],
        "fresh_tokens": report["fresh_tokens"],
        "artifact": {
            "path": "head.safetensors",
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
    }
    (building / "BUILD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    building.replace(output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
