#!/usr/bin/env python3
"""Create a self-contained FP64 mean from three exported OmniASR LLM models.

The input contract is intentionally exported ``model.pt`` files plus their
export/reload evidence.  FSDP shards, trainer state, and optimizer state are
not accepted or needed by this program.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch


EXPECTED_PARAMETERS = 4_380_578_432
ALGORITHM = "uniform-parameter-mean-float64-v1"


def repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError("unable to locate WAXAL3 root")


ROOT = repo_root(Path(__file__).resolve())


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_file(value: Path | str, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes WAXAL3: {value}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {resolved}")
    return resolved


def resolve_repo_dir(value: Path | str, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes WAXAL3: {value}") from error
    if resolved.is_symlink() or not resolved.is_dir():
        raise FileNotFoundError(f"missing or unsafe {label}: {resolved}")
    return resolved


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def ensure_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_export(
    model: Path, expected_sha256: str, expected_step: int
) -> dict[str, Any]:
    export_dir = model.parent
    export_path = export_dir / "EXPORT.json"
    verify_path = export_dir / "VERIFY.json"
    if export_path.is_symlink() or verify_path.is_symlink():
        raise RuntimeError("export evidence may not be symlinked")
    export = read_json(export_path)
    verify = read_json(verify_path)
    observed = sha256_file(model)
    if observed != expected_sha256:
        raise RuntimeError(f"source model SHA-256 drift: {model}")
    if (
        export.get("status") != "PASS"
        or export.get("model_file") != "model.pt"
        or export.get("model_sha256") != expected_sha256
        or int(export.get("source_checkpoint_step", -1)) != expected_step
        or int(export.get("state_parameter_values", -1)) != EXPECTED_PARAMETERS
        or int(export.get("state_tensors", 0)) <= 0
        or verify.get("status") != "PASS"
        or verify.get("strict_load") is not True
        or verify.get("model_sha256") != expected_sha256
        or int(verify.get("source_checkpoint_step", -1)) != expected_step
        or int(verify.get("model_parameters", -1)) != EXPECTED_PARAMETERS
        or float(verify.get("maximum_absolute_logit_delta", float("inf"))) != 0.0
        or float(verify.get("absolute_loss_delta", float("inf"))) != 0.0
    ):
        raise RuntimeError(f"standalone export/reload evidence drift: {export_dir}")
    return {
        "step": expected_step,
        "model": relative(model),
        "model_sha256": expected_sha256,
        "model_bytes": model.stat().st_size,
        "export_record": relative(export_path),
        "export_record_sha256": sha256_file(export_path),
        "verify_record": relative(verify_path),
        "verify_record_sha256": sha256_file(verify_path),
        "source_packet_digest": export.get("packet_digest"),
        "source_run_id": export.get("source_run_id"),
    }


def load_primitives(model_family_src: Path) -> tuple[Any, ...]:
    source = str(model_family_src)
    if source not in sys.path:
        sys.path.insert(0, source)
    from checkpoint_average import (  # noqa: PLC0415
        _forward,
        average_states,
        extract_model_state,
        load_checkpoint,
        tensor_content_sha256,
        tensor_schema,
    )
    from runtime_assets import render_asset_cards  # noqa: PLC0415

    return (
        _forward,
        average_states,
        extract_model_state,
        load_checkpoint,
        tensor_content_sha256,
        tensor_schema,
        render_asset_cards,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--model-sha256", action="append", required=True)
    parser.add_argument("--step", type=int, action="append", required=True)
    parser.add_argument("--model-family-src", type=Path, required=True)
    parser.add_argument("--card-template", type=Path, required=True)
    parser.add_argument("--parent-card", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-card", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language-code", default="lin_Latn")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-packet-digest", required=True)
    args = parser.parse_args()

    if not (
        len(args.model) == len(args.model_sha256) == len(args.step) == 3
        and len(set(args.step)) == 3
    ):
        raise RuntimeError("exactly three distinct model/SHA-256/step values are required")
    if not args.source_run_id.startswith("RUN"):
        raise RuntimeError("source run ID is invalid")
    ensure_sha256(args.source_packet_digest, label="source packet digest")
    expected_hashes = [ensure_sha256(value, label="source model SHA-256") for value in args.model_sha256]
    models = [resolve_repo_file(value, label="standalone model") for value in args.model]
    if len(set(models)) != 3:
        raise RuntimeError("three distinct standalone model paths are required")
    family_src = resolve_repo_dir(args.model_family_src, label="model-family source")
    template = resolve_repo_file(args.card_template, label="asset-card template")
    parent_checkpoint = resolve_repo_file(args.parent_checkpoint, label="parent checkpoint")
    if not args.candidate_card.startswith("waxal3_"):
        raise RuntimeError("candidate model card is invalid")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError("average output escapes WAXAL3") from error
    if output.exists():
        raise FileExistsError(output)
    if output.parent.is_symlink():
        raise RuntimeError("average output parent may not be symlinked")

    sources = [
        validate_export(model, digest, step)
        for model, digest, step in zip(models, expected_hashes, args.step, strict=True)
    ]
    if any(
        source["source_packet_digest"] != args.source_packet_digest for source in sources
    ):
        raise RuntimeError("source export packet digest drift")
    (
        forward,
        average_states,
        extract_model_state,
        load_checkpoint,
        tensor_content_sha256,
        tensor_schema,
        render_asset_cards,
    ) = load_primitives(family_src)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        states = [extract_model_state(load_checkpoint(model)) for model in models]
        averaged, contract = average_states(states)
        parameter_count = sum(tensor.numel() for tensor in averaged.values())
        if parameter_count != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"averaged parameter count drift: {parameter_count} != {EXPECTED_PARAMETERS}"
            )
        tensor_sha256 = tensor_content_sha256(averaged)
        checkpoint = temporary / "model.pt"
        torch.save({"model": averaged, "fs2": True}, checkpoint)
        reloaded = extract_model_state(load_checkpoint(checkpoint))
        if tensor_schema(reloaded) != contract["schema"]:
            raise RuntimeError("average schema changed on strict reload")
        if any(not torch.equal(averaged[name], reloaded[name]) for name in averaged):
            raise RuntimeError("average tensor changed on strict reload")
        if tensor_content_sha256(reloaded) != tensor_sha256:
            raise RuntimeError("average tensor-content hash changed on strict reload")

        cards = temporary / "asset_cards"
        rendered = render_asset_cards(template, cards)
        card_text = rendered.read_text(encoding="utf-8")
        card_text = card_text.replace(args.parent_card, args.candidate_card).replace(
            str(parent_checkpoint), str(checkpoint)
        )
        if args.candidate_card not in card_text or str(checkpoint) not in card_text:
            raise RuntimeError("average model-card rendering drift")
        rendered.write_text(card_text, encoding="utf-8")
        before = forward(
            reloaded,
            card_dir=cards,
            model_card=args.candidate_card,
        )
        again = extract_model_state(load_checkpoint(checkpoint))
        after = forward(
            again,
            card_dir=cards,
            model_card=args.candidate_card,
        )
        if (
            before[2] != after[2]
            or not torch.equal(before[0], after[0])
            or not torch.equal(before[1], after[1])
        ):
            raise RuntimeError("average strict reload/forward parity failed")

        checkpoint_sha256 = sha256_file(checkpoint)
        average = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "standalone_llm_export_parameter_average",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": ALGORITHM,
            "source_run_id": args.source_run_id,
            "source_packet_digest": args.source_packet_digest,
            "source_count": 3,
            "member_order": [source["step"] for source in sources],
            "uniform_weight": 1 / 3,
            "sources": sources,
            "accumulator_dtype": "torch.float64",
            "output_dtype_policy": "cast_each_average_to_source_dtype",
            "non_floating_policy": "require_exact_match",
            "tensor_count": contract["tensor_count"],
            "parameter_count": parameter_count,
            "floating_tensor_count": contract["floating_tensor_count"],
            "non_floating_tensor_count": contract["non_floating_tensor_count"],
            "output_checkpoint": "model.pt",
            "output_checkpoint_sha256": checkpoint_sha256,
            "output_checkpoint_bytes": checkpoint.stat().st_size,
            "output_tensor_content_sha256": tensor_sha256,
            "model_card": args.candidate_card,
            "language_code": args.language_code,
            "strict_reload_equal": True,
            "strict_reload_forward_parity": True,
            "implementation": relative(Path(__file__)),
            "implementation_sha256": sha256_file(Path(__file__)),
            "averaging_core": relative(family_src / "checkpoint_average.py"),
            "averaging_core_sha256": sha256_file(family_src / "checkpoint_average.py"),
        }
        average_path = temporary / "AVERAGE.json"
        average_path.write_text(
            json.dumps(average, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "standalone_llm_average_reload_forward_parity",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "average_record_sha256": sha256_file(average_path),
            "model_sha256": checkpoint_sha256,
            "model_parameters": parameter_count,
            "strict_load": True,
            "tensor_content_sha256": tensor_sha256,
            "absolute_loss_delta": float((before[0] - after[0]).abs()),
            "maximum_absolute_logit_delta": float((before[1] - after[1]).abs().max()),
            "mean_absolute_logit_delta": float((before[1] - after[1]).abs().mean()),
            "logits_shape": list(before[1].shape),
            "output_seq_lens": before[2],
            "model_card": args.candidate_card,
            "language_code": args.language_code,
            "implementation_sha256": sha256_file(Path(__file__)),
        }
        (temporary / "VERIFY.json").write_text(
            json.dumps(verify, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(average, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
