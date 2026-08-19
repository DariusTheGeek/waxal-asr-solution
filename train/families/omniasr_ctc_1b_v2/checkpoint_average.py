#!/usr/bin/env python3
"""Fail-closed uniform parameter averaging for WAXAL3 CTC checkpoints."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

import torch
from torch import Tensor

from consistency import TRAINING_ONLY_MASK_KEY, strip_training_only_masker


ROOT = Path(__file__).resolve().parents[3]
ALGORITHM = "uniform-parameter-mean-float64-v1"
SOURCE_IMPLEMENTATION = {
    "repository": "WAXAL2",
    "path": "experiments/amn/full/mono/S004_mms300_cv007/src/checkpoint_average.py",
    "sha256": "dea53c2e7bea9073fcb7e4f0d3c748e89087def7bb919a5efe2d0882deae36cb",
}


class CheckpointAverageError(RuntimeError):
    """Raised when any source, tensor, publication, or reload gate fails."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckpointAverageError(f"JSON object required: {path}")
    return value


def load_checkpoint(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def extract_model_state(checkpoint: object) -> Mapping[str, Tensor]:
    candidate: object = checkpoint
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        candidate = checkpoint["model"]
    if not isinstance(candidate, Mapping) or not candidate:
        raise CheckpointAverageError("checkpoint lacks a non-empty model state")
    if not all(
        isinstance(name, str) and torch.is_tensor(tensor)
        for name, tensor in candidate.items()
    ):
        raise CheckpointAverageError("model state must map names to tensors only")
    return candidate  # type: ignore[return-value]


def tensor_schema(state: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(state[name].shape),
            "dtype": str(state[name].dtype),
            "layout": str(state[name].layout),
        }
        for name in sorted(state)
    ]


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_content_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, schemas, and values independent of torch encoding."""

    digest = hashlib.sha256()
    with torch.inference_mode():
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            descriptor = json.dumps(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            digest.update(len(descriptor).to_bytes(8, "big"))
            digest.update(descriptor)
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def validate_selected_steps(
    observed_steps: Sequence[int],
    curve_ranked_steps: Sequence[int],
    gate_ranked_steps: Sequence[int],
) -> None:
    """Require the same top-three set while keeping source reads chronological.

    Uniform averaging is order independent. Registry sources are deliberately
    consumed in ascending step order for deterministic I/O, whereas the curve
    and production gate list them in descending score order. Requiring both
    lists to have identical ordering incorrectly rejects any run whose best
    checkpoint is not also the earliest selected checkpoint.
    """

    observed = [int(step) for step in observed_steps]
    curve_ranked = [int(step) for step in curve_ranked_steps]
    gate_ranked = [int(step) for step in gate_ranked_steps]
    if len(observed) != 3 or len(set(observed)) != 3:
        raise CheckpointAverageError("three unique observed steps are required")
    if observed != sorted(observed):
        raise CheckpointAverageError("source checkpoint steps must be chronological")
    if len(curve_ranked) != 3 or set(observed) != set(curve_ranked):
        raise CheckpointAverageError(
            f"checkpoint steps {observed} are not curve top three {curve_ranked}"
        )
    if len(gate_ranked) != 3 or set(observed) != set(gate_ranked):
        raise CheckpointAverageError("production gate top-three checkpoint drift")


def validate_state_contract(states: Sequence[Mapping[str, Tensor]]) -> dict[str, Any]:
    if len(states) != 3:
        raise CheckpointAverageError("exactly three model states are required")
    schemas = [tensor_schema(state) for state in states]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise CheckpointAverageError("source tensor key/shape/dtype schema drift")
    if any(
        tensor.layout != torch.strided for state in states for tensor in state.values()
    ):
        raise CheckpointAverageError("only dense strided tensors are supported")
    return {
        "schema": schemas[0],
        "schema_sha256": canonical_json_sha256(schemas[0]),
        "tensor_count": len(schemas[0]),
        "parameter_count": sum(tensor.numel() for tensor in states[0].values()),
    }


def average_states(
    states: Sequence[Mapping[str, Tensor]],
) -> tuple[OrderedDict[str, Tensor], dict[str, Any]]:
    """Uniformly average floats in float64 and require exact non-float state."""

    contract = validate_state_contract(states)
    output: OrderedDict[str, Tensor] = OrderedDict()
    floating = 0
    non_floating = 0
    with torch.inference_mode():
        for item in contract["schema"]:
            name = str(item["name"])
            tensors = [state[name] for state in states]
            reference = tensors[0]
            if torch.is_complex(reference):
                raise CheckpointAverageError(f"complex tensor is unsupported: {name}")
            if torch.is_floating_point(reference):
                floating += 1
                accumulator = torch.zeros(reference.shape, dtype=torch.float64)
                for index, tensor in enumerate(tensors):
                    if not bool(torch.isfinite(tensor).all()):
                        raise CheckpointAverageError(
                            f"non-finite source tensor {name} at source {index}"
                        )
                    accumulator.add_(tensor.to(dtype=torch.float64))
                averaged = accumulator.div_(len(tensors)).to(reference.dtype)
                if not bool(torch.isfinite(averaged).all()):
                    raise CheckpointAverageError(f"non-finite average: {name}")
                output[name] = averaged.contiguous()
            else:
                non_floating += 1
                if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                    raise CheckpointAverageError(
                        f"non-floating source tensor differs: {name}"
                    )
                output[name] = reference.detach().clone().contiguous()
    contract.update(
        {
            "floating_tensor_count": floating,
            "non_floating_tensor_count": non_floating,
        }
    )
    return output, contract


def resolve_sources(
    checkpoint_ids: Sequence[str],
    *,
    experiment_id: str,
    run_id: str,
    packet_digest: str,
    curve_path: Path,
    gate_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(checkpoint_ids) != 3 or len(set(checkpoint_ids)) != 3:
        raise CheckpointAverageError("three unique checkpoint IDs are required")
    if any(not value.startswith("CKPT") for value in checkpoint_ids):
        raise CheckpointAverageError("invalid checkpoint ID")
    curve_path = curve_path.resolve()
    gate_path = gate_path.resolve()
    gate = _json(gate_path)
    if (
        gate.get("status") != "PASS"
        or gate.get("experiment_id") != experiment_id
        or gate.get("production_run_id") != run_id
        or gate.get("packet_digest") != packet_digest
    ):
        raise CheckpointAverageError("L1 production gate identity drift")

    with curve_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {"checkpoint_step", "weighted_score"} - set(rows[0]):
        raise CheckpointAverageError("curve lacks checkpoint_step/weighted_score")
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["weighted_score"]), int(row["checkpoint_step"])),
    )
    curve_top_steps = [int(row["checkpoint_step"]) for row in ranked[:3]]

    records: list[dict[str, Any]] = []
    for checkpoint_id in checkpoint_ids:
        record_path = ROOT / "registry/checkpoints" / f"{checkpoint_id}.json"
        record = _json(record_path)
        checkpoint_path = ROOT / str(record.get("checkpoint_path", ""))
        if (
            record.get("id") != checkpoint_id
            or record.get("experiment_id") != experiment_id
            or record.get("run_id") != run_id
            or record.get("packet_digest") != packet_digest
            or not checkpoint_path.is_file()
            or record.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        ):
            raise CheckpointAverageError(f"checkpoint registry drift: {checkpoint_id}")
        records.append(
            {
                "checkpoint_id": checkpoint_id,
                "step": int(record["step"]),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "registry_record": str(record_path.relative_to(ROOT)),
                "registry_record_sha256": sha256_file(record_path),
            }
        )
    observed_steps = [record["step"] for record in records]
    gate_top = [
        int(value) for value in gate.get("curve", {}).get("top_five_steps", [])[:3]
    ]
    validate_selected_steps(observed_steps, curve_top_steps, gate_top)
    return records, {
        "curve": str(curve_path.relative_to(ROOT)),
        "curve_sha256": sha256_file(curve_path),
        "selection_metric": "target_slot_weighted_raw_q",
        "selected_steps": observed_steps,
        "score_ranked_selected_steps": curve_top_steps,
        "gate": str(gate_path.relative_to(ROOT)),
        "gate_sha256": sha256_file(gate_path),
    }


def _forward(
    state: Mapping[str, Tensor], *, card_dir: Path, model_card: str
) -> tuple[Tensor, list[int]]:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.environ["FAIRSEQ2_ASSET_DIR"] = str(card_dir.resolve())
    import omnilingual_asr  # noqa: F401
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout

    model = load_model(
        model_card,
        device=torch.device("cpu"),
        dtype=torch.float32,
        mmap=True,
        progress=False,
    )
    inference_state = strip_training_only_masker(
        state, require_masker=TRAINING_ONLY_MASK_KEY in state
    )
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointAverageError(
            f"strict averaged-state load failed: {incompatible}"
        )
    model.eval()
    samples = torch.linspace(-0.1, 0.1, 16_000, dtype=torch.float32).unsqueeze(0)
    layout = BatchLayout.of(samples, [samples.shape[1]])
    with torch.inference_mode():
        logits, output_layout = model(samples, layout)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 10_288:
        raise CheckpointAverageError(
            f"unexpected CPU-forward shape: {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise CheckpointAverageError("non-finite CPU-forward logits")
    return logits.detach().clone(), [int(value) for value in output_layout.seq_lens]


def build_average(
    *,
    checkpoint_ids: Sequence[str],
    experiment_id: str,
    run_id: str,
    packet_digest: str,
    curve_path: Path,
    gate_path: Path,
    card_dir: Path,
    model_card: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"create-only output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_records, selection = resolve_sources(
        checkpoint_ids,
        experiment_id=experiment_id,
        run_id=run_id,
        packet_digest=packet_digest,
        curve_path=curve_path,
        gate_path=gate_path,
    )
    checkpoints = [
        load_checkpoint(ROOT / record["checkpoint"]) for record in source_records
    ]
    states = [extract_model_state(checkpoint) for checkpoint in checkpoints]
    averaged, contract = average_states(states)
    content_sha256 = tensor_content_sha256(averaged)
    logits_before, lengths_before = _forward(
        averaged, card_dir=card_dir, model_card=model_card
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        checkpoint_path = temporary / "model.pt"
        torch.save({"model": averaged, "fs2": True}, checkpoint_path)
        reloaded = extract_model_state(load_checkpoint(checkpoint_path))
        if tensor_schema(reloaded) != contract["schema"]:
            raise CheckpointAverageError("averaged checkpoint schema changed on reload")
        for name in averaged:
            if not torch.equal(averaged[name], reloaded[name]):
                raise CheckpointAverageError(
                    f"averaged tensor changed on reload: {name}"
                )
        if tensor_content_sha256(reloaded) != content_sha256:
            raise CheckpointAverageError(
                "averaged tensor-content hash changed on reload"
            )
        logits_after, lengths_after = _forward(
            reloaded, card_dir=card_dir, model_card=model_card
        )
        if lengths_after != lengths_before or not torch.equal(
            logits_after, logits_before
        ):
            raise CheckpointAverageError(
                "CPU forward changed after averaged-state reload"
            )

        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "ctc_checkpoint_parameter_average",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": ALGORITHM,
            "source_implementation": SOURCE_IMPLEMENTATION,
            "experiment_id": experiment_id,
            "source_run_id": run_id,
            "packet_digest": packet_digest,
            "source_count": 3,
            "uniform_weight": 1 / 3,
            "sources": source_records,
            "selection": selection,
            "accumulator_dtype": "torch.float64",
            "output_dtype_policy": "cast_each_average_to_source_dtype",
            "non_floating_policy": "require_exact_match",
            "tensor_schema_sha256": contract["schema_sha256"],
            "tensor_count": contract["tensor_count"],
            "parameter_count": contract["parameter_count"],
            "floating_tensor_count": contract["floating_tensor_count"],
            "non_floating_tensor_count": contract["non_floating_tensor_count"],
            "training_only_state_keys": (
                [TRAINING_ONLY_MASK_KEY] if TRAINING_ONLY_MASK_KEY in averaged else []
            ),
            "inference_export_required": TRAINING_ONLY_MASK_KEY in averaged,
            "output_checkpoint": "model.pt",
            "output_checkpoint_sha256": sha256_file(checkpoint_path),
            "output_tensor_content_sha256": content_sha256,
            "strict_reload_equal": True,
            "finite_sources_and_output": True,
            "cpu_forward_reload_equivalent": True,
            "cpu_forward_logits_shape": list(logits_before.shape),
            "cpu_forward_output_lengths": lengths_before,
            "implementation_sha256": sha256_file(Path(__file__)),
        }
        manifest_path = temporary / "average_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
        return manifest
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (temporary / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        failed = output_dir.with_name(f"{output_dir.name}.failed.{uuid.uuid4().hex}")
        os.rename(temporary, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-id", action="append", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--packet-digest", required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--card-dir", type=Path, required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_average(
        checkpoint_ids=args.checkpoint_id,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        packet_digest=args.packet_digest,
        curve_path=(ROOT / args.curve).resolve(),
        gate_path=(ROOT / args.gate).resolve(),
        card_dir=(ROOT / args.card_dir).resolve(),
        model_card=args.model_card,
        output_dir=(ROOT / args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
