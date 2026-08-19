#!/usr/bin/env python3
"""Fail-closed uniform averaging for three MMS native-adapter checkpoints."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[3]
ALGORITHM = "uniform-parameter-mean-float64-v1"
SOURCE_IMPLEMENTATION = {
    "repository": "WAXAL3",
    "path": "experiments/omniasr_ctc_300m_v2/code/checkpoint_average.py",
}


class CheckpointAverageError(RuntimeError):
    """Raised when source selection, averaging, publication, or reload fails."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckpointAverageError(f"JSON object required: {path}")
    return value


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as exc:
        raise CheckpointAverageError(f"artifact escapes WAXAL3: {resolved}") from exc


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
    """Hash tensor names, schemas, and values independently of file encoding."""

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


def validate_state_contract(
    states: Sequence[Mapping[str, Tensor]],
) -> dict[str, Any]:
    if len(states) != 3:
        raise CheckpointAverageError("exactly three model states are required")
    schemas = [tensor_schema(state) for state in states]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise CheckpointAverageError("source tensor key/shape/dtype schema drift")
    if any(
        tensor.layout != torch.strided for state in states for tensor in state.values()
    ):
        raise CheckpointAverageError("only dense strided tensors are supported")
    if any(torch.is_complex(tensor) for state in states for tensor in state.values()):
        raise CheckpointAverageError("complex tensors are unsupported")
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
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    with torch.inference_mode():
        for item in contract["schema"]:
            name = str(item["name"])
            tensors = [state[name] for state in states]
            reference = tensors[0]
            if torch.is_floating_point(reference):
                floating += 1
                accumulator = reference.to(dtype=torch.float64)
                if not bool(torch.isfinite(accumulator).all()):
                    raise CheckpointAverageError(
                        f"non-finite source tensor {name} at source 0"
                    )
                for index, tensor in enumerate(tensors[1:], 1):
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


def _ranked_top_three(rankings: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        rows = rankings["rankings"]["target_weighted_raw_q"][:3]
    except (KeyError, TypeError) as exc:
        raise CheckpointAverageError("checkpoint rankings schema drift") from exc
    if len(rows) != 3 or [int(row["rank"]) for row in rows] != [1, 2, 3]:
        raise CheckpointAverageError("checkpoint rankings do not contain top three")
    if any(not bool(row.get("checkpoint_retained")) for row in rows):
        raise CheckpointAverageError("a selected top-three checkpoint was not retained")
    return rows


def resolve_sources(
    checkpoint_ids: Sequence[str],
    *,
    experiment_id: str,
    run_id: str,
    packet_digest: str,
    rankings_path: Path,
    winner_status_path: Path,
    pair_summary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    if len(checkpoint_ids) != 3 or len(set(checkpoint_ids)) != 3:
        raise CheckpointAverageError("three unique checkpoint IDs are required")
    if any(not value.startswith("CKPT") for value in checkpoint_ids):
        raise CheckpointAverageError("invalid checkpoint ID")

    rankings_path = rankings_path.resolve()
    winner_status_path = winner_status_path.resolve()
    pair_summary_path = pair_summary_path.resolve()
    rankings = read_json(rankings_path)
    top = _ranked_top_three(rankings)
    top_steps = [int(row["step"]) for row in top]

    status = read_json(winner_status_path)
    if (
        status.get("experiment_id") != experiment_id
        or status.get("production_run_id") != run_id
        or status.get("packet_digest") != packet_digest
        or status.get("recipe_comparison_status") != "WINNER"
        or [int(value) for value in status.get("selected_average_steps", [])]
        != top_steps
    ):
        raise CheckpointAverageError("winning-recipe status or top-three selection drift")

    pair = read_json(pair_summary_path)
    metrics = pair.get("metrics", {})
    baseline = metrics.get("baseline_target_weighted", {})
    candidate = metrics.get("candidate_target_weighted", {})
    if (
        not str(pair.get("baseline_id", "")).startswith(f"{experiment_id}_{run_id}")
        or int(pair.get("rows", 0)) != 900
        or float(metrics.get("target_weighted_score_delta", 0.0)) >= 0.0
        or float(baseline.get("score", 0.0))
        <= float(candidate.get("score", 0.0))
    ):
        raise CheckpointAverageError("paired recipe comparison does not select source run")

    run_record_path = ROOT / "registry/runs" / f"{run_id}.json"
    run_record = read_json(run_record_path)
    source_run = ROOT / str(run_record.get("run_dir", ""))
    if (
        run_record.get("id") != run_id
        or run_record.get("experiment_id") != experiment_id
        or run_record.get("packet_digest") != packet_digest
        or not source_run.is_dir()
        or rankings_path != source_run / "checkpoint_rankings.json"
        or winner_status_path.parent != source_run.parent.parent
    ):
        raise CheckpointAverageError("source run registry identity drift")
    final = read_json(source_run / "FINAL.json")
    if (
        final.get("completion_status") != "FIXED_HORIZON_REACHED"
        or int(final.get("global_step", 0)) < max(top_steps)
        or not set(top_steps).issubset(
            {int(value) for value in final.get("retained_checkpoint_steps", [])}
        )
    ):
        raise CheckpointAverageError("source run terminal or retention drift")

    records: list[dict[str, Any]] = []
    for checkpoint_id, ranking in zip(checkpoint_ids, top, strict=True):
        expected_step = int(ranking["step"])
        record_path = ROOT / "registry/checkpoints" / f"{checkpoint_id}.json"
        record = read_json(record_path)
        checkpoint_path = ROOT / str(record.get("checkpoint_path", ""))
        expected_path = (
            source_run
            / "checkpoints"
            / f"checkpoint-{expected_step}"
            / "model.safetensors"
        )
        ranked_files = {
            str(item["name"]): item
            for item in ranking["checkpoint_hashes"]["files"]
        }
        ranked_weight = ranked_files.get("model.safetensors", {})
        observed_hash = sha256_file(checkpoint_path)
        if (
            record.get("id") != checkpoint_id
            or record.get("experiment_id") != experiment_id
            or record.get("run_id") != run_id
            or record.get("packet_digest") != packet_digest
            or int(record.get("step", -1)) != expected_step
            or checkpoint_path.resolve() != expected_path.resolve()
            or not checkpoint_path.is_file()
            or record.get("checkpoint_sha256") != observed_hash
            or ranked_weight.get("sha256") != observed_hash
            or int(ranked_weight.get("bytes", -1)) != checkpoint_path.stat().st_size
        ):
            raise CheckpointAverageError(f"checkpoint source drift: {checkpoint_id}")
        records.append(
            {
                "rank": int(ranking["rank"]),
                "checkpoint_id": checkpoint_id,
                "step": expected_step,
                "target_weighted_raw_q": float(ranking["value"]),
                "checkpoint": relative(checkpoint_path),
                "checkpoint_sha256": observed_hash,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "registry_record": relative(record_path),
                "registry_record_sha256": sha256_file(record_path),
            }
        )
    return records, {
        "selection_metric": "target_weighted_raw_q",
        "selection_tie_break": "earlier_global_step",
        "selected_steps_in_rank_order": top_steps,
        "rankings": relative(rankings_path),
        "rankings_sha256": sha256_file(rankings_path),
        "winner_status": relative(winner_status_path),
        "winner_status_sha256": sha256_file(winner_status_path),
        "recipe_pair_summary": relative(pair_summary_path),
        "recipe_pair_summary_sha256": sha256_file(pair_summary_path),
        "source_run_registry": relative(run_record_path),
        "source_run_registry_sha256": sha256_file(run_record_path),
        "source_final": relative(source_run / "FINAL.json"),
        "source_final_sha256": sha256_file(source_run / "FINAL.json"),
    }, source_run


def _copy_portable_files(source_run: Path, source_checkpoint: Path, target: Path) -> None:
    config_hashes = []
    for checkpoint in source_run.joinpath("checkpoints").glob("checkpoint-*/config.json"):
        config_hashes.append(sha256_file(checkpoint))
    if not config_hashes or len(set(config_hashes)) != 1:
        raise CheckpointAverageError("source checkpoint configs are not identical")
    shutil.copyfile(source_checkpoint.parent / "config.json", target / "config.json")
    for name in ("preprocessor_config.json", "vocab.json"):
        source = source_run / "best_model" / name
        if not source.is_file():
            raise CheckpointAverageError(f"missing portable model artifact: {source}")
        shutil.copyfile(source, target / name)


def _strict_reload_and_forward(model_dir: Path, expected_content_hash: str) -> dict[str, Any]:
    reloaded = load_file(model_dir / "model.safetensors", device="cpu")
    observed_content_hash = tensor_content_sha256(reloaded)
    if observed_content_hash != expected_content_hash:
        raise CheckpointAverageError("averaged tensor content changed after reload")

    from transformers import Wav2Vec2ForCTC

    model, loading = Wav2Vec2ForCTC.from_pretrained(
        model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise CheckpointAverageError(f"strict Hugging Face reload failed: {loading}")
    model.eval()
    samples = torch.linspace(-0.1, 0.1, 16_000, dtype=torch.float32).unsqueeze(0)
    attention = torch.ones_like(samples, dtype=torch.long)
    with torch.inference_mode():
        logits = model(samples, attention_mask=attention).logits
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[-1] != int(model.config.vocab_size)
    ):
        raise CheckpointAverageError(f"unexpected CPU-forward shape: {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise CheckpointAverageError("non-finite CPU-forward logits")
    return {
        "strict_tensor_content_reload": True,
        "strict_huggingface_reload": True,
        "cpu_forward_finite": True,
        "cpu_forward_logits_shape": list(logits.shape),
        "loading_info": loading,
    }


def build_average(
    *,
    checkpoint_ids: Sequence[str],
    experiment_id: str,
    run_id: str,
    packet_digest: str,
    rankings_path: Path,
    winner_status_path: Path,
    pair_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"create-only output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    records, selection, source_run = resolve_sources(
        checkpoint_ids,
        experiment_id=experiment_id,
        run_id=run_id,
        packet_digest=packet_digest,
        rankings_path=rankings_path,
        winner_status_path=winner_status_path,
        pair_summary_path=pair_summary_path,
    )
    if (
        output_dir.name != "model"
        or output_dir.parent.parent.resolve() != (source_run / "derived").resolve()
    ):
        raise CheckpointAverageError(
            "output must be runs/<source-run>/derived/<candidate>/model"
        )

    states = [
        load_file(ROOT / record["checkpoint"], device="cpu") for record in records
    ]
    averaged, contract = average_states(states)
    content_sha256 = tensor_content_sha256(averaged)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        checkpoint_path = temporary / "model.safetensors"
        save_file(
            averaged,
            checkpoint_path,
            metadata={
                "format": "pt",
                "waxal3_algorithm": ALGORITHM,
                "waxal3_experiment_id": experiment_id,
                "waxal3_source_run_id": run_id,
            },
        )
        _copy_portable_files(
            source_run, ROOT / records[0]["checkpoint"], temporary
        )
        reload_evidence = _strict_reload_and_forward(temporary, content_sha256)
        artifacts = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "mms_native_adapter_checkpoint_parameter_average",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": ALGORITHM,
            "source_implementation": {
                **SOURCE_IMPLEMENTATION,
                "sha256": sha256_file(ROOT / SOURCE_IMPLEMENTATION["path"]),
            },
            "experiment_id": experiment_id,
            "source_run_id": run_id,
            "packet_digest": packet_digest,
            "source_count": 3,
            "uniform_weight": 1 / 3,
            "sources": records,
            "selection": selection,
            "accumulator_dtype": "torch.float64",
            "output_dtype_policy": "cast_each_average_to_source_dtype",
            "non_floating_policy": "require_exact_match",
            "tensor_schema_sha256": contract["schema_sha256"],
            "tensor_count": contract["tensor_count"],
            "parameter_count": contract["parameter_count"],
            "floating_tensor_count": contract["floating_tensor_count"],
            "non_floating_tensor_count": contract["non_floating_tensor_count"],
            "output_checkpoint": "model.safetensors",
            "output_checkpoint_sha256": sha256_file(checkpoint_path),
            "output_tensor_content_sha256": content_sha256,
            "output_artifacts_before_manifest": artifacts,
            **reload_evidence,
            "finite_sources_and_output": True,
            "implementation": relative(Path(__file__)),
            "implementation_sha256": sha256_file(Path(__file__)),
            "submission_created": False,
        }
        (temporary / "average_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-id", action="append", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--packet-digest", required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--winner-status", type=Path, required=True)
    parser.add_argument("--recipe-pair-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_average(
        checkpoint_ids=args.checkpoint_id,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        packet_digest=args.packet_digest,
        rankings_path=(ROOT / args.rankings).resolve(),
        winner_status_path=(ROOT / args.winner_status).resolve(),
        pair_summary_path=(ROOT / args.recipe_pair_summary).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
