#!/usr/bin/env python3
"""Locked multi-GPU high-mask SSL training of one MMS-1B native adapter."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pyarrow.parquet as pq
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from .composition import (
    adapter_l2sp_penalty,
    adapter_reference,
    compose_native_adapter_pretraining_model,
    export_adapter_only,
)
from .contract import (
    experiment_root_from,
    git_state,
    hardware_state,
    read_json,
    repo_root,
    require_distributed_runtime,
    require_fresh_pass,
    resume_trajectory_hash,
    resume_trajectory_payload,
    sha256_file,
    training_critical_hash,
    utc_now,
    validate_global_config,
    write_json_create_only,
)
from .data import (
    CPTAudioDataset,
    CPTCollator,
    SpeakerInterleavedDistributedSampler,
    StagePaddedDistributedSampler,
)


def scheduler_multiplier(step: int, *, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max(float(step + 1) / max(1, warmup_steps), 1e-12)
    return max(float(max_steps - step) / max(1, max_steps - warmup_steps), 0.0)


def ddp_mask_normalized_loss(
    local_sum: torch.Tensor,
    global_masks: torch.Tensor,
    *,
    world_size: int,
) -> torch.Tensor:
    """Scale a rank-local summed loss so DDP averaging yields a global mean."""

    if int(world_size) < 1 or global_masks.numel() != 1:
        raise ValueError("invalid distributed normalization dimensions")
    return local_sum.float() * int(world_size) / global_masks.float()


def _append_json_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _best_effort_stdout(value: dict[str, Any]) -> None:
    """Never let an SSH/terminal pipe failure terminate authoritative training."""

    try:
        print(json.dumps(value, sort_keys=True), flush=True)
    except (BrokenPipeError, OSError):
        # metrics.jsonl is durable and authoritative.  Replacing the broken
        # stream also prevents an interpreter-shutdown flush from failing.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _frozen_sentinels(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = [
        "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
        "wav2vec2.encoder.layers.0.attention.k_proj.weight",
        "wav2vec2.encoder.layers.24.attention.k_proj.weight",
        "wav2vec2.encoder.layers.47.attention.k_proj.weight",
        "quantizer.codevectors",
        "project_hid.weight",
        "project_q.weight",
    ]
    named = dict(model.named_parameters())
    missing = sorted(set(names) - set(named))
    if missing or any(named[name].requires_grad for name in names):
        raise RuntimeError(f"frozen sentinel contract failed: missing={missing}")
    return {name: named[name].detach().cpu().clone() for name in names}


def _assert_frozen_sentinels(
    model: torch.nn.Module, reference: dict[str, torch.Tensor]
) -> None:
    named = dict(model.named_parameters())
    changed = [
        name
        for name, initial in reference.items()
        if not torch.equal(named[name].detach().cpu(), initial)
    ]
    if changed:
        raise RuntimeError(f"frozen tensor drift detected: {changed}")


def _load_adapter(model: torch.nn.Module, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    if len(state) != 288 or any("adapter_layer" not in name for name in state):
        raise RuntimeError("resume adapter tensor inventory drift")
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_trainable = [
        name
        for name in result.missing_keys
        if dict(model.named_parameters()).get(name) is not None
        and dict(model.named_parameters())[name].requires_grad
    ]
    if unexpected or missing_trainable:
        raise RuntimeError(
            f"resume adapter load drift: missing={missing_trainable} unexpected={unexpected}"
        )


def _adapter_filename(language: str) -> str:
    return f"adapter.{language}.safetensors"


def _resume_state_names(world_size: int, language: str = "lin") -> set[str]:
    return {
        _adapter_filename(language),
        "optimizer_scheduler.pt",
        *(f"rng.rank-{rank}.pt" for rank in range(int(world_size))),
    }
RECOVERY_SPEC_KEYS = {
    "schema_version",
    "source_experiment_id",
    "source_experiment_path",
    "source_run_id",
    "source_sweep",
    "source_checkpoint_path",
    "source_packet_digest",
    "source_packet_json_sha256",
    "source_packet_audit_sha256",
    "source_profile_sha256",
    "source_launch_sha256",
    "source_start_sha256",
    "source_metrics_sha256",
    "source_config_sha256",
    "source_training_critical_hash",
    "source_resume_trajectory_hash",
    "source_checkpoint_json_sha256",
    "source_state_files",
    "source_collapse_logs",
}


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _verify_packet_inventory(packet: Path, expected_digest: str) -> dict[str, Any]:
    record_path = packet / "PACKET.json"
    record = _read_strict_json(record_path)
    if str(record.get("content_digest")) != expected_digest:
        raise RuntimeError("source packet record digest does not match declared lineage")
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in packet.rglob("*") if item.is_file()):
        relative = path.relative_to(packet).as_posix()
        if relative in {"PACKET.json", "SHA256SUMS"} or relative.startswith("audit/"):
            continue
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if (
        inventory != record.get("inventory")
        or len(inventory) != int(record.get("content_files", -1))
        or digest != expected_digest
    ):
        raise RuntimeError("source packet content/inventory drift")
    return record


def _validate_checkpoint_files(
    checkpoint: Path,
    record: dict[str, Any],
    *,
    horizon_sweeps: int,
    updates_per_sweep: int,
    world_size: int = 4,
    language: str = "lin",
) -> tuple[int, int]:
    sweep = int(record.get("sweep", 0))
    global_step = int(record.get("global_step", 0))
    if (
        int(record.get("schema_version", 0)) != 1
        or record.get("status") != "COMPLETE"
        or record.get("resumable") is not True
        or not 1 <= sweep < horizon_sweeps
        or global_step != sweep * updates_per_sweep
        or checkpoint.name != f"sweep-{sweep:02d}"
    ):
        raise RuntimeError("resume checkpoint is not a complete sweep boundary")
    files = record.get("files")
    expected_state_names = _resume_state_names(world_size, language)
    if not isinstance(files, list) or len(files) != len(expected_state_names):
        raise RuntimeError("resume checkpoint file inventory schema drift")
    names = [str(item.get("name")) for item in files if isinstance(item, dict)]
    if (
        len(names) != len(files)
        or set(names) != expected_state_names
        or len(set(names)) != len(names)
    ):
        raise RuntimeError("resume checkpoint state inventory drift")
    for item in files:
        name = str(item["name"])
        if Path(name).name != name:
            raise RuntimeError("resume checkpoint contains a non-local state path")
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != str(item["sha256"])
        ):
            raise RuntimeError(f"resume checkpoint file/hash drift: {path}")
    adapter = record.get("adapter")
    adapter_record = next(
        item for item in files if item["name"] == _adapter_filename(language)
    )
    if adapter != adapter_record:
        raise RuntimeError("resume checkpoint adapter record drift")

    state = torch.load(
        checkpoint / "optimizer_scheduler.pt", map_location="cpu", weights_only=True
    )
    if (
        int(state.get("schema_version", 0)) != 1
        or int(state.get("sweep", 0)) != sweep
        or int(state.get("global_step", 0)) != global_step
        or state.get("config_sha256") != record.get("config_sha256")
        or state.get("training_critical_hash")
        != record.get("training_critical_hash")
        or (
            "collapse_logs" in record
            and int(state.get("collapse_logs", -1)) != int(record["collapse_logs"])
        )
        or not isinstance(state.get("optimizer"), dict)
        or not isinstance(state.get("scheduler"), dict)
        or len(state["optimizer"].get("state", {})) != 288
        or int(state["scheduler"].get("last_epoch", -1)) != global_step
        or int(state["scheduler"].get("_step_count", -1)) != global_step + 1
    ):
        raise RuntimeError("resume optimizer/scheduler metadata drift")
    for rank in range(int(world_size)):
        rng = torch.load(
            checkpoint / f"rng.rank-{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            int(rng.get("schema_version", 0)) != 1
            or int(rng.get("rank", -1)) != rank
            or int(rng.get("sweep", 0)) != sweep
            or int(rng.get("global_step", 0)) != global_step
            or not isinstance(rng.get("cpu_rng"), torch.Tensor)
            or not isinstance(rng.get("cuda_rng"), torch.Tensor)
        ):
            raise RuntimeError(f"resume RNG metadata drift for rank {rank}")
    return sweep, global_step


def _collapse_logs_at_step(
    metrics_path: Path, *, global_step: int, collapse_floor: float
) -> int:
    collapse_logs = 0
    boundary_seen = False
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            step = int(record["step"])
            if step > global_step:
                break
            collapse_logs = (
                collapse_logs + 1
                if float(record["codevector_perplexity"]) < collapse_floor
                else 0
            )
            boundary_seen = step == global_step
    if not boundary_seen:
        raise RuntimeError("source metrics do not contain the resume sweep boundary")
    return collapse_logs


def _resume_checkpoint(
    value: Path,
    *,
    experiment_root: Path,
    config: dict[str, Any],
    expected_config_hash: str,
    expected_critical_hash: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    checkpoint = value.resolve()
    root = repo_root(experiment_root)
    if not _is_beneath(checkpoint, root):
        raise RuntimeError("resume checkpoint escapes the WAXAL3 repository")
    record_path = checkpoint / "CHECKPOINT.json"
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    record = _read_strict_json(record_path)
    sweep, global_step = _validate_checkpoint_files(
        checkpoint,
        record,
        horizon_sweeps=int(config["scheduler_horizon_sweeps"]),
        updates_per_sweep=int(config["updates_per_sweep"]),
        world_size=int(config["world_size"]),
        language=str(config["language"]),
    )

    current_runs = (experiment_root / "runs").resolve()
    if _is_beneath(checkpoint, current_runs):
        relative = checkpoint.relative_to(current_runs)
        if len(relative.parts) != 3 or relative.parts[1] != "checkpoints":
            raise RuntimeError("same-packet resume checkpoint path schema drift")
        if (
            record.get("config_sha256") != expected_config_hash
            or record.get("training_critical_hash") != expected_critical_hash
        ):
            raise RuntimeError("same-packet resume checkpoint contract mismatch")
        resume_collapse_logs = int(record.get("collapse_logs", -1))
        if resume_collapse_logs < 0:
            raise RuntimeError("same-packet checkpoint lacks resumable health state")
        return checkpoint, record, {
            "kind": "same_packet",
            "source_experiment_id": config["experiment_id"],
            "source_run_id": relative.parts[0],
            "source_sweep": sweep,
            "source_global_step": global_step,
            "resume_trajectory_hash": resume_trajectory_hash(config),
            "resume_collapse_logs": resume_collapse_logs,
        }

    if config.get("resume_policy") != "sweep_boundary_verified_lineage":
        raise RuntimeError("cross-packet resume is not authorized by this profile")
    specification = read_json(experiment_root / "experiment.yaml")
    recovery = specification.get("recovery")
    if not isinstance(recovery, dict) or set(recovery) != RECOVERY_SPEC_KEYS:
        raise RuntimeError("verified-lineage recovery specification drift")
    if int(recovery.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported recovery specification schema")

    source_experiment = (root / str(recovery["source_experiment_path"])).resolve()
    expected_checkpoint = (root / str(recovery["source_checkpoint_path"])).resolve()
    if (
        not _is_beneath(source_experiment, root)
        or checkpoint != expected_checkpoint
        or experiment_root_from(checkpoint) != source_experiment
        or read_json(source_experiment / "experiment.yaml").get("experiment_id")
        != recovery["source_experiment_id"]
        or checkpoint.parent.parent.name != recovery["source_run_id"]
        or sweep != int(recovery["source_sweep"])
    ):
        raise RuntimeError("resume checkpoint does not match the declared parent lineage")

    source_packet = source_experiment / "packet"
    if sha256_file(source_packet / "PACKET.json") != recovery["source_packet_json_sha256"]:
        raise RuntimeError("source PACKET.json hash drift")
    _verify_packet_inventory(source_packet, str(recovery["source_packet_digest"]))
    source_audit_path = source_packet / "audit/AUDIT.json"
    source_audit = _read_strict_json(source_audit_path)
    if (
        sha256_file(source_audit_path) != recovery["source_packet_audit_sha256"]
        or source_audit.get("status") != "PASS"
        or source_audit.get("packet_digest") != recovery["source_packet_digest"]
    ):
        raise RuntimeError("source packet audit drift")

    source_profile = source_packet / "profiles/production.yaml"
    source_run = source_experiment / "runs" / str(recovery["source_run_id"])
    source_launch_path = source_run / "launch.json"
    source_start_path = source_run / "START.json"
    source_metrics_path = source_run / "metrics.jsonl"
    source_config_path = source_run / "config.yaml"
    for path, expected in (
        (source_profile, recovery["source_profile_sha256"]),
        (source_launch_path, recovery["source_launch_sha256"]),
        (source_start_path, recovery["source_start_sha256"]),
        (source_metrics_path, recovery["source_metrics_sha256"]),
        (source_config_path, recovery["source_config_sha256"]),
        (record_path, recovery["source_checkpoint_json_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"declared recovery lineage hash drift: {path}")

    source_launch = _read_strict_json(source_launch_path)
    source_start = _read_strict_json(source_start_path)
    source_config = read_json(source_config_path)
    if (
        sha256_file(source_profile) != sha256_file(source_config_path)
        or source_launch.get("run_id") != recovery["source_run_id"]
        or source_launch.get("packet_digest") != recovery["source_packet_digest"]
        or source_start.get("run_id") != recovery["source_run_id"]
        or source_start.get("config") != source_config
        or source_start.get("config_sha256") != recovery["source_config_sha256"]
        or source_start.get("training_critical_hash")
        != recovery["source_training_critical_hash"]
        or record.get("config_sha256") != recovery["source_config_sha256"]
        or record.get("training_critical_hash")
        != recovery["source_training_critical_hash"]
    ):
        raise RuntimeError("source run/checkpoint lineage metadata drift")

    source_files = sorted(recovery["source_state_files"], key=lambda item: item["name"])
    record_files = sorted(record["files"], key=lambda item: item["name"])
    if source_files != record_files:
        raise RuntimeError("source state-file declaration does not match checkpoint")
    source_trajectory_hash = resume_trajectory_hash(source_config)
    source_collapse_logs = _collapse_logs_at_step(
        source_metrics_path,
        global_step=global_step,
        collapse_floor=float(config["codebook_collapse_floor"]),
    )
    if (
        source_trajectory_hash != recovery["source_resume_trajectory_hash"]
        or resume_trajectory_payload(source_config) != resume_trajectory_payload(config)
        or source_collapse_logs != int(recovery["source_collapse_logs"])
    ):
        raise RuntimeError("source/current optimization trajectory mismatch")

    return checkpoint, record, {
        "kind": "verified_cross_packet",
        "source_experiment_id": recovery["source_experiment_id"],
        "source_run_id": recovery["source_run_id"],
        "source_sweep": sweep,
        "source_global_step": global_step,
        "source_packet_digest": recovery["source_packet_digest"],
        "source_config_sha256": recovery["source_config_sha256"],
        "source_training_critical_hash": recovery["source_training_critical_hash"],
        "resume_trajectory_hash": source_trajectory_hash,
        "resume_collapse_logs": source_collapse_logs,
    }


def _save_sweep_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_dir: Path,
    sweep: int,
    global_step: int,
    rank: int,
    local_rank: int,
    config_hash: str,
    critical_hash: str,
    collapse_logs: int,
    world_size: int,
    language: str,
) -> dict[str, Any] | None:
    directory = run_dir / "checkpoints" / f"sweep-{sweep:02d}"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
        export_adapter_only(
            model,
            directory / _adapter_filename(language),
            metadata={
                "schema_version": "1",
                "stage": f"waxal3_mms1b_{language}_adapter_cpt",
                "language": language,
                "sweep": str(sweep),
                "global_step": str(global_step),
                "config_sha256": config_hash,
                "training_critical_hash": critical_hash,
                "collapse_logs": str(collapse_logs),
            },
        )
        torch.save(
            {
                "schema_version": 1,
                "sweep": int(sweep),
                "global_step": int(global_step),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config_sha256": config_hash,
                "training_critical_hash": critical_hash,
                "collapse_logs": int(collapse_logs),
            },
            directory / "optimizer_scheduler.pt",
        )
    dist.barrier(device_ids=[local_rank])
    torch.save(
        {
            "schema_version": 1,
            "rank": rank,
            "sweep": int(sweep),
            "global_step": int(global_step),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(local_rank),
        },
        directory / f"rng.rank-{rank}.pt",
    )
    dist.barrier(device_ids=[local_rank])
    record = None
    if rank == 0:
        names = [
            _adapter_filename(language),
            "optimizer_scheduler.pt",
            *(f"rng.rank-{index}.pt" for index in range(int(world_size))),
        ]
        files = [
            {
                "name": name,
                "bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in names
        ]
        record = {
            "schema_version": 1,
            "status": "COMPLETE",
            "created_at_utc": utc_now(),
            "sweep": int(sweep),
            "global_step": int(global_step),
            "config_sha256": config_hash,
            "training_critical_hash": critical_hash,
            "collapse_logs": int(collapse_logs),
            "files": files,
            "adapter": next(item for item in files if item["name"].endswith(".safetensors")),
            "resumable": True,
        }
        write_json_create_only(directory / "CHECKPOINT.json", record)
    dist.barrier(device_ids=[local_rank])
    return record


def _load_resume_training_state(
    *,
    checkpoint: Path,
    record: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    local_rank: int,
    language: str,
) -> tuple[int, int]:
    _load_adapter(model, checkpoint / _adapter_filename(language))
    state = torch.load(
        checkpoint / "optimizer_scheduler.pt",
        map_location=torch.device("cuda", local_rank),
        weights_only=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sweep = int(state["sweep"])
    global_step = int(state["global_step"])
    if sweep != int(record["sweep"]) or global_step != int(record["global_step"]):
        raise RuntimeError("loaded resume state disagrees with checkpoint record")
    return sweep, global_step


def _restore_resume_rng(
    *, checkpoint: Path, rank: int, local_rank: int, sweep: int, global_step: int
) -> None:
    rng = torch.load(
        checkpoint / f"rng.rank-{rank}.pt", map_location="cpu", weights_only=True
    )
    if int(rng["sweep"]) != sweep or int(rng["global_step"]) != global_step:
        raise RuntimeError("loaded resume RNG disagrees with checkpoint record")
    torch.set_rng_state(rng["cpu_rng"])
    torch.cuda.set_rng_state(rng["cuda_rng"], local_rank)


def _resume_bounds(
    *, mode: str, completed_sweep: int, global_step: int, config: dict[str, Any]
) -> tuple[int, int, int]:
    updates = int(config["updates_per_sweep"])
    gate = int(config["gate_sweep"])
    horizon = int(config["scheduler_horizon_sweeps"])
    if mode == "smoke":
        return 1, 1, int(config["smoke_updates"])
    if mode == "gate":
        return 1, gate, gate * updates
    if mode == "resume-smoke":
        if not 1 <= completed_sweep < horizon:
            raise RuntimeError("resume smoke requires a completed sweep before the horizon")
        return completed_sweep + 1, completed_sweep + 1, global_step + int(
            config["smoke_updates"]
        )
    if mode == "resume-probe":
        if not 1 <= completed_sweep < horizon:
            raise RuntimeError("resume probe requires a completed sweep before the horizon")
        return completed_sweep + 1, completed_sweep + 1, global_step + int(
            config["smoke_updates"]
        )
    if mode == "resume":
        if not 1 <= completed_sweep < horizon:
            raise RuntimeError("resume requires a completed sweep before the horizon")
        stop_sweep = gate if completed_sweep < gate else horizon
        return completed_sweep + 1, stop_sweep, stop_sweep * updates
    if mode == "tail":
        if completed_sweep != gate:
            raise RuntimeError("tail must resume exactly after the configured gate")
        return completed_sweep + 1, horizon, horizon * updates
    raise RuntimeError(f"unsupported CPT mode: {mode}")


def _run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yaml"
    experiment_root = experiment_root_from(run_dir)
    config = read_json(config_path)
    smoke_modes = {"smoke", "resume-smoke"}
    short_modes = {*smoke_modes, "resume-probe"}
    resume_modes = {"resume-smoke", "resume-probe", "resume", "tail"}
    expected_phase = "smoke" if args.mode in smoke_modes else "production"
    if config.get("phase") != expected_phase:
        raise RuntimeError(f"mode/profile phase mismatch: {args.mode}/{config.get('phase')}")
    if (args.mode in resume_modes) != (args.resume_checkpoint is not None):
        raise RuntimeError(
            "resume-smoke, resume-probe, resume, and tail require a checkpoint exclusively"
        )
    paths = validate_global_config(
        config,
        experiment_root=experiment_root,
        run_dir=run_dir,
        require_authorization=True,
        verify_large_hashes=False,
    )
    critical_hash = training_critical_hash(
        experiment_root=experiment_root,
        config_path=config_path,
        paths=paths,
    )
    require_fresh_pass(run_dir, critical_hash)
    collisions = [
        run_dir / name
        for name in (
            "START.json",
            "FINAL.json",
            "FAILURE.json",
            "metrics.jsonl",
            "checkpoints",
            "RESULT.md",
        )
        if (run_dir / name).exists()
    ]
    if collisions:
        raise RuntimeError(f"create-only CPT output collision: {collisions}")

    config_hash = sha256_file(config_path)
    resume_checkpoint = None
    resume_record = None
    resume_lineage = None
    if args.resume_checkpoint is not None:
        resume_checkpoint, resume_record, resume_lineage = _resume_checkpoint(
            args.resume_checkpoint,
            experiment_root=experiment_root,
            config=config,
            expected_config_hash=config_hash,
            expected_critical_hash=critical_hash,
        )

    rank, local_rank = require_distributed_runtime(config)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    device = torch.device("cuda", local_rank)
    seed = int(config["seed"])
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows = pq.read_table(paths["audio_manifest"]).to_pylist()
    if config["phase"] == "smoke":
        # Start with the longest clips to make the disposable smoke a memory
        # and numerical stress test, while retaining broad-before-tail staging.
        stress_rows = []
        for stage in ("broad", "tail"):
            selected = sorted(
                (row for row in rows if row["stage"] == stage),
                key=lambda row: (-int(row["decoded_frames"]), str(row["id"])),
            )
            stress_rows.extend(
                {**row, "stage_order_index": index}
                for index, row in enumerate(selected)
            )
        rows = stress_rows
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        paths["asr_base"], local_files_only=True
    )
    model, composition = compose_native_adapter_pretraining_model(
        asr_base=paths["asr_base"],
        ssl_base=paths["ssl_base"],
        native_adapter=paths["native_adapter"],
        mask_time_prob=float(config["mask_time_prob"]),
        mask_time_length=int(config["mask_time_length"]),
        mask_time_min_masks=int(config["mask_time_min_masks"]),
        num_negatives=int(config["num_negatives"]),
        layerdrop=float(config["layerdrop"]),
    )
    model.set_gumbel_temperature(float(config["gumbel_temperature"]))
    reference = adapter_reference(model)
    l2_cache: dict[str, torch.Tensor] = {}
    frozen_reference = _frozen_sentinels(model)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )
    horizon_steps = int(config["scheduler_horizon_sweeps"]) * int(
        config["updates_per_sweep"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_multiplier(
            step,
            warmup_steps=int(config["warmup_steps"]),
            max_steps=horizon_steps,
        ),
    )
    completed_sweep = 0
    global_step = 0
    if resume_checkpoint is not None:
        assert resume_record is not None
        completed_sweep, global_step = _load_resume_training_state(
            checkpoint=resume_checkpoint,
            record=resume_record,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            local_rank=local_rank,
            language=str(config["language"]),
        )
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    dataset = CPTAudioDataset(
        rows,
        audio_root=paths["audio_root"],
        sample_rate=int(config["sample_rate"]),
        max_audio_seconds=float(config["max_audio_seconds"]),
        seed=seed,
        crop_seed_mode=(
            "s008_source_row"
            if config["sampler_mode"] == "s008_speaker_interleaved"
            else "presentation_slot"
        ),
    )
    sampler_class = (
        SpeakerInterleavedDistributedSampler
        if config["sampler_mode"] == "s008_speaker_interleaved"
        else StagePaddedDistributedSampler
    )
    sampler = sampler_class(
        rows,
        rank=rank,
        world_size=int(config["world_size"]),
        per_device_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        seed=seed,
    )
    if (
        sampler.optimizer_updates_per_sweep != int(config["updates_per_sweep"])
        or sampler.synchronization_padding_slots != int(config["expected_sync_padding"])
        or int(getattr(sampler, "dropped_rows", 0))
        != int(config["expected_dropped_rows"])
    ):
        raise RuntimeError("CPT sampler sweep arithmetic drift")
    collator = CPTCollator(
        feature_extractor=feature_extractor,
        sample_rate=int(config["sample_rate"]),
        max_samples=dataset.max_samples,
        conv_kernel=[int(value) for value in model.config.conv_kernel],
        conv_stride=[int(value) for value in model.config.conv_stride],
        mask_time_prob=float(config["mask_time_prob"]),
        mask_time_length=int(config["mask_time_length"]),
        mask_time_min_masks=int(config["mask_time_min_masks"]),
        num_negatives=int(config["num_negatives"]),
    )

    start_sweep, stop_sweep, stop_step = _resume_bounds(
        mode=args.mode,
        completed_sweep=completed_sweep,
        global_step=global_step,
        config=config,
    )
    collapse_logs = (
        int(resume_lineage["resume_collapse_logs"])
        if resume_lineage is not None
        else 0
    )
    if collapse_logs < 0:
        raise RuntimeError("resume health state is invalid")
    dataset.set_sweep(start_sweep)
    sampler.set_sweep(start_sweep)

    if rank == 0:
        (run_dir / "checkpoints").mkdir(exist_ok=False)
        metrics_handle = (run_dir / "metrics.jsonl").open("x", encoding="utf-8")
        write_json_create_only(
            run_dir / "START.json",
            {
                "schema_version": 1,
                "started_at_utc": utc_now(),
                "mode": args.mode,
                "command": [sys.executable, *sys.argv],
                "run_id": run_dir.name,
                "config": config,
                "config_sha256": config_hash,
                "training_critical_hash": critical_hash,
                "git": git_state(repo_root(experiment_root)),
                "hardware": hardware_state(),
                "composition": composition,
                "sampler": {
                    "source_rows": len(rows),
                    "unique_rows_per_sweep": sampler.unique_rows_per_sweep,
                    "dropped_rows_per_sweep": int(
                        getattr(sampler, "dropped_rows", 0)
                    ),
                    "synchronization_padding_slots_per_sweep": sampler.synchronization_padding_slots,
                    "updates_per_sweep": sampler.optimizer_updates_per_sweep,
                    "global_batch": (
                        int(config["world_size"])
                        * int(config["per_device_batch_size"])
                        * int(config["gradient_accumulation_steps"])
                    ),
                    "runtime_order": (
                        "max_length_descending_stress"
                        if config["phase"] == "smoke"
                        else (
                            "s008_deterministic_speaker_interleaved"
                            if config["sampler_mode"]
                            == "s008_speaker_interleaved"
                            else "frozen_manifest_order"
                        )
                    ),
                    "padding_records_start_sweep": sampler.padding_records(),
                    "dropped_records_start_sweep": (
                        sampler.dropped_records()
                        if hasattr(sampler, "dropped_records")
                        else []
                    ),
                },
                "scheduler_horizon_steps": horizon_steps,
                "start_sweep": start_sweep,
                "stop_sweep": stop_sweep,
                "start_global_step": global_step,
                "stop_global_step": stop_step,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                "resume_checkpoint_record": resume_record,
                "resume_lineage": resume_lineage,
                "transcripts_accessed": False,
                "test_labels_accessed": False,
                "optimized_objective": "contrastive_loss_per_mask_plus_adapter_l2sp",
                "quantizer_target_mode": "frozen_hard_eval",
                "diversity_loss_role": "telemetry_only_no_adapter_gradient",
            },
        )
    else:
        metrics_handle = None
    dist.barrier(device_ids=[local_rank])

    log_every = int(config["log_every_steps"])
    window = torch.zeros(11, dtype=torch.float64, device=device)
    window_updates = 0
    checkpoints: list[dict[str, Any]] = []
    started = time.monotonic()
    last_log = started
    model.train()
    if bool(config["quantizer_eval"]):
        model.quantizer.eval()
    optimizer.zero_grad(set_to_none=True)
    if resume_checkpoint is not None:
        _restore_resume_rng(
            checkpoint=resume_checkpoint,
            rank=rank,
            local_rank=local_rank,
            sweep=completed_sweep,
            global_step=global_step,
        )

    for sweep in range(start_sweep, stop_sweep + 1):
        dataset.set_sweep(sweep)
        sampler.set_sweep(sweep)
        loader = DataLoader(
            dataset,
            batch_size=int(config["per_device_batch_size"]),
            sampler=sampler,
            collate_fn=collator,
            num_workers=int(config["dataloader_num_workers"]),
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )
        if len(loader) != int(config["updates_per_sweep"]):
            raise RuntimeError("rank-local loader update count drift")
        updates_this_sweep = 0
        for batch in loader:
            local_masks = batch["masked_positions"].to(device=device, dtype=torch.float64)
            global_masks = local_masks.clone()
            dist.all_reduce(global_masks, op=dist.ReduceOp.SUM)
            valid_features = model._get_feat_extract_output_lengths(
                batch["valid_samples"].to(device)
            ).sum().to(torch.float64)
            with nullcontext():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = wrapped(
                        input_values=batch["input_values"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                        mask_time_indices=batch["mask_time_indices"].to(device, non_blocking=True),
                        sampled_negative_indices=batch["sampled_negative_indices"].to(
                            device, non_blocking=True
                        ),
                    )
                    contrastive_loss = ddp_mask_normalized_loss(
                        outputs.contrastive_loss,
                        global_masks,
                        world_size=int(config["world_size"]),
                    )
                l2_value = adapter_l2sp_penalty(model, reference, l2_cache)
                loss = contrastive_loss + float(config["adapter_l2sp"]) * l2_value
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite CPT objective: {loss}")
                loss.backward()

            sync_padding = batch["synchronization_padding"].sum().to(
                device=device, dtype=torch.float64
            )
            window += torch.stack(
                [
                    outputs.loss.detach().double(),
                    outputs.contrastive_loss.detach().double(),
                    outputs.diversity_loss.detach().double(),
                    local_masks,
                    valid_features,
                    outputs.codevector_perplexity.detach().double() * local_masks,
                    batch["valid_samples"].sum().to(device=device, dtype=torch.float64),
                    l2_value.detach().double(),
                    loss.detach().double(),
                    sync_padding,
                    torch.tensor(len(batch["ids"]), device=device, dtype=torch.float64),
                ]
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config["max_grad_norm"])
            )
            if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0.0:
                raise FloatingPointError(f"invalid adapter gradient norm: {gradient_norm}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            updates_this_sweep += 1
            window_updates += 1

            log_now = (
                global_step % log_every == 0
                or global_step == stop_step
                or updates_this_sweep == int(config["updates_per_sweep"])
            )
            if log_now:
                reduced = window.clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                now = time.monotonic()
                interval = now - last_log
                total = now - started
                masks = float(reduced[3])
                valid = float(reduced[4])
                record = {
                    "step": global_step,
                    "sweep": sweep,
                    "updates_in_sweep": updates_this_sweep,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "loss_per_mask": float(reduced[1] / masks),
                    "hf_total_loss_per_mask": float(reduced[0] / masks),
                    "contrastive_loss_per_mask": float(reduced[1] / masks),
                    "diversity_loss_per_mask": float(reduced[2] / masks),
                    "codevector_perplexity": float(reduced[5] / masks),
                    "masked_positions": int(masks),
                    "valid_feature_positions": int(valid),
                    "realized_mask_coverage": float(masks / valid),
                    "adapter_l2sp": float(
                        reduced[7]
                        / (window_updates * int(config["world_size"]))
                    ),
                    "gradient_norm": float(gradient_norm),
                    "audio_seconds": float(reduced[6] / int(config["sample_rate"])),
                    "synchronization_padding_presentations": int(reduced[9]),
                    "presentations": int(reduced[10]),
                    "updates_per_second": float(window_updates / interval),
                    "audio_seconds_per_second": float(
                        reduced[6] / int(config["sample_rate"]) / interval
                    ),
                    "elapsed_seconds": total,
                    "eta_to_process_stop_seconds": float(
                        total / max(1, global_step - (completed_sweep * int(config["updates_per_sweep"])))
                        * (stop_step - global_step)
                    ),
                    "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "recorded_at_utc": utc_now(),
                }
                finite = torch.tensor(
                    [
                        record["loss_per_mask"],
                        record["codevector_perplexity"],
                        record["realized_mask_coverage"],
                        record["gradient_norm"],
                    ],
                    device=device,
                )
                if not torch.isfinite(finite).all():
                    raise FloatingPointError(f"non-finite CPT telemetry: {record}")
                if global_step >= int(config["collapse_check_after_steps"]):
                    coverage = record["realized_mask_coverage"]
                    if not float(config["effective_mask_hard_min"]) <= coverage <= float(
                        config["effective_mask_hard_max"]
                    ):
                        raise RuntimeError(f"effective-mask kill criterion: {record}")
                    collapse_logs = (
                        collapse_logs + 1
                        if record["codevector_perplexity"]
                        < float(config["codebook_collapse_floor"])
                        else 0
                    )
                    if collapse_logs >= 3:
                        raise RuntimeError("codebook collapse kill criterion reached")
                if rank == 0:
                    assert metrics_handle is not None
                    _append_json_line(metrics_handle, record)
                    _best_effort_stdout(record)
                window.zero_()
                window_updates = 0
                last_log = now
                torch.cuda.reset_peak_memory_stats(device)

            if global_step >= stop_step:
                break

        if args.mode not in short_modes and updates_this_sweep != int(
            config["updates_per_sweep"]
        ):
            raise RuntimeError("production stopped before a complete CPT sweep")
        if args.mode not in short_modes:
            _assert_frozen_sentinels(model, frozen_reference)
            checkpoint = _save_sweep_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_dir=run_dir,
                sweep=sweep,
                global_step=global_step,
                rank=rank,
                local_rank=local_rank,
                config_hash=config_hash,
                critical_hash=critical_hash,
                collapse_logs=collapse_logs,
                world_size=int(config["world_size"]),
                language=str(config["language"]),
            )
            if rank == 0:
                assert checkpoint is not None
                checkpoints.append(checkpoint)
        if global_step >= stop_step:
            break

    _assert_frozen_sentinels(model, frozen_reference)
    if rank == 0:
        assert metrics_handle is not None
        if args.mode in short_modes:
            export_adapter_only(
                model,
                run_dir
                / f"checkpoints/smoke-final-adapter.{config['language']}.safetensors",
                metadata={
                    "mode": args.mode,
                    "step": str(global_step),
                    "language": str(config["language"]),
                },
            )
        metrics_handle.close()
        if args.mode == "resume":
            status = (
                f"RESUME_GATE_SWEEP_{int(config['gate_sweep'])}_REACHED"
                if stop_sweep == int(config["gate_sweep"])
                else f"RESUME_HORIZON_SWEEP_{int(config['scheduler_horizon_sweeps'])}_REACHED"
            )
        else:
            status = {
                "smoke": "SMOKE_PASS",
                "resume-smoke": "RESUME_SMOKE_PASS",
                "resume-probe": "RESUME_PROBE_PASS",
                "gate": f"GATE_SWEEP_{int(config['gate_sweep'])}_REACHED",
                "tail": f"HORIZON_SWEEP_{int(config['scheduler_horizon_sweeps'])}_REACHED",
            }[args.mode]
        final = {
            "schema_version": 1,
            "status": status,
            "completed_at_utc": utc_now(),
            "run_id": run_dir.name,
            "mode": args.mode,
            "global_step": global_step,
            "last_completed_sweep": (
                completed_sweep if args.mode in short_modes else stop_sweep
            ),
            "elapsed_seconds": time.monotonic() - started,
            "config_sha256": config_hash,
            "training_critical_hash": critical_hash,
            "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
            "checkpoints": checkpoints,
            "resume_source": str(resume_checkpoint) if resume_checkpoint else None,
            "resume_lineage": resume_lineage,
            "frozen_sentinel_status": "BIT_IDENTICAL",
            "transcripts_accessed": False,
            "test_labels_accessed": False,
            "optimized_objective": "contrastive_loss_per_mask_plus_adapter_l2sp",
            "quantizer_target_mode": "frozen_hard_eval",
            "diversity_loss_role": "telemetry_only_no_adapter_gradient",
        }
        write_json_create_only(run_dir / "FINAL.json", final)
        with (run_dir / "RESULT.md").open("x", encoding="utf-8") as handle:
            handle.write(
                f"# {run_dir.name}\n\n"
                f"- Status: `{status}`\n"
                f"- Mode: `{args.mode}`\n"
                f"- Global step: `{global_step}`\n"
                f"- Last completed sweep: `{final['last_completed_sweep']}`\n"
                f"- Frozen sentinels: `BIT_IDENTICAL`\n"
                f"- Transcripts/test labels accessed: `false` / `false`\n"
            )
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "resume-smoke", "resume-probe", "gate", "resume", "tail"),
        required=True,
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        return _run(args)
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            failure = args.run_dir.resolve() / "FAILURE.json"
            if not failure.exists():
                write_json_create_only(
                    failure,
                    {
                        "schema_version": 1,
                        "failed_at_utc": utc_now(),
                        "error": traceback.format_exc(),
                    },
                )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
