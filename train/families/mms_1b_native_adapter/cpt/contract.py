#!/usr/bin/env python3
"""Fail-closed multilingual WAXAL3 MMS-1B native-adapter CPT contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

import pyarrow.parquet as pq
from safetensors import safe_open
import torch
import yaml


TRANSCRIPT_MARKERS = {
    "sentence",
    "transcript",
    "transcription",
    "text",
    "target",
    "targets",
    "label",
    "labels",
    "reference",
    "hypothesis",
}
EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "pyarrow": "17.0.0",
    "safetensors": "0.7.0",
    "soundfile": "0.12.1",
    "torch": "2.5.1+cu124",
    "transformers": "4.46.3",
}
ROWS = 139_239
HOURS = 372.7011875347242
HORIZON_SWEEPS = 15
GATE_SWEEP = 10
ADAPTER_TENSORS = 288
ADAPTER_PARAMETERS = 2_151_168
GEOMETRY_CONTRACTS = {
    4: {
        "per_device_batch_size": 2,
        "global_batch": 8,
        "broad_padding": 5,
        "tail_padding": 4,
        "sync_padding": 9,
        "updates_per_sweep": 17_406,
        "smoke_updates": 2,
        "smoke_warmup_steps": 8_703,
        "production_warmup_steps": 8_703,
        "learning_rate": 1e-4,
    },
    8: {
        "per_device_batch_size": 4,
        "global_batch": 32,
        "broad_padding": 21,
        "tail_padding": 4,
        "sync_padding": 25,
        "updates_per_sweep": 4_352,
        "smoke_updates": 200,
        "smoke_warmup_steps": 50,
        "production_warmup_steps": 2_176,
        "learning_rate": 2e-4,
    },
}
S008_SHONA_CONTRACT = {
    "language": "sna",
    "sampler_mode": "s008_speaker_interleaved",
    "world_size": 4,
    "per_device_batch_size": 2,
    "global_batch": 8,
    "expected_rows": 85_372,
    "expected_broad_rows": 85_372,
    "expected_broad_padding": 0,
    "expected_tail_rows": 0,
    "expected_tail_padding": 0,
    "expected_sync_padding": 0,
    "expected_dropped_rows": 4,
    "updates_per_sweep": 10_671,
    "scheduler_horizon_sweeps": 10,
    "gate_sweep": 10,
    "smoke_updates": 2,
    "smoke_warmup_steps": 5_335,
    "production_warmup_steps": 5_335,
    "learning_rate": 1e-4,
}
RESUME_POLICIES = {
    "sweep_boundary_same_packet",
    "sweep_boundary_verified_lineage",
}
RESUME_TRAJECTORY_EXCLUDED_KEYS = {
    "experiment_id",
    "minimum_gpu_memory_bytes",
    "phase",
    "resume_policy",
}


REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "phase",
    "training_authorized",
    "language",
    "sampler_mode",
    "seed",
    "asr_base_path",
    "asr_model_sha256",
    "asr_model_bytes",
    "asr_config_sha256",
    "asr_preprocessor_sha256",
    "ssl_base_path",
    "ssl_model_sha256",
    "ssl_model_bytes",
    "ssl_config_sha256",
    "ssl_preprocessor_sha256",
    "native_adapter_path",
    "native_adapter_sha256",
    "native_adapter_bytes",
    "audio_root",
    "audio_build_path",
    "audio_build_sha256",
    "audio_manifest_path",
    "audio_manifest_sha256",
    "expected_rows",
    "expected_hours",
    "expected_identity_digest",
    "expected_broad_rows",
    "expected_broad_padding",
    "expected_tail_rows",
    "expected_tail_padding",
    "expected_sync_padding",
    "expected_dropped_rows",
    "updates_per_sweep",
    "scheduler_horizon_sweeps",
    "gate_sweep",
    "sample_rate",
    "max_audio_seconds",
    "mask_time_prob",
    "mask_time_length",
    "mask_time_min_masks",
    "num_negatives",
    "layerdrop",
    "quantizer_eval",
    "gumbel_temperature",
    "precision",
    "tf32",
    "gradient_checkpointing",
    "minimum_gpu_memory_bytes",
    "world_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "dataloader_num_workers",
    "smoke_updates",
    "warmup_steps",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "weight_decay",
    "max_grad_norm",
    "adapter_l2sp",
    "log_every_steps",
    "save_every_sweep",
    "collapse_check_after_steps",
    "codebook_collapse_floor",
    "effective_mask_expected_min",
    "effective_mask_expected_max",
    "effective_mask_hard_min",
    "effective_mask_hard_max",
    "resume_policy",
    "expected_adapter_tensors",
    "expected_adapter_parameters",
    "model_audit_path",
    "model_audit_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_trajectory_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return only values capable of changing the optimization trajectory.

    Experiment identity, smoke/production phase, and the provenance policy do
    not affect model, optimizer, scheduler, sampler, or RNG state. Logging
    cadence is intentionally retained because health-stop counters are
    evaluated on log events. Every other locked CPT field must also match
    across a recovery lineage.
    """

    return {
        key: config[key]
        for key in sorted(config)
        if key not in RESUME_TRAJECTORY_EXCLUDED_KEYS
    }


def resume_trajectory_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            resume_trajectory_payload(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"mapping required: {path}")
    return value


def write_json_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def experiment_root_from(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while candidate != candidate.parent:
        if (candidate / "experiment.yaml").is_file():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(f"cannot locate experiment root above {path}")


def repo_root(experiment_root: Path) -> Path:
    candidate = experiment_root.resolve()
    while candidate != candidate.parent:
        if (candidate / "README.md").is_file() and (candidate / "models").is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError("cannot locate WAXAL3 repository root")


def resolve_path(value: str, *, experiment_root: Path) -> Path:
    root = repo_root(experiment_root)
    if value.startswith("repo://"):
        path = root / value.removeprefix("repo://")
    elif value.startswith("exp://"):
        path = experiment_root / value.removeprefix("exp://")
    elif value.startswith("packet://"):
        relative = Path(value.removeprefix("packet://"))
        packet_path = experiment_root / "packet" / relative
        if packet_path.exists() or (experiment_root / "packet").exists():
            path = packet_path
        else:
            try:
                suffix = relative.relative_to("src/model_family")
            except ValueError as exc:
                raise RuntimeError(f"unsupported prepacket URI: {value}") from exc
            specification = read_json(experiment_root / "experiment.yaml")
            path = root / str(specification["family_root"]) / "code" / suffix
    else:
        raise ValueError(f"path must use repo://, exp://, or packet://: {value}")
    resolved = path.resolve()
    if not (
        resolved == root
        or str(resolved).startswith(str(root) + os.sep)
        or resolved == experiment_root
        or str(resolved).startswith(str(experiment_root) + os.sep)
    ):
        raise RuntimeError(f"resolved path escapes WAXAL3: {resolved}")
    return resolved


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise RuntimeError(f"{label} hash drift: {observed} != {expected}")


def git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str, required: bool = True) -> str | None:
        process = subprocess.run(
            ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
        )
        if process.returncode:
            if required:
                raise RuntimeError(process.stderr.strip() or "git command failed")
            return None
        return process.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD", required=False),
        "branch": run("branch", "--show-current") or None,
        "status_short": (run("status", "--short") or "").splitlines(),
    }


def hardware_state() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES
        },
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def require_environment() -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES
    }
    if observed != EXPECTED_PACKAGES:
        raise RuntimeError(f"training environment drift: {observed}")
    if platform.python_version() != "3.11.13" or torch.version.cuda != "12.4":
        raise RuntimeError("pinned Python/CUDA contract drift")
    return observed


def require_distributed_runtime(config: dict[str, Any]) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != int(config["world_size"]) or world_size not in GEOMETRY_CONTRACTS:
        raise RuntimeError("CPT distributed topology does not match a locked recipe")
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise RuntimeError("invalid distributed rank topology")
    if torch.cuda.device_count() != world_size or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{world_size} BF16-capable visible CUDA devices are required")
    required_memory = int(config["minimum_gpu_memory_bytes"])
    if required_memory < 1_000_000_000:
        raise RuntimeError("minimum GPU memory contract is invalid")
    minimum_memory = min(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(world_size)
    )
    if minimum_memory < required_memory:
        raise RuntimeError(
            "visible GPU memory is below the packet requirement: "
            f"{minimum_memory} < {required_memory}"
        )
    return rank, local_rank


def validate_global_config(
    config: dict[str, Any],
    *,
    experiment_root: Path,
    run_dir: Path | None,
    require_authorization: bool,
    verify_large_hashes: bool,
) -> dict[str, Path]:
    require_environment()
    if set(config) != REQUIRED_KEYS:
        raise ValueError(
            f"CPT config schema drift: missing={sorted(REQUIRED_KEYS-set(config))} "
            f"extra={sorted(set(config)-REQUIRED_KEYS)}"
        )
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported CPT config schema")
    specification = read_json(experiment_root / "experiment.yaml")
    if str(config["experiment_id"]) != str(specification.get("experiment_id")):
        raise RuntimeError("profile/experiment ID mismatch")
    if run_dir is not None and run_dir.resolve().parent != (
        experiment_root / "runs"
    ).resolve():
        raise RuntimeError("CPT run directory is outside the experiment runs lane")
    if require_authorization and config["training_authorized"] is not True:
        raise RuntimeError("CPT training is not authorized")
    if not isinstance(config["training_authorized"], bool):
        raise ValueError("training_authorized must be boolean")

    world_size = int(config["world_size"])
    language = str(config["language"])
    sampler_mode = str(config["sampler_mode"])
    if language == "lin" and sampler_mode == "stage_padded":
        geometry = GEOMETRY_CONTRACTS.get(world_size)
        if geometry is None:
            raise ValueError(f"unsupported Lingala CPT world size: {world_size}")
        recipe = {
            "language": "lin",
            "sampler_mode": "stage_padded",
            "expected_rows": ROWS,
            "expected_broad_rows": 138_347,
            "expected_broad_padding": int(geometry["broad_padding"]),
            "expected_tail_rows": 892,
            "expected_tail_padding": int(geometry["tail_padding"]),
            "expected_sync_padding": int(geometry["sync_padding"]),
            "expected_dropped_rows": 0,
            "updates_per_sweep": int(geometry["updates_per_sweep"]),
            "scheduler_horizon_sweeps": HORIZON_SWEEPS,
            "gate_sweep": GATE_SWEEP,
            "per_device_batch_size": int(geometry["per_device_batch_size"]),
            "smoke_updates": int(geometry["smoke_updates"]),
            "smoke_warmup_steps": int(geometry["smoke_warmup_steps"]),
            "production_warmup_steps": int(geometry["production_warmup_steps"]),
            "learning_rate": float(geometry["learning_rate"]),
            "global_batch": int(geometry["global_batch"]),
        }
    elif language == "sna" and sampler_mode == "s008_speaker_interleaved":
        recipe = dict(S008_SHONA_CONTRACT)
        if world_size != int(recipe["world_size"]):
            raise ValueError("S008 Shona CPT requires exactly four GPUs")
    else:
        raise ValueError(
            f"unsupported CPT language/sampler recipe: {language}/{sampler_mode}"
        )
    phase = str(config["phase"])
    warmup_steps = (
        int(recipe["smoke_warmup_steps"])
        if phase == "smoke"
        else int(recipe["production_warmup_steps"])
    )
    exact = {
        "language": language,
        "sampler_mode": sampler_mode,
        "expected_rows": int(recipe["expected_rows"]),
        "expected_broad_rows": int(recipe["expected_broad_rows"]),
        "expected_broad_padding": int(recipe["expected_broad_padding"]),
        "expected_tail_rows": int(recipe["expected_tail_rows"]),
        "expected_tail_padding": int(recipe["expected_tail_padding"]),
        "expected_sync_padding": int(recipe["expected_sync_padding"]),
        "expected_dropped_rows": int(recipe["expected_dropped_rows"]),
        "updates_per_sweep": int(recipe["updates_per_sweep"]),
        "scheduler_horizon_sweeps": int(recipe["scheduler_horizon_sweeps"]),
        "gate_sweep": int(recipe["gate_sweep"]),
        "sample_rate": 16_000,
        "max_audio_seconds": 15.0,
        "mask_time_prob": 0.65,
        "mask_time_length": 10,
        "mask_time_min_masks": 2,
        "num_negatives": 100,
        "layerdrop": 0.0,
        "quantizer_eval": True,
        "gumbel_temperature": 0.5,
        "precision": "bf16",
        "tf32": False,
        "gradient_checkpointing": False,
        "world_size": world_size,
        "per_device_batch_size": int(recipe["per_device_batch_size"]),
        "gradient_accumulation_steps": 1,
        "smoke_updates": int(recipe["smoke_updates"]),
        "warmup_steps": warmup_steps,
        "learning_rate": float(recipe["learning_rate"]),
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "adam_epsilon": 1e-6,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "adapter_l2sp": 1e-5,
        "save_every_sweep": True,
        "codebook_collapse_floor": 5.0,
        "effective_mask_expected_min": 0.47,
        "effective_mask_expected_max": 0.52,
        "effective_mask_hard_min": 0.45,
        "effective_mask_hard_max": 0.55,
        "expected_adapter_tensors": ADAPTER_TENSORS,
        "expected_adapter_parameters": ADAPTER_PARAMETERS,
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ValueError(f"locked CPT recipe drift: {key}={config[key]!r}")
    if str(config["resume_policy"]) not in RESUME_POLICIES:
        raise ValueError(f"unsupported CPT resume policy: {config['resume_policy']!r}")
    if language == "lin":
        if not math.isclose(float(config["expected_hours"]), HOURS, abs_tol=1e-9):
            raise ValueError("Lingala CPT hour contract drift")
    elif not 475.49 < float(config["expected_hours"]) < 475.51:
        raise ValueError("Shona source-row CPT hour contract drift")
    if phase not in {"smoke", "production"}:
        raise ValueError("CPT phase must be smoke or production")
    if int(config["dataloader_num_workers"]) < 0:
        raise ValueError("invalid CPT dataloader workers")
    if int(config["log_every_steps"]) < 1:
        raise ValueError("invalid CPT logging interval")
    if int(config["collapse_check_after_steps"]) < 1:
        raise ValueError("invalid collapse ignition gate")
    if int(recipe["global_batch"]) != (
        int(config["world_size"])
        * int(config["per_device_batch_size"])
        * int(config["gradient_accumulation_steps"])
    ):
        raise ValueError("global CPT batch drift")

    paths = {
        "asr_base": resolve_path(str(config["asr_base_path"]), experiment_root=experiment_root),
        "ssl_base": resolve_path(str(config["ssl_base_path"]), experiment_root=experiment_root),
        "native_adapter": resolve_path(
            str(config["native_adapter_path"]), experiment_root=experiment_root
        ),
        "audio_root": resolve_path(str(config["audio_root"]), experiment_root=experiment_root),
        "audio_build": resolve_path(
            str(config["audio_build_path"]), experiment_root=experiment_root
        ),
        "audio_manifest": resolve_path(
            str(config["audio_manifest_path"]), experiment_root=experiment_root
        ),
        "model_audit": resolve_path(
            str(config["model_audit_path"]), experiment_root=experiment_root
        ),
    }
    small_hashes = (
        (paths["asr_base"] / "config.json", config["asr_config_sha256"], "ASR config"),
        (
            paths["asr_base"] / "preprocessor_config.json",
            config["asr_preprocessor_sha256"],
            "ASR preprocessor",
        ),
        (paths["ssl_base"] / "config.json", config["ssl_config_sha256"], "SSL config"),
        (
            paths["ssl_base"] / "preprocessor_config.json",
            config["ssl_preprocessor_sha256"],
            "SSL preprocessor",
        ),
        (
            paths["native_adapter"],
            config["native_adapter_sha256"],
            f"native {language} adapter",
        ),
        (paths["audio_build"], config["audio_build_sha256"], "CPT data build"),
        (paths["audio_manifest"], config["audio_manifest_sha256"], "CPT manifest"),
        (paths["model_audit"], config["model_audit_sha256"], "model composition audit"),
    )
    for path, expected, label in small_hashes:
        require_hash(path, str(expected), label)
    large = (
        (
            paths["asr_base"] / "model.safetensors",
            config["asr_model_sha256"],
            config["asr_model_bytes"],
            "ASR model",
        ),
        (
            paths["ssl_base"] / "pytorch_model.bin",
            config["ssl_model_sha256"],
            config["ssl_model_bytes"],
            "SSL model",
        ),
    )
    for path, expected_hash, expected_bytes, label in large:
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise RuntimeError(f"{label} file/size drift")
        if verify_large_hashes:
            require_hash(path, str(expected_hash), label)
    if paths["native_adapter"].stat().st_size != int(config["native_adapter_bytes"]):
        raise RuntimeError("native adapter byte-size drift")
    with safe_open(paths["native_adapter"], framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        adapter_keys = [name for name in keys if "adapter_layer" in name]
        parameters = sum(
            math.prod(handle.get_slice(name).get_shape()) for name in adapter_keys
        )
    if (
        len(adapter_keys) != ADAPTER_TENSORS
        or parameters != ADAPTER_PARAMETERS
        or set(keys) - set(adapter_keys) != {"lm_head.bias", "lm_head.weight"}
    ):
        raise RuntimeError(f"native {language} adapter tensor contract drift")

    table = pq.read_table(paths["audio_manifest"])
    forbidden = sorted(
        {str(name).casefold() for name in table.column_names} & TRANSCRIPT_MARKERS
    )
    if forbidden:
        raise RuntimeError(f"transcript-bearing CPT fields detected: {forbidden}")
    if len(table) != int(config["expected_rows"]) or set(
        table.column("language").to_pylist()
    ) != {language}:
        raise RuntimeError(f"{language} CPT manifest row/language drift")
    hours = sum(float(value) for value in table.column("duration_s").to_pylist()) / 3600.0
    if not math.isclose(hours, float(config["expected_hours"]), abs_tol=1e-9):
        raise RuntimeError(f"{language} CPT presentation hours drift")
    stages = table.column("stage").to_pylist()
    if stages.count("broad") != int(config["expected_broad_rows"]) or stages.count(
        "tail"
    ) != int(config["expected_tail_rows"]):
        raise RuntimeError("CPT broad/tail stage count drift")
    build = read_json(paths["audio_build"])
    if str(build["identity_digest"]) != str(config["expected_identity_digest"]):
        raise RuntimeError("CPT identity digest drift")
    if build.get("transcripts_accessed") or build.get("test_labels_accessed"):
        raise RuntimeError("CPT provenance reports forbidden label access")
    stage_contracts = build.get("stage_contracts", {})
    if (
        int(build.get("updates_per_sweep", -1))
        != int(config["updates_per_sweep"])
        or int(build.get("synchronization_padding_slots_per_sweep", -1))
        != int(config["expected_sync_padding"])
        or int(build.get("dropped_rows_per_sweep", 0))
        != int(config["expected_dropped_rows"])
        or int(stage_contracts.get("broad", {}).get("unique_rows", -1))
        != int(config["expected_broad_rows"])
        or int(stage_contracts.get("broad", {}).get("synchronization_padding_slots", -1))
        != int(config["expected_broad_padding"])
        or int(stage_contracts.get("tail", {}).get("unique_rows", -1))
        != int(config["expected_tail_rows"])
        or int(stage_contracts.get("tail", {}).get("synchronization_padding_slots", -1))
        != int(config["expected_tail_padding"])
    ):
        raise RuntimeError("CPT manifest/build geometry drift")
    if sampler_mode == "s008_speaker_interleaved":
        broad = stage_contracts.get("broad", {})
        if (
            build.get("sampling_mode")
            != "speaker_interleaved_drop_remainder_sourcecrop"
            or build.get("runtime_manifest_transcript_free") is not True
            or int(broad.get("usable_unique_rows_per_sweep", -1))
            != int(config["expected_rows"]) - int(config["expected_dropped_rows"])
            or int(broad.get("dropped_rows_per_sweep", -1))
            != int(config["expected_dropped_rows"])
            or not math.isclose(
                float(build.get("presentation_hours_full_source", -1.0)),
                float(config["expected_hours"]),
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError("S008 source-row build contract drift")
    elif int(build.get("dropped_rows_per_sweep", 0)) != 0:
        raise RuntimeError("stage-padded build unexpectedly drops source rows")
    if not paths["audio_root"].is_dir():
        raise FileNotFoundError(paths["audio_root"])
    audit = read_json(paths["model_audit"])
    if (
        audit.get("status") != "PASS"
        or audit.get("language", "lin") != language
        or not audit.get("zero_update_transfer", {}).get(
            "native_package_tensor_bit_identical"
        )
    ):
        raise RuntimeError("model composition/zero-update transfer audit is not PASS")
    return paths


def training_critical_hash(
    *, experiment_root: Path, config_path: Path, paths: dict[str, Path]
) -> str:
    packet = experiment_root / "packet/PACKET.json"
    payload = {
        "config_sha256": sha256_file(config_path),
        "packet_digest": read_json(packet).get("content_digest") if packet.is_file() else None,
        "references": {
            key: sha256_file(value) for key, value in paths.items() if value.is_file()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_fresh_pass(run_dir: Path, critical_hash: str) -> None:
    del run_dir, critical_hash
    # tools/run_recorded.py verifies the immutable packet and its digest-bound
    # independent audit immediately before the child process is created.


def require_free_gpus() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(f"GPU compute processes are active: {active}")


def require_run_disk_capacity(experiment_root: Path) -> tuple[int, int]:
    # Adapter + Adam moments + scheduler/RNG remain compact at every sweep.
    required = 2 * 1024**3
    free = shutil.disk_usage(experiment_root).free
    if free < required:
        raise RuntimeError(f"insufficient CPT disk: {free} < {required}")
    return free, required
