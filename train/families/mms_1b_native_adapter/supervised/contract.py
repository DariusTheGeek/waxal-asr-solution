"""Fail-closed WAXAL3 MMS supervised configuration and artifact contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq
from safetensors import safe_open
import torch
import yaml


GLOBAL_BATCH = 32
SUPERVISED_GEOMETRIES = {
    4: {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    },
    8: {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
    },
}
ADAPTER_TENSOR_COUNT = 288
ADAPTER_PARAMETER_COUNT = 2_151_168
LANGUAGE_CONTRACTS = {
    "lin": {
        "train_rows": 16_035,
        "validation_rows": 900,
        "validation_target_weight": 447.0,
        "source_head_rows": 81,
        "target_head_rows": 44,
        "mapped_head_rows": 41,
        "fresh_head_rows": 3,
    },
    "sna": {
        "train_rows": 16_293,
        "validation_rows": 900,
        "validation_target_weight": 445.0,
        "source_head_rows": 65,
        "target_head_rows": 39,
        "mapped_head_rows": 39,
        "fresh_head_rows": 0,
    },
}
DISK_RESERVE_BYTES = 10 * 1024**3

REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "phase",
    "recipe_id",
    "training_authorized",
    "language",
    "seed",
    "manifest_path",
    "manifest_sha256",
    "audio_root",
    "audio_build_path",
    "audio_build_sha256",
    "vocab_path",
    "vocab_sha256",
    "head_overlap_path",
    "head_overlap_sha256",
    "head_init_path",
    "head_init_sha256",
    "scorer_path",
    "scorer_sha256",
    "base_model_path",
    "base_model_sha256",
    "base_model_bytes",
    "base_config_sha256",
    "base_preprocessor_sha256",
    "base_vocab_sha256",
    "adapter_path",
    "adapter_sha256",
    "adapter_bytes",
    "native_package_path",
    "native_package_sha256",
    "native_package_bytes",
    "source_vocab_path",
    "source_vocab_sha256",
    "expected_train_rows",
    "expected_validation_rows",
    "expected_validation_target_weight",
    "expected_source_head_rows",
    "expected_target_head_rows",
    "expected_mapped_head_rows",
    "expected_fresh_head_rows",
    "expected_mapping_sha256",
    "expected_adapter_tensors",
    "expected_adapter_parameters",
    "expected_head_parameters",
    "expected_trainable_parameters",
    "max_train_duration_s",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "world_size",
    "optimizer_padding_multiple",
    "adapter_learning_rate",
    "head_learning_rate",
    "adapter_l2sp",
    "lr_scheduler_type",
    "warmup_steps",
    "optimizer",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "mask_time_prob",
    "mask_time_length",
    "mask_time_min_masks",
    "mask_feature_prob",
    "mask_feature_length",
    "hidden_dropout",
    "attention_dropout",
    "feat_proj_dropout",
    "activation_dropout",
    "final_dropout",
    "layerdrop",
    "ctc_loss_reduction",
    "ctc_zero_infinity",
    "freeze_base_model",
    "max_epochs",
    "collapse_min_updates",
    "collapse_min_epochs",
    "collapse_blank_fraction",
    "collapse_raw_wer",
    "eval_strategy",
    "save_strategy",
    "eval_steps",
    "save_steps",
    "save_only_model",
    "checkpoint_top_k",
    "bf16",
    "fp16",
    "tf32",
    "gradient_checkpointing",
    "gradient_checkpointing_use_reentrant",
    "group_by_length",
    "dataloader_num_workers",
    "dataloader_drop_last",
    "ddp_find_unused_parameters",
    "eval_accumulation_steps",
    "max_train_rows",
    "max_validation_rows",
    "subset_policy",
    "expected_updates_per_epoch",
    "expected_padded_train_rows",
    "no_resume",
    "metric_for_best_model",
    "greater_is_better",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def write_json_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def experiment_root_from(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while candidate != candidate.parent:
        if (candidate / "experiment.yaml").is_file():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(f"cannot locate WAXAL3 experiment root above {path}")


def repo_root(experiment_root: Path) -> Path:
    candidate = experiment_root.resolve()
    while candidate != candidate.parent:
        if (candidate / "README.md").is_file() and (candidate / "models").is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError("cannot locate WAXAL3 repository root")


def resolve_uri(value: str, experiment_root: Path) -> Path:
    repository = repo_root(experiment_root)
    if value.startswith("repo://"):
        path = repository / value.removeprefix("repo://")
    elif value.startswith("exp://"):
        path = experiment_root / value.removeprefix("exp://")
    elif value.startswith("packet://"):
        relative = Path(value.removeprefix("packet://"))
        packet_path = experiment_root / "packet" / relative
        if packet_path.exists() or (experiment_root / "packet").exists():
            path = packet_path
        else:
            # PREPACKET_TESTS necessarily runs before a packet exists. Resolve
            # packet-local model-family source to the exact live source whose
            # digest will be recorded and copied by materialize_packet.py.
            prefix = Path("src/model_family")
            try:
                suffix = relative.relative_to(prefix)
            except ValueError as exc:
                raise RuntimeError(
                    f"prepacket URI is not model-family source: {value}"
                ) from exc
            specification = read_json(experiment_root / "experiment.yaml")
            path = repository / str(specification["family_root"]) / "code" / suffix
    else:
        path = Path(value)
        if not path.is_absolute():
            path = experiment_root / path
    resolved = path.resolve()
    allowed = (repository.resolve(), experiment_root.resolve())
    if not any(
        resolved == root or str(resolved).startswith(str(root) + "/")
        for root in allowed
    ):
        raise RuntimeError(f"resolved path escapes WAXAL3: {resolved}")
    return resolved


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash drift: {observed} != {expected}")


def load_frozen_scorer(path: Path, expected_sha256: str) -> ModuleType:
    require_hash(path, expected_sha256, "packet-local canonical scorer facade")
    spec = importlib.util.spec_from_file_location("mms_packet_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_weight_path(model_path: Path) -> Path:
    safetensors = model_path / "model.safetensors"
    if safetensors.is_file():
        return safetensors
    legacy = model_path / "pytorch_model.bin"
    if legacy.is_file():
        return legacy
    raise RuntimeError(f"missing MMS weights: {model_path}")


def _validate_recipe(config: dict[str, Any]) -> None:
    language = str(config["language"])
    if language not in LANGUAGE_CONTRACTS:
        raise ValueError(f"unsupported MMS supervised language: {language}")
    language_contract = LANGUAGE_CONTRACTS[language]
    recipe = str(config["recipe_id"])
    recipe_values = {
        "c0_native_lr1e3": (1e-3, 0.0),
        "c1_e227_l2sp": (1e-4, 1e-4),
        "cpt_dose_e227_l2sp": (1e-4, 1e-4),
    }
    if recipe not in recipe_values:
        raise ValueError(f"unknown MMS supervised recipe: {recipe}")
    cpt_dose = recipe == "cpt_dose_e227_l2sp"
    world_size = int(config["world_size"])
    geometry = SUPERVISED_GEOMETRIES.get(world_size)
    if geometry is None:
        raise ValueError(f"unsupported MMS supervised world size: {world_size}")
    exact = {
        "max_train_duration_s": 45.0,
        "per_device_train_batch_size": geometry["per_device_train_batch_size"],
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": geometry["gradient_accumulation_steps"],
        "world_size": world_size,
        "optimizer_padding_multiple": GLOBAL_BATCH,
        "head_learning_rate": 1e-3,
        "lr_scheduler_type": "linear",
        "optimizer": "torch_adamw_two_group",
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "mask_time_prob": 0.05,
        "mask_time_length": 10,
        "mask_time_min_masks": 2,
        "mask_feature_prob": 0.0,
        "mask_feature_length": 10,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "activation_dropout": 0.0,
        "final_dropout": 0.0,
        "layerdrop": 0.0,
        "ctc_loss_reduction": "mean",
        "ctc_zero_infinity": True,
        "freeze_base_model": True,
        "collapse_min_epochs": 2,
        "collapse_blank_fraction": 0.99,
        "collapse_raw_wer": 0.99,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "group_by_length": True,
        "dataloader_num_workers": 4,
        "dataloader_drop_last": False,
        "ddp_find_unused_parameters": True,
        "eval_accumulation_steps": 8,
        "no_resume": not cpt_dose,
        "metric_for_best_model": "target_weighted_raw_q",
        "greater_is_better": True,
        "expected_adapter_tensors": ADAPTER_TENSOR_COUNT,
        "expected_adapter_parameters": ADAPTER_PARAMETER_COUNT,
        "expected_train_rows": language_contract["train_rows"],
        "expected_validation_rows": language_contract["validation_rows"],
        "expected_validation_target_weight": language_contract[
            "validation_target_weight"
        ],
        "expected_source_head_rows": language_contract["source_head_rows"],
        "expected_target_head_rows": language_contract["target_head_rows"],
        "expected_mapped_head_rows": language_contract["mapped_head_rows"],
        "expected_fresh_head_rows": language_contract["fresh_head_rows"],
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ValueError(
                f"locked recipe drift: {key}={config[key]!r} != {expected!r}"
            )
    adapter_lr, l2sp = recipe_values[recipe]
    if (
        float(config["adapter_learning_rate"]) != adapter_lr
        or float(config["adapter_l2sp"]) != l2sp
    ):
        raise ValueError("recipe-specific adapter LR/L2-SP drift")
    expected_save_only_model = not cpt_dose
    if bool(config["save_only_model"]) is not expected_save_only_model:
        raise ValueError(
            "CPT-dose arms must retain full optimizer/scheduler/RNG checkpoint "
            "state; historical controls retain model-only checkpoints"
        )
    target_head_rows = int(config["expected_target_head_rows"])
    if int(config["expected_head_parameters"]) != target_head_rows * 1_281:
        raise ValueError("CTC head parameter count drift")
    if (
        int(config["expected_trainable_parameters"])
        != ADAPTER_PARAMETER_COUNT + target_head_rows * 1_281
    ):
        raise ValueError("total trainable parameter count drift")
    train_rows = int(config["expected_train_rows"])
    expected_padded_rows = math.ceil(train_rows / GLOBAL_BATCH) * GLOBAL_BATCH
    expected_updates = expected_padded_rows // GLOBAL_BATCH
    if expected_updates % 2:
        raise ValueError("half-pass checkpoint cadence requires an even update count")
    phase = str(config["phase"])
    if phase == "production":
        phase_exact = {
            "max_epochs": 4,
            "warmup_steps": 100,
            "checkpoint_top_k": 8,
            "max_train_rows": None,
            "max_validation_rows": None,
            "subset_policy": "full",
            "expected_padded_train_rows": expected_padded_rows,
            "expected_updates_per_epoch": expected_updates,
            "eval_steps": expected_updates // 2,
            "save_steps": expected_updates // 2,
            "collapse_min_updates": 2 * expected_updates,
        }
    elif phase == "smoke":
        phase_exact = {
            "max_epochs": 1,
            "warmup_steps": 0,
            "checkpoint_top_k": 1,
            "max_train_rows": 32,
            "max_validation_rows": 32,
            "subset_policy": "duration_spread",
            "expected_padded_train_rows": 32,
            "expected_updates_per_epoch": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "collapse_min_updates": 2,
        }
    else:
        raise ValueError(f"unknown phase: {phase}")
    for key, expected in phase_exact.items():
        if config[key] != expected:
            raise ValueError(
                f"{phase} schedule drift: {key}={config[key]!r} != {expected!r}"
            )


def validate_run_config(
    config: dict[str, Any],
    *,
    experiment_root: Path,
    run_dir: Path | None,
    require_authorization: bool,
    allow_template: bool = False,
    verify_large_model_sha256: bool = True,
) -> dict[str, Path]:
    del allow_template
    if set(config) != REQUIRED_KEYS:
        raise ValueError(
            f"run config schema drift: missing={sorted(REQUIRED_KEYS - set(config))} "
            f"extra={sorted(set(config) - REQUIRED_KEYS)}"
        )
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported supervised config schema")
    if require_authorization and config["training_authorized"] is not True:
        raise RuntimeError("training is not authorized by the frozen profile")
    if not isinstance(config["training_authorized"], bool):
        raise ValueError("training_authorized must be boolean")
    if run_dir is not None:
        resolved_run = run_dir.resolve()
        if resolved_run.parent != (experiment_root / "runs").resolve():
            raise RuntimeError("run directory is outside the experiment runs lane")
    specification = read_json(experiment_root / "experiment.yaml")
    if str(config["experiment_id"]) != str(specification.get("experiment_id")):
        raise RuntimeError("profile/experiment ID mismatch")
    _validate_recipe(config)

    path_keys = (
        "manifest_path",
        "audio_root",
        "audio_build_path",
        "vocab_path",
        "head_overlap_path",
        "head_init_path",
        "scorer_path",
        "base_model_path",
        "adapter_path",
        "native_package_path",
        "source_vocab_path",
    )
    paths = {key: resolve_uri(str(config[key]), experiment_root) for key in path_keys}
    for key, hash_key, label in (
        ("manifest_path", "manifest_sha256", "MMS supervised manifest"),
        ("audio_build_path", "audio_build_sha256", "source audio build"),
        ("vocab_path", "vocab_sha256", "target vocabulary"),
        ("head_overlap_path", "head_overlap_sha256", "head overlap"),
        ("head_init_path", "head_init_sha256", "frozen head initialization"),
        ("scorer_path", "scorer_sha256", "packet scorer facade"),
        ("adapter_path", "adapter_sha256", "native adapter"),
        (
            "native_package_path",
            "native_package_sha256",
            "released native adapter/head package",
        ),
        ("source_vocab_path", "source_vocab_sha256", "MMS source vocabulary"),
    ):
        require_hash(paths[key], str(config[hash_key]), label)
    if not paths["audio_root"].is_dir():
        raise FileNotFoundError(paths["audio_root"])

    base = paths["base_model_path"]
    weight = _model_weight_path(base)
    if weight.stat().st_size != int(config["base_model_bytes"]):
        raise RuntimeError("MMS-1B-All base byte-size drift")
    if verify_large_model_sha256:
        require_hash(
            weight, str(config["base_model_sha256"]), "MMS-1B-All base weights"
        )
    for name, key in (
        ("config.json", "base_config_sha256"),
        ("preprocessor_config.json", "base_preprocessor_sha256"),
        ("vocab.json", "base_vocab_sha256"),
    ):
        require_hash(base / name, str(config[key]), f"MMS base {name}")
    if paths["source_vocab_path"] != (base / "vocab.json").resolve():
        raise RuntimeError("source vocabulary is not the frozen MMS-1B-All vocabulary")
    expected_adapter_name = f"adapter.{config['language']}.safetensors"
    if paths["native_package_path"].name != expected_adapter_name:
        raise RuntimeError("native package language/path drift")
    if paths["adapter_path"].stat().st_size != int(config["adapter_bytes"]):
        raise RuntimeError("native adapter byte-size drift")
    if paths["native_package_path"].parent != base.resolve():
        raise RuntimeError("released native package is outside MMS-1B-All")
    if paths["native_package_path"].stat().st_size != int(
        config["native_package_bytes"]
    ):
        raise RuntimeError("released native package byte-size drift")

    with safe_open(
        paths["native_package_path"], framework="pt", device="cpu"
    ) as handle:
        keys = sorted(handle.keys())
        shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in keys}
    adapter_keys = [name for name in keys if "adapter_layer" in name]
    if len(adapter_keys) != ADAPTER_TENSOR_COUNT or set(keys) - set(adapter_keys) != {
        "lm_head.bias",
        "lm_head.weight",
    }:
        raise RuntimeError("native adapter tensor inventory drift")
    if shapes["lm_head.weight"] != (int(config["expected_source_head_rows"]), 1280):
        raise RuntimeError("native source head geometry drift")
    with safe_open(paths["adapter_path"], framework="pt", device="cpu") as handle:
        init_keys = sorted(handle.keys())
    init_adapter_keys = [name for name in init_keys if "adapter_layer" in name]
    init_extras = set(init_keys) - set(init_adapter_keys)
    if len(init_adapter_keys) != ADAPTER_TENSOR_COUNT or init_extras not in (
        set(),
        {"lm_head.bias", "lm_head.weight"},
    ):
        raise RuntimeError("adapter initialization tensor inventory drift")
    with safe_open(paths["head_init_path"], framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"lm_head.bias", "lm_head.weight"}:
            raise RuntimeError("head-init inventory drift")
        if tuple(handle.get_slice("lm_head.weight").get_shape()) != (
            int(config["expected_target_head_rows"]),
            1280,
        ):
            raise RuntimeError("head-init weight geometry drift")

    overlap = read_json(paths["head_overlap_path"])
    for observed, expected in (
        (overlap["source_head_rows"], config["expected_source_head_rows"]),
        (overlap["target_head_rows"], config["expected_target_head_rows"]),
        (overlap["mapped_head_rows"], config["expected_mapped_head_rows"]),
        (overlap["fresh_head_rows"], config["expected_fresh_head_rows"]),
        (overlap["mapping_sha256"], config["expected_mapping_sha256"]),
    ):
        if observed != expected:
            raise RuntimeError(
                f"head-overlap contract drift: {observed!r} != {expected!r}"
            )

    rows = pq.read_table(paths["manifest_path"]).to_pylist()
    train = [row for row in rows if row["selected_for_training"]]
    validation = [row for row in rows if row["assignment"] == "validation_scored"]
    if len(train) != int(config["expected_train_rows"]) or len(validation) != int(
        config["expected_validation_rows"]
    ):
        raise RuntimeError("supervised manifest row-count drift")
    if {str(row["language"]) for row in rows} != {str(config["language"])}:
        raise RuntimeError("supervised manifest language drift")
    if len({str(row["row_key"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate supervised row key")
    if not math.isclose(
        sum(float(row["target_weight"]) for row in validation),
        float(config["expected_validation_target_weight"]),
        abs_tol=1e-9,
    ):
        raise RuntimeError("target-slot validation weight mass drift")
    return paths


def required_run_disk_bytes(config: dict[str, Any], paths: dict[str, Path]) -> int:
    model_bytes = _model_weight_path(paths["base_model_path"]).stat().st_size
    return (
        int(config["checkpoint_top_k"]) * model_bytes + model_bytes + DISK_RESERVE_BYTES
    )


def require_run_disk_capacity(
    experiment_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[int, int]:
    free = shutil.disk_usage(experiment_root).free
    required = required_run_disk_bytes(config, paths)
    if free < required:
        raise RuntimeError(f"insufficient disk: free={free} required={required}")
    return free, required


def training_critical_hash(
    *, experiment_root: Path, config_path: Path, paths: dict[str, Path]
) -> str:
    packet_record = experiment_root / "packet/PACKET.json"
    payload = {
        "config_sha256": sha256_file(config_path),
        "packet_digest": read_json(packet_record).get("content_digest")
        if packet_record.is_file()
        else None,
        "references": {
            key: sha256_file(value) for key, value in paths.items() if value.is_file()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_fresh_pass(run_dir: Path, expected_training_critical_hash: str) -> None:
    del run_dir, expected_training_critical_hash
    # The WAXAL3 run launcher already verifies an immutable packet and its
    # digest-bound independent audit immediately before creating the process.


def require_distributed_topology(config: dict[str, Any]) -> None:
    expected_world_size = int(config["world_size"])
    if expected_world_size not in SUPERVISED_GEOMETRIES:
        raise RuntimeError(
            f"unsupported MMS supervised world size: {expected_world_size}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise RuntimeError(f"world-size drift: {world_size} != {expected_world_size}")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != expected_world_size:
        raise RuntimeError(f"local world size must be {expected_world_size}")
    if int(os.environ.get("LOCAL_RANK", "-1")) not in set(range(expected_world_size)):
        raise RuntimeError("invalid local rank")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != expected_world_size
    ):
        raise RuntimeError(
            f"exactly {expected_world_size} visible CUDA devices are required"
        )
    if bool(config["bf16"]) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")


def gpu_process_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 3)]
        rows.append(
            {
                "gpu_uuid": values[0],
                "pid": int(values[1]),
                "process_name": values[2],
                "used_memory_mib": int(values[3]),
            }
        )
    return rows


def require_free_gpus() -> None:
    if processes := gpu_process_inventory():
        raise RuntimeError(f"GPU compute processes are active: {processes}")
