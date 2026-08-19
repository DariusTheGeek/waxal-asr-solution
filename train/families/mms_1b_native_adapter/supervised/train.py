#!/usr/bin/env python3
"""Train one isolated WAXAL3 MMS-1B native-adapter/head specialist."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoFeatureExtractor, TrainingArguments, Wav2Vec2ForCTC

from .contract import (
    experiment_root_from,
    load_frozen_scorer,
    read_json,
    require_distributed_topology,
    require_fresh_pass,
    sha256_file,
    training_critical_hash,
    validate_run_config,
    write_json_create_only,
)
from .data import AudioDataset, CharacterCodec, CTCCollator, minimum_ctc_frames
from .mms_adapter import (
    export_native_specialist,
    initialize_native_adapter_specialist,
    parameter_drift,
)
from .model import (
    CollapseGuardCallback,
    FiniteMetricCallback,
    SpecialistTrainer,
    TopKCheckpointCallback,
    align_prediction_bundle,
    checkpoint_artifact_hashes,
    load_completed_checkpoint_hash_index,
    make_metric_function,
    require_resumable_checkpoint,
    score_aligned,
    validate_loading_info,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_path_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def hardware_state() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
                "capability": torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _duration_spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError("invalid duration-spread count")
    ordered = sorted(rows, key=lambda row: (float(row["duration_s"]), str(row["id"])))
    if count == 1:
        return [ordered[-1]]
    positions = [(index * (len(ordered) - 1)) // (count - 1) for index in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError("duration-spread selector produced duplicate positions")
    return [ordered[position] for position in positions]


def select_rows(
    config: dict[str, Any],
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        row
        for row in pq.read_table(manifest_path).to_pylist()
        if row["language"] == config["language"]
    ]
    train = sorted(
        [row for row in rows if row["selected_for_training"]],
        key=lambda row: str(row["id"]),
    )
    validation = sorted(
        [row for row in rows if row["assignment"] == "validation_scored"],
        key=lambda row: int(row["evaluation_index"]),
    )
    if len(train) != int(config["expected_train_rows"]):
        raise RuntimeError("selected training row count drift")
    if len(validation) != int(config["expected_validation_rows"]):
        raise RuntimeError("selected validation row count drift")
    if config["subset_policy"] == "duration_spread":
        train = _duration_spread(train, int(config["max_train_rows"]))
        validation = _duration_spread(validation, int(config["max_validation_rows"]))
        for index, row in enumerate(validation):
            row = dict(row)
            row["evaluation_index"] = index
            validation[index] = row
    elif config["subset_policy"] != "full":
        raise RuntimeError("unknown subset policy")
    return train, validation


def verify_audio_payloads(
    rows: list[dict[str, Any]],
    audio_root: Path,
    *,
    verify_sha256: bool,
) -> dict[str, Any]:
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        identifier = str(row["id"])
        if identifier in seen:
            raise RuntimeError(f"duplicate selected audio ID: {identifier}")
        seen.add(identifier)
        path = audio_root / str(row["audio_relpath"])
        if not path.is_file() or path.stat().st_size != int(row["audio_bytes"]):
            raise RuntimeError(f"audio file/size drift: {identifier}")
        if verify_sha256 and sha256_file(path) != str(row["audio_sha256"]):
            raise RuntimeError(f"audio content hash drift: {identifier}")
        total_bytes += int(row["audio_bytes"])
    return {
        "rows": len(rows),
        "bytes": total_bytes,
        "sha256_verified": verify_sha256,
    }


def ctc_geometry(
    model: Wav2Vec2ForCTC,
    codec: CharacterCodec,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_lengths = torch.tensor(
        [int(row["input_samples"]) for row in rows], dtype=torch.long
    )
    output_lengths = model._get_feat_extract_output_lengths(sample_lengths)
    records = []
    for row, output_length in zip(rows, output_lengths.tolist()):
        required = minimum_ctc_frames(codec.encode(str(row["target_ctc"])))
        records.append(
            {
                "id": str(row["id"]),
                "input_samples": int(row["input_samples"]),
                "output_frames": int(output_length),
                "required_frames": int(required),
                "margin": int(output_length) - int(required),
            }
        )
    worst = min(records, key=lambda row: (row["margin"], row["id"]))
    if worst["margin"] < 0:
        raise RuntimeError(f"CTC geometry failed: {worst}")
    return {
        "rows": len(records),
        "minimum_margin": int(worst["margin"]),
        "minimum_margin_row": worst,
        "maximum_output_frames": max(row["output_frames"] for row in records),
        "maximum_required_frames": max(row["required_frames"] for row in records),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "evaluation_index",
        "id",
        "language",
        "model_language",
        "speaker_key",
        "stratum",
        "duration_s",
        "output_frames",
        "original_split",
        "is_phase2_test_speaker",
        "is_phase2_test_prompt",
        "target_weight",
        "slot_id",
        "target_phase2_id",
        "target_official_order",
        "reference_raw",
        "reference_ctc",
        "hypothesis",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def checkpoint_curve(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "eval_loss",
        "eval_wer",
        "eval_cer",
        "eval_weighted_error",
        "eval_q",
        "eval_raw_wer",
        "eval_raw_cer",
        "eval_raw_weighted_error",
        "eval_raw_q",
        "eval_target_weighted_raw_wer",
        "eval_target_weighted_raw_cer",
        "eval_target_weighted_raw_weighted_error",
        "eval_target_weighted_raw_q",
        "eval_blank_fraction",
    )
    curve = []
    for record in history:
        if "eval_target_weighted_raw_q" not in record:
            continue
        curve.append(
            {
                "epoch": float(record["epoch"]),
                "step": int(record["step"]),
                **{key: float(record[key]) for key in metrics if key in record},
            }
        )
    return curve


def annotate_checkpoint_retention(
    curve: list[dict[str, Any]],
    checkpoint_root: Path,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    if not curve:
        raise RuntimeError("no exact validation points recorded")
    ranked = sorted(
        curve,
        key=lambda row: (
            -float(row["eval_target_weighted_raw_q"]),
            int(row["step"]),
        ),
    )
    expected = sorted(int(row["step"]) for row in ranked[: min(limit, len(ranked))])
    observed = sorted(
        int(path.name.split("-", 1)[1])
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir()
    )
    if observed != expected:
        raise RuntimeError(
            f"retained checkpoint drift: observed={observed} expected={expected}"
        )
    recorded = load_completed_checkpoint_hash_index(
        checkpoint_root.parent / "retention_events"
    )
    retained_hashes = []
    for step in observed:
        path = checkpoint_root / f"checkpoint-{step}"
        hashes = checkpoint_artifact_hashes(path)
        if recorded.get(step) != hashes:
            raise RuntimeError(f"retained checkpoint hash/event drift at step {step}")
        retained_hashes.append(
            {
                "step": step,
                "path": f"checkpoints/checkpoint-{step}",
                "hashes": hashes,
            }
        )
    hashes_by_step = {
        int(record["step"]): record["hashes"] for record in retained_hashes
    }
    retained = set(observed)
    for row in curve:
        step = int(row["step"])
        row["checkpoint_retained"] = step in retained
        row["checkpoint_path"] = (
            f"checkpoints/checkpoint-{step}" if step in retained else None
        )
        row["checkpoint_hashes"] = hashes_by_step.get(step)
    return curve, observed, retained_hashes


def checkpoint_rankings(
    curve: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    specifications = {
        "loss": ("eval_loss", False),
        "target_weighted_raw_q": ("eval_target_weighted_raw_q", True),
        "target_weighted_raw_error": (
            "eval_target_weighted_raw_weighted_error",
            False,
        ),
        "raw_wer": ("eval_raw_wer", False),
        "raw_cer": ("eval_raw_cer", False),
        "raw_combined_error": ("eval_raw_weighted_error", False),
        "raw_q": ("eval_raw_q", True),
    }
    rankings = {}
    for name, (metric, descending) in specifications.items():
        ordered = sorted(
            [row for row in curve if metric in row],
            key=lambda row: (
                -float(row[metric]) if descending else float(row[metric]),
                int(row["step"]),
            ),
        )
        rankings[name] = [
            {
                "rank": rank,
                "step": int(row["step"]),
                "epoch": float(row["epoch"]),
                "value": float(row[metric]),
                "checkpoint_retained": bool(row["checkpoint_retained"]),
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_hashes": row["checkpoint_hashes"],
            }
            for rank, row in enumerate(ordered[: min(limit, len(ordered))], 1)
        ]
    return {
        "schema_version": 1,
        "limit": int(limit),
        "retention_authority": "target_weighted_raw_q",
        "retention_tie_break": "earlier_global_step",
        "rankings": rankings,
    }


def assert_metric_recomputation(
    prediction_metrics: dict[str, Any],
    scores: dict[str, Any],
) -> None:
    expected = {
        "validation_wer": scores["content"]["wer"],
        "validation_cer": scores["content"]["cer"],
        "validation_weighted_error": scores["content"]["weighted_error"],
        "validation_q": scores["content"]["q"],
        "validation_raw_wer": scores["raw"]["wer"],
        "validation_raw_cer": scores["raw"]["cer"],
        "validation_raw_weighted_error": scores["raw"]["weighted_error"],
        "validation_raw_q": scores["raw"]["q"],
        "validation_target_weighted_raw_wer": scores["target_weighted_raw"]["wer"],
        "validation_target_weighted_raw_cer": scores["target_weighted_raw"]["cer"],
        "validation_target_weighted_raw_weighted_error": scores["target_weighted_raw"][
            "weighted_error"
        ],
        "validation_target_weighted_raw_q": scores["target_weighted_raw"]["q"],
        "validation_blank_fraction": scores["blank_fraction"],
    }
    for key, value in expected.items():
        observed = prediction_metrics.get(key)
        if observed is None or not math.isclose(
            float(observed), float(value), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"metric recomputation drift: {key} "
                f"observed={observed} expected={value}"
            )


def _run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    experiment_root = experiment_root_from(config_path)
    if config_path != run_dir / "config.json":
        raise RuntimeError("immutable config must be <run-dir>/config.json")
    config = read_json(config_path)
    run_id = run_dir.name
    paths = validate_run_config(
        config,
        experiment_root=experiment_root,
        run_dir=run_dir,
        require_authorization=True,
    )
    require_distributed_topology(config)
    critical_hash = training_critical_hash(
        experiment_root=experiment_root,
        config_path=config_path,
        paths=paths,
    )
    require_fresh_pass(run_dir, critical_hash)
    resume_path = (
        args.resume_from_checkpoint.resolve()
        if args.resume_from_checkpoint is not None
        else None
    )
    resume_record: dict[str, Any] | None = None
    initial_outputs = (
        run_dir / "checkpoints",
        run_dir / "best_model",
        run_dir / "native_specialist",
        run_dir / "retention_events",
        run_dir / "START.json",
        run_dir / "FINAL.json",
        run_dir / "predictions.csv",
        run_dir / "checkpoint_curve.json",
        run_dir / "checkpoint_rankings.json",
    )
    terminal_outputs = (
        run_dir / "best_model",
        run_dir / "native_specialist",
        run_dir / "FINAL.json",
        run_dir / "predictions.csv",
        run_dir / "checkpoint_curve.json",
        run_dir / "checkpoint_rankings.json",
    )
    if resume_path is None:
        if any(path.exists() for path in initial_outputs):
            raise RuntimeError(
                "create-only output exists: "
                f"{[str(path) for path in initial_outputs if path.exists()]}"
            )
    else:
        if bool(config["no_resume"]) or bool(config["save_only_model"]):
            raise RuntimeError(
                "the frozen profile does not authorize exact-state resume"
            )
        if any(path.exists() for path in terminal_outputs):
            raise RuntimeError(
                "cannot resume a terminal run: "
                f"{[str(path) for path in terminal_outputs if path.exists()]}"
            )
        start_path = run_dir / "START.json"
        if not start_path.is_file():
            raise RuntimeError("resume run lacks its immutable START.json")
        start = read_json(start_path)
        if (
            start.get("training_critical_hash") != critical_hash
            or start.get("config_sha256") != sha256_file(config_path)
            or start.get("resume") is not False
        ):
            raise RuntimeError("resume run/config/packet lineage drift")
        resume_record = require_resumable_checkpoint(
            resume_path,
            checkpoint_root=run_dir / "checkpoints",
            world_size=int(config["world_size"]),
        )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_rows, validation_rows = select_rows(config, paths["manifest_path"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    audio_verification = verify_audio_payloads(
        train_rows + validation_rows,
        paths["audio_root"],
        verify_sha256=local_rank == 0,
    )
    vocabulary = read_json(paths["vocab_path"])
    codec = CharacterCodec({str(key): int(value) for key, value in vocabulary.items()})
    for row in train_rows + validation_rows:
        if codec.unk_id in codec.encode(str(row["target_ctc"])):
            raise RuntimeError(f"OOV target: {row['id']}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        paths["base_model_path"], local_files_only=True
    )
    if (
        int(feature_extractor.sampling_rate) != 16_000
        or not bool(feature_extractor.do_normalize)
        or not bool(feature_extractor.return_attention_mask)
    ):
        raise RuntimeError("MMS feature-extractor contract failed")
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(
        paths["base_model_path"],
        local_files_only=True,
        vocab_size=len(vocabulary),
        pad_token_id=codec.pad_id,
        ctc_loss_reduction=str(config["ctc_loss_reduction"]),
        ctc_zero_infinity=bool(config["ctc_zero_infinity"]),
        apply_spec_augment=True,
        mask_time_prob=float(config["mask_time_prob"]),
        mask_time_length=int(config["mask_time_length"]),
        mask_time_min_masks=int(config["mask_time_min_masks"]),
        mask_feature_prob=float(config["mask_feature_prob"]),
        mask_feature_length=int(config["mask_feature_length"]),
        hidden_dropout=float(config["hidden_dropout"]),
        attention_dropout=float(config["attention_dropout"]),
        feat_proj_dropout=float(config["feat_proj_dropout"]),
        activation_dropout=float(config["activation_dropout"]),
        final_dropout=float(config["final_dropout"]),
        layerdrop=float(config["layerdrop"]),
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
        output_loading_info=True,
    )
    validate_loading_info(loading_info)
    model.config.architectures = ["Wav2Vec2ForCTC"]
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError("E04 requires the standard linear CTC head")
    partition, l2sp_reference = initialize_native_adapter_specialist(
        model,
        adapter_path=paths["adapter_path"],
        native_package_path=paths["native_package_path"],
        source_vocab_path=paths["source_vocab_path"],
        target_vocab_path=paths["vocab_path"],
        language=str(config["language"]),
        head_init_path=paths["head_init_path"],
        expected=config,
    )
    if bool(config["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(config["gradient_checkpointing_use_reentrant"])
            }
        )
    meta_tensors = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if meta_tensors:
        raise RuntimeError(f"unmaterialized model tensors: {meta_tensors}")
    geometry = ctc_geometry(model, codec, train_rows + validation_rows)

    train_dataset = AudioDataset(
        train_rows,
        audio_root=paths["audio_root"],
        feature_extractor=feature_extractor,
        codec=codec,
    )
    validation_dataset = AudioDataset(
        validation_rows,
        audio_root=paths["audio_root"],
        feature_extractor=feature_extractor,
        codec=codec,
    )
    scorer = load_frozen_scorer(paths["scorer_path"], config["scorer_sha256"])
    metric_function = make_metric_function(
        codec=codec,
        reference_rows=validation_rows,
        scorer=scorer,
    )
    torch.set_float32_matmul_precision("high" if bool(config["tf32"]) else "highest")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy=str(config["eval_strategy"]),
        save_strategy=str(config["save_strategy"]),
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        logging_strategy="steps",
        logging_steps=1,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["head_learning_rate"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        warmup_steps=int(config["warmup_steps"]),
        weight_decay=float(config["weight_decay"]),
        adam_beta1=float(config["adam_beta1"]),
        adam_beta2=float(config["adam_beta2"]),
        adam_epsilon=float(config["adam_epsilon"]),
        max_grad_norm=float(config["max_grad_norm"]),
        num_train_epochs=float(config["max_epochs"]),
        bf16=bool(config["bf16"]),
        fp16=bool(config["fp16"]),
        tf32=bool(config["tf32"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": bool(config["gradient_checkpointing_use_reentrant"])}
            if bool(config["gradient_checkpointing"])
            else None
        ),
        group_by_length=bool(config["group_by_length"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_drop_last=bool(config["dataloader_drop_last"]),
        eval_accumulation_steps=int(config["eval_accumulation_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model=str(config["metric_for_best_model"]),
        greater_is_better=bool(config["greater_is_better"]),
        save_total_limit=None,
        save_safetensors=True,
        save_only_model=bool(config["save_only_model"]),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=bool(config["ddp_find_unused_parameters"]),
        label_names=["labels"],
        report_to=[],
        logging_nan_inf_filter=False,
    )
    collapse_callback = CollapseGuardCallback(
        minimum_updates=int(config["collapse_min_updates"]),
        minimum_epochs=int(config["collapse_min_epochs"]),
        blank_fraction=float(config["collapse_blank_fraction"]),
        raw_wer=float(config["collapse_raw_wer"]),
    )
    finite_callback = FiniteMetricCallback(
        max_consecutive_overflows=0,
        max_total_overflows=0,
    )
    trainer = SpecialistTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CTCCollator(feature_extractor, codec.pad_id),
        metric_function=metric_function,
        sampler_padding_multiple=int(config["optimizer_padding_multiple"]),
        sampler_seed=seed,
        adapter_learning_rate=float(config["adapter_learning_rate"]),
        head_learning_rate=float(config["head_learning_rate"]),
        adapter_l2sp_strength=float(config["adapter_l2sp"]),
        adapter_l2sp_reference=l2sp_reference,
        callbacks=[
            collapse_callback,
            finite_callback,
            TopKCheckpointCallback(
                limit=int(config["checkpoint_top_k"]),
                metric=f"eval_{config['metric_for_best_model']}",
                greater_is_better=bool(config["greater_is_better"]),
            ),
        ],
    )
    finite_callback.bind_accelerator(trainer.accelerator)
    train_dataloader = trainer.get_train_dataloader()
    train_sampler = getattr(trainer, "_e04_train_sampler", None)
    if train_sampler is None or int(train_sampler.group_batch_size) != int(
        config["optimizer_padding_multiple"]
    ):
        raise RuntimeError("length grouping is not aligned to the global batch")
    batches_per_rank = len(train_dataloader)
    if batches_per_rank % int(config["gradient_accumulation_steps"]) != 0:
        raise RuntimeError("partial gradient accumulation remains at epoch boundary")
    observed_updates = batches_per_rank // int(config["gradient_accumulation_steps"])
    expected_updates = int(config["expected_updates_per_epoch"])
    if observed_updates != expected_updates:
        raise RuntimeError(
            f"prepared dataloader update drift: {observed_updates} != {expected_updates}"
        )

    if trainer.is_world_process_zero():
        if resume_record is None:
            packet_content_digest = read_json(experiment_root / "packet/PACKET.json")[
                "content_digest"
            ]
            write_json_create_only(
                run_dir / "START.json",
                {
                    "schema_version": 1,
                    "started_at_utc": utc_now(),
                    "command": [sys.executable, *sys.argv],
                    "config": config,
                    "config_sha256": sha256_file(config_path),
                    "training_critical_hash": critical_hash,
                    "packet_content_digest": packet_content_digest,
                    "platform": platform.platform(),
                    "python": sys.version,
                    "hardware": hardware_state(),
                    "world_size": training_args.world_size,
                    "global_batch": (
                        int(config["per_device_train_batch_size"])
                        * int(config["gradient_accumulation_steps"])
                        * int(config["world_size"])
                    ),
                    "length_group_batch": int(train_sampler.group_batch_size),
                    "precision": "bf16",
                    "optimizer": {
                        "class": "torch.optim.AdamW",
                        "fused": False,
                        "adapter_learning_rate": config["adapter_learning_rate"],
                        "head_learning_rate": config["head_learning_rate"],
                        "adapter_l2sp": config["adapter_l2sp"],
                    },
                    "batches_per_rank_per_epoch": batches_per_rank,
                    "updates_per_epoch": observed_updates,
                    "train_rows": len(train_rows),
                    "padded_train_rows": (
                        (
                            len(train_rows)
                            + int(config["optimizer_padding_multiple"])
                            - 1
                        )
                        // int(config["optimizer_padding_multiple"])
                    )
                    * int(config["optimizer_padding_multiple"]),
                    "train_hours": sum(float(row["duration_s"]) for row in train_rows)
                    / 3600.0,
                    "validation_rows": len(validation_rows),
                    "validation_hours": sum(
                        float(row["duration_s"]) for row in validation_rows
                    )
                    / 3600.0,
                    "audio_verification": audio_verification,
                    "ctc_geometry": geometry,
                    "loading_info": loading_info,
                    "parameter_partition": partition,
                    "phase2_transcripts_accessed": False,
                    "released_native_specialist_weights_loaded": True,
                    "resume": False,
                },
            )
        else:
            event_dir = run_dir / "resume_events"
            event_dir.mkdir(exist_ok=True)
            write_json_create_only(
                event_dir / f"RESUME_{utc_path_timestamp()}.json",
                {
                    "schema_version": 1,
                    "started_at_utc": utc_now(),
                    "command": [sys.executable, *sys.argv],
                    "config_sha256": sha256_file(config_path),
                    "training_critical_hash": critical_hash,
                    "hardware": hardware_state(),
                    "world_size": training_args.world_size,
                    "global_batch": (
                        int(config["per_device_train_batch_size"])
                        * int(config["gradient_accumulation_steps"])
                        * int(config["world_size"])
                    ),
                    "resume": True,
                    "resume_checkpoint": resume_record,
                },
            )
    trainer.accelerator.wait_for_everyone()

    result = trainer.train(
        resume_from_checkpoint=str(resume_path) if resume_path is not None else None
    )
    successful_optimizer_steps = int(trainer.state.global_step) - int(
        finite_callback.total_overflows
    )
    if successful_optimizer_steps < 1:
        raise RuntimeError("training completed without a successful optimizer step")
    prediction = trainer.predict(validation_dataset, metric_key_prefix="validation")
    if not trainer.is_world_process_zero():
        return 0

    trainer.save_model(str(run_dir / "best_model"))
    feature_extractor.save_pretrained(str(run_dir / "best_model"))
    shutil.copyfile(paths["vocab_path"], run_dir / "best_model" / "vocab.json")
    best_base = trainer.accelerator.unwrap_model(trainer.model)
    drift = parameter_drift(best_base, l2sp_reference)
    native_dir = run_dir / "native_specialist"
    native_dir.mkdir()
    export = export_native_specialist(
        best_base,
        native_dir / f"adapter.{config['language']}.safetensors",
        metadata={
            "schema_version": "1",
            "language": str(config["language"]),
            "run_id": run_id,
            "base_model_sha256": str(config["base_model_sha256"]),
            "target_vocab_sha256": str(config["vocab_sha256"]),
            "source_adapter_sha256": str(config["adapter_sha256"]),
        },
    )
    shutil.copyfile(paths["vocab_path"], native_dir / "vocab.json")
    write_json_create_only(native_dir / "parameter_drift.json", drift)
    write_json_create_only(
        native_dir / "metadata.json",
        {
            "schema_version": 1,
            "language": config["language"],
            "run_id": run_id,
            "base_model_path": config["base_model_path"],
            "base_model_sha256": config["base_model_sha256"],
            "source_adapter_path": config["adapter_path"],
            "source_adapter_sha256": config["adapter_sha256"],
            "target_vocab_sha256": config["vocab_sha256"],
            "head_overlap": partition,
            "parameter_drift": drift,
            "export": export,
        },
    )
    aligned = align_prediction_bundle(
        predictions=prediction.predictions,
        label_bundle=prediction.label_ids,
        codec=codec,
        reference_rows=validation_rows,
    )
    scores = score_aligned(aligned, scorer)
    assert_metric_recomputation(prediction.metrics, scores)
    write_predictions(run_dir / "predictions.csv", aligned)

    best_weight = run_dir / "best_model" / "model.safetensors"
    if not best_weight.is_file() or trainer.state.best_model_checkpoint is None:
        raise RuntimeError("Trainer did not retain an exact-score best model")
    checkpoint_weight = Path(trainer.state.best_model_checkpoint) / "model.safetensors"
    if sha256_file(best_weight) != sha256_file(checkpoint_weight):
        raise RuntimeError("best-model copy is not byte-identical to checkpoint")
    curve, retained_steps, retained_checkpoint_hashes = annotate_checkpoint_retention(
        checkpoint_curve(trainer.state.log_history),
        run_dir / "checkpoints",
        limit=int(config["checkpoint_top_k"]),
    )
    write_json_create_only(run_dir / "checkpoint_curve.json", curve)
    rankings = checkpoint_rankings(curve, limit=int(config["checkpoint_top_k"]))
    write_json_create_only(run_dir / "checkpoint_rankings.json", rankings)
    best_index = min(
        range(len(curve)),
        key=lambda index: (
            -float(curve[index]["eval_target_weighted_raw_q"]),
            int(curve[index]["step"]),
        ),
    )
    if trainer.state.best_metric is None or not math.isclose(
        float(trainer.state.best_metric),
        float(curve[best_index]["eval_target_weighted_raw_q"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Trainer best metric disagrees with exact curve")
    trailing = len(curve) - best_index - 1
    if collapse_callback.triggered:
        completion = "FAILED_COLLAPSE"
    elif float(trainer.state.epoch or 0.0) >= int(config["max_epochs"]):
        completion = "FIXED_HORIZON_REACHED"
    else:
        completion = "FAILED_EARLY_TERMINATION"
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "run_id": run_id,
        "config_sha256": sha256_file(config_path),
        "phase": config["phase"],
        "language": config["language"],
        "completion_status": completion,
        "global_step": int(trainer.state.global_step),
        "successful_optimizer_steps": successful_optimizer_steps,
        "bf16_optimizer_steps_skipped": int(finite_callback.total_overflows),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "best_checkpoint": str(trainer.state.best_model_checkpoint),
        "best_metric": float(trainer.state.best_metric),
        "best_curve_index": best_index,
        "non_improving_evaluations_after_best": trailing,
        "retained_checkpoint_steps": retained_steps,
        "retained_checkpoint_hashes": retained_checkpoint_hashes,
        "best_model_sha256": sha256_file(best_weight),
        "best_model_config_sha256": sha256_file(run_dir / "best_model" / "config.json"),
        "best_model_preprocessor_sha256": sha256_file(
            run_dir / "best_model" / "preprocessor_config.json"
        ),
        "best_model_vocab_sha256": sha256_file(run_dir / "best_model" / "vocab.json"),
        "native_specialist_export": export,
        "native_specialist_metadata_sha256": sha256_file(native_dir / "metadata.json"),
        "native_specialist_vocab_sha256": sha256_file(native_dir / "vocab.json"),
        "parameter_drift": drift,
        "predictions_sha256": sha256_file(run_dir / "predictions.csv"),
        "train_metrics": result.metrics,
        "validation_metrics": prediction.metrics,
        "exact_scores": scores,
        "phase2_transcripts_accessed": False,
        "released_native_specialist_weights_loaded": True,
        "submission_created": False,
    }
    write_json_create_only(run_dir / "FINAL.json", final)
    with (run_dir / "RESULT.md").open("x", encoding="utf-8") as handle:
        handle.write(
            f"# {run_id}\n\n"
            f"- Language: `{config['language']}`\n"
            f"- Completion: `{completion}`\n"
            f"- Epoch / step: `{trainer.state.epoch}` / `{trainer.state.global_step}`\n"
            f"- Best checkpoint: `{trainer.state.best_model_checkpoint}`\n"
            f"- Raw WER/CER/Q: `{scores['raw']['wer']:.9f}` / "
            f"`{scores['raw']['cer']:.9f}` / `{scores['raw']['q']:.9f}`\n"
            f"- Target-weighted raw WER/CER/Q: "
            f"`{scores['target_weighted_raw']['wer']:.9f}` / "
            f"`{scores['target_weighted_raw']['cer']:.9f}` / "
            f"`{scores['target_weighted_raw']['q']:.9f}`\n"
            f"- Blank fraction: `{scores['blank_fraction']:.9f}`\n"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        return _run(args)
    except Exception:
        if int(os.environ.get("RANK", "0")) == 0:
            payload = {
                "schema_version": 1,
                "failed_at_utc": utc_now(),
                "resume_from_checkpoint": (
                    str(args.resume_from_checkpoint.resolve())
                    if args.resume_from_checkpoint is not None
                    else None
                ),
                "error": traceback.format_exc(),
            }
            failure_events = args.run_dir.resolve() / "failure_events"
            failure_events.mkdir(parents=True, exist_ok=True)
            write_json_create_only(
                failure_events / f"FAILURE_{utc_path_timestamp()}.json",
                payload,
            )
            failure = args.run_dir.resolve() / "FAILURE.json"
            if not failure.exists():
                write_json_create_only(failure, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
