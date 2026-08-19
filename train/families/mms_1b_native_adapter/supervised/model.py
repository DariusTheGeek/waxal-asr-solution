#!/usr/bin/env python3
"""E04 exact metrics, adapter-only Trainer, and checkpoint callbacks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from types import MethodType
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import IntervalStrategy

from .data import (
    CharacterCodec,
    OptimizationBatchPaddedLengthSampler,
)
from .mms_adapter import adapter_l2sp_penalty, two_group_optimizer_parameters


EXPECTED_RESIZED_HEAD_KEYS = {"lm_head.bias", "lm_head.weight"}
LEGACY_CHECKPOINT_HASH_FILES = (
    "model.safetensors",
    "config.json",
    "trainer_state.json",
)
FULL_STATE_CHECKPOINT_FILES = {
    "model.safetensors",
    "config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_artifact_hashes(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(path.iterdir(), key=lambda candidate: candidate.name)
    unsafe = [candidate for candidate in candidates if candidate.is_symlink()]
    non_files = [
        candidate
        for candidate in candidates
        if not candidate.is_symlink() and not candidate.is_file()
    ]
    if unsafe or non_files:
        raise RuntimeError(
            f"unsafe checkpoint inventory: symlinks={unsafe} non_files={non_files}"
        )
    names = [candidate.name for candidate in candidates]
    if not set(LEGACY_CHECKPOINT_HASH_FILES) <= set(names):
        raise RuntimeError(
            "checkpoint hash input missing: "
            f"{sorted(set(LEGACY_CHECKPOINT_HASH_FILES) - set(names))}"
        )
    files = []
    for candidate in candidates:
        files.append(
            {
                "name": candidate.name,
                "bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 2,
        "files": files,
        "checkpoint_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_cached_checkpoint_hashes(
    path: Path,
    hashes: dict[str, Any],
) -> None:
    if set(hashes) != {"schema_version", "files", "checkpoint_sha256"}:
        raise RuntimeError(f"checkpoint hash schema drift: {path}")
    schema_version = int(hashes["schema_version"])
    names = [record.get("name") for record in hashes["files"]]
    if schema_version == 1:
        if names != list(LEGACY_CHECKPOINT_HASH_FILES):
            raise RuntimeError(f"legacy checkpoint hash schema drift: {path}")
    elif schema_version == 2:
        if (
            names != sorted(names)
            or len(names) != len(set(names))
            or not set(LEGACY_CHECKPOINT_HASH_FILES) <= set(names)
        ):
            raise RuntimeError(f"complete checkpoint inventory schema drift: {path}")
        observed_names = sorted(
            candidate.name
            for candidate in path.iterdir()
            if candidate.is_file() and not candidate.is_symlink()
        )
        if observed_names != names:
            raise RuntimeError(
                f"checkpoint inventory drift: {observed_names} != {names}"
            )
    else:
        raise RuntimeError(f"unsupported checkpoint hash schema: {schema_version}")
    for record in hashes["files"]:
        candidate = path / str(record["name"])
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != int(record["bytes"])
            or len(str(record.get("sha256", ""))) != 64
        ):
            raise RuntimeError(f"cached checkpoint file/size drift: {candidate}")
    encoded = json.dumps(
        hashes["files"], sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != hashes["checkpoint_sha256"]:
        raise RuntimeError(f"cached checkpoint digest drift: {path}")


def require_resumable_checkpoint(
    checkpoint: Path,
    *,
    checkpoint_root: Path,
    world_size: int,
) -> dict[str, Any]:
    """Verify one fully committed same-run checkpoint before exact-state resume."""

    checkpoint = checkpoint.resolve()
    checkpoint_root = checkpoint_root.resolve()
    if checkpoint.parent != checkpoint_root or not checkpoint.name.startswith(
        "checkpoint-"
    ):
        raise RuntimeError("resume checkpoint must be a direct same-run checkpoint")
    try:
        step = int(checkpoint.name.split("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"malformed resume checkpoint: {checkpoint}") from exc
    if step < 1:
        raise RuntimeError("resume checkpoint step must be positive")
    cached = load_completed_checkpoint_hash_index(
        checkpoint_root.parent / "retention_events"
    ).get(step)
    if cached is None or int(cached.get("schema_version", -1)) != 2:
        raise RuntimeError("resume checkpoint lacks a complete inventory event")
    _validate_cached_checkpoint_hashes(checkpoint, cached)
    observed = checkpoint_artifact_hashes(checkpoint)
    if observed != cached:
        raise RuntimeError("resume checkpoint content does not match its commit event")
    names = {str(record["name"]) for record in observed["files"]}
    required = FULL_STATE_CHECKPOINT_FILES | {
        f"rng_state_{rank}.pth" for rank in range(int(world_size))
    }
    if missing := sorted(required - names):
        raise RuntimeError(f"resume checkpoint is missing full state: {missing}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError("resume checkpoint trainer-state step drift")
    if int(state.get("max_steps", -1)) < step:
        raise RuntimeError("resume checkpoint exceeds its declared schedule")
    return {
        "schema_version": 1,
        "step": step,
        "path": str(checkpoint),
        "checkpoint_sha256": observed["checkpoint_sha256"],
        "files": observed["files"],
    }


def load_completed_checkpoint_hash_index(
    event_dir: Path,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for event_path in sorted(
        event_dir.glob("step-*.complete.json"),
        key=lambda path: int(path.name.split(".", 1)[0].split("-", 1)[1]),
    ):
        with event_path.open(encoding="utf-8") as handle:
            event = json.load(handle)
        if (
            event.get("schema_version") != 2
            or event.get("event_status") != "COMPLETED"
            or not isinstance(event.get("checkpoint_records"), list)
        ):
            raise RuntimeError(f"completed retention-event schema drift: {event_path}")
        for record in event["checkpoint_records"]:
            step = int(record["step"])
            hashes = record["hashes"]
            if not isinstance(hashes, dict):
                raise RuntimeError(f"checkpoint hashes missing in {event_path}")
            result[step] = hashes
    return result


def validate_loading_info(loading_info: dict[str, Any]) -> None:
    observed_missing = set(loading_info.get("missing_keys", []))
    observed_unexpected = set(loading_info.get("unexpected_keys", []))
    mismatched = loading_info.get("mismatched_keys", [])
    errors = loading_info.get("error_msgs", [])
    mismatched_names = set()
    for item in mismatched:
        if isinstance(item, (tuple, list)) and item:
            mismatched_names.add(str(item[0]))
        elif isinstance(item, dict) and "key" in item:
            mismatched_names.add(str(item["key"]))
        else:
            raise RuntimeError(f"unknown loading-info mismatch record: {item!r}")
    affected = observed_missing | mismatched_names
    if affected != EXPECTED_RESIZED_HEAD_KEYS or observed_unexpected or errors:
        raise RuntimeError(
            "MMS-1B resized-head loading contract failed: "
            f"missing={sorted(observed_missing)} "
            f"unexpected={sorted(observed_unexpected)} "
            f"mismatched={mismatched} errors={errors}"
        )


def minimum_ctc_lengths(labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0
    lengths = mask.sum(-1)
    repeated = ((labels[:, 1:] == labels[:, :-1]) & mask[:, 1:] & mask[:, :-1]).sum(-1)
    return lengths + repeated


class SpecialistTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        metric_function: Callable[[Any], dict[str, float]],
        sampler_padding_multiple: int,
        sampler_seed: int,
        adapter_learning_rate: float,
        head_learning_rate: float,
        adapter_l2sp_strength: float,
        adapter_l2sp_reference: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        self.sampler_padding_multiple = int(sampler_padding_multiple)
        self.sampler_seed = int(sampler_seed)
        self.adapter_learning_rate = float(adapter_learning_rate)
        self.head_learning_rate = float(head_learning_rate)
        self.adapter_l2sp_strength = float(adapter_l2sp_strength)
        self.adapter_l2sp_reference = adapter_l2sp_reference
        self._adapter_l2sp_device_cache: dict[str, torch.Tensor] = {}
        if self.adapter_learning_rate <= 0.0 or self.head_learning_rate <= 0.0:
            raise ValueError("E04 optimizer learning rates must be positive")
        if self.adapter_l2sp_strength < 0.0:
            raise ValueError("native-adapter L2-SP strength must be non-negative")
        if len(self.adapter_l2sp_reference) != 288:
            raise ValueError("E04 requires exactly 288 L2-SP reference tensors")
        super().__init__(*args, compute_metrics=metric_function, **kwargs)

    def _get_train_sampler(self, *args: Any, **kwargs: Any):
        if self.args.group_by_length and hasattr(self.train_dataset, "lengths"):
            sampler = OptimizationBatchPaddedLengthSampler(
                lengths=self.train_dataset.lengths,
                group_batch_size=(
                    int(self.args.per_device_train_batch_size)
                    * int(self.args.gradient_accumulation_steps)
                    * int(self.args.world_size)
                ),
                padding_multiple=self.sampler_padding_multiple,
                seed=self.sampler_seed,
            )
            self._e04_train_sampler = sampler
            return sampler
        raise RuntimeError("E04 requires its exact length-grouped padded sampler")

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        sampler = getattr(self, "_e04_train_sampler", None)
        if sampler is None:
            raise RuntimeError("exact train sampler was not constructed")
        inherited_set_epoch = getattr(dataloader, "set_epoch", None)
        if inherited_set_epoch is None:
            raise RuntimeError("prepared train dataloader lost set_epoch")

        def set_epoch(loader, epoch: int) -> None:
            sampler.set_epoch(int(epoch))
            inherited_set_epoch(int(epoch))

        # Accelerate 1.1.1's DataLoaderShard updates its own iteration but does
        # not forward set_epoch to a custom sampler. Bind an exact forwarding
        # method so Trainer's epoch loop changes both in lockstep.
        dataloader.set_epoch = MethodType(set_epoch, dataloader)
        return dataloader

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def create_optimizer(self):
        if self.optimizer is None:
            base = self.accelerator.unwrap_model(self.model)
            adapters, head = two_group_optimizer_parameters(base)
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": adapters,
                        "lr": self.adapter_learning_rate,
                        "group_name": "native_adapter",
                    },
                    {
                        "params": head,
                        "lr": self.head_learning_rate,
                        "group_name": "ctc_head",
                    },
                ],
                lr=self.head_learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise RuntimeError("E04 optimizer is not torch.optim.AdamW")
        groups = self.optimizer.param_groups
        if (
            len(groups) != 2
            or groups[0].get("group_name") != "native_adapter"
            or groups[1].get("group_name") != "ctc_head"
            or float(groups[0]["lr"]) != self.adapter_learning_rate
            or float(groups[1]["lr"]) != self.head_learning_rate
            or self.optimizer.defaults.get("fused") is True
        ):
            raise RuntimeError(
                f"E04 two-group non-fused AdamW contract failed: {groups}"
            )
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        mutable = dict(inputs)
        mutable.pop("example_index", None)
        loss = super().compute_loss(
            model,
            mutable,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        value = loss[0] if return_outputs else loss
        if not torch.isfinite(value):
            raise FloatingPointError("non-finite CTC loss")
        if model.training and self.adapter_l2sp_strength > 0.0:
            base = self.accelerator.unwrap_model(model)
            penalty = adapter_l2sp_penalty(
                base,
                self.adapter_l2sp_reference,
                self._adapter_l2sp_device_cache,
            )
            value = value.float() + self.adapter_l2sp_strength * penalty
        if not torch.isfinite(value):
            raise FloatingPointError("non-finite E04 CTC + L2-SP loss")
        if return_outputs:
            return value, loss[1]
        return value

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        mutable = dict(inputs)
        example_index = mutable.pop("example_index", None)
        if example_index is None:
            raise RuntimeError("evaluation batch lost example_index")
        attention_mask = mutable.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("evaluation batch lost attention_mask")
        base = self.accelerator.unwrap_model(model)
        output_lengths = base._get_feat_extract_output_lengths(
            attention_mask.sum(-1).to(self.args.device)
        ).to(torch.long)
        loss, logits, labels = super().prediction_step(
            model,
            mutable,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if prediction_loss_only:
            return loss, logits, labels
        if labels is None or logits is None:
            raise RuntimeError("evaluation lost logits or labels")
        if bool((output_lengths <= 0).any()) or bool(
            (output_lengths > int(logits.shape[1])).any()
        ):
            raise RuntimeError("evaluation output-length contract failed")
        return (
            loss,
            logits,
            (
                labels,
                example_index.to(self.args.device),
                output_lengths.unsqueeze(-1),
            ),
        )


class DelayedEarlyStoppingCallback(TrainerCallback):
    """Strict raw-error improvement with a fixed minimum completed epoch."""

    def __init__(
        self,
        *,
        minimum_epochs: int,
        patience: int,
        metric: str = "eval_raw_weighted_error",
    ) -> None:
        self.minimum_epochs = int(minimum_epochs)
        self.patience = int(patience)
        self.metric = metric
        self.best = math.inf
        self.bad = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        if self.metric not in metrics:
            raise RuntimeError(f"early-stop metric missing: {self.metric}")
        value = float(metrics[self.metric])
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite early-stop metric: {value}")
        if value < self.best:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        completed = int(math.floor(float(state.epoch or 0.0) + 1e-8))
        if completed >= self.minimum_epochs and self.bad >= self.patience:
            control.should_training_stop = True
        return control


class CollapseGuardCallback(TrainerCallback):
    def __init__(
        self,
        *,
        minimum_updates: int,
        minimum_epochs: int,
        blank_fraction: float,
        raw_wer: float,
    ) -> None:
        self.minimum_updates = int(minimum_updates)
        self.minimum_epochs = int(minimum_epochs)
        self.blank_fraction = float(blank_fraction)
        self.raw_wer = float(raw_wer)
        self.triggered = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        required = {"eval_blank_fraction", "eval_raw_wer"}
        if required - set(metrics):
            raise RuntimeError("collapse diagnostics missing from evaluation")
        completed = int(math.floor(float(state.epoch or 0.0) + 1e-8))
        eligible = (
            int(state.global_step) >= self.minimum_updates
            and completed >= self.minimum_epochs
        )
        collapsed = (
            float(metrics["eval_blank_fraction"]) >= self.blank_fraction
            and float(metrics["eval_raw_wer"]) >= self.raw_wer
        )
        if eligible and collapsed:
            self.triggered = True
            control.should_training_stop = True
            print(
                "COLLAPSE_KILL_TRIGGER "
                f"step={state.global_step} epoch={completed} "
                f"blank_fraction={metrics['eval_blank_fraction']} "
                f"raw_wer={metrics['eval_raw_wer']}",
                flush=True,
            )
        return control


class FiniteMetricCallback(TrainerCallback):
    """Reject non-finite state while permitting bounded dynamic-AMP retries."""

    def __init__(
        self,
        *,
        max_consecutive_overflows: int,
        max_total_overflows: int,
    ) -> None:
        self.max_consecutive_overflows = int(max_consecutive_overflows)
        self.max_total_overflows = int(max_total_overflows)
        if self.max_consecutive_overflows < 0 or self.max_total_overflows < 0:
            raise ValueError("AMP overflow limits must be non-negative")
        self.total_overflows = 0
        self.scaler_overflows = 0
        self.gradient_norm_overflows = 0
        self.consecutive_overflows = 0
        self.maximum_consecutive_overflows = 0
        self.last_optimizer_step_skipped = False
        self.accelerator = None

    def _record_overflow(self, kind: str, attempted_step: int) -> None:
        self.total_overflows += 1
        if kind == "scaler":
            self.scaler_overflows += 1
        elif kind == "positive_inf_grad_norm":
            self.gradient_norm_overflows += 1
        else:
            raise ValueError(f"unknown overflow kind: {kind}")
        self.consecutive_overflows += 1
        self.maximum_consecutive_overflows = max(
            self.maximum_consecutive_overflows,
            self.consecutive_overflows,
        )
        scaler = getattr(self.accelerator, "scaler", None)
        scale = (
            float(scaler.get_scale())
            if scaler is not None and hasattr(scaler, "get_scale")
            else None
        )
        print(
            f"AMP_{kind.upper()}_OVERFLOW "
            f"attempted_step={int(attempted_step)} "
            f"consecutive={self.consecutive_overflows} "
            f"total={self.total_overflows} scale={scale}",
            flush=True,
        )
        if (
            self.consecutive_overflows > self.max_consecutive_overflows
            or self.total_overflows > self.max_total_overflows
        ):
            raise FloatingPointError(
                "native FP16 overflow budget exhausted: "
                f"consecutive={self.consecutive_overflows}/"
                f"{self.max_consecutive_overflows} "
                f"total={self.total_overflows}/{self.max_total_overflows}"
            )

    def bind_accelerator(self, accelerator: Any) -> None:
        if self.accelerator is not None:
            raise RuntimeError("finite guard accelerator is already bound")
        if not hasattr(accelerator, "optimizer_step_was_skipped"):
            raise TypeError("finite guard requires an Accelerate accelerator")
        self.accelerator = accelerator

    def on_optimizer_step(
        self,
        args,
        state,
        control,
        optimizer=None,
        **kwargs,
    ):
        if self.accelerator is None:
            raise RuntimeError("finite guard accelerator was not bound")
        skipped = bool(self.accelerator.optimizer_step_was_skipped)
        self.last_optimizer_step_skipped = skipped
        if not skipped:
            return control
        if not bool(args.fp16):
            raise FloatingPointError("optimizer step skipped outside native FP16")
        self._record_overflow("scaler", int(state.global_step) + 1)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in (logs or {}).items():
            if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                if (
                    key == "grad_norm"
                    and bool(args.fp16)
                    and self.last_optimizer_step_skipped
                    and math.isinf(float(value))
                    and float(value) > 0
                ):
                    continue
                if (
                    key == "grad_norm"
                    and bool(args.fp16)
                    and not self.last_optimizer_step_skipped
                    and math.isinf(float(value))
                    and float(value) > 0
                ):
                    # The installed Trainer/Accelerate combination can expose
                    # a positive-infinite clipped norm while its public skip
                    # flag remains false, even when the scale subsequently
                    # backs off. Treat the observed norm as authoritative:
                    # count this as ineffective and require a later finite-norm
                    # optimizer update.
                    self._record_overflow(
                        "positive_inf_grad_norm", int(state.global_step)
                    )
                    continue
                raise FloatingPointError(f"non-finite logged value: {key}={value}")
        if (
            "grad_norm" in (logs or {})
            and math.isfinite(float(logs["grad_norm"]))
            and not self.last_optimizer_step_skipped
        ):
            self.consecutive_overflows = 0
        return control


class TopKCheckpointCallback(TrainerCallback):
    """Retain the exact best K checkpoints by the declared promotion metric."""

    def __init__(
        self,
        *,
        limit: int,
        metric: str = "eval_target_weighted_raw_q",
        greater_is_better: bool = True,
    ) -> None:
        self.limit = int(limit)
        self.metric = metric
        self.greater_is_better = bool(greater_is_better)
        if self.limit < 1:
            raise ValueError("checkpoint top-k limit must be positive")

    def on_step_end(self, args, state, control, **kwargs):
        if (
            int(state.global_step) >= int(state.max_steps)
            and args.eval_strategy == IntervalStrategy.EPOCH
            and args.save_strategy == IntervalStrategy.EPOCH
        ):
            # DefaultFlowCallback requests an unscored terminal save just before
            # its final epoch evaluation. The epoch-end callback requests the
            # canonical evaluated save at the same step.
            control.should_save = False
        return control

    def on_save(self, args, state, control, **kwargs):
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed:
            # Every rank writes its own RNG state. Do not commit or hash the
            # checkpoint until all rank-local files are durable.
            torch.distributed.barrier()
        if not args.should_save:
            if distributed:
                torch.distributed.barrier()
            return control
        output = Path(args.output_dir)
        current = output / f"checkpoint-{int(state.global_step)}"
        if not current.is_dir():
            raise RuntimeError(f"new checkpoint missing before top-k: {current}")
        scores: dict[int, float] = {}
        for record in state.log_history:
            if self.metric in record and "step" in record:
                step = int(record["step"])
                value = float(record[self.metric])
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite checkpoint metric")
                scores[step] = value
        if int(state.global_step) not in scores:
            raise RuntimeError("current checkpoint has no exact validation score")
        ranked = sorted(
            scores,
            key=lambda step: (
                -scores[step] if self.greater_is_better else scores[step],
                step,
            ),
        )
        retained = set(ranked[: self.limit])
        existing: dict[int, Path] = {}
        for path in output.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.split("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"malformed checkpoint: {path}") from exc
            existing[step] = path
        deleted = sorted(set(existing) - retained)
        best = Path(str(state.best_model_checkpoint or ""))
        if best.name.startswith("checkpoint-"):
            best_step = int(best.name.split("-", 1)[1])
            if best_step not in retained:
                raise RuntimeError("top-k policy would delete Trainer best")
        event_dir = output.parent / "retention_events"
        event_dir.mkdir(exist_ok=True)
        cached = load_completed_checkpoint_hash_index(event_dir)
        checkpoint_records = []
        for step, path in sorted(existing.items()):
            prior_hashes = cached.get(step)
            if (
                prior_hashes is None
                or step == int(state.global_step)
                or step in deleted
            ):
                hashes = checkpoint_artifact_hashes(path)
                if prior_hashes is not None and hashes != prior_hashes:
                    raise RuntimeError(f"checkpoint content drift before prune: {path}")
            else:
                hashes = prior_hashes
                _validate_cached_checkpoint_hashes(path, prior_hashes)
            checkpoint_records.append(
                {
                    "step": step,
                    "path": f"checkpoints/checkpoint-{step}",
                    "decision": "retained" if step in retained else "deleted",
                    "metric_value": scores[step],
                    "hashes": hashes,
                }
            )
        event_payload = {
            "schema_version": 2,
            "step": int(state.global_step),
            "metric": self.metric,
            "greater_is_better": self.greater_is_better,
            "limit": self.limit,
            "ranked_steps": ranked,
            "retained_steps": sorted(retained),
            "deleted_steps": deleted,
            "checkpoint_records": checkpoint_records,
        }
        intent = event_dir / f"step-{int(state.global_step)}.intent.json"
        with intent.open("x", encoding="utf-8") as handle:
            json.dump(
                {**event_payload, "event_status": "HASHED_BEFORE_PRUNE"},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        for step in deleted:
            shutil.rmtree(existing[step])
        observed_after = sorted(
            int(path.name.split("-", 1)[1])
            for path in output.glob("checkpoint-*")
            if path.is_dir()
        )
        if observed_after != sorted(retained):
            raise RuntimeError(
                "checkpoint prune result drift: "
                f"observed={observed_after} expected={sorted(retained)}"
            )
        complete = event_dir / f"step-{int(state.global_step)}.complete.json"
        with complete.open("x", encoding="utf-8") as handle:
            json.dump(
                {**event_payload, "event_status": "COMPLETED"},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print(
            f"TOPK_CHECKPOINTS step={state.global_step} "
            f"retained={sorted(retained)} deleted={deleted}",
            flush=True,
        )
        if distributed:
            torch.distributed.barrier()
        return control


def align_prediction_bundle(
    *,
    predictions: np.ndarray,
    label_bundle: Any,
    codec: CharacterCodec,
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(label_bundle, (tuple, list)) or len(label_bundle) != 3:
        raise RuntimeError("evaluation labels lost index/output-length bundle")
    label_ids = np.asarray(label_bundle[0])
    indices = np.asarray(label_bundle[1]).reshape(-1)
    output_lengths = np.asarray(label_bundle[2]).reshape(-1)
    logits = np.asarray(predictions)
    if logits.ndim != 3:
        raise RuntimeError("prediction logits must be batch/time/vocabulary")
    prediction_ids = np.argmax(logits, axis=-1)
    if not (
        prediction_ids.shape[0]
        == label_ids.shape[0]
        == indices.shape[0]
        == output_lengths.shape[0]
    ):
        raise RuntimeError("prediction bundle row count mismatch")
    expected = set(range(len(reference_rows)))
    observed = [int(value) for value in indices]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError(
            "distributed evaluation ID coverage drift: "
            f"rows={len(observed)} unique={len(set(observed))} expected={len(expected)}"
        )
    aligned = []
    for source_position, evaluation_index in enumerate(observed):
        row = reference_rows[evaluation_index]
        output_length = int(output_lengths[source_position])
        if output_length <= 0 or output_length > int(prediction_ids.shape[1]):
            raise RuntimeError(f"invalid output length for {row['id']}")
        decoded_reference = codec.decode_labels(
            [int(value) for value in label_ids[source_position]]
        )
        if decoded_reference != str(row["target_ctc"]):
            raise RuntimeError(f"label/ID alignment drift at {row['id']}")
        aligned.append(
            {
                "evaluation_index": evaluation_index,
                "id": str(row["id"]),
                "language": str(row["language"]),
                "model_language": str(row["language"]),
                "stratum": str(row["stratum"]),
                "speaker_key": str(row["speaker_key"]),
                "duration_s": float(row["duration_s"]),
                "output_frames": output_length,
                "original_split": str(row["original_split"]),
                "is_phase2_test_speaker": bool(row["is_phase2_test_speaker"]),
                "is_phase2_test_prompt": bool(row["is_phase2_test_prompt"]),
                "target_weight": float(row["target_weight"]),
                "slot_id": str(row["slot_id"]),
                "target_phase2_id": str(row["target_phase2_id"]),
                "target_official_order": int(row["target_official_order"]),
                "reference_raw": str(row["target_raw"]),
                "reference_ctc": str(row["target_ctc"]),
                "hypothesis": codec.decode_ctc(
                    [
                        int(value)
                        for value in prediction_ids[source_position, :output_length]
                    ]
                ),
            }
        )
    return sorted(aligned, key=lambda row: int(row["evaluation_index"]))


def score_aligned(rows: list[dict[str, Any]], scorer) -> dict[str, Any]:
    def score(selected: list[dict[str, Any]]) -> dict[str, Any]:
        hypotheses = [str(row["hypothesis"]) for row in selected]
        raw = scorer.score_texts(
            [str(row["reference_raw"]) for row in selected],
            hypotheses,
        )
        content = scorer.score_texts(
            [str(row["reference_ctc"]) for row in selected],
            hypotheses,
            normalized=True,
        )
        target_weighted_raw = scorer.score_weighted_texts(
            [str(row["reference_raw"]) for row in selected],
            hypotheses,
            [float(row["target_weight"]) for row in selected],
        )
        target_weighted_content = scorer.score_weighted_texts(
            [str(row["reference_ctc"]) for row in selected],
            hypotheses,
            [float(row["target_weight"]) for row in selected],
            normalized=True,
        )
        return {
            "rows": len(selected),
            "raw": {key: float(value) for key, value in raw.items()},
            "content": {key: float(value) for key, value in content.items()},
            "target_weighted_raw": {
                key: float(value) for key, value in target_weighted_raw.items()
            },
            "target_weighted_content": {
                key: float(value) for key, value in target_weighted_content.items()
            },
            "blank_rows": sum(not value.strip() for value in hypotheses),
            "blank_fraction": sum(not value.strip() for value in hypotheses)
            / len(hypotheses),
        }

    result = score(rows)
    result["strata"] = {
        stratum: score([row for row in rows if row["stratum"] == stratum])
        for stratum in ("warm", "cold")
        if any(row["stratum"] == stratum for row in rows)
    }
    return result


def make_metric_function(
    *,
    codec: CharacterCodec,
    reference_rows: list[dict[str, Any]],
    scorer,
):
    def compute_metrics(prediction) -> dict[str, float]:
        aligned = align_prediction_bundle(
            predictions=prediction.predictions,
            label_bundle=prediction.label_ids,
            codec=codec,
            reference_rows=reference_rows,
        )
        scored = score_aligned(aligned, scorer)
        return {
            "wer": float(scored["content"]["wer"]),
            "cer": float(scored["content"]["cer"]),
            "weighted_error": float(scored["content"]["weighted_error"]),
            "q": float(scored["content"]["q"]),
            "raw_wer": float(scored["raw"]["wer"]),
            "raw_cer": float(scored["raw"]["cer"]),
            "raw_weighted_error": float(scored["raw"]["weighted_error"]),
            "raw_q": float(scored["raw"]["q"]),
            "target_weighted_raw_wer": float(scored["target_weighted_raw"]["wer"]),
            "target_weighted_raw_cer": float(scored["target_weighted_raw"]["cer"]),
            "target_weighted_raw_weighted_error": float(
                scored["target_weighted_raw"]["weighted_error"]
            ),
            "target_weighted_raw_q": float(scored["target_weighted_raw"]["q"]),
            "blank_fraction": float(scored["blank_fraction"]),
        }

    return compute_metrics
