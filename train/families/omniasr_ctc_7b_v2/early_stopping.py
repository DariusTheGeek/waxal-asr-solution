"""Target-aligned, fail-closed early stopping for WAXAL3 CTC fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from fairseq2.early_stopper import EarlyStopper
import polars as pl

from canonical_scoring import (
    CANONICAL_SCORER_SOURCE_SHA256,
    score_texts,
    score_weighted_texts,
)


SCHEMA_VERSION = 1
MODEL_CARD_SUFFIX = "_target_es_parent"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    payload = json.dumps(
        lines, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_lines_atomic(path: Path, lines: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_completed_validation_epochs(
    trainer_output_dir: Path, *, updates_per_epoch: int = 501
) -> int:
    """Read the durable target-score prefix without constructing stream users."""

    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be positive")
    evidence_dir = Path(trainer_output_dir) / "early_stopping"
    state_path = evidence_dir / "STATE.json"
    if (evidence_dir / "EARLY_STOP.json").exists():
        raise RuntimeError("terminal early-stop evidence forbids resume")
    if not state_path.exists():
        unexpected = sorted(evidence_dir.glob("step_*.json")) if evidence_dir.exists() else []
        if unexpected:
            raise RuntimeError("early-stop step evidence exists without durable STATE.json")
        return 0
    value = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "CONTINUE"
        or value.get("promotion_authority") != "target_slot_weighted_raw_q"
    ):
        raise RuntimeError("early-stop pre-open resume state is invalid")
    history = value.get("history")
    if not isinstance(history, list):
        raise RuntimeError("early-stop pre-open history is not a list")
    for expected_epoch, item in enumerate(history, start=1):
        if (
            not isinstance(item, dict)
            or int(item.get("epoch", -1)) != expected_epoch
            or int(item.get("step", -1)) != expected_epoch * updates_per_epoch
            or not (
                evidence_dir
                / f"step_{expected_epoch * updates_per_epoch:08d}.json"
            ).is_file()
        ):
            raise RuntimeError("early-stop pre-open history/evidence drift")
    return len(history)


def assert_validation_log_counts(
    *,
    trainer_output_dir: Path,
    rank_map_path: Path,
    completed_epochs: int,
    world_size: int,
) -> dict[str, Any]:
    """Assert the exact committed log prefix without replacing open files."""

    mapping = pl.read_csv(rank_map_path)
    if sorted(mapping["rank"].unique().to_list()) != list(range(world_size)):
        raise RuntimeError("validation log assertion rank-map world-size drift")
    counts: list[dict[str, int]] = []
    transcript_dir = Path(trainer_output_dir) / "transcriptions"
    for rank in range(world_size):
        rows = mapping.filter(pl.col("rank") == rank).height
        expected = rows * completed_epochs
        observed: dict[str, int] = {}
        for kind in ("hyp", "ref"):
            path = transcript_dir / f"rank_{rank}.{kind}.txt"
            count = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
            if count != expected:
                raise RuntimeError(
                    f"validation log count drift: {path}: {count} != {expected}"
                )
            observed[kind] = count
        counts.append({"rank": rank, "rows_per_epoch": rows, **observed})
    return {
        "status": "PASS",
        "completed_epochs": completed_epochs,
        "world_size": world_size,
        "counts": counts,
    }


def prepare_validation_logs_for_resume(
    *,
    trainer_output_dir: Path,
    rank_map_path: Path,
    completed_epochs: int,
    world_size: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Preserve and trim only uncommitted validation output before replay."""

    if completed_epochs < 0 or world_size <= 0:
        raise ValueError("invalid validation-resume counters")
    mapping = pl.read_csv(rank_map_path)
    if sorted(mapping["rank"].unique().to_list()) != list(range(world_size)):
        raise RuntimeError("validation resume rank-map world-size drift")
    transcript_dir = Path(trainer_output_dir) / "transcriptions"
    recovery_dir = (
        Path(trainer_output_dir)
        / "early_stopping"
        / "recovery"
        / f"checkpoint_step_{checkpoint_step:08d}"
    )
    actions: list[dict[str, Any]] = []
    for rank in range(world_size):
        rows = mapping.filter(pl.col("rank") == rank).height
        expected = rows * completed_epochs
        for kind in ("hyp", "ref"):
            path = transcript_dir / f"rank_{rank}.{kind}.txt"
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            if len(lines) < expected:
                raise RuntimeError(
                    f"committed validation log is truncated: {path}: {len(lines)} < {expected}"
                )
            if len(lines) == expected:
                continue
            recovery_dir.mkdir(parents=True, exist_ok=True)
            backup = recovery_dir / f"{path.name}.{uuid.uuid4().hex}.uncommitted"
            shutil.copy2(path, backup)
            _write_lines_atomic(path, lines[:expected])
            actions.append(
                {
                    "path": str(path),
                    "observed_lines": len(lines),
                    "retained_lines": expected,
                    "backup": str(backup),
                    "backup_sha256": _sha256_file(backup),
                }
            )
    return {
        "schema_version": 1,
        "status": "PASS",
        "checkpoint_step": checkpoint_step,
        "completed_epochs": completed_epochs,
        "actions": actions,
    }


@dataclass
class StrictPatiencePolicy:
    """Warm up, then stop after consecutive non-improving validation epochs."""

    warmup_epochs: int = 4
    patience: int = 3
    best_score: float = -math.inf
    best_epoch: int | None = None
    bad_epochs: int = 0
    last_epoch: int = 0

    def update(self, epoch: int, score: float) -> dict[str, Any]:
        if epoch != self.last_epoch + 1:
            raise ValueError(
                f"non-consecutive early-stop epoch: {epoch} after {self.last_epoch}"
            )
        if not math.isfinite(score):
            raise ValueError(f"non-finite promotion score: {score}")

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch

        if epoch <= self.warmup_epochs:
            self.bad_epochs = 0
            phase = "warmup"
        else:
            phase = "patience"
            self.bad_epochs = 0 if improved else self.bad_epochs + 1

        should_stop = (
            epoch > self.warmup_epochs and self.bad_epochs >= self.patience
        )
        self.last_epoch = epoch
        return {
            "epoch": epoch,
            "phase": phase,
            "strict_improvement": improved,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "consecutive_non_improving_epochs": self.bad_epochs,
            "should_stop": should_stop,
        }


def validate_export_validation_state(
    *,
    trainer_output_dir: Path,
    rank_map_path: Path,
    checkpoint_step: int,
    world_size: int,
    profile: str,
    updates_per_epoch: int = 501,
) -> dict[str, Any]:
    """Semantically validate the exact validation state selected for export."""

    if profile not in {"smoke", "production"}:
        raise RuntimeError(f"unsupported export validation profile: {profile}")
    if checkpoint_step <= 0 or world_size <= 0 or updates_per_epoch <= 0:
        raise ValueError(
            "export checkpoint step, world size, and updates_per_epoch must be positive"
        )
    if profile == "production" and checkpoint_step % updates_per_epoch:
        raise RuntimeError("production export requires an epoch-boundary checkpoint")
    completed_epochs = (
        1 if profile == "smoke" else checkpoint_step // updates_per_epoch
    )
    counts = assert_validation_log_counts(
        trainer_output_dir=trainer_output_dir,
        rank_map_path=rank_map_path,
        completed_epochs=completed_epochs,
        world_size=world_size,
    )
    transcript_hashes: list[dict[str, object]] = []
    transcript_dir = Path(trainer_output_dir) / "transcriptions"
    for rank in range(world_size):
        for kind in ("hyp", "ref"):
            path = transcript_dir / f"rank_{rank}.{kind}.txt"
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"unsafe export transcript evidence: {path}")
            transcript_hashes.append(
                {
                    "rank": rank,
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )

    evidence_dir = Path(trainer_output_dir) / "early_stopping"
    state_path = evidence_dir / "STATE.json"
    terminal_path = evidence_dir / "EARLY_STOP.json"
    step_paths = sorted(evidence_dir.glob("step_*.json")) if evidence_dir.exists() else []
    if profile == "smoke":
        if state_path.exists() or terminal_path.exists() or step_paths:
            raise RuntimeError("smoke export contains unexpected target-stopper evidence")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "profile": profile,
            "checkpoint_step": checkpoint_step,
            "completed_validation_events": completed_epochs,
            "target_stopper": "DISABLED_BY_FROZEN_SMOKE_PROFILE",
            "validation_log_counts": counts,
            "transcript_hashes": transcript_hashes,
            "target_evidence_sha256": None,
            "state_sha256": None,
            "terminal_sha256": None,
        }

    if not state_path.is_file() or state_path.is_symlink():
        raise RuntimeError("production export lacks safe target-stopper STATE.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("status") not in {"CONTINUE", "EARLY_STOP_REQUESTED"}
        or state.get("promotion_authority") != "target_slot_weighted_raw_q"
        or not isinstance(state.get("inputs"), dict)
        or not isinstance(state.get("policy"), dict)
        or not isinstance(state.get("history"), list)
    ):
        raise RuntimeError("production export target-stopper STATE semantics drift")
    history = state["history"]
    if len(history) != completed_epochs:
        raise RuntimeError(
            "production export target history/checkpoint mismatch: "
            f"history={len(history)} checkpoint_epoch={completed_epochs}"
        )
    policy_record = state["policy"]
    try:
        policy = StrictPatiencePolicy(
            warmup_epochs=int(policy_record["warmup_epochs"]),
            patience=int(policy_record["patience"]),
        )
        if (
            float(policy_record["strict_improvement_min_delta"]) != 0.0
            or int(policy_record["max_epochs"]) != 12
            or policy.warmup_epochs != 4
            or policy.patience != 3
        ):
            raise RuntimeError("production export target policy drift")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("production export target policy is malformed") from error

    expected_paths: list[Path] = []
    evidence_hashes: list[dict[str, object]] = []
    last_decision: dict[str, Any] | None = None
    for expected_epoch, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            raise RuntimeError("production export target history item is malformed")
        step = expected_epoch * updates_per_epoch
        try:
            score = float(item["promotion_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("production export promotion score is malformed") from error
        decision = policy.update(expected_epoch, score)
        expected_item = {
            "step": step,
            "epoch": expected_epoch,
            "promotion_score": score,
            **decision,
        }
        if item != expected_item:
            raise RuntimeError("production export target history replay drift")
        evidence_path = evidence_dir / f"step_{step:08d}.json"
        expected_paths.append(evidence_path)
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise RuntimeError(f"production export target evidence is unsafe: {evidence_path}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_policy = evidence.get("policy") if isinstance(evidence, dict) else None
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema_version") != SCHEMA_VERSION
            or evidence.get("status") != "PASS"
            or int(evidence.get("step", -1)) != step
            or int(evidence.get("epoch", -1)) != expected_epoch
            or evidence.get("promotion_authority") != "target_slot_weighted_raw_q"
            or float(evidence.get("promotion_score", math.nan)) != score
            or evidence.get("inputs") != state["inputs"]
            or not isinstance(evidence_policy, dict)
            or any(evidence_policy.get(key) != value for key, value in decision.items())
        ):
            raise RuntimeError("production export selected target evidence semantics drift")
        evidence_hashes.append(
            {
                "epoch": expected_epoch,
                "step": step,
                "sha256": _sha256_file(evidence_path),
            }
        )
        last_decision = decision
    if step_paths != expected_paths or not evidence_hashes:
        raise RuntimeError("production export target evidence file set drift")
    assert last_decision is not None
    terminal_expected = bool(last_decision["should_stop"])
    if (state["status"] == "EARLY_STOP_REQUESTED") != terminal_expected:
        raise RuntimeError("production export terminal state/policy disagreement")
    terminal_sha256: str | None = None
    if terminal_expected:
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise RuntimeError("production export terminal record is absent or unsafe")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if (
            not isinstance(terminal, dict)
            or any(terminal.get(key) != value for key, value in state.items())
            or int(terminal.get("terminal_step", -1)) != checkpoint_step
            or int(terminal.get("terminal_epoch", -1)) != completed_epochs
            or terminal.get("best_epoch") != last_decision["best_epoch"]
            or float(terminal.get("best_score", math.nan))
            != float(last_decision["best_score"])
        ):
            raise RuntimeError("production export EARLY_STOP/STATE coherence drift")
        terminal_sha256 = _sha256_file(terminal_path)
    elif terminal_path.exists():
        raise RuntimeError("production export has unexpected terminal record")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": profile,
        "checkpoint_step": checkpoint_step,
        "completed_validation_events": completed_epochs,
        "target_stopper": state["status"],
        "validation_log_counts": counts,
        "transcript_hashes": transcript_hashes,
        "target_evidence_sha256": evidence_hashes[-1]["sha256"],
        "target_evidence_history": evidence_hashes,
        "state_sha256": _sha256_file(state_path),
        "terminal_sha256": terminal_sha256,
    }


def target_early_stopping_enabled(config: object) -> bool:
    """Recognize only the frozen 12-pass CPT downstream production shape."""

    try:
        model_name = str(config.model.name)  # type: ignore[attr-defined]
        num_steps = int(config.regime.num_steps)  # type: ignore[attr-defined]
        validate_every = int(  # type: ignore[attr-defined]
            config.regime.validate_every_n_steps
        )
        checkpoint_every = int(  # type: ignore[attr-defined]
            config.regime.checkpoint_every_n_steps
        )
        valid_split = str(config.dataset.valid_split)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        model_name.endswith(MODEL_CARD_SUFFIX)
        and validate_every >= 100
        and num_steps == 12 * validate_every
        and checkpoint_every == validate_every
        and bool(valid_split)
        and valid_split != "None"
        and "," not in valid_split
    )


def load_validation_epoch(
    *,
    trainer_output_dir: Path,
    manifest_path: Path,
    rank_map_path: Path,
    epoch: int,
    world_size: int,
    require_exact_event_count: bool = True,
) -> dict[str, Any]:
    """Reconstruct one validation event from fairseq2's rank-local text logs."""

    if epoch <= 0 or world_size <= 0:
        raise ValueError("epoch and world_size must be positive")
    rows = pl.read_parquet(manifest_path).sort("manifest_index")
    rank_map = pl.read_csv(rank_map_path).sort(["rank", "rank_line_index"])
    required_rows = {
        "manifest_index",
        "row_key",
        "transcription_nfc",
        "training_target",
        "target_weight",
    }
    required_map = {"row_key", "manifest_index", "rank", "rank_line_index"}
    if missing := required_rows - set(rows.columns):
        raise ValueError(f"validation manifest missing columns: {sorted(missing)}")
    if missing := required_map - set(rank_map.columns):
        raise ValueError(f"validation rank map missing columns: {sorted(missing)}")
    if rows.height == 0 or rank_map.height != rows.height:
        raise ValueError("validation manifest/rank-map row count mismatch")
    if rows["row_key"].n_unique() != rows.height:
        raise ValueError("validation row keys are not unique")
    if rank_map["row_key"].n_unique() != rank_map.height:
        raise ValueError("validation rank-map row keys are not unique")
    if set(rows["row_key"].to_list()) != set(rank_map["row_key"].to_list()):
        raise ValueError("validation manifest/rank-map membership mismatch")
    if sorted(rank_map["rank"].unique().to_list()) != list(range(world_size)):
        raise ValueError("validation rank map does not cover the exact world size")

    index_check = rows.select("row_key", "manifest_index").join(
        rank_map.select(
            "row_key", pl.col("manifest_index").alias("mapped_manifest_index")
        ),
        on="row_key",
        how="left",
        validate="1:1",
    )
    if not (
        index_check["manifest_index"] == index_check["mapped_manifest_index"]
    ).all():
        raise ValueError("validation manifest indices disagree with rank map")

    transcript_dir = trainer_output_dir / "transcriptions"
    predictions: list[dict[str, Any]] = []
    rank_evidence: list[dict[str, Any]] = []
    for rank in range(world_size):
        mapping = rank_map.filter(pl.col("rank") == rank).sort("rank_line_index")
        if mapping["rank_line_index"].to_list() != list(range(mapping.height)):
            raise ValueError(f"rank {rank} line indices are not contiguous")
        mapped = mapping.join(
            rows.select("row_key", "training_target"),
            on="row_key",
            how="left",
            validate="1:1",
        ).sort("rank_line_index")
        hyp_path = transcript_dir / f"rank_{rank}.hyp.txt"
        ref_path = transcript_dir / f"rank_{rank}.ref.txt"
        hypotheses = hyp_path.read_text(encoding="utf-8").splitlines()
        emitted_references = ref_path.read_text(encoding="utf-8").splitlines()
        event_rows = mapping.height
        minimum_lines = event_rows * epoch
        observed_counts = (len(hypotheses), len(emitted_references))
        if require_exact_event_count:
            counts_valid = observed_counts == (minimum_lines, minimum_lines)
        else:
            counts_valid = all(count >= minimum_lines for count in observed_counts)
        if not counts_valid:
            raise ValueError(
                f"rank {rank} transcript count mismatch at epoch {epoch}: "
                f"hyp={observed_counts[0]} ref={observed_counts[1]} "
                f"expected={'exactly ' if require_exact_event_count else 'at least '}{minimum_lines}"
            )
        start = (epoch - 1) * event_rows
        end = epoch * event_rows
        epoch_hypotheses = hypotheses[start:end]
        epoch_emitted_references = emitted_references[start:end]
        expected_training_targets = mapped["training_target"].fill_null("").to_list()
        if epoch_emitted_references != expected_training_targets:
            mismatch = next(
                index
                for index, (observed, expected) in enumerate(
                    zip(
                        epoch_emitted_references,
                        expected_training_targets,
                        strict=True,
                    )
                )
                if observed != expected
            )
            raise ValueError(
                f"rank {rank} emitted reference drift at epoch {epoch}, "
                f"rank line {mismatch}"
            )
        predictions.extend(
            {
                "row_key": row_key,
                "rank": rank,
                "rank_line_index": line_index,
                "hypothesis": hypothesis,
            }
            for row_key, line_index, hypothesis in zip(
                mapping["row_key"].to_list(),
                mapping["rank_line_index"].to_list(),
                epoch_hypotheses,
                strict=True,
            )
        )
        rank_evidence.append(
            {
                "rank": rank,
                "rows": event_rows,
                "cumulative_lines": minimum_lines,
                "hypothesis_slice_sha256": _sha256_lines(epoch_hypotheses),
                "emitted_reference_slice_sha256": _sha256_lines(
                    epoch_emitted_references
                ),
            }
        )

    prediction_frame = pl.DataFrame(predictions)
    if prediction_frame.height != rows.height:
        raise RuntimeError("reconstructed validation prediction count drift")
    joined = rows.join(
        prediction_frame, on="row_key", how="left", validate="1:1"
    ).sort("manifest_index")
    if joined["hypothesis"].null_count():
        raise RuntimeError("reconstructed validation hypotheses are incomplete")
    canonical_references = joined["transcription_nfc"].fill_null("").to_list()
    canonical_hypotheses = joined["hypothesis"].fill_null("").to_list()
    weights = joined["target_weight"].cast(pl.Float64).to_list()
    raw = score_texts(canonical_references, canonical_hypotheses)
    weighted = score_weighted_texts(
        canonical_references, canonical_hypotheses, weights
    )
    return {
        "rows": rows.height,
        "blank_rows": sum(not value.strip() for value in canonical_hypotheses),
        "raw": raw,
        "target_weighted": weighted,
        "canonical_reference_sha256": _sha256_lines(canonical_references),
        "hypothesis_sha256": _sha256_lines(canonical_hypotheses),
        "rank_evidence": rank_evidence,
    }


class TargetWeightedEarlyStopper(EarlyStopper):
    """Stop on target-slot-weighted raw Q after a four-epoch warmup."""

    def __init__(
        self,
        *,
        trainer_output_dir: Path,
        manifest_path: Path,
        rank_map_path: Path,
        world_size: int,
        validation_interval_steps: int,
        warmup_epochs: int = 4,
        patience: int = 3,
        max_epochs: int = 12,
    ) -> None:
        if validation_interval_steps <= 0 or max_epochs <= 0:
            raise ValueError("validation interval and max epochs must be positive")
        if warmup_epochs <= 0 or patience <= 0:
            raise ValueError("warmup and patience must be positive")
        if warmup_epochs + patience > max_epochs:
            raise ValueError("early stop can never trigger before the maximum epoch")
        self._trainer_output_dir = Path(trainer_output_dir)
        self._manifest_path = Path(manifest_path)
        self._rank_map_path = Path(rank_map_path)
        self._world_size = int(world_size)
        self._interval = int(validation_interval_steps)
        self._max_epochs = int(max_epochs)
        self._policy = StrictPatiencePolicy(warmup_epochs, patience)
        self._evidence_dir = self._trainer_output_dir / "early_stopping"
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        state_path = self._evidence_dir / "STATE.json"
        scorer_source = Path(inspect.getsourcefile(score_texts) or "")
        if not self._manifest_path.is_file() or not self._rank_map_path.is_file():
            raise FileNotFoundError(
                self._manifest_path
                if not self._manifest_path.is_file()
                else self._rank_map_path
            )
        if not scorer_source.is_file():
            raise FileNotFoundError("canonical scorer source cannot be resolved")
        self._input_hashes = {
            "manifest_sha256": _sha256_file(self._manifest_path),
            "rank_map_sha256": _sha256_file(self._rank_map_path),
            "embedded_scorer_sha256": _sha256_file(scorer_source),
            "canonical_scorer_contract_sha256": CANONICAL_SCORER_SOURCE_SHA256,
        }
        self._history: list[dict[str, Any]] = []
        if (self._evidence_dir / "EARLY_STOP.json").exists():
            raise RuntimeError("a terminal early-stop record forbids accidental resume")
        if state_path.exists():
            self._restore_state(state_path)

    def _restore_state(self, state_path: Path) -> None:
        """Replay the durable policy history before fairseq restores tensors."""

        value = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("status") != "CONTINUE"
            or value.get("promotion_authority")
            != "target_slot_weighted_raw_q"
            or value.get("inputs") != self._input_hashes
        ):
            raise RuntimeError("early-stop resume state identity drift")
        history = value.get("history")
        if not isinstance(history, list):
            raise RuntimeError("early-stop resume history is not a list")
        replayed: list[dict[str, Any]] = []
        for expected_epoch, item in enumerate(history, start=1):
            if not isinstance(item, dict):
                raise RuntimeError("early-stop history item is not an object")
            epoch = int(item.get("epoch", -1))
            step = int(item.get("step", -1))
            score = float(item.get("promotion_score", math.nan))
            if epoch != expected_epoch or step != epoch * self._interval:
                raise RuntimeError("early-stop history epoch/step drift")
            decision = self._policy.update(epoch, score)
            expected = {
                "step": step,
                "epoch": epoch,
                "promotion_score": score,
                **decision,
            }
            if item != expected or decision["should_stop"]:
                raise RuntimeError("early-stop history replay drift")
            evidence = self._evidence_dir / f"step_{step:08d}.json"
            if not evidence.is_file():
                raise RuntimeError(f"early-stop evidence is absent: {evidence}")
            replayed.append(expected)
        self._history = replayed

    @property
    def completed_epochs(self) -> int:
        return self._policy.last_epoch

    def should_stop(self, step_nr: int, score: float) -> bool:
        if step_nr <= 0 or step_nr % self._interval:
            raise ValueError(
                f"early-stop callback step {step_nr} is not a positive multiple "
                f"of {self._interval}"
            )
        if not math.isfinite(float(score)):
            raise ValueError(f"non-finite fairseq validation score: {score}")
        epoch = step_nr // self._interval
        if epoch > self._max_epochs:
            raise ValueError(
                f"early-stop callback epoch {epoch} exceeds maximum {self._max_epochs}"
            )
        validation = load_validation_epoch(
            trainer_output_dir=self._trainer_output_dir,
            manifest_path=self._manifest_path,
            rank_map_path=self._rank_map_path,
            epoch=epoch,
            world_size=self._world_size,
            require_exact_event_count=True,
        )
        promotion_score = float(validation["target_weighted"]["score"])
        decision = self._policy.update(epoch, promotion_score)
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(step_nr),
            "epoch": epoch,
            "fairseq_validation_score_diagnostic": float(score),
            "promotion_authority": "target_slot_weighted_raw_q",
            "promotion_score": promotion_score,
            "validation": validation,
            "policy": {
                "warmup_epochs": self._policy.warmup_epochs,
                "patience": self._policy.patience,
                "strict_improvement_min_delta": 0.0,
                "max_epochs": self._max_epochs,
                **decision,
            },
            "inputs": self._input_hashes,
        }
        _write_json_create_only(
            self._evidence_dir / f"step_{step_nr:08d}.json", record
        )
        self._history.append(
            {
                "step": int(step_nr),
                "epoch": epoch,
                "promotion_score": promotion_score,
                **decision,
            }
        )
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "EARLY_STOP_REQUESTED"
            if decision["should_stop"]
            else "CONTINUE",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "promotion_authority": "target_slot_weighted_raw_q",
            "policy": {
                "warmup_epochs": self._policy.warmup_epochs,
                "patience": self._policy.patience,
                "strict_improvement_min_delta": 0.0,
                "max_epochs": self._max_epochs,
            },
            "inputs": self._input_hashes,
            "history": self._history,
        }
        _write_json_atomic(self._evidence_dir / "STATE.json", state)
        if decision["should_stop"]:
            _write_json_create_only(
                self._evidence_dir / "EARLY_STOP.json",
                {
                    **state,
                    "terminal_step": int(step_nr),
                    "terminal_epoch": epoch,
                    "best_epoch": self._policy.best_epoch,
                    "best_score": self._policy.best_score,
                },
            )
        return bool(decision["should_stop"])
