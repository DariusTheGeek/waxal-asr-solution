from __future__ import annotations

import hashlib
import json
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from early_stopping import (
    TargetWeightedEarlyStopper,
    assert_validation_log_counts,
    load_completed_validation_epochs,
    prepare_validation_logs_for_resume,
)
from trainer_runtime import attach_blocking_checkpoints_and_terminal_scoring
from train import prepare_validation_prefix_for_resume


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partial_next_epoch_is_trimmed_before_append_and_replayed_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    evidence = output / "early_stopping"
    transcripts = output / "transcriptions"
    evidence.mkdir(parents=True)
    transcripts.mkdir()
    history = []
    for epoch in (1, 2):
        step = epoch * 501
        history.append({"epoch": epoch, "step": step})
        (evidence / f"step_{step:08d}.json").write_text("{}\n", encoding="utf-8")
    (evidence / "STATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTINUE",
                "promotion_authority": "target_slot_weighted_raw_q",
                "history": history,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rank_map = tmp_path / "dev.rank_map.world2.csv"
    pl.DataFrame(
        {
            "row_key": ["a", "b", "c", "d"],
            "manifest_index": [0, 1, 2, 3],
            "rank": [0, 0, 1, 1],
            "rank_line_index": [0, 1, 0, 1],
        }
    ).write_csv(rank_map)
    committed_hashes: dict[str, str] = {}
    for rank in range(2):
        for kind in ("hyp", "ref"):
            path = transcripts / f"rank_{rank}.{kind}.txt"
            committed = [f"{rank}-{kind}-e{epoch}-{row}" for epoch in (1, 2) for row in (0, 1)]
            path.write_text(
                "\n".join([*committed, f"{rank}-{kind}-partial-e3"]) + "\n",
                encoding="utf-8",
            )
            expected_path = tmp_path / f"expected-{rank}-{kind}.txt"
            expected_path.write_text("\n".join(committed) + "\n", encoding="utf-8")
            committed_hashes[path.name] = _sha256(expected_path)

    assert load_completed_validation_epochs(output) == 2
    recovery = prepare_validation_logs_for_resume(
        trainer_output_dir=output,
        rank_map_path=rank_map,
        completed_epochs=2,
        world_size=2,
        checkpoint_step=1503,
    )
    assert len(recovery["actions"]) == 4
    assert_validation_log_counts(
        trainer_output_dir=output,
        rank_map_path=rank_map,
        completed_epochs=2,
        world_size=2,
    )
    for rank in range(2):
        for kind in ("hyp", "ref"):
            path = transcripts / f"rank_{rank}.{kind}.txt"
            assert _sha256(path) == committed_hashes[path.name]
            # This models the vendored calculator's FileMode.APPEND handle,
            # which is created only after the atomic trim above.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{rank}-{kind}-e3-0\n{rank}-{kind}-e3-1\n")
    assert_validation_log_counts(
        trainer_output_dir=output,
        rank_map_path=rank_map,
        completed_epochs=3,
        world_size=2,
    )


class _Stopper:
    def __init__(self) -> None:
        self.completed_epochs = 11


class _TrainerState(Enum):
    END_OF_TRAINING = auto()
    POST_STEP = auto()
    DATA_LOAD = auto()
    STOP_REQUESTED = auto()


class _FakeTrainer:
    def __init__(self, stopper: _Stopper) -> None:
        self._step_nr = 6012
        self._gangs = SimpleNamespace(root=SimpleNamespace(rank=0))
        self._state = _TrainerState.END_OF_TRAINING
        self._stop_requested = False
        self.stopper = stopper
        self.blocking_arguments: list[bool] = []
        self.validation_calls = 0
        self.target_score_calls = 0
        self.post_step_calls = 0

    def _maybe_save_checkpoint(self, blocking: bool) -> None:
        self.blocking_arguments.append(blocking)

    def _validate(self) -> float:
        self.validation_calls += 1
        return -42.0

    def _maybe_request_early_stop(self, score: float) -> bool:
        assert score == -42.0
        self.target_score_calls += 1
        self.stopper.completed_epochs += 1
        return False

    def _should_save_checkpoint(self) -> bool:
        return True

    def _run_post_step(self) -> _TrainerState:
        self.post_step_calls += 1
        return _TrainerState.DATA_LOAD

    def _stop(self) -> str:
        self._maybe_save_checkpoint(blocking=False)
        self._validate()
        return "STOPPED"


def test_maximum_epoch_is_scored_once_and_checkpoint_is_blocking() -> None:
    stopper = _Stopper()
    trainer = _FakeTrainer(stopper)
    attach_blocking_checkpoints_and_terminal_scoring(
        trainer,
        stopper=stopper,
        validation_interval_steps=501,
    )
    assert trainer._stop() == "STOPPED"
    assert trainer.blocking_arguments == [True]
    assert trainer.validation_calls == 1
    assert trainer.target_score_calls == 1
    assert stopper.completed_epochs == 12

    # A repeated terminal transition cannot append or score epoch 12 again.
    assert trainer._stop() == "STOPPED"
    assert trainer.validation_calls == 1
    assert trainer.target_score_calls == 1


def test_stop_request_preempts_configured_max_step_post_logic() -> None:
    trainer = _FakeTrainer(_Stopper())
    trainer._state = _TrainerState.POST_STEP
    attach_blocking_checkpoints_and_terminal_scoring(
        trainer,
        stopper=None,
        validation_interval_steps=501,
    )
    trainer._stop_requested = True
    assert trainer._run_post_step() is _TrainerState.STOP_REQUESTED
    assert trainer.post_step_calls == 0


def test_graceful_non_boundary_checkpoint_preserves_completed_epoch_prefix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    early = output / "early_stopping"
    transcripts = output / "transcriptions"
    manifests = tmp_path / "manifests"
    early.mkdir(parents=True)
    transcripts.mkdir()
    manifests.mkdir()
    (early / "STATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTINUE",
                "promotion_authority": "target_slot_weighted_raw_q",
                "history": [{"epoch": 1, "step": 501}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (early / "step_00000501.json").write_text("{}\n", encoding="utf-8")
    pl.DataFrame(
        {
            "row_key": ["a", "b"],
            "manifest_index": [0, 1],
            "rank": [0, 1],
            "rank_line_index": [0, 0],
        }
    ).write_csv(manifests / "dev.rank_map.world8.csv")
    for rank in range(2):
        for kind in ("hyp", "ref"):
            (transcripts / f"rank_{rank}.{kind}.txt").write_text(
                f"committed-{rank}-{kind}\nuncommitted-{rank}-{kind}\n",
                encoding="utf-8",
            )
    record = prepare_validation_prefix_for_resume(
        output=output,
        manifest_dir=manifests,
        steps=[501, 700],
        world_size=2,
        export_mode=False,
    )
    assert record["checkpoint_mode"] == "graceful_non_boundary"
    assert record["completed_epochs"] == 1
    for rank in range(2):
        for kind in ("hyp", "ref"):
            assert (
                transcripts / f"rank_{rank}.{kind}.txt"
            ).read_text(encoding="utf-8") == f"committed-{rank}-{kind}\n"


def test_unscored_epoch_boundary_accepts_previous_completed_prefix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    early = output / "early_stopping"
    transcripts = output / "transcriptions"
    manifests = tmp_path / "manifests"
    early.mkdir(parents=True)
    transcripts.mkdir()
    manifests.mkdir()
    (early / "STATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTINUE",
                "promotion_authority": "target_slot_weighted_raw_q",
                "history": [{"epoch": 1, "step": 501}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (early / "step_00000501.json").write_text("{}\n", encoding="utf-8")
    pl.DataFrame(
        {
            "row_key": ["a", "b"],
            "manifest_index": [0, 1],
            "rank": [0, 1],
            "rank_line_index": [0, 0],
        }
    ).write_csv(manifests / "dev.rank_map.world8.csv")
    for rank in range(2):
        for kind in ("hyp", "ref"):
            (transcripts / f"rank_{rank}.{kind}.txt").write_text(
                f"epoch-one-{rank}-{kind}\npartial-epoch-two-{rank}-{kind}\n",
                encoding="utf-8",
            )
    record = prepare_validation_prefix_for_resume(
        output=output,
        manifest_dir=manifests,
        steps=[501, 1002],
        world_size=2,
        export_mode=False,
    )
    assert record["checkpoint_mode"] == "epoch_boundary"
    assert record["completed_epochs"] == 1
    for rank in range(2):
        for kind in ("hyp", "ref"):
            assert (
                transcripts / f"rank_{rank}.{kind}.txt"
            ).read_text(encoding="utf-8") == f"epoch-one-{rank}-{kind}\n"


def test_shona_509_resume_boundaries_are_not_treated_as_lingala_501(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    early = output / "early_stopping"
    transcripts = output / "transcriptions"
    manifests = tmp_path / "manifests"
    early.mkdir(parents=True)
    transcripts.mkdir()
    manifests.mkdir()
    history = [{"epoch": 1, "step": 509}]
    (early / "STATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTINUE",
                "promotion_authority": "target_slot_weighted_raw_q",
                "history": history,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (early / "step_00000509.json").write_text("{}\n", encoding="utf-8")
    pl.DataFrame(
        {
            "row_key": ["a"],
            "manifest_index": [0],
            "rank": [0],
            "rank_line_index": [0],
        }
    ).write_csv(manifests / "dev.rank_map.world8.csv")
    for kind in ("hyp", "ref"):
        (transcripts / f"rank_0.{kind}.txt").write_text(
            f"epoch-one-{kind}\npartial-two-{kind}\n", encoding="utf-8"
        )
    assert load_completed_validation_epochs(
        output, updates_per_epoch=509
    ) == 1
    with pytest.raises(RuntimeError, match="history/evidence drift"):
        load_completed_validation_epochs(output, updates_per_epoch=501)
    record = prepare_validation_prefix_for_resume(
        output=output,
        manifest_dir=manifests,
        steps=[509, 1_018],
        world_size=1,
        export_mode=False,
        updates_per_epoch=509,
    )
    assert record["checkpoint_mode"] == "epoch_boundary"
    assert record["checkpoint_epoch_floor"] == 2
    assert record["completed_epochs"] == 1


def test_smoke_export_requires_complete_validation_and_preserves_logs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    transcripts = output / "transcriptions"
    manifests = tmp_path / "manifests"
    transcripts.mkdir(parents=True)
    manifests.mkdir()
    pl.DataFrame(
        {
            "row_key": ["a"],
            "manifest_index": [0],
            "rank": [0],
            "rank_line_index": [0],
        }
    ).write_csv(manifests / "dev.rank_map.world8.csv")
    sentinel = transcripts / "rank_0.hyp.txt"
    sentinel.write_text("smoke-hypothesis\n", encoding="utf-8")
    (transcripts / "rank_0.ref.txt").write_text(
        "smoke-reference\n", encoding="utf-8"
    )
    record = prepare_validation_prefix_for_resume(
        output=output,
        manifest_dir=manifests,
        steps=[3],
        world_size=1,
        export_mode=True,
        export_profile="smoke",
    )
    assert record["mode"] == "export_validates_and_preserves_training_terminal_evidence"
    assert record["validation_evidence"]["completed_validation_events"] == 1
    assert sentinel.read_text(encoding="utf-8") == "smoke-hypothesis\n"

    (transcripts / "rank_0.hyp.txt").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="validation log count drift"):
        prepare_validation_prefix_for_resume(
            output=output,
            manifest_dir=manifests,
            steps=[3],
            world_size=1,
            export_mode=True,
            export_profile="smoke",
        )


def test_actual_target_stopper_terminal_is_exportable_but_not_train_resumable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trainer"
    transcripts = output / "transcriptions"
    transcripts.mkdir(parents=True)
    manifest = tmp_path / "dev.rows.parquet"
    rank_map = tmp_path / "dev.rank_map.world8.csv"
    pl.DataFrame(
        {
            "manifest_index": [0, 1],
            "row_key": ["a", "b"],
            "transcription_nfc": ["hello", "world"],
            "training_target": ["hello", "world"],
            "target_weight": [1.0, 2.0],
        }
    ).write_parquet(manifest)
    pl.DataFrame(
        {
            "row_key": ["a", "b"],
            "manifest_index": [0, 1],
            "rank": [0, 0],
            "rank_line_index": [0, 1],
        }
    ).write_csv(rank_map)
    reference = transcripts / "rank_0.ref.txt"
    hypothesis = transcripts / "rank_0.hyp.txt"
    reference.write_text("", encoding="utf-8")
    hypothesis.write_text("", encoding="utf-8")
    stopper = TargetWeightedEarlyStopper(
        trainer_output_dir=output,
        manifest_path=manifest,
        rank_map_path=rank_map,
        world_size=1,
        validation_interval_steps=501,
        warmup_epochs=4,
        patience=3,
        max_epochs=12,
    )
    decisions = []
    for epoch in range(1, 8):
        with reference.open("a", encoding="utf-8") as handle:
            handle.write("hello\nworld\n")
        with hypothesis.open("a", encoding="utf-8") as handle:
            handle.write("hello\nwrong\n")
        decisions.append(stopper.should_stop(epoch * 501, -50.0))
    assert decisions == [False, False, False, False, False, False, True]
    assert (output / "early_stopping/EARLY_STOP.json").is_file()

    export_record = prepare_validation_prefix_for_resume(
        output=output,
        manifest_dir=tmp_path,
        steps=[3507],
        world_size=1,
        export_mode=True,
        export_profile="production",
    )
    assert (
        export_record["mode"]
        == "export_validates_and_preserves_training_terminal_evidence"
    )
    assert export_record["validation_evidence"]["target_stopper"] == "EARLY_STOP_REQUESTED"
    with pytest.raises(RuntimeError, match="terminal early-stop evidence"):
        prepare_validation_prefix_for_resume(
            output=output,
            manifest_dir=tmp_path / "unused",
            steps=[3507],
            world_size=1,
            export_mode=False,
        )
    terminal_path = output / "early_stopping/EARLY_STOP.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["terminal_step"] = 999
    terminal_path.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="EARLY_STOP/STATE coherence"):
        prepare_validation_prefix_for_resume(
            output=output,
            manifest_dir=tmp_path,
            steps=[3507],
            world_size=1,
            export_mode=True,
            export_profile="production",
        )
