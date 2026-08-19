"""Fail-closed trainer repairs for the pinned OmniASR CTC-7B-v2 runtime."""

from __future__ import annotations

from types import MethodType
from typing import Any


def attach_blocking_checkpoints_and_terminal_scoring(
    trainer: Any,
    *,
    stopper: Any | None,
    validation_interval_steps: int,
) -> None:
    """Remove the evidence race and score a maximum-step validation once."""

    required = (
        "_maybe_save_checkpoint",
        "_stop",
        "_validate",
        "_maybe_request_early_stop",
        "_run_post_step",
        "_should_save_checkpoint",
        "_step_nr",
        "_state",
        "_stop_requested",
    )
    missing = [name for name in required if not hasattr(trainer, name)]
    if missing:
        raise RuntimeError(f"pinned Trainer integration points disappeared: {missing}")
    if validation_interval_steps <= 0:
        raise ValueError("validation interval must be positive")

    original_maybe_save = trainer._maybe_save_checkpoint

    def maybe_save_checkpoint(self: Any, blocking: bool) -> None:
        del blocking
        # The original method still evaluates the configured save boundary; we
        # only force completion before validation can mutate runtime evidence.
        should_save = bool(self._should_save_checkpoint())
        step = int(self._step_nr)
        if should_save and getattr(
            self, "_waxal3_last_blocking_checkpoint_step", None
        ) == step:
            return
        original_maybe_save(True)
        if should_save:
            self._waxal3_last_blocking_checkpoint_step = step

    trainer._maybe_save_checkpoint = MethodType(maybe_save_checkpoint, trainer)
    original_run_post_step = trainer._run_post_step

    def run_post_step(self: Any):
        if bool(self._stop_requested):
            return type(self._state).STOP_REQUESTED
        state = original_run_post_step()
        if bool(self._stop_requested) and str(state.name) == "DATA_LOAD":
            return type(self._state).STOP_REQUESTED
        return state

    trainer._run_post_step = MethodType(run_post_step, trainer)
    original_stop = trainer._stop

    def stop(self: Any):
        original_validate = self._validate

        def validate_and_score_terminal(_self: Any):
            step = int(self._step_nr)
            is_boundary = step > 0 and step % validation_interval_steps == 0
            if stopper is not None and is_boundary:
                expected_epoch = step // validation_interval_steps
                completed = int(stopper.completed_epochs)
                if completed == expected_epoch:
                    # A repeated terminal transition must not append a duplicate
                    # transcript event or duplicate target-score evidence.
                    return None
                if completed != expected_epoch - 1:
                    raise RuntimeError(
                        "terminal validation/early-stop epoch drift: "
                        f"completed={completed} expected_next={expected_epoch}"
                    )
            score = original_validate()
            if score is not None and is_boundary:
                self._maybe_request_early_stop(score)
                root_rank = int(getattr(getattr(self._gangs, "root", None), "rank", 0))
                if (
                    stopper is not None
                    and root_rank == 0
                    and int(stopper.completed_epochs) != expected_epoch
                ):
                    raise RuntimeError("terminal target score was not committed exactly once")
            return score

        self._validate = MethodType(validate_and_score_terminal, self)
        try:
            return original_stop()
        finally:
            self._validate = original_validate

    trainer._stop = MethodType(stop, trainer)
    trainer._waxal3_blocking_checkpoint_contract = True
    trainer._waxal3_terminal_scoring_contract = True
