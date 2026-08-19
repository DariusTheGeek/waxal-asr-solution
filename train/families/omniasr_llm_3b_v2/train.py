#!/usr/bin/env python3
"""Relocatable, fenced OmniASR LLM-3B-v2 training entry point."""

from __future__ import annotations

import os
from pathlib import Path

from checkpoint_contract import coordinate_resume_preparation
from early_stopping import (
    load_completed_validation_epochs,
    prepare_validation_logs_for_resume,
    validate_export_validation_state,
)
from fsdp_compat import install_fsdp2_gradient_sync_compat
from activation_checkpointing import install_wav2vec2_llama_layerwise_ac
from runtime_assets import render_asset_cards, resolve_repo_root
from runtime_config import runtime_geometry_from_environment


def prepare_validation_prefix_for_resume(
    *,
    output: Path,
    manifest_dir: Path,
    steps: list[int],
    world_size: int,
    export_mode: bool,
    export_profile: str | None = None,
    updates_per_epoch: int = 501,
) -> dict[str, object]:
    """Reconcile training logs, or preserve terminal evidence for export."""

    if not steps:
        return {"status": "PASS", "mode": "fresh", "completed_epochs": 0}
    latest = max(steps)
    if export_mode:
        if export_profile not in {"smoke", "production"}:
            raise RuntimeError("export mode requires an exact smoke/production profile")
        validation_evidence = validate_export_validation_state(
            trainer_output_dir=output,
            rank_map_path=manifest_dir / "dev.rank_map.world8.csv",
            checkpoint_step=latest,
            world_size=world_size,
            profile=export_profile,
            updates_per_epoch=updates_per_epoch,
        )
        return {
            "status": "PASS",
            "mode": "export_validates_and_preserves_training_terminal_evidence",
            "checkpoint_step": latest,
            "validation_evidence": validation_evidence,
        }
    completed = load_completed_validation_epochs(
        output, updates_per_epoch=updates_per_epoch
    )
    checkpoint_epoch = latest // updates_per_epoch
    if latest % updates_per_epoch:
        allowed = {checkpoint_epoch}
        boundary_mode = "graceful_non_boundary"
    else:
        allowed = {checkpoint_epoch - 1, checkpoint_epoch}
        boundary_mode = "epoch_boundary"
    if completed not in allowed:
        raise RuntimeError(
            "checkpoint/validation prefix drift before stream construction: "
            f"step={latest} completed={completed} allowed={sorted(allowed)}"
        )
    recovery = prepare_validation_logs_for_resume(
        trainer_output_dir=output,
        rank_map_path=manifest_dir / "dev.rank_map.world8.csv",
        completed_epochs=completed,
        world_size=world_size,
        checkpoint_step=latest,
    )
    return {
        "status": "PASS",
        "mode": "reconciled_before_transcript_stream_open",
        "checkpoint_mode": boundary_mode,
        "checkpoint_step": latest,
        "checkpoint_epoch_floor": checkpoint_epoch,
        "completed_epochs": completed,
        "recovery": recovery,
    }


def prepare_runtime() -> None:
    install_wav2vec2_llama_layerwise_ac()
    install_fsdp2_gradient_sync_compat()
    root = resolve_repo_root()
    geometry = runtime_geometry_from_environment(root)
    runtime_raw = os.environ.get("WAXAL3_RUNTIME_DIR")
    output_raw = os.environ.get("WAXAL3_TRAINER_OUTPUT_DIR")
    if not runtime_raw or not output_raw:
        raise RuntimeError(
            "WAXAL3_RUNTIME_DIR and WAXAL3_TRAINER_OUTPUT_DIR are required"
        )
    runtime = Path(runtime_raw).expanduser().resolve()
    output = Path(output_raw).expanduser().resolve()
    render_asset_cards(
        Path(__file__).resolve().parent / "cards/waxal3.yaml.template",
        runtime / "asset_cards",
    )
    manifest_dir = geometry.manifest_dir

    def prepare_validation_prefix(steps: list[int]) -> dict[str, object]:
        return prepare_validation_prefix_for_resume(
            output=output,
            manifest_dir=manifest_dir,
            steps=steps,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            export_mode=bool(os.environ.get("WAXAL3_EXPORT_FULL_STATE_DIR")),
            export_profile=os.environ.get("WAXAL3_PROFILE"),
            updates_per_epoch=geometry.updates_per_epoch,
        )

    coordinate_resume_preparation(output, prepare_validation_prefix)
    os.chdir(root)


def main() -> None:
    prepare_runtime()
    from fairseq2.recipe.cli import train_main
    from recipe import WaxalWav2Vec2AsrRecipe

    train_main(WaxalWav2Vec2AsrRecipe())


if __name__ == "__main__":
    main()
