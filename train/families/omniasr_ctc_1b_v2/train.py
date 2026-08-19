#!/usr/bin/env python3
"""Relocatable, fenced OmniASR CTC-1B-v2 training entry point."""

from __future__ import annotations

import os
from pathlib import Path

from checkpoint_contract import coordinate_resume_preparation
from runtime_assets import render_asset_cards, resolve_repo_root


def prepare_runtime() -> None:
    root = resolve_repo_root()
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
    coordinate_resume_preparation(output)
    os.chdir(root)


def main() -> None:
    prepare_runtime()
    from fairseq2.recipe.cli import train_main
    from recipe import WaxalWav2Vec2AsrRecipe

    train_main(WaxalWav2Vec2AsrRecipe())


if __name__ == "__main__":
    main()
