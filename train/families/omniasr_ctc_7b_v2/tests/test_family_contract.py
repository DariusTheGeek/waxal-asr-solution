from __future__ import annotations

import importlib
import os
from pathlib import Path

import polars as pl
import yaml

from early_stopping import StrictPatiencePolicy
from runtime_assets import render_asset_cards
from runtime_config import runtime_geometry_from_experiment


ROOT_RAW = os.environ.get("WAXAL3_REPO_ROOT")
if not ROOT_RAW:
    raise RuntimeError("packet/source tests require WAXAL3_REPO_ROOT")
ROOT = Path(ROOT_RAW).resolve()
CODE_ROOT = Path(__file__).resolve().parents[1]


def _frozen_spec_and_profiles() -> tuple[Path, Path]:
    packet = CODE_ROOT.parents[1]
    if (packet / "PACKET.json").is_file():
        return packet / "resolved_experiment.yaml", packet / "profiles"
    explicit = os.environ.get("WAXAL3_TEST_EXPERIMENT")
    if explicit:
        experiment = (ROOT / explicit).resolve()
        experiment.relative_to(ROOT)
        return experiment / "experiment.yaml", experiment / "profiles"
    experiment = (
        ROOT
        / "experiments/omniasr_ctc_7b_v2/mono/lin/supervised_ft"
        / "control_max12_target_es/X0024_20260804T091235637461Z_ctc7b_v2_cv002_fsdp2_control"
    )
    return experiment / "experiment.yaml", experiment / "profiles"


def test_recipe_runtime_import_closure() -> None:
    """The packet must carry the upstream workflow modules used at launch."""

    recipe = importlib.import_module("recipe")
    criterion = importlib.import_module(
        "workflows.recipes.wav2vec2.asr.criterion"
    )
    assert recipe.WaxalWav2Vec2AsrRecipe is not None
    assert criterion.Wav2Vec2AsrCriterion is not None


def test_portable_supervised_view_and_world8_map() -> None:
    specification_path, _ = _frozen_spec_and_profiles()
    geometry = runtime_geometry_from_experiment(specification_path.parent, ROOT)
    manifests = geometry.manifest_dir
    assert (
        pl.read_parquet(manifests / "train.rows.parquet").height
        == geometry.expected_train_rows
    )
    rows = pl.read_parquet(manifests / "dev.rows.parquet")
    mapping = pl.read_csv(manifests / "dev.rank_map.world8.csv")
    assert rows.height == mapping.height == 900
    assert sorted(mapping["rank"].unique().to_list()) == list(range(8))
    for split in ("train", "dev"):
        header = (manifests / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[0]
        assert not Path(header).is_absolute()
        assert (ROOT / header).is_dir()


def test_relocatable_card_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WAXAL3_REPO_ROOT", str(ROOT))
    template = Path(__file__).resolve().parents[1] / "cards/waxal3.yaml.template"
    rendered = render_asset_cards(template, tmp_path)
    text = rendered.read_text(encoding="utf-8")
    assert "@WAXAL3_REPO_ROOT@" not in text
    assert str(ROOT / "models/omniasr-ctc-7b-v2") in text
    assert "waxal3_lin_cv002_portable" in text
    assert "waxal3_sna_cv002_portable" in text


def test_target_patience_earliest_stop_is_epoch_seven() -> None:
    policy = StrictPatiencePolicy(warmup_epochs=4, patience=3)
    decisions = [policy.update(epoch, 0.5) for epoch in range(1, 8)]
    assert not any(item["should_stop"] for item in decisions[:6])
    assert decisions[6]["should_stop"]


def test_epoch_checkpoint_and_storage_contract() -> None:
    specification_path, profiles = _frozen_spec_and_profiles()
    specification = yaml.safe_load(
        specification_path.read_text(encoding="utf-8")
    )
    production = yaml.safe_load(
        (profiles / "production.yaml").read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load(
        (profiles / "smoke.yaml").read_text(encoding="utf-8")
    )
    resume_smoke = yaml.safe_load(
        (profiles / "resume_smoke.yaml").read_text(encoding="utf-8")
    )
    runtime = specification.get("runtime_contract")
    if runtime is None:
        assert specification["experiment_id"] == "X0024"
        expected_updates = 501
        expected_steps = 6_012
    else:
        assert runtime["language"] == specification["language"]
        expected_updates = int(runtime["updates_per_epoch"])
        expected_steps = expected_updates * 12
    assert production["regime"]["num_steps"] == expected_steps
    assert production["regime"]["validate_every_n_steps"] == expected_updates
    assert production["regime"]["checkpoint_every_n_steps"] == expected_updates
    assert production["regime"]["keep_last_n_checkpoints"] == 1
    assert production["regime"]["save_model_only"] is False
    assert resume_smoke["regime"]["num_steps"] == 3
    assert resume_smoke["regime"]["checkpoint_every_n_steps"] == 3
    assert resume_smoke["regime"]["validate_every_n_steps"] == 3
    assert smoke["gang"]["timeout"] == 60
    assert resume_smoke["gang"]["timeout"] == 60
    assert production["gang"]["timeout"] == 60
    if specification["experiment_id"] == "X0024":
        continuation = specification["continuation"]
        assert continuation["source_experiment_id"] == "X0024"
        assert continuation["source_run_id"] == "RUN0122"
        assert continuation["source_packet_digest"] == continuation[
            "checkpoint_namespace_digest"
        ]
        assert continuation["source_epoch"] == 4
        assert continuation["source_step"] == 2_004
        assert continuation["required_retained_steps"] == [1_002, 1_503, 2_004]
        assert continuation["preserve_origin_commit_markers"] is True
    elif specification["language"] == "sna":
        assert specification["language"] == "sna"
        assert "continuation" not in specification
        assert expected_updates == 509
        assert expected_steps == 6_108
        assert production["dataset"]["name"] == "waxal3_sna_cv002_portable"
        assert (
            production["dataset"]["asr_task_config"]["example_shuffle_window"]
            == 16_293
        )
    else:
        assert specification["experiment_id"] in {"X0031", "X0034"}
        assert specification["language"] == "lin"
        assert "continuation" not in specification
        assert specification["parent_asset"]["initialization"] == (
            "untouched_official_parent"
        )
        assert expected_updates == 501
        assert expected_steps == 6_012
        assert production["dataset"]["name"] == "waxal3_lin_cv002_portable"
        assert (
            production["dataset"]["asr_task_config"]["example_shuffle_window"]
            == 16_035
        )
    assert specification["resume"]["checkpoint_every_pass"] is True
    assert specification["resume"]["local_retain_latest"] == 1
    assert specification["resume"]["remote_retain_all_epoch_boundaries"] is False
    assert specification["resume"]["remote_retain_top3_full_plus_latest_full"] is True
    assert specification["resume"]["automatic_local_checkpoint_downloads"] is False
    assert specification["resume"]["final_download_top3_only"] is True
    assert specification["resume"]["interruption_preserves_vast_instance_disk"] is True
