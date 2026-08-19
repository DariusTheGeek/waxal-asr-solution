from __future__ import annotations

import importlib
import os
from pathlib import Path

import polars as pl
import torch
import yaml
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from fairseq2.composition.lib import _register_library
from fairseq2.data.data_pipeline import Collater
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import DependencyContainer

from activation_checkpointing import install_wav2vec2_llama_layerwise_ac
from data import WaxalAsrTask
from early_stopping import StrictPatiencePolicy
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig
from omnilingual_asr.models.wav2vec2_llama.config import WAV2VEC2_LLAMA_FAMILY
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
from runtime_assets import render_asset_cards
from runtime_config import runtime_geometry_from_experiment
from infer import resolve_audio_root


ROOT_RAW = os.environ.get("WAXAL3_REPO_ROOT")
if not ROOT_RAW:
    raise RuntimeError("packet/source tests require WAXAL3_REPO_ROOT")
ROOT = Path(ROOT_RAW).resolve()
CODE_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_CODES = {"lin": "lin_Latn", "sna": "sna_Latn"}
AUDIO_ROOTS = {
    "lin": "data/derived/omniasr/lin_cv002_supervised_v1/audio",
    "sna": "data/derived/omniasr/sna_cv002_supervised_v1/audio",
}
EXPERIMENT_GEOMETRY = {
    "X0038": {
        "language": "lin",
        "language_code": "lin_Latn",
        "dataset": "waxal3_lin_cv002_portable",
        "updates_per_epoch": 501,
        "production_steps": 6_012,
        "publish_interval": 167,
    },
    "X0040": {
        "language": "sna",
        "language_code": "sna_Latn",
        "dataset": "waxal3_sna_cv002_portable",
        "updates_per_epoch": 509,
        "production_steps": 6_108,
        "publish_interval": 509,
    },
}


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
        / "experiments/omniasr_llm_3b_v2/mono/lin/supervised_ft"
        / "control_max12_target_es"
        / "X0038_20260806T060715192280Z_llm3b_v2_lin_cv002_fsdp2_seed42"
    )
    return experiment / "experiment.yaml", experiment / "profiles"


def test_recipe_runtime_import_closure() -> None:
    recipe = importlib.import_module("recipe")
    criterion = importlib.import_module(
        "workflows.recipes.wav2vec2.asr.criterion"
    )
    assert recipe.WaxalWav2Vec2AsrRecipe is not None
    assert criterion.Wav2Vec2AsrCriterion is not None


def test_packet_registers_and_executes_wav2vec2_llama_layerwise_ac() -> None:
    first = install_wav2vec2_llama_layerwise_ac()
    second = install_wav2vec2_llama_layerwise_ac()
    assert first["status"] == second["status"] == "PASS"
    assert second["mode"] == "already_installed"

    container = DependencyContainer()
    _register_library(container)
    family = container.resolve(ModelFamily, key=WAV2VEC2_LLAMA_FAMILY)
    assert family.supports_layerwise_ac

    class Stack(nn.Module):
        def __init__(self, layers: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(layers))

    model = Wav2Vec2LlamaModel.__new__(Wav2Vec2LlamaModel)
    nn.Module.__init__(model)
    model.encoder = Stack(3)
    model.llama_decoder = Stack(2)
    assert family.apply_layerwise_ac(model, every_nth_layer=1) is model
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.encoder.layers)
    assert all(
        isinstance(layer, CheckpointWrapper) for layer in model.llama_decoder.layers
    )
    value = torch.randn(2, 4, requires_grad=True)
    for layer in [*model.encoder.layers, *model.llama_decoder.layers]:
        value = layer(value).relu()
    value.sum().backward()
    assert value.isfinite().all()


def test_portable_supervised_view_and_world8_map() -> None:
    specification_path, _ = _frozen_spec_and_profiles()
    geometry = runtime_geometry_from_experiment(specification_path.parent, ROOT)
    manifests = geometry.manifest_dir
    assert geometry.language in LANGUAGE_CODES
    assert geometry.language_code == LANGUAGE_CODES[geometry.language]
    assert (
        pl.read_parquet(manifests / "train.rows.parquet").height
        == geometry.expected_train_rows
    )
    rows = pl.read_parquet(manifests / "dev.rows.parquet")
    mapping = pl.read_csv(manifests / "dev.rank_map.world8.csv")
    assert rows.height == mapping.height == 900
    assert sorted(mapping["rank"].unique().to_list()) == list(range(8))
    for split in ("train", "dev"):
        header = (manifests / f"{split}.tsv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        assert not Path(header).is_absolute()
        assert (ROOT / header).is_dir()


def test_language_code_is_attached_and_collated() -> None:
    for language_code in ("lin_Latn", "sna_Latn"):
        task = WaxalAsrTask(AsrTaskConfig(), language_code=language_code)
        first = task.attach_language_code({"value": 1})
        second = task.attach_language_code({"value": 2})
        assert first["lang"] == second["lang"] == language_code
        collated = Collater(pad_value=0)([first, second])
        assert collated["lang"] == [language_code, language_code]


def test_inference_resolves_frozen_tsv_audio_root() -> None:
    specification_path, _ = _frozen_spec_and_profiles()
    geometry = runtime_geometry_from_experiment(specification_path.parent, ROOT)
    observed = resolve_audio_root(
        geometry.manifest_dir / "dev.rows.parquet", ROOT
    )
    assert geometry.language in AUDIO_ROOTS
    assert observed == (ROOT / AUDIO_ROOTS[geometry.language]).resolve()


def test_relocatable_llm_card_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WAXAL3_REPO_ROOT", str(ROOT))
    template = CODE_ROOT / "cards/waxal3.yaml.template"
    rendered = render_asset_cards(template, tmp_path)
    text = rendered.read_text(encoding="utf-8")
    assert "@WAXAL3_REPO_ROOT@" not in text
    assert str(ROOT / "models/omniasr-llm-3b-v2") in text
    assert "model_family: wav2vec2_llama" in text
    assert "model_arch: 3b_v2" in text
    assert "waxal3_lin_cv002_portable" in text
    assert "waxal3_sna_cv002_portable" in text
    assert "omniasr-ctc-7b-v2" not in text


def test_target_patience_earliest_stop_is_epoch_seven() -> None:
    policy = StrictPatiencePolicy(warmup_epochs=4, patience=3)
    decisions = [policy.update(epoch, 0.5) for epoch in range(1, 8)]
    assert not any(item["should_stop"] for item in decisions[:6])
    assert decisions[6]["should_stop"]


def test_llm_training_and_checkpoint_contract() -> None:
    specification_path, profiles = _frozen_spec_and_profiles()
    specification = yaml.safe_load(
        specification_path.read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load((profiles / "smoke.yaml").read_text(encoding="utf-8"))
    production = yaml.safe_load(
        (profiles / "production.yaml").read_text(encoding="utf-8")
    )
    runtime = specification["runtime_contract"]
    experiment_id = specification["experiment_id"]
    assert experiment_id in EXPERIMENT_GEOMETRY
    expected = EXPERIMENT_GEOMETRY[experiment_id]
    assert "continuation" not in specification
    assert specification["language"] == expected["language"]
    assert runtime["language_code"] == expected["language_code"]
    assert runtime["updates_per_epoch"] == expected["updates_per_epoch"]
    assert runtime["expected_parameter_count"] == 4_380_578_432
    assert runtime["expected_encoder_layers"] == 60
    assert runtime["expected_encoder_model_dim"] == 2_048
    assert runtime["expected_encoder_ffn_inner_dim"] == 8_192
    assert runtime["expected_decoder_layers"] == 12
    assert smoke["model"]["name"] == production["model"]["name"]
    assert production["model"]["name"] == "waxal3_omni_llm_3b_v2_target_es_parent"
    assert smoke["dataset"]["name"] == expected["dataset"]
    assert production["dataset"]["name"] == expected["dataset"]
    assert production["optimizer"]["config"]["lr"] == 5e-5
    assert production["regime"]["num_steps"] == expected["production_steps"]
    assert (
        production["regime"]["validate_every_n_steps"]
        == expected["updates_per_epoch"]
    )
    assert (
        production["regime"]["checkpoint_every_n_steps"]
        == expected["updates_per_epoch"]
    )
    assert (
        production["regime"]["publish_metrics_every_n_steps"]
        == expected["publish_interval"]
    )
    assert expected["production_steps"] == expected["updates_per_epoch"] * 12
    assert production["regime"]["keep_last_n_checkpoints"] == 1
    assert production["regime"]["save_model_only"] is False
    assert smoke["regime"]["num_steps"] == 2
    for profile in (smoke, production):
        assert profile["gang"]["timeout"] == 60
        assert profile["trainer"]["data_parallelism"] == "fsdp"
        assert profile["trainer"]["fsdp"] == {
            "version": "v2",
            "granularity": "layer",
            "hybrid": False,
            "reshard_after_forward": True,
            "fp32_reduce": True,
        }
        assert profile["trainer"]["grad_accumulation"]["num_batches"] == 4
        assert profile["trainer"]["activation_checkpointing"] == {
            "mode": "layerwise",
            "every_nth_layer": 1,
        }
    resume = specification["resume"]
    assert resume["checkpoint_every_pass"] is True
    assert resume["local_retain_latest"] == 1
    assert resume["remote_retain_top3_full_plus_latest_full"] is True
    assert resume["automatic_local_checkpoint_downloads"] is False
    assert resume["final_download_top3_only"] is True
