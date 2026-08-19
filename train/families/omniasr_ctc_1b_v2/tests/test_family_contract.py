from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import threading

import polars as pl

from early_stopping import StrictPatiencePolicy
from runtime_assets import render_asset_cards


ROOT = Path(
    os.environ.get("WAXAL3_REPO_ROOT", Path(__file__).resolve().parents[4])
).resolve()


def test_recipe_runtime_import_closure() -> None:
    """The packet must carry the upstream workflow modules used at launch."""

    recipe = importlib.import_module("recipe")
    consistency = importlib.import_module("consistency")
    criterion = importlib.import_module("workflows.recipes.wav2vec2.asr.criterion")
    assert recipe.WaxalWav2Vec2AsrRecipe is not None
    assert (
        recipe.WaxalWav2Vec2AsrRecipe().config_kls is consistency.WaxalCtcRecipeConfig
    )
    assert consistency.DualViewCtcCriterion is not None
    assert criterion.Wav2Vec2AsrCriterion is not None


def test_portable_supervised_view_and_world8_map() -> None:
    manifests = ROOT / "data/derived/portable/omniasr1b_lin_cv002_v1/manifests"
    assert pl.read_parquet(manifests / "train.rows.parquet").height == 16_035
    rows = pl.read_parquet(manifests / "dev.rows.parquet")
    mapping = pl.read_csv(manifests / "dev.rank_map.world8.csv")
    assert rows.height == mapping.height == 900
    assert sorted(mapping["rank"].unique().to_list()) == list(range(8))
    for split in ("train", "dev"):
        header = (
            (manifests / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[0]
        )
        assert not Path(header).is_absolute()
        assert (ROOT / header).is_dir()


def test_relocatable_card_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WAXAL3_REPO_ROOT", str(ROOT))
    template = Path(__file__).resolve().parents[1] / "cards/waxal3.yaml.template"
    rendered = render_asset_cards(template, tmp_path)
    text = rendered.read_text(encoding="utf-8")
    assert "@WAXAL3_REPO_ROOT@" not in text
    assert str(ROOT / "models/omniasr-ctc-1b") in text


def test_relocatable_card_render_is_rank_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WAXAL3_REPO_ROOT", str(ROOT))
    template = tmp_path / "waxal3.yaml.template"
    template.write_text(
        "model_root: @WAXAL3_REPO_ROOT@/models/omniasr-ctc-1b\n"
        + ("padding: deterministic-rank-race-probe\n" * 8_192),
        encoding="utf-8",
    )
    output = tmp_path / "cards"
    workers = 32
    barrier = threading.Barrier(workers)

    def render() -> bytes:
        barrier.wait(timeout=10)
        return render_asset_cards(template, output).read_bytes()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rendered = list(pool.map(lambda _index: render(), range(workers)))

    assert len(set(rendered)) == 1
    assert b"@WAXAL3_REPO_ROOT@" not in rendered[0]
    assert not list(output.glob(".waxal3.yaml.*.tmp"))


def test_target_patience_earliest_stop_is_epoch_seven() -> None:
    policy = StrictPatiencePolicy(warmup_epochs=4, patience=3)
    decisions = [policy.update(epoch, 0.5) for epoch in range(1, 8)]
    assert not any(item["should_stop"] for item in decisions[:6])
    assert decisions[6]["should_stop"]
