"""The TTIA assembly: boundary-norm, ROVER candidate, selection, and its guards."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.fuse.fuse import fuse_ttia  # noqa: E402


@pytest.fixture()
def train_texts(tmp_path):
    """Two profiles with opposite word-boundary conventions, well past the
    boundary model's minimum support of 10."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        # The parquet's own column name; it holds profile keys.
        "speaker_key": ["split"] * 12 + ["join"] * 12,
        "training_target_stable_raw": ["na mo tema kaka"] * 12
                                      + ["na motema kaka"] * 12,
    })
    path = tmp_path / "train.rows.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_hypotheses_bend_toward_the_matched_profile(train_texts):
    surfaces = [{"c1": "motema", "c2": "mo tema"},
                {"c1": "motema", "c2": "mo tema"},
                {"c1": "motema", "c2": "mo tema"}]
    keys = {"c1": "split", "c2": "join"}
    rows, selections = fuse_ttia(surfaces, ["c1", "c2"], keys, train_texts)
    fused = {r["ID"]: r["Target"] for r in rows}
    assert fused["c1"] == "mo tema"      # same input text, opposite outputs,
    assert fused["c2"] == "motema"       # driven only by the matched profile
    assert sum(selections.values()) == 2


def test_a_blank_member_never_wins_while_text_exists(train_texts):
    """An empty candidate scores as zero-cost text; the guard keeps it out."""
    surfaces = [{"c1": "motema"}, {"c1": "motema"}, {"c1": ""}]
    rows, _ = fuse_ttia(surfaces, ["c1"], {"c1": "split"}, train_texts)
    assert rows[0]["Target"] == "mo tema"


def test_missing_profile_key_fails_loud(train_texts):
    surfaces = [{"c1": "a"}, {"c1": "a"}, {"c1": "a"}]
    with pytest.raises(SystemExit, match="no matched profile key"):
        fuse_ttia(surfaces, ["c1"], {}, train_texts)
