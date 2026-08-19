"""TTIA idiolect models on toy data: boundary conventions and LM ranking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.ttia.idiolect import (  # noqa: E402
    BoundaryConfig, BoundaryModel, TextLanguageModel)

CONFIG = BoundaryConfig(scope="profile", minimum_support=10, preference_ratio=2.0)

# Profile "split" writes the compound as two words; profile "join" writes it
# as one. Twelve observations clear the minimum_support of 10.
PROFILES = ["split"] * 12 + ["join"] * 12
TEXTS = ["na mo tema kaka"] * 12 + ["na motema kaka"] * 12


def test_boundary_splits_toward_the_profile_convention():
    model = BoundaryModel(PROFILES, TEXTS)
    assert model.normalize("motema", "split", CONFIG) == "mo tema"


def test_boundary_merges_toward_the_profile_convention():
    model = BoundaryModel(PROFILES, TEXTS)
    assert model.normalize("mo tema", "join", CONFIG) == "motema"


def test_boundary_respects_initial_case():
    model = BoundaryModel(PROFILES, TEXTS)
    assert model.normalize("Motema", "split", CONFIG) == "Mo tema"


def test_unknown_profile_falls_back_to_global_counts():
    """A key with no training text must not crash or change the text: the two
    conventions tie globally (12 each), which clears neither preference ratio."""
    model = BoundaryModel(PROFILES, TEXTS)
    assert model.normalize("motema", "unseen", CONFIG) == "motema"
    assert model.normalize("mo tema", "unseen", CONFIG) == "mo tema"


def test_lm_prefers_the_profile_wording():
    lm = TextLanguageModel(PROFILES, TEXTS)
    args = {"lexical": True, "scope": "profile", "order": 1, "prior": 20.0}
    in_profile = lm.score("na motema kaka", "join", **args)
    off_profile = lm.score("zoba pamba lobi", "join", **args)
    assert in_profile < off_profile  # score is a cost


def test_lm_ranking_is_profile_scoped():
    """The same candidate pair ranks differently under different profiles."""
    lm = TextLanguageModel(PROFILES, TEXTS)
    args = {"lexical": True, "scope": "profile", "order": 1, "prior": 20.0}
    split_form = "na mo tema kaka"
    join_form = "na motema kaka"
    assert lm.score(split_form, "split", **args) < lm.score(join_form, "split", **args)
    assert lm.score(join_form, "join", **args) < lm.score(split_form, "join", **args)
