"""Asset cards must name model families fairseq2 actually registers.

A wrong `model_family` does not fail when the card is written; it fails when the
model is loaded, which on this pipeline is after a multi-GB checkpoint has
already been read. Check it cheaply instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CARDS = sorted((ROOT / "models/cards").glob("*/card.yaml"))

# CTC models are wav2vec2_asr; the LLM-decoder models are wav2vec2_llama.
VALID_FAMILIES = {"wav2vec2_asr", "wav2vec2_llama"}
EXPECTED = {
    "waxal-joint-ctc-1b-lid": "wav2vec2_asr",
    "waxal-lin-omniasr-ctc-3b": "wav2vec2_asr",
    "waxal-lin-omniasr-ctc-1b": "wav2vec2_asr",
    "waxal-sna-omniasr-ctc-7b": "wav2vec2_asr",
    "waxal-lin-omniasr-llm-1b": "wav2vec2_llama",
    "waxal-sna-omniasr-llm-1b": "wav2vec2_llama",
    "waxal-lin-omniasr-llm-3b": "wav2vec2_llama",
    "waxal-sna-omniasr-llm-3b": "wav2vec2_llama",
}


def documents(path: Path) -> list[dict]:
    """Parse a card the way the loader does: render the placeholder first.

    A card with `@WAXAL_MODEL_DIR@` still in it is not valid YAML -- `@` cannot
    start a token. That is intentional: the card is a template, and
    `inference/decode/omniasr.py` substitutes the real weights directory before
    fairseq2 ever sees it.
    """
    rendered = path.read_text().replace("@WAXAL_MODEL_DIR@", "/rendered")
    return [d for d in yaml.safe_load_all(rendered) if d]


@pytest.mark.parametrize("card", CARDS, ids=lambda p: p.parent.name)
def test_model_family_is_registered(card: Path) -> None:
    model = [d for d in documents(card) if "model_family" in d]
    assert len(model) == 1, f"{card} should declare exactly one model"
    family = model[0]["model_family"]
    assert family in VALID_FAMILIES, f"{card}: unknown model_family {family!r}"
    assert family == EXPECTED[card.parent.name], \
        f"{card}: expected {EXPECTED[card.parent.name]}, found {family}"


@pytest.mark.parametrize("card", CARDS, ids=lambda p: p.parent.name)
def test_card_has_tokenizer_and_placeholder(card: Path) -> None:
    text = card.read_text()
    assert "@WAXAL_MODEL_DIR@" in text, "card must be relocatable"
    tokenizer = [d for d in documents(card) if "tokenizer_family" in d]
    assert len(tokenizer) == 1, f"{card} should declare exactly one tokenizer"
    model = [d for d in documents(card) if "model_family" in d][0]
    assert model["tokenizer_ref"] == tokenizer[0]["name"]


def test_every_omniasr_repo_ships_a_card() -> None:
    """Every OmniASR repo needs an asset card; the two MMS repos load without one."""
    spec = json.loads((ROOT / "models/publish/repos.json").read_text())
    omni = [r["name"] for r in spec["repos"] if "mms" not in r["name"]]
    assert len(omni) == 8, f"expected 8 OmniASR repos, found {len(omni)}"
    for name in omni:
        assert (ROOT / "models/cards" / name / "card.yaml").is_file(), \
            f"{name} is an OmniASR repo with no asset card"


def test_every_repo_ships_pinned_requirements() -> None:
    """Each HF repo carries a requirements.txt pinning its inference runtime."""
    spec = json.loads((ROOT / "models/publish/repos.json").read_text())
    for r in spec["repos"]:
        req = ROOT / "models/cards" / r["name"] / "requirements.txt"
        assert req.is_file(), f"{r['name']} has no requirements.txt"
        assert "==" in req.read_text(), f"{req} pins nothing"
