"""The shared normalisation contract: every stage must agree byte for byte."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.text import normalize_text, token_key, word_tokens  # noqa: E402


def test_nfc_composes_combining_marks():
    decomposed = "éza"          # e + combining acute
    assert normalize_text(decomposed) == "éza"


def test_whitespace_collapses_and_trims():
    assert normalize_text("  na   motema\tkaka\n") == "na motema kaka"


def test_idempotent():
    once = normalize_text("  Éza   makolo ")
    assert normalize_text(once) == once


def test_none_and_empty_become_empty_string():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_token_key_strips_edges_and_folds_case():
    assert token_key('"Motema,"') == "motema"
    assert token_key("...") == ""


def test_word_tokens_normalises_first():
    assert word_tokens("  Na   motema ") == ["Na", "motema"]
