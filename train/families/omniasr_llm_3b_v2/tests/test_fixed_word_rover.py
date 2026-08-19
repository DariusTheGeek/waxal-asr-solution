from __future__ import annotations

import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from fixed_word_rover import conservative_word_rover  # noqa: E402


def test_fixed_word_rover_adopts_a_strict_majority_replacement() -> None:
    assert conservative_word_rover(["one two three", "one x three", "one x three"]) == "one x three"


def test_fixed_word_rover_ignores_insertions_and_retains_pivot_on_ties() -> None:
    assert conservative_word_rover(["one three", "one two three", "one three"]) == "one three"
    assert conservative_word_rover(["one two", "one x", "one two"]) == "one two"
