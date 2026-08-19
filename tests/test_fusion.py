"""Fusion behaviour that the pipeline depends on and that is easy to break silently."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.fuse.fuse import conservative_word_rover, word_medoid  # noqa: E402


def test_rover_refuses_two_sources():
    """With two hypotheses the pivot always wins, so the fusion is a no-op."""
    with pytest.raises(ValueError, match="at least 3"):
        conservative_word_rover(["a b c", "a x c"])


def test_rover_adopts_on_strict_majority():
    assert conservative_word_rover(["a b c", "a x c", "a x c"]) == "a x c"


def test_rover_keeps_pivot_without_majority():
    """Two against two is not a majority; the pivot's surface survives."""
    assert conservative_word_rover(["a b c", "a b c", "a x c", "a x c"]) == "a b c"


def test_rover_never_adopts_insertions():
    assert conservative_word_rover(["a c", "a b c", "a b c"]) == "a c"


def test_rover_is_case_and_punctuation_insensitive_when_voting():
    """Votes are cast on the folded key, but the emitted surface is the pivot's."""
    assert conservative_word_rover(["a B, c", "a b c", "a b c"]) == "a B, c"


def test_rover_output_is_normalised():
    assert conservative_word_rover(["  a   b  ", "a b", "a b"]) == "a b"


def test_medoid_picks_the_consensus_member():
    hypotheses = ["a b c", "a b c", "z z z"]
    target, index = word_medoid(hypotheses)
    assert target == "a b c" and index == 0


def test_medoid_breaks_ties_by_input_order():
    """Member order is part of the contract; reordering can change the output."""
    _, index = word_medoid(["a b", "c d"])
    assert index == 0


def test_medoid_is_stable_under_reordering_of_equal_members():
    assert word_medoid(["x y", "x y", "x y"]) == ("x y", 0)


def test_medoid_folds_case_when_measuring_distance():
    """Case-only differences are not real disagreement.

    Comparing case-sensitively shifts the Shona selection counts from the
    shipped 316/100/29/0 to 304/108/33/0 -- a silent, plausible-looking
    regression that changes the submission.
    """
    target, index = word_medoid(["A B", "a b", "z z"])
    assert index == 0
    assert target == "A B", "the selected member's original casing must survive"


def test_rover_falls_back_when_the_pivot_is_empty():
    """A blank pivot must not silently discard what other members transcribed.

    Near-silent clips make a CTC pivot collapse to empty while an LLM decoder
    still emits text. Voting at pivot positions would return nothing.
    """
    out = conservative_word_rover(["", "hello world", "hello world"])
    assert out == "hello world"


def test_rover_returns_empty_only_when_every_member_is_empty():
    assert conservative_word_rover(["", "", ""]) == ""


def test_rover_fallback_does_not_disturb_the_normal_path():
    """With a non-empty pivot, behaviour is unchanged."""
    assert conservative_word_rover(["a b c", "a x c", "a x c"]) == "a x c"


def test_rover_fallback_prefers_the_stronger_member_not_the_longer_one():
    """Member order decides the fallback, not string length.

    The medoid over two hypotheses normalises each distance by that
    hypothesis's own length, so it systematically favours the longer string --
    and on a near-silent clip the longer string is usually a decoder repeating
    itself. Configured member order is strongest-first, so use that instead.
    """
    short_strong = "awa namoni balakisi"
    long_weak = "awa namoni balakisi eza eza eza eza eza eza eza"
    assert conservative_word_rover(["", short_strong, long_weak]) == short_strong


def test_rover_fallback_votes_when_three_members_speak():
    """With three speaking members the fallback re-pivots and votes properly."""
    out = conservative_word_rover(["", "a b c", "a x c", "a x c"])
    assert out == "a x c"
