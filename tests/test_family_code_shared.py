"""The model families genuinely share code; assert that they still do.

`train/families/` keeps one verbatim copy per family rather than hoisting the
shared modules into a common package. These tests document the sharing and
fail if the copies drift apart.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

FAMILIES = Path(__file__).resolve().parents[1] / "train/families"
OMNIASR = ["omniasr_ctc_1b_v2", "omniasr_ctc_3b_v2", "omniasr_ctc_7b_v2",
           "omniasr_llm_1b_v2", "omniasr_llm_3b_v2"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(names: list[str], relpath: str) -> dict[str, str]:
    return {n: digest(FAMILIES / n / relpath) for n in names
            if (FAMILIES / n / relpath).is_file()}


@pytest.mark.parametrize("relpath", [
    "canonical_scoring.py",
    "workflows/recipes/wav2vec2/asr/criterion.py",
    "workflows/recipes/wav2vec2/asr/metrics.py",
    "workflows/recipes/wav2vec2/asr/default_config.py",
    "workflows/recipes/wav2vec2/asr/recipe.py",
])
def test_shared_across_all_omniasr_families(relpath: str) -> None:
    found = group(OMNIASR, relpath)
    if len(found) < len(OMNIASR):
        pytest.skip(f"{relpath} absent from {set(OMNIASR) - set(found)}")
    assert len(set(found.values())) == 1, f"{relpath} has drifted: {found}"


def test_ctc_families_share_the_data_module() -> None:
    found = group(["omniasr_ctc_1b_v2", "omniasr_ctc_3b_v2", "omniasr_ctc_7b_v2"], "data.py")
    assert len(set(found.values())) == 1, f"CTC data.py has drifted: {found}"


def test_llm_families_share_the_data_module() -> None:
    found = group(["omniasr_llm_1b_v2", "omniasr_llm_3b_v2"], "data.py")
    assert len(set(found.values())) == 1, f"LLM data.py has drifted: {found}"


def test_ctc_3b_and_7b_share_the_recipe() -> None:
    """CTC-3B and CTC-7B differ only in scale; one recipe drives both."""
    found = group(["omniasr_ctc_3b_v2", "omniasr_ctc_7b_v2"], "recipe.py")
    assert len(set(found.values())) == 1, f"recipe.py has drifted: {found}"


def test_every_family_has_a_training_entry_point() -> None:
    missing = [n for n in OMNIASR if not (FAMILIES / n / "train.py").is_file()]
    assert not missing, f"families without train.py: {missing}"
    assert (FAMILIES / "mms_1b_native_adapter/supervised/train.py").is_file()
