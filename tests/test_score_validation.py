"""The validation scorer must agree with the frozen canonical contract.

`tools/score_validation.py` is what produced the Val CER/WER/Q column in the
README. It reimplements the metric so the tool runs without importing family
code, which is exactly the kind of copy that drifts silently -- so pin it
against the real `canonical_scoring.score_texts` here.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = _load(ROOT / "tools/score_validation.py", "score_validation")
canonical = _load(
    ROOT / "train/families/omniasr_ctc_3b_v2/canonical_scoring.py", "canonical_scoring"
)

CASES = [
    (["Toza komona ndako"], ["toza komona ndaku"]),
    (["na moni oyo eza ba biki", "Foto oyo elakisi biso"],
     ["na moni oyo eza ba bike", "Foto oyo elakis biso"]),
    (["Sapatu ya moyindo"], [""]),                      # blank hypothesis
    (["eza malamu"], ["eza malamu"]),                   # exact match
]


@pytest.mark.parametrize("references,hypotheses", CASES)
def test_matches_canonical_scoring(references, hypotheses):
    mine = scorer.score_texts(references, hypotheses)
    theirs = canonical.score_texts(references, hypotheses)
    for key in ("cer", "wer", "score"):
        assert mine[key] == pytest.approx(theirs[key], abs=1e-12), key


def test_q_is_one_minus_mean_error():
    result = scorer.score_texts(["eza malamu mingi"], ["eza malamu mingu"])
    assert result["score"] == pytest.approx(1.0 - 0.5 * (result["cer"] + result["wer"]))


def test_rejects_misaligned_input():
    with pytest.raises(ValueError):
        scorer.score_texts(["a", "b"], ["a"])
    with pytest.raises(ValueError):
        scorer.score_texts([], [])


def test_published_scores_are_self_consistent():
    """The committed evidence must satisfy the identity the README states."""
    record = json.loads((ROOT / "tools/validation_scores.json").read_text())
    assert record["surfaces"], "no surfaces recorded"
    for name, row in record["surfaces"].items():
        assert row["rows"] == 900, f"{name} scored {row['rows']} rows, expected 900"
        assert row["score"] == pytest.approx(1.0 - 0.5 * (row["cer"] + row["wer"])), name
