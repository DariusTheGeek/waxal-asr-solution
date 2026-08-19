from pathlib import Path

from supervised.analyze import cluster_bootstrap, score
from supervised.contract import load_frozen_scorer, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict[str, str]]:
    return [
        {
            "id": "a",
            "speaker_key": "s1",
            "reference_raw": "one two",
            "reference_ctc": "one two",
            "hypothesis": "one two",
            "target_weight": "0.5",
        },
        {
            "id": "b",
            "speaker_key": "s2",
            "reference_raw": "three",
            "reference_ctc": "three",
            "hypothesis": "",
            "target_weight": "1.0",
        },
    ]


def test_cluster_bootstrap_is_deterministic_and_scorer_is_mixed_case() -> None:
    scorer_path = ROOT / "scorer_compat.py"
    scorer = load_frozen_scorer(scorer_path, sha256_file(scorer_path))
    first = cluster_bootstrap(_rows(), scorer, replicates=128, seed=7)
    second = cluster_bootstrap(_rows(), scorer, replicates=128, seed=7)
    assert first == second
    assert first["clusters"] == 2
    assert score(_rows(), scorer)["blank_fraction"] == 0.5
    mixed = scorer.score_texts(["Ab C."], ["ab c."])
    assert mixed["wer"] == 0.0
    assert mixed["cer"] > 0.0


def test_target_weighted_scorer_changes_row_influence() -> None:
    scorer_path = ROOT / "scorer_compat.py"
    scorer = load_frozen_scorer(scorer_path, sha256_file(scorer_path))
    unweighted = scorer.score_texts(["correct", "lost"], ["correct", ""])
    weighted = scorer.score_weighted_texts(
        ["correct", "lost"], ["correct", ""], [10.0, 1.0]
    )
    assert weighted["q"] > unweighted["q"]
