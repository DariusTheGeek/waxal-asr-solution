from __future__ import annotations

import numpy as np
from pyctcdecode import build_ctcdecoder

from decoding.beam_postprocess import (
    PostprocessAssets,
    apply_policy,
    greedy_ctc,
    labels_from_vocab,
    sentence_case,
)


VOCAB = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}


def test_waxal2_label_mapping_and_greedy_ctc() -> None:
    assert labels_from_vocab(VOCAB) == ["", "⁇", " ", "a", "b"]
    assert greedy_ctc([0, 3, 3, 0, 2, 4, 4, 0], VOCAB) == "a b"


def test_lm_free_beam_width_64_smoke() -> None:
    logits = np.full((8, 5), -20.0, dtype=np.float32)
    for frame, token in enumerate([0, 3, 3, 0, 2, 4, 4, 0]):
        logits[frame, token] = 0.0
    decoder = build_ctcdecoder(labels_from_vocab(VOCAB))
    assert decoder.decode(logits, beam_width=64) == "a b"


def test_waxal2_terminal_policy_and_always_period_improvement() -> None:
    texts = [
        "Alpha moto.",
        "Beta moto.",
        "Gamma moto.",
        "Delta moto.",
        "Epsilon moto.",
        "Zeta nde",
    ]
    assets = PostprocessAssets.fit(texts)
    assert apply_policy("oyo moto", "waxal2_terminal_sentence_case", assets) == (
        "Oyo moto."
    )
    assert apply_policy("oyo nde", "waxal2_terminal_sentence_case", assets) == (
        "Oyo nde."
    )
    assert apply_policy("oyo", "always_period_sentence_case", assets) == "Oyo."


def test_truecase_uses_only_fitted_training_lexicon() -> None:
    texts = [f"oyo Africa ezali {index}." for index in range(5)]
    assets = PostprocessAssets.fit(texts)
    assert "africa" in assets.proper
    assert apply_policy("oyo africa ezali", "truecase", assets) == (
        "Oyo Africa ezali"
    )
    assert sentence_case("oyo rdc") == "Oyo rdc"
