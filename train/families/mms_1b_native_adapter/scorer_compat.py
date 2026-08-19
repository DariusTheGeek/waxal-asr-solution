"""Packet-local compatibility facade over the canonical WAXAL3 scorer."""

from __future__ import annotations

import jiwer

from canonical_asr import (
    ctc_text,
    raw_text,
    score_texts as _score_texts,
    score_weighted_texts as _score_weighted_texts,
)


def raw_wer_text(value: object) -> str:
    return raw_text(value, casefold=True)


def raw_cer_text(value: object) -> str:
    return raw_text(value, casefold=False)


def score_texts(
    references: list[str],
    hypotheses: list[str],
    *,
    normalized: bool = False,
) -> dict[str, float]:
    if normalized:
        references = [ctc_text(value) for value in references]
        hypotheses = [ctc_text(value) for value in hypotheses]
    result = _score_texts(references, hypotheses)
    return {
        "wer": float(result["wer"]),
        "cer": float(result["cer"]),
        "weighted_error": float(result["combined_error"]),
        "q": float(result["score"]),
    }


def score_weighted_texts(
    references: list[str],
    hypotheses: list[str],
    weights: list[float],
    *,
    normalized: bool = False,
) -> dict[str, float]:
    """Expose the frozen target-slot-weighted WAXAL3 scorer contract."""

    if normalized:
        references = [ctc_text(value) for value in references]
        hypotheses = [ctc_text(value) for value in hypotheses]
    result = _score_weighted_texts(references, hypotheses, weights)
    return {
        "wer": float(result["wer"]),
        "cer": float(result["cer"]),
        "weighted_error": float(result["combined_error"]),
        "q": float(result["score"]),
        "weight_sum": float(result["weight_sum"]),
        "weighted_reference_words": float(result["weighted_reference_words"]),
        "weighted_reference_characters": float(
            result["weighted_reference_characters"]
        ),
    }
