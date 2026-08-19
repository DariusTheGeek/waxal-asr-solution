"""Frozen subset of WAXAL3's canonical raw ASR scoring contract."""

from __future__ import annotations

import math
import unicodedata

import jiwer


CANONICAL_SCORER_SOURCE_SHA256 = (
    "e01a5ee0d7c562ce817ae5dd8096d24937bf58cc04d44766835e4250a604195a"
)


def raw_text(text: object, *, casefold: bool = False) -> str:
    """Apply only declared raw-metric normalization."""

    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFC", value)
    if casefold:
        value = value.casefold()
    return " ".join(value.split())


def score_texts(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Score aligned text lists using the raw leaderboard contract."""

    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be non-empty and aligned")
    refs_cer = [raw_text(value, casefold=False) for value in references]
    hyps_cer = [raw_text(value, casefold=False) for value in hypotheses]
    refs_wer = [raw_text(value, casefold=True) for value in references]
    hyps_wer = [raw_text(value, casefold=True) for value in hypotheses]
    cer = float(jiwer.cer(refs_cer, hyps_cer))
    wer = float(jiwer.wer(refs_wer, hyps_wer))
    combined_error = 0.5 * (cer + wer)
    return {
        "cer": cer,
        "wer": wer,
        "combined_error": combined_error,
        "score": 1.0 - combined_error,
    }


def score_weighted_texts(
    references: list[str], hypotheses: list[str], weights: list[float]
) -> dict[str, float]:
    """Score aligned text with fixed non-negative per-row target weights."""

    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be non-empty and aligned")
    if len(weights) != len(references):
        raise ValueError("weights must align with references")
    numeric_weights = [float(value) for value in weights]
    if any(not (value >= 0.0) for value in numeric_weights):
        raise ValueError("weights must be finite and non-negative")
    if any(value == float("inf") for value in numeric_weights) or sum(
        numeric_weights
    ) <= 0:
        raise ValueError("weights must be finite with positive total mass")

    wer_errors = 0.0
    wer_reference = 0.0
    cer_errors = 0.0
    cer_reference = 0.0
    for reference, hypothesis, weight in zip(
        references, hypotheses, numeric_weights, strict=True
    ):
        if weight == 0.0:
            continue
        ref_wer = raw_text(reference, casefold=True)
        hyp_wer = raw_text(hypothesis, casefold=True)
        word_output = jiwer.process_words(ref_wer, hyp_wer)
        wer_errors += weight * (
            word_output.substitutions
            + word_output.deletions
            + word_output.insertions
        )
        wer_reference += weight * (
            word_output.hits
            + word_output.substitutions
            + word_output.deletions
        )

        ref_cer = raw_text(reference, casefold=False)
        hyp_cer = raw_text(hypothesis, casefold=False)
        char_output = jiwer.process_characters(ref_cer, hyp_cer)
        cer_errors += weight * (
            char_output.substitutions
            + char_output.deletions
            + char_output.insertions
        )
        cer_reference += weight * (
            char_output.hits
            + char_output.substitutions
            + char_output.deletions
        )

    if wer_reference <= 0 or cer_reference <= 0:
        raise ValueError("weighted references must contain words and characters")
    wer = wer_errors / wer_reference
    cer = cer_errors / cer_reference
    combined_error = 0.5 * (cer + wer)
    result = {
        "cer": float(cer),
        "wer": float(wer),
        "combined_error": float(combined_error),
        "score": float(1.0 - combined_error),
        "weight_sum": float(sum(numeric_weights)),
        "weighted_reference_words": float(wer_reference),
        "weighted_reference_characters": float(cer_reference),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("weighted score produced a non-finite value")
    return result
