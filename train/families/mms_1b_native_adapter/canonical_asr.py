#!/usr/bin/env python3
"""Canonical WAXAL competition scorer.

The raw metric is corpus-micro CER/WER after NFC and whitespace collapse.
CER remains case-sensitive; WER is casefolded. Punctuation is retained in both.
The leaderboard quantity is ``1 - (CER + WER) / 2``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import jiwer


ANNOTATION = re.compile(r'<\s*"?[^>]*"?\s*>|\[[^\]]*\]')


def raw_text(text: object, *, casefold: bool = False) -> str:
    """Apply only declared raw-metric normalization."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFC", value)
    if casefold:
        value = value.casefold()
    return " ".join(value.split())


def ctc_text(text: object, *, preserve_case: bool = False) -> str:
    """Return the explicit punctuation-light diagnostic/training text view."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKD", value)
    if not preserve_case:
        value = value.casefold()
    value = value.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = ANNOTATION.sub(" ", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    cleaned: list[str] = []
    for index, char in enumerate(value):
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            cleaned.append(char)
        elif (
            char == "'"
            and 0 < index < len(value) - 1
            and unicodedata.category(value[index - 1])[0] in {"L", "M", "N"}
            and unicodedata.category(value[index + 1])[0] in {"L", "M", "N"}
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


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
    """Score aligned text with fixed non-negative per-row target weights.

    This is a declared target-aligned diagnostic. It does not replace
    :func:`score_texts`, which remains the canonical unweighted raw scorer.
    """
    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be non-empty and aligned")
    if len(weights) != len(references):
        raise ValueError("weights must align with references")
    numeric_weights = [float(value) for value in weights]
    if any(not (value >= 0.0) for value in numeric_weights):
        raise ValueError("weights must be finite and non-negative")
    if any(value == float("inf") for value in numeric_weights) or sum(numeric_weights) <= 0:
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
            word_output.substitutions + word_output.deletions + word_output.insertions
        )
        wer_reference += weight * (
            word_output.hits + word_output.substitutions + word_output.deletions
        )

        ref_cer = raw_text(reference, casefold=False)
        hyp_cer = raw_text(hypothesis, casefold=False)
        char_output = jiwer.process_characters(ref_cer, hyp_cer)
        cer_errors += weight * (
            char_output.substitutions + char_output.deletions + char_output.insertions
        )
        cer_reference += weight * (
            char_output.hits + char_output.substitutions + char_output.deletions
        )

    if wer_reference <= 0 or cer_reference <= 0:
        raise ValueError("weighted references must contain words and characters")
    wer = wer_errors / wer_reference
    cer = cer_errors / cer_reference
    combined_error = 0.5 * (cer + wer)
    return {
        "cer": float(cer),
        "wer": float(wer),
        "combined_error": float(combined_error),
        "score": float(1.0 - combined_error),
        "weight_sum": float(sum(numeric_weights)),
        "weighted_reference_words": float(wer_reference),
        "weighted_reference_characters": float(cer_reference),
    }


def read_column(path: Path, id_column: str, value_column: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {id_column, value_column} - set(reader.fieldnames):
            raise ValueError(f"missing required columns in {path}")
        output: dict[str, str] = {}
        for row in reader:
            identifier = row[id_column]
            if identifier in output:
                raise ValueError(f"duplicate ID in {path}: {identifier}")
            output[identifier] = row[value_column]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--reference-column", default="Target")
    parser.add_argument("--hypothesis-column", default="Target")
    args = parser.parse_args()
    references = read_column(args.references, args.id_column, args.reference_column)
    hypotheses = read_column(args.hypotheses, args.id_column, args.hypothesis_column)
    if references.keys() != hypotheses.keys():
        missing = sorted(references.keys() - hypotheses.keys())
        extra = sorted(hypotheses.keys() - references.keys())
        raise ValueError(f"ID mismatch: missing={missing[:5]} extra={extra[:5]}")
    identifiers = list(references)
    result = {"rows": len(identifiers), **score_texts(
        [references[item] for item in identifiers],
        [hypotheses[item] for item in identifiers],
    )}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
