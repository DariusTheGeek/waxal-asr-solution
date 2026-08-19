#!/usr/bin/env python3
"""Exact scoring and speaker-cluster diagnostics for one completed E04 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .contract import (
    experiment_root_from,
    load_frozen_scorer,
    read_json,
    resolve_uri,
    sha256_file,
    write_json_create_only,
)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "id",
        "language",
        "model_language",
        "speaker_key",
        "stratum",
        "duration_s",
        "output_frames",
        "original_split",
        "is_phase2_test_speaker",
        "is_phase2_test_prompt",
        "target_weight",
        "slot_id",
        "target_phase2_id",
        "target_official_order",
        "reference_raw",
        "reference_ctc",
        "hypothesis",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("prediction schema failed")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prediction IDs")
    if any(row["model_language"] != row["language"] for row in rows):
        raise ValueError("specialist model/language drift")
    if any(
        not math.isfinite(float(row["duration_s"]))
        or float(row["duration_s"]) <= 0.0
        or int(row["output_frames"]) <= 0
        for row in rows
    ):
        raise ValueError("prediction duration/frame contract failed")
    return rows


def score(rows: list[dict[str, str]], scorer) -> dict[str, object]:
    if not rows:
        return {"rows": 0}
    hypotheses = [row["hypothesis"] for row in rows]
    raw = scorer.score_texts([row["reference_raw"] for row in rows], hypotheses)
    content = scorer.score_texts(
        [row["reference_ctc"] for row in rows],
        hypotheses,
        normalized=True,
    )
    target_weighted_raw = scorer.score_weighted_texts(
        [row["reference_raw"] for row in rows],
        hypotheses,
        [float(row["target_weight"]) for row in rows],
    )
    target_weighted_content = scorer.score_weighted_texts(
        [row["reference_ctc"] for row in rows],
        hypotheses,
        [float(row["target_weight"]) for row in rows],
        normalized=True,
    )
    blanks = [row["id"] for row in rows if not row["hypothesis"].strip()]
    return {
        "rows": len(rows),
        "raw": {key: float(value) for key, value in raw.items()},
        "content": {key: float(value) for key, value in content.items()},
        "target_weighted_raw": {
            key: float(value) for key, value in target_weighted_raw.items()
        },
        "target_weighted_content": {
            key: float(value) for key, value in target_weighted_content.items()
        },
        "blank_rows": len(blanks),
        "blank_fraction": len(blanks) / len(rows),
    }


def worst_utterances(
    rows: list[dict[str, str]],
    scorer,
    *,
    limit: int = 25,
) -> list[dict[str, object]]:
    values = []
    for row in rows:
        raw = scorer.score_texts([row["reference_raw"]], [row["hypothesis"]])
        content = scorer.score_texts(
            [row["reference_ctc"]],
            [row["hypothesis"]],
            normalized=True,
        )
        values.append(
            {
                "id": row["id"],
                "speaker_key": row["speaker_key"],
                "stratum": row["stratum"],
                "duration_s": float(row["duration_s"]),
                "reference_raw": row["reference_raw"],
                "hypothesis": row["hypothesis"],
                "raw": raw,
                "content": content,
            }
        )
    return sorted(
        values,
        key=lambda row: (
            -float(row["raw"]["weighted_error"]),
            -float(row["content"]["weighted_error"]),
            str(row["id"]),
        ),
    )[:limit]


def _error_counts(rows: list[dict[str, str]], scorer) -> tuple[np.ndarray, np.ndarray]:
    word_references = [scorer.raw_wer_text(row["reference_raw"]) for row in rows]
    word_hypotheses = [scorer.raw_wer_text(row["hypothesis"]) for row in rows]
    character_references = [scorer.raw_cer_text(row["reference_raw"]) for row in rows]
    character_hypotheses = [scorer.raw_cer_text(row["hypothesis"]) for row in rows]
    words = scorer.jiwer.process_words(word_references, word_hypotheses)
    characters = scorer.jiwer.process_characters(
        character_references, character_hypotheses
    )
    return (
        np.asarray(
            [words.substitutions, words.deletions, words.insertions, words.hits],
            dtype=np.int64,
        ),
        np.asarray(
            [
                characters.substitutions,
                characters.deletions,
                characters.insertions,
                characters.hits,
            ],
            dtype=np.int64,
        ),
    )


def cluster_bootstrap(
    rows: list[dict[str, str]],
    scorer,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    clusters: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        clusters.setdefault(row["speaker_key"], []).append(row)
    keys = sorted(clusters)
    words = []
    characters = []
    for key in keys:
        word_counts, character_counts = _error_counts(clusters[key], scorer)
        words.append(word_counts)
        characters.append(character_counts)
    word_array = np.stack(words)
    character_array = np.stack(characters)
    point = scorer.score_texts(
        [row["reference_raw"] for row in rows],
        [row["hypothesis"] for row in rows],
    )
    total_words = word_array.sum(axis=0)
    total_characters = character_array.sum(axis=0)
    check_wer = float(
        total_words[:3].sum() / (total_words[0] + total_words[1] + total_words[3])
    )
    check_cer = float(
        total_characters[:3].sum()
        / (total_characters[0] + total_characters[1] + total_characters[3])
    )
    if not (
        math.isclose(check_wer, float(point["wer"]), abs_tol=1e-12)
        and math.isclose(check_cer, float(point["cer"]), abs_tol=1e-12)
    ):
        raise RuntimeError("bootstrap sufficient statistics disagree with scorer")

    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 3), dtype=np.float64)
    probabilities = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    for start in range(0, replicates, 2_000):
        stop = min(start + 2_000, replicates)
        weights = rng.multinomial(len(keys), probabilities, size=stop - start)
        sampled_words = weights @ word_array
        sampled_characters = weights @ character_array
        word_denominator = (
            sampled_words[:, 0] + sampled_words[:, 1] + sampled_words[:, 3]
        )
        character_denominator = (
            sampled_characters[:, 0]
            + sampled_characters[:, 1]
            + sampled_characters[:, 3]
        )
        wer = sampled_words[:, :3].sum(axis=1) / word_denominator
        cer = sampled_characters[:, :3].sum(axis=1) / character_denominator
        values[start:stop, 0] = wer
        values[start:stop, 1] = cer
        values[start:stop, 2] = 0.5 * (wer + cer)
    percentiles = {
        metric: {
            "2.5": float(np.quantile(values[:, index], 0.025)),
            "50": float(np.quantile(values[:, index], 0.5)),
            "97.5": float(np.quantile(values[:, index], 0.975)),
        }
        for index, metric in enumerate(("wer", "cer", "weighted_error"))
    }
    return {
        "clusters": len(keys),
        "replicates": replicates,
        "seed": seed,
        "raw_percentiles": percentiles,
        "raw_q_percentiles": {
            "2.5": 1.0 - percentiles["weighted_error"]["97.5"],
            "50": 1.0 - percentiles["weighted_error"]["50"],
            "97.5": 1.0 - percentiles["weighted_error"]["2.5"],
        },
    }


def _bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid serialized boolean: {value!r}")
    return value == "True"


def validated_analysis_run_dir(
    predictions: Path,
    experiment_root: Path,
) -> Path:
    """Accept direct runs and a run's create-only derived evaluation lane."""

    predictions = predictions.resolve()
    experiment_root = experiment_root.resolve()
    run_dir = predictions.parent
    if predictions != run_dir / "predictions.csv":
        raise RuntimeError("analysis input must be named predictions.csv")
    direct = run_dir.parent == experiment_root / "runs"
    source_run = (
        run_dir.parent.parent.parent
        if run_dir.name == "evaluation" and run_dir.parent.parent.name == "derived"
        else None
    )
    derived = (
        source_run is not None
        and source_run.parent == experiment_root / "runs"
        and (source_run / "config.json").is_file()
        and (run_dir.parent / "model" / "average_manifest.json").is_file()
    )
    if not direct and not derived:
        raise RuntimeError(
            "predictions must be runs/<run-id>/predictions.csv or "
            "runs/<run-id>/derived/<candidate>/evaluation/predictions.csv"
        )
    return run_dir


def source_run_for_analysis(run_dir: Path, experiment_root: Path) -> Path:
    if run_dir.parent == experiment_root / "runs":
        return run_dir
    source_run = run_dir.parent.parent.parent
    if (
        run_dir.name != "evaluation"
        or run_dir.parent.parent.name != "derived"
        or source_run.parent != experiment_root / "runs"
    ):
        raise RuntimeError("cannot resolve analysis source run")
    return source_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--scorer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    predictions = args.predictions.resolve()
    output = args.output.resolve()
    scorer_path = args.scorer.resolve()
    experiment_root = experiment_root_from(predictions)
    run_dir = validated_analysis_run_dir(predictions, experiment_root)
    source_run = source_run_for_analysis(run_dir, experiment_root)
    source_config = read_json(source_run / "config.json")
    expected_scorer = resolve_uri(str(source_config["scorer_path"]), experiment_root)
    if scorer_path != expected_scorer:
        raise RuntimeError("analysis must use the source run's hash-bound scorer")
    if args.scorer_sha256 != str(source_config["scorer_sha256"]):
        raise RuntimeError("analysis scorer hash must match the source run config")
    if output != run_dir / "analysis.json":
        raise RuntimeError("analysis output must be <run-dir>/analysis.json")
    checksum_path = run_dir / "ANALYSIS_SHA256SUMS"
    if output.exists() or checksum_path.exists():
        raise RuntimeError("create-only analysis output exists")
    if args.replicates != 50_000 or args.seed != 20260730:
        raise RuntimeError("canonical bootstrap contract drift")
    rows = read_predictions(predictions)
    scorer = load_frozen_scorer(scorer_path, args.scorer_sha256)
    result = {
        "schema_version": 1,
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "overall": score(rows, scorer),
        "strata": {
            name: score(selected, scorer)
            for name, selected in {
                "warm": [row for row in rows if row["stratum"] == "warm"],
                "cold": [row for row in rows if row["stratum"] == "cold"],
                "original_train": [
                    row for row in rows if row["original_split"] == "train"
                ],
                "original_validation": [
                    row for row in rows if row["original_split"] == "validation"
                ],
                "phase2_speaker": [
                    row for row in rows if _bool(row["is_phase2_test_speaker"])
                ],
                "non_phase2_speaker": [
                    row for row in rows if not _bool(row["is_phase2_test_speaker"])
                ],
                "phase2_prompt": [
                    row for row in rows if _bool(row["is_phase2_test_prompt"])
                ],
                "non_phase2_prompt": [
                    row for row in rows if not _bool(row["is_phase2_test_prompt"])
                ],
            }.items()
            if selected
        },
        "worst_utterances": worst_utterances(rows, scorer),
        "speaker_cluster_bootstrap": cluster_bootstrap(
            rows,
            scorer,
            replicates=args.replicates,
            seed=args.seed,
        ),
    }
    write_json_create_only(output, result)
    with checksum_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(output)}  analysis.json\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
