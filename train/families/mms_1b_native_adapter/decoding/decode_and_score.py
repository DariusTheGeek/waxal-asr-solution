#!/usr/bin/env python3
"""Decode frozen MMS log probabilities and select fold-safe text reconstruction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import jiwer
import numpy as np
import pyarrow.parquet as pq
from pyctcdecode import build_ctcdecoder
import yaml


MODEL_FAMILY = Path(__file__).resolve().parents[1]
PACKET_SRC = MODEL_FAMILY.parent
sys.path.insert(0, str(MODEL_FAMILY))
sys.path.insert(0, str(PACKET_SRC))

import scorer_compat as scorer  # noqa: E402
from decoding.beam_postprocess import (  # noqa: E402
    POLICIES,
    PostprocessAssets,
    apply_policy,
    labels_from_vocab,
    normalize_text,
)
from common.hashing import sha256_file, write_json_atomic  # noqa: E402
from common.packet import verify as verify_packet  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes WAXAL3: {value}") from exc
    return path


def checked_file(root: Path, item: dict[str, Any]) -> Path:
    path = resolve(root, str(item["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != str(item["sha256"]):
        raise RuntimeError(f"file hash drift: {path}: {observed}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def write_csv_create(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(
    root: Path, profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    path = checked_file(root, profile["validation_manifest"])
    all_rows = pq.read_table(path).to_pylist()
    train = [
        dict(row)
        for row in all_rows
        if str(row["language"]) == "lin"
        and str(row["assignment"]) == "train"
        and bool(row["selected_for_training"])
    ]
    if len(train) != int(profile["expected_train_rows"]):
        raise RuntimeError("post-processing training row-count drift")
    validation = [
        dict(row)
        for row in all_rows
        if str(row["language"]) == "lin"
        and str(row["assignment"]) == "validation_scored"
    ]
    validation.sort(key=lambda row: int(row["evaluation_index"]))
    if len(validation) != int(profile["expected_validation_rows"]):
        raise RuntimeError("validation row-count drift")
    limit = profile.get("validation_limit")
    if limit is not None:
        validation = validation[: int(limit)]
    return validation, [str(row["target_raw"]) for row in train]


def load_test_rows(root: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    path = checked_file(root, profile["test_master"])
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: int(row["official_order"]))
    if len(rows) != int(profile["expected_test_rows"]):
        raise RuntimeError("corrected-test row-count drift")
    selected = [dict(row) for row in rows if str(row["language"]) == "lin"]
    if len(selected) != int(profile["expected_test_lingala_rows"]):
        raise RuntimeError("corrected-test Lingala count drift")
    limit = profile.get("test_limit")
    return selected if limit is None else selected[: int(limit)]


def load_capture(
    capture: Path,
    *,
    packet_record: dict[str, Any],
    profile_path: Path,
    split: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray], list[dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    matrices: dict[int, np.ndarray] = {}
    terminals: list[dict[str, Any]] = []
    expected_profile_sha = sha256_file(profile_path)
    for rank in range(4):
        rank_dir = capture / f"rank_{rank}"
        terminal_path = rank_dir / "TERMINAL.json"
        terminal = read_json(terminal_path)
        required = {
            "status": "PASS",
            "rank": rank,
            "world_size": 4,
            "split": split,
            "packet_digest": packet_record["content_digest"],
            "profile_sha256": expected_profile_sha,
        }
        if any(terminal.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"capture terminal drift at rank {rank}")
        records_path = rank_dir / str(terminal["records"])
        logits_path = rank_dir / str(terminal["log_probabilities"])
        if (
            sha256_file(records_path) != terminal["records_sha256"]
            or sha256_file(logits_path) != terminal["log_probabilities_sha256"]
        ):
            raise RuntimeError(f"capture payload hash drift at rank {rank}")
        with records_path.open(encoding="utf-8") as handle:
            rank_records = [json.loads(line) for line in handle if line.strip()]
        with np.load(logits_path, allow_pickle=False) as archive:
            for row in rank_records:
                index = int(row["capture_index"])
                if index in records or index % 4 != rank:
                    raise RuntimeError("capture index duplication/rank drift")
                key = str(row["array_key"])
                matrix = archive[key]
                if (
                    matrix.dtype != np.float16
                    or matrix.ndim != 2
                    or matrix.shape[0] != int(row["output_frames"])
                    or not np.isfinite(matrix).all()
                ):
                    raise RuntimeError(f"invalid captured logits: {row['id']}")
                records[index] = row
                matrices[index] = matrix
        terminals.append(terminal)
    return records, matrices, terminals


def decode_beam64(
    matrices: list[np.ndarray], labels: list[str], processes: int
) -> list[str]:
    decoder = build_ctcdecoder(labels)
    if processes <= 1 or len(matrices) == 1:
        return [
            normalize_text(decoder.decode(matrix.astype(np.float32), beam_width=64))
            for matrix in matrices
        ]
    context = mp.get_context("fork")
    with context.Pool(processes=min(processes, len(matrices))) as pool:
        values = decoder.decode_batch(
            pool,
            [matrix.astype(np.float32) for matrix in matrices],
            beam_width=64,
        )
    return [normalize_text(value) for value in values]


def score_variant(rows: list[dict[str, Any]], hypotheses: list[str]) -> dict[str, Any]:
    references = [str(row["reference_raw"]) for row in rows]
    weights = [float(row["target_weight"]) for row in rows]
    raw = scorer.score_texts(references, hypotheses)
    target = scorer.score_weighted_texts(references, hypotheses, weights)
    return {
        "rows": len(rows),
        "raw": raw,
        "target_weighted_raw": target,
        "blank_rows": sum(not hypothesis for hypothesis in hypotheses),
    }


def _counts(reference: str, hypothesis: str) -> np.ndarray:
    words = jiwer.process_words(
        scorer.raw_wer_text(reference), scorer.raw_wer_text(hypothesis)
    )
    characters = jiwer.process_characters(
        scorer.raw_cer_text(reference), scorer.raw_cer_text(hypothesis)
    )
    return np.asarray(
        [
            words.substitutions + words.deletions + words.insertions,
            words.hits + words.substitutions + words.deletions,
            characters.substitutions + characters.deletions + characters.insertions,
            characters.hits + characters.substitutions + characters.deletions,
        ],
        dtype=np.float64,
    )


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]],
    baseline: list[str],
    candidate: list[str],
    *,
    cluster_key: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        weight = float(row["target_weight"])
        if weight > 0.0:
            clusters.setdefault(str(row[cluster_key]), []).append(index)
    keys = sorted(clusters)
    if not keys or any(not key for key in keys):
        raise RuntimeError(f"invalid bootstrap clusters: {cluster_key}")
    arrays: list[np.ndarray] = []
    for hypotheses in (baseline, candidate):
        cluster_counts: list[np.ndarray] = []
        for key in keys:
            value = np.zeros(4, dtype=np.float64)
            for index in clusters[key]:
                value += float(rows[index]["target_weight"]) * _counts(
                    str(rows[index]["reference_raw"]), hypotheses[index]
                )
            cluster_counts.append(value)
        arrays.append(np.stack(cluster_counts))

    def q(counts: np.ndarray) -> np.ndarray:
        if np.any(counts[..., 1] <= 0) or np.any(counts[..., 3] <= 0):
            raise RuntimeError("invalid bootstrap reference denominator")
        return 1.0 - 0.5 * (
            counts[..., 0] / counts[..., 1]
            + counts[..., 2] / counts[..., 3]
        )

    points = [float(q(array.sum(axis=0))) for array in arrays]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    probabilities = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    for start in range(0, replicates, 2_000):
        stop = min(start + 2_000, replicates)
        sampled = rng.multinomial(len(keys), probabilities, size=stop - start)
        deltas[start:stop] = q(sampled @ arrays[1]) - q(sampled @ arrays[0])
    return {
        "cluster_key": cluster_key,
        "clusters": len(keys),
        "replicates": replicates,
        "seed": seed,
        "baseline_target_q": points[0],
        "candidate_target_q": points[1],
        "delta_target_q": points[1] - points[0],
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "p_delta_positive": float(np.mean(deltas > 0.0)),
    }


def validation_mode(
    *,
    root: Path,
    profile: dict[str, Any],
    records: dict[int, dict[str, Any]],
    matrices: dict[int, np.ndarray],
    vocabulary: dict[str, int],
    assets: PostprocessAssets,
    output: Path,
) -> dict[str, Any]:
    rows, _ = load_manifest_rows(root, profile)
    indices = [int(row["evaluation_index"]) for row in rows]
    if set(indices) != set(records) or set(indices) != set(matrices):
        raise RuntimeError("validation capture coverage drift")
    baseline_rows = read_csv(checked_file(root, profile["baseline_validation"]))
    baseline_by_id = {row["id"]: row["hypothesis"] for row in baseline_rows}
    greedy = []
    greedy_fp16_changes = 0
    for row in rows:
        index = int(row["evaluation_index"])
        capture = records[index]
        if capture["id"] != row["id"]:
            raise RuntimeError("validation capture ID drift")
        value = normalize_text(capture["greedy_hypothesis"])
        if value != normalize_text(baseline_by_id[str(row["id"])]):
            raise RuntimeError(f"source greedy fails frozen baseline: {row['id']}")
        greedy.append(value)
        greedy_fp16_changes += value != normalize_text(
            capture["greedy_hypothesis_fp16"]
        )
    beam = decode_beam64(
        [matrices[index] for index in indices],
        labels_from_vocab(vocabulary),
        int(profile["decoder_processes"]),
    )
    sources = {"greedy": greedy, "beam64": beam}
    scores: dict[str, dict[str, Any]] = {}
    hypotheses_by_variant: dict[str, list[str]] = {}
    variants_dir = output / "variants"
    variants_dir.mkdir()
    fieldnames = [
        "evaluation_index",
        "id",
        "speaker_key",
        "stratum",
        "target_weight",
        "slot_id",
        "target_phase2_id",
        "reference_raw",
        "hypothesis",
    ]
    for decoder_name, base in sources.items():
        for policy in POLICIES:
            variant = f"{decoder_name}__{policy}"
            hypotheses = [apply_policy(value, policy, assets) for value in base]
            hypotheses_by_variant[variant] = hypotheses
            scores[variant] = score_variant(rows, hypotheses)
            write_csv_create(
                variants_dir / f"{variant}.csv",
                fieldnames,
                [
                    {
                        "evaluation_index": row["evaluation_index"],
                        "id": row["id"],
                        "speaker_key": row["speaker_key"],
                        "stratum": row["stratum"],
                        "target_weight": row["target_weight"],
                        "slot_id": row["slot_id"],
                        "target_phase2_id": row["target_phase2_id"],
                        "reference_raw": row["reference_raw"],
                        "hypothesis": hypothesis,
                    }
                    for row, hypothesis in zip(rows, hypotheses, strict=True)
                ],
            )
    winner = max(
        scores,
        key=lambda name: (
            float(scores[name]["target_weighted_raw"]["q"]),
            float(scores[name]["raw"]["q"]),
            name,
        ),
    )
    baseline_name = "greedy__sentence_case"
    winner_hypotheses = hypotheses_by_variant[winner]
    baseline_hypotheses = hypotheses_by_variant[baseline_name]
    wins = ties = losses = 0
    for row, base, candidate in zip(
        rows, baseline_hypotheses, winner_hypotheses, strict=True
    ):
        base_q = scorer.score_texts([str(row["reference_raw"])], [base])["q"]
        candidate_q = scorer.score_texts(
            [str(row["reference_raw"])], [candidate]
        )["q"]
        if candidate_q > base_q:
            wins += 1
        elif candidate_q < base_q:
            losses += 1
        else:
            ties += 1
    replicates = int(profile["bootstrap_replicates"])
    selection = {
        "schema_version": 1,
        "selection_authority": "maximum target-slot-weighted canonical raw Q",
        "tie_break": "unweighted canonical raw Q, then stable variant name",
        "winner": winner,
        "winner_decoder": winner.split("__", 1)[0],
        "winner_policy": winner.split("__", 1)[1],
        "winner_scores": scores[winner],
        "baseline": baseline_name,
        "baseline_scores": scores[baseline_name],
        "delta_target_q": float(scores[winner]["target_weighted_raw"]["q"])
        - float(scores[baseline_name]["target_weighted_raw"]["q"]),
        "exact_waxal2_variant": "beam64__waxal2_terminal_truecase",
        "exact_waxal2_scores": scores["beam64__waxal2_terminal_truecase"],
        "row_win_tie_loss_vs_baseline": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "target_id_bootstrap": paired_cluster_bootstrap(
            rows,
            baseline_hypotheses,
            winner_hypotheses,
            cluster_key="target_phase2_id",
            replicates=replicates,
            seed=20260803,
        ),
        "speaker_bootstrap": paired_cluster_bootstrap(
            rows,
            baseline_hypotheses,
            winner_hypotheses,
            cluster_key="speaker_key",
            replicates=replicates,
            seed=20260804,
        ),
        "fp16_greedy_hypothesis_changes": greedy_fp16_changes,
        "postprocess_assets": assets.provenance(),
    }
    write_json_atomic(output / "scores.json", scores)
    write_json_atomic(output / "selection.json", selection)
    return selection


def test_mode(
    *,
    root: Path,
    profile: dict[str, Any],
    records: dict[int, dict[str, Any]],
    matrices: dict[int, np.ndarray],
    vocabulary: dict[str, int],
    assets: PostprocessAssets,
    selection_path: Path,
    output: Path,
) -> dict[str, Any]:
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = read_json(selection_path)
    decoder_name = str(selection["winner_decoder"])
    policy = str(selection["winner_policy"])
    if decoder_name not in {"greedy", "beam64"} or policy not in POLICIES:
        raise RuntimeError("invalid frozen validation selection")
    rows = load_test_rows(root, profile)
    indices = [int(row["official_order"]) for row in rows]
    if set(indices) != set(records) or set(indices) != set(matrices):
        raise RuntimeError("test capture coverage drift")
    greedy: list[str] = []
    for row in rows:
        record = records[int(row["official_order"])]
        if record["id"] != row["id"]:
            raise RuntimeError("test capture ID drift")
        greedy.append(normalize_text(record["greedy_hypothesis"]))
    base = (
        greedy
        if decoder_name == "greedy"
        else decode_beam64(
            [matrices[index] for index in indices],
            labels_from_vocab(vocabulary),
            int(profile["decoder_processes"]),
        )
    )
    hypotheses: list[str] = []
    fallback_rows: list[str] = []
    for row, decoded, greedy_value in zip(rows, base, greedy, strict=True):
        hypothesis = apply_policy(decoded, policy, assets)
        if not hypothesis:
            hypothesis = apply_policy(greedy_value, policy, assets)
            fallback_rows.append(str(row["id"]))
        if not hypothesis:
            hypothesis = "A"
        hypotheses.append(hypothesis)
    prediction_rows = [
        {
            "official_order": row["official_order"],
            "id": row["id"],
            "language": "lin",
            "decoder": decoder_name,
            "policy": policy,
            "hypothesis": hypothesis,
            "blank_fallback": str(row["id"]) in fallback_rows,
        }
        for row, hypothesis in zip(rows, hypotheses, strict=True)
    ]
    write_csv_create(
        output / "predictions.csv",
        [
            "official_order",
            "id",
            "language",
            "decoder",
            "policy",
            "hypothesis",
            "blank_fallback",
        ],
        prediction_rows,
    )
    submission_rows = [
        {"ID": row["id"], "Target": hypothesis}
        for row, hypothesis in zip(rows, hypotheses, strict=True)
    ]
    write_csv_create(output / "lingala_submission.csv", ["ID", "Target"], submission_rows)
    return {
        "schema_version": 1,
        "validation_selection": str(selection_path),
        "validation_selection_sha256": sha256_file(selection_path),
        "decoder": decoder_name,
        "beam_width": 64 if decoder_name == "beam64" else None,
        "language_model": None,
        "hotwords": None,
        "policy": policy,
        "rows": len(rows),
        "blank_fallback_rows": fallback_rows,
        "prediction_sha256": sha256_file(output / "predictions.csv"),
        "lingala_submission_sha256": sha256_file(output / "lingala_submission.csv"),
        "postprocess_assets": assets.provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    packet = args.packet.resolve()
    profile_path = args.profile.resolve()
    capture = args.capture.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    packet_record = read_json(packet / "PACKET.json")
    packet_check = verify_packet(packet)
    if packet_check["status"] != "PASS":
        raise RuntimeError(f"packet verification failed: {packet_check}")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if profile.get("experiment_id") != packet_record["experiment_id"]:
        raise RuntimeError("profile/packet identity drift")
    vocabulary = {
        str(key): int(value)
        for key, value in read_json(checked_file(root, profile["vocab"])).items()
    }
    labels_from_vocab(vocabulary)
    records, matrices, terminals = load_capture(
        capture,
        packet_record=packet_record,
        profile_path=profile_path,
        split=args.split,
    )
    _, train_texts = load_manifest_rows(root, profile)
    assets = PostprocessAssets.fit(train_texts)
    output.mkdir(parents=True)
    if args.split == "validation":
        if args.selection is not None:
            raise RuntimeError("validation decode must not receive --selection")
        result = validation_mode(
            root=root,
            profile=profile,
            records=records,
            matrices=matrices,
            vocabulary=vocabulary,
            assets=assets,
            output=output,
        )
    else:
        if args.selection is None:
            raise RuntimeError("test decode requires frozen --selection")
        result = test_mode(
            root=root,
            profile=profile,
            records=records,
            matrices=matrices,
            vocabulary=vocabulary,
            assets=assets,
            selection_path=args.selection.resolve(),
            output=output,
        )
    terminal = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "experiment_id": profile["experiment_id"],
        "run_id": args.run_id,
        "split": args.split,
        "profile": str(profile_path.relative_to(packet)),
        "profile_sha256": sha256_file(profile_path),
        "packet_digest": packet_record["content_digest"],
        "capture": str(capture),
        "capture_run_ids": sorted({str(item["run_id"]) for item in terminals}),
        "implementation": str(Path(__file__).relative_to(PACKET_SRC)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "decoder": "pyctcdecode==0.5.0",
        "external_action": False,
        "submission_created": False,
        "result": result,
    }
    write_json_atomic(output / "TERMINAL.json", terminal)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
