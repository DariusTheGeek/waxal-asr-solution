#!/usr/bin/env python3
"""Build fixed strongest-first conservative word ROVER artifacts.

The rule is reference-free during construction: the first input is the pivot,
insertions are never adopted, and ties retain the pivot surface.  Validation
references are used only afterwards for canonical scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
import unicodedata

import polars as pl


def repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "scoring").is_dir():
            return candidate
    raise RuntimeError("unable to locate WAXAL3 root")


ROOT = repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))
from scoring.asr import raw_text, score_texts, score_weighted_texts  # noqa: E402


TOKEN_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
EXPECTED_LANGUAGE_COUNTS = {"lin": 447, "sna": 445}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_LANGUAGE_COUNTS.values())


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def resolve_repo_file(value: Path | str, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes WAXAL3: {value}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {resolved}")
    return resolved


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def token_key(token: str) -> str:
    return TOKEN_EDGE.sub("", token.casefold())


def conservative_word_rover(hypotheses: list[str]) -> str:
    """Equal-weight word ROVER with first-member pivot and no insertions."""

    if len(hypotheses) != 3:
        raise RuntimeError("fixed word ROVER requires exactly three hypotheses")
    tokenized = [normalize_text(hypothesis).split() for hypothesis in hypotheses]
    pivot = tokenized[0]
    pivot_keys = [token_key(token) for token in pivot]
    votes: list[dict[str, float]] = [defaultdict(float) for _ in pivot]
    surfaces: list[dict[str, tuple[float, str]]] = [dict() for _ in pivot]
    for member_index, tokens in enumerate(tokenized):
        keys = [token_key(token) for token in tokens]
        matcher = difflib.SequenceMatcher(None, pivot_keys, keys)
        for tag, start_pivot, end_pivot, start_member, end_member in matcher.get_opcodes():
            same_width = (end_pivot - start_pivot) == (end_member - start_member)
            if tag == "equal" or (tag == "replace" and same_width):
                for offset in range(end_pivot - start_pivot):
                    position = start_pivot + offset
                    token = tokens[start_member + offset]
                    key = keys[start_member + offset]
                    if not key:
                        continue
                    votes[position][key] += 1.0
                    preference = 100.0 if member_index == 0 else 1.0
                    previous = surfaces[position].get(key)
                    if previous is None or preference > previous[0]:
                        surfaces[position][key] = (preference, token)
            elif tag == "delete":
                for position in range(start_pivot, end_pivot):
                    votes[position][""] += 1.0
            # An insertion is intentionally ignored.
    output: list[str] = []
    for position, pivot_token in enumerate(pivot):
        pivot_key = pivot_keys[position]
        if not votes[position]:
            output.append(pivot_token)
            continue
        winner, winner_votes = max(
            votes[position].items(), key=lambda item: (item[1], item[0] == pivot_key)
        )
        if winner != pivot_key and winner_votes > votes[position].get(pivot_key, 0.0):
            if winner:
                output.append(surfaces[position][winner][1])
        else:
            output.append(pivot_token)
    return normalize_text(" ".join(output))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_member_labels(labels: list[str]) -> list[str]:
    if len(labels) != 3 or len(set(labels)) != 3:
        raise RuntimeError("exactly three unique ordered member labels are required")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", label) for label in labels):
        raise RuntimeError("member labels contain unsafe characters")
    return labels


def validation(args: argparse.Namespace, paths: list[Path], labels: list[str]) -> int:
    required = {
        "row_key",
        "manifest_index",
        "id",
        "transcription_nfc",
        "target_weight",
        "hypothesis",
    }
    frames = [pl.read_parquet(path).sort("manifest_index") for path in paths]
    for path, frame in zip(paths, frames, strict=True):
        if missing := required - set(frame.columns):
            raise RuntimeError(f"validation prediction schema drift {path}: {sorted(missing)}")
        if frame.height != args.expected_rows or frame["row_key"].n_unique() != frame.height:
            raise RuntimeError(f"validation prediction row-key drift: {path}")
    anchor = frames[0]
    keys = anchor["row_key"].to_list()
    metadata = anchor.select(
        "row_key", "manifest_index", "id", "transcription_nfc", "target_weight"
    )
    references = anchor["transcription_nfc"].fill_null("").to_list()
    weights = anchor["target_weight"].cast(pl.Float64).to_list()
    hypotheses = [frame["hypothesis"].fill_null("").to_list() for frame in frames]
    for path, frame, member in zip(paths[1:], frames[1:], hypotheses[1:], strict=True):
        if frame["row_key"].to_list() != keys:
            raise RuntimeError(f"validation row-key alignment drift: {path}")
        if frame["transcription_nfc"].fill_null("").to_list() != references:
            raise RuntimeError(f"validation reference alignment drift: {path}")
        if frame["target_weight"].cast(pl.Float64).to_list() != weights:
            raise RuntimeError(f"validation target-weight alignment drift: {path}")
        if len(member) != len(keys):
            raise RuntimeError(f"validation member row count drift: {path}")
    rover = [conservative_word_rover(values) for values in zip(*hypotheses, strict=True)]
    raw_score = score_texts(references, rover)
    weighted_score = score_weighted_texts(references, rover, weights)
    ledger = metadata.with_columns(
        [pl.Series(f"{label}_hypothesis", member) for label, member in zip(labels, hypotheses, strict=True)]
        + [pl.Series("rover_hypothesis", rover)]
    )

    output = args.output
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        prediction_frame = anchor.with_columns(pl.Series("hypothesis", rover))
        parquet_path = temporary / "predictions.parquet"
        csv_path = temporary / "predictions.csv"
        ledger_parquet = temporary / "decision_ledger.parquet"
        ledger_csv = temporary / "decision_ledger.csv"
        prediction_frame.write_parquet(parquet_path, compression="zstd")
        prediction_frame.select("id", "hypothesis").rename(
            {"id": "ID", "hypothesis": "Target"}
        ).write_csv(csv_path)
        ledger.write_parquet(ledger_parquet, compression="zstd")
        ledger.write_csv(ledger_csv)
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "fixed_word_rover_validation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "uniform_equal_weight_conservative_word_rover_strongest_first_no_insertions",
            "construction_is_reference_free": True,
            "member_order_is_tie_break_order": True,
            "members": [
                {"label": label, "predictions": relative(path), "sha256": sha256_file(path)}
                for label, path in zip(labels, paths, strict=True)
            ],
            "rows": len(rover),
            "raw": raw_score,
            "target_weighted": weighted_score,
            "blank_rows": sum(not raw_text(value) for value in rover),
            "changed_rows_vs_pivot": sum(
                value != hypotheses[0][index] for index, value in enumerate(rover)
            ),
            "predictions": {
                "parquet": relative(output / "predictions.parquet"),
                "parquet_sha256": sha256_file(parquet_path),
                "csv": relative(output / "predictions.csv"),
                "csv_sha256": sha256_file(csv_path),
            },
            "decision_ledger": {
                "parquet": relative(output / "decision_ledger.parquet"),
                "parquet_sha256": sha256_file(ledger_parquet),
                "csv": relative(output / "decision_ledger.csv"),
                "csv_sha256": sha256_file(ledger_csv),
            },
            "implementation": relative(Path(__file__)),
            "implementation_sha256": sha256_file(Path(__file__)),
            "external_action": False,
            "submission_created": False,
        }
        write_json(temporary / "summary.json", summary)
        write_checksums(temporary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def phase2_ids(master: Path, language: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = pq.read_table(master, columns=["official_order", "id", "language"]).to_pylist()
    rows.sort(key=lambda row: int(row["official_order"]))
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("corrected Phase-2 row count drift")
    if [int(row["official_order"]) for row in rows] != list(range(EXPECTED_TOTAL_ROWS)):
        raise RuntimeError("corrected Phase-2 official-order drift")
    route = [row for row in rows if str(row["language"]) == language]
    if len(route) != EXPECTED_LANGUAGE_COUNTS[language]:
        raise RuntimeError("corrected Phase-2 language-route drift")
    if len({str(row["id"]) for row in route}) != len(route):
        raise RuntimeError("corrected Phase-2 route ID uniqueness drift")
    return route


def read_phase2_predictions(path: Path, expected_ids: list[str]) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Target"]:
            raise RuntimeError(f"unsupported Phase-2 prediction schema: {path}")
        rows = list(reader)
    identifiers = [str(row["ID"]) for row in rows]
    if identifiers != expected_ids or len(set(identifiers)) != len(expected_ids):
        raise RuntimeError(f"Phase-2 ID/order drift: {path}")
    values = {identifier: normalize_text(row["Target"]) for identifier, row in zip(identifiers, rows, strict=True)}
    if any(not values[identifier] for identifier in expected_ids):
        raise RuntimeError(f"blank Phase-2 member hypothesis: {path}")
    return values


def phase2(args: argparse.Namespace, paths: list[Path], labels: list[str]) -> int:
    if args.test_master is None:
        raise RuntimeError("--test-master is required for Phase-2 ROVER")
    master = resolve_repo_file(args.test_master, label="test master")
    route = phase2_ids(master, args.language)
    identifiers = [str(row["id"]) for row in route]
    members = [read_phase2_predictions(path, identifiers) for path in paths]
    rover = [
        conservative_word_rover([member[identifier] for member in members])
        for identifier in identifiers
    ]
    if any(not value for value in rover):
        raise RuntimeError("fixed word ROVER generated a blank Phase-2 hypothesis")
    ledger_rows: list[dict[str, Any]] = []
    for row, hypothesis in zip(route, rover, strict=True):
        identifier = str(row["id"])
        ledger_row: dict[str, Any] = {
            "official_order": int(row["official_order"]),
            "ID": identifier,
            "rover_target": hypothesis,
            "changed_vs_pivot": hypothesis != members[0][identifier],
        }
        for label, member in zip(labels, members, strict=True):
            ledger_row[f"{label}_hypothesis"] = member[identifier]
        ledger_rows.append(ledger_row)

    output = args.output
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        predictions_path = temporary / "predictions.raw.csv"
        ledger_path = temporary / "row_source_ledger.csv"
        write_csv(
            predictions_path,
            ["ID", "Target"],
            [
                {"ID": identifier, "Target": hypothesis}
                for identifier, hypothesis in zip(identifiers, rover, strict=True)
            ],
        )
        write_csv(ledger_path, list(ledger_rows[0]), ledger_rows)
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "fixed_word_rover_phase2_candidate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "uniform_equal_weight_conservative_word_rover_strongest_first_no_insertions",
            "construction_is_reference_free": True,
            "member_order_is_tie_break_order": True,
            "language": args.language,
            "rows": len(rover),
            "members": [
                {"label": label, "predictions": relative(path), "sha256": sha256_file(path)}
                for label, path in zip(labels, paths, strict=True)
            ],
            "test_master": relative(master),
            "test_master_sha256": sha256_file(master),
            "predictions": {
                "path": relative(output / "predictions.raw.csv"),
                "sha256": sha256_file(predictions_path),
            },
            "row_source_ledger": {
                "path": relative(output / "row_source_ledger.csv"),
                "sha256": sha256_file(ledger_path),
            },
            "changed_rows_vs_pivot": sum(
                value != members[0][identifier]
                for identifier, value in zip(identifiers, rover, strict=True)
            ),
            "implementation": relative(Path(__file__)),
            "implementation_sha256": sha256_file(Path(__file__)),
            "external_action": False,
            "submission_created": False,
        }
        write_json(temporary / "summary.json", summary)
        write_checksums(temporary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def write_checksums(directory: Path) -> None:
    checksum = directory / "SHA256SUMS"
    with checksum.open("x", encoding="utf-8") as handle:
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            if path.name == checksum.name:
                continue
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validation", "phase2"), required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--member-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=900)
    parser.add_argument("--test-master", type=Path)
    parser.add_argument("--language", choices=sorted(EXPECTED_LANGUAGE_COUNTS), default="lin")
    args = parser.parse_args()
    if len(args.input) != 3:
        raise RuntimeError("exactly three ordered input files are required")
    labels = check_member_labels(args.member_label)
    paths = [resolve_repo_file(path, label="ROVER input") for path in args.input]
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    args.output = output.resolve()
    try:
        args.output.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError("ROVER output escapes WAXAL3") from error
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "validation":
        return validation(args, paths, labels)
    return phase2(args, paths, labels)


if __name__ == "__main__":
    raise SystemExit(main())
