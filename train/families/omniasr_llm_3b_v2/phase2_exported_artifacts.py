#!/usr/bin/env python3
"""Prepare and materialize label-free Phase-2 artifacts for exported LLM models.

The inference implementation remains the frozen model-family ``infer.py``.
This companion only turns the immutable Phase-2 audio manifest into its portable
form and then validates/materializes rank-sharded hypotheses.  It deliberately
selects no transcript or solution columns from the Phase-2 master.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unicodedata

import pyarrow.parquet as pq


EXPECTED_LANGUAGE_COUNTS = {"lin": 447, "sna": 445}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_LANGUAGE_COUNTS.values())
EXPECTED_PARAMETERS = 4_380_578_432
# Immutable public routing metadata.  Pin this before reading any fields so a
# replaced local master cannot silently route a different Phase-2 set.
EXPECTED_PHASE2_MASTER_SHA256 = (
    "a6e34f2b1aff4d79a0353b1b880a4b2f10ec451527a22ca67841071f70fa280e"
)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_inside(root: Path, value: Path | str, *, kind: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{kind} escapes WAXAL3: {value}") from error
    return resolved


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def sentence_case(value: object) -> str:
    characters = list(normalize_text(value))
    for index, character in enumerate(characters):
        if character.isalpha():
            characters[index] = character.upper()
            break
    return "".join(characters)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def load_test_rows(root: Path, master: Path, language: str) -> list[dict[str, Any]]:
    """Read only immutable Phase-2 route/audio identity columns."""

    columns = [
        "official_order",
        "id",
        "language",
        "pool_id",
        "pool_split",
        "test_wav_path",
        "test_wav_bytes",
        "test_wav_sha256",
    ]
    if master.is_symlink() or not master.is_file():
        raise FileNotFoundError(master)
    if sha256_file(master) != EXPECTED_PHASE2_MASTER_SHA256:
        raise RuntimeError("corrected Phase-2 master SHA-256 drift")
    rows = pq.read_table(master, columns=columns).to_pylist()
    rows.sort(key=lambda row: int(row["official_order"]))
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"corrected Phase-2 row drift: {len(rows)}")
    if [int(row["official_order"]) for row in rows] != list(
        range(EXPECTED_TOTAL_ROWS)
    ):
        raise RuntimeError("corrected Phase-2 official-order drift")
    identifiers = [str(row["id"]) for row in rows]
    if len(set(identifiers)) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("corrected Phase-2 IDs are not unique")
    counts = {
        key: sum(str(row["language"]) == key for row in rows)
        for key in EXPECTED_LANGUAGE_COUNTS
    }
    if counts != EXPECTED_LANGUAGE_COUNTS:
        raise RuntimeError(f"corrected Phase-2 route-count drift: {counts}")
    selected = [row for row in rows if str(row["language"]) == language]
    if len(selected) != EXPECTED_LANGUAGE_COUNTS[language]:
        raise RuntimeError(f"{language} route coverage drift")
    for row in selected:
        audio = resolve_inside(root, str(row["test_wav_path"]), kind="test audio")
        if audio.is_symlink() or not audio.is_file():
            raise FileNotFoundError(f"missing or unsafe test audio: {row['id']}")
        if audio.stat().st_size != int(row["test_wav_bytes"]):
            raise RuntimeError(f"test audio byte-size drift: {row['id']}")
        row["resolved_audio_path"] = audio
    return selected


def prepare(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    master = resolve_inside(root, args.test_master, kind="test master")
    output = resolve_inside(root, args.output, kind="prepared output")
    if output.exists():
        raise FileExistsError(output)
    rows = load_test_rows(root, master, args.language)
    audio_roots = {Path(str(row["test_wav_path"])).parent.as_posix() for row in rows}
    if len(audio_roots) != 1:
        raise RuntimeError("Phase-2 route has multiple audio roots")
    audio_root = next(iter(audio_roots))
    for row in rows:
        if sha256_file(Path(row["resolved_audio_path"])) != str(
            row["test_wav_sha256"]
        ):
            raise RuntimeError(f"test audio SHA-256 drift: {row['id']}")

    output.mkdir(parents=True)
    manifest_rows = []
    for index, row in enumerate(rows):
        source_path = Path(str(row["test_wav_path"]))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise RuntimeError("unsafe Phase-2 audio relative path")
        if source_path.parent.as_posix() != audio_root:
            raise RuntimeError("Phase-2 audio-root relative-path drift")
        manifest_rows.append(
            {
                "row_key": f"phase2:{args.language}:{int(row['official_order'])}",
                "id": str(row["id"]),
                "manifest_index": index,
                "official_order": int(row["official_order"]),
                "pool_id": str(row["pool_id"]),
                "pool_split": str(row["pool_split"]),
                "derived_audio_relpath": source_path.name,
                "derived_audio_sha256": str(row["test_wav_sha256"]),
            }
        )
    import polars as pl  # noqa: PLC0415

    manifest_path = output / f"phase2_{args.language}.rows.parquet"
    pl.DataFrame(manifest_rows).write_parquet(manifest_path, compression="zstd")
    tsv_path = output / f"phase2_{args.language}.tsv"
    tsv_path.write_text(f"{audio_root}\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "language": args.language,
        "rows": len(manifest_rows),
        "test_master": relative(root, master),
        "test_master_sha256": sha256_file(master),
        "audio_root": audio_root,
        "audio_identity_verified": "sha256_each_route_audio_once",
        "manifest": relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tsv": relative(root, tsv_path),
        "tsv_sha256": sha256_file(tsv_path),
        "official_orders_sha256": hashlib.sha256(
            json.dumps([row["official_order"] for row in manifest_rows]).encode()
        ).hexdigest(),
        "external_action": False,
        "submission_created": False,
    }
    write_json_atomic(output / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def materialize(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    prepared = resolve_inside(root, args.prepared, kind="prepared manifest")
    rank_dir = resolve_inside(root, args.rank_dir, kind="rank output")
    output = resolve_inside(root, args.output, kind="materialized output")
    if output.exists():
        raise FileExistsError(output)
    if args.expected_world_size != 8:
        raise RuntimeError("exported LLM Phase-2 inference requires eight ranks")
    if args.max_rows is not None and args.max_rows < args.expected_world_size:
        raise RuntimeError("Phase-2 smoke must exercise every rank")
    manifest_summary_path = prepared / "manifest.json"
    manifest_summary = read_json(manifest_summary_path)
    if (
        manifest_summary.get("status") != "PASS"
        or manifest_summary.get("language") != args.language
        or int(manifest_summary.get("rows", 0)) != EXPECTED_LANGUAGE_COUNTS[args.language]
    ):
        raise RuntimeError("prepared Phase-2 manifest identity drift")
    import polars as pl  # noqa: PLC0415

    manifest_path = prepared / f"phase2_{args.language}.rows.parquet"
    if sha256_file(manifest_path) != manifest_summary.get("manifest_sha256"):
        raise RuntimeError("prepared Phase-2 manifest SHA-256 drift")
    manifest = pl.read_parquet(manifest_path).sort("manifest_index")
    if manifest.height != EXPECTED_LANGUAGE_COUNTS[args.language]:
        raise RuntimeError("prepared Phase-2 manifest row-count drift")
    if args.max_rows is not None:
        manifest = manifest.head(args.max_rows)
    expected = manifest.to_dicts()
    expected_by_index = {int(row["manifest_index"]): row for row in expected}
    if len(expected_by_index) != len(expected):
        raise RuntimeError("prepared Phase-2 manifest index drift")

    captured: list[dict[str, Any]] = []
    rank_sources: list[dict[str, Any]] = []
    for rank in range(args.expected_world_size):
        predictions_path = rank_dir / f"rank_{rank}.jsonl"
        terminal_path = rank_dir / f"rank_{rank}.terminal.json"
        if predictions_path.is_symlink() or terminal_path.is_symlink():
            raise RuntimeError(f"unsafe rank artifact: {rank}")
        terminal = read_json(terminal_path)
        if (
            terminal.get("status") != "PASS"
            or int(terminal.get("rank", -1)) != rank
            or int(terminal.get("world_size", 0)) != args.expected_world_size
            or int(terminal.get("beam_size", 0)) != args.beam_size
            or int(terminal.get("model_parameters", 0)) != EXPECTED_PARAMETERS
            or terminal.get("predictions_sha256") != sha256_file(predictions_path)
        ):
            raise RuntimeError(f"rank terminal identity drift: {rank}")
        records = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(records) != int(terminal.get("rows", -1)):
            raise RuntimeError(f"rank prediction count drift: {rank}")
        if any(int(row.get("rank", -1)) != rank for row in records):
            raise RuntimeError(f"rank prediction assignment drift: {rank}")
        captured.extend(records)
        rank_sources.append(
            {
                "rank": rank,
                "predictions": relative(root, predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "terminal": relative(root, terminal_path),
                "terminal_sha256": sha256_file(terminal_path),
                "rows": len(records),
            }
        )
    captured.sort(key=lambda row: int(row["manifest_index"]))
    if len(captured) != len(expected):
        raise RuntimeError("merged Phase-2 row-count drift")

    raw_rows: list[dict[str, str]] = []
    sentence_rows: list[dict[str, str]] = []
    ledger_rows: list[dict[str, Any]] = []
    fallback_ids: list[str] = []
    for source, prediction in zip(expected, captured, strict=True):
        required = ("row_key", "id", "manifest_index", "derived_audio_sha256")
        if any(source[key] != prediction.get(key) for key in required):
            raise RuntimeError(
                f"Phase-2 inference/source identity drift: {source['manifest_index']}"
            )
        if int(source["manifest_index"]) % args.expected_world_size != int(
            prediction.get("rank", -1)
        ):
            raise RuntimeError(f"Phase-2 rank sharding drift: {source['id']}")
        raw = normalize_text(prediction.get("hypothesis", ""))
        if not raw:
            raw = "A"
            fallback_ids.append(str(source["id"]))
        capped = sentence_case(raw)
        raw_rows.append({"ID": str(source["id"]), "Target": raw})
        sentence_rows.append({"ID": str(source["id"]), "Target": capped})
        ledger_rows.append(
            {
                "official_order": int(source["official_order"]),
                "ID": str(source["id"]),
                "language": args.language,
                "pool_id": str(source["pool_id"]),
                "pool_split": str(source["pool_split"]),
                "model_label": args.model_label,
                "model_sha256": args.model_sha256,
                "beam_size": args.beam_size,
                "rank": int(prediction["rank"]),
                "raw_hypothesis": raw,
                "sentence_case_hypothesis": capped,
                "blank_fallback_applied": str(source["id"]) in fallback_ids,
            }
        )

    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        raw_path = temporary / "predictions.raw.csv"
        sentence_path = temporary / "predictions.sentence_case.csv"
        ledger_path = temporary / "row_source_ledger.csv"
        write_csv(raw_path, ["ID", "Target"], raw_rows)
        write_csv(sentence_path, ["ID", "Target"], sentence_rows)
        write_csv(ledger_path, list(ledger_rows[0]), ledger_rows)
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "created_at_utc": utc_now(),
            "run_id": args.run_id,
            "language": args.language,
            "rows": len(raw_rows),
            "beam_size": args.beam_size,
            "model_label": args.model_label,
            "model_sha256": args.model_sha256,
            "prepared_manifest": relative(root, manifest_summary_path),
            "prepared_manifest_sha256": sha256_file(manifest_summary_path),
            "rank_sources": rank_sources,
            "raw_predictions": relative(root, output / "predictions.raw.csv"),
            "raw_predictions_sha256": sha256_file(raw_path),
            "sentence_case_predictions": relative(
                root, output / "predictions.sentence_case.csv"
            ),
            "sentence_case_predictions_sha256": sha256_file(sentence_path),
            "row_source_ledger": relative(root, output / "row_source_ledger.csv"),
            "row_source_ledger_sha256": sha256_file(ledger_path),
            "sentence_case_changed_rows": sum(
                raw["Target"] != capped["Target"]
                for raw, capped in zip(raw_rows, sentence_rows, strict=True)
            ),
            "blank_fallback": "A",
            "blank_fallback_ids": fallback_ids,
            "max_rows": args.max_rows,
            "external_action": False,
            "submission_created": False,
        }
        write_json_atomic(temporary / "inference.json", summary)
        os.replace(temporary, output)
    except BaseException:
        import shutil  # noqa: PLC0415

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    subparsers = parser.add_subparsers(required=True, dest="command")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--test-master", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--language", choices=sorted(EXPECTED_LANGUAGE_COUNTS), default="lin")
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--prepared", type=Path, required=True)
    materialize_parser.add_argument("--rank-dir", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--run-id", required=True)
    materialize_parser.add_argument("--model-label", required=True)
    materialize_parser.add_argument("--model-sha256", required=True)
    materialize_parser.add_argument("--beam-size", type=int, choices=(1, 5), required=True)
    materialize_parser.add_argument("--language", choices=sorted(EXPECTED_LANGUAGE_COUNTS), default="lin")
    materialize_parser.add_argument("--expected-world-size", type=int, default=8)
    materialize_parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if len(args.model_sha256) != 64 or set(args.model_sha256) - set("0123456789abcdef"):
        raise RuntimeError("model SHA-256 must be lowercase hexadecimal")
    if not args.run_id.startswith("RUN") or not args.model_label:
        raise RuntimeError("run ID and model label are required")
    return materialize(args)


if __name__ == "__main__":
    raise SystemExit(main())
