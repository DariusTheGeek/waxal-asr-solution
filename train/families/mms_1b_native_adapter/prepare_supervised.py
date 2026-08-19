#!/usr/bin/env python3
"""Build language-specific MMS CV-002 rows over verified WAXAL3 16 kHz audio."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from supervised.mms_adapter import inspect_head_overlap


ROOT = Path(__file__).resolve().parents[3]
LANGUAGE_CONTRACTS = {
    "lin": {"train_rows": 16_035, "validation_rows": 900, "target_weight": 447.0},
    "sna": {"train_rows": 16_293, "validation_rows": 900, "target_weight": 445.0},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def target_vocabulary(rows: list[dict[str, Any]]) -> dict[str, int]:
    characters = sorted(
        {
            character
            for row in rows
            for character in str(row["target_ctc"])
            if character != " "
        }
    )
    forbidden = {"<pad>", "<unk>", "|"} & set(characters)
    if forbidden:
        raise RuntimeError(f"literal special token collision: {sorted(forbidden)}")
    return {
        "<pad>": 0,
        "<unk>": 1,
        "|": 2,
        **{character: index + 3 for index, character in enumerate(characters)},
    }


def _selected_columns(
    row: dict[str, Any], *, language: str, partition: str, index: int
) -> dict[str, Any]:
    target = " ".join(str(row["target_ctc"]).split())
    if not target:
        raise RuntimeError(f"empty CTC target: {row['row_key']}")
    is_validation = partition == "validation"
    warm = bool(row.get("target_effective_warm", row.get("speaker_is_phase2_target", False)))
    return {
        "row_key": str(row["row_key"]),
        "id": str(row["id"]),
        "language": language,
        "selected_for_training": not is_validation,
        "assignment": "validation_scored" if is_validation else "train",
        "evaluation_index": int(index) if is_validation else -1,
        "audio_relpath": str(row["derived_audio_relpath"]),
        "audio_bytes": 0,
        "audio_sha256": str(row["derived_audio_sha256"]),
        "input_samples": int(row["derived_num_samples"]),
        "duration_s": float(row["derived_duration_s"]),
        "target_ctc": target,
        "target_raw": " ".join(str(row["transcription_nfc"]).split()),
        "reference_ctc": target,
        "reference_raw": " ".join(str(row["transcription_nfc"]).split()),
        "speaker_key": str(row["speaker_key"]),
        "stratum": "warm" if warm else "cold",
        "original_split": str(row["split"]),
        "is_phase2_test_speaker": bool(row.get("speaker_is_phase2_target", False)),
        "is_phase2_test_prompt": bool(row.get("is_phase2_matched_pool_row", False)),
        "target_weight": float(row.get("target_weight") or 0.0),
        "slot_id": str(row.get("slot_id") or ""),
        "target_phase2_id": str(row.get("target_phase2_id") or ""),
        "target_official_order": int(row.get("target_official_order") or -1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=sorted(LANGUAGE_CONTRACTS), required=True)
    parser.add_argument(
        "--audio-build", type=Path
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--adapter",
        type=Path,
    )
    parser.add_argument(
        "--source-vocab",
        type=Path,
        default=ROOT / "models/mms-1b-all/vocab.json",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    language = str(args.language)
    contract = LANGUAGE_CONTRACTS[language]
    audio_build = (
        args.audio_build
        or ROOT / f"data/derived/omniasr/{language}_cv002_supervised_v1"
    ).resolve()
    output = (
        args.output
        or ROOT / f"data/derived/mms/{language}_cv002_native_adapter_v1"
    ).resolve()
    adapter = (
        args.adapter
        or ROOT / f"models/mms-1b-all/adapter.{language}.safetensors"
    ).resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(f".{output.name}.building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)

    train_source = audio_build / "manifests/train.rows.parquet"
    validation_source = audio_build / "manifests/dev.rows.parquet"
    audio_root = audio_build / "audio"
    source_build = audio_build / "BUILD.json"
    for path in (train_source, validation_source, audio_root, source_build):
        if not path.exists():
            raise FileNotFoundError(path)
    source_record = json.loads(source_build.read_text(encoding="utf-8"))
    if (
        source_record.get("language") != language
        or int(source_record.get("train_rows", -1)) != int(contract["train_rows"])
        or int(source_record.get("validation_rows", -1))
        != int(contract["validation_rows"])
    ):
        raise RuntimeError("source 16 kHz audio-build language/row contract drift")

    train_raw = pq.read_table(train_source).to_pylist()
    validation_raw = pq.read_table(validation_source).to_pylist()
    if (
        len(train_raw) != int(contract["train_rows"])
        or len(validation_raw) != int(contract["validation_rows"])
    ):
        raise RuntimeError(
            f"CV row-count drift: train={len(train_raw)} validation={len(validation_raw)}"
        )
    train = [
        _selected_columns(row, language=language, partition="train", index=i)
        for i, row in enumerate(train_raw)
    ]
    validation = [
        _selected_columns(row, language=language, partition="validation", index=i)
        for i, row in enumerate(validation_raw)
    ]
    if {row["row_key"] for row in train} & {row["row_key"] for row in validation}:
        raise RuntimeError("train/validation row overlap")
    if len({row["id"] for row in train + validation}) != len(train) + len(validation):
        raise RuntimeError("duplicate MMS supervised ID")
    validation_target_weight = sum(float(row["target_weight"]) for row in validation)
    if abs(validation_target_weight - float(contract["target_weight"])) > 1e-9:
        raise RuntimeError("validation target-weight mass drift")

    # Freeze one comparison vocabulary that can encode the complete fixed CV.
    # Validation characters define output geometry only; validation examples
    # are never selected for gradient updates.
    vocabulary = target_vocabulary(train + validation)
    train_characters = {
        character
        for row in train
        for character in str(row["target_ctc"])
        if character != " "
    }
    validation_characters = {
        character
        for row in validation
        for character in str(row["target_ctc"])
        if character != " "
    }
    unseen = sorted(validation_characters - train_characters)

    def verify_audio(row: dict[str, Any]) -> tuple[str, int]:
        path = audio_root / str(row["audio_relpath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != row["audio_sha256"]:
            raise RuntimeError(f"derived audio hash drift: {row['row_key']}")
        return str(row["row_key"]), int(path.stat().st_size)

    workers = max(1, min(int(args.workers), 64))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        byte_sizes = dict(pool.map(verify_audio, train + validation, chunksize=16))
    for row in train + validation:
        row["audio_bytes"] = byte_sizes[str(row["row_key"])]

    manifest = train + validation
    pq.write_table(
        pa.Table.from_pylist(manifest),
        building / "manifest.parquet",
        compression="zstd",
    )
    write_json(building / "vocab.json", vocabulary)
    (building / "train.ids").write_text(
        "\n".join(str(row["row_key"]) for row in train) + "\n",
        encoding="utf-8",
    )
    (building / "validation.ids").write_text(
        "\n".join(str(row["row_key"]) for row in validation) + "\n",
        encoding="utf-8",
    )
    overlap = inspect_head_overlap(
        adapter_path=adapter,
        source_vocab_path=args.source_vocab.resolve(),
        target_vocab_path=building / "vocab.json",
        language=language,
    )
    write_json(building / "HEAD_OVERLAP.json", overlap)

    files = []
    for path in sorted(building.iterdir()):
        if path.name == "BUILD.json":
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    record = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "cv_id": "CV-P2T-D0-WARM-900X2-002",
        "source_audio_build": str(audio_build.relative_to(ROOT)),
        "source_audio_build_sha256": sha256_file(source_build),
        "source_train_sha256": sha256_file(train_source),
        "source_validation_sha256": sha256_file(validation_source),
        "audio_root": str(audio_root.relative_to(ROOT)),
        "audio_files_verified": len(manifest),
        "audio_bytes_verified": sum(int(row["audio_bytes"]) for row in manifest),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_target_weight_sum": validation_target_weight,
        "vocabulary_rows": len(vocabulary),
        "vocabulary_source": "fixed_train_plus_validation_character_inventory",
        "validation_only_characters": unseen,
        "validation_labels_used_for_vocabulary_only": bool(unseen),
        "validation_rows_used_for_gradient_updates": 0,
        "row_identity_digest": canonical_sha256(
            [
                {
                    "row_key": row["row_key"],
                    "audio_sha256": row["audio_sha256"],
                    "target_ctc": row["target_ctc"],
                    "assignment": row["assignment"],
                    "target_weight": row["target_weight"],
                }
                for row in manifest
            ]
        ),
        "files": files,
        "transcript_source": "fixed CV train/validation labels",
        "test_labels_accessed": False,
    }
    write_json(building / "BUILD.json", record)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.replace(output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
