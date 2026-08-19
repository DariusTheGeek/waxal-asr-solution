#!/usr/bin/env python3
"""Build and full-byte audit a transcript-free source-row MMS CPT view.

This is the WAXAL3 continuation of the WAXAL2 S008 data contract: one complete
source waveform per official unlabeled row, with fresh deterministic <=15 s
crops selected by the training dataset on every sweep.  It intentionally does
not perform full-content segmentation and does not append a second corrected
Phase-2 raw-WAV view.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import warnings
import wave
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parents[2]
TARGET_SAMPLE_RATE = 16_000
EXPECTED = {
    "sna": {
        "official_rows": 85_384,
        "official_hours": 475.6491891609977,
        "excluded_rows": 12,
        "excluded_hours": 0.15217999999999998,
        "retained_rows": 85_372,
        "retained_source_hours": 475.4970091609978,
        "retained_speakers": 218,
        "exact_phase2_source_views": 445,
        "digital_silence_rows": 12,
        "over_60_second_rows": 3,
        "non_mono_rows": 0,
    }
}
FORBIDDEN_ROOTS = {
    "hypothesis",
    "label",
    "reference",
    "sentence",
    "target",
    "text",
    "transcript",
    "transcription",
    "wrd",
}
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_transcript_free_columns(columns: list[str]) -> None:
    forbidden = []
    for column in columns:
        tokens = re.sub(r"[^a-z0-9]+", "_", column.casefold()).split("_")
        if any(
            token.startswith(root)
            for token in tokens
            for root in FORBIDDEN_ROOTS
        ):
            forbidden.append(column)
    if forbidden:
        raise RuntimeError(f"transcript-like columns entered CPT view: {forbidden}")


def expected_resampled_samples(
    source_samples: int,
    source_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> int:
    if min(source_samples, source_sample_rate, target_sample_rate) <= 0:
        raise ValueError("sample counts and rates must be positive")
    return math.ceil(source_samples * target_sample_rate / source_sample_rate)


def pcm16_bytes(waveform: torch.Tensor) -> bytes:
    samples = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    quantized = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(
        "<i2", copy=False
    )
    return quantized.tobytes(order="C")


def wav_bytes(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm)
    return output.getvalue()


def build_plan(eda_rows_path: Path, language: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    contract = EXPECTED.get(language)
    if contract is None:
        raise ValueError(f"unsupported source-crop language: {language}")
    eda = pl.read_parquet(eda_rows_path).filter(
        (pl.col("supervision") == "unlabeled") & (pl.col("language") == language)
    )
    digital_silence = pl.col("rms_dbfs") <= -159.0
    over_60_seconds = pl.col("duration_s") > 60.0
    non_mono = pl.col("channels") != 1
    exclusion = digital_silence | over_60_seconds | non_mono
    retained_frame = eda.filter(~exclusion).sort("id", "parquet_path", "row_index")
    excluded_frame = eda.filter(exclusion).sort("id", "parquet_path", "row_index")

    safe_columns = [
        "row_key",
        "id",
        "language",
        "speaker_key",
        "parquet_path",
        "row_index",
        "audio_path",
        "encoded_audio_bytes",
        "encoded_audio_sha256",
        "sample_rate",
        "channels",
        "decoded_frames",
        "duration_s",
        "rms_dbfs",
        "decoded_pcm_sha256",
        "trimmed_pcm_sha256",
        "phase2_id",
        "is_phase2_matched_pool_row",
    ]
    assert_transcript_free_columns(safe_columns)
    retained: list[dict[str, Any]] = []
    for order, row in enumerate(retained_frame.select(safe_columns).to_dicts()):
        retained.append(
            {
                "source_order_index": order,
                "source_row_key": str(row["row_key"]),
                "source_id": str(row["id"]),
                "language": str(row["language"]),
                "speaker_key": str(row["speaker_key"]),
                "source_parquet_path": str(row["parquet_path"]),
                "source_row_index": int(row["row_index"]),
                "source_audio_path": str(row["audio_path"]),
                "source_encoded_bytes": int(row["encoded_audio_bytes"]),
                "source_encoded_sha256": str(row["encoded_audio_sha256"]),
                "source_sample_rate": int(row["sample_rate"]),
                "source_channels": int(row["channels"]),
                "source_decoded_frames": int(row["decoded_frames"]),
                "source_duration_s": float(row["duration_s"]),
                "source_decoded_pcm_sha256": str(row["decoded_pcm_sha256"]),
                "source_trimmed_pcm_sha256": str(row["trimmed_pcm_sha256"]),
                "expected_16k_samples": expected_resampled_samples(
                    int(row["decoded_frames"]), int(row["sample_rate"])
                ),
                "phase2_id": str(row["phase2_id"] or ""),
                "exact_phase2_source_view": bool(row["is_phase2_matched_pool_row"]),
            }
        )

    excluded: list[dict[str, Any]] = []
    for row in excluded_frame.select(safe_columns).to_dicts():
        reasons = []
        if float(row["rms_dbfs"]) <= -159.0:
            reasons.append("digital_silence")
        if float(row["duration_s"]) > 60.0:
            reasons.append("duration_over_60_seconds")
        if int(row["channels"]) != 1:
            reasons.append("non_mono")
        excluded.append(
            {
                "source_row_key": str(row["row_key"]),
                "source_id": str(row["id"]),
                "language": str(row["language"]),
                "speaker_key": str(row["speaker_key"]),
                "source_parquet_path": str(row["parquet_path"]),
                "source_row_index": int(row["row_index"]),
                "source_audio_path": str(row["audio_path"]),
                "source_encoded_bytes": int(row["encoded_audio_bytes"]),
                "source_encoded_sha256": str(row["encoded_audio_sha256"]),
                "source_sample_rate": int(row["sample_rate"]),
                "source_channels": int(row["channels"]),
                "source_decoded_frames": int(row["decoded_frames"]),
                "source_duration_s": float(row["duration_s"]),
                "source_decoded_pcm_sha256": str(row["decoded_pcm_sha256"]),
                "source_trimmed_pcm_sha256": str(row["trimmed_pcm_sha256"]),
                "phase2_id": str(row["phase2_id"] or ""),
                "exact_phase2_source_view": bool(row["is_phase2_matched_pool_row"]),
                "exclusion_reasons_json": json.dumps(reasons, separators=(",", ":")),
            }
        )

    summary = {
        "language": language,
        "official_rows": eda.height,
        "official_hours": float(eda["duration_s"].sum() / 3600.0),
        "excluded_rows": len(excluded),
        "excluded_hours": sum(float(row["source_duration_s"]) for row in excluded) / 3600.0,
        "retained_rows": len(retained),
        "retained_source_hours": sum(float(row["source_duration_s"]) for row in retained) / 3600.0,
        "retained_speakers": len({str(row["speaker_key"]) for row in retained}),
        "capped_15s_source_hours": sum(
            min(float(row["source_duration_s"]), 15.0) for row in retained
        ) / 3600.0,
        "exact_phase2_source_views": sum(
            bool(row["exact_phase2_source_view"]) for row in retained
        ),
        "excluded_exact_phase2_source_views": sum(
            bool(row["exact_phase2_source_view"]) for row in excluded
        ),
        "digital_silence_rows": sum(
            "digital_silence" in str(row["exclusion_reasons_json"])
            for row in excluded
        ),
        "over_60_second_rows": sum(
            "duration_over_60_seconds" in str(row["exclusion_reasons_json"])
            for row in excluded
        ),
        "non_mono_rows": sum(
            "non_mono" in str(row["exclusion_reasons_json"])
            for row in excluded
        ),
    }
    for key, expected in contract.items():
        observed = summary[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, abs_tol=1e-8):
                raise RuntimeError(f"source plan drift for {key}: {observed} != {expected}")
        elif observed != expected:
            raise RuntimeError(f"source plan drift for {key}: {observed} != {expected}")
    if summary["excluded_exact_phase2_source_views"] != 0:
        raise RuntimeError("an exact Phase-2 source view was excluded")
    return retained, excluded, summary


def decode_and_materialize(row: dict[str, Any], encoded: bytes, audio_root: Path) -> dict[str, Any]:
    if len(encoded) != int(row["source_encoded_bytes"]):
        raise RuntimeError(f"encoded byte drift: {row['source_id']}")
    if hashlib.sha256(encoded).hexdigest() != row["source_encoded_sha256"]:
        raise RuntimeError(f"encoded hash drift: {row['source_id']}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        waveform, sample_rate = torchaudio.load(
            io.BytesIO(encoded), format="mp3", backend="ffmpeg"
        )
    if (
        waveform.ndim != 2
        or int(waveform.shape[0]) != int(row["source_channels"])
        or int(waveform.shape[1]) != int(row["source_decoded_frames"])
        or int(sample_rate) != int(row["source_sample_rate"])
    ):
        raise RuntimeError(f"source decode geometry drift: {row['source_id']}")
    if waveform.shape[0] != 1:
        raise RuntimeError(f"non-mono source reached materializer: {row['source_id']}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, int(sample_rate), TARGET_SAMPLE_RATE
        )
    waveform = waveform.squeeze(0).contiguous()
    if waveform.numel() != int(row["expected_16k_samples"]):
        raise RuntimeError(
            f"resampled length drift: {row['source_id']} "
            f"{waveform.numel()} != {row['expected_16k_samples']}"
        )
    if waveform.numel() < TARGET_SAMPLE_RATE or not torch.isfinite(waveform).all():
        raise RuntimeError(f"invalid model waveform: {row['source_id']}")

    pcm = pcm16_bytes(waveform)
    payload = wav_bytes(pcm)
    identity_hash = hashlib.sha256(str(row["source_row_key"]).encode()).hexdigest()
    clean_id = SAFE.sub("_", str(row["source_id"]))[:64]
    relpath = Path(identity_hash[:2]) / f"{clean_id}-{identity_hash[:16]}.wav"
    destination = audio_root / relpath
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".wav.tmp")
    if destination.exists() or staging.exists():
        raise FileExistsError(destination)
    with staging.open("xb") as handle:
        handle.write(payload)
    staging.replace(destination)
    frames = int(waveform.numel())
    return {
        "id": str(row["source_row_key"]),
        "source_id": str(row["source_id"]),
        "language": str(row["language"]),
        "stage": "broad",
        "stage_order_index": int(row["source_order_index"]),
        "audio_relpath": relpath.as_posix(),
        "audio_bytes": len(payload),
        "audio_sha256": hashlib.sha256(payload).hexdigest(),
        "decoded_frames": frames,
        "duration_s": frames / TARGET_SAMPLE_RATE,
        "capped_duration_s": min(frames / TARGET_SAMPLE_RATE, 15.0),
        "speaker_key": str(row["speaker_key"]),
        "source_kind": "official_unlabeled_mp3",
        "source_row_key": str(row["source_row_key"]),
        "source_encoded_sha256": str(row["source_encoded_sha256"]),
        "source_decoded_pcm_sha256": str(row["source_decoded_pcm_sha256"]),
        "source_trimmed_pcm_sha256": str(row["source_trimmed_pcm_sha256"]),
        "source_16k_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "source_duration_s": float(row["source_duration_s"]),
        "source_sample_rate": int(row["source_sample_rate"]),
        "source_decoded_frames": int(row["source_decoded_frames"]),
        "phase2_id": str(row["phase2_id"]),
        "exact_phase2_source_view": bool(row["exact_phase2_source_view"]),
        "corrected_phase2_raw_view": False,
    }


def process_parquet_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    parquet_path = Path(str(task["parquet_path"]))
    rows: list[dict[str, Any]] = task["rows"]
    audio_root = Path(str(task["audio_root"]))
    table = pq.read_table(parquet_path, columns=["id", "audio"])
    indices = pa.array([int(row["source_row_index"]) for row in rows], type=pa.int64())
    selected = table.take(indices).to_pylist()
    output = []
    for row, source in zip(rows, selected, strict=True):
        if str(source["id"]) != str(row["source_id"]):
            raise RuntimeError(f"source row index drift: {row['source_row_key']}")
        audio = source.get("audio")
        encoded = None if not isinstance(audio, dict) else audio.get("bytes")
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise RuntimeError(f"missing embedded audio: {row['source_row_key']}")
        output.append(decode_and_materialize(row, bytes(encoded), audio_root))
    return output


def build(args: argparse.Namespace) -> int:
    language = str(args.language)
    eda_rows = args.eda_rows.resolve()
    retained, excluded, plan = build_plan(eda_rows, language)
    global_batch = int(args.global_batch)
    usable_rows = (len(retained) // global_batch) * global_batch
    dropped_rows = len(retained) - usable_rows
    updates_per_sweep = usable_rows // global_batch
    plan_record = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "view_id": str(args.view_id),
        "sampling_mode": "speaker_interleaved_drop_remainder_sourcecrop",
        "plan": plan,
        "geometry": {
            "global_batch": global_batch,
            "usable_unique_rows_per_sweep": usable_rows,
            "dropped_rows_per_sweep": dropped_rows,
            "updates_per_sweep": updates_per_sweep,
        },
        "source_inputs": [
            {
                "role": "eda_rows",
                "path": eda_rows.relative_to(REPO_ROOT).as_posix(),
                "bytes": eda_rows.stat().st_size,
                "sha256": sha256_file(eda_rows),
            }
        ],
        "transcripts_accessed": False,
        "test_labels_accessed": False,
    }
    if global_batch != 8 or dropped_rows != 4 or updates_per_sweep != 10_671:
        raise RuntimeError(f"S008 Shona geometry drift: {plan_record['geometry']}")
    if args.plan_only:
        print(json.dumps(plan_record, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(f".{output.name}.building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)
    audio_root = building / "audio"
    audio_root.mkdir()

    frame = pl.DataFrame(retained, infer_schema_length=None)
    tasks = []
    for parquet_key, group in frame.group_by("source_parquet_path", maintain_order=True):
        relative = Path(str(parquet_key[0]))
        parquet_path = relative if relative.is_absolute() else REPO_ROOT / relative
        tasks.append(
            {
                "parquet_path": str(parquet_path.resolve()),
                "audio_root": str(audio_root),
                "rows": group.sort("source_order_index").to_dicts(),
            }
        )
    workers = max(1, min(int(args.workers), len(tasks), 32))
    materialized: list[dict[str, Any]] = []
    completed_rows = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_parquet_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                rows = future.result()
                materialized.extend(rows)
                completed_rows += len(task["rows"])
                print(
                    f"materialized {index}/{len(tasks)} shards; "
                    f"rows={completed_rows}/{len(retained)}",
                    flush=True,
                )
    except BaseException as error:
        (building / "FAILURE.json").write_text(
            json.dumps(
                {
                    "created_at_utc": utc_now(),
                    "error": repr(error),
                    "completed_rows": completed_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    if completed_rows != len(retained) or len(materialized) != len(retained):
        raise RuntimeError("materialized source-row count drift")
    manifest = pl.DataFrame(materialized, infer_schema_length=None).sort(
        "stage_order_index"
    )
    assert_transcript_free_columns(manifest.columns)
    if (
        manifest["id"].n_unique() != manifest.height
        or manifest["audio_relpath"].n_unique() != manifest.height
        or manifest["audio_sha256"].n_unique() != manifest.height
    ):
        raise RuntimeError("source-row identity/audio inventory is not unique")
    if manifest["stage_order_index"].to_list() != list(range(manifest.height)):
        raise RuntimeError("stage order indices are not contiguous")

    manifest_path = building / "manifest.parquet"
    excluded_path = building / "excluded.parquet"
    manifest.write_parquet(manifest_path, compression="zstd")
    pl.DataFrame(excluded, infer_schema_length=None).write_parquet(
        excluded_path, compression="zstd"
    )
    identity = manifest.select(
        "id", "stage_order_index", "audio_relpath", "audio_sha256"
    ).to_dicts()
    build_record = {
        **plan_record,
        "workers": workers,
        "manifest": {
            "path": "manifest.parquet",
            "rows": manifest.height,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "columns": manifest.columns,
        },
        "excluded_manifest": {
            "path": "excluded.parquet",
            "rows": len(excluded),
            "bytes": excluded_path.stat().st_size,
            "sha256": sha256_file(excluded_path),
        },
        "identity_digest": canonical_sha256(identity),
        "presentation_hours_full_source": float(manifest["duration_s"].sum() / 3600.0),
        "presentation_hours_capped_15s": float(
            manifest["capped_duration_s"].sum() / 3600.0
        ),
        "audio_bytes": int(manifest["audio_bytes"].sum()),
        "stage_contracts": {
            "broad": {
                "unique_rows": manifest.height,
                "usable_unique_rows_per_sweep": usable_rows,
                "dropped_rows_per_sweep": dropped_rows,
                "synchronization_padding_slots": 0,
                "updates_per_sweep": updates_per_sweep,
            },
            "tail": {
                "unique_rows": 0,
                "usable_unique_rows_per_sweep": 0,
                "dropped_rows_per_sweep": 0,
                "synchronization_padding_slots": 0,
                "updates_per_sweep": 0,
            },
        },
        "updates_per_sweep": updates_per_sweep,
        "synchronization_padding_slots_per_sweep": 0,
        "dropped_rows_per_sweep": dropped_rows,
        "runtime_manifest_transcript_free": True,
        "transcripts_accessed": False,
        "test_labels_accessed": False,
    }
    (building / "BUILD.json").write_text(
        json.dumps(build_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    building.replace(output)
    print(json.dumps(build_record, indent=2, sort_keys=True))
    return 0


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != TARGET_SAMPLE_RATE
        ):
            raise RuntimeError(f"WAV format drift: {path}")
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    return pcm, frames


def audit_audio(row: dict[str, Any], audio_root: Path) -> dict[str, Any]:
    path = audio_root / str(row["audio_relpath"])
    raw = path.read_bytes()
    if len(raw) != int(row["audio_bytes"]) or hashlib.sha256(raw).hexdigest() != row["audio_sha256"]:
        raise RuntimeError(f"derived WAV byte/hash drift: {row['id']}")
    pcm, frames = read_wav_pcm(path)
    if frames != int(row["decoded_frames"]):
        raise RuntimeError(f"derived WAV frame drift: {row['id']}")
    if hashlib.sha256(pcm).hexdigest() != row["source_16k_pcm_sha256"]:
        raise RuntimeError(f"derived PCM hash drift: {row['id']}")
    return {"bytes": len(raw), "frames": frames}


def audit(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    audit_path = root / "AUDIT.json"
    exposure_path = root / "EXPOSURE.json"
    if audit_path.exists() or exposure_path.exists():
        raise FileExistsError("create-only source-crop audit/exposure record exists")
    build_record = json.loads((root / "BUILD.json").read_text(encoding="utf-8"))
    manifest_path = root / "manifest.parquet"
    excluded_path = root / "excluded.parquet"
    manifest = pl.read_parquet(manifest_path).sort("stage_order_index")
    excluded = pl.read_parquet(excluded_path)
    assert_transcript_free_columns(manifest.columns)
    assert_transcript_free_columns(excluded.columns)
    if manifest.height != EXPECTED[str(args.language)]["retained_rows"]:
        raise RuntimeError("audit manifest row-count drift")
    if sha256_file(manifest_path) != build_record["manifest"]["sha256"]:
        raise RuntimeError("audit manifest hash drift")
    if sha256_file(excluded_path) != build_record["excluded_manifest"]["sha256"]:
        raise RuntimeError("audit exclusion hash drift")

    audio_root = root / "audio"
    expected_paths = set(manifest["audio_relpath"].to_list())
    observed_paths = {
        path.relative_to(audio_root).as_posix() for path in audio_root.rglob("*.wav")
    }
    if observed_paths != expected_paths:
        raise RuntimeError(
            f"audio path inventory drift: missing={len(expected_paths-observed_paths)} "
            f"extra={len(observed_paths-expected_paths)}"
        )
    workers = max(1, min(int(args.workers), 64))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda row: audit_audio(row, audio_root),
                manifest.select(
                    "id",
                    "audio_relpath",
                    "audio_bytes",
                    "audio_sha256",
                    "decoded_frames",
                    "source_16k_pcm_sha256",
                ).to_dicts(),
                chunksize=64,
            )
        )

    validation_path = args.validation_manifest.resolve()
    validation = pl.read_parquet(validation_path)
    validation_root = validation_path.parents[1] / "audio"
    validation_pcm_hashes = set()
    for relpath in validation["derived_audio_relpath"].to_list():
        pcm, _frames = read_wav_pcm(validation_root / str(relpath))
        validation_pcm_hashes.add(hashlib.sha256(pcm).hexdigest())
    overlaps = {
        "encoded_sha256": len(
            set(manifest["source_encoded_sha256"].to_list())
            & set(validation["encoded_audio_sha256"].to_list())
        ),
        "decoded_pcm_sha256": len(
            set(manifest["source_decoded_pcm_sha256"].to_list())
            & set(validation["decoded_pcm_sha256"].to_list())
        ),
        "trimmed_pcm_sha256": len(
            set(manifest["source_trimmed_pcm_sha256"].to_list())
            & set(validation["trimmed_pcm_sha256"].to_list())
        ),
        "derived_16k_pcm_sha256": len(
            set(manifest["source_16k_pcm_sha256"].to_list())
            & validation_pcm_hashes
        ),
        "derived_wav_sha256": len(
            set(manifest["audio_sha256"].to_list())
            & set(validation["derived_audio_sha256"].to_list())
        ),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"validation audio entered CPT view: {overlaps}")

    audit_record = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "view_id": build_record["view_id"],
        "language": str(args.language),
        "workers": workers,
        "rows": manifest.height,
        "excluded_rows": excluded.height,
        "all_audio_headers_and_hashes_verified": True,
        "audited_audio_bytes": sum(int(item["bytes"]) for item in results),
        "audited_audio_hours": sum(int(item["frames"]) for item in results)
        / TARGET_SAMPLE_RATE
        / 3600.0,
        "manifest_sha256": sha256_file(manifest_path),
        "excluded_manifest_sha256": sha256_file(excluded_path),
        "build_sha256": sha256_file(root / "BUILD.json"),
        "runtime_manifest_transcript_free": True,
        "validation_manifest": validation_path.relative_to(REPO_ROOT).as_posix(),
        "validation_manifest_sha256": sha256_file(validation_path),
        "validation_audio_hash_overlap": overlaps,
        "transcripts_accessed": False,
        "test_labels_accessed": False,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    exposure_record = {
        **audit_record,
        "audit_sha256": sha256_file(audit_path),
        "source_view_exact_phase2_audio_rows": int(
            manifest["exact_phase2_source_view"].sum()
        ),
        "corrected_phase2_raw_wav_rows": int(
            manifest["corrected_phase2_raw_view"].sum()
        ),
        "phase2_audio_exposure": True,
        "phase2_label_exposure": False,
    }
    with exposure_path.open("x", encoding="utf-8") as handle:
        json.dump(exposure_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(exposure_record, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--language", default="sna")
    build_parser.add_argument("--view-id", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument(
        "--eda-rows",
        type=Path,
        default=REPO_ROOT / "eda/runs/A002_phase2_rank1_eda_v2/rows.parquet",
    )
    build_parser.add_argument("--global-batch", type=int, default=8)
    build_parser.add_argument("--workers", type=int, default=32)
    build_parser.add_argument("--plan-only", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--language", default="sna")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/derived/omniasr/sna_cv002_supervised_v1/manifests/dev.rows.parquet",
    )
    audit_parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    return build(args) if args.command == "build" else audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
