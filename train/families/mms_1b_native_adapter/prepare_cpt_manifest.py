#!/usr/bin/env python3
"""Freeze an MMS-readable transcript-free manifest from a declared CPT order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/derived/cpt_views/CPT-LIN-UALL-P2RAW-V1"
DEFAULT_ORDER_VIEW_ID = "CPT-LIN-UALL-P2RAW-V2-LENSORT"
DEFAULT_VIEW_ID = "CPT-LIN-UALL-P2RAW-V2-LENSORT-MMS-G8-V1"
FORBIDDEN = ("transcript", "target", "sentence", "label", "text")


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


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def stage_rows(
    *,
    stage: str,
    order: Path,
    source_by_id: dict[str, dict[str, Any]],
    audio_root_holder: list[Path],
) -> list[dict[str, Any]]:
    ids_path = order / f"manifests/{stage}.ids"
    tsv_path = order / f"manifests/{stage}.tsv"
    identifiers = read_lines(ids_path)
    lines = read_lines(tsv_path)
    if len(lines) != len(identifiers) + 1:
        raise RuntimeError(f"{stage} TSV/ID row-count drift")
    root_value = lines[0]
    root = (
        ROOT / root_value.removeprefix("repo://")
        if root_value.startswith("repo://")
        else Path(root_value)
    ).resolve()
    if not root.is_dir() or not str(root).startswith(str(ROOT.resolve()) + "/"):
        raise RuntimeError(f"invalid WAXAL3 CPT audio root: {root}")
    if audio_root_holder and audio_root_holder[0] != root:
        raise RuntimeError("CPT stages disagree on audio root")
    if not audio_root_holder:
        audio_root_holder.append(root)
    output = []
    for order_index, (identifier, line) in enumerate(
        zip(identifiers, lines[1:], strict=True)
    ):
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            raise RuntimeError(f"invalid {stage} TSV line {order_index + 2}")
        relpath, frames_raw = parts
        source = source_by_id.get(identifier)
        if source is None:
            raise RuntimeError(f"V2 order references unknown segment: {identifier}")
        if str(source["derived_audio_relpath"]) != relpath:
            raise RuntimeError(f"V2/source relative-path drift: {identifier}")
        frames = int(frames_raw)
        if int(source["segment_num_samples"]) != frames:
            raise RuntimeError(f"V2/source frame-count drift: {identifier}")
        path = root / relpath
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(
            {
                "id": identifier,
                "language": "lin",
                "stage": stage,
                "stage_order_index": order_index,
                "audio_relpath": relpath,
                "audio_bytes": int(path.stat().st_size),
                "audio_sha256": str(source["derived_audio_sha256"]),
                "decoded_frames": frames,
                "duration_s": float(source["segment_duration_s"]),
                "speaker_key": str(source["speaker_key"]),
                "source_kind": str(source["source_kind"]),
                "source_row_key": str(source["source_row_key"]),
                "source_encoded_sha256": str(source["source_encoded_sha256"]),
                "segment_pcm_sha256": str(source["segment_pcm_sha256"]),
                "phase2_id": str(source["phase2_id"] or ""),
                "exact_phase2_source_view": bool(source["exact_phase2_source_view"]),
                "corrected_phase2_raw_view": bool(source["corrected_phase2_raw_view"]),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-view-id", default=DEFAULT_ORDER_VIEW_ID)
    parser.add_argument("--view-id", default=DEFAULT_VIEW_ID)
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument(
        "--created-at-utc",
        help="frozen provenance timestamp for byte-reproducible cloud reconstruction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.global_batch < 1:
        raise ValueError("global batch must be positive")
    if args.created_at_utc:
        parsed_created_at = datetime.fromisoformat(args.created_at_utc)
        if parsed_created_at.tzinfo is None:
            raise ValueError("created-at-utc must include a timezone")
    order_view_id = str(args.order_view_id)
    view_id = str(args.view_id)
    if any("/" in value or not value.startswith("CPT-") for value in (order_view_id, view_id)):
        raise ValueError("invalid CPT view ID")
    order = ROOT / "data/derived/cpt_views" / order_view_id
    default_output = (
        ROOT
        / "data/derived/mms"
        / (
            "cpt_lin_v2_lensort_manifest_v1"
            if view_id == DEFAULT_VIEW_ID
            else "cpt_lin_v3_lensort_g32_manifest_v1"
        )
    )
    output = (args.output or default_output).resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(f".{output.name}.building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)

    source_segments = SOURCE / "segments.parquet"
    source_audit_path = SOURCE / "AUDIT.json"
    order_build_path = order / "BUILD.json"
    order_audit_path = order / "AUDIT.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    order_build = json.loads(order_build_path.read_text(encoding="utf-8"))
    order_audit = json.loads(order_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status") != "PASS" or order_audit.get("status") != "PASS":
        raise RuntimeError("source CPT audit is not PASS")
    if not source_audit.get("all_audio_headers_and_hashes_verified"):
        raise RuntimeError("source CPT audio was not fully byte-audited")
    if not order_audit.get("runtime_manifest_transcript_free"):
        raise RuntimeError("ordered runtime view is not transcript-free")
    if (
        order_audit.get("view_id") != order_view_id
        or int(order_audit.get("global_batch_size", args.global_batch))
        != int(args.global_batch)
    ):
        raise RuntimeError("ordered runtime view geometry drift")
    if sha256_file(source_segments) != source_audit["segments_parquet_sha256"]:
        raise RuntimeError("source segments parquet hash drift")

    source_rows = pq.read_table(source_segments).to_pylist()
    source_by_id = {str(row["segment_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows) != 0:
        raise RuntimeError("duplicate source CPT segment ID")
    audio_roots: list[Path] = []
    broad = stage_rows(
        stage="broad",
        order=order,
        source_by_id=source_by_id,
        audio_root_holder=audio_roots,
    )
    tail = stage_rows(
        stage="tail",
        order=order,
        source_by_id=source_by_id,
        audio_root_holder=audio_roots,
    )
    rows = broad + tail
    if len(rows) != 139_239 or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("MMS CPT presentation identity/count drift")
    if set(source_by_id) != {str(row["id"]) for row in rows}:
        raise RuntimeError("MMS CPT view is not an exact source permutation")
    columns = set(rows[0])
    violations = sorted(
        column
        for column in columns
        if any(token in column.casefold() for token in FORBIDDEN)
    )
    if violations:
        raise RuntimeError(f"forbidden transcript-like columns: {violations}")

    manifest_path = building / "segments.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest_path, compression="zstd")
    global_batch = int(args.global_batch)
    stage_contracts = {}
    for stage, selected in (("broad", broad), ("tail", tail)):
        padding = (-len(selected)) % global_batch
        stage_contracts[stage] = {
            "unique_rows": len(selected),
            "synchronization_padding_slots": padding,
            "updates_per_sweep": math.ceil(len(selected) / global_batch),
        }
    identity = [
        {
            "id": row["id"],
            "stage": row["stage"],
            "stage_order_index": row["stage_order_index"],
            "audio_sha256": row["audio_sha256"],
        }
        for row in rows
    ]
    record = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": args.created_at_utc or datetime.now(timezone.utc).isoformat(),
        "view_id": view_id,
        "source_view_id": order_view_id,
        "source_segments": str(source_segments.relative_to(ROOT)),
        "source_segments_sha256": sha256_file(source_segments),
        "source_audit": str(source_audit_path.relative_to(ROOT)),
        "source_audio_inventory_digest": order_build[
            "source_audio_inventory_digest"
        ],
        "order_audit": str(order_audit_path.relative_to(ROOT)),
        "order_audit_sha256": sha256_file(order_audit_path),
        "audio_root": str(audio_roots[0].relative_to(ROOT)),
        "audio_bytes_inherited_full_audit": int(source_audit["audited_audio_bytes"]),
        "segments": len(rows),
        "presentation_hours": sum(float(row["duration_s"]) for row in rows) / 3600.0,
        "stage_contracts": stage_contracts,
        "updates_per_sweep": sum(
            int(value["updates_per_sweep"]) for value in stage_contracts.values()
        ),
        "synchronization_padding_slots_per_sweep": sum(
            int(value["synchronization_padding_slots"])
            for value in stage_contracts.values()
        ),
        "manifest": {
            "path": "segments.parquet",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "columns": sorted(columns),
        },
        "identity_digest": canonical_sha256(identity),
        "all_source_presentations_exactly_once": True,
        "runtime_manifest_transcript_free": True,
        "transcripts_accessed": False,
        "test_labels_accessed": False,
    }
    (building / "BUILD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    building.replace(output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
