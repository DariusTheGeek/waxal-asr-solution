#!/usr/bin/env python3
"""Capture cropped MMS CTC log probabilities on four GPUs for offline decoding."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
import yaml


MODEL_FAMILY = Path(__file__).resolve().parents[1]
PACKET_SRC = MODEL_FAMILY.parent
sys.path.insert(0, str(MODEL_FAMILY))
sys.path.insert(0, str(PACKET_SRC))

from decoding.beam_postprocess import greedy_ctc  # noqa: E402
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
    if sha256_file(path) != str(item["sha256"]):
        raise RuntimeError(f"file hash drift: {path}")
    return path


class CaptureDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], feature_extractor: Any) -> None:
        self.rows = rows
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(str(row["resolved_audio_path"]))
        if not path.is_file() or path.stat().st_size != int(row["audio_bytes"]):
            raise RuntimeError(f"audio file/size drift: {row['id']}")
        if sha256_file(path) != str(row["audio_sha256"]):
            raise RuntimeError(f"audio SHA-256 drift: {row['id']}")
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if waveform.shape[1] != 1 or waveform.size == 0:
            raise RuntimeError(f"invalid waveform: {row['id']}")
        values = torch.from_numpy(np.asarray(waveform[:, 0], dtype=np.float32))
        if row.get("expected_samples") is not None and values.numel() != int(
            row["expected_samples"]
        ):
            raise RuntimeError(f"sample-count drift: {row['id']}")
        if int(sample_rate) != 16_000:
            values = torchaudio.functional.resample(values, int(sample_rate), 16_000)
        array = values.numpy(force=True).astype(np.float32, copy=False)
        if array.size == 0 or not np.isfinite(array).all():
            raise RuntimeError(f"invalid resampled waveform: {row['id']}")
        processed = self.feature_extractor(
            array, sampling_rate=16_000, return_attention_mask=True
        )
        features = np.asarray(processed["input_values"][0], dtype=np.float32)
        if features.shape != array.shape or not np.isfinite(features).all():
            raise RuntimeError(f"feature-extractor output drift: {row['id']}")
        return {
            "capture_index": int(row["capture_index"]),
            "input_values": features,
        }


class CaptureCollator:
    def __init__(self, feature_extractor: Any) -> None:
        self.feature_extractor = feature_extractor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.feature_extractor.pad(
            [{"input_values": item["input_values"]} for item in examples],
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "capture_indices": [int(item["capture_index"]) for item in examples],
            **batch,
        }


def validation_rows(root: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = checked_file(root, profile["validation_manifest"])
    audio_root = resolve(root, str(profile["validation_audio_root"]))
    rows = [
        dict(row)
        for row in pq.read_table(manifest).to_pylist()
        if str(row["language"]) == "lin"
        and str(row["assignment"]) == "validation_scored"
    ]
    rows.sort(key=lambda row: int(row["evaluation_index"]))
    if len(rows) != int(profile["expected_validation_rows"]):
        raise RuntimeError("validation row-count drift")
    limit = profile.get("validation_limit")
    if limit is not None:
        rows = rows[: int(limit)]
    for row in rows:
        row["capture_index"] = int(row["evaluation_index"])
        row["resolved_audio_path"] = str(
            (audio_root / str(row["audio_relpath"])).resolve()
        )
        row["expected_samples"] = int(row["input_samples"])
    return rows


def test_rows(root: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    master = checked_file(root, profile["test_master"])
    all_rows = pq.read_table(master).to_pylist()
    all_rows.sort(key=lambda row: int(row["official_order"]))
    if len(all_rows) != int(profile["expected_test_rows"]):
        raise RuntimeError("corrected-test row-count drift")
    rows = [dict(row) for row in all_rows if str(row["language"]) == "lin"]
    if len(rows) != int(profile["expected_test_lingala_rows"]):
        raise RuntimeError("corrected-test Lingala route-count drift")
    limit = profile.get("test_limit")
    if limit is not None:
        rows = rows[: int(limit)]
    for row in rows:
        row["capture_index"] = int(row["official_order"])
        row["resolved_audio_path"] = str(resolve(root, str(row["test_wav_path"])))
        row["audio_bytes"] = int(row["test_wav_bytes"])
        row["audio_sha256"] = str(row["test_wav_sha256"])
        row["expected_samples"] = None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    packet = args.packet.resolve()
    profile_path = args.profile.resolve()
    output = args.output.resolve()
    packet_record = read_json(packet / "PACKET.json")
    packet_check = verify_packet(packet)
    if packet_check["status"] != "PASS":
        raise RuntimeError(f"packet verification failed: {packet_check}")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("experiment_id") != packet_record["experiment_id"]
        or int(profile.get("world_size", 0)) != 4
        or int(profile.get("batch_size", 0)) != 8
    ):
        raise RuntimeError("profile/packet identity drift")

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise RuntimeError("logit capture requires exactly four local ranks")
    if torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("four local BF16-capable GPUs are required")

    checkpoint_record_path = checked_file(root, profile["checkpoint_record"])
    checkpoint_record = read_json(checkpoint_record_path)
    model_dir = resolve(root, str(profile["model_dir"]))
    model_path = model_dir / "model.safetensors"
    if (
        model_path.resolve()
        != resolve(root, str(checkpoint_record["checkpoint_path"]))
        or model_path.stat().st_size != int(checkpoint_record["checkpoint_bytes"])
    ):
        raise RuntimeError("checkpoint record/model identity drift")
    feature_extractor_dir = resolve(root, str(profile["feature_extractor_dir"]))
    vocab_path = checked_file(root, profile["vocab"])
    vocabulary = {
        str(key): int(value) for key, value in read_json(vocab_path).items()
    }
    rows = (
        validation_rows(root, profile)
        if args.split == "validation"
        else test_rows(root, profile)
    )
    rank_rows = [
        row for row in rows if int(row["capture_index"]) % world_size == rank
    ]
    if not rank_rows:
        raise RuntimeError(f"rank {rank} received no rows")

    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    digest_pass = torch.tensor(
        [
            int(
                rank != 0
                or sha256_file(model_path)
                == str(checkpoint_record["checkpoint_sha256"])
            )
        ],
        dtype=torch.int32,
        device=device,
    )
    dist.broadcast(digest_pass, src=0)
    if int(digest_pass.item()) != 1:
        raise RuntimeError("checkpoint SHA-256 drift")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        feature_extractor_dir, local_files_only=True
    )
    if (
        int(feature_extractor.sampling_rate) != 16_000
        or not bool(feature_extractor.do_normalize)
        or not bool(feature_extractor.return_attention_mask)
    ):
        raise RuntimeError("MMS feature-extractor contract failed")
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"strict MMS model load failed: {loading}")
    if int(model.config.vocab_size) != len(vocabulary):
        raise RuntimeError("model/vocabulary size drift")
    model.to(device)
    model.eval()
    loader = DataLoader(
        CaptureDataset(rank_rows, feature_extractor),
        batch_size=int(profile["batch_size"]),
        shuffle=False,
        num_workers=int(profile["dataloader_num_workers"]),
        pin_memory=True,
        persistent_workers=False,
        collate_fn=CaptureCollator(feature_extractor),
    )
    row_by_index = {int(row["capture_index"]): row for row in rows}
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            indices = [int(value) for value in batch.pop("capture_indices")]
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            output_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_values=input_values, attention_mask=attention_mask
                ).logits
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite MMS logits")
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            historical_top_indices = torch.topk(
                log_probabilities, k=2, dim=-1
            ).indices
            for offset, capture_index in enumerate(indices):
                if capture_index % world_size != rank:
                    raise RuntimeError("row reached wrong capture rank")
                frames = int(output_lengths[offset])
                if frames <= 0 or frames > int(logits.shape[1]):
                    raise RuntimeError("invalid model output length")
                matrix = (
                    log_probabilities[offset, :frames]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                if matrix.shape != (frames, len(vocabulary)) or not np.isfinite(
                    matrix
                ).all():
                    raise RuntimeError("captured log-probability contract failed")
                key = f"row_{capture_index:06d}"
                arrays[key] = matrix
                row = row_by_index[capture_index]
                historical_ids = (
                    historical_top_indices[offset, :frames, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                argmax_ids = (
                    log_probabilities[offset, :frames]
                    .argmax(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                greedy = greedy_ctc(historical_ids, vocabulary)
                greedy_argmax = greedy_ctc(argmax_ids, vocabulary)
                greedy_fp16 = greedy_ctc(matrix.argmax(axis=-1), vocabulary)
                records.append(
                    {
                        "array_key": key,
                        "capture_index": capture_index,
                        "id": str(row["id"]),
                        "language": "lin",
                        "output_frames": frames,
                        "greedy_hypothesis": greedy,
                        "greedy_hypothesis_argmax": greedy_argmax,
                        "greedy_hypothesis_fp16": greedy_fp16,
                    }
                )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed": min(
                            (batch_index + 1) * int(profile["batch_size"]),
                            len(rank_rows),
                        ),
                        "total": len(rank_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    records.sort(key=lambda row: int(row["capture_index"]))
    if len(records) != len(rank_rows) or len(arrays) != len(rank_rows):
        raise RuntimeError("rank capture coverage drift")
    output.mkdir(parents=True, exist_ok=True)
    rank_dir = output / f"rank_{rank}"
    rank_dir.mkdir(exist_ok=False)
    records_path = rank_dir / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    logits_path = rank_dir / "log_probs_fp16.npz"
    np.savez_compressed(logits_path, **arrays)
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
        "rank": rank,
        "world_size": world_size,
        "batch_size": int(profile["batch_size"]),
        "rows": len(records),
        "checkpoint_id": checkpoint_record["id"],
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "checkpoint_sha256_verified_on_this_rank": rank == 0,
        "records": records_path.name,
        "records_sha256": sha256_file(records_path),
        "log_probabilities": logits_path.name,
        "log_probabilities_sha256": sha256_file(logits_path),
        "log_probabilities_dtype": "float16",
        "implementation": str(Path(__file__).relative_to(PACKET_SRC)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "precision": "BF16 autocast; FP32 log-softmax; FP16 persisted",
        "tf32": False,
        "external_action": False,
        "submission_created": False,
    }
    write_json_atomic(rank_dir / "TERMINAL.json", terminal)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
