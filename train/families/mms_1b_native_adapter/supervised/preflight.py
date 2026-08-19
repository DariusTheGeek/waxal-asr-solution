#!/usr/bin/env python3
"""Read-only exact-run preflight and adversarial-review hash emitter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .contract import (
    experiment_root_from,
    read_json,
    require_free_gpus,
    require_run_disk_capacity,
    training_critical_hash,
    validate_run_config,
)
from .train import select_rows, verify_audio_payloads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--verify-large-model-sha256", action="store_true")
    parser.add_argument("--require-free-gpus", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    experiment_root = experiment_root_from(config_path)
    config = read_json(config_path)
    paths = validate_run_config(
        config,
        experiment_root=experiment_root,
        run_dir=run_dir,
        require_authorization=False,
        verify_large_model_sha256=bool(args.verify_large_model_sha256),
    )
    train, validation = select_rows(config, paths["manifest_path"])
    padding_multiple = int(config["optimizer_padding_multiple"])
    padded = math.ceil(len(train) / padding_multiple) * padding_multiple
    updates = padded // padding_multiple
    expected_updates = int(config["expected_updates_per_epoch"])
    if updates != expected_updates:
        raise RuntimeError(
            f"optimizer updates/epoch drift: {updates} != {expected_updates}"
        )
    audio = verify_audio_payloads(
        train + validation,
        paths["audio_root"],
        verify_sha256=bool(args.verify_audio),
    )
    if args.require_free_gpus:
        require_free_gpus()
    free_bytes, required_bytes = require_run_disk_capacity(
        experiment_root, config, paths
    )
    critical = training_critical_hash(
        experiment_root=experiment_root,
        config_path=config_path,
        paths=paths,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": run_dir.name,
                "phase": config["phase"],
                "language": config["language"],
                "training_authorized": config["training_authorized"],
                "train_rows": len(train),
                "padded_train_rows": padded,
                "padding_duplicates": padded - len(train),
                "validation_rows": len(validation),
                "updates_per_epoch": updates,
                "maximum_epochs": config["max_epochs"],
                "maximum_updates": updates * int(config["max_epochs"]),
                "global_batch": padding_multiple,
                "audio_verification": audio,
                "large_model_sha256_verified": bool(args.verify_large_model_sha256),
                "disk_free_bytes": free_bytes,
                "disk_required_bytes": required_bytes,
                "disk_headroom_bytes": free_bytes - required_bytes,
                "free_gpu_gate_checked": bool(args.require_free_gpus),
                "training_critical_hash": critical,
                "review_line": f"training-critical-hash: {critical}",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
