#!/usr/bin/env python3
"""CPU-only prepacket validation for MMS supervised and CPT experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from cpt.contract import repo_root, sha256_file, validate_global_config
from cpt.data import (
    SpeakerInterleavedDistributedSampler,
    StagePaddedDistributedSampler,
)
from cpt.global_cpt import _resume_checkpoint
from supervised.contract import validate_run_config
from supervised.train import select_rows


def _supervised(experiment: Path, profiles: list[Path]) -> dict[str, object]:
    records = []
    for index, profile in enumerate(profiles):
        config = json.loads(json.dumps(__import__("yaml").safe_load(profile.read_text())))
        paths = validate_run_config(
            config,
            experiment_root=experiment,
            run_dir=experiment / "runs" / "PREPACKET_DUMMY",
            require_authorization=False,
            verify_large_model_sha256=index == 0,
        )
        train, validation = select_rows(config, paths["manifest_path"])
        padded = math.ceil(len(train) / int(config["optimizer_padding_multiple"])) * int(
            config["optimizer_padding_multiple"]
        )
        if padded != int(config["expected_padded_train_rows"]):
            raise RuntimeError("supervised preflight padding arithmetic drift")
        records.append(
            {
                "profile": profile.name,
                "phase": config["phase"],
                "train_rows": len(train),
                "validation_rows": len(validation),
                "padded_train_rows": padded,
                "updates_per_epoch": padded
                // int(config["optimizer_padding_multiple"]),
                "large_parent_hash_verified": index == 0,
            }
        )
    return {"method": "supervised_ft", "profiles": records}


def _cpt(experiment: Path, profiles: list[Path]) -> dict[str, object]:
    records = []
    rows = None
    specification = __import__("yaml").safe_load(
        (experiment / "experiment.yaml").read_text(encoding="utf-8")
    )
    recovery = specification.get("recovery")
    for index, profile in enumerate(profiles):
        config = __import__("yaml").safe_load(profile.read_text(encoding="utf-8"))
        paths = validate_global_config(
            config,
            experiment_root=experiment,
            run_dir=experiment / "runs" / "PREPACKET_DUMMY",
            require_authorization=False,
            verify_large_hashes=index == 0,
        )
        if rows is None:
            rows = pq.read_table(paths["audio_manifest"]).to_pylist()
        sampler_class = (
            SpeakerInterleavedDistributedSampler
            if config["sampler_mode"] == "s008_speaker_interleaved"
            else StagePaddedDistributedSampler
        )
        samplers = [
            sampler_class(
                rows,
                rank=rank,
                world_size=int(config["world_size"]),
                per_device_batch_size=int(config["per_device_batch_size"]),
                gradient_accumulation_steps=int(
                    config["gradient_accumulation_steps"]
                ),
                seed=int(config["seed"]),
            )
            for rank in range(int(config["world_size"]))
        ]
        if config["sampler_mode"] == "s008_speaker_interleaved":
            canonical = samplers[0].global_order()[: samplers[0].unique_rows_per_sweep]
            unique = set(canonical)
            sync_padding = 0
        else:
            canonical = samplers[0].global_presentations()
            unique = {
                item.row_index
                for item in canonical
                if not item.synchronization_padding
            }
            sync_padding = sum(
                item.synchronization_padding for item in canonical
            )
        if any(
            list(sampler) != canonical[rank :: int(config["world_size"])]
            for rank, sampler in enumerate(samplers)
        ):
            raise RuntimeError("CPT rank partition drift")
        if len(unique) != len(rows) - int(config["expected_dropped_rows"]):
            raise RuntimeError("CPT unique sweep coverage drift")
        resume_validation = None
        if recovery is not None and config["phase"] == "production":
            checkpoint = repo_root(experiment) / str(recovery["source_checkpoint_path"])
            _, checkpoint_record, lineage = _resume_checkpoint(
                checkpoint,
                experiment_root=experiment,
                config=config,
                expected_config_hash=sha256_file(profile),
                expected_critical_hash="PREPACKET_CROSS_LINEAGE_ONLY",
            )
            resume_validation = {
                "status": "PASS",
                "source_sweep": int(checkpoint_record["sweep"]),
                "source_global_step": int(checkpoint_record["global_step"]),
                "lineage_kind": lineage["kind"],
                "resume_trajectory_hash": lineage["resume_trajectory_hash"],
                "resume_collapse_logs": lineage["resume_collapse_logs"],
            }
        elif recovery is not None:
            resume_validation = {
                "status": "NOT_APPLICABLE_NONRESUME_SMOKE",
            }
        records.append(
            {
                "profile": profile.name,
                "phase": config["phase"],
                "source_rows": len(rows),
                "unique_rows": len(unique),
                "dropped_rows": int(getattr(samplers[0], "dropped_rows", 0)),
                "synchronization_padding_slots": sync_padding,
                "updates_per_sweep": samplers[0].optimizer_updates_per_sweep,
                "large_parent_hashes_verified": index == 0,
                "resume_validation": resume_validation,
            }
        )
    return {"method": "cpt", "profiles": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    specification = __import__("yaml").safe_load(
        (experiment / "experiment.yaml").read_text(encoding="utf-8")
    )
    profiles = [experiment / "profiles/smoke.yaml", experiment / "profiles/production.yaml"]
    if not all(path.is_file() for path in profiles):
        raise FileNotFoundError("smoke.yaml and production.yaml are required")
    method = str(specification["method"])
    if method == "supervised_ft":
        result = _supervised(experiment, profiles)
    elif method == "cpt":
        result = _cpt(experiment, profiles)
    else:
        raise RuntimeError(f"unsupported MMS method: {method}")
    print(json.dumps({"status": "PASS", **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
