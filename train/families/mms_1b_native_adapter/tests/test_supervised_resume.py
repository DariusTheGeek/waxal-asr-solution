from __future__ import annotations

import json
from pathlib import Path

import pytest

from supervised.model import (
    checkpoint_artifact_hashes,
    require_resumable_checkpoint,
)


def _committed_checkpoint(tmp_path: Path, *, world_size: int = 2) -> Path:
    run_dir = tmp_path / "RUN0001_20260804T000000000000Z_test"
    checkpoint = run_dir / "checkpoints/checkpoint-5"
    checkpoint.mkdir(parents=True)
    payloads = {
        "model.safetensors": b"model",
        "config.json": b"{}\n",
        "trainer_state.json": json.dumps({"global_step": 5, "max_steps": 10}).encode(),
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "training_args.bin": b"arguments",
        **{
            f"rng_state_{rank}.pth": f"rng-{rank}".encode()
            for rank in range(world_size)
        },
    }
    for name, payload in payloads.items():
        (checkpoint / name).write_bytes(payload)
    hashes = checkpoint_artifact_hashes(checkpoint)
    events = run_dir / "retention_events"
    events.mkdir()
    (events / "step-5.complete.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "event_status": "COMPLETED",
                "checkpoint_records": [
                    {
                        "step": 5,
                        "path": "checkpoints/checkpoint-5",
                        "decision": "retained",
                        "metric_value": 0.5,
                        "hashes": hashes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_exact_resume_requires_and_verifies_complete_rank_state(tmp_path: Path) -> None:
    checkpoint = _committed_checkpoint(tmp_path)
    record = require_resumable_checkpoint(
        checkpoint,
        checkpoint_root=checkpoint.parent,
        world_size=2,
    )
    assert record["step"] == 5
    assert len(record["checkpoint_sha256"]) == 64
    assert {item["name"] for item in record["files"]} >= {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_0.pth",
        "rng_state_1.pth",
    }


def test_exact_resume_rejects_state_content_drift(tmp_path: Path) -> None:
    checkpoint = _committed_checkpoint(tmp_path)
    (checkpoint / "optimizer.pt").write_bytes(b"X" * len(b"optimizer"))
    with pytest.raises(RuntimeError, match="does not match its commit event"):
        require_resumable_checkpoint(
            checkpoint,
            checkpoint_root=checkpoint.parent,
            world_size=2,
        )


def test_exact_resume_rejects_missing_rank_rng(tmp_path: Path) -> None:
    checkpoint = _committed_checkpoint(tmp_path, world_size=1)
    with pytest.raises(RuntimeError, match="missing full state"):
        require_resumable_checkpoint(
            checkpoint,
            checkpoint_root=checkpoint.parent,
            world_size=2,
        )
