from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cpt.contract import resume_trajectory_hash, resume_trajectory_payload
from cpt.global_cpt import (
    _collapse_logs_at_step,
    _resume_bounds,
    _validate_checkpoint_files,
)
from cpt.contract import sha256_file


def test_resume_trajectory_ignores_only_non_optimizing_identity_fields() -> None:
    source = {
        "experiment_id": "X0008",
        "minimum_gpu_memory_bytes": 75_000_000_000,
        "phase": "production",
        "log_every_steps": 100,
        "resume_policy": "sweep_boundary_same_packet",
        "learning_rate": 1.0e-4,
        "seed": 42,
    }
    child = {
        **source,
        "experiment_id": "X0012",
        "phase": "smoke",
        "resume_policy": "sweep_boundary_verified_lineage",
        "minimum_gpu_memory_bytes": 38_000_000_000,
    }
    assert resume_trajectory_payload(source) == resume_trajectory_payload(child)
    assert resume_trajectory_hash(source) == resume_trajectory_hash(child)
    child["learning_rate"] = 2.0e-4
    assert resume_trajectory_hash(source) != resume_trajectory_hash(child)


def test_resume_trajectory_includes_health_check_cadence() -> None:
    source = {"log_every_steps": 100, "codebook_collapse_floor": 5.0}
    changed = {**source, "log_every_steps": 1}
    assert resume_trajectory_hash(source) != resume_trajectory_hash(changed)


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (1, (2, 10, 100)),
        (9, (10, 10, 100)),
        (10, (11, 15, 150)),
        (14, (15, 15, 150)),
    ],
)
def test_generic_resume_routes_every_complete_sweep(
    completed: int, expected: tuple[int, int, int]
) -> None:
    config = {
        "updates_per_sweep": 10,
        "gate_sweep": 10,
        "scheduler_horizon_sweeps": 15,
        "smoke_updates": 2,
    }
    assert (
        _resume_bounds(
            mode="resume",
            completed_sweep=completed,
            global_step=completed * 10,
            config=config,
        )
        == expected
    )


def test_resume_probe_is_short_and_stays_inside_next_sweep() -> None:
    config = {
        "updates_per_sweep": 4_352,
        "gate_sweep": 10,
        "scheduler_horizon_sweeps": 15,
        "smoke_updates": 200,
    }
    assert _resume_bounds(
        mode="resume-probe",
        completed_sweep=2,
        global_step=8_704,
        config=config,
    ) == (3, 3, 8_904)


def _complete_checkpoint(
    root: Path, *, sweep: int, updates: int, world_size: int = 4
) -> dict[str, object]:
    checkpoint = root / f"sweep-{sweep:02d}"
    checkpoint.mkdir()
    global_step = sweep * updates
    (checkpoint / "adapter.lin.safetensors").write_bytes(b"adapter-state")
    torch.save(
        {
            "schema_version": 1,
            "sweep": sweep,
            "global_step": global_step,
            "optimizer": {"state": {index: {} for index in range(288)}},
            "scheduler": {
                "last_epoch": global_step,
                "_step_count": global_step + 1,
            },
            "config_sha256": "config",
            "training_critical_hash": "critical",
            "collapse_logs": 0,
        },
        checkpoint / "optimizer_scheduler.pt",
    )
    for rank in range(world_size):
        torch.save(
            {
                "schema_version": 1,
                "rank": rank,
                "sweep": sweep,
                "global_step": global_step,
                "cpu_rng": torch.arange(8, dtype=torch.uint8),
                "cuda_rng": torch.arange(8, dtype=torch.uint8),
            },
            checkpoint / f"rng.rank-{rank}.pt",
        )
    names = [
        "adapter.lin.safetensors",
        "optimizer_scheduler.pt",
        *(f"rng.rank-{rank}.pt" for rank in range(world_size)),
    ]
    files = [
        {
            "name": name,
            "bytes": (checkpoint / name).stat().st_size,
            "sha256": sha256_file(checkpoint / name),
        }
        for name in names
    ]
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "resumable": True,
        "sweep": sweep,
        "global_step": global_step,
        "config_sha256": "config",
        "training_critical_hash": "critical",
        "collapse_logs": 0,
        "files": files,
        "adapter": files[0],
    }


def test_complete_checkpoint_validation_accepts_any_pre_horizon_sweep(
    tmp_path: Path,
) -> None:
    record = _complete_checkpoint(tmp_path, sweep=3, updates=7)
    assert _validate_checkpoint_files(
        tmp_path / "sweep-03",
        record,
        horizon_sweeps=15,
        updates_per_sweep=7,
    ) == (3, 21)


def test_complete_checkpoint_validation_fails_closed_on_state_drift(
    tmp_path: Path,
) -> None:
    record = _complete_checkpoint(tmp_path, sweep=3, updates=7)
    with (tmp_path / "sweep-03/rng.rank-2.pt").open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RuntimeError, match="file/hash drift"):
        _validate_checkpoint_files(
            tmp_path / "sweep-03",
            record,
            horizon_sweeps=15,
            updates_per_sweep=7,
        )


def test_eight_rank_checkpoint_requires_all_eight_rng_states(tmp_path: Path) -> None:
    record = _complete_checkpoint(tmp_path, sweep=2, updates=11, world_size=8)
    assert _validate_checkpoint_files(
        tmp_path / "sweep-02",
        record,
        horizon_sweeps=15,
        updates_per_sweep=11,
        world_size=8,
    ) == (2, 22)
    (tmp_path / "sweep-02/rng.rank-7.pt").unlink()
    with pytest.raises(RuntimeError, match="file/hash drift"):
        _validate_checkpoint_files(
            tmp_path / "sweep-02",
            record,
            horizon_sweeps=15,
            updates_per_sweep=11,
            world_size=8,
        )


def test_collapse_health_state_is_reconstructed_at_boundary(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {"step": 10, "codevector_perplexity": 4.0},
        {"step": 20, "codevector_perplexity": 4.5},
        {"step": 30, "codevector_perplexity": 8.0},
        {"step": 40, "codevector_perplexity": 3.0},
    ]
    metrics.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert _collapse_logs_at_step(metrics, global_step=30, collapse_floor=5.0) == 0
    assert _collapse_logs_at_step(metrics, global_step=40, collapse_floor=5.0) == 1
