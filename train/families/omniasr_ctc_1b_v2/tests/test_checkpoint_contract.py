from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint_contract import (
    LeaseGuard,
    acquire_lease,
    commit_local_checkpoint,
    verify_existing_checkpoints,
)


def _checkpoint(root: Path, step: int, world_size: int = 2) -> Path:
    path = root / "checkpoints" / f"step_{step}"
    for rank in range(world_size):
        trainer = path / "trainer" / f"rank_{rank:02d}.pt"
        reader = path / "data_reader" / f"dp_{rank:02d}.pt"
        trainer.parent.mkdir(parents=True, exist_ok=True)
        reader.parent.mkdir(parents=True, exist_ok=True)
        trainer.write_bytes(f"trainer-{rank}".encode())
        reader.write_bytes(f"reader-{rank}".encode())
    model = path / "model/pp_00/tp_00/sdp_00.pt"
    optimizer = path / "optimizer/pp_00/tp_00/sdp_00.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    optimizer.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    optimizer.write_bytes(b"optimizer")
    return path


def test_full_checkpoint_commit_and_corruption_rejection(tmp_path: Path) -> None:
    step = _checkpoint(tmp_path, 500)
    commit_local_checkpoint(
        step,
        experiment_id="X0010",
        packet_digest="abc",
        step=500,
        world_size=2,
        lease_generation=1,
        lease_token="host-a",
    )
    assert verify_existing_checkpoints(tmp_path) == [500]
    (step / "model/pp_00/tp_00/sdp_00.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="inventory corruption"):
        verify_existing_checkpoints(tmp_path)


def test_incomplete_tmp_checkpoint_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "checkpoints/step_12.tmp").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        verify_existing_checkpoints(tmp_path)


def test_multiple_checkpoint_steps_are_returned_in_numeric_order(
    tmp_path: Path,
) -> None:
    for step_number in (500, 1_000, 8_703):
        step = _checkpoint(tmp_path, step_number)
        commit_local_checkpoint(
            step,
            experiment_id="X0009",
            packet_digest="packet",
            step=step_number,
            world_size=2,
            lease_generation=1,
            lease_token="host-a",
        )
    assert verify_existing_checkpoints(tmp_path) == [500, 1_000, 8_703]


def test_generation_takeover_fences_old_host(tmp_path: Path) -> None:
    first = acquire_lease(
        tmp_path,
        experiment_id="X0010",
        packet_digest="packet",
        token="host-a",
        takeover=False,
    )
    old = LeaseGuard(tmp_path, "X0010", "packet", int(first["generation"]), "host-a")
    old.assert_current()
    second = acquire_lease(
        tmp_path,
        experiment_id="X0010",
        packet_digest="packet",
        token="host-b",
        takeover=True,
    )
    assert second["generation"] == 2
    with pytest.raises(RuntimeError, match="stale or foreign"):
        old.assert_current()
    LeaseGuard(tmp_path, "X0010", "packet", 2, "host-b").assert_current()


def test_lease_identity_is_bound_to_packet(tmp_path: Path) -> None:
    acquire_lease(
        tmp_path,
        experiment_id="X0009",
        packet_digest="packet-a",
        token="host-a",
        takeover=False,
    )
    foreign = LeaseGuard(tmp_path, "X0009", "packet-b", 1, "host-a")
    with pytest.raises(RuntimeError, match="stale or foreign"):
        foreign.assert_current()
