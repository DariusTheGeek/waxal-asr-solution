from __future__ import annotations

import json
from pathlib import Path

import pytest
import checkpoint_contract as checkpoint_module

from checkpoint_contract import (
    CheckpointCommitter,
    CheckpointLineage,
    LeaseGuard,
    RETENTION_APPLIED_MARKER,
    acquire_lease,
    commit_local_checkpoint,
    export_source_identity,
    restore_remote_checkpoint,
    select_committed_checkpoint_step,
    verify_epoch_evidence,
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
        model = path / f"model/pp_00/tp_00/sdp_{rank:02d}.pt"
        optimizer = path / f"optimizer/pp_00/tp_00/sdp_{rank:02d}.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        optimizer.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f"model-{rank}".encode())
        optimizer.write_bytes(f"optimizer-{rank}".encode())
    return path


def _write_target_score_prefix(
    output: Path, scores: list[float], *, interval: int
) -> None:
    evidence_dir = output / "early_stopping"
    evidence_dir.mkdir(exist_ok=True)
    history = []
    for epoch, score in enumerate(scores, start=1):
        step = epoch * interval
        history.append(
            {"step": step, "epoch": epoch, "promotion_score": score}
        )
        path = evidence_dir / f"step_{step:08d}.json"
        if not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "PASS",
                        "step": step,
                        "epoch": epoch,
                        "promotion_authority": "target_slot_weighted_raw_q",
                        "promotion_score": score,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    (evidence_dir / "STATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTINUE",
                "promotion_authority": "target_slot_weighted_raw_q",
                "history": history,
            }
        )
        + "\n",
        encoding="utf-8",
    )


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


@pytest.mark.parametrize(
    ("relative", "replacement", "message"),
    [
        ("model/pp_00/tp_00/sdp_01.pt", None, "topology drift"),
        ("optimizer/pp_00/tp_00/sdp_01.pt", None, "topology drift"),
        ("trainer/rank_01.pt", None, "topology drift"),
        ("data_reader/dp_01.pt", None, "topology drift"),
        (
            "model/pp_00/tp_00/sdp_01.pt",
            "model/pp_00/tp_00/sdp_08.pt",
            "topology drift",
        ),
    ],
)
def test_exact_world_topology_rejects_missing_or_wrong_rank(
    tmp_path: Path,
    relative: str,
    replacement: str | None,
    message: str,
) -> None:
    step = _checkpoint(tmp_path, 500)
    target = step / relative
    if replacement is None:
        target.unlink()
    else:
        replacement_path = step / replacement
        replacement_path.parent.mkdir(parents=True, exist_ok=True)
        target.rename(replacement_path)
    with pytest.raises(RuntimeError, match=message):
        commit_local_checkpoint(
            step,
            experiment_id="X0024",
            packet_digest="packet",
            step=500,
            world_size=2,
            lease_generation=1,
            lease_token="host-a",
        )


def test_exact_world_topology_rejects_extra_and_zero_byte_state(
    tmp_path: Path,
) -> None:
    extra_step = _checkpoint(tmp_path / "extra", 500)
    extra = extra_step / "model/pp_01/tp_00/sdp_00.pt"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="topology drift"):
        commit_local_checkpoint(
            extra_step,
            experiment_id="X0024",
            packet_digest="packet",
            step=500,
            world_size=2,
            lease_generation=1,
            lease_token="host-a",
        )

    zero_step = _checkpoint(tmp_path / "zero", 500)
    (zero_step / "optimizer/pp_00/tp_00/sdp_01.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="zero-byte"):
        commit_local_checkpoint(
            zero_step,
            experiment_id="X0024",
            packet_digest="packet",
            step=500,
            world_size=2,
            lease_generation=1,
            lease_token="host-a",
        )


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


def test_exact_checkpoint_selector_can_export_a_nonlatest_best_step() -> None:
    steps = [501, 1_002, 1_503]
    assert select_committed_checkpoint_step(
        steps, resume_latest=False, resume_step=1_002
    ) == 1_002
    assert select_committed_checkpoint_step(
        steps, resume_latest=True, resume_step=None
    ) == 1_503
    assert select_committed_checkpoint_step(
        steps, resume_latest=False, resume_step=None
    ) is None
    with pytest.raises(RuntimeError, match="exactly one"):
        select_committed_checkpoint_step(
            steps, resume_latest=True, resume_step=1_002
        )
    with pytest.raises(RuntimeError, match="not remotely committed"):
        select_committed_checkpoint_step(
            steps, resume_latest=False, resume_step=2_004
        )


def test_export_identity_binds_checkpoint_inventory_and_target_evidence(
    tmp_path: Path,
) -> None:
    step_number = 501
    output = tmp_path / "smoke-output"
    store = tmp_path / "smoke-remote"
    _checkpoint(output, step_number)
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest="packet-v3",
        token="host-a",
        takeover=False,
    )
    guard = LeaseGuard(
        store,
        "X0024",
        "packet-v3",
        int(lease["generation"]),
        str(lease["token"]),
    )
    transcripts = output / "transcriptions"
    transcripts.mkdir()
    for rank in range(2):
        for kind in ("hyp", "ref"):
            (transcripts / f"rank_{rank}.{kind}.txt").write_text(
                f"{rank}-{kind}\n", encoding="utf-8"
            )
    committer = CheckpointCommitter(
        output_dir=output,
        guard=guard,
        world_size=2,
        profile="smoke",
        validation_interval_steps=step_number,
    )
    marker = committer.commit(step_number)
    committer.commit_runtime_evidence(step_number)
    smoke = export_source_identity(
        output,
        remote_store=store,
        experiment_id="X0024",
        packet_digest="packet-v3",
        step=step_number,
        world_size=2,
        profile="smoke",
    )
    assert smoke["source_checkpoint_step"] == step_number
    assert smoke["checkpoint_inventory_digest"] == marker["inventory_digest"]
    assert smoke["target_evidence_sha256"] is None
    with pytest.raises(RuntimeError, match="lacks selected-step target evidence"):
        export_source_identity(
            output,
            remote_store=store,
            experiment_id="X0024",
            packet_digest="packet-v3",
            step=step_number,
            world_size=2,
            profile="production",
        )

    production_output = tmp_path / "production-output"
    production_store = tmp_path / "production-remote"
    _checkpoint(production_output, step_number)
    production_lease = acquire_lease(
        production_store,
        experiment_id="X0024",
        packet_digest="packet-v3",
        token="host-b",
        takeover=False,
    )
    production_guard = LeaseGuard(
        production_store,
        "X0024",
        "packet-v3",
        int(production_lease["generation"]),
        str(production_lease["token"]),
    )
    production_transcripts = production_output / "transcriptions"
    production_transcripts.mkdir()
    for rank in range(2):
        for kind in ("hyp", "ref"):
            (production_transcripts / f"rank_{rank}.{kind}.txt").write_text(
                f"{rank}-{kind}\n", encoding="utf-8"
            )
    evidence_dir = production_output / "early_stopping"
    evidence_dir.mkdir()
    _write_target_score_prefix(production_output, [0.5], interval=step_number)
    production_committer = CheckpointCommitter(
        output_dir=production_output,
        guard=production_guard,
        world_size=2,
        profile="production",
        validation_interval_steps=step_number,
    )
    production_committer.commit(step_number)
    production_committer.commit_runtime_evidence(step_number)
    production = export_source_identity(
        production_output,
        remote_store=production_store,
        experiment_id="X0024",
        packet_digest="packet-v3",
        step=step_number,
        world_size=2,
        profile="production",
    )
    assert production["target_evidence_sha256"] is not None
    with pytest.raises(RuntimeError, match="checkpoint identity drift"):
        export_source_identity(
            production_output,
            remote_store=production_store,
            experiment_id="X0024",
            packet_digest="foreign-packet",
            step=step_number,
            world_size=2,
            profile="production",
        )


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


def test_post_validation_evidence_is_separate_and_preferred_on_restore(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    step = _checkpoint(output, 500)
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest="replacement-packet",
        token="host-a",
        takeover=False,
    )
    guard = LeaseGuard(
        store,
        "X0024",
        "replacement-packet",
        int(lease["generation"]),
        str(lease["token"]),
    )
    transcripts = output / "transcriptions"
    transcripts.mkdir()
    transcript = transcripts / "rank_0.hyp.txt"
    transcript.write_text("before-validation\n", encoding="utf-8")
    committer = CheckpointCommitter(
        output_dir=output,
        guard=guard,
        world_size=2,
        profile="production",
        validation_interval_steps=500,
    )
    committer.commit(500)
    transcript.write_text("after-validation\n", encoding="utf-8")
    _write_target_score_prefix(output, [0.5], interval=500)
    marker = committer.commit_runtime_evidence(500)
    assert marker["status"] == "EPOCH_EVIDENCE_COMMITTED"
    epoch_path = committer.remote_namespace / "epoch_evidence/step_500"
    verify_epoch_evidence(epoch_path)

    restored = tmp_path / "restored"
    restore_remote_checkpoint(committer.remote_namespace / "step_500", restored)
    assert (restored / "transcriptions/rank_0.hyp.txt").read_text(
        encoding="utf-8"
    ) == "after-validation\n"
    assert (restored / "early_stopping/STATE.json").is_file()
    assert (step / "LOCAL_COMMITTED.json").is_file()


def test_remote_retention_keeps_exact_target_top3_plus_latest_full(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest="lean-retention-packet",
        token="host-a",
        takeover=False,
    )
    guard = LeaseGuard(
        store,
        "X0024",
        "lean-retention-packet",
        int(lease["generation"]),
        str(lease["token"]),
    )
    committer = CheckpointCommitter(
        output_dir=output,
        guard=guard,
        world_size=2,
        profile="production",
        validation_interval_steps=500,
    )
    scores = [0.5, 0.8, 0.6, 0.4, 0.9, 0.7]
    expected = [
        {500},
        {500, 1_000},
        {500, 1_000, 1_500},
        {500, 1_000, 1_500, 2_000},
        {1_000, 1_500, 2_500},
        {1_000, 2_500, 3_000},
    ]
    for epoch, score in enumerate(scores, start=1):
        step = epoch * 500
        _checkpoint(output, step)
        committer.commit(step)
        pre_score = {
            int(path.name.removeprefix("step_"))
            for path in committer.remote_namespace.glob("step_*")
            if path.is_dir()
        }
        assert len(pre_score) <= 4
        assert step in pre_score
        _write_target_score_prefix(output, scores[:epoch], interval=500)
        committer.commit_runtime_evidence(step)
        observed = {
            int(path.name.removeprefix("step_"))
            for path in committer.remote_namespace.glob("step_*")
            if path.is_dir()
        }
        assert observed == expected[epoch - 1]
        assert len(observed) <= 4
        retention = json.loads(
            (committer.remote_namespace / RETENTION_APPLIED_MARKER).read_text(
                encoding="utf-8"
            )
        )
        assert retention["status"] == "RETENTION_APPLIED"
        assert retention["newest_full_step"] == step
        assert retention["observed_full_steps_after"] == sorted(observed)
        assert retention["automatic_local_checkpoint_downloads"] is False


def test_audited_packet_lineage_continues_one_top3_namespace_without_relabeling(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    predecessor_digest = "predecessor-packet"
    current_digest = "timeout-repair-packet"
    first_lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest=predecessor_digest,
        token="host-old",
        takeover=False,
    )
    predecessor = CheckpointCommitter(
        output_dir=output,
        guard=LeaseGuard(
            store,
            "X0024",
            predecessor_digest,
            int(first_lease["generation"]),
            str(first_lease["token"]),
        ),
        world_size=2,
        profile="production",
        validation_interval_steps=500,
    )
    scores = [0.5, 0.8, 0.6, 0.9]
    for epoch, score in enumerate(scores, start=1):
        step = epoch * 500
        _checkpoint(output, step)
        predecessor.commit(step)
        _write_target_score_prefix(output, scores[:epoch], interval=500)
        predecessor.commit_runtime_evidence(step)
    assert _remote_steps(predecessor) == {1_000, 1_500, 2_000}

    second_lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest=current_digest,
        token="host-new",
        takeover=True,
    )
    lineage = CheckpointLineage(
        current_packet_digest=current_digest,
        namespace_digest=predecessor_digest,
        predecessor_packet_digest=predecessor_digest,
        predecessor_max_step=2_000,
    )
    continuation = CheckpointCommitter(
        output_dir=output,
        guard=LeaseGuard(
            store,
            "X0024",
            current_digest,
            int(second_lease["generation"]),
            str(second_lease["token"]),
        ),
        world_size=2,
        profile="production",
        validation_interval_steps=500,
        lineage=lineage,
    )
    assert continuation.remote_namespace == predecessor.remote_namespace
    _checkpoint(output, 2_500)
    continuation.commit(2_500)
    assert _remote_steps(continuation) == {1_000, 1_500, 2_000, 2_500}
    _write_target_score_prefix(output, [*scores, 0.7], interval=500)
    continuation.commit_runtime_evidence(2_500)
    assert _remote_steps(continuation) == {1_000, 2_000, 2_500}

    origins = {
        step: checkpoint_module.verify_remote_checkpoint(
            continuation.remote_namespace / f"step_{step}"
        )["local"]["packet_digest"]
        for step in (1_000, 2_000, 2_500)
    }
    assert origins == {
        1_000: predecessor_digest,
        2_000: predecessor_digest,
        2_500: current_digest,
    }
    old_epoch = verify_epoch_evidence(
        continuation.remote_namespace / "epoch_evidence/step_2000"
    )
    new_epoch = verify_epoch_evidence(
        continuation.remote_namespace / "epoch_evidence/step_2500"
    )
    assert old_epoch["packet_digest"] == predecessor_digest
    assert new_epoch["packet_digest"] == current_digest
    applied = json.loads(
        (continuation.remote_namespace / RETENTION_APPLIED_MARKER).read_text(
            encoding="utf-8"
        )
    )
    assert applied["packet_digest"] == current_digest
    assert applied["observed_full_steps_after"] == [1_000, 2_000, 2_500]
    with pytest.raises(RuntimeError, match="predecessor step range"):
        continuation.commit(2_000)


def _primed_production_committer(
    tmp_path: Path, *, packet_digest: str
) -> tuple[Path, CheckpointCommitter, list[float]]:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest=packet_digest,
        token="host-a",
        takeover=False,
    )
    guard = LeaseGuard(
        store,
        "X0024",
        packet_digest,
        int(lease["generation"]),
        str(lease["token"]),
    )
    committer = CheckpointCommitter(
        output_dir=output,
        guard=guard,
        world_size=2,
        profile="production",
        validation_interval_steps=500,
    )
    scores = [0.5, 0.8, 0.6, 0.4, 0.9]
    for epoch in range(1, 5):
        step = epoch * 500
        _checkpoint(output, step)
        committer.commit(step)
        _write_target_score_prefix(output, scores[:epoch], interval=500)
        committer.commit_runtime_evidence(step)
    return output, committer, scores


def _remote_steps(committer: CheckpointCommitter) -> set[int]:
    return {
        int(path.name.removeprefix("step_"))
        for path in committer.remote_namespace.glob("step_*")
        if path.is_dir()
    }


def test_interrupted_remote_copy_is_discarded_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest="copy-replay-packet",
        token="host-a",
        takeover=False,
    )
    committer = CheckpointCommitter(
        output_dir=output,
        guard=LeaseGuard(
            store,
            "X0024",
            "copy-replay-packet",
            int(lease["generation"]),
            str(lease["token"]),
        ),
        world_size=2,
        profile="smoke",
        validation_interval_steps=500,
    )
    _checkpoint(output, 500)

    def interrupted_copytree(source: Path, destination: Path, **_: object) -> None:
        del source
        destination.mkdir(parents=True)
        (destination / "partial-copy").write_bytes(b"interrupted")
        raise RuntimeError("simulated copy preemption")

    with monkeypatch.context() as scoped:
        scoped.setattr(checkpoint_module.shutil, "copytree", interrupted_copytree)
        with pytest.raises(RuntimeError, match="simulated copy preemption"):
            committer.commit(500)
    assert len(list(committer.remote_namespace.glob(".step_500.*.tmp"))) == 1

    committer.commit(500)
    committer.commit_runtime_evidence(500)
    assert _remote_steps(committer) == {500}
    assert not list(committer.remote_namespace.glob(".step_500.*.tmp"))
    events = list((committer.remote_namespace / "recovery_events").glob("RECOVERY_*.json"))
    assert events
    assert any(
        action["kind"] == "discard_incomplete_remote_copy"
        for action in json.loads(events[-1].read_text(encoding="utf-8"))["actions"]
    )


def test_interrupted_pre_score_prune_replays_a_partially_deleted_obsolete_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, committer, scores = _primed_production_committer(
        tmp_path, packet_digest="pending-prune-replay-packet"
    )
    _checkpoint(output, 2_500)
    original_rmtree = checkpoint_module.shutil.rmtree

    def interrupted_rmtree(path: Path) -> None:
        target = Path(path)
        assert target.name == "step_2000"
        next(item for item in target.rglob("*") if item.is_file()).unlink()
        raise RuntimeError("simulated pre-score prune preemption")

    with monkeypatch.context() as scoped:
        scoped.setattr(checkpoint_module.shutil, "rmtree", interrupted_rmtree)
        with pytest.raises(RuntimeError, match="pre-score prune preemption"):
            committer.commit(2_500)
    assert (committer.remote_namespace / "step_2000").is_dir()

    # Replay trusts the durable pending intent for obsolete-state deletion,
    # while rehashing every state that the intent requires us to retain.
    assert checkpoint_module.shutil.rmtree is original_rmtree
    committer.commit(2_500)
    _write_target_score_prefix(output, scores, interval=500)
    committer.commit_runtime_evidence(2_500)
    assert _remote_steps(committer) == {1_000, 1_500, 2_500}


def test_fresh_process_launch_replays_partial_prune_before_strict_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, interrupted, _scores = _primed_production_committer(
        tmp_path, packet_digest="fresh-launch-replay-packet"
    )
    _checkpoint(output, 2_500)

    def interrupted_rmtree(path: Path) -> None:
        target = Path(path)
        assert target.name == "step_2000"
        next(item for item in target.rglob("*") if item.is_file()).unlink()
        raise RuntimeError("simulated fresh-process prune preemption")

    with monkeypatch.context() as scoped:
        scoped.setattr(checkpoint_module.shutil, "rmtree", interrupted_rmtree)
        with pytest.raises(RuntimeError, match="fresh-process prune preemption"):
            interrupted.commit(2_500)
    assert (interrupted.remote_namespace / "step_2000").is_dir()

    # Construct a genuinely new generation and committer, as a replacement
    # launcher would after the original process is gone.
    lease = acquire_lease(
        interrupted.guard.store,
        experiment_id="X0024",
        packet_digest="fresh-launch-replay-packet",
        token="host-b",
        takeover=True,
    )
    replacement = CheckpointCommitter(
        output_dir=output,
        guard=LeaseGuard(
            interrupted.guard.store,
            "X0024",
            "fresh-launch-replay-packet",
            int(lease["generation"]),
            str(lease["token"]),
        ),
        world_size=2,
        profile="production",
        validation_interval_steps=500,
    )
    recovery, records = replacement.recover_for_launch()
    assert recovery["status"] == "PASS"
    assert recovery["lease_generation"] == 2
    assert sorted(records) == [500, 1_000, 1_500, 2_500]
    assert any(
        action["kind"] == "finish_pre_score_retention"
        for action in recovery["recovery_actions"]
    )
    assert not (replacement.remote_namespace / "step_2000").exists()


def test_interrupted_scored_prune_replays_a_partially_deleted_obsolete_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, committer, scores = _primed_production_committer(
        tmp_path, packet_digest="desired-prune-replay-packet"
    )
    _checkpoint(output, 2_500)
    committer.commit(2_500)
    _write_target_score_prefix(output, scores, interval=500)

    def interrupted_rmtree(path: Path) -> None:
        target = Path(path)
        assert target.name == "step_500"
        next(item for item in target.rglob("*") if item.is_file()).unlink()
        raise RuntimeError("simulated scored prune preemption")

    with monkeypatch.context() as scoped:
        scoped.setattr(checkpoint_module.shutil, "rmtree", interrupted_rmtree)
        with pytest.raises(RuntimeError, match="scored prune preemption"):
            committer.commit_runtime_evidence(2_500)
    assert (committer.remote_namespace / "step_500").is_dir()

    committer.commit_runtime_evidence(2_500)
    assert _remote_steps(committer) == {1_000, 1_500, 2_500}
    applied = json.loads(
        (committer.remote_namespace / RETENTION_APPLIED_MARKER).read_text(
            encoding="utf-8"
        )
    )
    assert applied["newest_full_step"] == 2_500


def test_interrupted_applied_marker_is_reconstructed_from_durable_desire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, committer, scores = _primed_production_committer(
        tmp_path, packet_digest="applied-marker-replay-packet"
    )
    _checkpoint(output, 2_500)
    committer.commit(2_500)
    _write_target_score_prefix(output, scores, interval=500)
    original_write = checkpoint_module._write_json_atomic

    def interrupted_applied_write(path: Path, value: object) -> None:
        if (
            path.name == RETENTION_APPLIED_MARKER
            and isinstance(value, dict)
            and value.get("status") == "RETENTION_APPLIED"
            and int(value.get("newest_full_step", -1)) == 2_500
        ):
            raise RuntimeError("simulated applied-marker preemption")
        original_write(path, value)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            checkpoint_module, "_write_json_atomic", interrupted_applied_write
        )
        with pytest.raises(RuntimeError, match="applied-marker preemption"):
            committer.commit_runtime_evidence(2_500)
    assert _remote_steps(committer) == {1_000, 1_500, 2_500}

    # Re-entering the full-checkpoint commit must finish the scored intent
    # before consulting the formerly stale RETENTION_APPLIED record.
    committer.commit(2_500)
    committer.commit_runtime_evidence(2_500)
    assert _remote_steps(committer) == {1_000, 1_500, 2_500}
    events = list((committer.remote_namespace / "recovery_events").glob("RECOVERY_*.json"))
    assert any(
        action["kind"] == "finish_scored_retention"
        for event in events
        for action in json.loads(event.read_text(encoding="utf-8"))["actions"]
    )


def test_smoke_resume_retention_replaces_source_with_newest_full(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = tmp_path / "remote"
    lease = acquire_lease(
        store,
        experiment_id="X0024",
        packet_digest="smoke-retention-packet",
        token="host-a",
        takeover=False,
    )
    guard = LeaseGuard(
        store,
        "X0024",
        "smoke-retention-packet",
        int(lease["generation"]),
        str(lease["token"]),
    )
    committer = CheckpointCommitter(
        output_dir=output,
        guard=guard,
        world_size=2,
        profile="smoke",
        validation_interval_steps=1,
    )
    for step in (2, 3):
        _checkpoint(output, step)
        committer.commit(step)
        committer.commit_runtime_evidence(step)
    observed = sorted(
        int(path.name.removeprefix("step_"))
        for path in committer.remote_namespace.glob("step_*")
        if path.is_dir()
    )
    assert observed == [3]
