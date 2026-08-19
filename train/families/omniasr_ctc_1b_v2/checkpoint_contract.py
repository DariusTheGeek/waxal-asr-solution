"""Atomic, corruption-detecting, generation-fenced checkpoint contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MethodType
from typing import Any
import uuid
import time


LOCAL_MARKER = "LOCAL_COMMITTED.json"
REMOTE_MARKER = "REMOTE_COMMITTED.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_inventory(step_dir: Path) -> list[dict[str, object]]:
    step_dir = step_dir.resolve()
    if not step_dir.is_dir() or step_dir.is_symlink():
        raise RuntimeError(f"checkpoint is not a real directory: {step_dir}")
    records: list[dict[str, object]] = []
    for path in sorted(step_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"checkpoint symlink is forbidden: {path}")
        relative = path.relative_to(step_dir)
        if relative.parts and relative.parts[0] == "run_evidence":
            continue
        if not path.is_file() or path.name in {LOCAL_MARKER, REMOTE_MARKER}:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError(f"empty checkpoint: {step_dir}")
    return records


def _validate_full_state_layout(
    inventory: list[dict[str, object]], world_size: int
) -> None:
    paths = {str(item["path"]) for item in inventory}
    required_prefixes = ("trainer/", "model/", "optimizer/", "data_reader/")
    if any(not any(path.startswith(prefix) for path in paths) for prefix in required_prefixes):
        raise RuntimeError("checkpoint omits trainer/model/optimizer/data-reader state")
    trainer = [path for path in paths if path.startswith("trainer/rank_")]
    readers = [path for path in paths if path.startswith("data_reader/dp_")]
    if len(trainer) != world_size or len(readers) != world_size:
        raise RuntimeError(
            f"rank-state count drift: trainer={len(trainer)} reader={len(readers)} world={world_size}"
        )


def commit_local_checkpoint(
    step_dir: Path,
    *,
    experiment_id: str,
    packet_digest: str,
    step: int,
    world_size: int,
    lease_generation: int,
    lease_token: str,
) -> dict[str, object]:
    step_dir = step_dir.resolve()
    if step_dir.name != f"step_{step}" or step <= 0:
        raise RuntimeError(f"checkpoint step/path mismatch: {step_dir}")
    marker = step_dir / LOCAL_MARKER
    if marker.exists():
        return verify_local_checkpoint(step_dir)
    inventory = checkpoint_inventory(step_dir)
    _validate_full_state_layout(inventory, world_size)
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "COMMITTED",
        "created_at_utc": _utc_now(),
        "experiment_id": experiment_id,
        "packet_digest": packet_digest,
        "step": step,
        "world_size": world_size,
        "lease_generation": lease_generation,
        "lease_token": lease_token,
        "inventory": inventory,
        "inventory_digest": _json_digest(inventory),
        "full_state": True,
    }
    _write_json_atomic(marker, record)
    return record


def verify_local_checkpoint(
    step_dir: Path, *, require_canonical_name: bool = True
) -> dict[str, object]:
    step_dir = step_dir.resolve()
    marker = step_dir / LOCAL_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise RuntimeError(f"checkpoint lacks local commit marker: {step_dir}")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("status") != "COMMITTED":
        raise RuntimeError(f"invalid local checkpoint marker: {marker}")
    try:
        step = int(record["step"])
        world_size = int(record["world_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("local marker scalar contract drift") from error
    if require_canonical_name and step_dir.name != f"step_{step}":
        raise RuntimeError("local marker step/path mismatch")
    inventory = checkpoint_inventory(step_dir)
    if record.get("inventory") != inventory:
        raise RuntimeError(f"checkpoint inventory corruption: {step_dir}")
    if record.get("inventory_digest") != _json_digest(inventory):
        raise RuntimeError(f"checkpoint inventory digest corruption: {step_dir}")
    _validate_full_state_layout(inventory, world_size)
    return record


def verify_existing_checkpoints(output_dir: Path) -> list[int]:
    checkpoint_dir = output_dir.resolve() / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise RuntimeError(f"invalid checkpoint root: {checkpoint_dir}")
    steps: list[int] = []
    for path in checkpoint_dir.glob("step_*"):
        if path.name.endswith(".tmp"):
            raise RuntimeError(f"incomplete checkpoint directory blocks resume: {path}")
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"unexpected checkpoint entry: {path}")
        record = verify_local_checkpoint(path)
        steps.append(int(record["step"]))
    if len(steps) != len(set(steps)):
        raise RuntimeError("checkpoint steps are not unique")
    return sorted(steps)


def coordinate_resume_preparation(
    output_dir: Path,
    prepare_callback: Any | None = None,
    *,
    timeout_seconds: float = 600.0,
) -> list[int]:
    """Hash checkpoints once on rank zero and release all launch ranks together."""

    token = os.environ.get("WAXAL3_LEASE_TOKEN")
    if not token:
        raise RuntimeError("WAXAL3_LEASE_TOKEN is required for resume coordination")
    rank = int(os.environ.get("RANK", "0"))
    gate = output_dir.resolve() / "runtime_gates" / f"RESUME_{token}.json"
    if rank == 0:
        try:
            steps = verify_existing_checkpoints(output_dir)
            callback_record = (
                prepare_callback(steps) if prepare_callback is not None else None
            )
            value = {
                "schema_version": 1,
                "status": "PASS",
                "lease_token": token,
                "steps": steps,
                "prepared_at_utc": _utc_now(),
                "callback": callback_record,
            }
        except BaseException as error:
            _write_json_atomic(
                gate,
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "lease_token": token,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "prepared_at_utc": _utc_now(),
                },
            )
            raise
        _write_json_atomic(gate, value)
        return steps
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if gate.is_file():
            value = json.loads(gate.read_text(encoding="utf-8"))
            if value.get("lease_token") != token:
                raise RuntimeError("resume coordination token drift")
            if value.get("status") != "PASS":
                raise RuntimeError(f"rank-zero resume preparation failed: {value}")
            return [int(step) for step in value.get("steps", [])]
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for rank-zero resume preparation: {gate}")


@dataclass(frozen=True)
class LeaseGuard:
    store: Path
    experiment_id: str
    packet_digest: str
    generation: int
    token: str

    @property
    def lease_path(self) -> Path:
        return self.store / "leases" / f"{self.experiment_id}.json"

    def assert_current(self) -> dict[str, object]:
        if not self.lease_path.is_file() or self.lease_path.is_symlink():
            raise RuntimeError(f"lease is absent: {self.lease_path}")
        value = json.loads(self.lease_path.read_text(encoding="utf-8"))
        expected = {
            "status": "ACTIVE",
            "experiment_id": self.experiment_id,
            "packet_digest": self.packet_digest,
            "generation": self.generation,
            "token": self.token,
        }
        if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
            raise RuntimeError("stale or foreign remote lease generation")
        return value

    def heartbeat(self, step: int) -> None:
        self.assert_current()
        path = (
            self.store
            / "heartbeats"
            / self.experiment_id
            / f"generation_{self.generation}_{self.token}.json"
        )
        _write_json_atomic(
            path,
            {
                "schema_version": 1,
                "status": "ACTIVE",
                "experiment_id": self.experiment_id,
                "packet_digest": self.packet_digest,
                "generation": self.generation,
                "token": self.token,
                "step": int(step),
                "updated_at_utc": _utc_now(),
            },
        )

    @classmethod
    def from_environment(cls) -> "LeaseGuard":
        required = (
            "WAXAL3_REMOTE_STORE",
            "WAXAL3_EXPERIMENT_ID",
            "WAXAL3_PACKET_DIGEST",
            "WAXAL3_LEASE_GENERATION",
            "WAXAL3_LEASE_TOKEN",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"missing fenced runtime variables: {missing}")
        return cls(
            Path(os.environ["WAXAL3_REMOTE_STORE"]).expanduser().resolve(),
            os.environ["WAXAL3_EXPERIMENT_ID"],
            os.environ["WAXAL3_PACKET_DIGEST"],
            int(os.environ["WAXAL3_LEASE_GENERATION"]),
            os.environ["WAXAL3_LEASE_TOKEN"],
        )


def acquire_lease(
    store: Path,
    *,
    experiment_id: str,
    packet_digest: str,
    token: str,
    takeover: bool,
) -> dict[str, object]:
    store = store.expanduser().resolve()
    lease_dir = store / "leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lease_dir / f".{experiment_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = lease_dir / f"{experiment_id}.json"
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if not takeover:
                raise RuntimeError("an active/prior lease exists; explicit takeover is required")
            generation = int(current["generation"]) + 1
        else:
            generation = 1
        value = {
            "schema_version": 1,
            "status": "ACTIVE",
            "experiment_id": experiment_id,
            "packet_digest": packet_digest,
            "generation": generation,
            "token": token,
            "created_at_utc": _utc_now(),
            "takeover": bool(takeover),
        }
        _write_json_atomic(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return value


class CheckpointCommitter:
    def __init__(
        self,
        *,
        output_dir: Path,
        guard: LeaseGuard,
        world_size: int,
        sweep_boundaries: set[int],
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.guard = guard
        self.world_size = int(world_size)
        self.sweep_boundaries = set(int(step) for step in sweep_boundaries)

    @property
    def remote_namespace(self) -> Path:
        return (
            self.guard.store
            / "checkpoints"
            / self.guard.experiment_id
            / self.guard.packet_digest
        )

    def commit(self, step: int) -> dict[str, object]:
        lease = self.guard.assert_current()
        step_dir = self.output_dir / "checkpoints" / f"step_{step}"
        local = commit_local_checkpoint(
            step_dir,
            experiment_id=self.guard.experiment_id,
            packet_digest=self.guard.packet_digest,
            step=step,
            world_size=self.world_size,
            lease_generation=int(lease["generation"]),
            lease_token=str(lease["token"]),
        )
        namespace = self.remote_namespace
        namespace.mkdir(parents=True, exist_ok=True)
        target = namespace / f"step_{step}"
        if target.exists():
            existing = verify_remote_checkpoint(target)
            existing_local = existing["local"]
            existing_remote = existing["remote"]
            if (
                existing_local != local
                or existing_remote.get("status") != "REMOTE_COMMITTED"
                or existing_remote.get("experiment_id") != self.guard.experiment_id
                or existing_remote.get("packet_digest") != self.guard.packet_digest
                or existing_remote.get("step") != step
            ):
                raise RuntimeError("existing remote checkpoint identity drift")
            return local
        temporary = namespace / f".step_{step}.{uuid.uuid4().hex}.tmp"
        shutil.copytree(step_dir, temporary, symlinks=False)
        copied = verify_local_checkpoint(temporary, require_canonical_name=False)
        if copied != local:
            raise RuntimeError("remote checkpoint copy changed its local commit record")
        evidence = self._snapshot_runtime_evidence(temporary / "run_evidence")
        self.guard.assert_current()
        _write_json_atomic(
            temporary / REMOTE_MARKER,
            {
                "schema_version": 1,
                "status": "REMOTE_COMMITTED",
                "created_at_utc": _utc_now(),
                "experiment_id": self.guard.experiment_id,
                "packet_digest": self.guard.packet_digest,
                "step": step,
                "lease_generation": self.guard.generation,
                "lease_token": self.guard.token,
                "local_marker_sha256": sha256_file(temporary / LOCAL_MARKER),
                "inventory_digest": local["inventory_digest"],
                "evidence_inventory_digest": evidence["inventory_digest"],
                "evidence_files": evidence["files"],
            },
        )
        os.replace(temporary, target)
        self.guard.heartbeat(step)
        self._prune_remote(step)
        return local

    def _snapshot_runtime_evidence(self, destination: Path) -> dict[str, object]:
        destination.mkdir(parents=True, exist_ok=False)
        direct_names = {
            "MODEL_PARTITION.json",
            "RUNTIME_SAFETY_ATTEST.json",
            "RUNTIME_SAFETY_EVENTS.jsonl",
            "RUNTIME_SAFETY_TERMINAL.json",
        }
        sources: list[Path] = []
        for name in direct_names:
            path = self.output_dir / name
            if path.exists():
                sources.append(path)
        sources.extend(sorted(self.output_dir.glob("DATA_SWEEP_*.json")))
        for directory_name in ("early_stopping", "transcriptions", "metrics"):
            directory = self.output_dir / directory_name
            if directory.is_dir() and not directory.is_symlink():
                sources.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
        inventory: list[dict[str, object]] = []
        for source in sources:
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"unsafe runtime evidence source: {source}")
            relative = source.relative_to(self.output_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        inventory.sort(key=lambda item: str(item["path"]))
        record: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "files": len(inventory),
            "inventory": inventory,
            "inventory_digest": _json_digest(inventory),
        }
        _write_json_atomic(destination / "EVIDENCE.json", record)
        return record

    def _prune_remote(self, newest_step: int) -> None:
        """Enforce exact latest-two plus sweep-boundary retention."""

        namespace = self.remote_namespace.resolve()
        steps: list[tuple[int, Path]] = []
        for path in namespace.glob("step_*"):
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"unsafe remote checkpoint entry: {path}")
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError as error:
                raise RuntimeError(f"invalid remote checkpoint name: {path}") from error
            verified = verify_remote_checkpoint(path)
            remote = verified["remote"]
            if (
                remote.get("status") != "REMOTE_COMMITTED"
                or remote.get("experiment_id") != self.guard.experiment_id
                or remote.get("packet_digest") != self.guard.packet_digest
                or remote.get("step") != step
            ):
                raise RuntimeError(f"remote retention identity drift: {path}")
            steps.append((step, path))
        latest = {step for step, _ in sorted(steps)[-2:]}
        keep = latest | self.sweep_boundaries | {newest_step}
        for step, path in steps:
            if step in keep:
                continue
            if path.parent.resolve() != namespace or path.is_symlink():
                raise RuntimeError(f"unsafe remote prune target: {path}")
            shutil.rmtree(path)


def verify_remote_checkpoint(step_dir: Path) -> dict[str, object]:
    step_dir = step_dir.resolve()
    local = verify_local_checkpoint(step_dir)
    marker_path = step_dir / REMOTE_MARKER
    evidence_path = step_dir / "run_evidence/EVIDENCE.json"
    if not marker_path.is_file() or not evidence_path.is_file():
        raise RuntimeError(f"remote checkpoint is not fully committed: {step_dir}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    inventory: list[dict[str, object]] = []
    for item in evidence.get("inventory", []):
        path = step_dir / "run_evidence" / str(item["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"remote evidence file is absent/unsafe: {path}")
        inventory.append(
            {
                "path": str(item["path"]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if (
        marker.get("status") != "REMOTE_COMMITTED"
        or marker.get("experiment_id") != local.get("experiment_id")
        or marker.get("packet_digest") != local.get("packet_digest")
        or marker.get("step") != local.get("step")
        or marker.get("local_marker_sha256") != sha256_file(step_dir / LOCAL_MARKER)
        or marker.get("inventory_digest") != local.get("inventory_digest")
        or evidence.get("status") != "PASS"
        or evidence.get("inventory") != inventory
        or evidence.get("inventory_digest") != _json_digest(inventory)
        or marker.get("evidence_inventory_digest") != evidence.get("inventory_digest")
        or marker.get("evidence_files") != len(inventory)
    ):
        raise RuntimeError(f"remote checkpoint/evidence corruption: {step_dir}")
    return {"local": local, "remote": marker, "evidence": evidence}


def restore_remote_checkpoint(step_dir: Path, output_dir: Path) -> dict[str, object]:
    """Create-only restore of one verified remote checkpoint and its run evidence."""

    step_dir = step_dir.resolve()
    output_dir = output_dir.resolve()
    record = verify_remote_checkpoint(step_dir)
    step = int(record["local"]["step"])
    target = output_dir / "checkpoints" / f"step_{step}"
    if target.exists():
        raise FileExistsError(f"local restore checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        step_dir,
        target,
        ignore=shutil.ignore_patterns("run_evidence", REMOTE_MARKER),
        symlinks=False,
    )
    verify_local_checkpoint(target)
    for item in record["evidence"]["inventory"]:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe evidence restore path: {relative}")
        source = step_dir / "run_evidence" / relative
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or sha256_file(destination) != item["sha256"]:
                raise RuntimeError(f"runtime evidence restore collision: {destination}")
            continue
        shutil.copy2(source, destination)
    return record


def attach_checkpoint_contract(
    trainer: Any,
    *,
    output_dir: Path,
    sweep_boundaries: set[int],
) -> CheckpointCommitter:
    if not hasattr(trainer, "_on_checkpoint_saved") or not hasattr(trainer, "_gangs"):
        raise RuntimeError("pinned Trainer checkpoint integration point disappeared")
    guard = LeaseGuard.from_environment()
    guard.assert_current()
    gangs = trainer._gangs
    committer = CheckpointCommitter(
        output_dir=output_dir,
        guard=guard,
        world_size=int(gangs.root.size),
        sweep_boundaries=sweep_boundaries,
    )
    original = trainer._on_checkpoint_saved

    def on_checkpoint_saved(self: Any, step_nr: int, blocking: bool) -> None:
        if gangs.root.rank == 0:
            committer.commit(int(step_nr))
        gangs.root.barrier()
        original(step_nr, blocking)

    trainer._on_checkpoint_saved = MethodType(on_checkpoint_saved, trainer)
    optimizer = trainer._optimizer

    def assert_fence(optimizer_object: Any, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        del optimizer_object, args, kwargs
        guard.assert_current()

    handle = optimizer.register_step_pre_hook(assert_fence)
    trainer._waxal3_lease_guard = guard
    trainer._waxal3_lease_hook = handle
    trainer._waxal3_checkpoint_committer = committer
    return committer


def attach_boundary_retention(trainer: Any, boundaries: set[int]) -> None:
    manager = trainer._checkpoint_manager
    original = manager.get_stale_step_numbers

    def get_stale(
        self: Any,
        keep_last_n: int | None,
        keep_best_n: int | None,
        keep_every_n_steps: int | None,
    ) -> list[int]:
        stale = original(keep_last_n, keep_best_n, keep_every_n_steps)
        return [step for step in stale if int(step) not in boundaries]

    manager.get_stale_step_numbers = MethodType(get_stale, manager)
