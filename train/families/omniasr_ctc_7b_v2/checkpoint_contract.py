"""Atomic, corruption-detecting, generation-fenced checkpoint contract."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from types import MethodType
from typing import Any
import uuid
import time


LOCAL_MARKER = "LOCAL_COMMITTED.json"
REMOTE_MARKER = "REMOTE_COMMITTED.json"
EPOCH_EVIDENCE_MARKER = "EPOCH_EVIDENCE_COMMITTED.json"
RETENTION_DESIRED_MARKER = "RETENTION_DESIRED.json"
RETENTION_APPLIED_MARKER = "RETENTION_APPLIED.json"
RETENTION_PENDING_SCORE_MARKER = "RETENTION_PENDING_SCORE.json"


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
    if world_size <= 0:
        raise RuntimeError(f"invalid checkpoint world size: {world_size}")
    paths: set[str] = set()
    for item in inventory:
        path = str(item.get("path", ""))
        try:
            size = int(item.get("bytes", -1))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid checkpoint byte count: {item}") from error
        if not path or size <= 0:
            raise RuntimeError(f"zero-byte or malformed checkpoint entry: {item}")
        if path in paths:
            raise RuntimeError(f"duplicate checkpoint inventory path: {path}")
        paths.add(path)

    expected = {
        *(f"trainer/rank_{rank:02d}.pt" for rank in range(world_size)),
        *(f"data_reader/dp_{rank:02d}.pt" for rank in range(world_size)),
        *(
            f"model/pp_00/tp_00/sdp_{rank:02d}.pt"
            for rank in range(world_size)
        ),
        *(
            f"optimizer/pp_00/tp_00/sdp_{rank:02d}.pt"
            for rank in range(world_size)
        ),
    }
    state_path = re.compile(
        r"^(?:trainer/|data_reader/|model/|optimizer/).+"
    )
    observed = {path for path in paths if state_path.fullmatch(path)}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "exact FSDP checkpoint topology drift: "
            f"world={world_size} missing={missing} unexpected={unexpected}"
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


def select_committed_checkpoint_step(
    steps: list[int], *, resume_latest: bool, resume_step: int | None
) -> int | None:
    """Resolve latest or exact checkpoint selection without silently falling back."""

    if resume_latest and resume_step is not None:
        raise RuntimeError("choose exactly one of latest or exact checkpoint selection")
    normalized = sorted(set(int(step) for step in steps))
    if any(step <= 0 for step in normalized):
        raise RuntimeError(f"invalid committed checkpoint steps: {normalized}")
    if resume_step is not None:
        selected = int(resume_step)
        if selected <= 0 or selected not in normalized:
            raise RuntimeError(
                f"requested checkpoint is not remotely committed: {selected}"
            )
        return selected
    if resume_latest:
        if not normalized:
            raise RuntimeError("latest checkpoint requested but none is committed")
        return normalized[-1]
    return None


def export_source_identity(
    output_dir: Path,
    *,
    remote_store: Path,
    experiment_id: str,
    packet_digest: str,
    step: int,
    world_size: int,
    profile: str,
) -> dict[str, object]:
    """Bind an export to a verified checkpoint and its target-score evidence."""

    if profile not in {"smoke", "production"}:
        raise RuntimeError(f"unsupported OmniASR 7B export profile: {profile}")
    output_dir = output_dir.expanduser().resolve()
    step_dir = output_dir / "checkpoints" / f"step_{step}"
    marker_path = step_dir / LOCAL_MARKER
    local = verify_local_checkpoint(step_dir)
    lineage = checkpoint_lineage_from_environment(packet_digest)
    source_packet_digest = str(local.get("packet_digest", ""))
    expected = {
        "experiment_id": experiment_id,
        "step": int(step),
        "world_size": int(world_size),
    }
    if (
        any(local.get(key) != value for key, value in expected.items())
        or not lineage.accepts(source_packet_digest, step)
    ):
        raise RuntimeError(
            "export source checkpoint identity drift: "
            f"expected={expected} lineage={lineage} observed="
            f"{ {key: local.get(key) for key in (*expected, 'packet_digest')} }"
        )

    evidence_path = output_dir / "early_stopping" / f"step_{step:08d}.json"
    evidence_sha256: str | None = None
    if evidence_path.exists():
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise RuntimeError(f"unsafe target evidence path: {evidence_path}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            not isinstance(evidence, dict)
            or evidence.get("status") != "PASS"
            or int(evidence.get("step", -1)) != step
            or evidence.get("promotion_authority")
            != "target_slot_weighted_raw_q"
        ):
            raise RuntimeError("export target evidence identity drift")
        evidence_sha256 = sha256_file(evidence_path)
    elif profile == "production":
        raise RuntimeError(
            f"production export lacks selected-step target evidence: {evidence_path}"
        )

    def optional_safe_hash(path: Path) -> str | None:
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsafe optional export evidence path: {path}")
        return sha256_file(path)

    state_sha256 = optional_safe_hash(output_dir / "early_stopping" / "STATE.json")
    terminal_sha256 = optional_safe_hash(
        output_dir / "early_stopping" / "EARLY_STOP.json"
    )
    epoch_evidence = verified_epoch_evidence_identity(
        remote_store,
        experiment_id=experiment_id,
        packet_digest=source_packet_digest,
        namespace_digest=lineage.namespace_digest,
        step=step,
        restored_output_dir=output_dir,
    )
    epoch_inventory = {
        str(item["path"]): item
        for item in epoch_evidence["epoch_evidence_inventory"]
    }
    required_evidence_paths = {
        *(f"transcriptions/rank_{rank}.{kind}.txt" for rank in range(world_size) for kind in ("hyp", "ref")),
    }
    if profile == "production":
        required_evidence_paths.update(
            {
                "early_stopping/STATE.json",
                f"early_stopping/step_{step:08d}.json",
            }
        )
        if terminal_sha256 is not None:
            required_evidence_paths.add("early_stopping/EARLY_STOP.json")
    if not required_evidence_paths.issubset(epoch_inventory):
        raise RuntimeError(
            "post-validation commit lacks required export evidence: "
            f"{sorted(required_evidence_paths - set(epoch_inventory))}"
        )
    return {
        "experiment_id": experiment_id,
        "packet_digest": packet_digest,
        "source_checkpoint_packet_digest": source_packet_digest,
        "checkpoint_namespace_digest": lineage.namespace_digest,
        "profile": profile,
        "source_checkpoint_step": int(step),
        "source_checkpoint_world_size": int(world_size),
        "checkpoint_inventory_digest": str(local["inventory_digest"]),
        "source_local_marker_sha256": sha256_file(marker_path),
        "target_evidence_sha256": evidence_sha256,
        "target_state_sha256": state_sha256,
        "early_stop_terminal_sha256": terminal_sha256,
        **epoch_evidence,
    }


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


@dataclass(frozen=True)
class CheckpointLineage:
    """Bind one logical checkpoint namespace across one audited packet repair."""

    current_packet_digest: str
    namespace_digest: str
    predecessor_packet_digest: str | None = None
    predecessor_max_step: int | None = None

    def __post_init__(self) -> None:
        if not self.current_packet_digest or not self.namespace_digest:
            raise ValueError("checkpoint lineage digests must be non-empty")
        has_predecessor = self.predecessor_packet_digest is not None
        has_boundary = self.predecessor_max_step is not None
        if has_predecessor != has_boundary:
            raise ValueError("checkpoint lineage predecessor fields are atomic")
        if not has_predecessor:
            if self.namespace_digest != self.current_packet_digest:
                raise ValueError("unlineaged checkpoint namespace must use current digest")
            return
        assert self.predecessor_packet_digest is not None
        assert self.predecessor_max_step is not None
        if (
            self.predecessor_packet_digest == self.current_packet_digest
            or self.namespace_digest != self.predecessor_packet_digest
            or self.predecessor_max_step <= 0
        ):
            raise ValueError("invalid predecessor checkpoint lineage")

    def accepts(self, packet_digest: object, step: int) -> bool:
        """Accept only predecessor steps at/before the frozen repair boundary."""

        if step <= 0 or not isinstance(packet_digest, str):
            return False
        if self.predecessor_packet_digest is None:
            return packet_digest == self.current_packet_digest
        assert self.predecessor_max_step is not None
        if step <= self.predecessor_max_step:
            return packet_digest == self.predecessor_packet_digest
        return packet_digest == self.current_packet_digest


def checkpoint_lineage_from_environment(current_packet_digest: str) -> CheckpointLineage:
    """Resolve the packet-audited lineage passed by the launcher."""

    namespace = os.environ.get(
        "WAXAL3_CHECKPOINT_NAMESPACE_DIGEST", current_packet_digest
    )
    predecessor = os.environ.get("WAXAL3_PREDECESSOR_PACKET_DIGEST")
    raw_boundary = os.environ.get("WAXAL3_PREDECESSOR_MAX_STEP")
    if predecessor is None and raw_boundary is None:
        return CheckpointLineage(current_packet_digest, namespace)
    if predecessor is None or raw_boundary is None:
        raise RuntimeError("incomplete checkpoint lineage environment")
    try:
        boundary = int(raw_boundary)
    except ValueError as error:
        raise RuntimeError("invalid predecessor checkpoint boundary") from error
    return CheckpointLineage(
        current_packet_digest=current_packet_digest,
        namespace_digest=namespace,
        predecessor_packet_digest=predecessor,
        predecessor_max_step=boundary,
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
        profile: str,
        validation_interval_steps: int,
        top_k: int = 3,
        lineage: CheckpointLineage | None = None,
    ) -> None:
        if profile not in {"smoke", "production"}:
            raise ValueError(f"unsupported checkpoint retention profile: {profile}")
        if validation_interval_steps <= 0 or top_k != 3:
            raise ValueError(
                "OmniASR 7B retention requires a positive interval and top_k=3"
            )
        self.output_dir = output_dir.resolve()
        self.guard = guard
        self.world_size = int(world_size)
        self.profile = profile
        self.validation_interval_steps = int(validation_interval_steps)
        self.top_k = int(top_k)
        self.lineage = lineage or CheckpointLineage(
            current_packet_digest=guard.packet_digest,
            namespace_digest=guard.packet_digest,
        )
        if self.lineage.current_packet_digest != self.guard.packet_digest:
            raise ValueError("checkpoint lineage/current lease packet drift")

    @property
    def remote_namespace(self) -> Path:
        return (
            self.guard.store
            / "checkpoints"
            / self.guard.experiment_id
            / self.lineage.namespace_digest
        )

    @contextmanager
    def _remote_mutation_lock(self):
        """Serialize checkpoint mutation with lease takeover.

        ``acquire_lease()`` uses the same lock.  A replacement generation can
        therefore never overtake a live copy, prune, or replay transaction
        after the old generation has passed its fence check.
        """

        lock_path = (
            self.guard.store
            / "leases"
            / f".{self.guard.experiment_id}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self.guard.assert_current()
            try:
                yield
                self.guard.assert_current()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def commit(self, step: int) -> dict[str, object]:
        if not self.lineage.accepts(self.guard.packet_digest, step):
            raise RuntimeError("current packet cannot commit inside predecessor step range")
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
        with self._remote_mutation_lock():
            self._reconcile_interrupted_remote_transactions()
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
                self._apply_prevalidation_retention(step)
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
            self._apply_prevalidation_retention(step)
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

    def commit_runtime_evidence(self, step: int) -> dict[str, object]:
        """Commit the small post-validation state separately from model shards."""

        if step <= 0:
            raise ValueError("runtime evidence step must be positive")
        if not self.lineage.accepts(self.guard.packet_digest, step):
            raise RuntimeError("current packet cannot score inside predecessor step range")
        with self._remote_mutation_lock():
            self._reconcile_interrupted_remote_transactions()
            checkpoint = self.remote_namespace / f"step_{step}"
            verified_checkpoint = verify_remote_checkpoint(checkpoint)
            if int(verified_checkpoint["local"]["step"]) != step:
                raise RuntimeError("post-validation evidence/checkpoint step drift")
            root = self.remote_namespace / "epoch_evidence"
            root.mkdir(parents=True, exist_ok=True)
            target = root / f"step_{step}"
            if target.exists():
                existing = verify_epoch_evidence(target)
                if (
                    existing.get("experiment_id") != self.guard.experiment_id
                    or existing.get("packet_digest") != self.guard.packet_digest
                    or int(existing.get("step", -1)) != step
                ):
                    raise RuntimeError("existing epoch evidence identity drift")
                marker = existing
            else:
                temporary = root / f".step_{step}.{uuid.uuid4().hex}.tmp"
                temporary.mkdir()
                evidence = self._snapshot_runtime_evidence(temporary / "run_evidence")
                self.guard.assert_current()
                marker = {
                    "schema_version": 1,
                    "status": "EPOCH_EVIDENCE_COMMITTED",
                    "created_at_utc": _utc_now(),
                    "experiment_id": self.guard.experiment_id,
                    "packet_digest": self.guard.packet_digest,
                    "step": step,
                    "lease_generation": self.guard.generation,
                    "lease_token": self.guard.token,
                    "inventory_digest": evidence["inventory_digest"],
                    "evidence_files": evidence["files"],
                }
                _write_json_atomic(temporary / EPOCH_EVIDENCE_MARKER, marker)
                os.replace(temporary, target)
                verify_epoch_evidence(target)
            self._apply_remote_retention(step)
        return marker

    def _unverified_remote_step_entries(self) -> dict[int, Path]:
        """Resolve direct checkpoint children without reading partial payloads."""

        namespace = self.remote_namespace.resolve()
        steps: dict[int, Path] = {}
        if not namespace.exists():
            return steps
        for path in namespace.iterdir():
            match = re.fullmatch(r"step_([1-9][0-9]*)", path.name)
            if match is None:
                if path.name.startswith("step_"):
                    raise RuntimeError(f"invalid remote checkpoint name: {path}")
                continue
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"unsafe remote checkpoint entry: {path}")
            step = int(match.group(1))
            if step in steps:
                raise RuntimeError(f"duplicate remote checkpoint step: {step}")
            steps[step] = path
        return steps

    def _validated_retention_intent(
        self, path: Path, *, expected_status: str
    ) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsafe retention intent: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        top_three = value.get("top_three") if isinstance(value, dict) else None
        newest = int(value.get("newest_full_step", -1)) if isinstance(value, dict) else -1
        if (
            not isinstance(value, dict)
            or value.get("status") != expected_status
            or value.get("experiment_id") != self.guard.experiment_id
            or not self.lineage.accepts(value.get("packet_digest"), newest)
            or value.get("profile") != self.profile
            or int(value.get("top_k", -1)) != self.top_k
            or not isinstance(top_three, list)
        ):
            raise RuntimeError(f"retention intent identity drift: {path}")
        keep_values = value.get("keep_full_steps")
        if newest <= 0 or not isinstance(keep_values, list):
            raise RuntimeError(f"retention intent scalar drift: {path}")
        keep = [int(step) for step in keep_values]
        if (
            len(keep) != len(set(keep))
            or keep != sorted(keep)
            or newest not in keep
            or len(keep) > self.top_k + 1
        ):
            raise RuntimeError(f"retention keep-set drift: {path}")
        ranked_steps: list[int] = []
        for expected_rank, item in enumerate(top_three, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"malformed retention ranking: {path}")
            step = int(item.get("step", -1))
            rank = int(item.get("rank", -1))
            if step <= 0 or rank != expected_rank:
                raise RuntimeError(f"retention ranking drift: {path}")
            ranked_steps.append(step)
        if (
            len(ranked_steps) != len(set(ranked_steps))
            or len(ranked_steps) > self.top_k
            or set(keep) != set(ranked_steps) | {newest}
        ):
            raise RuntimeError(f"retention ranking/keep mismatch: {path}")
        return value

    def _validated_pending_intent(self, path: Path) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsafe pending retention intent: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        prior = value.get("prior_top_three_full_steps") if isinstance(value, dict) else None
        keep_values = value.get("keep_full_steps") if isinstance(value, dict) else None
        newest = int(value.get("newest_full_step", -1)) if isinstance(value, dict) else -1
        if (
            not isinstance(value, dict)
            or value.get("status") != "AWAITING_TARGET_SCORE"
            or value.get("experiment_id") != self.guard.experiment_id
            or not self.lineage.accepts(value.get("packet_digest"), newest)
            or value.get("profile") != self.profile
            or int(value.get("top_k", -1)) != self.top_k
            or not isinstance(prior, list)
            or not isinstance(keep_values, list)
        ):
            raise RuntimeError(f"pending retention intent identity drift: {path}")
        prior_steps = [int(step) for step in prior]
        keep = [int(step) for step in keep_values]
        if (
            newest <= 0
            or len(prior_steps) != len(set(prior_steps))
            or len(prior_steps) > self.top_k
            or prior_steps != sorted(prior_steps)
            or keep != sorted(set(prior_steps) | {newest})
            or len(keep) > self.top_k + 1
        ):
            raise RuntimeError(f"pending retention keep-set drift: {path}")
        return value

    def _remove_remote_step(self, step: int, path: Path, *, reason: str) -> None:
        namespace = self.remote_namespace.resolve()
        if (
            step <= 0
            or path.parent.resolve() != namespace
            or path.name != f"step_{step}"
            or not path.is_dir()
            or path.is_symlink()
        ):
            raise RuntimeError(f"unsafe {reason} target: {path}")
        self.guard.assert_current()
        shutil.rmtree(path)
        if path.exists():
            raise RuntimeError(f"{reason} target survived removal: {path}")

    def _write_recovery_event(self, actions: list[dict[str, object]]) -> None:
        if not actions:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = (
            self.remote_namespace
            / "recovery_events"
            / f"RECOVERY_{timestamp}_{uuid.uuid4().hex}.json"
        )
        _write_json_atomic(
            path,
            {
                "schema_version": 1,
                "status": "RECOVERED",
                "created_at_utc": _utc_now(),
                "experiment_id": self.guard.experiment_id,
                "packet_digest": self.guard.packet_digest,
                "profile": self.profile,
                "lease_generation": self.guard.generation,
                "lease_token": self.guard.token,
                "actions": actions,
            },
        )

    def _reconcile_intent_steps(
        self,
        *,
        intent: dict[str, object],
        intent_kind: str,
    ) -> tuple[list[int], list[int]]:
        """Finish an intent whose destructive prune may have been interrupted."""

        newest = int(intent["newest_full_step"])
        keep = {int(step) for step in intent["keep_full_steps"]}
        entries = self._unverified_remote_step_entries()
        newer = sorted(step for step in entries if step > newest)
        if newer:
            raise RuntimeError(
                f"{intent_kind} replay found a newer ungoverned checkpoint: {newer}"
            )
        missing = sorted(keep - set(entries))
        if missing:
            raise RuntimeError(
                f"{intent_kind} replay lost a required checkpoint: {missing}"
            )
        # Required states must remain byte-exact.  Obsolete states are allowed
        # to be partially deleted because the durable intent predates pruning.
        for step in sorted(keep):
            verified = verify_remote_checkpoint(entries[step])
            local = verified["local"]
            remote = verified["remote"]
            if (
                local.get("experiment_id") != self.guard.experiment_id
                or int(local.get("step", -1)) != step
                or int(local.get("world_size", -1)) != self.world_size
                or local.get("full_state") is not True
                or not self.lineage.accepts(local.get("packet_digest"), step)
                or remote.get("experiment_id") != self.guard.experiment_id
                or not self.lineage.accepts(remote.get("packet_digest"), step)
                or int(remote.get("step", -1)) != step
            ):
                raise RuntimeError(
                    f"{intent_kind} replay retained a foreign checkpoint: {step}"
                )
        removed = sorted(set(entries) - keep)
        for step in removed:
            self._remove_remote_step(
                step, entries[step], reason=f"interrupted {intent_kind} replay"
            )
        after = self._verified_remote_steps()
        if set(after) != keep:
            raise RuntimeError(
                f"{intent_kind} replay postcondition drift: "
                f"{sorted(after)} != {sorted(keep)}"
            )
        return sorted(keep), removed

    def _reconcile_interrupted_remote_transactions(self) -> list[dict[str, object]]:
        """Make copy and prune transactions idempotent after hard preemption."""

        self.guard.assert_current()
        namespace = self.remote_namespace.resolve()
        namespace.mkdir(parents=True, exist_ok=True)
        actions: list[dict[str, object]] = []

        temporary_pattern = re.compile(r"^\.step_([1-9][0-9]*)\.([0-9a-f]{32})\.tmp$")
        for path in sorted(namespace.iterdir(), key=lambda item: item.name):
            if not path.name.startswith(".step_"):
                continue
            match = temporary_pattern.fullmatch(path.name)
            if match is None or not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"unsafe interrupted checkpoint copy: {path}")
            self.guard.assert_current()
            shutil.rmtree(path)
            if path.exists():
                raise RuntimeError(f"interrupted checkpoint copy survived cleanup: {path}")
            actions.append(
                {
                    "kind": "discard_incomplete_remote_copy",
                    "step": int(match.group(1)),
                    "path": path.name,
                }
            )

        desired_path = namespace / RETENTION_DESIRED_MARKER
        applied_path = namespace / RETENTION_APPLIED_MARKER
        applied: dict[str, object] | None = None
        if applied_path.exists():
            applied = self._validated_retention_intent(
                applied_path, expected_status="RETENTION_APPLIED"
            )
        if desired_path.exists():
            desired = self._validated_retention_intent(
                desired_path, expected_status="RETENTION_DESIRED"
            )
            desired_sha256 = sha256_file(desired_path)
            applied_matches = (
                applied is not None
                and applied.get("desired_sha256") == desired_sha256
                and int(applied.get("newest_full_step", -1))
                == int(desired["newest_full_step"])
                and applied.get("keep_full_steps") == desired.get("keep_full_steps")
                and applied.get("top_three") == desired.get("top_three")
            )
            if not applied_matches:
                kept, removed = self._reconcile_intent_steps(
                    intent=desired, intent_kind="scored retention"
                )
                recovered_applied: dict[str, object] = {
                    **desired,
                    "status": "RETENTION_APPLIED",
                    "applied_at_utc": _utc_now(),
                    "desired_sha256": desired_sha256,
                    "observed_full_steps_after": kept,
                    "recovered_from_interruption": True,
                }
                _write_json_atomic(applied_path, recovered_applied)
                applied = recovered_applied
                actions.append(
                    {
                        "kind": "finish_scored_retention",
                        "newest_full_step": int(desired["newest_full_step"]),
                        "kept_full_steps": kept,
                        "removed_full_steps": removed,
                    }
                )

        pending_path = namespace / RETENTION_PENDING_SCORE_MARKER
        if pending_path.exists():
            pending = self._validated_pending_intent(pending_path)
            applied_newest = (
                int(applied.get("newest_full_step", -1)) if applied is not None else -1
            )
            pending_newest = int(pending["newest_full_step"])
            if pending_newest > applied_newest:
                kept, removed = self._reconcile_intent_steps(
                    intent=pending, intent_kind="pre-score retention"
                )
                if removed:
                    actions.append(
                        {
                            "kind": "finish_pre_score_retention",
                            "newest_full_step": pending_newest,
                            "kept_full_steps": kept,
                            "removed_full_steps": removed,
                        }
                    )
        self._write_recovery_event(actions)
        return actions

    def recover_for_launch(
        self,
    ) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
        """Fence, replay, and strictly inventory a namespace before restore.

        A hard preemption can leave an incomplete copy or an obsolete checkpoint
        partially removed after a durable retention intent was committed.  A new
        process must replay those transactions *before* strict checkpoint hashing;
        otherwise the recoverable partial directory makes resume unreachable.
        The takeover lease and checkpoint mutation lock jointly exclude the prior
        generation for the entire replay-and-enumerate critical section.
        """

        with self._remote_mutation_lock():
            actions = self._reconcile_interrupted_remote_transactions()
            records = self._verified_remote_records()
        return (
            {
                "schema_version": 1,
                "status": "PASS",
                "experiment_id": self.guard.experiment_id,
                "packet_digest": self.guard.packet_digest,
                "namespace_digest": self.lineage.namespace_digest,
                "lease_generation": self.guard.generation,
                "recovery_actions": actions,
                "committed_steps": sorted(records),
            },
            records,
        )

    def _verified_remote_records(self) -> dict[int, dict[str, object]]:
        """Hash every committed checkpoint and retain its verified record."""

        namespace = self.remote_namespace.resolve()
        temporary = sorted(namespace.glob(".step_*.tmp"))
        if temporary:
            raise RuntimeError(
                f"incomplete remote checkpoint copies block retention: {temporary}"
            )
        records: dict[int, dict[str, object]] = {}
        for path in namespace.glob("step_*"):
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"unsafe remote checkpoint entry: {path}")
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError as error:
                raise RuntimeError(f"invalid remote checkpoint name: {path}") from error
            verified = verify_remote_checkpoint(path)
            local = verified["local"]
            remote = verified["remote"]
            if (
                local.get("experiment_id") != self.guard.experiment_id
                or int(local.get("step", -1)) != step
                or int(local.get("world_size", -1)) != self.world_size
                or local.get("full_state") is not True
                or not self.lineage.accepts(local.get("packet_digest"), step)
                or remote.get("status") != "REMOTE_COMMITTED"
                or remote.get("experiment_id") != self.guard.experiment_id
                or not self.lineage.accepts(remote.get("packet_digest"), step)
                or remote.get("step") != step
            ):
                raise RuntimeError(f"remote retention identity drift: {path}")
            if step in records:
                raise RuntimeError(f"duplicate remote checkpoint step: {step}")
            records[step] = verified
        return records

    def _verified_remote_steps(self) -> dict[int, Path]:
        namespace = self.remote_namespace.resolve()
        return {
            step: namespace / f"step_{step}"
            for step in self._verified_remote_records()
        }

    def _target_score_top_three(self, newest_step: int) -> list[dict[str, object]]:
        evidence_dir = self.output_dir / "early_stopping"
        state_path = evidence_dir / "STATE.json"
        step_evidence = sorted(evidence_dir.glob("step_*.json"))
        if self.profile == "smoke":
            if state_path.exists() or step_evidence:
                raise RuntimeError("smoke retention contains target-score evidence")
            return []
        if not state_path.is_file() or state_path.is_symlink():
            raise RuntimeError("production retention lacks safe target-score state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = state.get("history") if isinstance(state, dict) else None
        if (
            not isinstance(state, dict)
            or state.get("promotion_authority") != "target_slot_weighted_raw_q"
            or state.get("status") not in {"CONTINUE", "EARLY_STOP_REQUESTED"}
            or not isinstance(history, list)
            or not history
        ):
            raise RuntimeError("production retention target-score state drift")
        scored: list[dict[str, object]] = []
        expected_paths: list[Path] = []
        for epoch, item in enumerate(history, start=1):
            if not isinstance(item, dict):
                raise RuntimeError("production retention history item is malformed")
            step = int(item.get("step", -1))
            score = float(item.get("promotion_score", float("nan")))
            expected_step = epoch * self.validation_interval_steps
            if (
                int(item.get("epoch", -1)) != epoch
                or step != expected_step
                or not math.isfinite(score)
            ):
                raise RuntimeError("production retention history sequence drift")
            evidence_path = evidence_dir / f"step_{step:08d}.json"
            if not evidence_path.is_file() or evidence_path.is_symlink():
                raise RuntimeError(f"unsafe retention score evidence: {evidence_path}")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (
                not isinstance(evidence, dict)
                or evidence.get("status") != "PASS"
                or evidence.get("promotion_authority")
                != "target_slot_weighted_raw_q"
                or int(evidence.get("step", -1)) != step
                or int(evidence.get("epoch", -1)) != epoch
                or float(evidence.get("promotion_score", float("nan"))) != score
            ):
                raise RuntimeError("retention score evidence/state drift")
            expected_paths.append(evidence_path)
            scored.append(
                {
                    "rank": 0,
                    "step": step,
                    "epoch": epoch,
                    "target_weighted_raw_q": score,
                    "evidence_sha256": sha256_file(evidence_path),
                }
            )
        if step_evidence != expected_paths or int(scored[-1]["step"]) != newest_step:
            raise RuntimeError("retention score prefix/newest checkpoint drift")
        ranked = sorted(
            scored,
            key=lambda item: (
                -float(item["target_weighted_raw_q"]),
                int(item["step"]),
            ),
        )[: self.top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked

    def _apply_prevalidation_retention(self, newest_step: int) -> dict[str, object]:
        """Bound transient disk to prior top three plus the new resume state."""

        namespace = self.remote_namespace.resolve()
        steps = self._verified_remote_steps()
        if newest_step not in steps or newest_step != max(steps):
            raise RuntimeError("prevalidation newest full checkpoint is absent or non-latest")
        prior_top_three: set[int] = set()
        applied_path = namespace / RETENTION_APPLIED_MARKER
        if applied_path.exists():
            if not applied_path.is_file() or applied_path.is_symlink():
                raise RuntimeError(f"unsafe prior retention marker: {applied_path}")
            applied = json.loads(applied_path.read_text(encoding="utf-8"))
            if (
                not isinstance(applied, dict)
                or applied.get("status") != "RETENTION_APPLIED"
                or applied.get("experiment_id") != self.guard.experiment_id
                or not self.lineage.accepts(
                    applied.get("packet_digest"),
                    int(applied.get("newest_full_step", -1)),
                )
                or applied.get("profile") != self.profile
                or int(applied.get("top_k", -1)) != self.top_k
                or not isinstance(applied.get("top_three"), list)
            ):
                raise RuntimeError("prior remote retention marker identity drift")
            prior_top_three = {
                int(item["step"])
                for item in applied["top_three"]
                if isinstance(item, dict) and "step" in item
            }
            if len(prior_top_three) != len(applied["top_three"]):
                raise RuntimeError("prior remote top-three marker is malformed")
        keep = prior_top_three | {newest_step}
        if not prior_top_three.issubset(steps):
            raise RuntimeError(
                "prior top-three checkpoint disappeared before new score: "
                f"{sorted(prior_top_three - set(steps))}"
            )
        pending: dict[str, object] = {
            "schema_version": 1,
            "status": "AWAITING_TARGET_SCORE",
            "created_at_utc": _utc_now(),
            "experiment_id": self.guard.experiment_id,
            "packet_digest": self.guard.packet_digest,
            "profile": self.profile,
            "top_k": self.top_k,
            "prior_top_three_full_steps": sorted(prior_top_three),
            "newest_full_step": newest_step,
            "keep_full_steps": sorted(keep),
            "observed_full_steps_before": sorted(steps),
        }
        _write_json_atomic(namespace / RETENTION_PENDING_SCORE_MARKER, pending)
        for step in sorted(set(steps) - keep):
            self._remove_remote_step(
                step, steps[step], reason="prevalidation prune"
            )
        after = self._verified_remote_steps()
        if set(after) != keep or len(after) > self.top_k + 1:
            raise RuntimeError("prevalidation remote retention postcondition drift")
        return pending

    def _apply_remote_retention(self, newest_step: int) -> dict[str, object]:
        """Keep exactly target-Q top three full states plus newest full state."""

        self.guard.assert_current()
        namespace = self.remote_namespace.resolve()
        steps = self._verified_remote_steps()
        if newest_step not in steps or newest_step != max(steps):
            raise RuntimeError("retention newest full checkpoint is absent or non-latest")
        ranked = self._target_score_top_three(newest_step)
        top_three = {int(item["step"]) for item in ranked}
        keep = top_three | {newest_step}
        if not keep.issubset(steps):
            raise RuntimeError(
                f"ranked checkpoint is absent before retention: {sorted(keep - set(steps))}"
            )
        desired: dict[str, object] = {
            "schema_version": 1,
            "status": "RETENTION_DESIRED",
            "created_at_utc": _utc_now(),
            "experiment_id": self.guard.experiment_id,
            "packet_digest": self.guard.packet_digest,
            "profile": self.profile,
            "promotion_authority": "target_slot_weighted_raw_q",
            "top_k": self.top_k,
            "newest_full_step": newest_step,
            "top_three": ranked,
            "keep_full_steps": sorted(keep),
            "observed_full_steps_before": sorted(steps),
            "automatic_local_checkpoint_downloads": False,
        }
        _write_json_atomic(namespace / RETENTION_DESIRED_MARKER, desired)
        for step in sorted(set(steps) - keep):
            self._remove_remote_step(step, steps[step], reason="remote prune")
        after = self._verified_remote_steps()
        if set(after) != keep:
            raise RuntimeError(
                f"remote retention postcondition drift: {sorted(after)} != {sorted(keep)}"
            )
        applied: dict[str, object] = {
            **desired,
            "status": "RETENTION_APPLIED",
            "applied_at_utc": _utc_now(),
            "desired_sha256": sha256_file(namespace / RETENTION_DESIRED_MARKER),
            "observed_full_steps_after": sorted(after),
        }
        _write_json_atomic(namespace / RETENTION_APPLIED_MARKER, applied)
        return applied


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


def verify_epoch_evidence(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"invalid epoch evidence directory: {path}")
    marker_path = path / EPOCH_EVIDENCE_MARKER
    evidence_path = path / "run_evidence/EVIDENCE.json"
    if not marker_path.is_file() or not evidence_path.is_file():
        raise RuntimeError(f"incomplete epoch evidence: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    inventory: list[dict[str, object]] = []
    for item in evidence.get("inventory", []):
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe epoch evidence path: {relative}")
        source = path / "run_evidence" / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"epoch evidence file is absent/unsafe: {source}")
        inventory.append(
            {
                "path": relative.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    if (
        not isinstance(marker, dict)
        or marker.get("status") != "EPOCH_EVIDENCE_COMMITTED"
        or evidence.get("status") != "PASS"
        or evidence.get("inventory") != inventory
        or evidence.get("inventory_digest") != _json_digest(inventory)
        or marker.get("inventory_digest") != evidence.get("inventory_digest")
        or marker.get("evidence_files") != len(inventory)
    ):
        raise RuntimeError(f"epoch evidence corruption: {path}")
    return marker


def verified_epoch_evidence_identity(
    remote_store: Path,
    *,
    experiment_id: str,
    packet_digest: str,
    step: int,
    restored_output_dir: Path | None = None,
    namespace_digest: str | None = None,
) -> dict[str, object]:
    """Verify and bind the separate post-validation commit for one step."""

    remote_store = remote_store.expanduser().resolve()
    namespace_digest = namespace_digest or packet_digest
    path = (
        remote_store
        / "checkpoints"
        / experiment_id
        / namespace_digest
        / "epoch_evidence"
        / f"step_{step}"
    )
    marker = verify_epoch_evidence(path)
    if (
        marker.get("experiment_id") != experiment_id
        or marker.get("packet_digest") != packet_digest
        or int(marker.get("step", -1)) != step
    ):
        raise RuntimeError("selected post-validation evidence identity drift")
    evidence = json.loads(
        (path / "run_evidence" / "EVIDENCE.json").read_text(encoding="utf-8")
    )
    inventory = evidence["inventory"]
    if restored_output_dir is not None:
        restored_output_dir = restored_output_dir.expanduser().resolve()
        for item in inventory:
            local = restored_output_dir / str(item["path"])
            if (
                not local.is_file()
                or local.is_symlink()
                or local.stat().st_size != int(item["bytes"])
                or sha256_file(local) != item["sha256"]
            ):
                raise RuntimeError(
                    f"restored post-validation evidence changed: {local}"
                )
    return {
        "epoch_evidence_marker_sha256": sha256_file(
            path / EPOCH_EVIDENCE_MARKER
        ),
        "epoch_evidence_inventory_digest": str(marker["inventory_digest"]),
        "epoch_evidence_files": int(marker["evidence_files"]),
        "epoch_evidence_inventory": inventory,
    }


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
    evidence_root = step_dir / "run_evidence"
    evidence_record = record["evidence"]
    post_validation = step_dir.parent / "epoch_evidence" / f"step_{step}"
    if post_validation.exists():
        post_marker = verify_epoch_evidence(post_validation)
        if (
            post_marker.get("experiment_id") != record["local"].get("experiment_id")
            or post_marker.get("packet_digest") != record["local"].get("packet_digest")
            or int(post_marker.get("step", -1)) != step
        ):
            raise RuntimeError("post-validation evidence/checkpoint identity drift")
        evidence_root = post_validation / "run_evidence"
        evidence_record = json.loads(
            (evidence_root / "EVIDENCE.json").read_text(encoding="utf-8")
        )
    for item in evidence_record["inventory"]:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe evidence restore path: {relative}")
        source = evidence_root / relative
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
    profile: str,
    validation_interval_steps: int,
) -> CheckpointCommitter:
    if not hasattr(trainer, "_on_checkpoint_saved") or not hasattr(trainer, "_gangs"):
        raise RuntimeError("pinned Trainer checkpoint integration point disappeared")
    guard = LeaseGuard.from_environment()
    guard.assert_current()
    lineage = checkpoint_lineage_from_environment(guard.packet_digest)
    gangs = trainer._gangs
    committer = CheckpointCommitter(
        output_dir=output_dir,
        guard=guard,
        world_size=int(gangs.root.size),
        profile=profile,
        validation_interval_steps=validation_interval_steps,
        lineage=lineage,
    )
    original = trainer._on_checkpoint_saved

    def on_checkpoint_saved(self: Any, step_nr: int, blocking: bool) -> None:
        if gangs.root.rank == 0:
            committer.commit(int(step_nr))
        gangs.root.barrier()
        original(step_nr, blocking)

    trainer._on_checkpoint_saved = MethodType(on_checkpoint_saved, trainer)
    original_early_stop_request = trainer._maybe_request_early_stop

    def maybe_request_early_stop(self: Any, score: float) -> bool:
        result = bool(original_early_stop_request(score))
        if gangs.root.rank == 0:
            committer.commit_runtime_evidence(int(self._step_nr))
        gangs.root.barrier()
        return result

    trainer._maybe_request_early_stop = MethodType(maybe_request_early_stop, trainer)
    optimizer = trainer._optimizer

    def assert_fence(optimizer_object: Any, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        del optimizer_object, args, kwargs
        guard.assert_current()

    handle = optimizer.register_step_pre_hook(assert_fence)
    trainer._waxal3_lease_guard = guard
    trainer._waxal3_lease_hook = handle
    trainer._waxal3_checkpoint_committer = committer
    return committer
