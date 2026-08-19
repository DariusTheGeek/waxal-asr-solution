"""Distributed full-state export for a restored OmniASR 3B FSDP2 model."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MethodType
from typing import Any
import uuid

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from fairseq2.nn import BatchLayout

from checkpoint_contract import export_source_identity
from early_stopping import validate_export_validation_state
from runtime_config import runtime_geometry_from_environment


EXPECTED_PARAMETERS = 3_081_398_960


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parity_input(device: torch.device) -> tuple[Tensor, BatchLayout]:
    samples = torch.linspace(
        -0.1, 0.1, 16_000, device=device, dtype=torch.float32
    ).unsqueeze(0)
    return samples, BatchLayout.of(samples, [16_000])


def parity_forward(
    module: torch.nn.Module,
    inputs: Tensor,
    layout: BatchLayout,
    *,
    device: torch.device,
) -> tuple[Tensor, BatchLayout]:
    """Match the frozen BF16 training/inference compute contract explicitly."""

    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"unsupported parity device type: {device.type}")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        return module(inputs, layout)


def consolidate_fsdp_state_dict(
    module: torch.nn.Module, *, rank: int
) -> dict[str, Tensor] | None:
    """Collect every DTensor on all ranks and retain the full state on rank zero."""

    consolidated: dict[str, Tensor] | None = {} if rank == 0 else None
    state = module.state_dict()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, DTensor):
            full = value.full_tensor()
        elif isinstance(value, Tensor):
            full = value
        else:
            raise RuntimeError(f"non-tensor model state is forbidden: {key}: {type(value)}")
        if rank == 0:
            assert consolidated is not None
            consolidated[key] = full.detach().cpu().contiguous()
        del full
    return consolidated


def export_restored_fsdp_model(trainer: Any, destination: Path) -> dict[str, object]:
    """Export one restored trainer model and a sharded-forward parity anchor."""

    gangs = trainer._gangs
    rank = int(gangs.root.rank)
    world_size = int(gangs.root.size)
    if world_size != 8:
        raise RuntimeError(f"OmniASR 3B export requires exactly eight ranks, got {world_size}")
    step = int(trainer._step_nr)
    if step <= 0:
        raise RuntimeError("OmniASR 3B export requires a restored positive checkpoint step")
    destination = destination.resolve()
    if rank == 0 and destination.exists():
        raise FileExistsError(f"OmniASR 3B export destination already exists: {destination}")
    gangs.root.barrier()

    required_environment = (
        "WAXAL3_TRAINER_OUTPUT_DIR",
        "WAXAL3_REMOTE_STORE",
        "WAXAL3_EXPERIMENT_ID",
        "WAXAL3_PACKET_DIGEST",
        "WAXAL3_PROFILE",
        "WAXAL3_REPO_ROOT",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"OmniASR 3B export identity environment is incomplete: {missing}")
    geometry = runtime_geometry_from_environment(
        Path(os.environ["WAXAL3_REPO_ROOT"])
    )
    identity_payload: list[object | None] = [None, None]
    if rank == 0:
        try:
            identity_payload[1] = export_source_identity(
                Path(os.environ["WAXAL3_TRAINER_OUTPUT_DIR"]),
                remote_store=Path(os.environ["WAXAL3_REMOTE_STORE"]),
                experiment_id=os.environ["WAXAL3_EXPERIMENT_ID"],
                packet_digest=os.environ["WAXAL3_PACKET_DIGEST"],
                step=step,
                world_size=world_size,
                profile=os.environ["WAXAL3_PROFILE"],
            )
            validation_evidence = validate_export_validation_state(
                trainer_output_dir=Path(
                    os.environ["WAXAL3_TRAINER_OUTPUT_DIR"]
                ),
                rank_map_path=geometry.manifest_dir / "dev.rank_map.world8.csv",
                checkpoint_step=step,
                world_size=world_size,
                profile=os.environ["WAXAL3_PROFILE"],
                updates_per_epoch=geometry.updates_per_epoch,
            )
            identity_payload[1].update(
                {
                    "validation_evidence": validation_evidence,
                    "validation_evidence_digest": canonical_json_sha256(
                        validation_evidence
                    ),
                }
            )
        except BaseException as error:
            identity_payload[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(identity_payload, src=0)
    if identity_payload[0] is not None:
        raise RuntimeError(f"OmniASR 3B export source identity failed: {identity_payload[0]}")
    if not isinstance(identity_payload[1], dict):
        raise RuntimeError("OmniASR 3B export source identity was not broadcast")
    source_identity = identity_payload[1]

    module = trainer._unit.model.module
    module.eval()
    inputs, layout = parity_input(gangs.root.device)
    logits, output_layout = parity_forward(
        module, inputs, layout, device=gangs.root.device
    )
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("sharded parity forward produced non-finite logits")
    local_logits = logits.detach().float()
    maximum = local_logits.clone()
    minimum = local_logits.clone()
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    rank_delta = float((maximum - minimum).abs().max())
    if rank_delta > 0.0:
        raise RuntimeError(f"sharded parity logits differ across ranks: {rank_delta}")
    del maximum, minimum

    consolidated = consolidate_fsdp_state_dict(module, rank=rank)
    record: dict[str, object] = {
        "schema_version": 2,
        "status": "PASS",
        "step": step,
        "world_size": world_size,
        "rank_forward_maximum_delta": rank_delta,
        **source_identity,
    }
    if rank == 0:
        assert consolidated is not None
        parameter_values = sum(value.numel() for value in consolidated.values())
        if parameter_values != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"consolidated state size drift: {parameter_values} != {EXPECTED_PARAMETERS}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temporary.mkdir()
        try:
            checkpoint = temporary / "model.pt"
            torch.save({"model": consolidated, "fs2": True}, checkpoint)
            probe = temporary / "sharded_forward.pt"
            torch.save(
                {
                    "logits": local_logits.cpu(),
                    "output_seq_lens": [int(value) for value in output_layout.seq_lens],
                    "input_samples": 16_000,
                },
                probe,
            )
            record.update(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "format": "fairseq2_native_full_state_fs2",
                    "model_file": checkpoint.name,
                    "model_bytes": checkpoint.stat().st_size,
                    "model_sha256": sha256_file(checkpoint),
                    "probe_file": probe.name,
                    "probe_bytes": probe.stat().st_size,
                    "probe_sha256": sha256_file(probe),
                    "state_tensors": len(consolidated),
                    "state_parameter_values": parameter_values,
                    "strict_unsharded_reload": "PENDING_FRESH_PROCESS",
                }
            )
            (temporary / "EXPORT.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    gangs.root.barrier()
    return record


def attach_export_after_restore(trainer: Any) -> None:
    raw = os.environ.get("WAXAL3_EXPORT_FULL_STATE_DIR")
    if not raw:
        return
    destination = Path(raw).expanduser().resolve()
    original_restore = trainer._maybe_restore_state

    def restore_and_export(self: Any):
        state = original_restore()
        export_restored_fsdp_model(self, destination)
        return type(state).STOPPED

    trainer._maybe_restore_state = MethodType(restore_and_export, trainer)
    trainer._waxal3_distributed_export_contract = True
