"""Fail-closed inspection contract for fairseq2 FSDP checkpoint shards.

FSDP state is partitioned by `sdp_XX.pt`; it must be restored/consolidated by a
matching distributed fairseq2 runtime before a parameter average or inference
export. Treating an individual shard as a standalone model is forbidden.
"""

from __future__ import annotations

from pathlib import Path

from checkpoint_contract import checkpoint_inventory, _validate_full_state_layout


def inspect_fsdp_checkpoint(step_dir: Path, *, world_size: int) -> dict[str, object]:
    step_dir = step_dir.resolve()
    inventory = checkpoint_inventory(step_dir)
    _validate_full_state_layout(inventory, world_size)
    model = sorted((step_dir / "model").glob("pp_*/tp_*/sdp_*.pt"))
    optimizer = sorted((step_dir / "optimizer").glob("pp_*/tp_*/sdp_*.pt"))
    trainer = sorted((step_dir / "trainer").glob("rank_*.pt"))
    readers = sorted((step_dir / "data_reader").glob("dp_*.pt"))
    if len(model) != world_size or len(optimizer) != world_size:
        raise RuntimeError("FSDP model/optimizer shard count does not match world size")
    if len(trainer) != world_size or len(readers) != world_size:
        raise RuntimeError("FSDP trainer/data-reader shard count does not match world size")
    return {
        "format": "fairseq2_fsdp_sharded_v1",
        "world_size": world_size,
        "exact_topology": True,
        "model_shards": [path.relative_to(step_dir).as_posix() for path in model],
        "optimizer_shards": [path.relative_to(step_dir).as_posix() for path in optimizer],
        "export_policy": "distributed_consolidation_required_before_average_or_single_process_inference",
    }
