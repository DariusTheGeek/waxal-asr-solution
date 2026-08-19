from pathlib import Path

import pytest

from fsdp_artifacts import inspect_fsdp_checkpoint


def _checkpoint(root: Path, world_size: int = 2) -> Path:
    step = root / "step_2"
    for rank in range(world_size):
        for rel in (f"trainer/rank_{rank:02d}.pt", f"data_reader/dp_{rank:02d}.pt", f"model/pp_00/tp_00/sdp_{rank:02d}.pt", f"optimizer/pp_00/tp_00/sdp_{rank:02d}.pt"):
            path = step / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    return step


def test_fsdp_shard_inventory_requires_every_rank(tmp_path: Path) -> None:
    step = _checkpoint(tmp_path)
    record = inspect_fsdp_checkpoint(step, world_size=2)
    assert record["format"] == "fairseq2_fsdp_sharded_v1"
    assert len(record["model_shards"]) == 2
    (step / "model/pp_00/tp_00/sdp_01.pt").unlink()
    with pytest.raises(RuntimeError, match="topology drift"):
        inspect_fsdp_checkpoint(step, world_size=2)
