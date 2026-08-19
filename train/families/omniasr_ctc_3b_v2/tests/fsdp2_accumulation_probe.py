#!/usr/bin/env python3
"""Two-rank CPU/Gloo numerical gate for OmniASR CTC-3B-v2 FSDP2 accumulation."""

from __future__ import annotations

from contextlib import nullcontext
import json

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from fairseq2.nn.fsdp.fsdp2 import fsdp2_no_sync
from fairseq2.nn.utils.grad import clip_grad_norm
from fsdp_compat import install_fsdp2_gradient_sync_compat
from fsdp_export import consolidate_fsdp_state_dict


def _model() -> nn.Module:
    torch.manual_seed(1729)
    return nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 3))


def _batches(rank: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(8800 + rank)
    return [
        (
            torch.randn(2, 5, generator=generator),
            torch.randn(2, 3, generator=generator),
        )
        for _ in range(4)
    ]


def main() -> int:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"probe requires exactly two ranks, got {world_size}")
    torch.set_num_threads(1)
    compatibility = install_fsdp2_gradient_sync_compat()
    batches = _batches(rank)
    mesh = init_device_mesh("cpu", (world_size,))

    def train_once(*, accumulated_no_sync: bool) -> tuple[dict[str, torch.Tensor], float]:
        sharded = _model()
        fully_shard(sharded, mesh=mesh, reshard_after_forward=True)
        optimizer = torch.optim.SGD(sharded.parameters(), lr=0.03)
        for index, (inputs, targets) in enumerate(batches):
            use_no_sync = accumulated_no_sync and index < len(batches) - 1
            context = fsdp2_no_sync(sharded) if use_no_sync else nullcontext()
            with context:
                loss = (
                    torch.nn.functional.mse_loss(sharded(inputs), targets)
                    / len(batches)
                )
                loss.backward()
        gradient_norm = clip_grad_norm(sharded, 0.7)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite FSDP2 gradient norm: {gradient_norm}")
        optimizer.step()
        consolidated = consolidate_fsdp_state_dict(sharded, rank=rank)
        payload: list[dict[str, torch.Tensor] | None] = [consolidated]
        dist.broadcast_object_list(payload, src=0)
        state = payload[0]
        if state is None:
            raise RuntimeError("rank-zero consolidated state was not broadcast")
        return state, float(gradient_norm)

    reference_state, reference_norm = train_once(accumulated_no_sync=False)
    dist.barrier()
    observed_state, observed_norm = train_once(accumulated_no_sync=True)
    maximum_delta = 0.0
    for key, value in observed_state.items():
        maximum_delta = max(
            maximum_delta,
            float((value - reference_state[key]).abs().max()),
        )
    norm_delta = abs(observed_norm - reference_norm)
    if rank == 0:
        reloaded = _model()
        incompatibility = reloaded.load_state_dict(observed_state, strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(f"strict consolidated reload drift: {incompatibility}")
        reload_delta = max(
            float((value - observed_state[key]).abs().max())
            for key, value in reloaded.state_dict().items()
        )
    else:
        reload_delta = 0.0
    reload_tensor = torch.tensor(reload_delta)
    dist.broadcast(reload_tensor, src=0)
    reload_delta = float(reload_tensor)
    if maximum_delta > 2e-6 or norm_delta > 2e-6 or reload_delta > 0.0:
        raise RuntimeError(
            "FSDP2/reference drift: "
            f"parameters={maximum_delta} norm={norm_delta} "
            f"reload={reload_delta} observed_norm={observed_norm} "
            f"reference_norm={reference_norm}"
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "world_size": world_size,
                    "microbatches": len(batches),
                    "optimizer_steps": 1,
                    "clip_max_norm": 0.7,
                    "gradient_norm": observed_norm,
                    "gradient_norm_delta": norm_delta,
                    "maximum_parameter_delta": maximum_delta,
                    "strict_consolidated_reload_maximum_delta": reload_delta,
                    "compatibility": compatibility,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
