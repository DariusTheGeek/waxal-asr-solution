#!/usr/bin/env python3
"""World-2 CPU proof that two DDP forwards may share one backward."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag

from consistency import ConsistencyConfig, DualViewCtcCriterion


class TinyCtc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        seqs,
        seqs_layout,
        targets,
        targets_layout,
        *,
        return_logits,
    ):
        del targets, targets_layout
        logits = torch.stack(
            [seqs.float() * self.scale + index / 10.0 for index in range(5)],
            dim=-1,
        )
        assert return_logits
        return logits.square().sum(), logits, seqs_layout


class RecipeModelStub:
    def __init__(self, module: torch.nn.Module, base_module: torch.nn.Module) -> None:
        self.module = module
        self.base_module = base_module


def main() -> int:
    dist.init_process_group("gloo")
    try:
        rank = int(os.environ["RANK"])
        torch.manual_seed(42 + rank)
        base = TinyCtc()
        ddp = torch.nn.parallel.DistributedDataParallel(base)
        criterion = DualViewCtcCriterion(
            RecipeModelStub(ddp, base),  # type: ignore[arg-type]
            ConsistencyConfig(
                speed_factors=(1.0,),
                view_b_noise_prob=1.0,
                view_b_snr_db_min=20.0,
                view_b_snr_db_max=20.0,
            ),
        )
        criterion.set_step_nr(501)
        metrics = MetricBag(torch.device("cpu"))
        criterion.prepare_metric_bag(metrics)
        batch = Seq2SeqBatch(
            source_seqs=torch.randn((1, 80)) + rank,
            source_seq_lens=[80],
            target_seqs=torch.tensor([[1, 2]]),
            target_seq_lens=[2],
        )
        objective, _ = criterion(batch, metrics)
        objective.backward()
        gradient = base.scale.grad
        if (
            gradient is None
            or not torch.isfinite(gradient)
            or float(gradient.abs()) <= 0
        ):
            raise RuntimeError("missing/non-finite DDP gradient")
        gathered = [torch.zeros_like(gradient) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, gradient)
        if any(not torch.equal(gathered[0], value) for value in gathered[1:]):
            raise RuntimeError("DDP gradients differ after all-reduce")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "world_size": dist.get_world_size(),
                        "forwards_per_backward": 2,
                        "gradient": float(gradient),
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
