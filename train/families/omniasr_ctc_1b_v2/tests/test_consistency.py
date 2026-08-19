from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.nn import BatchLayout

from consistency import (
    ConsistencyConfig,
    DualViewCtcCriterion,
    TRAINING_ONLY_MASK_KEY,
    build_training_masker,
    cr_weight,
    detached_kl_direction,
    make_aligned_views,
    strip_training_only_masker,
    symmetric_consistency_loss,
    validate_consistency_config,
)


def test_cr_weight_is_linear_and_saturates() -> None:
    config = ConsistencyConfig(cr_max_weight=0.2, cr_warmup_steps=501)
    assert cr_weight(0, config) == 0.0
    assert cr_weight(1, config) == pytest.approx(0.2 / 501)
    assert cr_weight(501, config) == pytest.approx(0.2)
    assert cr_weight(9_999, config) == pytest.approx(0.2)


def test_frozen_recipe_constraints_fail_closed() -> None:
    with pytest.raises(ValueError, match="dual_view"):
        validate_consistency_config(ConsistencyConfig(dual_view=False))
    with pytest.raises(ValueError, match="spatial"):
        validate_consistency_config(ConsistencyConfig(spatial_mask_prob=0.1))
    with pytest.raises(ValueError, match="blank"):
        validate_consistency_config(ConsistencyConfig(blank_idx=1))


def test_masker_is_neutral_and_requested_fraction_is_realized() -> None:
    torch.manual_seed(42)
    config = ConsistencyConfig()
    masker = build_training_masker(16, config, device="cpu", dtype=torch.float32)
    assert torch.equal(masker.temporal_mask_embed, torch.zeros(16))
    features = torch.ones((32, 400, 16))
    layout = BatchLayout.of(features, [400] * 32)
    _, mask = masker(features, layout)
    fraction = float(mask.float().mean())
    assert 0.08 <= fraction <= 0.14


def test_two_masker_calls_draw_independent_temporal_views() -> None:
    torch.manual_seed(42)
    masker = build_training_masker(8, ConsistencyConfig())
    features = torch.ones((4, 200, 8))
    layout = BatchLayout.of(features, [200] * 4)
    _, first = masker(features, layout)
    _, second = masker(features, layout)
    assert not torch.equal(first, second)


def test_one_way_kl_detaches_the_target_distribution() -> None:
    input_logits = torch.randn((2, 4, 7), requires_grad=True)
    target_logits = torch.randn((2, 4, 7), requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    loss = detached_kl_direction(input_logits, target_logits, valid)
    loss.backward()
    assert input_logits.grad is not None
    assert float(input_logits.grad.abs().sum()) > 0.0
    assert target_logits.grad is None


def test_symmetric_kl_is_swap_invariant_and_padding_is_zero() -> None:
    logits_a = torch.randn((1, 5, 9), requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    layout = BatchLayout.of(torch.zeros((1, 5)), [3])
    logits_b.data[:, 3:] = 100.0
    forward = symmetric_consistency_loss(
        logits_a, layout, logits_b, layout, blank_idx=0
    )
    reverse = symmetric_consistency_loss(
        logits_b, layout, logits_a, layout, blank_idx=0
    )
    assert forward.valid_frames == 3
    assert float(forward.loss.detach()) == pytest.approx(0.0, abs=1.0e-6)
    assert torch.equal(forward.loss, reverse.loss)


def test_output_layout_mismatch_fails_closed() -> None:
    logits = torch.randn((1, 5, 9))
    first = BatchLayout.of(torch.zeros((1, 5)), [5])
    second = BatchLayout.of(torch.zeros((1, 5)), [4])
    with pytest.raises(RuntimeError, match="layout mismatch"):
        symmetric_consistency_loss(logits, first, logits, second, blank_idx=0)


def test_view_generation_preserves_alignment_and_padding() -> None:
    torch.manual_seed(17)
    source = torch.linspace(-1.0, 1.0, 200).reshape(2, 100)
    layout = BatchLayout.of(source, [100, 73])
    config = ConsistencyConfig(
        speed_factors=(1.0,),
        view_b_noise_prob=1.0,
        view_b_snr_db_min=20.0,
        view_b_snr_db_max=20.0,
    )
    view_a, view_b, output_layout, stats = make_aligned_views(source, layout, config)
    assert view_a.shape == view_b.shape == source.shape
    assert list(output_layout.seq_lens) == [100, 73]
    assert torch.equal(view_a[1, 73:], torch.zeros(27))
    assert torch.equal(view_b[1, 73:], torch.zeros(27))
    assert not torch.equal(view_a[:, :73], view_b[:, :73])
    assert stats.noise_examples == 2
    assert stats.noise_snr_db_mean == pytest.approx(20.0)
    signal = view_a[0, :100].float()
    noise = view_b[0, :100].float() - signal
    observed_snr = 20.0 * torch.log10(
        signal.square().mean().sqrt() / noise.square().mean().sqrt()
    )
    assert float(observed_snr) == pytest.approx(20.0, abs=1.0e-4)


def test_training_only_state_filter_is_exact() -> None:
    state = OrderedDict(
        {
            "encoder.weight": torch.ones((2, 2)),
            TRAINING_ONLY_MASK_KEY: torch.zeros(2),
            "final_proj.bias": torch.zeros(3),
        }
    )
    output = strip_training_only_masker(state, require_masker=True)
    assert list(output) == ["encoder.weight", "final_proj.bias"]
    with pytest.raises(RuntimeError, match="missing training-only"):
        strip_training_only_masker(
            {"encoder.weight": torch.ones(1)}, require_masker=True
        )


class _TinyCtc(torch.nn.Module):
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
        base = seqs.float()
        if base.ndim == 3:
            base = base.squeeze(-1)
        logits = torch.stack(
            [base * self.scale + float(index) / 10.0 for index in range(5)],
            dim=-1,
        )
        loss = logits.square().sum()
        assert return_logits
        return loss, logits, seqs_layout


class _RecipeModelStub:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.base_module = module


def test_dual_view_objective_runs_backward_and_records_metrics() -> None:
    torch.manual_seed(73)
    module = _TinyCtc()
    model = _RecipeModelStub(module)
    config = ConsistencyConfig(
        cr_max_weight=0.2,
        cr_warmup_steps=2,
        speed_factors=(1.0,),
        view_b_noise_prob=1.0,
        view_b_snr_db_min=20.0,
        view_b_snr_db_max=20.0,
    )
    criterion = DualViewCtcCriterion(model, config)  # type: ignore[arg-type]
    criterion.set_step_nr(2)
    metrics = MetricBag(torch.device("cpu"))
    criterion.prepare_metric_bag(metrics)
    batch = Seq2SeqBatch(
        source_seqs=torch.randn((2, 80)),
        source_seq_lens=[80, 67],
        target_seqs=torch.tensor([[1, 2, 3], [1, 3, 0]]),
        target_seq_lens=[3, 2],
    )
    objective, batch_size = criterion(batch, metrics)
    objective.backward()
    assert batch_size == 2
    assert module.scale.grad is not None
    assert torch.isfinite(module.scale.grad)
    assert {
        "ctc_loss",
        "objective_loss",
        "cr_loss",
        "cr_weight",
        "blank_argmax_rate",
        "view_argmax_disagreement",
    } <= set(metrics.metrics)


def test_two_forwards_share_one_ddp_backward(tmp_path: Path) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already active")
    init_file = tmp_path / "gloo-init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(42)
        base = _TinyCtc()
        ddp = torch.nn.parallel.DistributedDataParallel(base)
        model = _RecipeModelStub(ddp)
        model.base_module = base
        config = ConsistencyConfig(
            speed_factors=(1.0,),
            view_b_noise_prob=1.0,
            view_b_snr_db_min=20.0,
            view_b_snr_db_max=20.0,
        )
        criterion = DualViewCtcCriterion(model, config)  # type: ignore[arg-type]
        criterion.set_step_nr(501)
        metrics = MetricBag(torch.device("cpu"))
        criterion.prepare_metric_bag(metrics)
        batch = Seq2SeqBatch(
            source_seqs=torch.randn((1, 80)),
            source_seq_lens=[80],
            target_seqs=torch.tensor([[1, 2]]),
            target_seq_lens=[2],
        )
        objective, _ = criterion(batch, metrics)
        objective.backward()
        assert base.scale.grad is not None
        assert torch.isfinite(base.scale.grad)
    finally:
        dist.destroy_process_group()


def test_two_forwards_share_one_world2_ddp_backward() -> None:
    probe = Path(__file__).with_name("ddp_two_forward_probe.py")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(probe.parents[1]),
        }
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(probe),
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout
    assert '"status": "PASS"' in process.stdout
