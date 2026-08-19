import torch

from cpt.global_cpt import ddp_mask_normalized_loss, scheduler_multiplier


def test_ddp_average_is_exact_global_mask_mean() -> None:
    local_sums = [torch.tensor(15.0), torch.tensor(20.0), torch.tensor(9.0), torch.tensor(6.0)]
    global_masks = torch.tensor(25.0)
    rank_losses = [
        ddp_mask_normalized_loss(value, global_masks, world_size=4)
        for value in local_sums
    ]
    assert float(torch.stack(rank_losses).mean()) == 2.0


def test_eight_rank_ddp_average_is_exact_global_mask_mean() -> None:
    local_sums = [torch.tensor(float(value)) for value in range(1, 9)]
    global_masks = torch.tensor(18.0)
    rank_losses = [
        ddp_mask_normalized_loss(value, global_masks, world_size=8)
        for value in local_sums
    ]
    assert float(torch.stack(rank_losses).mean()) == 2.0


def test_scheduler_reaches_zero_only_after_declared_horizon() -> None:
    assert scheduler_multiplier(0, warmup_steps=5, max_steps=50) == 0.2
    assert scheduler_multiplier(4, warmup_steps=5, max_steps=50) == 1.0
    assert scheduler_multiplier(49, warmup_steps=5, max_steps=50) > 0.0
    assert scheduler_multiplier(50, warmup_steps=5, max_steps=50) == 0.0
