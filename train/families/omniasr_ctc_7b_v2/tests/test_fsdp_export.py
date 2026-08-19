from __future__ import annotations

import pytest
import torch

from fsdp_export import EXPECTED_PARAMETERS, parity_forward, parity_input


def test_exporter_is_pinned_to_exact_7b_v2_parameter_count() -> None:
    assert EXPECTED_PARAMETERS == 6_505_761_456


class _Bf16AudioFrontEnd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, kernel_size=3).to(torch.bfloat16)

    def forward(self, samples, layout):
        return self.conv(samples.unsqueeze(1)).transpose(1, 2), layout


def test_parity_forward_explicitly_autocasts_fp32_audio_for_bf16_model() -> None:
    device = torch.device("cpu")
    model = _Bf16AudioFrontEnd().eval()
    samples, layout = parity_input(device)
    assert samples.dtype is torch.float32
    with pytest.raises(RuntimeError, match="Input type"):
        model(samples, layout)
    logits, observed_layout = parity_forward(
        model, samples, layout, device=device
    )
    assert observed_layout is layout
    assert logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(logits).all())
