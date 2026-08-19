from __future__ import annotations

import pytest
import torch

from fsdp_export import EXPECTED_PARAMETERS, parity_forward, parity_input


def test_exporter_is_pinned_to_exact_llm_3b_v2_parameter_count() -> None:
    assert EXPECTED_PARAMETERS == 4_380_578_432


def test_parity_input_accepts_frozen_shona_language_code() -> None:
    batch = parity_input(torch.device("cpu"), language_code="sna_Latn")
    assert batch.example == {"lang": ["sna_Latn"]}


class _Bf16AudioFrontEnd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, kernel_size=3).to(torch.bfloat16)

    def forward(self, batch, *, return_logits=False):
        logits = self.conv(batch.source_seqs.unsqueeze(1)).transpose(1, 2)
        from fairseq2.nn import BatchLayout

        layout = BatchLayout.of(logits, [logits.shape[1]])
        loss = logits.float().square().mean()
        if return_logits:
            return loss, logits, layout, [], [], []
        return loss


def test_parity_forward_explicitly_autocasts_fp32_audio_for_bf16_model() -> None:
    device = torch.device("cpu")
    model = _Bf16AudioFrontEnd().eval()
    batch = parity_input(device)
    assert batch.source_seqs.dtype is torch.float32
    with pytest.raises(RuntimeError, match="Input type"):
        model(batch, return_logits=True)
    loss, logits, observed_layout = parity_forward(
        model, batch, device=device
    )
    assert bool(torch.isfinite(loss))
    assert observed_layout.seq_lens == [logits.shape[1]]
    assert logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(logits).all())
