from __future__ import annotations

from pathlib import Path

import torch
import pytest

import inference_export
from consistency import ConsistencyConfig, TRAINING_ONLY_MASK_KEY


def test_clean_inference_export_strips_only_masker_and_reloads(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "training.pt"
    output = tmp_path / "export"
    torch.save(
        {
            "model": {
                "encoder.weight": torch.tensor([[1.0, 2.0]]),
                TRAINING_ONLY_MASK_KEY: torch.zeros(2),
                "final_proj.bias": torch.tensor([0.5]),
            },
            "fs2": True,
        },
        source,
    )

    def fake_forward(state, **_kwargs):
        # The train-only embedding is inert in eval; all retained tensors bind
        # the parity fixture to the serialized inference state.
        value = sum(
            tensor.float().sum()
            for name, tensor in state.items()
            if name != TRAINING_ONLY_MASK_KEY
        )
        return value.reshape(1, 1, 1).repeat(1, 3, 10_288), [3]

    monkeypatch.setattr(inference_export, "_forward", fake_forward)
    record = inference_export.export_inference_checkpoint(
        source=source,
        output_dir=output,
        card_dir=tmp_path,
        model_card="unused",
        consistency=ConsistencyConfig(),
    )
    exported = inference_export.load_state(output / "model.pt")
    assert record["status"] == "PASS"
    assert record["forward_parity_exact"] is True
    assert TRAINING_ONLY_MASK_KEY not in exported
    assert set(exported) == {"encoder.weight", "final_proj.bias"}


def test_inference_export_preserves_failure_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "training.pt"
    torch.save(
        {
            "model": {
                "encoder.weight": torch.ones(1),
                TRAINING_ONLY_MASK_KEY: torch.zeros(1),
            }
        },
        source,
    )

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("intentional parity failure")

    monkeypatch.setattr(inference_export, "_forward", fail_forward)
    with pytest.raises(RuntimeError, match="intentional"):
        inference_export.export_inference_checkpoint(
            source=source,
            output_dir=tmp_path / "export",
            card_dir=tmp_path,
            model_card="unused",
            consistency=ConsistencyConfig(),
        )
    failures = list(tmp_path.glob("export.failed.*/FAILURE.json"))
    assert len(failures) == 1
    assert '"status": "FAIL"' in failures[0].read_text(encoding="utf-8")
