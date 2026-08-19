from __future__ import annotations

from pathlib import Path

import torch

from checkpoint_average import (
    CheckpointAverageError,
    average_states,
    validate_state_contract,
)


def test_llm_average_uses_float64_and_preserves_nonfloating_state() -> None:
    states = [
        {
            "decoder.weight": torch.tensor([value, value + 2], dtype=torch.bfloat16),
            "version": torch.tensor([7], dtype=torch.int64),
        }
        for value in (1.0, 3.0, 5.0)
    ]
    averaged, contract = average_states(states)
    assert averaged["decoder.weight"].dtype == torch.bfloat16
    assert torch.equal(
        averaged["decoder.weight"], torch.tensor([3.0, 5.0], dtype=torch.bfloat16)
    )
    assert torch.equal(averaged["version"], torch.tensor([7], dtype=torch.int64))
    assert contract["floating_tensor_count"] == 1
    assert contract["non_floating_tensor_count"] == 1


def test_llm_average_rejects_schema_drift() -> None:
    states = [
        {"weight": torch.zeros(shape, dtype=torch.float32)}
        for shape in ((2,), (3,), (2,))
    ]
    try:
        validate_state_contract(states)
    except CheckpointAverageError as error:
        assert "schema drift" in str(error)
    else:
        raise AssertionError("schema drift was accepted")


def test_llm_average_forward_contract_is_language_conditioned() -> None:
    source = (Path(__file__).resolve().parents[1] / "checkpoint_average.py").read_text(
        encoding="utf-8"
    )
    assert "Seq2SeqBatch" in source
    assert 'example={"lang": [language_code]}' in source
    assert 'language_code: str = "lin_Latn"' in source
    assert "4_380_578_432" in source
    assert '"kind": "llm_checkpoint_parameter_average"' in source
