from collections import OrderedDict

import pytest
import torch

from checkpoint_average import (
    CheckpointAverageError,
    average_states,
    tensor_content_sha256,
)


def _state(value: float, *, counter: int = 7) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "encoder.weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
            "lm_head.bias": torch.tensor([value, 0, 1], dtype=torch.float32),
            "counter": torch.tensor([counter], dtype=torch.int64),
        }
    )


def test_uniform_average_uses_float64_and_preserves_non_floats() -> None:
    output, contract = average_states([_state(1), _state(3), _state(8)])
    assert torch.equal(output["encoder.weight"], torch.tensor([[4.0, 5.0]]))
    assert torch.equal(output["counter"], torch.tensor([7]))
    assert contract["floating_tensor_count"] == 2
    assert contract["non_floating_tensor_count"] == 1


def test_non_floating_difference_fails_closed() -> None:
    with pytest.raises(CheckpointAverageError, match="non-floating"):
        average_states([_state(1), _state(3), _state(8, counter=9)])


def test_schema_and_nonfinite_drift_fail_closed() -> None:
    missing = _state(3)
    del missing["lm_head.bias"]
    with pytest.raises(CheckpointAverageError, match="schema"):
        average_states([_state(1), missing, _state(8)])
    bad = _state(float("nan"))
    with pytest.raises(CheckpointAverageError, match="non-finite"):
        average_states([_state(1), bad, _state(8)])


def test_tensor_content_hash_is_key_order_independent() -> None:
    first = _state(2)
    second = OrderedDict(reversed(list(first.items())))
    assert tensor_content_sha256(first) == tensor_content_sha256(second)
