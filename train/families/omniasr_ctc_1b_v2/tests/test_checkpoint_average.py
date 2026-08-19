from collections import OrderedDict

import pytest
import torch

from checkpoint_average import (
    CheckpointAverageError,
    average_states,
    tensor_content_sha256,
    validate_selected_steps,
)


def _state(value: float, *, counter: int = 7) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "encoder.weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
            "final_proj.bias": torch.tensor([value, 0, 1], dtype=torch.float32),
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
    del missing["final_proj.bias"]
    with pytest.raises(CheckpointAverageError, match="schema"):
        average_states([_state(1), missing, _state(8)])
    with pytest.raises(CheckpointAverageError, match="non-finite"):
        average_states([_state(1), _state(float("nan")), _state(8)])


def test_tensor_content_hash_is_key_order_independent() -> None:
    first = _state(2)
    second = OrderedDict(reversed(list(first.items())))
    assert tensor_content_sha256(first) == tensor_content_sha256(second)


def test_score_rank_and_chronological_source_order_may_differ() -> None:
    validate_selected_steps(
        observed_steps=[2505, 3006, 4008],
        curve_ranked_steps=[4008, 2505, 3006],
        gate_ranked_steps=[4008, 2505, 3006],
    )


@pytest.mark.parametrize(
    ("observed", "curve", "gate", "message"),
    [
        ([4008, 2505, 3006], [4008, 2505, 3006], [4008, 2505, 3006], "chronological"),
        ([2505, 3006, 4008], [4008, 2505, 4509], [4008, 2505, 3006], "curve top three"),
        ([2505, 3006, 4008], [4008, 2505, 3006], [4008, 2505, 4509], "gate top-three"),
    ],
)
def test_selected_step_drift_fails_closed(
    observed: list[int], curve: list[int], gate: list[int], message: str
) -> None:
    with pytest.raises(CheckpointAverageError, match=message):
        validate_selected_steps(observed, curve, gate)
