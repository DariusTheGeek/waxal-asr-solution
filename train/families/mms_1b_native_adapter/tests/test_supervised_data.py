from __future__ import annotations

import numpy as np
import torch

from supervised.data import (
    CharacterCodec,
    CTCCollator,
    OptimizationBatchPaddedLengthSampler,
    minimum_ctc_frames,
)
from supervised.model import align_prediction_bundle
from supervised.train import _duration_spread


def vocabulary() -> dict[str, int]:
    return {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4, "'": 5}


def test_codec_ctc_blank_reset_and_minimum_geometry() -> None:
    codec = CharacterCodec(vocabulary())
    assert codec.encode("a b") == [3, 2, 4]
    assert codec.decode_ctc([3, 3, 0, 3, 2, 4, 4]) == "aa b"
    assert minimum_ctc_frames([3, 3, 2, 4, 4]) == 7


def test_optimizer_sampler_is_complete_padded_and_epoch_deterministic() -> None:
    sampler = OptimizationBatchPaddedLengthSampler(
        lengths=list(range(1, 34)),
        group_batch_size=8,
        padding_multiple=32,
        seed=42,
    )
    assert len(sampler) == 64
    assert sampler.duplicate_rows == 31
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)
    sampler.set_epoch(4)
    third = list(sampler)
    assert first == second
    assert first != third
    assert len(first) == 64
    assert set(first) == set(range(33))
    counts = {index: first.count(index) for index in range(33)}
    assert all(value in {1, 2} for value in counts.values())
    assert sum(value == 2 for value in counts.values()) == 31


def test_duration_spread_contains_both_extremes() -> None:
    rows = [{"id": str(index), "duration_s": float(index)} for index in range(100)]
    selected = _duration_spread(rows, 32)
    assert len(selected) == 32
    assert selected[0]["duration_s"] == 0.0
    assert selected[-1]["duration_s"] == 99.0
    assert len({row["id"] for row in selected}) == 32


def test_collator_normalizes_padding_contract_and_keeps_rank_two_index() -> None:
    class FeaturePad:
        @staticmethod
        def pad(features, padding, return_attention_mask, return_tensors):
            assert padding is True
            assert return_attention_mask is True
            assert return_tensors == "pt"
            maximum = max(len(item["input_values"]) for item in features)
            values = torch.zeros((len(features), maximum), dtype=torch.float32)
            attention = torch.zeros((len(features), maximum), dtype=torch.long)
            for index, item in enumerate(features):
                length = len(item["input_values"])
                values[index, :length] = torch.from_numpy(item["input_values"])
                attention[index, :length] = 1
            return {"input_values": values, "attention_mask": attention}

    examples = [
        {
            "input_values": np.ones(5, dtype=np.float32),
            "labels": [3, 4],
            "example_index": 8,
        },
        {
            "input_values": np.ones(3, dtype=np.float32),
            "labels": [3],
            "example_index": 9,
        },
    ]
    batch = CTCCollator(FeaturePad(), 0)(examples)
    assert tuple(batch["input_values"].shape) == (2, 5)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["labels"].tolist() == [[3, 4], [3, -100]]
    assert batch["example_index"].tolist() == [[8], [9]]


def test_alignment_restores_carried_order_and_clips_padded_logits() -> None:
    codec = CharacterCodec(vocabulary())
    references = [
        {
            "evaluation_index": 0,
            "id": "first",
            "language": "ach",
            "stratum": "warm",
            "speaker_key": "ach:1",
            "duration_s": 10.0,
            "original_split": "train",
            "is_phase2_test_speaker": True,
            "is_phase2_test_prompt": False,
            "target_raw": "Ab.",
            "target_ctc": "ab",
            "target_weight": 0.5,
            "slot_id": "lin:a:r1",
            "target_phase2_id": "a",
            "target_official_order": 0,
        },
        {
            "evaluation_index": 1,
            "id": "second",
            "language": "ach",
            "stratum": "cold",
            "speaker_key": "ach:2",
            "duration_s": 20.0,
            "original_split": "validation",
            "is_phase2_test_speaker": False,
            "is_phase2_test_prompt": True,
            "target_raw": "AB!",
            "target_ctc": "ab",
            "target_weight": 1.0,
            "slot_id": "lin:b:r1",
            "target_phase2_id": "b",
            "target_official_order": 1,
        },
    ]
    logits = np.zeros((2, 4, len(vocabulary())), dtype=np.float32)
    logits[:, 0, 3] = 9
    logits[:, 1, 4] = 9
    logits[:, 2:, 5] = 9
    aligned = align_prediction_bundle(
        predictions=logits,
        label_bundle=(
            np.asarray([[3, 4], [3, 4]]),
            np.asarray([[1], [0]]),
            np.asarray([[2], [2]]),
        ),
        codec=codec,
        reference_rows=references,
    )
    assert [row["id"] for row in aligned] == ["first", "second"]
    assert [row["hypothesis"] for row in aligned] == ["ab", "ab"]
    assert aligned[0]["is_phase2_test_speaker"] is True
    assert aligned[1]["original_split"] == "validation"
