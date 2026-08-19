#!/usr/bin/env python3
"""Raw-waveform dataset, character codec, collator, and exact-step sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices


class CharacterCodec:
    def __init__(self, vocabulary: dict[str, int]) -> None:
        if vocabulary.get("<pad>") != 0 or vocabulary.get("<unk>") != 1:
            raise ValueError("pad/unknown token IDs must be 0/1")
        if vocabulary.get("|") != 2:
            raise ValueError("word delimiter ID must be 2")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("vocabulary IDs must be unique and contiguous")
        self.vocabulary = dict(vocabulary)
        self.inverse = {value: key for key, value in vocabulary.items()}
        self.pad_id = 0
        self.unk_id = 1

    def encode(self, text: str) -> list[int]:
        return [
            self.vocabulary.get("|" if character == " " else character, self.unk_id)
            for character in text
        ]

    def decode_labels(self, values: list[int]) -> str:
        tokens = [
            self.inverse[int(value)]
            for value in values
            if int(value) >= 0 and int(value) != self.pad_id
        ]
        return " ".join("".join(tokens).replace("|", " ").split())

    def decode_ctc(self, values: list[int]) -> str:
        tokens: list[str] = []
        previous: int | None = None
        for raw_value in values:
            value = int(raw_value)
            if value == self.pad_id:
                previous = value
                continue
            if value == previous:
                continue
            previous = value
            token = self.inverse.get(value, "<unk>")
            if token not in {"<pad>", "<unk>"}:
                tokens.append(token)
        return " ".join("".join(tokens).replace("|", " ").split())


def minimum_ctc_frames(token_ids: list[int]) -> int:
    return len(token_ids) + sum(
        left == right for left, right in zip(token_ids, token_ids[1:])
    )


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        audio_root: Path,
        feature_extractor: Any,
        codec: CharacterCodec,
    ) -> None:
        self.rows = rows
        self.audio_root = audio_root
        self.feature_extractor = feature_extractor
        self.codec = codec
        self.lengths = [int(row["input_samples"]) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.audio_root / str(row["audio_relpath"])
        if path.stat().st_size != int(row["audio_bytes"]):
            raise RuntimeError(f"audio byte-size drift: {row['id']}")
        waveform, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
        if int(sample_rate) != 16_000:
            raise RuntimeError(f"sample-rate drift for {row['id']}: {sample_rate}")
        if waveform.shape[1] != 1:
            raise RuntimeError(
                f"channel-count drift for {row['id']}: {waveform.shape[1]}"
            )
        waveform = np.asarray(waveform[:, 0], dtype=np.float32)
        if waveform.size != int(row["input_samples"]):
            raise RuntimeError(
                f"decoded sample-count drift for {row['id']}: "
                f"{waveform.size} != {row['input_samples']}"
            )
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError(f"invalid waveform: {row['id']}")
        processed = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_attention_mask=True,
        )
        values = np.asarray(processed["input_values"][0], dtype=np.float32)
        if values.shape != waveform.shape or not np.isfinite(values).all():
            raise RuntimeError(f"feature-extractor output drift: {row['id']}")
        labels = self.codec.encode(str(row["target_ctc"]))
        if not labels or self.codec.unk_id in labels:
            raise RuntimeError(f"empty/OOV label: {row['id']}")
        return {
            "input_values": values,
            "labels": labels,
            "example_index": int(row["evaluation_index"]),
        }


@dataclass
class CTCCollator:
    feature_extractor: Any
    pad_token_id: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        features = [{"input_values": item["input_values"]} for item in examples]
        batch = self.feature_extractor.pad(
            features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if "input_values" not in batch or "attention_mask" not in batch:
            raise RuntimeError(
                "feature extractor did not return values and attention mask"
            )
        maximum = max(len(item["labels"]) for item in examples)
        labels = torch.full((len(examples), maximum), -100, dtype=torch.long)
        for index, item in enumerate(examples):
            values = torch.tensor(item["labels"], dtype=torch.long)
            labels[index, : values.numel()] = values
        batch["labels"] = labels
        # Trainer pads nested labels on dim 1 before distributed gather. Keep the
        # carried row index rank-two, matching labels and runtime output lengths.
        batch["example_index"] = torch.tensor(
            [int(item["example_index"]) for item in examples],
            dtype=torch.long,
        ).unsqueeze(-1)
        return batch


class OptimizationBatchPaddedLengthSampler(Sampler[int]):
    """Length-grouped sampler padded to complete global optimization batches.

    Transformers 4.46.3 computes the epoch horizon using complete gradient
    accumulation groups. Rounding the sampler itself to the global optimization
    batch prevents a partial microbatch from carrying gradients across an epoch
    boundary. Every real example occurs once; only the declared padding rows are
    repeated.
    """

    def __init__(
        self,
        *,
        lengths: list[int],
        group_batch_size: int,
        padding_multiple: int,
        seed: int,
    ) -> None:
        if not lengths or any(int(value) <= 0 for value in lengths):
            raise ValueError("sampler lengths must be positive")
        if group_batch_size < 1 or padding_multiple < 1:
            raise ValueError("sampler batch sizes must be positive")
        if padding_multiple % group_batch_size != 0:
            raise ValueError("padding multiple must be divisible by group batch size")
        self.lengths = [int(value) for value in lengths]
        self.group_batch_size = int(group_batch_size)
        self.padding_multiple = int(padding_multiple)
        self.seed = int(seed)
        self.epoch = 0
        self.padded_size = (
            (len(self.lengths) + self.padding_multiple - 1) // self.padding_multiple
        ) * self.padding_multiple

    @property
    def duplicate_rows(self) -> int:
        return self.padded_size - len(self.lengths)

    def __len__(self) -> int:
        return self.padded_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed((self.seed * 1_000_003 + self.epoch) % (2**63 - 1))
        base_indices = list(range(len(self.lengths)))
        if self.duplicate_rows:
            repeated = torch.randperm(len(self.lengths), generator=generator)[
                : self.duplicate_rows
            ].tolist()
        else:
            repeated = []
        expanded_indices = base_indices + repeated
        expanded_lengths = [self.lengths[index] for index in expanded_indices]
        order = get_length_grouped_indices(
            expanded_lengths,
            batch_size=self.group_batch_size,
            generator=generator,
        )
        return iter(expanded_indices[position] for position in order)
