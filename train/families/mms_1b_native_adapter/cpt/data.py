#!/usr/bin/env python3
"""Deterministic audio crops, distributed epoch sampling, and wav2vec2 masks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Sampler
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)


def deterministic_u64(*values: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "big")


class CPTAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        audio_root: Path,
        sample_rate: int,
        max_audio_seconds: float,
        seed: int,
        crop_seed_mode: str = "presentation_slot",
    ) -> None:
        if not rows:
            raise ValueError("CPT dataset is empty")
        self.rows = rows
        self.audio_root = audio_root.resolve()
        self.sample_rate = int(sample_rate)
        self.max_samples = int(round(float(max_audio_seconds) * self.sample_rate))
        self.seed = int(seed)
        self.crop_seed_mode = str(crop_seed_mode)
        self.sweep = 0
        if self.max_samples <= 0:
            raise ValueError("maximum crop length must be positive")
        if self.crop_seed_mode not in {"presentation_slot", "s008_source_row"}:
            raise ValueError(f"unsupported CPT crop seed mode: {self.crop_seed_mode}")

    def __len__(self) -> int:
        return len(self.rows)

    def set_sweep(self, sweep: int) -> None:
        if int(sweep) < 1:
            raise ValueError("CPT sweep numbers are one-based")
        self.sweep = int(sweep)

    def set_epoch(self, epoch: int) -> None:
        # Kept as an explicit compatibility alias for torch-style loaders.
        self.set_sweep(int(epoch) + 1)

    def __getitem__(self, index: int | tuple[int, int, bool]) -> dict[str, Any]:
        if isinstance(index, tuple):
            row_index, presentation_slot, synchronization_padding = index
        else:
            row_index, presentation_slot, synchronization_padding = int(index), int(index), False
        row = self.rows[int(row_index)]
        path = self.audio_root / str(row["audio_relpath"])
        if not path.is_file() or path.stat().st_size != int(row["audio_bytes"]):
            raise RuntimeError(f"audio file/size drift: {row['id']}")
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if int(sample_rate) != self.sample_rate:
            raise RuntimeError(f"sample-rate drift for {row['id']}: {sample_rate}")
        if waveform.ndim != 2 or waveform.shape[1] != 1:
            raise RuntimeError(f"channel drift for {row['id']}: {waveform.shape}")
        waveform = np.asarray(waveform[:, 0], dtype=np.float32)
        if waveform.size != int(row["decoded_frames"]):
            raise RuntimeError(
                f"decoded-frame drift for {row['id']}: "
                f"{waveform.size} != {row['decoded_frames']}"
            )
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError(f"invalid waveform: {row['id']}")
        seed_values = (
            (self.seed, self.sweep - 1, row["id"])
            if self.crop_seed_mode == "s008_source_row"
            else (self.seed, self.sweep, presentation_slot, row["id"])
        )
        crop_seed = deterministic_u64(*seed_values, "crop")
        if waveform.size > self.max_samples:
            maximum_start = waveform.size - self.max_samples
            start = crop_seed % (maximum_start + 1)
            waveform = waveform[start : start + self.max_samples]
        return {
            "id": str(row["id"]),
            "input_values": np.ascontiguousarray(waveform),
            "valid_samples": int(waveform.size),
            "mask_seed": deterministic_u64(*seed_values, "mask"),
            "synchronization_padding": bool(synchronization_padding),
            "presentation_slot": int(presentation_slot),
        }


class PresentationIndex(NamedTuple):
    row_index: int
    presentation_slot: int
    synchronization_padding: bool


class StagePaddedDistributedSampler(Sampler[PresentationIndex]):
    """Preserve the frozen broad/tail order and pad each stage to global batch.

    Every source row appears exactly once per sweep. The few synchronization
    slots are declared duplicate presentations with fresh deterministic
    crop/mask seeds; they are never misreported as unique exposure.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
    ) -> None:
        if not rows or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid stage-padded sampler dimensions")
        self.rows = rows
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.sweep = 1
        self.global_update_rows = (
            self.world_size
            * self.per_device_batch_size
            * self.gradient_accumulation_steps
        )
        if self.global_update_rows <= 0:
            raise ValueError("global update size must be positive")
        stage_names = [str(row["stage"]) for row in rows]
        if set(stage_names) != {"broad", "tail"}:
            raise ValueError("CPT rows must contain exactly broad and tail stages")
        self.stage_indices: dict[str, list[int]] = {}
        for stage in ("broad", "tail"):
            indices = [index for index, row in enumerate(rows) if row["stage"] == stage]
            expected_order = list(range(len(indices)))
            observed_order = [int(rows[index]["stage_order_index"]) for index in indices]
            if observed_order != expected_order:
                raise ValueError(f"{stage} stage order is not the frozen contiguous order")
            self.stage_indices[stage] = indices
        self.padding_by_stage = {
            stage: (-len(indices)) % self.global_update_rows
            for stage, indices in self.stage_indices.items()
        }
        self.global_size = sum(
            len(self.stage_indices[stage]) + self.padding_by_stage[stage]
            for stage in ("broad", "tail")
        )
        if self.global_size % self.world_size:
            raise RuntimeError("stage-padded global size is not rank divisible")
        self.rank_size = self.global_size // self.world_size

    @property
    def unique_rows_per_sweep(self) -> int:
        return len(self.rows)

    @property
    def synchronization_padding_slots(self) -> int:
        return sum(self.padding_by_stage.values())

    @property
    def optimizer_updates_per_sweep(self) -> int:
        return self.global_size // self.global_update_rows

    def set_sweep(self, sweep: int) -> None:
        if int(sweep) < 1:
            raise ValueError("CPT sweep numbers are one-based")
        self.sweep = int(sweep)

    def set_epoch(self, epoch: int) -> None:
        self.set_sweep(int(epoch) + 1)

    def __len__(self) -> int:
        return self.rank_size

    def global_presentations(self) -> list[PresentationIndex]:
        presentations: list[PresentationIndex] = []
        slot = 0
        for stage in ("broad", "tail"):
            indices = self.stage_indices[stage]
            for row_index in indices:
                presentations.append(PresentationIndex(row_index, slot, False))
                slot += 1
            padding = self.padding_by_stage[stage]
            if padding:
                start = deterministic_u64(
                    self.seed, self.sweep, stage, "sync-padding"
                ) % len(indices)
                for offset in range(padding):
                    row_index = indices[(start + offset) % len(indices)]
                    presentations.append(PresentationIndex(row_index, slot, True))
                    slot += 1
        if len(presentations) != self.global_size:
            raise RuntimeError("stage-padded presentation count drift")
        return presentations

    def padding_records(self) -> list[dict[str, Any]]:
        records = []
        for item in self.global_presentations():
            if item.synchronization_padding:
                row = self.rows[item.row_index]
                records.append(
                    {
                        "stage": str(row["stage"]),
                        "id": str(row["id"]),
                        "row_index": item.row_index,
                        "presentation_slot": item.presentation_slot,
                    }
                )
        return records

    def __iter__(self) -> Iterator[PresentationIndex]:
        global_presentations = self.global_presentations()
        local = global_presentations[self.rank :: self.world_size]
        if len(local) != self.rank_size:
            raise RuntimeError("stage-padded rank sampler length drift")
        return iter(local)


class SpeakerInterleavedDistributedSampler(Sampler[int]):
    """WAXAL2 S008 source-row sweep with deterministic speaker interleaving.

    Each source row is visited at most once in a sweep.  The global order is
    truncated only to a complete optimizer update; the omitted tail changes
    deterministically with the sweep, and no omitted row is misreported as
    exposure or synchronization padding.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
    ) -> None:
        if not rows or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid speaker-interleaved sampler dimensions")
        if {str(row.get("stage", "")) for row in rows} != {"broad"}:
            raise ValueError("S008 source-row sampling requires one broad stage")
        observed_order = [int(row["stage_order_index"]) for row in rows]
        if observed_order != list(range(len(rows))):
            raise ValueError("S008 source-row stage order is not contiguous")
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            speaker = str(row.get("speaker_key", ""))
            if not speaker:
                raise ValueError(f"row lacks speaker_key: {row.get('id')}")
            groups[speaker].append(index)
        self.rows = rows
        self.groups = {speaker: indices for speaker, indices in sorted(groups.items())}
        self.size = len(rows)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.sweep = 1
        self.global_update_rows = (
            self.world_size
            * self.per_device_batch_size
            * self.gradient_accumulation_steps
        )
        self.usable_size = (self.size // self.global_update_rows) * self.global_update_rows
        if self.usable_size < self.global_update_rows:
            raise ValueError("manifest cannot fill one global optimizer update")
        self.rank_size = self.usable_size // self.world_size

    @property
    def source_rows(self) -> int:
        return self.size

    @property
    def unique_rows_per_sweep(self) -> int:
        return self.usable_size

    @property
    def dropped_rows(self) -> int:
        return self.size - self.usable_size

    @property
    def synchronization_padding_slots(self) -> int:
        return 0

    @property
    def optimizer_updates_per_sweep(self) -> int:
        return self.usable_size // self.global_update_rows

    def set_sweep(self, sweep: int) -> None:
        if int(sweep) < 1:
            raise ValueError("CPT sweep numbers are one-based")
        self.sweep = int(sweep)

    def set_epoch(self, epoch: int) -> None:
        self.set_sweep(int(epoch) + 1)

    def __len__(self) -> int:
        return self.rank_size

    def global_order(self) -> list[int]:
        epoch = self.sweep - 1
        queues: dict[str, deque[int]] = {}
        for speaker, indices in self.groups.items():
            generator = torch.Generator()
            generator.manual_seed(
                deterministic_u64(self.seed, epoch, speaker, "rows") % (2**63 - 1)
            )
            order = torch.randperm(len(indices), generator=generator).tolist()
            queues[speaker] = deque(indices[position] for position in order)
        speakers = list(queues)
        result: list[int] = []
        round_index = 0
        while speakers:
            generator = torch.Generator()
            generator.manual_seed(
                deterministic_u64(
                    self.seed, epoch, round_index, "speaker_round"
                )
                % (2**63 - 1)
            )
            permutation = torch.randperm(len(speakers), generator=generator).tolist()
            for position in permutation:
                speaker = speakers[position]
                result.append(queues[speaker].popleft())
            speakers = [speaker for speaker in speakers if queues[speaker]]
            round_index += 1
        if len(result) != self.size or len(set(result)) != self.size:
            raise RuntimeError("speaker-interleaved sweep lost or repeated rows")
        return result

    def dropped_records(self) -> list[dict[str, Any]]:
        return [
            {
                "id": str(self.rows[index]["id"]),
                "row_index": index,
                "speaker_key": str(self.rows[index]["speaker_key"]),
            }
            for index in self.global_order()[self.usable_size :]
        ]

    def padding_records(self) -> list[dict[str, Any]]:
        return []

    def __iter__(self) -> Iterator[int]:
        global_order = self.global_order()[: self.usable_size]
        rank_order = global_order[self.rank : self.usable_size : self.world_size]
        if len(rank_order) != self.rank_size:
            raise RuntimeError("speaker-interleaved rank sampler length drift")
        return iter(rank_order)


class DistributedEpochSampler(Sampler[int]):
    """Deterministic global permutation truncated to full optimizer updates."""

    def __init__(
        self,
        *,
        size: int,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
    ) -> None:
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed sampler dimensions")
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.epoch = 0
        global_update_rows = (
            self.world_size
            * self.per_device_batch_size
            * self.gradient_accumulation_steps
        )
        self.usable_size = (self.size // global_update_rows) * global_update_rows
        if self.usable_size == 0:
            raise ValueError("dataset is smaller than one global optimizer update")
        self.rank_size = self.usable_size // self.world_size

    @property
    def dropped_rows(self) -> int:
        return self.size - self.usable_size

    @property
    def optimizer_updates_per_epoch(self) -> int:
        denominator = self.per_device_batch_size * self.gradient_accumulation_steps
        return self.rank_size // denominator

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.rank_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            deterministic_u64(self.seed, self.epoch, "sampler") % (2**63 - 1)
        )
        permutation = torch.randperm(self.size, generator=generator)[: self.usable_size]
        indices = permutation[self.rank : self.usable_size : self.world_size]
        if indices.numel() != self.rank_size:
            raise RuntimeError("distributed rank sampler length drift")
        return iter(indices.tolist())


def convolution_output_length(
    input_length: int,
    kernels: list[int],
    strides: list[int],
) -> int:
    value = int(input_length)
    for kernel, stride in zip(kernels, strides):
        value = (value - int(kernel)) // int(stride) + 1
    return value


@dataclass
class CPTCollator:
    feature_extractor: Any
    sample_rate: int
    max_samples: int
    conv_kernel: list[int]
    conv_stride: list[int]
    mask_time_prob: float
    mask_time_length: int
    mask_time_min_masks: int
    num_negatives: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty CPT batch")
        waveforms = [item["input_values"] for item in examples]
        batch = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if tuple(batch["input_values"].shape) != (len(examples), self.max_samples):
            raise RuntimeError(f"feature-extractor padding drift: {batch['input_values'].shape}")
        raw_attention = batch["attention_mask"].to(torch.long)
        valid_samples = raw_attention.sum(-1).tolist()
        expected_samples = [int(item["valid_samples"]) for item in examples]
        if valid_samples != expected_samples:
            raise RuntimeError(
                f"raw attention-mask drift: {valid_samples} != {expected_samples}"
            )
        maximum_features = convolution_output_length(
            self.max_samples, self.conv_kernel, self.conv_stride
        )
        feature_attention = np.zeros(
            (len(examples), maximum_features), dtype=np.int64
        )
        for index, samples in enumerate(valid_samples):
            length = convolution_output_length(
                int(samples), self.conv_kernel, self.conv_stride
            )
            if length <= self.mask_time_length:
                raise RuntimeError(
                    f"audio too short for masking: {examples[index]['id']} -> {length}"
                )
            feature_attention[index, :length] = 1

        combined_seed = deterministic_u64(
            *(item["mask_seed"] for item in examples), "wav2vec2_masks"
        ) % (2**32 - 1)
        state = np.random.get_state()
        try:
            np.random.seed(combined_seed)
            mask_time_indices = _compute_mask_indices(
                shape=(len(examples), maximum_features),
                mask_prob=float(self.mask_time_prob),
                mask_length=int(self.mask_time_length),
                attention_mask=torch.from_numpy(feature_attention),
                min_masks=int(self.mask_time_min_masks),
            )
            sampled_negative_indices = _sample_negative_indices(
                features_shape=(len(examples), maximum_features),
                num_negatives=int(self.num_negatives),
                mask_time_indices=mask_time_indices,
            )
        finally:
            np.random.set_state(state)
        masked = int(mask_time_indices.sum())
        if masked < len(examples) * self.mask_time_min_masks:
            raise RuntimeError(f"insufficient masked positions: {masked}")
        return {
            "input_values": batch["input_values"],
            "attention_mask": raw_attention,
            "mask_time_indices": torch.from_numpy(mask_time_indices).to(torch.bool),
            "sampled_negative_indices": torch.from_numpy(
                sampled_negative_indices
            ).to(torch.long),
            "masked_positions": torch.tensor(masked, dtype=torch.long),
            "valid_samples": torch.tensor(valid_samples, dtype=torch.long),
            "ids": [str(item["id"]) for item in examples],
            "synchronization_padding": torch.tensor(
                [bool(item["synchronization_padding"]) for item in examples],
                dtype=torch.bool,
            ),
            "presentation_slots": torch.tensor(
                [int(item["presentation_slot"]) for item in examples],
                dtype=torch.long,
            ),
        }
