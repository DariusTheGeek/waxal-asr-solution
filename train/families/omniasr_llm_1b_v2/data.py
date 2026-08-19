"""Strict manifest reader with language conditioning and rank sharding."""

from __future__ import annotations

import io
from pathlib import Path

import torch
import torchaudio
from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataPipelineReader, DataReader, Seq2SeqBatch
from fairseq2.gang import Gangs
from typing_extensions import override

from omnilingual_asr.datasets.impl.manifest_asr_dataset import ManifestAsrDataset
from omnilingual_asr.datasets.storage.manifest_storage import (
    ManifestStorage,
    ManifestStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTask, AsrTaskConfig
from omnilingual_asr.datasets.utils.audio import (
    add_fbank_processing,
    add_waveform_processing,
    filter_by_audio_length,
)


class StrictPcmWavDecoder:
    """Decode derived PCM WAV bytes and enforce the frozen manifest contract."""

    def __init__(self, *, expected_sample_rate: int = 16_000) -> None:
        self.expected_sample_rate = expected_sample_rate

    def _decode_one(self, example: dict[str, object]) -> dict[str, object]:
        payload = bytes(example["audio"])
        waveform, sample_rate = torchaudio.load(io.BytesIO(payload), format="wav")
        if waveform.ndim != 2 or waveform.dtype != torch.float32:
            raise RuntimeError("unexpected WAV decoder output")
        channels, frames = (int(value) for value in waveform.shape)
        expected_frames = int(example["length"])
        if sample_rate != self.expected_sample_rate:
            raise RuntimeError(
                f"sample-rate drift: {sample_rate} != {self.expected_sample_rate}"
            )
        if channels != 1:
            raise RuntimeError(f"channel drift: {channels} != 1")
        if frames != expected_frames:
            raise RuntimeError(f"frame drift: {frames} != {expected_frames}")
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise RuntimeError("empty or non-finite decoded waveform")
        output = dict(example)
        output["audio"] = {
            "waveform": waveform.transpose(0, 1).contiguous(),
            "sample_rate": sample_rate,
            "format": "wav",
        }
        return output

    def __call__(
        self, examples: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return [self._decode_one(example) for example in examples]


class WaxalAsrTask(AsrTask):
    def __init__(self, config: AsrTaskConfig, *, language_code: str) -> None:
        super().__init__(config)
        self.language_code = language_code

    def attach_language_code(
        self, example: dict[str, object]
    ) -> dict[str, object]:
        """Attach the explicit OmniASR LID code before collation."""

        existing = example.get("lang")
        if existing is not None and existing != self.language_code:
            raise RuntimeError(
                f"language-code drift: {existing!r} != {self.language_code!r}"
            )
        output = dict(example)
        output["lang"] = self.language_code
        return output

    @override
    def apply_processing_pipeline(
        self,
        builder: DataPipelineBuilder,
        gangs: Gangs,
        tokenizer: Tokenizer,
        dtype: torch.dtype,
    ) -> DataPipelineBuilder:
        config = self.config
        builder = filter_by_audio_length(
            builder,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            length_selector="length",
        )
        builder = AsrTask.add_example_shuffling(
            builder,
            example_shuffle_window=config.example_shuffle_window,
            seed=config.seed,
        )
        config.seed += 1
        builder = AsrTask.add_tokenization_pipeline(
            builder,
            tokenizer,
            filter_long_text_threshold=config.filter_long_text_threshold,
            remove_unknown=config.remove_unknown,
            min_samples_per_char=config.min_samples_per_char,
            text_selector="text",
            audio_length_selector="length",
        )
        builder.map(self.attach_language_code)
        builder = AsrTask.add_bucketing_pipeline(
            builder,
            batching=config.batching_strategy,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            max_num_elements=config.max_num_elements,
            num_seqs_multiple_of=config.num_seqs_multiple_of,
            drop_remainder=config.drop_remainder,
            max_bucket_size=config.max_bucket_size,
            length_selector="length",
            batch_size=config.batch_size,
            no_padding=config.no_padding,
        )
        builder = AsrTask.add_batch_shuffling(
            builder,
            batch_shuffle_window=config.batch_shuffle_window,
            seed=config.seed,
        )
        builder = self.add_audio_processing_pipeline(
            builder,
            dtype=dtype,
            normalize_audio=config.normalize_audio,
            audio_selector="[*].audio",
            npc=config.npc,
            use_fbank=config.use_fbank,
            spec_aug_p=config.spec_aug_p,
            spec_aug_freq_mask_param=config.spec_aug_freq_mask_param,
            spec_aug_time_mask_param=config.spec_aug_time_mask_param,
            unified_audio_feature_keys=config.unified_audio_feature_keys,
        )
        return AsrTask.add_postprocessing_pipeline(
            builder,
            text_selector="text",
            pad_idx=tokenizer.vocab_info.pad_idx,
            npc=config.npc,
            max_num_batches=config.max_num_batches,
            num_prefetch=config.num_prefetch,
            no_padding=config.no_padding,
        )

    @staticmethod
    def add_audio_processing_pipeline(
        builder: DataPipelineBuilder,
        use_fbank: bool,
        audio_selector: str,
        dtype: torch.dtype,
        normalize_audio: bool,
        spec_aug_p: float | None,
        spec_aug_freq_mask_param: int,
        spec_aug_time_mask_param: int,
        npc: int,
        unified_audio_feature_keys: bool,
    ) -> DataPipelineBuilder:
        if audio_selector != "[*].audio":
            raise RuntimeError(f"unexpected audio selector: {audio_selector}")
        builder.map(StrictPcmWavDecoder(), num_parallel_calls=npc)
        if use_fbank:
            builder = add_fbank_processing(
                builder, dtype=dtype, selector="[*].audio", npc=npc
            )
        else:
            builder = add_waveform_processing(
                builder,
                normalize_audio=normalize_audio,
                dtype=dtype,
                selector="[*].audio.waveform",
                spec_aug_p=spec_aug_p,
                spec_aug_freq_mask_param=spec_aug_freq_mask_param,
                spec_aug_time_mask_param=spec_aug_time_mask_param,
            )
        return AsrTask.add_unified_naming(
            builder, unified_audio_feature_keys=unified_audio_feature_keys
        )


class WaxalManifestAsrDataset(ManifestAsrDataset):
    @classmethod
    def from_path(cls, path: Path) -> "WaxalManifestAsrDataset":
        splits, manifest_dir = ManifestStorage.discover_splits(path)
        return cls(manifest_dir, splits)

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gangs: Gangs,
        dtype: torch.dtype,
        num_accumulate: int,
        storage_config: ManifestStorageConfig,
        task_config: AsrTaskConfig,
        language_code: str,
    ) -> DataReader[Seq2SeqBatch]:
        storage = ManifestStorage(
            splits=self.splits,
            manifest_dir=self.manifest_dir,
            config=storage_config,
        )
        task = WaxalAsrTask(config=task_config, language_code=language_code)
        builder = storage.create_raw_data_pipeline(split, gangs)
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        builder = task.apply_processing_pipeline(
            builder, gangs, tokenizer=tokenizer, dtype=dtype
        )
        pipeline = builder.and_return()
        return DataPipelineReader[Seq2SeqBatch](
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=storage_config.sync_batches,
            sync_mode=storage_config.sync_mode,
        )


def open_waxal_manifest_asr_dataset(config) -> WaxalManifestAsrDataset:
    return WaxalManifestAsrDataset.from_path(config.data)
