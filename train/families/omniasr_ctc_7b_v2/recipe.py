"""Pinned OmniASR CTC recipe; FSDP-v2 layer sharding is selected by profile."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from types import MethodType

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SyncMode
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs
from fairseq2.recipe.base import RecipeContext
from fairseq2.recipe.trainer import Trainer
from fairseq2.runtime.dependency import DependencyContainer
from torch import Tensor

from omnilingual_asr.datasets.impl.manifest_asr_dataset import (
    MANIFEST_ASR_DATASET,
    ManifestAsrDatasetConfig,
)
from omnilingual_asr.datasets.storage.manifest_storage import ManifestStorageConfig

from workflows.recipes.wav2vec2.asr.criterion import Wav2Vec2AsrCriterion
from workflows.recipes.wav2vec2.asr.default_config import Wav2Vec2AsrRecipeConfig
from workflows.recipes.wav2vec2.asr.recipe import (
    Wav2Vec2AsrEvalUnit,
    Wav2Vec2AsrRecipe,
    Wav2Vec2AsrTrainUnit,
)
from workflows.recipes.wav2vec2.asr.wer_calculator import WerCalculator

from data import WaxalManifestAsrDataset, open_waxal_manifest_asr_dataset
from early_stopping import (
    TargetWeightedEarlyStopper,
    assert_validation_log_counts,
    target_early_stopping_enabled,
)
from checkpoint_contract import attach_checkpoint_contract
from trainer_runtime import attach_blocking_checkpoints_and_terminal_scoring
from fsdp_export import attach_export_after_restore
from runtime_config import runtime_geometry_from_environment
from runtime_assets import resolve_repo_root


MAX_PASSES = 12


def attach_stop_checkpoint_and_resume_validation(
    trainer: Trainer,
    *,
    output_dir,
    stopper: TargetWeightedEarlyStopper | None,
    rank_map_path,
    world_size: int,
    validation_interval_steps: int,
    updates_per_epoch: int,
) -> None:
    """Save graceful interruptions and replay only a missing boundary validation."""

    original_should_save = trainer._should_save_checkpoint

    def should_save(self: Trainer) -> bool:
        if str(self._state.name) == "STOP_REQUESTED" and int(self._step_nr) > 0:
            return True
        return bool(original_should_save())

    trainer._should_save_checkpoint = MethodType(should_save, trainer)
    if stopper is None:
        attach_blocking_checkpoints_and_terminal_scoring(
            trainer,
            stopper=None,
            validation_interval_steps=validation_interval_steps,
        )
        return
    original_restore = trainer._maybe_restore_state

    def restore(self: Trainer):
        state = original_restore()
        step = int(self._step_nr)
        if step <= 0 or step % updates_per_epoch:
            return state
        epoch = step // updates_per_epoch
        if stopper.completed_epochs == epoch:
            return state
        if stopper.completed_epochs != epoch - 1:
            raise RuntimeError("early-stop/checkpoint resume epoch drift")
        if self._gangs.root.rank == 0:
            recovery = assert_validation_log_counts(
                trainer_output_dir=output_dir,
                rank_map_path=rank_map_path,
                completed_epochs=stopper.completed_epochs,
                world_size=world_size,
            )
            path = output_dir / "early_stopping" / f"RESUME_REPLAY_{step:08d}.json"
            if path.exists():
                raise RuntimeError(f"validation replay record collision: {path}")
            path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._gangs.root.barrier()
        score = self._validate()
        if score is None:
            raise RuntimeError("resumed boundary validation produced no score")
        if self._maybe_request_early_stop(score):
            return type(state).EARLY_STOP
        return state

    trainer._maybe_restore_state = MethodType(restore, trainer)
    attach_blocking_checkpoints_and_terminal_scoring(
        trainer,
        stopper=stopper,
        validation_interval_steps=validation_interval_steps,
    )


class EmptySafeWerCalculator(WerCalculator):
    """Keep WER defined when an early CTC hypothesis is entirely blank."""

    def _generate_hypotheses(
        self, logits: Tensor, logit_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        hyp_seqs = []
        for sample_logits, logits_len in zip(logits, logit_layout.seq_lens):
            hyp_seq = sample_logits[:logits_len].argmax(-1).unique_consecutive()
            hyp_seq = hyp_seq[hyp_seq != self._blank_label]
            if hyp_seq.numel() == 0:
                hyp_seq = hyp_seq.new_tensor([self._pad_idx])
            hyp_seqs.append(hyp_seq)
        return pad_seqs(hyp_seqs, pad_value=self._pad_idx)


class WaxalWav2Vec2AsrRecipe(Wav2Vec2AsrRecipe):
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            MANIFEST_ASR_DATASET,
            WaxalManifestAsrDataset,
            ManifestAsrDatasetConfig,
            opener=open_waxal_manifest_asr_dataset,
        )

    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config.as_(Wav2Vec2AsrRecipeConfig)
        geometry = runtime_geometry_from_environment(resolve_repo_root())
        dataset = context.default_dataset.as_(WaxalManifestAsrDataset)
        if dataset.manifest_dir.resolve() != geometry.manifest_dir:
            raise RuntimeError(
                "dataset card/runtime manifest mismatch: "
                f"{dataset.manifest_dir.resolve()} != {geometry.manifest_dir}"
            )
        runtime_profile = os.environ.get("WAXAL3_PROFILE", "")
        if runtime_profile == "production":
            expected_steps = MAX_PASSES * geometry.updates_per_epoch
            if (
                int(config.regime.num_steps) != expected_steps
                or int(config.regime.validate_every_n_steps or -1)
                != geometry.updates_per_epoch
                or int(config.regime.checkpoint_every_n_steps or -1)
                != geometry.updates_per_epoch
            ):
                raise RuntimeError(
                    "production profile/runtime epoch geometry mismatch: "
                    f"steps={config.regime.num_steps}/{expected_steps} "
                    f"validate={config.regime.validate_every_n_steps}/"
                    f"{geometry.updates_per_epoch} "
                    f"checkpoint={config.regime.checkpoint_every_n_steps}/"
                    f"{geometry.updates_per_epoch}"
                )
        criterion = Wav2Vec2AsrCriterion(context.model)
        unit = Wav2Vec2AsrTrainUnit(
            criterion, config.trainer.freeze_encoder_for_n_steps
        )
        if config.dataset.train_split is None:
            raise ValueError("train_split must be defined")

        train_storage = deepcopy(config.dataset.manifest_storage_config)
        train_task = deepcopy(config.dataset.asr_task_config)
        train_task.seed = config.common.seed
        train_reader = dataset.create_reader(
            split=config.dataset.train_split,
            tokenizer=context.default_tokenizer,
            gangs=context.gangs,
            dtype=config.trainer.mixed_precision.dtype,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            storage_config=train_storage,
            task_config=train_task,
        )

        valid_units = []
        valid_readers = []
        if config.dataset.valid_split is not None:
            if not isinstance(context.model.base_module, Wav2Vec2AsrModel):
                raise RuntimeError("WAXAL3 control recipe is CTC-only")
            valid_criterion = Wav2Vec2AsrCriterion(
                model=context.model,
                wer_calculator=EmptySafeWerCalculator.from_context(context),
                llama_beam_search=None,
            )
            for split_index, split in enumerate(
                config.dataset.valid_split.split(",")
            ):
                valid_task = deepcopy(config.dataset.asr_task_config)
                valid_task.seed = config.common.seed + 1 + split_index
                valid_task.example_shuffle_window = 1
                valid_task.batch_shuffle_window = 1
                valid_task.batch_size = 8
                valid_storage: ManifestStorageConfig = deepcopy(
                    config.dataset.manifest_storage_config
                )
                valid_storage.sync_mode = SyncMode.UNTIL_LAST
                valid_units.append(Wav2Vec2AsrEvalUnit(valid_criterion))
                valid_readers.append(
                    dataset.create_reader(
                        split=split,
                        tokenizer=context.default_tokenizer,
                        gangs=context.gangs,
                        dtype=config.trainer.mixed_precision.dtype,
                        num_accumulate=1,
                        storage_config=valid_storage,
                        task_config=valid_task,
                    )
                )
        trainer = context.create_trainer(
            unit, train_reader, valid_units, valid_readers
        )
        stopper = None
        rank_map_path = dataset.manifest_dir / "dev.rank_map.world8.csv"
        export_mode = bool(os.environ.get("WAXAL3_EXPORT_FULL_STATE_DIR"))
        if target_early_stopping_enabled(config) and not export_mode:
            if config.dataset.valid_split is None:
                raise ValueError("target-aligned early stopping requires validation")
            if not hasattr(trainer, "_early_stopper"):
                raise RuntimeError(
                    "pinned fairseq2 Trainer no longer exposes _early_stopper"
                )
            if trainer._early_stopper is not None:
                raise RuntimeError("an unexpected early stopper is already configured")
            validation_interval = config.regime.validate_every_n_steps
            if validation_interval is None:
                raise ValueError("validation interval is required for early stopping")
            valid_split = config.dataset.valid_split
            stopper = TargetWeightedEarlyStopper(
                trainer_output_dir=context.output_dir,
                manifest_path=dataset.manifest_dir / f"{valid_split}.rows.parquet",
                rank_map_path=rank_map_path,
                world_size=context.gangs.dp.size,
                validation_interval_steps=validation_interval,
                warmup_epochs=4,
                patience=3,
                max_epochs=12,
            )
            trainer._early_stopper = stopper
        attach_stop_checkpoint_and_resume_validation(
            trainer,
            output_dir=context.output_dir,
            stopper=stopper,
            rank_map_path=rank_map_path,
            world_size=context.gangs.dp.size,
            validation_interval_steps=int(config.regime.validate_every_n_steps or 1),
            updates_per_epoch=geometry.updates_per_epoch,
        )
        retention_profile = (
            "smoke" if runtime_profile in {"smoke", "resume_smoke"} else runtime_profile
        )
        attach_checkpoint_contract(
            trainer,
            output_dir=context.output_dir,
            profile=retention_profile,
            validation_interval_steps=int(
                config.regime.validate_every_n_steps or 1
            ),
        )
        attach_export_after_restore(trainer)
        return trainer
