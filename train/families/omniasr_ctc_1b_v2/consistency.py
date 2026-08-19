"""Paper-faithful dual-view consistency regularization for OmniASR CTC.

The training-only latent masker is attached after the official parent is
loaded.  It is removed explicitly when an inference checkpoint is exported;
``Wav2Vec2AsrModel`` never applies the masker in evaluation mode.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.models.wav2vec2 import Wav2Vec2Masker
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn import BatchLayout
from fairseq2.recipe.model import RecipeModel
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F

from workflows.recipes.wav2vec2.asr.default_config import (
    Wav2Vec2AsrRecipeConfig,
)
from workflows.recipes.wav2vec2.asr.metrics import (
    add_asr_metrics,
    update_asr_batch_metrics,
    update_ctc_loss,
)


TRAINING_ONLY_MASK_KEY = "masker.temporal_mask_embed"


@dataclass(kw_only=True)
class ConsistencyConfig:
    """Frozen dual-view and CR-CTC hyperparameters."""

    dual_view: bool = True
    cr_max_weight: float = 0.2
    cr_warmup_steps: int = 501
    temporal_mask_prob: float = 0.125
    temporal_mask_span: int = 10
    min_temporal_mask_spans: int = 1
    spatial_mask_prob: float = 0.0
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    view_b_noise_prob: float = 0.5
    view_b_snr_db_min: float = 15.0
    view_b_snr_db_max: float = 30.0
    noise_colors: tuple[str, ...] = ("white", "lowpass", "highpass")
    blank_idx: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConsistencyConfig":
        fields = dict(value)
        if "speed_factors" in fields:
            fields["speed_factors"] = tuple(float(x) for x in fields["speed_factors"])
        if "noise_colors" in fields:
            fields["noise_colors"] = tuple(str(x) for x in fields["noise_colors"])
        config = cls(**fields)
        validate_consistency_config(config)
        return config


@dataclass(kw_only=True)
class WaxalCtcRecipeConfig(Wav2Vec2AsrRecipeConfig):
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)


@dataclass(frozen=True)
class ViewStats:
    speed_factor_mean: float
    noise_examples: int
    batch_size: int
    noise_snr_db_mean: float


@dataclass(frozen=True)
class ConsistencyStats:
    loss: Tensor
    valid_frames: int
    blank_argmax_rate: Tensor
    posterior_entropy: Tensor
    argmax_disagreement: Tensor


def validate_consistency_config(config: ConsistencyConfig) -> None:
    if not config.dual_view:
        raise ValueError("this recipe requires dual_view=true")
    if not 0.0 <= config.cr_max_weight <= 1.0:
        raise ValueError("cr_max_weight must be in [0, 1]")
    if config.cr_warmup_steps <= 0:
        raise ValueError("cr_warmup_steps must be positive")
    if not 0.0 < config.temporal_mask_prob < 1.0:
        raise ValueError("temporal_mask_prob must be in (0, 1)")
    if config.temporal_mask_span <= 0 or config.min_temporal_mask_spans <= 0:
        raise ValueError("temporal mask span settings must be positive")
    if config.spatial_mask_prob != 0.0:
        raise ValueError("the first CR-CTC packet forbids spatial masking")
    if not config.speed_factors or any(value <= 0.0 for value in config.speed_factors):
        raise ValueError("speed_factors must be non-empty and positive")
    if not 0.0 <= config.view_b_noise_prob <= 1.0:
        raise ValueError("view_b_noise_prob must be in [0, 1]")
    if not 0.0 < config.view_b_snr_db_min <= config.view_b_snr_db_max:
        raise ValueError("invalid view-B SNR interval")
    allowed_colors = {"white", "lowpass", "highpass"}
    if not config.noise_colors or not set(config.noise_colors) <= allowed_colors:
        raise ValueError("noise_colors contains an unsupported value")
    if config.blank_idx != 0:
        raise ValueError("fairseq2 CTC blank index must remain zero")


def cr_weight(step_nr: int, config: ConsistencyConfig) -> float:
    """Linear 0-to-max warmup over optimizer steps, saturating thereafter."""

    if step_nr < 0:
        raise ValueError("step number cannot be negative")
    progress = min(float(step_nr) / float(config.cr_warmup_steps), 1.0)
    return float(config.cr_max_weight) * progress


def build_training_masker(
    model_dim: int,
    config: ConsistencyConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> "AdaptiveWav2Vec2Masker":
    validate_consistency_config(config)
    masker = AdaptiveWav2Vec2Masker(
        model_dim=model_dim,
        temporal_span_len=config.temporal_mask_span,
        max_temporal_mask_prob=config.temporal_mask_prob,
        min_num_temporal_mask_spans=config.min_temporal_mask_spans,
        max_spatial_mask_prob=config.spatial_mask_prob,
        device=device,
        dtype=dtype,
    )
    return masker


def attach_training_masker(
    model: Wav2Vec2AsrModel, config: ConsistencyConfig
) -> "AdaptiveWav2Vec2Masker":
    """Attach or verify the one permitted train-only model extension."""

    existing = model.masker
    if existing is None:
        parameter = model.final_proj.weight
        masker = build_training_masker(
            model.model_dim,
            config,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        model.masker = masker
        return masker
    if not isinstance(existing, AdaptiveWav2Vec2Masker):
        raise RuntimeError(f"unexpected parent masker type: {type(existing)}")
    expected = (
        config.temporal_mask_span,
        config.temporal_mask_prob,
        config.min_temporal_mask_spans,
        config.spatial_mask_prob,
    )
    observed = (
        existing.temporal_span_len,
        existing.max_temporal_mask_prob,
        existing.min_num_temporal_mask_spans,
        existing.max_spatial_mask_prob,
    )
    if observed != expected:
        raise RuntimeError(
            f"training masker configuration drift: {observed} != {expected}"
        )
    return existing


class AdaptiveWav2Vec2Masker(Wav2Vec2Masker):
    """Contiguous time masking that remains defined for short utterances.

    fairseq2's stock probability/span implementation raises when a short row
    cannot fit its configured minimum span count.  Lingala contains short
    clips, so this implementation shortens the single minimum span to retain
    approximately the requested masked-frame fraction.
    """

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int,
        max_temporal_mask_prob: float,
        min_num_temporal_mask_spans: int,
        max_spatial_mask_prob: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if temporal_span_len <= 0 or min_num_temporal_mask_spans <= 0:
            raise ValueError("temporal mask span settings must be positive")
        if not 0.0 < max_temporal_mask_prob < 1.0:
            raise ValueError("max_temporal_mask_prob must be in (0, 1)")
        if max_spatial_mask_prob != 0.0:
            raise ValueError("adaptive first-wave recipe forbids spatial masking")
        self.temporal_mask_embed = Parameter(
            torch.zeros((model_dim,), device=device, dtype=dtype)
        )
        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans
        self.max_spatial_mask_prob = max_spatial_mask_prob

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> tuple[Tensor, Tensor]:
        if seqs_layout.packed:
            raise ValueError("adaptive masking forbids packed batches")
        if seqs.ndim != 3:
            raise ValueError("adaptive masking expects [N,T,M] features")
        batch_size, max_length, _ = seqs.shape
        if batch_size != len(seqs_layout.seq_lens):
            raise ValueError("feature batch and layout disagree")
        mask = torch.zeros(
            (batch_size, max_length), dtype=torch.bool, device=seqs.device
        )
        for row, raw_length in enumerate(seqs_layout.seq_lens):
            length = int(raw_length)
            if length <= 0 or length > max_length:
                raise ValueError("invalid feature sequence length")
            target_frames = max(1, int(round(length * self.max_temporal_mask_prob)))
            span_width = min(self.temporal_span_len, target_frames, length)
            span_count = max(
                self.min_num_temporal_mask_spans,
                int(round(target_frames / span_width)),
            )
            max_start = length - span_width
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(span_count,),
                device=seqs.device,
            )
            for start_value in starts:
                start = int(start_value)
                mask[row, start : start + span_width] = True
        output = seqs.clone()
        output[mask] = self.temporal_mask_embed.type_as(output)
        return output, mask

    def extra_repr(self) -> str:
        return (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}"
        )


def strip_training_only_masker(
    state: Mapping[str, Tensor], *, require_masker: bool
) -> OrderedDict[str, Tensor]:
    """Return a base-architecture state suitable for clean inference."""

    present = TRAINING_ONLY_MASK_KEY in state
    if require_masker and not present:
        raise RuntimeError(f"missing training-only state key: {TRAINING_ONLY_MASK_KEY}")
    output: OrderedDict[str, Tensor] = OrderedDict()
    for name, tensor in state.items():
        if name == TRAINING_ONLY_MASK_KEY:
            if (
                tensor.ndim != 1
                or tensor.numel() <= 0
                or not torch.isfinite(tensor).all()
            ):
                raise RuntimeError("invalid training-only mask embedding")
            continue
        output[name] = tensor
    if len(output) != len(state) - int(present):
        raise RuntimeError("inference state filtering removed an unexpected key")
    return output


@torch.no_grad()
def _shared_speed_perturb(
    seqs: Tensor, layout: BatchLayout, factors: Sequence[float]
) -> tuple[Tensor, BatchLayout, list[float]]:
    if layout.packed:
        raise ValueError("dual-view augmentation forbids packed batches")
    if seqs.ndim not in {2, 3}:
        raise ValueError(f"expected [N,S] or [N,S,C] waveform, got {tuple(seqs.shape)}")
    if seqs.ndim == 3 and seqs.shape[-1] != 1:
        raise ValueError("dual-view augmentation requires mono waveforms")
    lengths = [int(value) for value in layout.seq_lens]
    if len(lengths) != seqs.shape[0] or any(value <= 0 for value in lengths):
        raise ValueError("invalid source layout")
    factor_indices = (
        torch.randint(
            low=0, high=len(factors), size=(seqs.shape[0],), device=seqs.device
        )
        .cpu()
        .tolist()
    )
    selected = [float(factors[index]) for index in factor_indices]
    outputs: list[Tensor] = []
    output_lengths: list[int] = []
    for index, (length, factor) in enumerate(zip(lengths, selected, strict=True)):
        source = seqs[index, :length]
        new_length = max(1, int(round(length / factor)))
        if seqs.ndim == 2:
            source_ncl = source.view(1, 1, length)
            resized = F.interpolate(
                source_ncl.float(), size=new_length, mode="linear", align_corners=False
            ).view(new_length)
        else:
            source_ncl = source.transpose(0, 1).unsqueeze(0)
            resized = (
                F.interpolate(
                    source_ncl.float(),
                    size=new_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        outputs.append(resized.to(dtype=seqs.dtype))
        output_lengths.append(new_length)
    max_length = max(output_lengths)
    output_shape = (len(outputs), max_length, *seqs.shape[2:])
    padded = seqs.new_zeros(output_shape)
    for index, (output, length) in enumerate(zip(outputs, output_lengths, strict=True)):
        padded[index, :length] = output
    return padded, BatchLayout.of(padded, output_lengths), selected


def _colored_noise(noise: Tensor, color: str) -> Tensor:
    if color == "white":
        return noise
    low = F.avg_pool1d(noise.view(1, 1, -1), kernel_size=9, stride=1, padding=4).view(
        -1
    )
    if color == "lowpass":
        return low
    if color == "highpass":
        return noise - low
    raise ValueError(f"unsupported noise color: {color}")


@torch.no_grad()
def _add_view_b_noise(
    seqs: Tensor, layout: BatchLayout, config: ConsistencyConfig
) -> tuple[Tensor, int, float]:
    output = seqs.clone()
    snrs: list[float] = []
    for index, length_value in enumerate(layout.seq_lens):
        length = int(length_value)
        if float(torch.rand((), device=seqs.device)) >= config.view_b_noise_prob:
            continue
        source = output[index, :length]
        flat = source.float().reshape(-1)
        source_rms = flat.square().mean().sqrt()
        if float(source_rms) <= 1.0e-8:
            continue
        noise = torch.randn_like(flat)
        color_index = int(
            torch.randint(
                low=0,
                high=len(config.noise_colors),
                size=(),
                device=seqs.device,
            )
        )
        noise = _colored_noise(noise, config.noise_colors[color_index])
        noise = noise - noise.mean()
        noise_rms = noise.square().mean().sqrt().clamp_min(1.0e-8)
        unit = noise / noise_rms
        snr = config.view_b_snr_db_min + float(torch.rand((), device=seqs.device)) * (
            config.view_b_snr_db_max - config.view_b_snr_db_min
        )
        scale = source_rms * math.pow(10.0, -snr / 20.0)
        perturbed = flat + unit * scale
        output[index, :length] = perturbed.reshape_as(source).to(source.dtype)
        snrs.append(snr)
    return output, len(snrs), sum(snrs) / len(snrs) if snrs else 0.0


@torch.no_grad()
def make_aligned_views(
    seqs: Tensor, layout: BatchLayout, config: ConsistencyConfig
) -> tuple[Tensor, Tensor, BatchLayout, ViewStats]:
    """Build two equal-length views; only view B receives waveform noise."""

    validate_consistency_config(config)
    shared, shared_layout, speeds = _shared_speed_perturb(
        seqs, layout, config.speed_factors
    )
    view_a = shared
    view_b, noise_examples, noise_snr_mean = _add_view_b_noise(
        shared, shared_layout, config
    )
    if view_a.shape != view_b.shape or list(shared_layout.seq_lens) != list(
        BatchLayout.of(view_b, list(shared_layout.seq_lens)).seq_lens
    ):
        raise RuntimeError("dual-view waveform alignment drift")
    if not torch.isfinite(view_a).all() or not torch.isfinite(view_b).all():
        raise RuntimeError("dual-view augmentation produced non-finite samples")
    return (
        view_a,
        view_b,
        shared_layout,
        ViewStats(
            speed_factor_mean=sum(speeds) / len(speeds),
            noise_examples=noise_examples,
            batch_size=len(speeds),
            noise_snr_db_mean=noise_snr_mean,
        ),
    )


def _valid_frame_mask(logits: Tensor, layout: BatchLayout) -> Tensor:
    if layout.packed or logits.ndim != 3:
        raise ValueError("CR-CTC requires unpacked [N,T,V] logits")
    lengths = layout.seq_lens_pt.to(device=logits.device)
    if logits.shape[0] != lengths.numel() or int(lengths.max()) > logits.shape[1]:
        raise ValueError("logit layout is incompatible with logits")
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def detached_kl_direction(
    input_logits: Tensor, target_logits: Tensor, valid_mask: Tensor
) -> Tensor:
    """KL(stopgrad(target) || input), summed over valid frames and vocabulary."""

    if input_logits.shape != target_logits.shape:
        raise ValueError("KL input/target shape mismatch")
    if valid_mask.shape != input_logits.shape[:2]:
        raise ValueError("KL valid-frame mask shape mismatch")
    input_log_probs = F.log_softmax(input_logits.float(), dim=-1)
    target_log_probs = F.log_softmax(target_logits.float(), dim=-1).detach()
    per_token = F.kl_div(
        input_log_probs,
        target_log_probs,
        reduction="none",
        log_target=True,
    )
    return per_token.sum(dim=-1).masked_fill(~valid_mask, 0.0).sum()


def symmetric_consistency_loss(
    logits_a: Tensor,
    layout_a: BatchLayout,
    logits_b: Tensor,
    layout_b: BatchLayout,
    *,
    blank_idx: int,
) -> ConsistencyStats:
    if logits_a.shape != logits_b.shape:
        raise RuntimeError("A/B logit tensor shape mismatch")
    lengths_a = [int(value) for value in layout_a.seq_lens]
    lengths_b = [int(value) for value in layout_b.seq_lens]
    if lengths_a != lengths_b:
        raise RuntimeError("A/B output layout mismatch")
    if not 0 <= blank_idx < logits_a.shape[-1]:
        raise ValueError("blank index is outside the vocabulary")
    valid = _valid_frame_mask(logits_a, layout_a)
    valid_frames = int(valid.sum())
    if valid_frames <= 0:
        raise RuntimeError("CR-CTC batch has no valid output frames")
    b_to_a = detached_kl_direction(logits_a, logits_b, valid)
    a_to_b = detached_kl_direction(logits_b, logits_a, valid)
    loss = 0.5 * (b_to_a + a_to_b)
    with torch.no_grad():
        log_a = F.log_softmax(logits_a.float(), dim=-1)
        log_b = F.log_softmax(logits_b.float(), dim=-1)
        prob_a = log_a.exp()
        prob_b = log_b.exp()
        entropy_a = -(prob_a * log_a).sum(dim=-1)
        entropy_b = -(prob_b * log_b).sum(dim=-1)
        normalizer = valid.sum().to(dtype=torch.float32)
        posterior_entropy = (
            0.5
            * (
                (entropy_a.masked_fill(~valid, 0.0).sum())
                + (entropy_b.masked_fill(~valid, 0.0).sum())
            )
            / normalizer
        )
        argmax_a = logits_a.argmax(dim=-1)
        argmax_b = logits_b.argmax(dim=-1)
        blank_rate = (
            0.5
            * (
                (argmax_a.eq(blank_idx) & valid).sum()
                + (argmax_b.eq(blank_idx) & valid).sum()
            )
            / normalizer
        )
        disagreement = ((argmax_a != argmax_b) & valid).sum() / normalizer
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite CR loss")
    return ConsistencyStats(
        loss=loss,
        valid_frames=valid_frames,
        blank_argmax_rate=blank_rate,
        posterior_entropy=posterior_entropy,
        argmax_disagreement=disagreement,
    )


class DualViewCtcCriterion:
    """Two-view CTC objective with optional consistency-loss weight."""

    def __init__(self, model: RecipeModel, config: ConsistencyConfig) -> None:
        validate_consistency_config(config)
        self._model = model
        self._config = config
        self._step_nr = 0

    def set_step_nr(self, step_nr: int) -> None:
        if step_nr < 0:
            raise ValueError("step number cannot be negative")
        self._step_nr = int(step_nr)

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_asr_metrics(metric_bag)
        metric_bag.add("objective_loss", Mean())
        metric_bag.add("cr_loss", Mean())
        metric_bag.add("cr_weight", Mean())
        metric_bag.add("valid_logit_frames", Sum())
        metric_bag.add("blank_argmax_rate", Mean())
        metric_bag.add("posterior_entropy", Mean())
        metric_bag.add("view_argmax_disagreement", Mean())
        metric_bag.add("view_b_noise_fraction", Mean())
        metric_bag.add("view_b_noise_snr_db", Mean())
        metric_bag.add("speed_factor", Mean())

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        source_seqs, source_layout = batch.as_source_input()
        target_seqs, target_layout = batch.as_target_input()
        view_a, view_b, view_layout, view_stats = make_aligned_views(
            source_seqs, source_layout, self._config
        )
        loss_a, logits_a, logits_layout_a = self._model.module(
            view_a,
            view_layout,
            target_seqs,
            target_layout,
            return_logits=True,
        )
        loss_b, logits_b, logits_layout_b = self._model.module(
            view_b,
            view_layout,
            target_seqs,
            target_layout,
            return_logits=True,
        )
        ctc_loss = 0.5 * (loss_a + loss_b)
        consistency = symmetric_consistency_loss(
            logits_a,
            logits_layout_a,
            logits_b,
            logits_layout_b,
            blank_idx=self._config.blank_idx,
        )
        weight = cr_weight(self._step_nr, self._config)
        objective = ctc_loss + weight * consistency.loss
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("non-finite dual-view CTC objective")
        self._update_metrics(
            metric_bag,
            batch,
            ctc_loss,
            objective,
            consistency,
            weight,
            view_stats,
        )
        return objective, batch.batch_size

    @torch.inference_mode()
    def _update_metrics(
        self,
        metric_bag: MetricBag,
        batch: Seq2SeqBatch,
        ctc_loss: Tensor,
        objective: Tensor,
        consistency: ConsistencyStats,
        weight: float,
        view_stats: ViewStats,
    ) -> None:
        update_ctc_loss(metric_bag, ctc_loss, batch.batch_size)
        update_asr_batch_metrics(metric_bag, batch)
        metric_bag.get("objective_loss", Mean).update(
            objective.detach() / batch.batch_size / math.log(2),
            weight=batch.batch_size,
        )
        metric_bag.get("cr_loss", Mean).update(
            consistency.loss.detach() / consistency.valid_frames / math.log(2),
            weight=consistency.valid_frames,
        )
        metric_bag.get("cr_weight", Mean).update(weight)
        metric_bag.get("valid_logit_frames", Sum).update(consistency.valid_frames)
        metric_bag.get("blank_argmax_rate", Mean).update(
            consistency.blank_argmax_rate, weight=consistency.valid_frames
        )
        metric_bag.get("posterior_entropy", Mean).update(
            consistency.posterior_entropy, weight=consistency.valid_frames
        )
        metric_bag.get("view_argmax_disagreement", Mean).update(
            consistency.argmax_disagreement, weight=consistency.valid_frames
        )
        metric_bag.get("view_b_noise_fraction", Mean).update(
            view_stats.noise_examples / view_stats.batch_size,
            weight=view_stats.batch_size,
        )
        metric_bag.get("view_b_noise_snr_db", Mean).update(
            view_stats.noise_snr_db_mean,
            weight=view_stats.batch_size,
        )
        metric_bag.get("speed_factor", Mean).update(
            view_stats.speed_factor_mean, weight=view_stats.batch_size
        )

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        return None

    @property
    def model(self) -> RecipeModel:
        return self._model
