#!/usr/bin/env python3
"""CPU-only numerical closure for the untouched OmniASR CTC-1B-v2 parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import polars as pl
import torch
import yaml
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from fairseq2.nn import BatchLayout

from consistency import (
    ConsistencyConfig,
    TRAINING_ONLY_MASK_KEY,
    attach_training_masker,
    symmetric_consistency_loss,
)
from runtime_assets import render_asset_cards, resolve_repo_root, verify_anchor


EXPECTED_PARENT_PARAMETERS = 975_675_056
EXPECTED_VOCABULARY = 10_288


def feature_output_length(num_samples: int) -> int:
    """Exact no-padding convolutional length for the frozen CTC-1B frontend."""

    length = int(num_samples)
    for kernel, stride in (
        (10, 5),
        (3, 2),
        (3, 2),
        (3, 2),
        (3, 2),
        (2, 2),
        (2, 2),
    ):
        length = (length - kernel) // stride + 1
    return length


def validate_fastest_view_ctc_alignment(
    tokenizer, manifest_dir: Path, fastest_factor: float
) -> dict[str, int | float]:
    tsv_lines = (manifest_dir / "train.tsv").read_text(encoding="utf-8").splitlines()
    targets = (manifest_dir / "train.wrd").read_text(encoding="utf-8").splitlines()
    if len(tsv_lines) != len(targets) + 1:
        raise RuntimeError("training TSV/target row-count drift")
    encoder = tokenizer.create_encoder()
    margins: list[int] = []
    maximum_target_tokens = 0
    for row, (tsv_line, target) in enumerate(zip(tsv_lines[1:], targets, strict=True)):
        raw_samples = int(tsv_line.rsplit("\t", 1)[1])
        augmented_samples = max(1, int(round(raw_samples / fastest_factor)))
        output_frames = feature_output_length(augmented_samples)
        token_ids = encoder(target).tolist()
        required_frames = len(token_ids) + sum(
            left == right for left, right in zip(token_ids, token_ids[1:])
        )
        margin = output_frames - required_frames
        if margin < 0:
            raise RuntimeError(
                f"fastest speed view is CTC-infeasible at train row {row}: "
                f"output={output_frames} required={required_frames}"
            )
        margins.append(margin)
        maximum_target_tokens = max(maximum_target_tokens, len(token_ids))
    ordered = sorted(margins)
    return {
        "rows": len(margins),
        "fastest_speed_factor": fastest_factor,
        "minimum_ctc_frame_margin": ordered[0],
        "p01_ctc_frame_margin": ordered[len(ordered) // 100],
        "maximum_target_tokens": maximum_target_tokens,
        "infeasible_rows": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    torch.set_num_interop_threads(4)
    root = resolve_repo_root()
    experiment = args.experiment.expanduser().resolve()
    try:
        experiment.relative_to(root)
    except ValueError as error:
        raise RuntimeError("experiment must be inside WAXAL3") from error
    profile = yaml.safe_load(
        (experiment / "profiles/production.yaml").read_text(encoding="utf-8")
    )
    consistency = ConsistencyConfig.from_mapping(profile["consistency"])
    parent = root / "models/omniasr-ctc-1b/omniASR-CTC-1B-v2.pt"
    tokenizer_path = root / "models/omniasr-ctc-1b/omniASR_tokenizer_written_v2.model"
    verify_anchor(
        parent,
        "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c",
        3_902_956_068,
    )
    verify_anchor(
        tokenizer_path,
        "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
        91_481,
    )
    with tempfile.TemporaryDirectory(prefix="waxal3-ctc1b-cards-") as temporary:
        render_asset_cards(
            Path(__file__).resolve().parent / "cards/waxal3.yaml.template",
            Path(temporary),
        )
        import omnilingual_asr  # noqa: F401

        tokenizer = load_tokenizer("waxal3_omni_tokenizer_written_v2")
        model = load_model(
            "waxal3_omni_ctc_1b_v2_target_es_parent",
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
            progress=False,
        )
        parent_parameters = sum(parameter.numel() for parameter in model.parameters())
        if parent_parameters != EXPECTED_PARENT_PARAMETERS:
            raise RuntimeError("official parent parameter-count drift")
        parent_state_keys = set(model.state_dict())
        masker = attach_training_masker(model, consistency)
        training_state_keys = set(model.state_dict())
        parameters = sum(parameter.numel() for parameter in model.parameters())
        expected_parameters = EXPECTED_PARENT_PARAMETERS + model.model_dim
        if masker.temporal_mask_embed.numel() != model.model_dim:
            raise RuntimeError("training-only mask embedding has the wrong width")
        if not torch.equal(
            masker.temporal_mask_embed,
            torch.zeros_like(masker.temporal_mask_embed),
        ):
            raise RuntimeError("training-only mask embedding is not neutral")
        torch.manual_seed(42)
        probe = torch.zeros((32, 400, model.model_dim), dtype=torch.float32)
        probe_layout = BatchLayout.of(probe, [400] * 32)
        _, temporal_mask = masker(probe, probe_layout)
        effective_mask_fraction = float(temporal_mask.float().mean())
        torch.manual_seed(73)
        samples = torch.linspace(-0.1, 0.1, 16_000).unsqueeze(0)
        layout = BatchLayout.of(samples, [16_000])
        targets = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        target_layout = BatchLayout.of(targets, [3])
        model.train()
        with torch.inference_mode():
            ctc_a, logits_a, output_layout_a = model(
                samples,
                layout,
                targets,
                target_layout,
                return_logits=True,
            )
            ctc_b, logits_b, output_layout_b = model(
                samples,
                layout,
                targets,
                target_layout,
                return_logits=True,
            )
        cr_probe = symmetric_consistency_loss(
            logits_a,
            output_layout_a,
            logits_b,
            output_layout_b,
            blank_idx=consistency.blank_idx,
        )
        model.eval()
        with torch.inference_mode():
            logits, output_layout = model(samples, layout)
            training_masker = model.masker
            model.masker = None
            clean_logits, clean_output_layout = model(samples, layout)
            model.masker = training_masker
    manifest_dir = root / "data/derived/portable/omniasr1b_lin_cv002_v1/manifests"
    ctc_alignment = validate_fastest_view_ctc_alignment(
        tokenizer, manifest_dir, max(consistency.speed_factors)
    )
    train_rows = pl.read_parquet(manifest_dir / "train.rows.parquet").height
    validation_rows = pl.read_parquet(manifest_dir / "dev.rows.parquet").height
    rank_map = pl.read_csv(manifest_dir / "dev.rank_map.world8.csv")
    checks = {
        "parent_parameters": parent_parameters == EXPECTED_PARENT_PARAMETERS,
        "parameters": parameters == expected_parameters,
        "training_parameter_delta": parameters - parent_parameters == model.model_dim,
        "only_training_state_key": training_state_keys - parent_state_keys
        == {TRAINING_ONLY_MASK_KEY}
        and parent_state_keys <= training_state_keys,
        "mask_embedding_trainable": masker.temporal_mask_embed.requires_grad,
        "neutral_mask_embedding": bool(
            torch.equal(
                masker.temporal_mask_embed,
                torch.zeros_like(masker.temporal_mask_embed),
            )
        ),
        "effective_mask_fraction": 0.08 <= effective_mask_fraction <= 0.14,
        "two_view_output_layout_equal": list(output_layout_a.seq_lens)
        == list(output_layout_b.seq_lens),
        "two_view_cr_finite": bool(torch.isfinite(cr_probe.loss)),
        "two_view_cr_positive": float(cr_probe.loss) > 0.0,
        "two_view_ctc_finite": bool(torch.isfinite(ctc_a))
        and bool(torch.isfinite(ctc_b)),
        "two_view_ctc_positive": float(ctc_a) > 0.0 and float(ctc_b) > 0.0,
        "two_view_valid_frames": cr_probe.valid_frames
        == sum(int(value) for value in output_layout_a.seq_lens),
        "vocabulary": tokenizer.vocab_info.size == EXPECTED_VOCABULARY,
        "finite_logits": bool(torch.isfinite(logits).all()),
        "logit_vocabulary": int(logits.shape[-1]) == EXPECTED_VOCABULARY,
        "output_length_positive": int(output_layout.seq_lens[0]) > 0,
        "eval_masker_is_inert": bool(torch.equal(logits, clean_logits)),
        "clean_output_layout_equal": list(output_layout.seq_lens)
        == list(clean_output_layout.seq_lens),
        "train_rows": train_rows == 16_035,
        "validation_rows": validation_rows == 900,
        "rank_map_rows": rank_map.height == 900,
        "rank_map_world8": sorted(rank_map["rank"].unique().to_list())
        == list(range(8)),
        "fastest_view_ctc_alignment": ctc_alignment["infeasible_rows"] == 0
        and ctc_alignment["minimum_ctc_frame_margin"] >= 0,
    }
    record = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": "cpu",
        "experiment": str(experiment),
        "consistency": {
            "cr_max_weight": consistency.cr_max_weight,
            "cr_warmup_steps": consistency.cr_warmup_steps,
            "temporal_mask_prob": consistency.temporal_mask_prob,
            "effective_mask_fraction_probe": effective_mask_fraction,
            "training_only_parameter_key": "masker.temporal_mask_embed",
            "two_view_cr_loss_sum": float(cr_probe.loss),
            "two_view_ctc_loss_sum": [float(ctc_a), float(ctc_b)],
            "two_view_valid_frames": cr_probe.valid_frames,
            "two_view_output_lengths": [
                int(value) for value in output_layout_a.seq_lens
            ],
        },
        "parent_model_parameters": parent_parameters,
        "model_parameters": parameters,
        "training_parameter_delta": parameters - parent_parameters,
        "training_only_state_keys": sorted(training_state_keys - parent_state_keys),
        "vocabulary_size": tokenizer.vocab_info.size,
        "logits_shape": list(logits.shape),
        "output_lengths": [int(value) for value in output_layout.seq_lens],
        "fastest_view_ctc_alignment": ctc_alignment,
        "checks": checks,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
