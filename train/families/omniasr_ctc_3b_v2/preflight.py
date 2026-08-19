#!/usr/bin/env python3
"""CPU-only numerical closure for the untouched OmniASR CTC-3B-v2 parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import polars as pl
import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from fairseq2.nn import BatchLayout

from runtime_assets import render_asset_cards, resolve_repo_root, verify_anchor
from runtime_config import runtime_geometry_from_experiment


EXPECTED_VOCABULARY = 10_288
EXPECTED_PARAMETERS = 3_081_398_960


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    torch.set_num_interop_threads(4)
    root = resolve_repo_root()
    geometry = runtime_geometry_from_experiment(args.experiment, root)
    parent = root / "models/omniasr-ctc-3b-v2/omniASR-CTC-3B-v2.pt"
    tokenizer_path = (
        root
        / "models/omniasr-ctc-3b-v2/omniASR_tokenizer_written_v2.model"
    )
    if not parent.is_file():
        raise FileNotFoundError(
            "official CTC-3B-v2 asset is not locally verified"
        )
    if parent.stat().st_size != 12_325_920_624:
        raise RuntimeError("CTC-3B-v2 official asset byte-count drift")
    verify_anchor(
        tokenizer_path,
        "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
        91_481,
    )
    with tempfile.TemporaryDirectory(prefix="waxal3-ctc3b-cards-") as temporary:
        render_asset_cards(
            Path(__file__).resolve().parent / "cards/waxal3.yaml.template",
            Path(temporary),
        )
        import omnilingual_asr  # noqa: F401

        tokenizer = load_tokenizer("waxal3_omni_tokenizer_written_v2")
        model = load_model(
            "waxal3_omni_ctc_3b_v2_target_es_parent",
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
            progress=False,
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        model.eval()
        samples = torch.linspace(-0.1, 0.1, 16_000).unsqueeze(0)
        layout = BatchLayout.of(samples, [16_000])
        with torch.inference_mode():
            logits, output_layout = model(samples, layout)
    manifest_dir = geometry.manifest_dir
    train_rows = pl.read_parquet(manifest_dir / "train.rows.parquet").height
    validation_rows = pl.read_parquet(manifest_dir / "dev.rows.parquet").height
    rank_map = pl.read_csv(manifest_dir / "dev.rank_map.world8.csv")
    checks = {
        "exact_parameter_count": parameters == EXPECTED_PARAMETERS,
        "vocabulary": tokenizer.vocab_info.size == EXPECTED_VOCABULARY,
        "finite_logits": bool(torch.isfinite(logits).all()),
        "logit_vocabulary": int(logits.shape[-1]) == EXPECTED_VOCABULARY,
        "output_length_positive": int(output_layout.seq_lens[0]) > 0,
        "train_rows": train_rows == geometry.expected_train_rows,
        "validation_rows": validation_rows == 900,
        "rank_map_rows": rank_map.height == 900,
        "rank_map_world8": sorted(rank_map["rank"].unique().to_list())
        == list(range(8)),
    }
    record = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": "cpu",
        "experiment": str(args.experiment),
        "language": geometry.language,
        "manifest_dir": str(manifest_dir),
        "expected_train_rows": geometry.expected_train_rows,
        "updates_per_epoch": geometry.updates_per_epoch,
        "model_parameters": parameters,
        "vocabulary_size": tokenizer.vocab_info.size,
        "logits_shape": list(logits.shape),
        "output_lengths": [int(value) for value in output_layout.seq_lens],
        "checks": checks,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
