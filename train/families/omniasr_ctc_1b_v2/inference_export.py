#!/usr/bin/env python3
"""Export a CR-CTC checkpoint into the untouched clean-inference topology."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import torch
from torch import Tensor

from checkpoint_average import extract_model_state, tensor_content_sha256
from consistency import (
    ConsistencyConfig,
    TRAINING_ONLY_MASK_KEY,
    attach_training_masker,
    strip_training_only_masker,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> Mapping[str, Tensor]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return extract_model_state(checkpoint)


def _forward(
    state: Mapping[str, Tensor],
    *,
    card_dir: Path,
    model_card: str,
    attach_masker: bool,
    consistency: ConsistencyConfig,
) -> tuple[Tensor, list[int]]:
    os.environ["FAIRSEQ2_ASSET_DIR"] = str(card_dir.resolve())
    import omnilingual_asr  # noqa: F401
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout

    model = load_model(
        model_card,
        device=torch.device("cpu"),
        dtype=torch.float32,
        mmap=True,
        progress=False,
    )
    if attach_masker:
        masker = attach_training_masker(model, consistency)
        mask_tensor = state.get(TRAINING_ONLY_MASK_KEY)
        if (
            mask_tensor is None
            or mask_tensor.numel() != masker.temporal_mask_embed.numel()
        ):
            raise RuntimeError(
                "checkpoint mask embedding does not match the parent width"
            )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    model.eval()
    samples = torch.linspace(-0.1, 0.1, 16_000, dtype=torch.float32).unsqueeze(0)
    layout = BatchLayout.of(samples, [16_000])
    with torch.inference_mode():
        logits, output_layout = model(samples, layout)
    if logits.shape[:1] != (1,) or logits.shape[-1] != 10_288:
        raise RuntimeError(f"unexpected parity-forward shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite parity-forward logits")
    return logits.detach().clone(), [int(value) for value in output_layout.seq_lens]


def export_inference_checkpoint(
    *,
    source: Path,
    output_dir: Path,
    card_dir: Path,
    model_card: str,
    consistency: ConsistencyConfig,
) -> dict[str, object]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        full_state = load_state(source)
        inference_state = strip_training_only_masker(full_state, require_masker=True)
        full_logits, full_lengths = _forward(
            full_state,
            card_dir=card_dir,
            model_card=model_card,
            attach_masker=True,
            consistency=consistency,
        )
        inference_logits, inference_lengths = _forward(
            inference_state,
            card_dir=card_dir,
            model_card=model_card,
            attach_masker=False,
            consistency=consistency,
        )
        if full_lengths != inference_lengths or not torch.equal(
            full_logits, inference_logits
        ):
            raise RuntimeError("training-topology and clean-inference forwards differ")

        output_model = temporary / "model.pt"
        torch.save({"model": inference_state, "fs2": True}, output_model)
        reloaded = load_state(output_model)
        if list(reloaded) != list(inference_state):
            raise RuntimeError("inference export tensor ordering changed on reload")
        for name in inference_state:
            if not torch.equal(inference_state[name], reloaded[name]):
                raise RuntimeError(f"inference tensor changed on reload: {name}")
        reload_logits, reload_lengths = _forward(
            reloaded,
            card_dir=card_dir,
            model_card=model_card,
            attach_masker=False,
            consistency=consistency,
        )
        if reload_lengths != inference_lengths or not torch.equal(
            reload_logits, inference_logits
        ):
            raise RuntimeError("inference export changed after serialization")
        record: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "omniasr_ctc1b_crctc_clean_inference_export",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "removed_training_only_keys": [TRAINING_ONLY_MASK_KEY],
            "output_model": "model.pt",
            "output_model_sha256": sha256_file(output_model),
            "output_tensor_content_sha256": tensor_content_sha256(reloaded),
            "output_tensor_count": len(reloaded),
            "strict_clean_reload": True,
            "forward_parity_exact": True,
            "forward_logits_shape": list(reload_logits.shape),
            "forward_output_lengths": reload_lengths,
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
        }
        (temporary / "VERIFY.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
        return record
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (temporary / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        failed = output_dir.with_name(f"{output_dir.name}.failed.{uuid.uuid4().hex}")
        os.rename(temporary, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--card-dir", type=Path, required=True)
    parser.add_argument(
        "--model-card",
        default="waxal3_omni_ctc_1b_v2_target_es_parent",
    )
    parser.add_argument("--cr-max-weight", type=float, required=True)
    args = parser.parse_args()
    config = ConsistencyConfig(cr_max_weight=args.cr_max_weight)
    record = export_inference_checkpoint(
        source=args.source,
        output_dir=args.output_dir,
        card_dir=args.card_dir,
        model_card=args.model_card,
        consistency=config,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
