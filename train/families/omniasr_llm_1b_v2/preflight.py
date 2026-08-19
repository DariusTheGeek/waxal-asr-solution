#!/usr/bin/env python3
"""CPU-only numerical closure for the official OmniASR LLM-1B-v2 parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import polars as pl
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.models.family import ModelFamily
from fairseq2.models.hub import load_model
from fairseq2.runtime.dependency import get_dependency_resolver

from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import (
    WAV2VEC2_LLAMA_FAMILY,
    ModelType,
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
from omnilingual_asr.models.wav2vec2_llama.syntax import Modality

from runtime_assets import render_asset_cards, resolve_repo_root, verify_anchor
from runtime_config import runtime_geometry_from_experiment
from activation_checkpointing import install_wav2vec2_llama_layerwise_ac


EXPECTED_TOKENIZER_VOCABULARY = 10_288
EXPECTED_TARGET_VOCABULARY = 9_812
# The upstream README still advertises the v1-sized count (2,275,710,592).
# The strict-loaded v2 checkpoint has 952 additional 4096-wide language rows.
EXPECTED_PARAMETERS = 2_279_609_984
EXPECTED_LANGUAGE_ID = 820
EXPECTED_LANGUAGE_EMBEDDINGS = 1_694


def audit_transcripts(
    manifest_dir: Path,
    token_encoder,
    unknown_idx: int,
    *,
    expected_train_rows: int = 16_035,
) -> dict:
    splits: dict[str, object] = {}
    total_unknown = 0
    for split, expected_rows in (("train", expected_train_rows), ("dev", 900)):
        rows = (manifest_dir / f"{split}.wrd").read_text(
            encoding="utf-8"
        ).splitlines()
        unknown = 0
        empty = 0
        maximum_tokens = 0
        for text in rows:
            tokens = token_encoder(text)
            empty += int(tokens.numel() == 0)
            unknown += int((tokens == unknown_idx).sum().item())
            maximum_tokens = max(maximum_tokens, int(tokens.numel()))
        total_unknown += unknown
        splits[split] = {
            "rows": len(rows),
            "expected_rows": expected_rows,
            "empty_targets": empty,
            "unknown_tokens": unknown,
            "maximum_tokens": maximum_tokens,
        }
    return {"splits": splits, "total_unknown_tokens": total_unknown}


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
    parent = root / "models/omniasr-llm-1b-v2/omniASR-LLM-1B-v2.pt"
    tokenizer_path = (
        root
        / "models/omniasr-llm-1b-v2/omniASR_tokenizer_written_v2.model"
    )
    verify_anchor(
        parent,
        "cceb4d9ebac3d168a6af6b26c62ce11bafc562b38976c6bfa87e7d60422c6da5",
        9_118_733_852,
    )
    verify_anchor(
        tokenizer_path,
        "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
        91_481,
    )

    activation_checkpoint_repair = install_wav2vec2_llama_layerwise_ac()

    with tempfile.TemporaryDirectory(prefix="waxal3-llm1b-cards-") as temporary:
        render_asset_cards(
            Path(__file__).resolve().parent / "cards/waxal3.yaml.template",
            Path(temporary),
        )
        import omnilingual_asr  # noqa: F401

        tokenizer = load_tokenizer("waxal3_omni_tokenizer_written_v2")
        model = load_model(
            "waxal3_omni_llm_1b_v2_target_es_parent",
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
            progress=False,
        )
        if not isinstance(model, Wav2Vec2LlamaModel):
            raise RuntimeError(f"unexpected parent model type: {type(model)}")
        family = get_dependency_resolver().resolve(
            ModelFamily, key=WAV2VEC2_LLAMA_FAMILY
        )
        if not family.supports_layerwise_ac:
            raise RuntimeError(
                "Wav2Vec2-Llama family still lacks layerwise activation checkpointing"
            )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        token_encoder = tokenizer.create_encoder()
        unk_idx = tokenizer.vocab_info.unk_idx
        if unk_idx is None:
            raise RuntimeError("tokenizer has no unknown-token index")
        transcript_audit = audit_transcripts(
            geometry.manifest_dir,
            token_encoder,
            int(unk_idx),
            expected_train_rows=geometry.expected_train_rows,
        )

        probe_word = {"lin": "lingala", "sna": "shona"}.get(
            geometry.language, geometry.language
        )
        target_tokens = token_encoder(probe_word)
        if target_tokens.numel() == 0:
            raise RuntimeError("probe target tokenization is empty")
        samples = torch.linspace(-0.1, 0.1, 3_200).unsqueeze(0)
        batch = Seq2SeqBatch(
            source_seqs=samples,
            source_seq_lens=[samples.shape[1]],
            target_seqs=target_tokens.unsqueeze(0),
            target_seq_lens=[int(target_tokens.numel())],
            example={"lang": [geometry.language_code]},
        )
        model.eval()
        syntax = model.create_default_syntax(batch, torch.device("cpu"))
        language_inputs = [item for item in syntax if item.modality == Modality.LANG]
        with torch.inference_mode():
            (
                loss,
                logits,
                logits_layout,
                decoder_context,
                decoder_context_lens,
                audio_embeddings,
            ) = model(batch, return_logits=True)
            original_max_length = model.max_generation_length
            model.max_generation_length = max(decoder_context_lens[0]) + 8
            try:
                generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
                    model=model,
                    config=Wav2Vec2LlamaBeamSearchConfig(
                        nbest=1, length_norm=False
                    ),
                    streaming_config=model.streaming_config,
                )
                beam_tokens, beam_lengths = generator.generate_hypotheses(
                    decoder_context_inputs=decoder_context,
                    decoder_context_seq_lens=decoder_context_lens,
                    audio_embeddings=audio_embeddings,
                    batch=batch,
                )
            finally:
                model.max_generation_length = original_max_length
        family.apply_layerwise_ac(model, every_nth_layer=1)
        encoder_ac_wrappers = sum(
            isinstance(layer, CheckpointWrapper) for layer in model.encoder.layers
        )
        decoder_ac_wrappers = sum(
            isinstance(layer, CheckpointWrapper)
            for layer in model.llama_decoder.layers
        )

    manifest_dir = geometry.manifest_dir
    train_rows = pl.read_parquet(manifest_dir / "train.rows.parquet").height
    validation_rows = pl.read_parquet(manifest_dir / "dev.rows.parquet").height
    rank_map = pl.read_csv(manifest_dir / "dev.rank_map.world8.csv")
    language_id = (
        None
        if model.lang_mapping is None
        else model.lang_mapping.get(geometry.language_code.casefold())
    )
    language_input_id = (
        None
        if len(language_inputs) != 1
        else int(language_inputs[0].seqs[0, 0])
    )
    expected_language_id = (
        geometry.expected_model_language_id
        if geometry.expected_model_language_id is not None
        else EXPECTED_LANGUAGE_ID
    )
    checks = {
        "llm_model_type": model.model_type == ModelType.LLM_ASR_LID,
        "exact_parameter_count": parameters == EXPECTED_PARAMETERS,
        "tokenizer_vocabulary": tokenizer.vocab_info.size
        == EXPECTED_TOKENIZER_VOCABULARY,
        "target_vocabulary": model.target_vocab_info.size
        == EXPECTED_TARGET_VOCABULARY,
        "final_projection_vocabulary": int(model.final_proj.weight.shape[0])
        == EXPECTED_TOKENIZER_VOCABULARY,
        "text_embedding_rows": int(model.text_frontend.weight.shape[0])
        == EXPECTED_TOKENIZER_VOCABULARY + 1,
        "language_embedding_rows": model.lang_embeddings is not None
        and int(model.lang_embeddings.weight.shape[0])
        == EXPECTED_LANGUAGE_EMBEDDINGS,
        "language_dropout": float(model.lang_embeddings_p) == 0.5,
        "language_mapping": language_id == expected_language_id,
        "language_batch_injection": language_input_id == expected_language_id,
        "lid_marker": model.special_tokens.lid_marker
        == EXPECTED_TARGET_VOCABULARY,
        "default_beam": int(model.beam_search_config.nbest) == 5,
        "default_no_length_normalization": model.beam_search_config.length_norm
        is False,
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "logit_vocabulary": int(logits.shape[-1])
        == EXPECTED_TOKENIZER_VOCABULARY,
        "output_length_positive": int(logits_layout.seq_lens[0]) > 0,
        "bounded_beam_probe": beam_tokens.ndim == 2
        and len(beam_lengths) == 1
        and int(beam_lengths[0]) >= 0,
        "model_family_supports_layerwise_ac": family.supports_layerwise_ac,
        "encoder_activation_checkpoint_wrappers": encoder_ac_wrappers == 48,
        "decoder_activation_checkpoint_wrappers": decoder_ac_wrappers == 12,
        "transcript_rows": all(
            int(item["rows"]) == int(item["expected_rows"])
            for item in transcript_audit["splits"].values()
        ),
        "no_empty_targets": all(
            int(item["empty_targets"]) == 0
            for item in transcript_audit["splits"].values()
        ),
        "no_unknown_target_tokens": transcript_audit["total_unknown_tokens"]
        == 0,
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
        "language_code": geometry.language_code,
        "language_id": language_id,
        "expected_language_id": expected_language_id,
        "manifest_dir": str(manifest_dir),
        "expected_train_rows": geometry.expected_train_rows,
        "updates_per_epoch": geometry.updates_per_epoch,
        "model_parameters": parameters,
        "tokenizer_vocabulary_size": tokenizer.vocab_info.size,
        "target_vocabulary_size": model.target_vocab_info.size,
        "logits_shape": list(logits.shape),
        "output_lengths": [int(value) for value in logits_layout.seq_lens],
        "probe_loss": float(loss),
        "beam_probe_shape": list(beam_tokens.shape),
        "beam_probe_lengths": [int(value) for value in beam_lengths],
        "activation_checkpointing": {
            **activation_checkpoint_repair,
            "encoder_wrappers": encoder_ac_wrappers,
            "decoder_wrappers": decoder_ac_wrappers,
            "every_nth_layer": 1,
        },
        "transcript_audit": transcript_audit,
        "checks": checks,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
