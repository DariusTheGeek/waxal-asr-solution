#!/usr/bin/env python3
"""Create-only audit of an MMS-1B native-adapter CPT composition."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from transformers import Wav2Vec2ForCTC

from .composition import (
    compose_native_adapter_pretraining_model,
    export_adapter_only,
)
from .contract import sha256_file, write_json_create_only
from .transfer import build_native_ctc_package


def _load_ctc_model(base: Path, package: Path) -> Wav2Vec2ForCTC:
    state = load_file(str(package), device="cpu")
    head_rows = int(state["lm_head.weight"].shape[0])
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        base,
        local_files_only=True,
        vocab_size=head_rows,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
        output_loading_info=True,
    )
    affected = {
        str(item[0]) if isinstance(item, (tuple, list)) else str(item.get("key"))
        for item in loading.get("mismatched_keys", [])
    } | set(loading.get("missing_keys", []))
    if affected != {"lm_head.weight", "lm_head.bias"}:
        raise RuntimeError(f"unexpected CTC loading drift: {loading}")
    model.init_adapter_layers()
    result = model.load_state_dict(state, strict=False)
    missing = [name for name in result.missing_keys if "adapter_layer" in name or name.startswith("lm_head.")]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"native package load drift: missing={missing} unexpected={unexpected}"
        )
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("lin", "sna"), default="lin")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve()
    language = str(args.language)
    if output.exists():
        raise RuntimeError(f"create-only model audit exists: {output}")
    ssl_weight = root / "models/mms-1b/pytorch_model.bin"
    asr_weight = root / "models/mms-1b-all/model.safetensors"
    adapter_name = f"adapter.{language}.safetensors"
    adapter = root / "models/mms-1b-all" / adapter_name

    ssl_state = torch.load(ssl_weight, map_location="cpu", weights_only=True, mmap=True)
    with safe_open(asr_weight, framework="pt", device="cpu") as handle:
        asr_keys = set(handle.keys())
        common = sorted(
            (set(ssl_state) & asr_keys)
            - {
                "wav2vec2.encoder.pos_conv_embed.conv.weight_g",
                "wav2vec2.encoder.pos_conv_embed.conv.weight_v",
            }
        )
        equal = 0
        mismatch_maxima: list[tuple[float, str]] = []
        for name in common:
            left = ssl_state[name]
            right = handle.get_tensor(name)
            if torch.equal(left, right):
                equal += 1
            else:
                mismatch_maxima.append(
                    (float((left.float() - right.float()).abs().max()), name)
                )
    del ssl_state

    model, composition = compose_native_adapter_pretraining_model(
        asr_base=root / "models/mms-1b-all",
        ssl_base=root / "models/mms-1b",
        native_adapter=adapter,
        mask_time_prob=0.65,
        mask_time_length=10,
        mask_time_min_masks=2,
        num_negatives=100,
        layerdrop=0.0,
    )
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    with tempfile.TemporaryDirectory(prefix="waxal3_mms_audit_") as temporary:
        temporary_root = Path(temporary)
        exported = temporary_root / adapter_name
        export_record = export_adapter_only(
            model,
            exported,
            metadata={"language": language, "dose": "zero"},
        )
        transferred = temporary_root / f"transferred.{language}.safetensors"
        transfer = build_native_ctc_package(
            cpt_adapter=exported,
            native_package=adapter,
            output=transferred,
            metadata={"language": language, "dose": "zero"},
        )
        del model
        gc.collect()

        ctc_model = _load_ctc_model(root / "models/mms-1b-all", adapter)
        generator = torch.Generator().manual_seed(42003)
        probe = torch.randn((1, 8_000), generator=generator, dtype=torch.float32)
        attention = torch.ones_like(probe, dtype=torch.long)
        with torch.inference_mode():
            native_logits = ctc_model(
                input_values=probe, attention_mask=attention
            ).logits.detach().cpu()
        transferred_state = load_file(str(transferred), device="cpu")
        ctc_model.load_state_dict(transferred_state, strict=False)
        with torch.inference_mode():
            transferred_logits = ctc_model(
                input_values=probe, attention_mask=attention
            ).logits.detach().cpu()
        logits_identical = torch.equal(native_logits, transferred_logits)
        maximum_logit_difference = float(
            (native_logits.float() - transferred_logits.float()).abs().max()
        )
        if not logits_identical or not transfer["native_package_tensor_bit_identical"]:
            raise RuntimeError("zero-dose CTC package/logit identity failed")

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "language": language,
        "implementation": {
            "model_audit_sha256": sha256_file(Path(__file__)),
            "composition_sha256": sha256_file(Path(__file__).with_name("composition.py")),
            "transfer_sha256": sha256_file(Path(__file__).with_name("transfer.py")),
        },
        "inputs": {
            "ssl_weight": "models/mms-1b/pytorch_model.bin",
            "ssl_weight_sha256": sha256_file(ssl_weight),
            "asr_weight": "models/mms-1b-all/model.safetensors",
            "asr_weight_sha256": sha256_file(asr_weight),
            "native_adapter": f"models/mms-1b-all/{adapter_name}",
            "native_adapter_sha256": sha256_file(adapter),
        },
        "shared_backbone_comparison": {
            "common_tensors_compared": len(common),
            "bit_identical_tensors": equal,
            "different_tensors": len(common) - equal,
            "largest_absolute_differences": [
                {"name": name, "maximum_absolute_difference": value}
                for value, name in sorted(mismatch_maxima, reverse=True)[:20]
            ],
            "conclusion": (
                "MMS-1B and MMS-1B-all backbones are not interchangeable; "
                "the CPT composition uses the ASR backbone and only genuine SSL tensors."
            ),
        },
        "composition": composition,
        "trainable_tensor_count": len(trainable),
        "zero_update_export": {
            key: export_record[key]
            for key in ("bytes", "sha256", "tensors", "parameters")
        },
        "zero_update_transfer": {
            key: transfer[key]
            for key in (
                "status",
                "adapter_tensors",
                "adapter_parameters",
                "head_tensors",
                "source_head_bit_identical",
                "native_package_tensor_bit_identical",
                "reload_bit_identical",
            )
        },
        "zero_update_ctc_logits": {
            "bit_identical": logits_identical,
            "maximum_absolute_difference": maximum_logit_difference,
            "probe_samples": 8_000,
            "seed": 42_003,
        },
        "transcripts_accessed": False,
        "test_labels_accessed": False,
    }
    write_json_create_only(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
