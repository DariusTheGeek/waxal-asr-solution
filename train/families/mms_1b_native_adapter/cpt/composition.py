#!/usr/bin/env python3
"""Audited MMS-1B-ASR backbone + genuine SSL-head + native adapter composition."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining


SSL_OBJECTIVE_KEYS = {
    "quantizer.codevectors",
    "quantizer.weight_proj.weight",
    "quantizer.weight_proj.bias",
    "project_hid.weight",
    "project_hid.bias",
    "project_q.weight",
    "project_q.bias",
}
ASR_WEIGHT_NORM_KEY_MAP = {
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g":
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v":
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}
NATIVE_ADAPTER_TENSORS = 288
NATIVE_ADAPTER_PARAMETERS = 2_151_168


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_parameter_names(model: torch.nn.Module) -> list[str]:
    return sorted(
        name for name, _ in model.named_parameters() if "adapter_layer" in name
    )


def _copy_tensor(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    name: str,
) -> None:
    if target.shape != source.shape or target.dtype != source.dtype:
        raise RuntimeError(
            f"tensor contract mismatch for {name}: "
            f"target={tuple(target.shape)}/{target.dtype} "
            f"source={tuple(source.shape)}/{source.dtype}"
        )
    with torch.no_grad():
        target.copy_(source)


def _load_safetensor_subset(
    path: Path,
    target_state: dict[str, torch.Tensor],
    *,
    include: Iterable[str],
) -> list[str]:
    requested = sorted(set(include))
    with safe_open(path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted(set(requested) - available)
        if missing:
            raise RuntimeError(f"missing tensors in {path}: {missing[:10]}")
        for name in requested:
            if name not in target_state:
                raise RuntimeError(f"composition target lacks tensor: {name}")
            _copy_tensor(target_state[name], handle.get_tensor(name), name=name)
    return requested


def compose_native_adapter_pretraining_model(
    *,
    asr_base: Path,
    ssl_base: Path,
    native_adapter: Path,
    mask_time_prob: float,
    mask_time_length: int,
    mask_time_min_masks: int,
    num_negatives: int,
    layerdrop: float,
) -> tuple[Wav2Vec2ForPreTraining, dict[str, Any]]:
    """Use the ASR-trained shared backbone and only the genuine SSL head tensors.

    The common MMS-1B and MMS-1B-all backbone tensors are known to differ. This
    function therefore never lets `from_pretrained` silently choose a parent:
    every target tensor is filled explicitly and checked by name/shape/dtype.
    """

    asr_base = asr_base.resolve()
    ssl_base = ssl_base.resolve()
    native_adapter = native_adapter.resolve()
    asr_weights = asr_base / "model.safetensors"
    ssl_weights = ssl_base / "pytorch_model.bin"
    for path in (asr_weights, ssl_weights, native_adapter):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = Wav2Vec2Config.from_pretrained(asr_base, local_files_only=True)
    config.architectures = ["Wav2Vec2ForPreTraining"]
    config.mask_time_prob = float(mask_time_prob)
    config.mask_time_length = int(mask_time_length)
    config.mask_time_min_masks = int(mask_time_min_masks)
    config.num_negatives = int(num_negatives)
    config.layerdrop = float(layerdrop)
    if config.adapter_attn_dim != 16 or config.num_hidden_layers != 48:
        raise RuntimeError(
            "unexpected MMS-1B native-adapter architecture: "
            f"adapter_attn_dim={config.adapter_attn_dim} "
            f"layers={config.num_hidden_layers}"
        )

    model = Wav2Vec2ForPreTraining(config)
    target_state = model.state_dict()

    with safe_open(asr_weights, framework="pt", device="cpu") as handle:
        asr_keys = set(handle.keys())
    asr_backbone_keys = sorted(asr_keys - {"lm_head.weight", "lm_head.bias"})
    mapped_asr_backbone_keys = sorted(
        ASR_WEIGHT_NORM_KEY_MAP.get(name, name) for name in asr_backbone_keys
    )
    expected_backbone = sorted(set(target_state) - SSL_OBJECTIVE_KEYS)
    if mapped_asr_backbone_keys != expected_backbone:
        raise RuntimeError(
            "ASR backbone/model key contract mismatch: "
            f"missing={sorted(set(expected_backbone) - set(mapped_asr_backbone_keys))[:10]} "
            f"extra={sorted(set(mapped_asr_backbone_keys) - set(expected_backbone))[:10]}"
        )
    with safe_open(asr_weights, framework="pt", device="cpu") as handle:
        for source_name in asr_backbone_keys:
            target_name = ASR_WEIGHT_NORM_KEY_MAP.get(source_name, source_name)
            _copy_tensor(
                target_state[target_name],
                handle.get_tensor(source_name),
                name=f"{source_name}->{target_name}",
            )

    ssl_state = torch.load(
        ssl_weights,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not SSL_OBJECTIVE_KEYS.issubset(ssl_state):
        raise RuntimeError(
            f"genuine SSL objective tensors missing: "
            f"{sorted(SSL_OBJECTIVE_KEYS - set(ssl_state))}"
        )
    for name in sorted(SSL_OBJECTIVE_KEYS):
        _copy_tensor(target_state[name], ssl_state[name], name=name)
    del ssl_state

    with safe_open(native_adapter, framework="pt", device="cpu") as handle:
        adapter_keys = sorted(name for name in handle.keys() if "adapter_layer" in name)
        non_adapter = sorted(name for name in handle.keys() if "adapter_layer" not in name)
    if len(adapter_keys) != NATIVE_ADAPTER_TENSORS or non_adapter != [
        "lm_head.bias",
        "lm_head.weight",
    ]:
        raise RuntimeError(
            "native adapter package contract mismatch: "
            f"adapters={len(adapter_keys)} non_adapter={non_adapter}"
        )
    if adapter_keys != adapter_parameter_names(model):
        raise RuntimeError("native adapter/model tensor names do not match")
    _load_safetensor_subset(native_adapter, target_state, include=adapter_keys)
    adapter_parameters = sum(target_state[name].numel() for name in adapter_keys)
    if adapter_parameters != NATIVE_ADAPTER_PARAMETERS:
        raise RuntimeError(
            f"native adapter parameter drift: {adapter_parameters}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "adapter_layer" in name:
            parameter.requires_grad = True
    trainable = sorted(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != adapter_keys:
        raise RuntimeError("adapter-only trainable partition mismatch")

    report = {
        "schema_version": 1,
        "asr_backbone_tensors": len(asr_backbone_keys),
        "ssl_objective_tensors": len(SSL_OBJECTIVE_KEYS),
        "native_adapter_tensors": len(adapter_keys),
        "native_adapter_parameters": adapter_parameters,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_names": trainable,
        "composition": {
            "backbone": str(asr_weights),
            "ssl_objective": str(ssl_weights),
            "native_adapter": str(native_adapter),
        },
        "config": {
            "mask_time_prob": float(config.mask_time_prob),
            "mask_time_length": int(config.mask_time_length),
            "mask_time_min_masks": int(config.mask_time_min_masks),
            "num_negatives": int(config.num_negatives),
            "layerdrop": float(config.layerdrop),
        },
    }
    return model, report


def adapter_reference(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    reference = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(reference) != NATIVE_ADAPTER_TENSORS:
        raise RuntimeError("adapter reference tensor count drift")
    return reference


def adapter_l2sp_penalty(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    device_cache = {} if cache is None else cache
    named = dict(model.named_parameters())
    if set(reference) != {name for name, value in named.items() if value.requires_grad}:
        raise RuntimeError("L2-SP reference/trainable partition drift")
    penalty: torch.Tensor | None = None
    for name in sorted(reference):
        parameter = named[name]
        initial = device_cache.get(name)
        if initial is None or initial.device != parameter.device:
            initial = reference[name].to(parameter.device, dtype=torch.float32)
            device_cache[name] = initial
        term = (parameter.float() - initial).square().sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        raise RuntimeError("empty L2-SP partition")
    return penalty


def export_adapter_only(
    model: torch.nn.Module,
    path: Path,
    *,
    metadata: dict[str, str],
) -> dict[str, Any]:
    path = path.resolve()
    if path.exists():
        raise RuntimeError(f"create-only adapter export exists: {path}")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.named_parameters()
        if "adapter_layer" in name
    }
    if len(state) != NATIVE_ADAPTER_TENSORS:
        raise RuntimeError("adapter-only export tensor count drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={str(k): str(v) for k, v in metadata.items()})
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensors": len(state),
        "parameters": sum(value.numel() for value in state.values()),
    }
