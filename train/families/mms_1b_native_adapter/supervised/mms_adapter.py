#!/usr/bin/env python3
"""Exact MMS-1B native-adapter/head warm start and trainable partition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch


SPECIAL_TOKEN_ALIASES = {
    "[PAD]": "<pad>",
    "[UNK]": "<unk>",
    "<pad>": "<pad>",
    "<unk>": "<unk>",
}


def source_vocabulary(path: Path, language: str) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(language)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"missing MMS {language!r} vocabulary in {path}")
    vocabulary = {str(token): int(index) for token, index in value.items()}
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("MMS source vocabulary IDs are not contiguous")
    return vocabulary


def target_vocabulary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid target vocabulary: {path}")
    vocabulary = {str(token): int(index) for token, index in payload.items()}
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("target vocabulary IDs are not contiguous")
    if (
        vocabulary.get("<pad>") != 0
        or vocabulary.get("<unk>") != 1
        or vocabulary.get("|") != 2
    ):
        raise ValueError("target blank/unknown/delimiter IDs must be 0/1/2")
    return vocabulary


def build_head_overlap_mapping(
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    mapping = []
    for target_token, target_id in sorted(
        target_vocab.items(), key=lambda item: item[1]
    ):
        source_token = SPECIAL_TOKEN_ALIASES.get(target_token, target_token)
        if source_token in source_vocab:
            mapping.append(
                {
                    "target_token": target_token,
                    "target_id": int(target_id),
                    "source_token": source_token,
                    "source_id": int(source_vocab[source_token]),
                }
            )
    target_ids = [int(item["target_id"]) for item in mapping]
    source_ids = [int(item["source_id"]) for item in mapping]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("MMS head mapping repeats a target row")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("MMS head mapping repeats a source row")
    return mapping


def mapping_sha256(mapping: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def inspect_head_overlap(
    *,
    adapter_path: Path,
    source_vocab_path: Path,
    target_vocab_path: Path,
    language: str,
) -> dict[str, Any]:
    source_vocab = source_vocabulary(source_vocab_path, language)
    target_vocab = target_vocabulary(target_vocab_path)
    mapping = build_head_overlap_mapping(source_vocab, target_vocab)
    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    adapter_keys = [key for key in keys if "adapter_layer" in key]
    if len(adapter_keys) != 288 or set(keys) - set(adapter_keys) != {
        "lm_head.bias",
        "lm_head.weight",
    }:
        raise ValueError("native MMS adapter tensor inventory drift")
    if shapes["lm_head.weight"] != [len(source_vocab), 1280]:
        raise ValueError("native MMS source head weight shape drift")
    if shapes["lm_head.bias"] != [len(source_vocab)]:
        raise ValueError("native MMS source head bias shape drift")
    mapped_target_ids = sorted(int(item["target_id"]) for item in mapping)
    mapped_set = set(mapped_target_ids)
    fresh_tokens = [
        token
        for token, index in sorted(target_vocab.items(), key=lambda item: item[1])
        if int(index) not in mapped_set
    ]
    adapter_parameters = sum(
        int(torch.tensor(shapes[key]).prod().item()) for key in adapter_keys
    )
    return {
        "schema_version": 1,
        "language": language,
        "adapter_tensor_count": len(adapter_keys),
        "adapter_parameters": adapter_parameters,
        "source_head_rows": len(source_vocab),
        "target_head_rows": len(target_vocab),
        "mapped_head_rows": len(mapping),
        "fresh_head_rows": len(target_vocab) - len(mapping),
        "mapping_sha256": mapping_sha256(mapping),
        "mapped_target_ids": mapped_target_ids,
        "fresh_tokens": fresh_tokens,
        "mapping": mapping,
    }


def _adapter_projection_state(
    path: Path,
    *,
    require_head: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = load_file(str(path), device="cpu")
    adapters = {key: value for key, value in state.items() if "adapter_layer" in key}
    head = {key: value for key, value in state.items() if key.startswith("lm_head.")}
    extras = set(state) - set(adapters) - set(head)
    valid_head = set(head) == {"lm_head.weight", "lm_head.bias"}
    if (
        len(adapters) != 288
        or extras
        or (require_head and not valid_head)
        or (head and not valid_head)
    ):
        raise ValueError("MMS adapter initialization inventory drift")
    return adapters, head


def _apply_head_overlap(
    model: torch.nn.Module,
    source_head: dict[str, torch.Tensor],
    report: dict[str, Any],
) -> None:
    head = getattr(model, "lm_head", None)
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("E04 requires a one-layer linear CTC head")
    source_weight = source_head["lm_head.weight"]
    source_bias = source_head["lm_head.bias"]
    if list(head.weight.shape) != [int(report["target_head_rows"]), 1280]:
        raise ValueError("runtime target head weight shape drift")
    if list(head.bias.shape) != [int(report["target_head_rows"])]:
        raise ValueError("runtime target head bias shape drift")
    if list(source_weight.shape) != [int(report["source_head_rows"]), 1280]:
        raise ValueError("source head weight shape drift")
    if list(source_bias.shape) != [int(report["source_head_rows"])]:
        raise ValueError("source head bias shape drift")
    with torch.no_grad():
        for item in report["mapping"]:
            source_id = int(item["source_id"])
            target_id = int(item["target_id"])
            head.weight[target_id].copy_(source_weight[source_id].to(head.weight))
            head.bias[target_id].copy_(source_bias[source_id].to(head.bias))


def initialize_native_adapter_specialist(
    model: torch.nn.Module,
    *,
    adapter_path: Path,
    native_package_path: Path | None = None,
    source_vocab_path: Path,
    target_vocab_path: Path,
    language: str,
    head_init_path: Path | None = None,
    expected: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Warm-start all native adapters/head overlap, freeze base, and anchor L2-SP."""

    for attribute in ("init_adapter_layers", "freeze_base_model"):
        if not hasattr(model, attribute):
            raise ValueError(f"MMS model lacks required method: {attribute}")
    native_package_path = adapter_path if native_package_path is None else native_package_path
    report = inspect_head_overlap(
        adapter_path=native_package_path,
        source_vocab_path=source_vocab_path,
        target_vocab_path=target_vocab_path,
        language=language,
    )
    if expected:
        checks = {
            "source_head_rows": "expected_source_head_rows",
            "target_head_rows": "expected_target_head_rows",
            "mapped_head_rows": "expected_mapped_head_rows",
            "fresh_head_rows": "expected_fresh_head_rows",
            "mapping_sha256": "expected_mapping_sha256",
            "adapter_tensor_count": "expected_adapter_tensors",
            "adapter_parameters": "expected_adapter_parameters",
        }
        for observed_key, expected_key in checks.items():
            if report[observed_key] != expected[expected_key]:
                raise RuntimeError(
                    f"native warm-start contract drift: {observed_key}="
                    f"{report[observed_key]!r} expected={expected[expected_key]!r}"
                )

    model.init_adapter_layers()
    adapters, _ = _adapter_projection_state(adapter_path, require_head=False)
    _, source_head = _adapter_projection_state(native_package_path, require_head=True)
    model_adapter_names = {
        name for name, _ in model.named_parameters() if "adapter_layer" in name
    }
    if set(adapters) != model_adapter_names:
        raise ValueError(
            "native adapter/model key mismatch: "
            f"source={len(adapters)} model={len(model_adapter_names)} "
            f"missing={sorted(model_adapter_names - set(adapters))[:3]} "
            f"extra={sorted(set(adapters) - model_adapter_names)[:3]}"
        )
    model.load_state_dict(adapters, strict=False)
    if head_init_path is None:
        _apply_head_overlap(model, source_head, report)
    else:
        head_state = load_file(str(head_init_path), device="cpu")
        if set(head_state) != {"lm_head.weight", "lm_head.bias"}:
            raise ValueError("frozen CTC head-init inventory drift")
        head = getattr(model, "lm_head", None)
        if not isinstance(head, torch.nn.Linear):
            raise ValueError("MMS requires a one-layer linear CTC head")
        expected_shapes = {
            "lm_head.weight": tuple(head.weight.shape),
            "lm_head.bias": tuple(head.bias.shape),
        }
        if any(
            tuple(head_state[name].shape) != expected_shapes[name]
            for name in expected_shapes
        ):
            raise ValueError("frozen CTC head-init shape drift")
        with torch.no_grad():
            head.weight.copy_(head_state["lm_head.weight"].to(head.weight))
            head.bias.copy_(head_state["lm_head.bias"].to(head.bias))
        report["head_init_path"] = str(head_init_path)
        report["head_init_sha256"] = hashlib.sha256(
            head_init_path.read_bytes()
        ).hexdigest()

    model.freeze_base_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "adapter_layer" in name or name.startswith("lm_head."):
            parameter.requires_grad = True

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    adapter_trainable = {
        name: parameter
        for name, parameter in trainable.items()
        if "adapter_layer" in name
    }
    head_trainable = {
        name: parameter
        for name, parameter in trainable.items()
        if name.startswith("lm_head.")
    }
    if set(adapter_trainable) != model_adapter_names:
        raise RuntimeError("not all and only native adapter tensors are trainable")
    if set(head_trainable) != {"lm_head.weight", "lm_head.bias"}:
        raise RuntimeError("linear CTC head partition drift")
    adapter_parameters = sum(value.numel() for value in adapter_trainable.values())
    head_parameters = sum(value.numel() for value in head_trainable.values())
    trainable_parameters = adapter_parameters + head_parameters
    total_parameters = sum(value.numel() for value in model.parameters())
    if expected and (
        adapter_parameters != int(expected["expected_adapter_parameters"])
        or head_parameters != int(expected["expected_head_parameters"])
        or trainable_parameters != int(expected["expected_trainable_parameters"])
    ):
        raise RuntimeError(
            "MMS trainable partition count drift: "
            f"adapter={adapter_parameters} head={head_parameters} "
            f"total={trainable_parameters}"
        )
    if not (0 < trainable_parameters < total_parameters):
        raise RuntimeError("invalid MMS adapter-only partition")

    reference = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in adapter_trainable.items()
    }
    if len(reference) != 288:
        raise RuntimeError("L2-SP reference tensor-count drift")
    report = {
        **report,
        "adapter_parameters": adapter_parameters,
        "head_parameters": head_parameters,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_adapter_names": sorted(adapter_trainable),
        "trainable_head_names": sorted(head_trainable),
        "l2sp_reference_tensors": len(reference),
    }
    return report, reference


def adapter_l2sp_penalty(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    device_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the unnormalized FP32 L2 distance from native adapter initialization."""

    cache = {} if device_cache is None else device_cache
    named = dict(model.named_parameters())
    if set(reference) - set(named):
        raise ValueError("an MMS L2-SP reference parameter disappeared")
    penalty: torch.Tensor | None = None
    for name, initial_cpu in reference.items():
        parameter = named[name]
        initial = cache.get(name)
        if (
            initial is None
            or initial.device != parameter.device
            or initial.dtype != torch.float32
        ):
            initial = initial_cpu.to(device=parameter.device, dtype=torch.float32)
            cache[name] = initial
        term = (parameter.float() - initial).square().sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        raise ValueError("MMS L2-SP reference is empty")
    return penalty


def two_group_optimizer_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    adapters: list[torch.nn.Parameter] = []
    head: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("duplicate trainable parameter object")
        seen.add(id(parameter))
        if "adapter_layer" in name:
            adapters.append(parameter)
        elif name.startswith("lm_head."):
            head.append(parameter)
        else:
            raise ValueError(f"unexpected trainable MMS parameter: {name}")
    if len(adapters) != 288 or len(head) != 2:
        raise ValueError(
            f"MMS optimizer group tensor-count drift: adapters={len(adapters)} "
            f"head={len(head)}"
        )
    return adapters, head


def parameter_drift(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    named = dict(model.named_parameters())
    per_tensor = []
    squared_total = 0.0
    initial_squared_total = 0.0
    for name in sorted(reference):
        current = named[name].detach().float().cpu()
        initial = reference[name]
        squared = float((current - initial).square().sum().item())
        initial_squared = float(initial.square().sum().item())
        squared_total += squared
        initial_squared_total += initial_squared
        per_tensor.append(
            {
                "name": name,
                "l2": squared**0.5,
                "relative_l2": (
                    squared**0.5 / initial_squared**0.5
                    if initial_squared > 0.0
                    else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "tensors": len(per_tensor),
        "l2": squared_total**0.5,
        "relative_l2": (
            squared_total**0.5 / initial_squared_total**0.5
            if initial_squared_total > 0.0
            else None
        ),
        "maximum_tensor_l2": max(item["l2"] for item in per_tensor),
        "per_tensor": per_tensor,
    }


def export_native_specialist(
    model: torch.nn.Module,
    path: Path,
    *,
    metadata: dict[str, str],
) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"create-only adapter export exists: {path}")
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if "adapter_layer" in name or name.startswith("lm_head.")
    }
    if len([key for key in state if "adapter_layer" in key]) != 288 or set(
        key for key in state if key.startswith("lm_head.")
    ) != {"lm_head.weight", "lm_head.bias"}:
        raise RuntimeError("adapter export tensor inventory drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={str(k): str(v) for k, v in metadata.items()})
    reloaded = load_file(str(path), device="cpu")
    if set(reloaded) != set(state):
        raise RuntimeError("adapter export reload inventory drift")
    for name, value in state.items():
        if not torch.equal(value, reloaded[name]):
            raise RuntimeError(f"adapter export reload value drift: {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "tensor_count": len(state),
        "adapter_tensor_count": 288,
        "head_tensor_count": 2,
    }
