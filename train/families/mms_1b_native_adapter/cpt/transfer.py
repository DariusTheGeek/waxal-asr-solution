#!/usr/bin/env python3
"""Create a native CTC package from an adapter-only CPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .composition import NATIVE_ADAPTER_PARAMETERS, NATIVE_ADAPTER_TENSORS
from .contract import sha256_file, utc_now, write_json_create_only


HEAD_KEYS = {"lm_head.weight", "lm_head.bias"}


def _read(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    return state, metadata


def build_native_ctc_package(
    *,
    cpt_adapter: Path,
    native_package: Path,
    output: Path,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Combine CPT adapter tensors with an immutable released native CTC head."""

    cpt_adapter = cpt_adapter.resolve()
    native_package = native_package.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"create-only transfer output exists: {output}")
    cpt, _ = _read(cpt_adapter)
    native, _ = _read(native_package)
    cpt_keys = {name for name in cpt if "adapter_layer" in name}
    cpt_extras = set(cpt) - cpt_keys
    native_adapter_keys = {name for name in native if "adapter_layer" in name}
    if cpt_keys != native_adapter_keys or len(cpt_keys) != NATIVE_ADAPTER_TENSORS:
        raise RuntimeError(
            "CPT/native adapter inventory mismatch: "
            f"cpt={len(cpt_keys)} native={len(native_adapter_keys)}"
        )
    if cpt_extras not in (set(), HEAD_KEYS):
        raise RuntimeError(f"unexpected tensors in CPT input: {sorted(cpt_extras)}")
    if set(native) - native_adapter_keys != HEAD_KEYS:
        raise RuntimeError("released native package head inventory drift")
    parameters = 0
    for name in sorted(cpt_keys):
        if cpt[name].shape != native[name].shape or cpt[name].dtype != native[name].dtype:
            raise RuntimeError(f"CPT/native adapter tensor contract drift: {name}")
        parameters += cpt[name].numel()
    if parameters != NATIVE_ADAPTER_PARAMETERS:
        raise RuntimeError(f"CPT adapter parameter drift: {parameters}")
    if native["lm_head.weight"].ndim != 2 or native["lm_head.bias"].shape != (
        native["lm_head.weight"].shape[0],
    ):
        raise RuntimeError("released Myx head shape drift")

    output.parent.mkdir(parents=True, exist_ok=True)
    state = {**cpt, **{name: native[name] for name in sorted(HEAD_KEYS)}}
    transfer_metadata = {
        "schema_version": "1",
        "stage": "waxal3_mms1b_adapter_cpt_to_native_ctc_package",
        "created_at_utc": utc_now(),
        "cpt_adapter_sha256": sha256_file(cpt_adapter),
        "native_package_sha256": sha256_file(native_package),
        **({str(key): str(value) for key, value in metadata.items()} if metadata else {}),
    }
    save_file(state, str(output), metadata=transfer_metadata)
    reloaded, saved_metadata = _read(output)
    if set(reloaded) != set(state):
        raise RuntimeError("transferred package reload inventory drift")
    for name in sorted(state):
        if not torch.equal(reloaded[name], state[name]):
            raise RuntimeError(f"transferred package reload value drift: {name}")
    if saved_metadata != transfer_metadata:
        raise RuntimeError("transferred package metadata drift")
    return {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": transfer_metadata["created_at_utc"],
        "cpt_adapter": str(cpt_adapter),
        "cpt_adapter_sha256": transfer_metadata["cpt_adapter_sha256"],
        "native_package": str(native_package),
        "native_package_sha256": transfer_metadata["native_package_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "adapter_tensors": len(cpt_keys),
        "adapter_parameters": parameters,
        "head_tensors": len(HEAD_KEYS),
        "source_head_bit_identical": all(
            torch.equal(reloaded[name], native[name]) for name in HEAD_KEYS
        ),
        "native_package_tensor_bit_identical": set(state) == set(native)
        and all(torch.equal(reloaded[name], native[name]) for name in state),
        "reload_bit_identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpt-adapter", type=Path, required=True)
    parser.add_argument("--native-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = build_native_ctc_package(
        cpt_adapter=args.cpt_adapter,
        native_package=args.native_package,
        output=args.output,
    )
    write_json_create_only(args.audit.resolve(), audit)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
