"""Pinned PyTorch/Fairseq2 FSDP2 compatibility for OmniASR CTC-7B-v2."""

from __future__ import annotations

import inspect
from typing import Any


def install_fsdp2_gradient_sync_compat() -> dict[str, object]:
    """Install Fairseq2's legacy spelling as a guarded PyTorch delegation.

    Fairseq2 0.6 invokes ``set_requires_grad_sync`` while its pinned PyTorch
    2.8 runtime names the same FSDP2 operation
    ``set_requires_gradient_sync``.  The delegation is process-local and does
    not mutate site-packages on disk.
    """

    from torch.distributed.fsdp import FSDPModule

    legacy_name = "set_requires_grad_sync"
    current_name = "set_requires_gradient_sync"
    legacy = getattr(FSDPModule, legacy_name, None)
    current = getattr(FSDPModule, current_name, None)
    if legacy is not None:
        if not callable(legacy):
            raise RuntimeError(f"FSDPModule.{legacy_name} is not callable")
        return {
            "status": "PASS",
            "mode": "native_legacy_api",
            "legacy_name": legacy_name,
            "current_name": current_name,
        }
    if current is None or not callable(current):
        raise RuntimeError(
            "pinned FSDP2 exposes neither the Fairseq2 legacy gradient-sync "
            "method nor its PyTorch 2.8 replacement"
        )
    signature = inspect.signature(current)
    parameters = list(signature.parameters.values())
    if len(parameters) < 2 or parameters[1].name != "requires_gradient_sync":
        raise RuntimeError(f"unexpected FSDP2 gradient-sync signature: {signature}")

    def set_requires_grad_sync(self: Any, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("FSDP2 gradient-sync flag must be boolean")
        self.set_requires_gradient_sync(value)

    setattr(FSDPModule, legacy_name, set_requires_grad_sync)
    if not callable(getattr(FSDPModule, legacy_name, None)):
        raise RuntimeError("failed to install the packet-owned FSDP2 delegation")
    return {
        "status": "PASS",
        "mode": "packet_owned_guarded_delegation",
        "legacy_name": legacy_name,
        "current_name": current_name,
        "current_signature": str(signature),
    }
