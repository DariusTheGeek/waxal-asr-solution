from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from torch.distributed.fsdp import FSDPModule

from fsdp_compat import install_fsdp2_gradient_sync_compat


CODE_ROOT = Path(__file__).resolve().parents[1]


def test_guarded_compatibility_method_is_callable() -> None:
    record = install_fsdp2_gradient_sync_compat()
    assert record["status"] == "PASS"
    assert callable(getattr(FSDPModule, "set_requires_grad_sync", None))


def test_two_rank_gloo_accumulation_matches_synchronized_reference() -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(CODE_ROOT),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("fsdp2_accumulation_probe.py")),
        ],
        cwd=CODE_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert '"status": "PASS"' in result.stdout
    assert '"microbatches": 4' in result.stdout
