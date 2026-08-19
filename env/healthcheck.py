#!/usr/bin/env python3
"""Probe one environment and write a JSON health report.

Run through env/verify_envs.sh rather than directly; that script drives every
profile and collects the evidence files together.
"""
from __future__ import annotations

import argparse
import importlib
import os
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

IMPORTS = {
    "omni": ["torch", "torchaudio", "omnilingual_asr", "fairseq2", "fairseq2n",
             "polars", "pyarrow"],
    "hf":   ["torch", "torchaudio", "transformers", "datasets", "accelerate",
             "peft", "librosa", "soundfile"],
    "fuse": ["sklearn", "joblib", "editdistance", "Levenshtein", "rapidfuzz",
             "jiwer", "numpy", "pandas"],
}

# The shipped language classifier was fitted under these exact versions.
# scikit-learn pickles are not forward or backward compatible in general.
PINNED = {"fuse": {"sklearn": "1.5.2", "joblib": "1.4.2"}}

GPU_PROFILES = {"omni", "hf"}
EXPECTED_GPUS = 4


def gpu_check() -> dict[str, object]:
    import torch

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                value = (torch.ones(8, device=f"cuda:{index}") * 2).sum().item()
            prop = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": prop.name,
                            "total_memory": int(prop.total_memory),
                            "tensor_check": float(value)})
    return {"torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
            "devices": devices}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(IMPORTS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": 1,
        "profile": args.profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "imports": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]          # type: ignore[assignment]
    imports: dict[str, object] = report["imports"]  # type: ignore[assignment]

    for name in IMPORTS[args.profile]:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
            imports[name] = {"status": "pass", "version": version}
            want = PINNED.get(args.profile, {}).get(name)
            if want and version != want:
                errors.append(f"pinned:{name}:expected {want}, found {version}")
        except Exception as exc:                    # record every failure
            imports[name] = {"status": "fail", "error": repr(exc)}
            errors.append(f"import:{name}:{exc!r}")

    if args.profile in GPU_PROFILES:
        try:
            report["gpu"] = gpu_check()
            count = report["gpu"]["gpu_count"]      # type: ignore[index]
            if count != EXPECTED_GPUS:
                errors.append(f"gpu_count_is_{count}_expected_{EXPECTED_GPUS}")
        except Exception as exc:
            errors.append(f"gpu_check:{exc!r}")

    # uv-created virtual environments do not ship a `pip` module, so the
    # dependency check has to go through uv itself.
    uv_bin = os.environ.get("UV_BIN") or str(Path.home() / ".local/bin/uv")
    try:
        check = subprocess.run([uv_bin, "pip", "check", "--python", sys.executable],
                               check=False, capture_output=True, text=True)
        report["pip_check"] = {"returncode": check.returncode,
                              "stdout": check.stdout, "stderr": check.stderr}
        if check.returncode != 0:
            errors.append("pip_check_failed")
    except Exception as exc:
        errors.append(f"pip_check:{exc!r}")

    report["status"] = "pass" if not errors else "fail"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": args.profile, "status": report["status"], "errors": errors}))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
