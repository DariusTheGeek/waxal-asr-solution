#!/usr/bin/env python3
"""Launch training for one MMS-1B native-adapter specialist.

Thin driver, same shape as ``train_omniasr.py``. The training logic is
``train/families/mms_1b_native_adapter/supervised/train.py``, and the run
configuration is the record under ``train/recipes/<lang>/mms1b.yaml`` that
produced the released weight.

MMS trains under the ``hf`` environment (Torch 2.5.1 + transformers), not the
``omni`` one.

Invoke through ``run_train.sh`` rather than directly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "train/families/mms_1b_native_adapter"

# Keys the smoke run shortens. Names follow the shipped run configs.
SMOKE_OVERRIDES = {"num_train_epochs": 1, "max_steps": 4, "save_steps": 2,
                   "eval_steps": 2, "logging_steps": 1, "passes": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="configs/<lang>/mms1b.yaml")
    parser.add_argument("--recipe", type=Path, help="override the run config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    lane = args.config.parent.name
    if config["family"] != "mms_1b_native_adapter":
        raise SystemExit(f"{args.config} is not an MMS config (family={config['family']})")

    recipe = args.recipe or (ROOT / "train/recipes" / lane / "mms1b.yaml")
    if not recipe.is_file():
        raise SystemExit(f"no recipe config: {recipe}")

    output = args.output or (ROOT / "outputs/training" / f"{lane}_mms1b")
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        spec = yaml.safe_load(recipe.read_text())
        for key, value in SMOKE_OVERRIDES.items():
            if key in spec:
                spec[key] = value
        recipe = output / "config.smoke.yaml"
        recipe.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f">>> smoke mode: shortened schedule -> {recipe}")

    python = ROOT / ".venvs/hf/bin/python"
    if not python.exists():
        raise SystemExit("missing hf environment. Run: bash install.sh")

    command = [str(python), "-m", "supervised.train",
               "--config", str(recipe), "--run-dir", str(output)]
    if args.resume_from_checkpoint:
        command += ["--resume-from-checkpoint", str(args.resume_from_checkpoint)]

    env = dict(os.environ,
               WAXAL3_REPO_ROOT=str(ROOT),
               PYTHONPATH=f"{FAMILY}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"))

    print(f">>> family  mms_1b_native_adapter ({lane})")
    print(f">>> recipe  {recipe.relative_to(ROOT) if recipe.is_relative_to(ROOT) else recipe}")
    print(f">>> output  {output.relative_to(ROOT)}")
    print(">>> " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=FAMILY, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
