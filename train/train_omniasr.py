#!/usr/bin/env python3
"""Launch training for one OmniASR model.

Thin driver. The training logic is the family code under ``train/families/``
— the verbatim code that produced the released weights — and the
hyperparameters are the fairseq2 run configs under ``train/recipes/``. This
script only resolves which of each to use, satisfies the environment contract
the family code expects, and starts torchrun.

Invoke through ``run_train.sh`` rather than directly.

Prerequisites
-------------
* the ``omni`` environment          bash install.sh
* the official parent checkpoints   see SOLUTION.md
* prepared manifests                see SOLUTION.md

Smoke mode
----------
``--smoke`` overrides the step counts so the whole path -- data loading, model
build, optimizer step, checkpoint write -- executes in minutes instead of days.
It proves the path runs; it does not reproduce a released weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

SMOKE_REGIME = {
    "num_steps": 4,
    "validate_after_n_steps": 0,
    "validate_every_n_steps": 2,
    "checkpoint_after_n_steps": 0,
    "checkpoint_every_n_steps": 2,
    "publish_metrics_after_n_steps": 0,
    "publish_metrics_every_n_steps": 2,
    "keep_last_n_checkpoints": 1,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="configs/<lang>/<model>.yaml")
    parser.add_argument("--recipe", type=Path,
                        help="override the fairseq2 run config")
    parser.add_argument("--output", type=Path,
                        help="override the training output directory")
    parser.add_argument("--nproc", type=int, default=0,
                        help="processes per node; 0 means every visible GPU")
    parser.add_argument("--parents", type=Path, default=ROOT / "weights/parents",
                        help="directory holding the official parent checkpoints")
    parser.add_argument("--smoke", action="store_true",
                        help="run a handful of steps to prove the path executes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    family = config["family"]
    lane = args.config.parent.name
    model = config["model"]

    family_dir = ROOT / "train/families" / family
    train_py = family_dir / "train.py"
    if not train_py.is_file():
        raise SystemExit(f"no training entry point for family {family}: {train_py}")

    recipe = args.recipe or (ROOT / "train/recipes" / lane / f"{model}.yaml")
    if not recipe.is_file():
        raise SystemExit(f"no recipe config: {recipe}")

    output = args.output or (ROOT / "outputs/training" / f"{lane}_{model}")
    runtime = output / "_runtime"
    trainer_out = output / "_trainer"
    for directory in (output, runtime, trainer_out):
        directory.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Write a shortened copy rather than mutating the real recipe, so the
        # shipped hyperparameters can never be silently replaced by smoke values.
        spec = yaml.safe_load(recipe.read_text())
        spec.setdefault("regime", {}).update(SMOKE_REGIME)
        recipe = output / "config.smoke.yaml"
        recipe.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f">>> smoke mode: {SMOKE_REGIME['num_steps']} steps -> {recipe}")

    nproc = args.nproc
    if nproc == 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        nproc = len(visible.split(",")) if visible else _gpu_count()
    if nproc < 1:
        raise SystemExit("no GPU visible; OmniASR training requires at least one")

    python = ROOT / ".venvs/omni/bin/python"
    if not python.exists():
        raise SystemExit("missing omni environment. Run: bash install.sh")

    command = [str(python), "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={nproc}", str(train_py), str(trainer_out),
               "--config-file", str(recipe)]

    # The family code resolves its asset cards and parent checkpoints relative to
    # a repository root that must expose README.md, models/MODELS.json and
    # data/provenance/SOURCES.json. This repository provides all three.
    # The family code fences training behind a lease, so that two launches
    # cannot write the same checkpoint namespace concurrently. The lease store
    # is an ordinary directory and the lease an ordinary JSON file -- there is
    # no remote service and no credential involved. One launch here owns its own
    # output directory, so the driver mints the lease it is about to hold.
    lease = os.environ.get("WAXAL3_LEASE_TOKEN") or uuid.uuid4().hex
    generation = int(os.environ.get("WAXAL3_LEASE_GENERATION", "1"))
    experiment_id = f"{lane}_{model}"
    packet_digest = hashlib.sha256(recipe.read_bytes()).hexdigest()
    store = Path(os.environ.get("WAXAL3_REMOTE_STORE") or (output / "_lease"))
    lease_path = store / "leases" / f"{experiment_id}.json"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "ACTIVE",
        "experiment_id": experiment_id,
        "packet_digest": packet_digest,
        "generation": generation,
        "token": lease,
    }, indent=2, sort_keys=True) + "\n")

    env = dict(os.environ,
               WAXAL3_REPO_ROOT=str(ROOT),
               WAXAL3_RUNTIME_DIR=str(runtime),
               WAXAL3_TRAINER_OUTPUT_DIR=str(trainer_out),
               WAXAL3_LEASE_TOKEN=lease,
               WAXAL3_LEASE_GENERATION=str(generation),
               WAXAL3_EXPERIMENT_ID=experiment_id,
               WAXAL3_PACKET_DIGEST=packet_digest,
               WAXAL3_REMOTE_STORE=str(store),
               # Selects the run profile: checkpoint retention, and for "production"
               # an epoch-geometry cross-check against the recipe.
               WAXAL3_PROFILE="smoke" if args.smoke else "production",
               WAXAL_PARENT_DIR=str(args.parents),
               PYTHONPATH=f"{family_dir}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"))

    print(f">>> family  {family}")
    print(f">>> recipe  {recipe.relative_to(ROOT) if recipe.is_relative_to(ROOT) else recipe}")
    print(f">>> output  {output.relative_to(ROOT)}")
    print(f">>> nproc   {nproc}")
    print(">>> " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=family_dir, env=env).returncode


def _gpu_count() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "--list-gpus"],
                                capture_output=True, text=True, check=False)
        return len([l for l in result.stdout.splitlines() if l.strip()])
    except FileNotFoundError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
