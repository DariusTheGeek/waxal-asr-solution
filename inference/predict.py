#!/usr/bin/env python3
"""Run the full pipeline: test audio in, submission CSV out.

This is the orchestrator invoked by ``run_inference.sh``. It runs under the
``fuse`` environment and dispatches each decode stage to the environment that
stage's dependencies require, because fairseq2 and transformers pin
incompatible Torch builds and cannot share an interpreter.

    stage 0   joint tag-free decode        .venvs/omni
              text language identification .venvs/fuse
    stage 1   OmniASR decodes              .venvs/omni
              MMS decodes                  .venvs/hf
    stage 2   TTIA voice embedding         .venvs/hf
              TTIA profile matching        .venvs/fuse
    stage 3   TTIA / medoid fusion         .venvs/fuse
    stage 4   normalise and write          .venvs/fuse

Language routing comes from the audio via stage 0. Profile keys for the
Lingala lane come from the audio via stage 2 (Test-Time Idiolect Adaptation),
matched against the enrolment gallery built by inference/ttia/build_enrollment.py.

Usage
-----
bash run_inference.sh                 # normal path
bash run_inference.sh --dry-run       # print the plan without running anything
bash run_inference.sh --from-stage 3  # reuse existing decodes and keys
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENVS = {name: ROOT / ".venvs" / name / "bin" / "python" for name in ("omni", "hf", "fuse")}
ENV_FOR_MODEL = {"mms1b": "hf"}          # everything else is OmniASR -> omni


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def run(env: str, script: Path, *args: str, dry: bool = False) -> None:
    python = ENVS[env]
    if not python.exists():
        raise SystemExit(f"missing environment '{env}'. Run: bash install.sh")
    command = [str(python), str(script), *map(str, args)]
    if dry:
        print("   would run:", " ".join(command), flush=True)
        return
    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"stage failed ({env}): {' '.join(command)}")
    print(f"   [{time.time() - started:.1f}s]", flush=True)


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def expected_rows(route: Path, lane: str) -> int:
    """How many clips this lane owns, according to stage 0."""
    if not route.is_file():
        return -1
    with route.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for r in csv.DictReader(handle) if r["language"] == lane)


def check_submission(path: Path, expected: int) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [r["ID"] for r in rows]
    problems = []
    if len(rows) != expected:
        problems.append(f"expected {expected} rows, found {len(rows)}")
    if len(set(identifiers)) != len(identifiers):
        problems.append("duplicate IDs")
    blank = [r["ID"] for r in rows if not (r["Target"] or "").strip()]
    if blank:
        problems.append(f"{len(blank)} blank target(s), first: {blank[:3]}")
    if problems:
        raise SystemExit("submission failed validation:\n  " + "\n  ".join(problems))
    return {"rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference.yaml")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml")
    parser.add_argument("--from-stage", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = load_config(args.paths)
    out = {k: ROOT / v for k, v in paths["outputs"].items()}
    for directory in out.values():
        directory.mkdir(parents=True, exist_ok=True)

    audio = ROOT / paths["data"]["audio"]
    if not args.dry_run and not audio.is_dir():
        raise SystemExit(f"test audio not found at {audio} (see README.md, Data)")

    started = time.time()

    if args.from_stage <= 0:
        print("=== Stage 0: joint decode and language routing ===", flush=True)
        joint = out["route"] / "joint_hypotheses.csv"
        run("omni", ROOT / "inference/decode/omniasr.py",
            "--config", ROOT / "configs/joint/joint.yaml",
            "--audio", audio, "--output", joint, dry=args.dry_run)
        run("fuse", ROOT / "inference/route/route.py",
            "--hypotheses", joint,
            "--model", ROOT / load_config(ROOT / "configs/lid.yaml")["classifier"]["path"],
            "--output", out["route"] / "route.csv", dry=args.dry_run)

    route = out["route"] / "route.csv"
    if args.from_stage <= 1:
        print("=== Stage 1: per-model decode, per language lane ===", flush=True)
        for lane, lane_config in config["lanes"].items():
            for model in lane_config["members"]:
                env = ENV_FOR_MODEL.get(model, "omni")
                script = ROOT / ("inference/decode/mms.py" if env == "hf"
                                 else "inference/decode/omniasr.py")
                target = out["decodes"] / f"{lane}_{model}.csv"
                if target.exists() and _row_count(target) == expected_rows(route, lane):
                    print(f" -- {lane}/{model} [{env}] already complete, skipped", flush=True)
                    continue
                print(f" -- {lane}/{model} [{env}]", flush=True)
                run(env, script,
                    "--config", ROOT / f"configs/{lane}/{model}.yaml",
                    "--audio", audio, "--route", route, "--lane", lane,
                    "--output", out["decodes"] / f"{lane}_{model}.csv", dry=args.dry_run)

    ttia_lanes = [lane for lane, c in config["lanes"].items() if c["method"] == "ttia"]
    ttia = config.get("ttia", {})
    if args.from_stage <= 2 and ttia_lanes:
        print("=== Stage 2: TTIA profile matching ===", flush=True)
        gallery = ROOT / ttia["gallery"]
        gallery_manifest = ROOT / ttia["manifest"]
        if not args.dry_run and not gallery.is_file():
            raise SystemExit(f"enrolment gallery not found at {gallery}; "
                             "build it with inference/ttia/build_enrollment.py (see SOLUTION.md)")
        test_vectors = out["ttia"] / "test_clips.npz"
        run("hf", ROOT / "inference/ttia/embed.py",
            "--audio", audio, "--output", test_vectors, dry=args.dry_run)
        for lane in ttia_lanes:
            run("fuse", ROOT / "inference/ttia/match.py",
                "--enroll", gallery, "--enroll-manifest", gallery_manifest,
                "--query", test_vectors,
                "--layer", *map(str, ttia["layers"]),
                "--transform", ttia["transform"], "--scoring", ttia["scoring"],
                "--restrict", route, "--language", lane,
                "--output", out["ttia"] / f"keys_{lane}.csv", dry=args.dry_run)

    if args.from_stage <= 3:
        print("=== Stage 3: fusion ===", flush=True)
        for lane, lane_config in config["lanes"].items():
            sources = [out["decodes"] / f"{lane}_{m}.csv" for m in lane_config["members"]]
            print(f" -- {lane}: {lane_config['method']} over {len(sources)} members", flush=True)
            extra: list[str] = []
            if lane_config["method"] == "ttia":
                extra = ["--keys", str(out["ttia"] / f"keys_{lane}.csv"),
                         "--train-texts", str(ROOT / ttia["train_texts"])]
            run("fuse", ROOT / "inference/fuse/fuse.py",
                "--method", lane_config["method"], *extra,
                "--output", out["fused"] / f"{lane}.csv", *sources, dry=args.dry_run)

    print("=== Stage 4: post-process and write submission ===", flush=True)
    submission = ROOT / config["submission"]["path"]
    run("fuse", ROOT / "inference/submit/postprocess.py",
        "--lanes", *(out["fused"] / f"{lane}.csv" for lane in config["lanes"]),
        "--output", submission, dry=args.dry_run)

    if args.dry_run:
        print("\nDry run complete; nothing was executed.")
        return 0

    summary = check_submission(submission, config["submission"]["expected_rows"])
    elapsed = time.time() - started
    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "seed": config["seed"],
        "lanes": {k: v["method"] for k, v in config["lanes"].items()},
        **summary,
    }
    (out["submissions"] / "RUN_RECORD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"\nSubmission: {submission}")
    print(f"rows {summary['rows']}   sha256 {summary['sha256']}")
    print(f"total {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
