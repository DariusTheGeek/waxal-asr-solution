#!/usr/bin/env python3
"""Fetch the TTIA data assets the default Lingala lane needs.

Three files from one Hugging Face dataset repository, placed at the exact
paths configs/inference.yaml reads: the enrolment gallery (voice vectors and
its manifest) under outputs/ttia/, and the per-profile training text under
data/derived/. Every file's SHA-256 is recorded in ASSETS.json and verified
after download; a mismatch is fatal rather than a warning.

The assets are derived from google/WaxalNLP (CC-BY-SA-4.0); they can also be
rebuilt from that dataset with inference/ttia/build_enrollment.py, embed.py
and merge.py — this download and that rebuild produce the same bytes.

Total download is roughly 1 GB. Not needed when lanes.lin.method is "rover".

Usage
-----
.venvs/hf/bin/python models/download_assets.py               # fetch and verify
.venvs/hf/bin/python models/download_assets.py --verify-only # re-check disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "ASSETS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    spec = json.loads(MANIFEST.read_text())
    repo = spec["repo"]
    repo_id = f"{spec['namespace']}/{repo['name']}"

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit("huggingface_hub is required: bash install.sh")

    failures: list[str] = []
    print(f"=== {repo_id} ({repo['bytes'] / 1e9:.2f} GB) ===")
    for entry in repo["files"]:
        local = ROOT / entry["dest"]
        local.parent.mkdir(parents=True, exist_ok=True)
        if not args.verify_only and not local.exists():
            hf_hub_download(repo_id=repo_id, filename=entry["path"],
                            repo_type=repo["type"],
                            revision=repo.get("revision", "main"),
                            local_dir=str(local.parent))
        if not local.exists():
            failures.append(f"{repo_id}:{entry['path']} missing")
            print(f"  MISSING  {entry['dest']}")
            continue
        digest = sha256_file(local)
        if digest != entry["sha256"]:
            failures.append(f"{repo_id}:{entry['path']} sha256 mismatch")
            print(f"  BAD SHA  {entry['dest']}")
        else:
            print(f"  ok       {entry['dest']}  {local.stat().st_size:,} B")

    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"\nAll {len(repo['files'])} file(s) present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
