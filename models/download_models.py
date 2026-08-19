#!/usr/bin/env python3
"""Fetch every model weight this solution needs from the Hugging Face Hub.

Every file's SHA-256 is recorded in MODELS.json and verified after download;
a mismatch is fatal rather than a warning, so a rerun months later either
gets the same bytes or refuses.

Total download is roughly 159 GB: one routing model, five Lingala models
and four Shona models.

Usage
-----
python models/download_models.py                    # everything
python models/download_models.py --lane sna         # one language lane
python models/download_models.py --repo waxal-lin-mms-1b
python models/download_models.py --verify-only      # re-check what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "MODELS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["lin", "sna", "routing"])
    parser.add_argument("--repo")
    parser.add_argument("--dest", type=Path, default=ROOT / "weights")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    spec = json.loads(MANIFEST.read_text())
    repos = [r for r in spec["repos"]
             if (not args.lane or r["lane"] == args.lane)
             and (not args.repo or r["name"] == args.repo)]
    if not repos:
        raise SystemExit("no repository matched the given filters")

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit("huggingface_hub is required: bash install.sh")

    failures: list[str] = []
    for repo in repos:
        repo_id = f"{spec['namespace']}/{repo['name']}"
        target = args.dest / repo["name"]
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {repo_id} ({repo['bytes'] / 1e9:.2f} GB) ===")
        for entry in repo["files"]:
            local = target / entry["path"]
            if not args.verify_only and not local.exists():
                hf_hub_download(repo_id=repo_id, filename=entry["path"],
                                revision=repo.get("revision", "main"),
                                local_dir=str(target))
            if not local.exists():
                failures.append(f"{repo_id}:{entry['path']} missing")
                print(f"  MISSING  {entry['path']}")
                continue
            digest = sha256_file(local)
            if digest != entry["sha256"]:
                failures.append(f"{repo_id}:{entry['path']} sha256 mismatch")
                print(f"  BAD SHA  {entry['path']}")
            else:
                print(f"  ok       {entry['path']}  {local.stat().st_size:,} B")

    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"\nAll {sum(len(r['files']) for r in repos)} file(s) present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
