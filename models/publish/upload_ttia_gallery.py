#!/usr/bin/env python3
"""Publish the TTIA gallery dataset repository. One-shot; auth via HF_TOKEN.

Uploads the three files named in models/ASSETS.json from their local pipeline
paths, plus the dataset card in this directory. Created private, like the
model repositories, so an incomplete card never sits public; flip with
--public once it looks right.

Usage
-----
HF_TOKEN=... python models/publish/upload_ttia_gallery.py
HF_TOKEN=... python models/publish/upload_ttia_gallery.py --public
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "models/ASSETS.json").read_text())
CARD = Path(__file__).resolve().parent / "ttia-gallery/README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true",
                        help="flip the repository public instead of uploading")
    args = parser.parse_args()

    repo = SPEC["repo"]
    repo_id = f"{SPEC['namespace']}/{repo['name']}"
    api = HfApi(token=os.environ["HF_TOKEN"])

    if args.public:
        api.update_repo_settings(repo_id=repo_id, repo_type=repo["type"],
                                 private=False)
        print(f"PUBLIC  {repo_id}")
        return 0

    api.create_repo(repo_id=repo_id, repo_type=repo["type"],
                    private=True, exist_ok=True)
    uploads = [(CARD, "README.md")]
    uploads += [(ROOT / e["dest"], e["path"]) for e in repo["files"]]
    for local, path in uploads:
        size = local.stat().st_size
        started = time.time()
        print(f"START {repo_id}:{path}  {size:,} B", flush=True)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=path,
                        repo_id=repo_id, repo_type=repo["type"])
        elapsed = time.time() - started
        print(f"DONE  {repo_id}:{path}  in {elapsed:.1f}s", flush=True)
    print("\nUploaded private. Review the card, then rerun with --public.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
