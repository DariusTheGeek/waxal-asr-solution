#!/usr/bin/env python3
"""Flip repositories between private and public.

Uploads are staged privately so an incomplete model card never sits on the
`google/WaxalNLP` dataset page. This flips them once the cards are right.

Flipping to public also frees the account's private-storage quota, which is why
the upload runs in two batches: 159 GB does not fit in the 100 GB free private
tier at once, but neither batch alone exceeds it.

Usage
-----
python models/publish/set_visibility.py --batch A --public
python models/publish/set_visibility.py --status
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "repos.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=["A", "B"])
    parser.add_argument("--repo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--public", action="store_true")
    group.add_argument("--private", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    repos = [r for r in SPEC["repos"]
             if (not args.batch or r["batch"] == args.batch)
             and (not args.repo or r["name"] == args.repo)]

    for r in repos:
        rid = f"{SPEC['namespace']}/{r['name']}"
        if args.status:
            try:
                info = api.model_info(rid, files_metadata=True)
                files = len(info.siblings)
                size = sum((s.lfs.size if s.lfs else s.size) or 0 for s in info.siblings)
                state = "private" if info.private else "PUBLIC"
                print(f"  {state:<8} {rid:<45} {files:>2} files  {size/1e9:7.2f} GB")
            except Exception as exc:
                print(f"  MISSING  {rid:<45} {type(exc).__name__}")
            continue
        if not (args.public or args.private):
            raise SystemExit("choose --public, --private or --status")
        api.update_repo_settings(repo_id=rid, private=bool(args.private))
        print(f"  {'private' if args.private else 'PUBLIC':<8} {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
