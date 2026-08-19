#!/usr/bin/env python3
"""Create each HF repo private and upload its files, one file at a time.

Sequential by design: the uplink saturates at ~19 MB/s on a single stream, so
concurrent uploads split the same pipe and finish later (measured: 2 streams
-8%, 4 streams -10%).

Each file is uploaded in a child process under a hard wall-clock timeout. The
Xet client can wedge with its socket in CLOSE-WAIT and the main thread parked in
futex_wait -- that never raises, so an in-process try/except cannot recover from
it. A killed child can. Retries fall back to the classic LFS path.

Already-uploaded files of the right size are skipped, so a rerun resumes.

Local sources are not part of the released manifest: point ``--sources`` at a
JSON file mapping ``"<repo-name>/<dst>"`` to the local path holding that file,
and ``--tokenizer`` at the OmniASR tokenizer (91,481 B, byte-identical across
every OmniASR repository).

Usage
-----
python models/publish/upload_batch.py A --sources local_sources.json \\
    --tokenizer path/to/omniASR_tokenizer_written_v2.model
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "repos.json").read_text())
NS = SPEC["namespace"]
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)


def is_omni(name: str) -> bool:
    """OmniASR repos ship a fairseq2 asset card; the MMS repos do not."""
    return "omniasr" in name or name.endswith("-lid")


def remote_sizes(rid: str) -> dict[str, int]:
    try:
        return {s.rfilename: (s.lfs.size if s.lfs else s.size)
                for s in api.model_info(rid, files_metadata=True).siblings}
    except Exception:
        return {}


def send(rid: str, local: str, path: str, n: int) -> float:
    """Upload one file in a child process. Returns seconds taken."""
    budget = int(600 + n / 4e6)          # ~4x headroom over the 19 MB/s line rate
    for attempt in (1, 2, 3):
        # Xet wedges on multi-GB uploads from this host: the socket parks in
        # CLOSE-WAIT, the main thread in futex_wait, and the transfer sits at
        # zero bytes/s indefinitely without raising. Classic LFS multipart
        # reaches the same ~19 MB/s line rate and does not wedge.
        env = dict(os.environ, HF_TOKEN=TOKEN, HF_HUB_DISABLE_PROGRESS_BARS="1",
                   HF_HUB_DISABLE_XET="1")
        t = time.time()
        try:
            r = subprocess.run([sys.executable, str(ROOT / "_upload_one.py"), rid, local, path],
                               env=env, timeout=budget, capture_output=True, text=True)
            if r.returncode == 0:
                return time.time() - t
            print(f"    attempt {attempt} exit {r.returncode}: {r.stderr.strip()[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    attempt {attempt} TIMED OUT after {budget}s -- killed, retrying"
                  f"{' without Xet' if attempt == 1 else ''}", flush=True)
        time.sleep(15 * attempt)
    raise RuntimeError(f"upload failed after 3 attempts: {rid}:{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("repos", nargs="*", help="limit to these repo names")
    parser.add_argument("--sources", type=Path, required=True,
                        help='JSON mapping "<repo-name>/<dst>" -> local path')
    parser.add_argument("--tokenizer", type=Path,
                        help="omniASR_tokenizer_written_v2.model to ship in each OmniASR "
                             "repo; required only when the selection includes one")
    args = parser.parse_args()

    batch = args.batch
    only = set(args.repos) or None
    sources = json.loads(args.sources.read_text())

    selected = [r for r in SPEC["repos"]
                if r["batch"] == batch and not (only and r["name"] not in only)]
    # Validate the whole selection before creating anything: a missing
    # --tokenizer discovered mid-loop would leave empty private repos behind.
    needs_tokenizer = [r["name"] for r in selected if is_omni(r["name"])]
    if needs_tokenizer and not args.tokenizer:
        raise SystemExit(f"--tokenizer is required for OmniASR repos: "
                         f"{', '.join(needs_tokenizer)}")

    t_all = time.time(); sent = 0; skipped = 0

    for r in selected:
        rid = f"{NS}/{r['name']}"
        api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
        print(f"\n=== {rid} (private) ===", flush=True)

        d = ROOT.parent / "cards" / r["name"]
        jobs = [(str(d / s), s) for s in ("README.md", "card.yaml") if (d / s).exists()]
        if is_omni(r["name"]):
            jobs.append((str(args.tokenizer), "omniASR_tokenizer_written_v2.model"))
        for f in r["files"]:
            key = f"{r['name']}/{f['dst']}"
            if key not in sources:
                raise SystemExit(f"--sources has no local path for {key}")
            jobs.append((sources[key], f["dst"]))

        have = remote_sizes(rid)
        for local, path in jobs:
            n = os.path.getsize(local)
            if have.get(path) == n:
                print(f"  {path:<38} {n:>14,} B  already present, skipped", flush=True)
                skipped += n
                continue
            dt = send(rid, local, path, n)
            sent += n
            print(f"  {path:<38} {n:>14,} B  {dt:7.1f}s  {n/max(dt,1e-9)/1e6:6.2f} MB/s", flush=True)

    el = time.time() - t_all
    print(f"\nBATCH {batch} COMPLETE: uploaded {sent:,} B, skipped {skipped:,} B "
          f"in {el/60:.1f} min = {sent/max(el,1e-9)/1e6:.2f} MB/s", flush=True)


if __name__ == "__main__":
    main()
