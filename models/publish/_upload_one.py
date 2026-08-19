"""One-shot per-file upload helper, kept as an operational record.

Usage: _upload_one.py <repo_id> <local_path> <path_in_repo>; auth via HF_TOKEN.
"""
import os, sys, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
rid, local, path = sys.argv[1], sys.argv[2], sys.argv[3]
n = os.path.getsize(local); t = time.time()
print(f"START {rid}:{path}  {n:,} B", flush=True)
api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=rid, repo_type="model")
d = time.time()-t
print(f"DONE  {rid}:{path}  {n:,} B in {d:.1f}s = {n/d/1e6:.2f} MB/s", flush=True)
