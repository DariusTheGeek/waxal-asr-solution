#!/usr/bin/env python3
"""Build the TTIA enrolment gallery inputs: audio directory plus manifest.

The gallery is pinned: ``enrollment_ids.csv.gz`` beside this script names
every clip in it, with its profile key and source. Each of the 84 idiolect
profiles collects the clips the `google/WaxalNLP` release attributes to one
contributor; profile keys carry the release's own identifiers. Two sources:

``labeled``
    Training clips already on disk under the derived training audio root
    (the same 16 kHz files the ASR training reads).

``pool``
    Clips from the unlabelled release of `google/WaxalNLP`. Enrolment needs
    audio and a profile key, not a transcript, so these rows widen the
    gallery's per-profile coverage beyond the labelled data. Their audio
    bytes are extracted here into ``<out>/audio/``.

Output: ``<out>/enrollment.parquet`` (id, profile_key, derived_audio_relpath)
covering every pinned clip, ready for ``embed.py``. Embedding is GPU work and
is left to that script:

    python inference/ttia/build_enrollment.py --pool-parquets <dir-of-lin-unlabeled>
    for S in 0 1 2 3; do
      python inference/ttia/embed.py --manifest outputs/ttia/enrollment.parquet \\
        --audio outputs/ttia/audio --device cuda:$S --shard $S --num-shards 4 \\
        --output outputs/ttia/_shard$S.npz &
    done; wait
    python inference/ttia/merge.py --shards outputs/ttia/_shard*.npz \\
      --output outputs/ttia/enrollment.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PINNED = Path(__file__).resolve().parent / "enrollment_ids.csv.gz"
TRAIN_AUDIO = ROOT / "data/derived/omniasr/lin_cv002_supervised_v1/audio"
TRAIN_ROWS = (ROOT / "data/derived/portable/omniasr1b_lin_cv002_v1"
              / "manifests/train.rows.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-parquets", type=Path, required=True,
                    help="directory of lin unlabeled parquet shards from "
                         "google/WaxalNLP")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ttia")
    args = parser.parse_args()

    pinned = pd.read_csv(PINNED)
    labeled = pinned[pinned["source"] == "labeled"]
    pool = pinned[pinned["source"] == "pool"]
    print(f"pinned gallery: {len(pinned)} clips "
          f"({len(labeled)} labeled, {len(pool)} pool), "
          f"{pinned['profile_key'].nunique()} profiles")

    rows = []

    # Labelled clips resolve against the training audio already on disk.
    # Paths are stored absolute; embed.py joins them against --audio, and
    # pathlib keeps an absolute right-hand side as-is.
    train = pd.read_parquet(TRAIN_ROWS, columns=["id", "derived_audio_relpath"])
    relpath = dict(zip(train["id"].astype(str), train["derived_audio_relpath"]))
    missing = [i for i in labeled["id"].astype(str) if i not in relpath]
    if missing:
        raise SystemExit(f"{len(missing)} labeled ids not in {TRAIN_ROWS}, "
                         f"first: {missing[:3]}")
    for r in labeled.itertuples():
        rows.append({"id": str(r.id), "profile_key": r.profile_key,
                     "derived_audio_relpath": str(TRAIN_AUDIO
                                                  / relpath[str(r.id)])})

    # Pool clips are extracted from the unlabeled parquet shards by id.
    audio_dir = args.out / "audio/lin"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wanted = dict(zip(pool["id"].astype(str), pool["profile_key"]))
    found = 0
    for shard in sorted(args.pool_parquets.glob("*.parquet")):
        frame = pd.read_parquet(shard, columns=["id", "audio"])
        frame = frame[frame["id"].astype(str).isin(wanted)]
        for r in frame.itertuples():
            data = r.audio["bytes"]
            if not data:
                raise SystemExit(f"empty audio bytes for pool clip {r.id}")
            path = audio_dir / f"{r.id}.mp3"
            path.write_bytes(data)
            rows.append({"id": str(r.id), "profile_key": wanted[str(r.id)],
                         "derived_audio_relpath": str(path)})
            found += 1
        if found == len(wanted):
            break
    if found != len(wanted):
        raise SystemExit(f"only {found}/{len(wanted)} pool clips found under "
                         f"{args.pool_parquets}")

    manifest = (pd.DataFrame(rows).sort_values("id").reset_index(drop=True))
    out_path = args.out / "enrollment.parquet"
    manifest.to_parquet(out_path, index=False)
    print(f"wrote {len(manifest)} rows -> {out_path}")
    print("next: embed the manifest with embed.py (see module docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
