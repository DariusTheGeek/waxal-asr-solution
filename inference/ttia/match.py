#!/usr/bin/env python3
"""Match each test clip to an enrolled idiolect profile, from its voice vector.

The core of Test-Time Idiolect Adaptation: every clip's voice vector is
compared against the enrolment gallery and assigned the closest profile key.
Downstream fusion then consults that profile's idiolect (its training
transcripts). Emits the decision margin alongside each key so the decision
quality is inspectable; the keys are consumed ungated.

Method: per requested layer, LDA fitted on the gallery (it needs the many
clips per profile the enrolment build provides), cosine scoring against
profile centroids, cohort normalisation across profiles, then the normalised
similarities are summed over layers. Different depths make different mistakes,
so the sum beats any single layer.

The gallery is built from training and pool audio by ``build_enrollment.py``;
its profiles and their keys are described there.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def project(enroll, query, transform, labels, n_profiles):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if transform == "pca":
        fitted = PCA(n_components=min(200, enroll.shape[0] - 1), whiten=True).fit(enroll)
    elif transform == "lda":
        fitted = LinearDiscriminantAnalysis(
            n_components=min(n_profiles - 1, 150, enroll.shape[1])).fit(enroll, labels)
    else:
        return enroll, query
    return fitted.transform(enroll), fitted.transform(query)


def profile_scores(query, enroll, by_profile, profiles, scoring):
    from sklearn.preprocessing import normalize

    if scoring == "centroid":
        centroids = normalize(np.stack([enroll[by_profile[key]].mean(0) for key in profiles]))
        return query @ centroids.T
    similarities = query @ enroll.T
    if scoring == "nearest":
        return np.stack([similarities[:, by_profile[key]].max(1) for key in profiles], 1)
    out = []
    for key in profiles:
        profile_sims = similarities[:, by_profile[key]]
        k = min(5, profile_sims.shape[1])
        out.append(np.sort(profile_sims, axis=1)[:, -k:].mean(1))
    return np.stack(out, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", type=Path, required=True,
                    help="gallery npz from the enrolment build")
    parser.add_argument("--enroll-manifest", type=Path, required=True,
                    help="parquet with id and profile_key for every gallery clip")
    parser.add_argument("--query", type=Path, required=True,
                    help="test-clip npz from embed.py")
    parser.add_argument("--layer", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--transform", choices=["none", "pca", "lda"], default="lda")
    parser.add_argument("--scoring", choices=["centroid", "nearest", "top5"],
                    default="centroid")
    parser.add_argument("--no-snorm", action="store_true")
    parser.add_argument("--restrict", type=Path,
                    help="optional route.csv; only its IDs are matched")
    parser.add_argument("--language",
                    help="with --restrict, keep only this language's rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from sklearn.preprocessing import normalize

    enroll_data = np.load(args.enroll, allow_pickle=True)
    query_data = np.load(args.query, allow_pickle=True)
    for layer in args.layer:
        if f"layer_{layer}" not in enroll_data.files or f"layer_{layer}" not in query_data.files:
            raise SystemExit(f"layer_{layer} missing; enrolment has {sorted(enroll_data.files)}")

    enroll_ids = [str(x) for x in enroll_data["ids"]]
    query_ids = [str(x) for x in query_data["ids"]]
    idx = list(range(len(query_ids)))
    if args.restrict:
        with args.restrict.open(encoding="utf-8-sig", newline="") as handle:
            keep = {r["ID"] for r in csv.DictReader(handle)
                    if not args.language or r.get("language") == args.language}
        idx = [n for n, i in enumerate(query_ids) if i in keep]
        query_ids = [query_ids[n] for n in idx]

    manifest_frame = pd.read_parquet(args.enroll_manifest, columns=["id", "profile_key"])
    profile_of = dict(zip(manifest_frame["id"].astype(str), manifest_frame["profile_key"].astype(str)))
    labels = np.array([profile_of[i] for i in enroll_ids])
    by_profile: dict[str, list[int]] = {}
    for n, i in enumerate(enroll_ids):
        by_profile.setdefault(profile_of[i], []).append(n)
    profiles = sorted(by_profile)

    sims = None
    for layer in args.layer:
        enroll, query = project(enroll_data[f"layer_{layer}"], query_data[f"layer_{layer}"][idx],
                                args.transform, labels, len(profiles))
        enroll, query = normalize(enroll), normalize(query)
        per_layer = profile_scores(query, enroll, by_profile, profiles, args.scoring)
        if not args.no_snorm:
            per_layer = ((per_layer - per_layer.mean(1, keepdims=True))
                         / (per_layer.std(1, keepdims=True) + 1e-9))
        sims = per_layer if sims is None else sims + per_layer

    order = np.argsort(-sims, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n",
                                fieldnames=["ID", "profile_key", "margin",
                                            "second_key", "similarity"])
        writer.writeheader()
        for n, qid in enumerate(query_ids):
            top, second = order[n, 0], order[n, 1]
            writer.writerow({"ID": qid, "profile_key": profiles[top],
                             "margin": f"{sims[n, top] - sims[n, second]:.6f}",
                             "second_key": profiles[second],
                             "similarity": f"{sims[n, top]:.6f}"})
    args.output.with_suffix(".json").write_text(json.dumps(
        {"clips": len(query_ids), "enrolled_profiles": len(profiles),
         "layers": args.layer, "transform": args.transform,
         "scoring": args.scoring, "snorm": not args.no_snorm}, indent=2) + "\n")
    print(f"matched {len(query_ids)} clips over {len(profiles)} profiles "
          f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
