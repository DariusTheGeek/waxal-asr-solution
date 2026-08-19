#!/usr/bin/env python3
"""Fuse per-model hypotheses into one transcript per clip.

Three mechanisms:

``ttia``
    Test-Time Idiolect Adaptation, the Lingala lane. Each member hypothesis is
    boundary-normalised against the training text of the idiolect profile
    stage 2 matched to the clip; the conservative word ROVER of the normalised
    members is added as one further candidate; a count LM over that profile's
    training text then selects among the candidates. Requires ``--keys`` (from
    ``inference/ttia/match.py``) and ``--train-texts``.

``rover``
    Conservative word ROVER: word-level voting with a fixed pivot. Also the
    candidate generator inside ``ttia``.

``medoid``
    Word-medoid MBR: minimum mean peer-WER selection. Used on the Shona lane,
    where it selects one member's hypothesis whole.

``rover`` and ``medoid`` are reference-free; ``ttia`` additionally reads
training transcripts, keyed by the profile matched from the clip's voice
vector. None has a tunable threshold fitted on the evaluation set.

Usage
-----
python inference/fuse/fuse.py --method ttia --keys outputs/ttia/keys_lin.csv \\
    --train-texts data/derived/portable/omniasr1b_lin_cv002_v1/manifests/train.rows.parquet \\
    --output outputs/fused/lin.csv  a.csv b.csv c.csv d.csv
python inference/fuse/fuse.py --method rover  --output outputs/fused/lin.csv  a.csv b.csv c.csv d.csv
python inference/fuse/fuse.py --method medoid --output outputs/fused/sna.csv  w.csv x.csv y.csv z.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.text import normalize_text, token_key, word_tokens  # noqa: E402

PIVOT_PREFERENCE = 100.0   # keeps the pivot's surface form when votes tie
MEMBER_PREFERENCE = 1.0


def conservative_word_rover(hypotheses: list[str]) -> str:
    """Equal-weight word ROVER, first member as pivot, insertions never adopted.

    A position changes only on a *strict* majority against the pivot. With two
    hypotheses that can never happen -- the pivot always wins -- so the fusion
    would silently degenerate into "return the first input". Refuse instead.
    """
    if len(hypotheses) < 3:
        raise ValueError(
            f"conservative word ROVER needs at least 3 hypotheses, got {len(hypotheses)}; "
            "with 2 the pivot always wins and the fusion is a no-op"
        )
    tokenized = [word_tokens(h) for h in hypotheses]
    pivot = tokenized[0]

    if not pivot:
        # ROVER votes at pivot word positions. An empty pivot has none, so the
        # mechanism would emit nothing -- silently losing a row that other
        # members did transcribe. Observed on near-silent clips, where a CTC
        # pivot collapses to empty while an LLM decoder still produces text.
        #
        # Re-pivot on the strongest member that did produce text, keeping the
        # configured member order as the authority it already is elsewhere.
        # Voting needs three sources, so with fewer the strongest speaking
        # member stands on its own.
        #
        # Deliberately not the medoid: over exactly two hypotheses it normalises
        # each distance by that hypothesis's own length, which systematically
        # favours the longer string -- and the longer string of a near-silent
        # clip is usually a decoder repeating itself.
        spoken = [h for h in hypotheses if word_tokens(h)]
        if not spoken:
            return ""                          # genuinely nothing to say
        if len(spoken) < 3:
            return normalize_text(spoken[0])
        return conservative_word_rover(spoken)

    pivot_keys = [token_key(t) for t in pivot]

    votes: list[dict[str, float]] = [defaultdict(float) for _ in pivot]
    surfaces: list[dict[str, tuple[float, str]]] = [{} for _ in pivot]

    for member_index, tokens in enumerate(tokenized):
        keys = [token_key(t) for t in tokens]
        matcher = difflib.SequenceMatcher(None, pivot_keys, keys)
        for tag, p0, p1, m0, m1 in matcher.get_opcodes():
            same_width = (p1 - p0) == (m1 - m0)
            if tag == "equal" or (tag == "replace" and same_width):
                for offset in range(p1 - p0):
                    position, token = p0 + offset, tokens[m0 + offset]
                    key = keys[m0 + offset]
                    if not key:
                        continue
                    votes[position][key] += 1.0
                    preference = PIVOT_PREFERENCE if member_index == 0 else MEMBER_PREFERENCE
                    previous = surfaces[position].get(key)
                    if previous is None or preference > previous[0]:
                        surfaces[position][key] = (preference, token)
            elif tag == "delete":
                for position in range(p0, p1):
                    votes[position][""] += 1.0
            # Insertions are intentionally ignored: adopting them is what makes
            # ROVER hallucinate on this task.

    output: list[str] = []
    for position, pivot_token in enumerate(pivot):
        pivot_key = pivot_keys[position]
        if not votes[position]:
            output.append(pivot_token)
            continue
        winner, winner_votes = max(
            votes[position].items(), key=lambda item: (item[1], item[0] == pivot_key)
        )
        if winner != pivot_key and winner_votes > votes[position].get(pivot_key, 0.0):
            if winner:                       # an empty winner means "delete"
                output.append(surfaces[position][winner][1])
        else:
            output.append(pivot_token)
    return normalize_text(" ".join(output))


def _levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (token_a != token_b)))
        previous = current
    return previous[-1]


def word_medoid(hypotheses: list[str]) -> tuple[str, int]:
    """Return the hypothesis with the lowest mean word-error rate against its peers.

    Distances are computed on case-folded tokens -- a member that differs from
    its peers only in capitalisation is not a different hypothesis -- while the
    returned surface keeps the selected member's original casing.

    Ties are broken by input order, so the member ordering in the config is part
    of the contract: reordering members can change the output.
    """
    tokenized = [[t.casefold() for t in word_tokens(h)] for h in hypotheses]
    costs = [
        sum(_levenshtein(t, u) / max(1, len(t)) for k, u in enumerate(tokenized) if k != i)
        for i, t in enumerate(tokenized)
    ]
    selected = min(range(len(hypotheses)), key=lambda i: (costs[i], i))
    return normalize_text(hypotheses[selected]), selected


def read_surface(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty surface: {path}")
    identifiers = [str(r["ID"]) for r in rows]
    if len(set(identifiers)) != len(identifiers):
        duplicated = [k for k, v in Counter(identifiers).items() if v > 1]
        raise SystemExit(f"duplicate IDs in {path}: {duplicated[:5]}")
    return {str(r["ID"]): normalize_text(r["Target"]) for r in rows}


def read_keys(path: Path) -> dict[str, str]:
    """Profile key per clip, as matched by inference/ttia/match.py. Consumed
    ungated."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(r["ID"]): str(r["profile_key"]) for r in csv.DictReader(handle)}


def read_train_texts(path: Path) -> tuple[list[str], list[str]]:
    import pandas as pd
    # "speaker_key" is this parquet's own column name; it holds profile keys.
    frame = pd.read_parquet(path, columns=["speaker_key", "training_target_stable_raw"])
    profiles = frame["speaker_key"].astype(str).tolist()
    texts = ["" if v is None else str(v) for v in frame["training_target_stable_raw"]]
    return profiles, texts


def fuse_ttia(surfaces: list[dict[str, str]], identifiers: list[str],
              keys: dict[str, str], train_texts: Path,
              ) -> tuple[list[dict[str, str]], Counter[str]]:
    from inference.ttia.idiolect import BoundaryConfig, BoundaryModel, TextLanguageModel

    profiles, texts = read_train_texts(train_texts)
    boundary = BoundaryModel(profiles, texts)
    lm = TextLanguageModel(profiles, texts)
    bcfg = BoundaryConfig(scope="profile", minimum_support=10, preference_ratio=2.0)
    lm_args = {"lexical": True, "scope": "profile", "order": 1, "prior": 20.0}

    missing = [i for i in identifiers if i not in keys]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no matched profile key, "
                         f"first: {missing[:3]}")

    selections: Counter[str] = Counter()
    rows = []
    for identifier in identifiers:
        profile = keys[identifier]
        normalised = [boundary.normalize(s[identifier], profile, bcfg)
                      for s in surfaces]
        candidates = list(normalised)
        names = [f"member_{n}" for n in range(len(normalised))]
        candidates.append(conservative_word_rover(normalised))
        names.append("_rover")
        # An empty hypothesis scores as zero-cost text, so it would always
        # win. Score it only when every candidate is empty; the blank then
        # survives to the output check below, which fails the run.
        viable = [rank for rank, c in enumerate(candidates) if c.strip()] \
            or list(range(len(candidates)))
        scored = sorted((lm.score(candidates[rank], profile, **lm_args), rank)
                        for rank in viable)
        chosen = scored[0][1]
        selections[names[chosen]] += 1
        rows.append({"ID": identifier, "Target": normalize_text(candidates[chosen])})
    return rows, selections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rover", "medoid", "ttia"], required=True)
    parser.add_argument("--keys", type=Path,
                        help="ttia only: profile key per clip, "
                             "from inference/ttia/match.py")
    parser.add_argument("--train-texts", type=Path,
                        help="ttia only: parquet of per-profile training text "
                             "(speaker_key, training_target_stable_raw)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+",
                        help="per-model surfaces, strongest first (order is load-bearing)")
    args = parser.parse_args()

    if len({p.resolve() for p in args.sources}) != len(args.sources):
        raise SystemExit("the same surface was passed twice; that silently re-weights the fusion")
    if args.method == "ttia" and not (args.keys and args.train_texts):
        raise SystemExit("--method ttia needs --keys and --train-texts")

    surfaces = [read_surface(p) for p in args.sources]
    identifiers = list(surfaces[0])
    for path, surface in zip(args.sources[1:], surfaces[1:]):
        if set(surface) != set(identifiers):
            raise SystemExit(f"ID set mismatch between {args.sources[0]} and {path}")

    selections: Counter[int] = Counter()
    named_selections: Counter[str] = Counter()
    rows = []
    if args.method == "ttia":
        rows, named_selections = fuse_ttia(surfaces, identifiers,
                                           read_keys(args.keys), args.train_texts)
    else:
        for identifier in identifiers:
            hypotheses = [s[identifier] for s in surfaces]
            if args.method == "rover":
                target = conservative_word_rover(hypotheses)
            else:
                target, chosen = word_medoid(hypotheses)
                selections[chosen] += 1
            rows.append({"ID": identifier, "Target": target})

    blank = [r["ID"] for r in rows if not r["Target"].strip()]
    if blank:
        raise SystemExit(f"{len(blank)} blank fused target(s), first: {blank[:3]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "sources": [str(p) for p in args.sources],
        "rows": len(rows),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.method == "medoid":
        record["selection_counts"] = {str(args.sources[i]): selections.get(i, 0)
                                      for i in range(len(args.sources))}
    if args.method == "ttia":
        record["selection_counts"] = dict(sorted(named_selections.items()))
        record["keys"] = str(args.keys)
        record["train_texts"] = str(args.train_texts)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"fused {len(rows)} rows by {args.method} over {len(args.sources)} sources")
    if args.method == "medoid":
        for i, path in enumerate(args.sources):
            print(f"  selected {selections.get(i, 0):>4}  {path.name}")
    if args.method == "ttia":
        for name, count in sorted(named_selections.items()):
            print(f"  selected {count:>4}  {name}")
    print(f"sha256 {record['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
