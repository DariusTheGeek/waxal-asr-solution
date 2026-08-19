#!/usr/bin/env python3
"""Score decoded validation surfaces against the held-out validation manifest.

Reports CER, WER and Q under the training selection contract: CER pooled over
the corpus and case-sensitive, WER pooled over the corpus and case-folded,

    Q = 1 - (CER + WER) / 2

This is the metric the released checkpoints were selected on. It is *not* the
leaderboard metric, which averages WER per utterance; only the Q combination
step is shared. The scoring here is unweighted -- every validation row counts
once -- while model selection during training used a weighted variant of the
same metric.

Hypotheses are joined to references by audio-file stem: decode surfaces are
keyed by the stem of `derived_audio_relpath`, not by the manifest `id`.

Usage
-----
python tools/score_validation.py \\
    --manifest data/derived/omniasr/lin_cv002_supervised_v1/manifests/dev.rows.parquet \\
    --surfaces outputs/val_metrics/decodes/*.val.csv \\
    --output outputs/val_metrics/validation_scores.json
"""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import pandas as pd


def raw_text(text: object, *, casefold: bool = False) -> str:
    """Declared raw-metric normalisation: NFC, optional casefold, collapse space."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFC", value)
    if casefold:
        value = value.casefold()
    return " ".join(value.split())


def score_texts(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Pooled CER (case-sensitive) and pooled WER (case-folded), plus Q."""
    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be non-empty and aligned")
    cer = float(jiwer.cer([raw_text(r) for r in references],
                          [raw_text(h) for h in hypotheses]))
    wer = float(jiwer.wer([raw_text(r, casefold=True) for r in references],
                          [raw_text(h, casefold=True) for h in hypotheses]))
    return {"cer": cer, "wer": wer, "score": 1.0 - 0.5 * (cer + wer)}


def read_surface(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty surface: {path}")
    return {str(r["ID"]): r["Target"] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="validation manifest with derived_audio_relpath "
                             "and transcription_nfc")
    parser.add_argument("--surfaces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest,
                            columns=["derived_audio_relpath", "transcription_nfc"])
    reference = {Path(p).stem: t for p, t in
                 zip(frame["derived_audio_relpath"], frame["transcription_nfc"])}

    results = {}
    print(f"{'surface':<24} {'rows':>5} {'CER':>8} {'WER':>8} {'Q':>8} {'blank':>7}")
    for path in sorted(args.surfaces):
        surface = read_surface(path)
        missing = [k for k in surface if k not in reference]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} hypotheses have no reference, "
                             f"first: {missing[:3]}")
        keys = sorted(surface)
        if len(keys) != len(reference):
            raise SystemExit(f"{path}: {len(keys)} rows against "
                             f"{len(reference)} references")
        scored = score_texts([reference[k] for k in keys], [surface[k] for k in keys])
        # A blank hypothesis is real model output here, not a pipeline fault --
        # these are single-model surfaces, and the fusion stage is what repairs
        # them downstream. Report it rather than let it pass unremarked: it
        # scores as a whole-utterance deletion.
        blank = [k for k in keys if not surface[k].strip()]
        name = path.name.removesuffix(".val.csv")
        results[name] = {"rows": len(keys), "blank_hypotheses": len(blank), **scored}
        print(f"{name:<24} {len(keys):>5} {scored['cer']:>8.4f} "
              f"{scored['wer']:>8.4f} {scored['score']:>8.4f} {len(blank):>7}")
        if blank:
            print(f"    {len(blank)} blank hypothesis/es, scored as deletions: {blank[:3]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "metric": "pooled case-sensitive CER, pooled case-folded WER, "
                  "Q = 1 - (CER + WER) / 2; unweighted over all validation rows",
        "surfaces": results,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
