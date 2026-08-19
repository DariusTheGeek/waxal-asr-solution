#!/usr/bin/env python3
"""Compare freshly decoded per-model surfaces against reference decodes.

GPU decoding is not bit-exact across hardware and driver versions, so this
reports agreement rather than asserting equality. What it is really checking is
that each model was invoked the way it was invoked when the reference was
produced -- beam width, length normalisation, language conditioning, checkpoint.
A configuration mistake shows up here as a large disagreement; hardware
nondeterminism shows up as a small one.

That distinction is the point: a decode run with the wrong language
conditioning, beam width or checkpoint disagrees far more than hardware
nondeterminism ever does -- invisible without a reference to compare against.

References are not shipped with this repository; point --references at a JSON
file describing your own reference decodes:

    {"surfaces": [{"surface": "lin_llm3b", "reference": "/path/to/ref.csv"}]}

Usage
-----
python tools/verify_surfaces.py --references my_references.json
"""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Below this, treat the surface as misconfigured rather than merely nondeterministic.
AGREEMENT_FLOOR = 0.95


# Deliberate standalone copy of inference/text.py's normalisation, so this
# verifier runs with no dependency on the package layout it is checking.
def normalise(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


# Reference surfaces were written by several tools over time and do not share
# one schema. Detect rather than demand.
SCHEMAS = [("ID", "Target"), ("id", "hypothesis"), ("id", "text"), ("ID", "target")]


def load(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty surface: {path}")
    columns = set(rows[0])
    for id_field, text_field in SCHEMAS:
        if {id_field, text_field} <= columns:
            break
    else:
        raise SystemExit(f"{path}: unrecognised schema {sorted(columns)}")
    return {str(r[id_field]): normalise(r[text_field]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True,
                        help="JSON describing reference decodes (see module docstring)")
    parser.add_argument("--decodes", type=Path, default=ROOT / "outputs/decodes")
    parser.add_argument("--floor", type=float, default=AGREEMENT_FLOOR)
    args = parser.parse_args()

    spec = json.loads(args.references.read_text())
    results, low = [], []
    print(f"{'surface':<18} {'rows':>5} {'identical':>10} {'agreement':>10}  status")
    for entry in spec["surfaces"]:
        if entry.get("from") == "route":
            # Stage 0 decodes both languages in one pass; the references are
            # split per language, so compare on the IDs they share.
            mine_path = args.decodes.parent / "route/joint_hypotheses.csv"
        else:
            mine_path = args.decodes / f"{entry['surface']}.csv"
        ref_path = Path(entry["reference"])
        if not mine_path.is_file():
            print(f"{entry['surface']:<18} {'':>5} {'':>10} {'':>10}  not decoded yet")
            continue
        if not ref_path.is_file():
            print(f"{entry['surface']:<18} {'':>5} {'':>10} {'':>10}  no reference on disk")
            continue
        mine = load(mine_path)
        ref = load(ref_path)
        common = set(mine) & set(ref)
        if not common:
            print(f"{entry['surface']:<18} {'':>5} {'':>10} {'':>10}  no shared IDs")
            continue
        same = sum(1 for k in common if mine[k] == ref[k])
        rate = same / len(common)
        gating = entry.get("gating", True)
        if rate >= args.floor:
            status = "ok"
        elif gating:
            status = "CHECK CONFIG"
            low.append((entry["surface"], rate))
        else:
            # Reference predates the toolchain this repository pins; report it,
            # do not fail on it.
            status = f"informational ({entry.get('decoder', '?')})"
        print(f"{entry['surface']:<18} {len(common):>5} {same:>10} {rate:>9.2%}  {status}")
        results.append({"surface": entry["surface"], "reference": str(ref_path),
                        "decoder": entry.get("decoder"),
                        "gating": gating, "rows": len(common),
                        "identical": same, "agreement": rate})

    out = args.decodes.parent / "surface_verification.json"
    out.write_text(json.dumps({"floor": args.floor, "results": results},
                              indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    if low:
        print("\nBelow the agreement floor -- check the decode configuration, not the hardware:")
        for name, rate in low:
            print(f"  {name}: {rate:.2%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
