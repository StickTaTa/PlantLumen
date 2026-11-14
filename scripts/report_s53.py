# -*- coding: utf-8 -*-
"""Recompute the pathway-application numbers from the shipped per-fold table.

Reads artefacts/barley_stage_application/pathways.tsv and prints:

  * per-pathway mean Spearman rho on the stage-absent folds (3 and 6 DAP
    sections, where the held-out developmental stage is absent from training),
  * the same-stage reference and the stage-absent aggregate over identical
    (pathway, section) combinations, and the retained fraction.

Nothing is refitted here: this checks the aggregation that the manuscript
reports, so a reader can confirm that the numbers follow from the per-fold file.
The scores are per-spot Spearman correlations between predicted and measured
pathway profiles.

Usage
  python scripts/report_s53.py [--artefacts artefacts]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import csv

ABSENT_COL = "stage_absent"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artefacts", default=str(Path(__file__).resolve().parent.parent
                                              / "artefacts"))
    args = ap.parse_args()
    root = Path(args.artefacts)
    rows = read_rows(root / "barley_stage_application" / "pathways.tsv")
    scored = [r for r in rows if r["rho"] not in ("", None)]

    absent = [r for r in scored if r[ABSENT_COL] == "1"]
    same_stage = [r for r in scored
                  if r["protocol"] == "section" and r["section"] not in ("12DAP_A", "20DAP_A")]

    print("pathway                                       genes  n  mean rho  min rho")
    order = ["starch mobilisation", "chaperone / redox", "plastid fatty-acid synthesis",
             "starch biosynthesis", "storage protein / protease inhibitor",
             "oil body", "cell expansion"]
    for pathway in order:
        part = [r for r in absent if r["pathway"] == pathway]
        if not part:
            print(f"{pathway:45s} {'0':>5s}  0        --       --")
            continue
        rho = [float(r["rho"]) for r in part]
        sizes = [int(r["n_genes"]) for r in part]
        genes = (str(max(sizes)) if min(sizes) == max(sizes)
                 else f"{min(sizes)}-{max(sizes)}")
        print(f"{pathway:45s} {genes:>5s} {len(part):2d} "
              f"{sum(rho) / len(rho):+9.3f} {min(rho):+8.3f}")

    a = [float(r["rho"]) for r in absent]
    s = [float(r["rho"]) for r in same_stage]
    print()
    print(f"stage absent      n={len(a):3d}  mean rho {sum(a) / len(a):+.4f}")
    print(f"same-stage ref.   n={len(s):3d}  mean rho {sum(s) / len(s):+.4f}  "
          f"(same 7 sections)")
    print(f"retained {100 * (sum(a) / len(a)) / (sum(s) / len(s)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
