"""Reproduce the published scores from the shipped per-fold artefacts.

Two things are checked, and both fail loudly:

  1. the grid. The stored predicted and measured spot fields must have one row
     per spot of the stored coordinate array and one column per stored gene,
     and the scored gene set must be a subset of those genes.
  2. the score. Per-gene Pearson recomputed across the individual spots of each
     held-out section, averaged over the shipped score gene set, must equal the
     shipped per-section score and the shipped nine-fold mean to 1e-4.

This does not re-run the image encoder. It checks the half of the chain that a
reviewer can check without the raw slides; `plantlumen.run` re-runs the other
half when a bundle is supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

# Shipped per-section scores on the per-spot readout, fold-common gene set.
EXPECTED = {
    "grain": {
        "dir": "results/grain/ours_v15",
        "sections": {
            "3DAP_A": 0.462259, "3DAP_B": 0.470368, "3DAP_C": 0.508709,
            "3DAP_D": 0.511806, "6DAP_A": 0.401159, "6DAP_B": 0.322200,
            "6DAP_D": 0.384088, "12DAP_A": 0.238466, "20DAP_A": 0.278659,
        },
        "mean": 0.397524,
    },
}


def check_cohort(bench_root: Path, name: str, tol: float = 1e-4) -> list[str]:
    spec = EXPECTED[name]
    root = bench_root / spec["dir"]
    problems: list[str] = []
    scores = {}
    for file in sorted(root.glob("fold*/*.npz")):
        z = np.load(file, allow_pickle=True)
        tag = file.stem
        missing = [k for k in ("spot_est", "spot_true", "score_genes", "genes", "xy")
                   if k not in z.files]
        if missing:
            problems.append(f"{tag}: missing {', '.join(missing)}")
            continue
        genes = [str(g) for g in z["genes"]]
        true = np.asarray(z["spot_true"], np.float64)
        pred = np.asarray(z["spot_est"], np.float64)
        n_spots = len(np.asarray(z["xy"]))
        if true.shape != pred.shape or pred.shape != (n_spots, len(genes)):
            problems.append(f"{tag}: field shape {pred.shape} != ({n_spots}, {len(genes)})")
            continue
        wanted = [str(g) for g in z["score_genes"]]
        if not set(wanted).issubset(set(genes)):
            problems.append(f"{tag}: score genes are not a subset of the panel")
            continue
        cols = [genes.index(g) for g in wanted]
        r = core.gene_pearson(true[:, cols], pred[:, cols])
        scores[tag] = float(np.nanmean(r))
        if tag in spec["sections"]:
            if abs(scores[tag] - spec["sections"][tag]) > tol:
                problems.append(
                    f"{tag}: score {scores[tag]:.6f} != shipped "
                    f"{spec['sections'][tag]:.6f}")
    mean = float(np.mean(list(scores.values())))
    if abs(mean - spec["mean"]) > tol:
        problems.append(f"mean {mean:.6f} != shipped {spec['mean']:.6f}")
    print(f"{name}: {len(scores)} folds, mean {mean:.6f} (shipped {spec['mean']:.6f})")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="bench",
                    help="path to a directory holding results/<cohort>/ (or the "
                         "artefacts directory shipped with the package)")
    ap.add_argument("--cohort", default="grain", choices=sorted(EXPECTED))
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args(argv)
    bench = Path(args.bench)
    if not (bench / "results").exists() and (bench / "artefacts" / "results").exists():
        bench = bench / "artefacts"
    problems = check_cohort(bench, args.cohort)
    if args.report:
        Path(args.report).write_text(json.dumps(
            dict(cohort=args.cohort, problems=problems), indent=1), encoding="utf-8")
    if problems:
        print("FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: grids and scores reproduce the shipped artefacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
