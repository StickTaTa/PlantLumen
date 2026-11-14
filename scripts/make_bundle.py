# -*- coding: utf-8 -*-
"""Export a cohort bundle for `plantlumen run` from the project's own caches.

This script is for the authors and for anyone who clones the full analysis
repository. It does not ship with the demo: the bundles carry per-spot image
descriptors, which are large. A reviewer who only wants to check the published
scores should use `python -m plantlumen verify`, which needs nothing but the
shipped per-fold artefacts.

What it writes, for every section of one cohort:
    xy       physical spot coordinates, micrometres
    expr     expression on the cohort's chosen scale, spots x genes
    genes    gene identifiers
    image    R50 patch descriptor (2048) + mean of the 12 nearest physical
             neighbours (2048) = 4096
    hand     63-d colour/texture descriptor
    stage_feat  768-d ConvNeXt-Tiny patch descriptor, used only by the stage
                branch (blend weight 0.45 when the held-out stage is absent)
    stage    developmental stage label, taken from the section tag

Example
    python scripts/make_bundle.py --project G:/PlantLumen --cohort grain \
        --out bundles/grain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, help="root of the analysis repository")
    ap.add_argument("--cohort", default="grain")
    ap.add_argument("--readout", default="k8")
    ap.add_argument("--readout-seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.project)
    sys.path.insert(0, str(root / "src" / "bench"))
    sys.path.insert(0, str(root / "src" / "bench" / "exp"))
    import protocol as P
    import exp95_plantlumen_v7 as V7
    import exp129_histo_feats as H9

    tags = P.COHORTS[args.cohort]
    secs = {t: P.load_section(t) for t in tags}
    xy = {t: P.phys_xy(secs[t]) for t in tags}
    raw = V7.get_features(args.cohort, secs)
    hand = H9.get_histo(args.cohort, secs)
    stage_cache = root / "bench" / "cache" / "cnn_feats_bb" / f"convnext_{args.cohort}.npz"
    cnx = (np.load(stage_cache, allow_pickle=True) if stage_cache.exists() else None)

    from sklearn.neighbors import NearestNeighbors
    out = Path(args.out)
    (out / "sec").mkdir(parents=True, exist_ok=True)
    for tag in tags:
        nb = NearestNeighbors(n_neighbors=V7.N_CTX + 1).fit(xy[tag])
        idx = nb.kneighbors(xy[tag])[1][:, 1:]
        # the shipped barley chain concatenates the 2048-d patch descriptor with
        # the mean descriptor of the 12 physical neighbours (all columns)
        image = np.hstack([raw[tag][:, :2048],
                           raw[tag][idx[:, :V7.N_CTX]].mean(1)]).astype(np.float32)
        payload = dict(
            tag=np.asarray(tag), stage=np.asarray(tag[:4]),
            xy=xy[tag].astype(np.float64),
            expr=np.asarray(secs[tag]["e"], np.float32),
            genes=np.asarray([str(g) for g in secs[tag]["g"]]),
            image=image, hand=np.asarray(hand[tag], np.float32))
        if cnx is not None and tag in cnx.files:
            payload["stage_feat"] = np.asarray(cnx[tag], np.float32).reshape(
                len(cnx[tag]), -1)
        np.savez_compressed(out / "sec" / f"{tag}.npz", **payload)
        print(f"  {tag}: {image.shape[0]} spots, {image.shape[1]} image dims"
              + (", stage_feat present" if "stage_feat" in payload else ""))
    (out / "cohort.json").write_text(json.dumps(dict(
        name=args.cohort, readout=args.readout, readout_seed=args.readout_seed,
        gate_weight=0.45 if cnx is not None else None,
        sections=list(tags)), indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
