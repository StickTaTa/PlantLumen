"""Cohort bundles.

A bundle is a directory holding one compressed npz per section plus a
`cohort.json` that fixes the readout rule. It carries no model weights and no
fitted state: everything a prediction needs is fitted at run time from the
training sections, which is why a bundle can be shared without leaking a fold.

Bundle layout
    mycohort/
      cohort.json
      sec/3DAP_A.npz     xy, expr, genes, image, hand, stage
      sec/...

`image` is the R50 patch descriptor (2048-d) concatenated with the mean
descriptor of the spot's 12 physical nearest neighbours (2048-d), so 4096-d.
`hand` is the 63-d colour/texture descriptor. Both are computed from the image
only; see scripts/make_bundle.py for how the shipped bundles were produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import Section


def load_bundle(path: str | Path) -> tuple[dict[str, Section], dict]:
    root = Path(path)
    config = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    sections: dict[str, Section] = {}
    for file in sorted((root / "sec").glob("*.npz")):
        z = np.load(file, allow_pickle=True)
        tag = str(z["tag"])
        sections[tag] = Section(
            tag=tag,
            stage=str(z["stage"]),
            xy=np.asarray(z["xy"], np.float64),
            expr=np.asarray(z["expr"], np.float32),
            genes=[str(g) for g in z["genes"]],
            image=np.asarray(z["image"], np.float32),
            hand=np.asarray(z["hand"], np.float32),
            stage_feat=(np.asarray(z["stage_feat"], np.float32)
                        if "stage_feat" in z.files else None),
        )
    if not sections:
        raise FileNotFoundError(f"no sections under {root / 'sec'}")
    return sections, config
