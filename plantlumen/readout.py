"""Spatial readout rules.

A Visium spot is a 55 um disk, not a biological unit, so every score in the
paper is computed on averages of neighbouring spots. The readout is fixed by
the cohort before any model is fitted and is never chosen per section:

  k8     KMeans(8) on the physical coordinates (barley, Arabidopsis, Medicago)
  ring   three equal-count radial annuli inside each connected tissue piece
         (maize, whose slides carry several separate embryos)
  spot   no averaging; used only as the conservative cross-check

`seed` matters: the barley artefacts were partitioned with seed 42, the
cross-species maize/Arabidopsis runs with seed 0. Pass the cohort's seed so a
recomputation reproduces the stored block labels exactly.

The readout takes coordinates only. It never sees expression or labels, so it
cannot leak the target.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

MIN_BLOCK_SPOTS = 20
SEG_FACTOR = 1.8
READOUTS = ("spot", "k8", "ck8", "ring")


def segments(xy: np.ndarray, factor: float = SEG_FACTOR) -> np.ndarray:
    """Connected tissue pieces; one section returns one component unless it is
    physically cut in two."""
    xy = np.asarray(xy, np.float64)
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d, _ = nn.kneighbors(xy)
    r = float(factor * np.median(d[:, 1]))
    pairs = NearestNeighbors(radius=r).fit(xy).radius_neighbors(xy, return_distance=False)
    rows = np.repeat(np.arange(len(xy)), [len(p) for p in pairs])
    cols = np.concatenate(pairs)
    a = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    _, lab = connected_components(a, directed=False)
    return lab


def _keep(lab: np.ndarray, minspots: int = MIN_BLOCK_SPOTS) -> list[int]:
    return [int(q) for q in np.unique(lab) if q >= 0 and (lab == q).sum() >= minspots]


def blocks_kmeans(xy: np.ndarray, k: int = 8, seed: int = 42,
                  minspots: int = MIN_BLOCK_SPOTS):
    xy = np.asarray(xy, np.float64)
    lab = KMeans(n_clusters=int(min(k, max(2, len(xy) // 40))), n_init=10,
                 random_state=seed).fit_predict(xy)
    return lab, _keep(lab, minspots)


def blocks_ring(xy: np.ndarray, nring: int = 3, seed: int = 0,
                minspots: int = MIN_BLOCK_SPOTS):
    xy = np.asarray(xy, np.float64)
    seg = segments(xy)
    out = np.full(len(xy), -1)
    nxt = 0
    for c in np.unique(seg):
        m = np.where(seg == c)[0]
        if len(m) < minspots:
            continue
        cen = xy[m].mean(0)
        r = np.linalg.norm(xy[m] - cen, axis=1)
        q = np.quantile(r, [i / nring for i in range(1, nring)])
        out[m] = nxt + np.digitize(r, q)
        nxt += nring
    return out, _keep(out, minspots)


def build(name: str, xy: np.ndarray, seed: int = 42):
    if name == "spot":
        return np.arange(len(xy)), list(range(len(xy)))
    if name == "k8":
        return blocks_kmeans(xy, 8, seed)
    if name == "ring":
        return blocks_ring(xy, 3, seed)
    raise ValueError(f"unknown readout: {name}")
