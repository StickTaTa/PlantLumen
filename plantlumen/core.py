"""The frozen PlantLumen chain, in one place.

Every function here is a direct port of the code that produced the published
numbers (`src/bench/exp/exp321_unified_barley_v15_cross_species.py` and
`src/bench/exp/exp345_losto_stage_transfer.py`). The arithmetic is unchanged;
only names and packaging differ. If you change anything in this file the
shipped numbers stop being reproducible, which `plantlumen.verify` will tell
you.

The chain, in order
  1. z-score each feature block over the spots of each section
  2. PCA to 768 dimensions, fitted on training spots only
  3. cosine k-nearest-neighbour retrieval of expression, k=21, in a CORAL
     aligned space, blended with a raw-space k=9 retrieval for the HAND route
  4. fuse image route and HAND route at 0.80 / 0.20
  5. aggregate to the cohort's spatial readout (8 blocks or per-embryo rings)
  6. project the block prediction onto the rank-{16,20,32} principal subspace
     of the training block expression and average the three projections
  7. if the held-out developmental stage is absent from every training section,
     blend that prediction 0.55/0.45 with a pooled spot-level ridge on ConvNeXt
     patch features; otherwise the branch has weight zero

Nothing in the chain is fitted on the held-out section: the PCA, the CORAL
frame, the retrieval bank and the expression subspace all come from training
sections of the same species only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

SEED = 42
K_PANEL = 300
N_CTX = 12
K_RAW = 9
K_COR = 21
W_RAW = 0.5
FEATURE_DIM = 768
RANKS = (16, 20, 32)
W_IMAGE = 0.80
W_HAND = 0.20
# stage-coverage branch (exp181): only used when the held-out stage is absent
GATE_WEIGHT = 0.45
GATE_DIM = 64
GATE_ALPHA = 30.0
GATE_SEED = 0


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def knn_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(xtr)), metric="cosine").fit(xtr)
    d, idx = nn.kneighbors(xte)
    w = 1.0 / (d + 1e-6)
    w /= w.sum(1, keepdims=True)
    return np.einsum("nk,nkg->ng", w, ytr[idx])


def _eigh(x: np.ndarray, eps: float = 1e-3):
    xc = x - x.mean(0)
    cov = np.cov(xc, rowvar=False) + np.eye(x.shape[1]) * eps
    values, vectors = np.linalg.eigh(cov)
    return np.maximum(values, 1e-8), vectors


def whiten_frame(x: np.ndarray) -> np.ndarray:
    values, vectors = _eigh(x)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def coral_to(train_features: list[np.ndarray], test_features: np.ndarray,
             train_frame: np.ndarray):
    def convert(x):
        values, vectors = _eigh(x)
        root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
        return ((x - x.mean(0)) @ root) @ train_frame

    return [convert(x) for x in train_features], convert(test_features)


def retrieve(features: dict[str, np.ndarray], train_tags: list[str], test_tag: str,
             y_train: np.ndarray, k_raw: int | None, k_cor: int | None) -> np.ndarray:
    """Cosine retrieval fitted on training sections only.

    k_raw=k_cor=None is not valid; both routes always supply at least one k.
    When both are given the raw and CORAL predictions are averaged 50/50, which
    is the frozen rule for the HAND route.
    """
    standardized = {t: zscore(features[t]) for t in train_tags + [test_tag]}
    all_train = np.vstack([standardized[t] for t in train_tags])
    dim = min(FEATURE_DIM, all_train.shape[1], len(all_train) - 1)
    pca = PCA(n_components=int(dim), svd_solver="randomized", random_state=SEED).fit(all_train)
    projected = {t: pca.transform(standardized[t]) for t in train_tags + [test_tag]}
    x_train = np.vstack([projected[t] for t in train_tags])
    preds = []
    if k_raw is not None:
        preds.append(knn_predict(x_train, y_train, projected[test_tag], k_raw))
    if k_cor is not None:
        frame = whiten_frame(x_train)
        aligned_train, aligned_test = coral_to(
            [projected[t] for t in train_tags], projected[test_tag], frame)
        preds.append(knn_predict(np.vstack(aligned_train), y_train, aligned_test, k_cor))
    if len(preds) == 1:
        return preds[0]
    return W_RAW * preds[0] + (1.0 - W_RAW) * preds[1]


def block_means(values: np.ndarray, labels: np.ndarray, keep) -> np.ndarray:
    return np.stack([values[labels == q].mean(0) for q in keep])


def member_b(sections: dict[str, "Section"], train_tags: list[str], test_tag: str,
             labels: np.ndarray, keep) -> np.ndarray:
    """Pooled spot-level ridge on ConvNeXt patch features (the stage branch).

    Fitted exactly as in exp181: features z-scored per section, PCA-64 fitted on
    training spots only, ridge alpha=30 onto the untransformed log1p(CPM) target,
    prediction averaged over the readout rows.
    """
    missing = [t for t in train_tags + [test_tag] if sections[t].stage_feat is None]
    if missing:
        raise ValueError("stage branch needs stage_feat for " + ", ".join(missing))
    z = {t: zscore(sections[t].stage_feat) for t in train_tags + [test_tag]}
    all_train = np.vstack([z[t] for t in train_tags])
    dim = int(min(GATE_DIM, all_train.shape[1], len(all_train) - 1))
    pca = PCA(n_components=dim, svd_solver="randomized",
              random_state=GATE_SEED).fit(all_train)
    x_train = np.vstack([pca.transform(z[t]) for t in train_tags])
    y_train = np.vstack([sections[t].expr for t in train_tags])
    fit = Ridge(alpha=GATE_ALPHA).fit(x_train, y_train)
    spot_pred = fit.predict(pca.transform(z[test_tag]))
    return block_means(spot_pred, labels, keep)


def low_rank_project(train_block_y: list[np.ndarray], test_block_prediction: np.ndarray):
    y = np.vstack(train_block_y)
    covariance = np.cov(y, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    max_supported = min(y.shape[1], max(1, len(y) - 1))
    ranks = tuple(sorted(set(min(k, max_supported) for k in RANKS)))
    projected = []
    for rank in ranks:
        basis = vectors[:, -rank:]
        projected.append((test_block_prediction @ basis) @ basis.T)
    return np.mean(projected, axis=0), ranks


def gene_pearson(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-gene Pearson across the readout's rows; NaN where either is flat."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    yc, pc = y - y.mean(0), p - p.mean(0)
    den = np.sqrt((yc * yc).sum(0) * (pc * pc).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (yc * pc).sum(0) / den
    return np.where((den > 1e-12) & np.isfinite(r), r, np.nan)


@dataclass
class Section:
    """One spatial section: spots, expression and the two image feature routes."""

    tag: str
    stage: str
    xy: np.ndarray            # (n_spots, 2) physical coordinates, micrometres
    expr: np.ndarray          # (n_spots, n_genes) expression, already on the chosen scale
    genes: list[str]
    image: np.ndarray         # (n_spots, 1536) R50 patch + 12-neighbour context
    hand: np.ndarray          # (n_spots, 63) colour/texture description
    labels: np.ndarray | None = None   # readout labels; built by readout.build if None
    stage_feat: np.ndarray | None = None  # (n_spots, 768) ConvNeXt patch features


def predict_fold(sections: dict[str, Section], train_tags: list[str], test_tag: str,
                 readout, gate_weight: float | None = None) -> dict:
    """Run one leave-out fold. Returns block predictions in the panel's gene order."""
    test = sections[test_tag]
    genes = test.genes
    expr = {t: zscore(sections[t].expr) for t in train_tags + [test_tag]}
    y_train = np.vstack([expr[t] for t in train_tags])

    image_route = {t: sections[t].image for t in train_tags + [test_tag]}
    hand_route = {t: sections[t].hand for t in train_tags + [test_tag]}
    core = retrieve(image_route, train_tags, test_tag, y_train, None, K_COR)
    hand = retrieve(hand_route, train_tags, test_tag, y_train, K_RAW, K_COR)
    spot_pred = W_IMAGE * core + W_HAND * hand

    train_block_y = []
    for t in train_tags:
        lab, keep = readout(sections[t].xy)
        train_block_y.append(block_means(expr[t], lab, keep))
    labels, keep = readout(test.xy)
    target = block_means(expr[test_tag], labels, keep)
    raw = block_means(spot_pred, labels, keep)
    pred, ranks = low_rank_project(train_block_y, raw)

    # the stage branch is applied at block level, after the low-rank projection,
    # and only when no training section belongs to the held-out stage
    unseen = test.stage not in {sections[t].stage for t in train_tags}
    weight = GATE_WEIGHT if gate_weight is None else float(gate_weight)
    gate = weight if unseen else 0.0
    if gate > 0.0:
        pred = (1.0 - gate) * pred + gate * member_b(
            sections, train_tags, test_tag, labels, keep)

    return dict(pred_blocks=pred, true_blocks=target, labels=labels, keep=list(keep),
                ranks=ranks, gate_weight=gate, stage_unseen=unseen,
                r=float(np.nanmean(gene_pearson(target, pred))))
