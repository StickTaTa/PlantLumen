"""Fold execution and the published target definition.

Two rules, both fixed before any model is fitted:

  panel   the 300 genes with the largest total count over the *training*
          sections. Marker loci can be forced in for application runs; the
          benchmark panel is never altered.
  target  log1p(CPM) within the panel, library size computed from the panel
          columns of that spot.

A fold then fits the readout and the retrieval bank on the training sections,
predicts the held-out section and scores per-gene Pearson on the readout rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import core, readout as R

CPM_SCALE = 1e4


def select_panel(sections: dict[str, "core.Section"], train_tags: list[str],
                 k: int = core.K_PANEL, force: list[str] | None = None) -> list[str]:
    """Top-k genes by total count over the training spots, plus any forced loci."""
    totals: dict[str, float] = {}
    for tag in train_tags:
        section = sections[tag]
        sums = section.expr.sum(0)
        for gene, total in zip(section.genes, sums):
            totals[gene] = totals.get(gene, 0.0) + float(total)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    panel = [gene for gene, _ in ranked[:k]]
    have = set(panel)
    for gene in force or []:
        if gene in totals and gene not in have:
            panel.append(gene)
            have.add(gene)
    return panel


def subset(section: "core.Section", panel: list[str]) -> "core.Section":
    """Project a section onto the panel and apply the log1p(CPM) transform."""
    index = {g: i for i, g in enumerate(section.genes)}
    cols = [index[g] for g in panel if g in index]
    where = {g: i for i, g in enumerate(panel)}
    raw = np.zeros((section.expr.shape[0], len(panel)), np.float32)
    for local, col in enumerate(cols):
        raw[:, where[section.genes[col]]] = section.expr[:, col]
    lib = raw.sum(1, keepdims=True) + 1.0
    expr = np.log1p(raw / lib * CPM_SCALE).astype(np.float32)
    return core.Section(tag=section.tag, stage=section.stage, xy=section.xy,
                        expr=expr, genes=list(panel),
                        image=section.image, hand=section.hand,
                        stage_feat=section.stage_feat)


def run_cohort(sections: dict[str, "core.Section"], config: dict,
               protocol: str = "section", force: list[str] | None = None) -> list[dict]:
    """protocol "section" leaves one section out; "stage" leaves one stage out."""
    tags = list(sections)
    readout = config.get("readout", "k8")
    seed = int(config.get("readout_seed", 42))
    out = []
    for tag in tags:
        if protocol == "section":
            train = [t for t in tags if t != tag]
        elif protocol == "stage":
            train = [t for t in tags if sections[t].stage != sections[tag].stage]
        else:
            raise ValueError(protocol)
        panel = select_panel(sections, train, force=force)
        sub = {t: subset(sections[t], panel) for t in train + [tag]}
        result = core.predict_fold(
            sub, train, tag, readout=lambda xy: R.build(readout, xy, seed),
            gate_weight=config.get("gate_weight"))
        out.append(dict(section=tag, stage=sections[tag].stage, n_train=len(train),
                        protocol=protocol, r=result["r"], ranks=result["ranks"],
                        pred_blocks=result["pred_blocks"],
                        true_blocks=result["true_blocks"], genes=panel,
                        labels=result["labels"], keep=result["keep"]))
    return out


def save_folds(rows: list[dict], outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        np.savez_compressed(
            out / f"{row['section']}.npz",
            pred_blocks=row["pred_blocks"].astype(np.float32),
            true_blocks=row["true_blocks"].astype(np.float32),
            genes=np.asarray(row["genes"]), labels=row["labels"].astype(np.int32),
            kept_blocks=np.asarray(row["keep"], np.int32),
            stage=np.asarray(row["stage"]), protocol=np.asarray(row["protocol"]),
            r=np.float64(row["r"]))
