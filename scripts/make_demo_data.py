# -*- coding: utf-8 -*-
"""Export the browser demo's data from the frozen stage-absent predictions.

Reads runs/losto_stage_preds/stage/<tag>.npz (predicted and measured block
profiles, spot coordinates and block labels) and writes demo/data.json: per
section, the spot coordinates, block labels and, for a fixed gene list, the
predicted and measured block value of every spot.

Nothing is recomputed here; the values are the ones the paper scores. The page
therefore shows exactly what the model produced for a developmental stage that
had no reference section in training.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parents[3]
SRC = BASE / "runs" / "losto_stage_preds" / "stage"
TABLE = BASE / "bench" / "tables" / "barley_stage_application" / "pathways.tsv"
OUT = Path(__file__).resolve().parents[1] / "demo" / "data.json"

# Pre-declared barley grain panel; kept in the stored order so the page and the
# supplement list the same genes.
GENES = [
    "HORVU1Hr1G001120", "HORVU4Hr1G081640", "HORVU3Hr1G004140",
    "HORVU5Hr1G046550", "HORVU6Hr1G058570", "HORVU7Hr1G067620",
    "HORVU1Hr1G091600", "HORVU2Hr1G090980", "HORVU7Hr1G033230",
    "HORVU7Hr1G091250", "HORVU1Hr1G055140", "HORVU7Hr1G027860",
    "HORVU7Hr1G057470", "HORVU1Hr1G057570", "HORVU4Hr1G005800",
    "HORVU2Hr1G068670", "HORVU7Hr1G069460", "HORVU5Hr1G027910",
    "HORVU1Hr1G075560", "HORVU2Hr1G093690",
]
SYMBOL = {
    "HORVU1Hr1G001120": "B1 hordein", "HORVU4Hr1G081640": "CMd inhibitor",
    "HORVU3Hr1G004140": "Bowman-Birk", "HORVU5Hr1G046550": "LTP1",
    "HORVU6Hr1G058570": "GLTP1", "HORVU7Hr1G067620": "AGPase large",
    "HORVU1Hr1G091600": "AGPase large 2", "HORVU2Hr1G090980": "SS chloroplastic",
    "HORVU7Hr1G033230": "SUS", "HORVU7Hr1G091250": "alpha-amylase",
    "HORVU1Hr1G055140": "beta-amylase", "HORVU7Hr1G027860": "pullulanase",
    "HORVU7Hr1G057470": "ACP3", "HORVU1Hr1G057570": "ACP1",
    "HORVU4Hr1G005800": "oleosin", "HORVU2Hr1G068670": "ACPD",
    "HORVU7Hr1G069460": "p23-1", "HORVU5Hr1G027910": "Hsp90-6",
    "HORVU1Hr1G075560": "thioredoxin", "HORVU2Hr1G093690": "expansin",
}

# Mirrors PATHWAYS in src/bench/exp/exp359_barley_stage_program_application.py.
# The score at the end of main() is asserted against that script's table, so a
# drift between the two copies fails the build instead of quietly changing what
# the page shows.
PATHWAYS = [
    ("储藏蛋白 / 蛋白酶抑制剂", "storage protein / protease inhibitor",
     ["HORVU1Hr1G001120", "HORVU4Hr1G081640", "HORVU3Hr1G004140"]),
    ("淀粉合成", "starch biosynthesis",
     ["HORVU7Hr1G067620", "HORVU1Hr1G091600", "HORVU2Hr1G090980", "HORVU7Hr1G033230"]),
    ("淀粉动员", "starch mobilisation",
     ["HORVU7Hr1G091250", "HORVU1Hr1G055140", "HORVU7Hr1G027860"]),
    ("质体脂肪酸合成", "plastid fatty-acid synthesis",
     ["HORVU7Hr1G057470", "HORVU1Hr1G057570", "HORVU2Hr1G068670"]),
    ("伴侣 / 氧化还原", "chaperone / redox",
     ["HORVU7Hr1G069460", "HORVU5Hr1G027910", "HORVU1Hr1G075560"]),
    ("油体", "oil body", ["HORVU4Hr1G005800"]),
    ("细胞扩张", "cell expansion", ["HORVU2Hr1G093690"]),
]


def pathway_profile(tb, pb, panel, wanted):
    """Mean block profile over the pathway's scoreable genes.

    Mirrors exp359: the block values are already z-scores across spots for each
    gene, so the pathway profile is their plain average over the genes; a gene is
    scoreable when it is in the fold's panel and its measured block profile is
    non-constant, and a pathway needs at least two such genes.
    """
    cols = [panel.index(g) for g in wanted if g in panel]
    cols = [j for j in cols if tb[:, j].std() > 1e-12]
    if len(cols) < 2:
        return None
    tm = tb[:, cols].mean(1)
    pm = pb[:, cols].mean(1)
    rho = float(spearmanr(pm, tm).statistic)
    # the same 7 non-trivial cyclic rotations of the measured profile used in
    # the paper; the prediction is held fixed
    null = float(np.mean([spearmanr(pm, np.roll(tm, s)).statistic for s in range(1, 8)]))
    return dict(n_genes=len(cols), rho=rho, null=null,
                measured=[round(float(v), 3) for v in tm],
                predicted=[round(float(v), 3) for v in pm])


def check_against_table(sections) -> None:
    """Fail the build if the page's pathway scores drift from Table S53's file."""
    if not TABLE.exists():
        print(f"note: {TABLE.name} not found, skipping the table check")
        return
    want = {}
    with TABLE.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["protocol"] != "stage" or not row["rho"]:
                continue
            want[(row["section"], row["pathway"])] = float(row["rho"])
    for sec in sections:
        for en, got in sec["pathways"].items():
            key = (sec["tag"], en)
            if key in want and abs(got["rho"] - want[key]) > 1e-9:
                raise SystemExit(f"pathway score drift at {key}: "
                                 f"{got['rho']:.6f} vs table {want[key]:.6f}")
    print(f"pathway scores match {TABLE.name} for all {len(want)} (section, pathway) rows")


def main() -> None:
    sections = []
    for file in sorted(SRC.glob("*.npz")):
        z = np.load(file, allow_pickle=True)
        tag = file.stem
        genes = [str(g) for g in z["genes"]]
        xy = np.asarray(z["xy"], float)
        labels = np.asarray(z["block_id"], int)
        keep = [int(b) for b in z["kept_blocks"]]
        tb = np.asarray(z["true_blocks"], float)
        pb = np.asarray(z["pred_blocks"], float)
        by_block = {b: i for i, b in enumerate(keep)}
        values = {}
        for gene in GENES:
            if gene not in genes:
                continue
            j = genes.index(gene)
            t, p = tb[:, j], pb[:, j]
            if t.std() < 1e-12:
                values[gene] = dict(r=None, measured=None, predicted=None)
                continue
            r = float(pearsonr(t, p).statistic) if t.std() > 0 and p.std() > 0 else None
            values[gene] = dict(r=r,
                                measured=[round(float(v), 3) for v in t],
                                predicted=[round(float(v), 3) for v in p])
        pathways = {}
        for _zh, en, wanted in PATHWAYS:
            got = pathway_profile(tb, pb, genes, wanted)
            if got is not None:
                pathways[en] = got
        sections.append(dict(
            tag=tag,
            stage=tag[:4],
            x=[round(float(v), 0) for v in xy[:, 0] - xy[:, 0].mean()],
            y=[round(float(v), 0) for v in xy[:, 1] - xy[:, 1].mean()],
            block=[int(v) for v in labels],
            keep=keep,
            genes=values,
            pathways=pathways))
    check_against_table(sections)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(
        species="barley",
        note="predicted and measured expression in a developmental stage absent "
             "from training; colour = predicted or measured block z-score",
        symbol=SYMBOL, genes=GENES,
        pathways=[dict(name=zh, key=en, genes=wanted) for zh, en, wanted in PATHWAYS],
        sections=sections),
        ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(payload, encoding="utf-8")
    # inline the same payload into the page so it opens from the filesystem
    # without a server and without a CORS error
    template = (OUT.parent / "template.html").read_text(encoding="utf-8")
    page = template.replace("DATA_PLACEHOLDER", payload)
    (OUT.parent / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(sections)} sections)")
    print(f"wrote {OUT.parent / 'index.html'}")


if __name__ == "__main__":
    main()
