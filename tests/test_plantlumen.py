"""Unit tests on synthetic sections; no data download required.

Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from plantlumen import core, readout as R, run


def synthetic(n=400, n_genes=60, seed=0):
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0, 4000, size=(n, 2))
    expr = rng.gamma(0.5, 1.0, size=(n, n_genes)).astype(np.float32)
    genes = [f"G{i:03d}" for i in range(n_genes)]
    image = rng.normal(size=(n, 64)).astype(np.float32)
    hand = rng.normal(size=(n, 63)).astype(np.float32)
    section = core.Section(tag="A", stage="1", xy=xy, expr=expr, genes=genes,
                           image=image, hand=hand,
                           stage_feat=rng.normal(size=(n, 32)).astype(np.float32))
    return section, rng


def test_zscore_columns_are_standardised():
    x = np.random.RandomState(0).normal(size=(50, 5))
    z = core.zscore(x)
    assert np.allclose(z.mean(0), 0, atol=1e-5)
    assert np.allclose(z.std(0), 1, atol=1e-4)


def test_gene_pearson_flags_flat_genes():
    y = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    p = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    r = core.gene_pearson(y, p)
    assert r[0] == pytest.approx(1.0)
    assert np.isnan(r[1])


def test_low_rank_project_is_deterministic():
    rng = np.random.RandomState(1)
    train = [rng.normal(size=(8, 40))]
    pred = rng.normal(size=(8, 40))
    a, ranks = core.low_rank_project(train, pred)
    b, _ = core.low_rank_project(train, pred)
    assert np.allclose(a, b)
    # a centred covariance with 8 rows supports at most 7 dimensions, so the
    # three nominal ranks all cap to 7 and collapse to a single projection
    assert ranks == (7,)


def test_readout_is_deterministic_and_coordinate_only():
    section, _ = synthetic()
    lab1, keep1 = R.build("k8", section.xy, seed=42)
    lab2, keep2 = R.build("k8", section.xy, seed=42)
    assert np.array_equal(lab1, lab2)
    assert keep1 == keep2
    assert len(keep1) == 8


def test_fold_runs_and_panel_is_training_only():
    sections = {}
    for i, tag in enumerate("ABC"):
        section, _ = synthetic(seed=i)
        sections[tag] = core.Section(tag=tag, stage=tag, xy=section.xy,
                                     expr=section.expr, genes=section.genes,
                                     image=section.image, hand=section.hand,
                                     stage_feat=section.stage_feat)
    panel = run.select_panel(sections, ["A", "B"])
    sub = {t: run.subset(s, panel) for t, s in sections.items()}
    result = core.predict_fold(sub, ["A", "B"], "C",
                               readout=lambda xy: R.build("k8", xy, 42))
    assert result["pred_blocks"].shape == result["true_blocks"].shape
    assert np.isfinite(result["r"])


def test_stage_branch_runs_only_when_the_stage_is_absent():
    secs = {}
    for i, tag in enumerate(["1_A", "1_B", "2_A"]):
        s, rng = synthetic(seed=i)
        secs[tag] = core.Section(tag=tag, stage=tag[:1], xy=s.xy, expr=s.expr,
                                 genes=s.genes, image=s.image, hand=s.hand,
                                 stage_feat=rng.normal(size=(len(s.xy), 32)))
    readout = lambda xy: R.build("k8", xy, 42)
    present = core.predict_fold(secs, ["1_A", "2_A"], "1_B", readout)
    assert present["stage_unseen"] is False and present["gate_weight"] == 0.0
    absent = core.predict_fold(secs, ["1_A", "1_B"], "2_A", readout)
    assert absent["stage_unseen"] is True
    assert absent["gate_weight"] == pytest.approx(core.GATE_WEIGHT)


def test_stage_branch_needs_stage_features():
    s, _ = synthetic()
    only = core.Section(tag="A", stage="1", xy=s.xy, expr=s.expr, genes=s.genes,
                        image=s.image, hand=s.hand)
    with pytest.raises(ValueError):
        core.member_b({"A": only, "B": only}, ["A"], "B",
                      np.zeros(len(s.xy), int), [0])


def test_stage_protocol_leaves_the_whole_stage_out():
    sections = {}
    for i, tag in enumerate(["1_A", "1_B", "2_A"]):
        section, _ = synthetic(seed=i)
        sections[tag] = core.Section(tag=tag, stage=tag[:1], xy=section.xy,
                                     expr=section.expr, genes=section.genes,
                                     image=section.image, hand=section.hand,
                                     stage_feat=section.stage_feat)
    rows = run.run_cohort(sections, dict(readout="k8", readout_seed=42),
                          protocol="stage")
    row = [r for r in rows if r["section"] == "2_A"][0]
    assert row["n_train"] == 2
