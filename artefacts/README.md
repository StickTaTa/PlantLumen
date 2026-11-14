# Shipped artefacts

## Current manuscript summaries (`summary/`)

| Path | What it is |
|---|---|
| `summary/barley_summary.json` | barley U1 benchmark summary, current manuscript |
| `summary/maize_summary.json` | maize cohort summary, current manuscript |
| `summary/arabidopsis_summary.json` | Arabidopsis cohort summary, current manuscript |
| `summary/spruce_bench_summary.tsv` | Norway spruce benchmark summary, current manuscript |
| `summary/image_r2_cohort_summary.txt` | maize and Arabidopsis image-R² summaries, current manuscript |

These are derived tables. They do not contain raw expression matrices or image features.

## Legacy stage-application artefacts

These files let `plantlumen verify` and `scripts/report_s53.py` run with no
download and no image processing. They are the legacy 300-gene stage-application
demonstration, not the current U1 four-cohort benchmark.

| Path | What it is |
|---|---|
| `results/grain/ours_v15/fold*/<section>.npz` | the published per-fold predictions for the nine held-out barley sections: `spot_est` and `spot_true` (per-spot field x 300 panel genes), `score_genes` (the per-fold competitor gene intersection the benchmark is scored on), `pred_blocks` and `true_blocks` (their 8-block means), `genes`, `block_id`, `xy`, `r_block`, `w_star` (stage-branch weight) |
| `barley_stage_application/folds.tsv` | per-fold application scores under both protocols (`section`, `stage`) with the `stage_absent` flag |
| `barley_stage_application/genes.tsv` | the same for individual panel genes |
| `barley_stage_application/pathways.tsv` | the per-spot pathway x fold table behind the pathway application numbers |

`verify` scores `spot_est` against `spot_true` across the individual spots of
each held-out section, averaged over `score_genes`; that is the manuscript's
per-spot benchmark, mean 0.397524. The block arrays are kept because the block
means are what the shipped model originally returned, and every published block
value must remain checkable from the same file; they are not the manuscript's
scored readout for barley.

Provenance: `bench/results/grain/ours_v15/` and
`bench/tables/barley_stage_application/` in the analysis repository, produced by
`exp181_stage_gate.py` (chain) and `exp359_barley_stage_program_application.py`
(pathway application). The npz files are byte copies.
