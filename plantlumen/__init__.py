"""PlantLumen: reference-assisted prediction of spatial expression from plant
histology images.

The model is frozen: one set of input, retrieval, fusion and low-rank projection
rules is applied to every species, with all fitted state (feature transforms,
reference bank, expression subspace) estimated inside each species and each fold
from training sections only. Nothing is shared across species.
"""
from .core import Section, gene_pearson, predict_fold
from .run import run_cohort, select_panel

__version__ = "1.0.0"
__all__ = ["Section", "gene_pearson", "predict_fold", "run_cohort", "select_panel"]
