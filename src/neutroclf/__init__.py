"""neutroclf: interpretable and calibrated classifier for co-occurring neutrophil abnormalities.

The model grades four named morphological features (nuclear segmentation, chromatin
hypercondensation, granularity, Doehle bodies) and assigns one or more of seven
categories, with per-category temperature scaling, a split-conformal shortlist and a
confidence ranking for expert review.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
