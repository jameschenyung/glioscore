"""Glioblastoma intratumoral heterogeneity pipeline.

Load a whole-slide image, keep the stained tissue, embed patches with a
pretrained vision model, cluster morphological niches, and summarize how
mixed those niches are with a single entropy score.
"""

__version__ = "0.1.0"
