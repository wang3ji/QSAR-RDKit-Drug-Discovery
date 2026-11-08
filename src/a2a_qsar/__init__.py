"""Lab-in-the-loop QSAR for the adenosine A2A receptor.

A reproducible pipeline that models A2A receptor ligand potency, quantifies
the uncertainty of its own predictions, and simulates an iterative screening
campaign in which the assay feeding the model is itself imperfect.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "active",
    "assay",
    "data",
    "evaluate",
    "features",
    "models",
    "splits",
]
