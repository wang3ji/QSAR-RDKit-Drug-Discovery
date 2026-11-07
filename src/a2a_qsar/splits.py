"""Train/test splits, ordered from optimistic to realistic.

A random split asks "can the model interpolate within chemistry it has already
seen?". That is almost never the question a project team has. The question is
whether the model holds up on the *next* series, so scaffold and
publication-level splits are reported alongside the random one and treated as
the headline numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import murcko_scaffold


@dataclass(frozen=True)
class Split:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray

    def __len__(self) -> int:
        return len(self.train_index) + len(self.test_index)

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train_index), len(self.test_index)


def random_split(
    frame: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> Split:
    indices = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    cut = int(round(len(indices) * (1.0 - test_fraction)))
    return Split("random", indices[:cut], indices[cut:])


def _group_split(
    groups: pd.Series, name: str, test_fraction: float, seed: int
) -> Split:
    """Assign whole groups to train or test, largest groups first.

    Filling the training set with the largest groups first keeps the test set
    made of the smaller, more peripheral groups, which is the realistic case:
    a new series arrives with few compounds in it.
    """
    order = groups.value_counts().index.tolist()
    rng = random.Random(seed)
    rng.shuffle(order)
    target_train = int(round(len(groups) * (1.0 - test_fraction)))

    train_groups: set = set()
    running = 0
    counts = groups.value_counts().to_dict()
    for group in sorted(order, key=lambda key: -counts[key]):
        if running >= target_train:
            break
        train_groups.add(group)
        running += counts[group]

    mask = groups.isin(train_groups).to_numpy()
    return Split(name, np.flatnonzero(mask), np.flatnonzero(~mask))


def scaffold_split(
    frame: pd.DataFrame,
    smiles_column: str = "smiles",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Split:
    """Split so that no Bemis-Murcko scaffold spans train and test."""
    scaffolds = frame[smiles_column].map(murcko_scaffold)
    # Acyclic compounds share an empty scaffold; give each its own group so
    # they are not forced into a single artificial cluster.
    scaffolds = pd.Series(
        [
            value if value else f"__acyclic_{position}"
            for position, value in enumerate(scaffolds)
        ],
        index=frame.index,
    )
    return _group_split(scaffolds, "scaffold", test_fraction, seed)


def document_split(
    frame: pd.DataFrame,
    document_column: str = "document_chembl_id",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Split:
    """Split by source publication.

    Compounds reported together were usually made together, as one series,
    screened in one assay campaign. Holding out whole publications therefore
    approximates the deployment case — a new programme, a new assay run — more
    closely than a scaffold split does, because it also breaks the shared
    experimental conditions, not only the shared chemistry.
    """
    documents = frame[document_column]
    if documents.isna().all():
        raise ValueError(
            f"{document_column!r} is empty for every row; "
            "a document split needs publication identifiers"
        )
    documents = documents.fillna(
        pd.Series(
            [f"__unknown_{position}" for position in range(len(frame))],
            index=frame.index,
        )
    )
    return _group_split(documents, "document", test_fraction, seed)


def all_splits(
    frame: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> list[Split]:
    """Every split that the given frame supports, optimistic first."""
    splits = [
        random_split(frame, test_fraction, seed),
        scaffold_split(frame, test_fraction=test_fraction, seed=seed),
    ]
    if "document_chembl_id" in frame and frame["document_chembl_id"].notna().any():
        splits.append(document_split(frame, test_fraction=test_fraction, seed=seed))
    return splits


def scaffold_overlap(frame: pd.DataFrame, split: Split, smiles_column: str = "smiles") -> int:
    """Number of scaffolds present in both halves. Should be 0 for a scaffold split."""
    scaffolds = frame[smiles_column].map(murcko_scaffold)
    train = set(scaffolds.iloc[split.train_index])
    test = set(scaffolds.iloc[split.test_index])
    return len({value for value in train & test if value})
