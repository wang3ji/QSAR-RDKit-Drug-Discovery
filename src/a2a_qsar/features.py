"""Molecular featurisation.

Two representations are offered so that the choice can be measured rather than
assumed: count-based Morgan fingerprints, and those fingerprints concatenated
with a small set of interpretable physicochemical descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

#: Descriptors that a medicinal chemist would read off a compound card.
DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
)


@dataclass(frozen=True)
class FeatureMatrix:
    """Features plus the names needed to interpret them."""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"{self.values.shape[1]} columns but {len(self.names)} names"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape


def _morgan_generator(radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_counts(
    smiles: list[str] | pd.Series,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_BITS,
) -> np.ndarray:
    """Count-based Morgan fingerprints.

    Counts are kept rather than folded to bits: how many times a substructure
    occurs carries real SAR signal that a binary vector discards.
    """
    from rdkit import Chem

    generator = _morgan_generator(radius, n_bits)
    rows = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"row {index}: unparsable SMILES {smi!r}")
        rows[index] = np.asarray(
            generator.GetCountFingerprintAsNumPy(mol), dtype=np.float32
        )
    return rows


def descriptors(smiles: list[str] | pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    getters = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Crippen.MolLogP,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumHDonors": Lipinski.NumHDonors,
        "NumHAcceptors": Lipinski.NumHAcceptors,
        "NumRotatableBonds": Lipinski.NumRotatableBonds,
        "RingCount": Lipinski.RingCount,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "HeavyAtomCount": lambda mol: mol.GetNumHeavyAtoms(),
    }
    rows = np.zeros((len(smiles), len(DESCRIPTOR_NAMES)), dtype=np.float32)
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"row {index}: unparsable SMILES {smi!r}")
        rows[index] = [getters[name](mol) for name in DESCRIPTOR_NAMES]
    return rows


def featurize(
    smiles: list[str] | pd.Series,
    representation: str = "morgan",
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_BITS,
) -> FeatureMatrix:
    """Build a feature matrix.

    ``representation`` is ``"morgan"``, ``"descriptors"`` or ``"combined"``.
    """
    smiles = list(smiles)
    if representation == "morgan":
        values = morgan_counts(smiles, radius, n_bits)
        names = tuple(f"morgan_{i}" for i in range(values.shape[1]))
    elif representation == "descriptors":
        values = descriptors(smiles)
        names = DESCRIPTOR_NAMES
    elif representation == "combined":
        fingerprints = morgan_counts(smiles, radius, n_bits)
        physchem = descriptors(smiles)
        values = np.hstack([fingerprints, physchem])
        names = tuple(
            [f"morgan_{i}" for i in range(fingerprints.shape[1])] + list(DESCRIPTOR_NAMES)
        )
    else:
        raise ValueError(
            f"unknown representation {representation!r}; "
            "expected 'morgan', 'descriptors' or 'combined'"
        )
    return FeatureMatrix(values=values, names=names)


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold as canonical SMILES.

    Acyclic molecules have an empty scaffold; they are returned as the empty
    string so the caller can group them explicitly instead of silently
    merging them with a ring system.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def tanimoto_similarity_matrix(
    smiles_a: list[str], smiles_b: list[str] | None = None
) -> np.ndarray:
    """Pairwise Tanimoto similarity on binary Morgan fingerprints."""
    from rdkit import Chem, DataStructs

    generator = _morgan_generator()

    def fingerprints(items: list[str]):
        out = []
        for smi in items:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"unparsable SMILES {smi!r}")
            out.append(generator.GetFingerprint(mol))
        return out

    left = fingerprints(list(smiles_a))
    right = left if smiles_b is None else fingerprints(list(smiles_b))
    matrix = np.zeros((len(left), len(right)), dtype=np.float32)
    for row, fingerprint in enumerate(left):
        matrix[row] = DataStructs.BulkTanimotoSimilarity(fingerprint, right)
    return matrix
