"""Dataset assembly and curation for adenosine A2A receptor (CHEMBL251) ligands.

Two independently curated public sources are used, deliberately kept separate:

``annotated``
    Per-measurement records with assay and document identifiers. Because the
    same compound is often measured in several assays, this source carries the
    information needed to *estimate inter-assay variability empirically* rather
    than assuming it (see :mod:`a2a_qsar.assay`), and to split by publication
    rather than at random.

``external``
    A larger, independently curated set of aggregated pXC50 values with no
    assay metadata. It is never used for model selection. It serves as an
    external test set and as the virtual screening pool for the
    active-learning simulation in :mod:`a2a_qsar.active`.

Curation follows the conventions used for regression-ready ChEMBL extracts:
keep exact (``=``) relations only, standardize structures, aggregate repeated
measurements of the same structure by median, and record how much spread was
collapsed in the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGET_CHEMBL_ID = "CHEMBL251"
TARGET_UNIPROT = "P29274"

#: Repository-relative location of the committed snapshots.
DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

ANNOTATED_SNAPSHOT = DATA_RAW / "chembl251_annotated_bioactivity.csv"
EXTERNAL_SNAPSHOT = DATA_RAW / "chembl251_external_pxc50.tsv"

#: Activity types treated as interchangeable potency readouts.
ACCEPTED_ACTIVITY_TYPES = ("IC50", "EC50", "Ki", "Kd")

#: pActivity values outside this window are implausible for a binding assay
#: and usually indicate a unit or transcription error upstream.
PLAUSIBLE_PACTIVITY = (3.0, 12.0)


@dataclass(frozen=True)
class CurationReport:
    """What curation did, so the numbers in the README can be reproduced."""

    source: str
    records_in: int
    records_after_relation_filter: int
    records_after_type_filter: int
    records_after_structure_parsing: int
    records_after_range_filter: int
    compounds_out: int
    replicated_compounds: int
    median_within_compound_sd: float | None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "records_in": self.records_in,
            "records_after_relation_filter": self.records_after_relation_filter,
            "records_after_type_filter": self.records_after_type_filter,
            "records_after_structure_parsing": self.records_after_structure_parsing,
            "records_after_range_filter": self.records_after_range_filter,
            "compounds_out": self.compounds_out,
            "replicated_compounds": self.replicated_compounds,
            "median_within_compound_sd": self.median_within_compound_sd,
        }


def standardize_smiles(smiles: str) -> str | None:
    """Return a canonical SMILES for the largest organic fragment, or ``None``.

    Salts and co-crystallised counter-ions carry no SAR information but do
    create duplicate rows that differ only by counter-ion, so the largest
    fragment is kept and the structure is re-canonicalised.
    """
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    mol = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() < 6:
        return None
    return Chem.MolToSmiles(mol)


def inchikey(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToInchiKey(mol)


def _aggregate(
    records: pd.DataFrame, source: str, counts_in: dict[str, int]
) -> tuple[pd.DataFrame, CurationReport]:
    """Collapse repeated measurements of one structure into one row."""
    grouped = records.groupby("inchikey")
    aggregated = grouped.agg(
        smiles=("smiles", "first"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        pactivity=("pactivity", "median"),
        n_measurements=("pactivity", "size"),
        pactivity_sd=("pactivity", "std"),
        pactivity_range=("pactivity", lambda values: values.max() - values.min()),
        document_chembl_id=(
            "document_chembl_id",
            lambda values: sorted(set(values.dropna()))[0] if values.notna().any() else None,
        ),
        n_documents=("document_chembl_id", lambda values: values.dropna().nunique()),
    ).reset_index()

    replicated = aggregated[aggregated["n_measurements"] > 1]
    report = CurationReport(
        source=source,
        records_in=counts_in["records_in"],
        records_after_relation_filter=counts_in["after_relation"],
        records_after_type_filter=counts_in["after_type"],
        records_after_structure_parsing=counts_in["after_structure"],
        records_after_range_filter=len(records),
        compounds_out=len(aggregated),
        replicated_compounds=int(len(replicated)),
        median_within_compound_sd=(
            float(replicated["pactivity_sd"].median()) if len(replicated) else None
        ),
    )
    return aggregated.sort_values("inchikey").reset_index(drop=True), report


def load_annotated(path: Path | None = None) -> tuple[pd.DataFrame, CurationReport]:
    """Curate the per-measurement snapshot.

    Returns one row per unique structure plus the per-structure measurement
    spread, which downstream code uses as the empirical assay noise estimate.
    """
    path = Path(path) if path is not None else ANNOTATED_SNAPSHOT
    raw = pd.read_csv(path)
    # The upstream export repeats ``target_pref_name``; pandas disambiguates
    # duplicate column labels, so select by position to stay version-stable.
    raw = raw.loc[:, ~raw.columns.duplicated()]
    counts = {"records_in": len(raw)}

    frame = raw.rename(columns={"pActivity": "pactivity"})
    frame = frame[frame["standard_relation"].astype(str).str.strip() == "="]
    counts["after_relation"] = len(frame)

    frame = frame[frame["standard_type"].isin(ACCEPTED_ACTIVITY_TYPES)]
    counts["after_type"] = len(frame)

    frame = frame.assign(smiles=frame["smiles"].map(standardize_smiles))
    frame = frame[frame["smiles"].notna()]
    frame = frame.assign(inchikey=frame["smiles"].map(inchikey))
    frame = frame[frame["inchikey"].notna()]
    counts["after_structure"] = len(frame)

    frame = frame[frame["pactivity"].between(*PLAUSIBLE_PACTIVITY)]

    keep = [
        "inchikey",
        "smiles",
        "molecule_chembl_id",
        "pactivity",
        "document_chembl_id",
        "assay_chembl_id",
        "standard_type",
    ]
    return _aggregate(frame[keep].copy(), source=path.name, counts_in=counts)


def load_external(path: Path | None = None) -> tuple[pd.DataFrame, CurationReport]:
    """Curate the aggregated external snapshot (no assay metadata)."""
    path = Path(path) if path is not None else EXTERNAL_SNAPSHOT
    # Line 1 is the target identifier, line 2 the target sequence.
    raw = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        header=None,
        names=["molecule_chembl_id", "smiles", "pactivity"],
    )
    counts = {"records_in": len(raw)}
    counts["after_relation"] = len(raw)  # pre-aggregated; no relation column
    counts["after_type"] = len(raw)

    frame = raw.assign(smiles=raw["smiles"].map(standardize_smiles))
    frame = frame[frame["smiles"].notna()]
    frame = frame.assign(inchikey=frame["smiles"].map(inchikey))
    frame = frame[frame["inchikey"].notna()]
    counts["after_structure"] = len(frame)

    frame = frame[frame["pactivity"].between(*PLAUSIBLE_PACTIVITY)]
    frame["document_chembl_id"] = None
    return _aggregate(frame, source=path.name, counts_in=counts)


def build_dataset(
    annotated_path: Path | None = None,
    external_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[CurationReport]]:
    """Return ``(modelling_set, screening_pool, reports)``.

    The screening pool has every structure that also appears in the modelling
    set removed, so the active-learning simulation cannot "discover" a
    compound the model was already trained on.
    """
    modelling, annotated_report = load_annotated(annotated_path)
    external, external_report = load_external(external_path)

    overlap = set(modelling["inchikey"]) & set(external["inchikey"])
    pool = external[~external["inchikey"].isin(overlap)].reset_index(drop=True)
    LOGGER.info(
        "modelling=%d external=%d overlap=%d pool=%d",
        len(modelling),
        len(external),
        len(overlap),
        len(pool),
    )
    return modelling, pool, [annotated_report, external_report]


def label_actives(pactivity: np.ndarray | pd.Series, threshold: float = 8.0) -> np.ndarray:
    """Binary potency label. ``pActivity >= 8`` is 10 nM or better."""
    return np.asarray(pactivity) >= threshold
