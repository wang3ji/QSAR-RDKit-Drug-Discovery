from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from a2a_qsar import data


def test_standardize_smiles_strips_counterion():
    # Hydrochloride salt: the chloride carries no SAR information.
    assert data.standardize_smiles("CCCCCCc1ccccc1N.Cl") == "CCCCCCc1ccccc1N"


def test_standardize_smiles_rejects_unparsable_and_tiny():
    assert data.standardize_smiles("not_a_molecule") is None
    assert data.standardize_smiles("") is None
    assert data.standardize_smiles(None) is None
    assert data.standardize_smiles("CCO") is None  # too small to be a ligand


def test_standardize_smiles_is_canonical():
    first = data.standardize_smiles("c1ccccc1C(=O)Nc1ccccc1")
    second = data.standardize_smiles("O=C(Nc1ccccc1)c1ccccc1")
    assert first == second


def test_inchikey_matches_for_equivalent_smiles():
    assert data.inchikey("C1=CC=CC=C1C(=O)O") == data.inchikey("OC(=O)c1ccccc1")


def test_load_annotated_aggregates_replicates(tmp_path):
    frame = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL2"],
            "standard_relation": ["=", "=", "="],
            "standard_type": ["IC50", "IC50", "IC50"],
            "smiles": [
                "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
                "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
                "c1ccc(CCNc2nc3ccccc3c3cn(-c4ccccc4)nc23)cc1",
            ],
            "pActivity": [7.0, 8.0, 6.5],
            "document_chembl_id": ["DOC1", "DOC2", "DOC1"],
            "assay_chembl_id": ["A1", "A2", "A1"],
        }
    )
    path = tmp_path / "annotated.csv"
    frame.to_csv(path, index=False)

    curated, report = data.load_annotated(path)

    assert len(curated) == 2
    replicated = curated[curated["n_measurements"] == 2].iloc[0]
    assert replicated["pactivity"] == pytest.approx(7.5)
    assert replicated["pactivity_range"] == pytest.approx(1.0)
    assert replicated["n_documents"] == 2
    assert report.records_in == 3
    assert report.compounds_out == 2
    assert report.replicated_compounds == 1


def test_load_annotated_drops_inexact_relations(tmp_path):
    frame = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "standard_relation": ["=", ">"],
            "standard_type": ["IC50", "IC50"],
            "smiles": [
                "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
                "c1ccc(CCNc2nc3ccccc3c3cn(-c4ccccc4)nc23)cc1",
            ],
            "pActivity": [7.0, 5.0],
            "document_chembl_id": ["DOC1", "DOC1"],
            "assay_chembl_id": ["A1", "A1"],
        }
    )
    path = tmp_path / "annotated.csv"
    frame.to_csv(path, index=False)

    curated, report = data.load_annotated(path)

    assert len(curated) == 1
    assert report.records_after_relation_filter == 1


def test_load_annotated_drops_implausible_values(tmp_path):
    frame = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "standard_relation": ["=", "="],
            "standard_type": ["IC50", "IC50"],
            "smiles": [
                "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
                "c1ccc(CCNc2nc3ccccc3c3cn(-c4ccccc4)nc23)cc1",
            ],
            # 1.8 is below the plausible window: a mis-stated unit upstream.
            "pActivity": [7.0, 1.8],
            "document_chembl_id": ["DOC1", "DOC1"],
            "assay_chembl_id": ["A1", "A1"],
        }
    )
    path = tmp_path / "annotated.csv"
    frame.to_csv(path, index=False)

    curated, _ = data.load_annotated(path)
    assert len(curated) == 1
    assert curated.iloc[0]["pactivity"] == pytest.approx(7.0)


def test_build_dataset_pool_excludes_modelling_compounds():
    modelling, pool, reports = data.build_dataset()
    assert len(modelling) > 0
    assert len(pool) > 0
    assert not set(modelling["inchikey"]) & set(pool["inchikey"])
    assert len(reports) == 2
    for frame in (modelling, pool):
        assert frame["pactivity"].between(*data.PLAUSIBLE_PACTIVITY).all()
        assert frame["inchikey"].is_unique


def test_label_actives():
    labels = data.label_actives(np.array([7.9, 8.0, 8.1]), threshold=8.0)
    assert labels.tolist() == [False, True, True]
