from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from a2a_qsar import features, splits

SMILES = [
    "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
    "c1ccc(CCNc2nc3ccccc3c3cn(-c4ccccc4)nc23)cc1",
    "CCCn1c(=O)c2[nH]c(-c3ccccc3)cc2n(CCC)c1=O",
    "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
    "CCCC(=O)Nc1nc(-c2ccccc2)nc(-c2ccccc2)c1C#N",
    "N#Cc1sc(=S)n(-c2ccccc2)c1/N=C/c1ccc(Cl)cc1",
]


def test_featurize_morgan_shape_and_counts():
    matrix = features.featurize(SMILES, representation="morgan", n_bits=256)
    assert matrix.shape == (len(SMILES), 256)
    assert len(matrix.names) == 256
    # Count fingerprints must retain multiplicity, not just presence.
    assert matrix.values.max() > 1


def test_featurize_descriptors_names_align():
    matrix = features.featurize(SMILES, representation="descriptors")
    assert matrix.names == features.DESCRIPTOR_NAMES
    assert matrix.shape == (len(SMILES), len(features.DESCRIPTOR_NAMES))
    molwt = matrix.values[:, features.DESCRIPTOR_NAMES.index("MolWt")]
    assert (molwt > 100).all()


def test_featurize_combined_is_concatenation():
    fingerprints = features.featurize(SMILES, representation="morgan", n_bits=128)
    physchem = features.featurize(SMILES, representation="descriptors")
    combined = features.featurize(SMILES, representation="combined", n_bits=128)
    assert combined.shape == (len(SMILES), 128 + len(features.DESCRIPTOR_NAMES))
    assert combined.values[:, :128] == pytest.approx(fingerprints.values)
    assert combined.values[:, 128:] == pytest.approx(physchem.values)


def test_feature_matrix_rejects_mismatched_names():
    with pytest.raises(ValueError, match="columns but"):
        features.FeatureMatrix(values=np.zeros((2, 3)), names=("a", "b"))


def test_featurize_rejects_unknown_representation():
    with pytest.raises(ValueError, match="unknown representation"):
        features.featurize(SMILES, representation="graph")


def test_featurize_rejects_bad_smiles():
    with pytest.raises(ValueError, match="unparsable"):
        features.featurize(["not_a_molecule"])


def test_murcko_scaffold_strips_side_chains():
    scaffold = features.murcko_scaffold("CCCCc1ccc(C(=O)NCC)cc1")
    assert "c1ccc" in scaffold.replace("C1=CC=C", "c1ccc")
    # Two analogues of one series share a scaffold.
    a = features.murcko_scaffold("CCn1c(=O)c2[nH]c(-c3ccccc3)cc2n(CC)c1=O")
    b = features.murcko_scaffold("CCCn1c(=O)c2[nH]c(-c3ccccc3)cc2n(CCC)c1=O")
    assert a == b


def test_murcko_scaffold_of_acyclic_is_empty():
    assert features.murcko_scaffold("CCCCCCCC(=O)O") == ""


def test_tanimoto_matrix_properties():
    matrix = features.tanimoto_similarity_matrix(SMILES[:4])
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T, atol=1e-6)
    assert ((matrix >= 0) & (matrix <= 1)).all()


def _frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "smiles": [SMILES[index % len(SMILES)] for index in range(n)],
            "pactivity": rng.normal(7.0, 1.0, size=n),
            "document_chembl_id": [f"DOC{index % 7}" for index in range(n)],
        }
    )


def test_random_split_partitions_every_row():
    frame = _frame()
    split = splits.random_split(frame, test_fraction=0.2, seed=1)
    assert len(split) == len(frame)
    assert not set(split.train_index) & set(split.test_index)
    assert split.sizes[1] == pytest.approx(12, abs=1)


def test_random_split_is_reproducible():
    frame = _frame()
    first = splits.random_split(frame, seed=5).test_index
    second = splits.random_split(frame, seed=5).test_index
    assert first.tolist() == second.tolist()


def test_scaffold_split_leaks_no_scaffold():
    frame = _frame()
    split = splits.scaffold_split(frame, test_fraction=0.3, seed=2)
    assert splits.scaffold_overlap(frame, split) == 0
    assert len(split.test_index) > 0


def test_document_split_keeps_publications_whole():
    frame = _frame()
    split = splits.document_split(frame, test_fraction=0.3, seed=2)
    train_docs = set(frame["document_chembl_id"].iloc[split.train_index])
    test_docs = set(frame["document_chembl_id"].iloc[split.test_index])
    assert not train_docs & test_docs
    assert len(split.test_index) > 0


def test_document_split_requires_identifiers():
    frame = _frame().assign(document_chembl_id=None)
    with pytest.raises(ValueError, match="document"):
        splits.document_split(frame)


def test_all_splits_omits_document_when_unavailable():
    frame = _frame().assign(document_chembl_id=None)
    names = [split.name for split in splits.all_splits(frame)]
    assert names == ["random", "scaffold"]


def test_all_splits_includes_document_when_available():
    names = [split.name for split in splits.all_splits(_frame())]
    assert names == ["random", "scaffold", "document"]
