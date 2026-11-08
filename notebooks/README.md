# Notebooks

These are kept deliberately, as the exploratory work this project grew out of.
They are **not** the pipeline: everything load-bearing was rewritten as the
tested `a2a_qsar` package, and every number in the top-level README comes from
the CLI, not from here.

| Notebook | What it was for | Where it went |
|---|---|---|
| `01_original_qsar_exploration.ipynb` | First end-to-end A2A QSAR: ChEMBL pull, Morgan fingerprints, RF and XGBoost, scaffold split, SHAP on fingerprint bits | Became `data.py`, `features.py`, `splits.py`, `models.py` |
| `Binding affinity regression.ipynb` | Regression experiments on binding data | Folded into `models.py` |
| `QSAR basic.ipynb`, `QSAR Workflow Cheat Sheet.ipynb` | Working notes on the standard QSAR workflow | Reference only |
| `rdkit_tutorial.ipynb` | RDKit mechanics: molecule handling, fingerprint generators, drawing | Reference only |

## Merged from `A2aR-Allosteric-AI`

That repository held two notebooks (`ADOA2AR_CHEMBL251.ipynb` and
`Untitled0.ipynb`, near-duplicates of each other) that paged the ChEMBL
activity API for CHEMBL251 in Colab. That fetch is now `data.py`, with
pagination, structure standardisation, replicate aggregation and a recorded
curation report, so the notebooks are not carried over. `A2aR-Allosteric-AI`
can be archived with a pointer here.

Its stated ambition — allosteric modulators specifically — is kept as an
explicit limitation rather than an implied capability: the public potency data
for this target carries no mechanism annotation, so a model trained on it
cannot distinguish allosteric from orthosteric ligands. Doing that properly
needs assay-level labels (for example, readouts run at two agonist
concentrations to detect a shift in the curve), which is a wet-lab design
problem before it is a modelling one. See the model card.

## What changed in the rewrite, and why

The exploratory notebook reported R² ≈ 0.67 on a random split and ≈ 0.60 on a
scaffold split. The packaged pipeline reports 0.77 and 0.58 for the same
target, and the differences are instructive:

- **Count fingerprints instead of binary bits**, and median aggregation of
  repeated measurements rather than keeping the first row per SMILES, account
  for most of the gain on the random split.
- **Structure standardisation** (largest-fragment selection, re-canonicalisation,
  InChIKey deduplication) removes duplicate rows that differ only by
  counter-ion. In the notebook, `drop_duplicates(subset="canonical_smiles")`
  left those in, which inflates a random split by putting near-identical rows
  on both sides of it.
- **A harder scaffold split.** The notebook shuffled scaffold groups and cut at
  80%; the package fills training with the largest groups first, so the test
  set is made of the smaller, more peripheral series — the realistic case.
  That is why the scaffold number went slightly *down*: the split got harder,
  not the model worse.
- **A publication-level split was added**, and it is where the model actually
  breaks down. No notebook here would have caught that.

The SHAP analysis in the original notebook identified the fingerprint bits
driving predicted potency. That mapping is interpretable but fragile — a
Morgan bit is a hash, and the same bit can collide across substructures — so
it was not carried into the package as a load-bearing claim. It remains a
reasonable exploratory tool and is worth revisiting with an unhashed,
substructure-keyed representation.
