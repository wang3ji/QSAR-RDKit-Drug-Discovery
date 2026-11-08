# Data provenance

Both files are snapshots of public, independently curated extracts of ChEMBL
bioactivity data for the human adenosine A2A receptor
(**CHEMBL251** / UniProt **P29274**). They are committed so that every number
in the README can be regenerated without network access, and so that a result
cannot silently change when the upstream database is updated.

The two sources are kept separate on purpose: one supplies the assay metadata
needed to estimate measurement error, the other supplies scale and an
independent check.

## `chembl251_annotated_bioactivity.csv`

| | |
|---|---|
| Rows | 1,243 measurements |
| Unique compounds | 986 |
| Distinct assays | 55 |
| Distinct publications | 49 |
| Activity types | IC50 (1,225), EC50 (18) |
| Retrieved | 2026-11-07 |
| Source | [`coreytaylorcompchem/the_kitbag`](https://github.com/coreytaylorcompchem/the_kitbag), `pipelines/data_retrieval_and_curation/projects/gpcrs/class_a_gpcrs/outputs/gpcr_class_a/P29274_bioactivity.csv` |
| Licence | MIT (upstream repository) |

Each row is **one measurement**, carrying `assay_chembl_id`,
`document_chembl_id` and the assay description. That per-measurement grain is
what makes this snapshot valuable here:

- 170 compounds were measured in more than one assay, which gives an
  **empirical estimate of inter-assay reproducibility** instead of an assumed
  one (`a2a-qsar assay-noise`).
- Publication identifiers allow a **document-level split**, which holds out
  whole series together with their experimental conditions.

## `chembl251_external_pxc50.tsv`

| | |
|---|---|
| Rows | 4,081 compounds (pre-aggregated pXC50) |
| Retrieved | 2026-11-07 |
| Source | [`prtos/few_shot_regression`](https://github.com/prtos/few_shot_regression), `datasets/chembl/pXC50_CHEMBL251_4081.tsv` |
| Licence | MIT (upstream repository) |

Line 1 is the target identifier, line 2 the receptor sequence, and the
remaining lines are `compound_id`, SMILES and pXC50. There is no assay
metadata, so this snapshot is never used to estimate noise or to select a
model. After structure standardisation and removal of the 78 compounds it
shares with the annotated set, 3,995 compounds remain. They serve as

1. an **external test set** — a genuinely independent curation of the same
   target, and
2. the **screening deck** for the active-learning simulation.

## Refreshing from ChEMBL directly

The curation functions accept any path, so a fresh pull can be substituted.
The [ChEMBL web resource client](https://github.com/chembl/chembl_webresource_client)
query equivalent to the annotated snapshot is:

```python
from chembl_webresource_client.new_client import new_client

records = (
    new_client.activity
    .filter(target_chembl_id="CHEMBL251", standard_type="IC50", standard_relation="=")
    .only("molecule_chembl_id", "canonical_smiles", "standard_value",
          "standard_units", "assay_chembl_id", "document_chembl_id")
)
```

Re-running on a newer ChEMBL release will change the counts. Record the
release number alongside any result produced that way.
