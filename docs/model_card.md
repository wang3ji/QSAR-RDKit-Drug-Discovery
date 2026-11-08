# Model card: A2A receptor potency regressor

## Overview

| | |
|---|---|
| Task | Regression of ligand potency (pActivity, i.e. −log10 of IC50/EC50 in molar) |
| Target | Human adenosine A2A receptor (CHEMBL251 / UniProt P29274) |
| Input | SMILES, featurised as count-based Morgan fingerprints (radius 2, 2,048 bits), optionally concatenated with 10 physicochemical descriptors |
| Output | Point prediction plus a conformal prediction interval |
| Models | Random forest (500 trees) and bagged XGBoost (5 bags) |
| Training data | 983 unique structures, median-aggregated from 1,243 public measurements |
| Version | 0.1.0 |

## Intended use

**Use it to rank.** Across every evaluation here, rank order survives far
better than absolute potency. The defensible application is prioritising
which compounds from a library go onto the next plate, where a 4–5×
enrichment over random picking translates directly into fewer wells for the
same number of hits.

**Do not use it to predict a number you would act on.** On a held-out
publication the coefficient of determination is approximately zero: the model
cannot tell you that an untested compound from a new series will come in at
8.4 rather than 7.1. Treating a prediction as a measurement is the failure
mode this card exists to prevent.

## Out-of-scope uses

- Selectivity against other adenosine receptor subtypes (A1, A2B, A3). Only
  A2A data was used, so the model has no basis for a subtype comparison.
- Allosteric versus orthosteric mechanism. The training labels are potency
  values with no mechanism annotation, so the model cannot distinguish them.
- ADME, toxicity, or any property that is not A2A potency.
- Chemistry unlike the training set. Applicability domain should be checked
  by Tanimoto similarity to the training set before a prediction is used.

## Evaluation

Three splits are reported, deliberately ordered from optimistic to realistic,
because the gap between them is the result.

| Split | What it asks | R² (XGBoost, Morgan) | Spearman ρ |
|---|---|---|---|
| Random | Interpolation within known chemistry | 0.77 | 0.87 |
| Scaffold | A new scaffold | 0.58 | 0.81 |
| Document | A new publication, i.e. a new series and a new assay campaign | 0.00 | 0.53 |
| External set (3,995 compounds, independent curation) | A different curation of the same target | 0.15 | 0.38 |

Enrichment in the top 5% of the ranked external set is 4.5–4.9×.

### Error relative to the measurement floor

Inter-assay reproducibility estimated from 170 compounds measured in more than
one assay is **0.60 log units** (pooled within-compound standard deviation).
On a random split the model's RMSE is 0.52–0.54, which is *at or below* that
floor: the remaining error is indistinguishable from the error of the
experiment the labels came from. No change of algorithm addresses this. Only
more reproducible measurements, or more replicates, would.

### Calibration of the uncertainty

Conformal intervals at a nominal 80% level achieve:

| Split | Observed coverage |
|---|---|
| Random | 0.86 |
| Scaffold | 0.73–0.77 |
| Document | 0.47–0.55 |

Conformal prediction guarantees coverage only when calibration and test data
are exchangeable. A scaffold or document split breaks exchangeability by
construction, and the measured coverage degrades accordingly. **The intervals
are therefore most overconfident in exactly the situation where a project team
would most want to trust them.** Reported here rather than omitted, because a
silently overconfident interval is worse than no interval.

## Training data characteristics and bias

- **Publication bias.** Public potency data is reported because it was worth
  reporting. Weakly active compounds are under-represented, so the model sees
  a population shifted towards potency (training median pActivity 7.5).
- **Series clustering.** 983 compounds carry far fewer than 983 independent
  chemotypes. Performance on a random split is inflated by near-neighbours of
  training compounds appearing in the test set.
- **Mixed readouts.** IC50 and EC50 values from 55 assays are pooled on the
  pActivity scale. This is standard practice for regression-ready extracts and
  is part of why the measured inter-assay spread is as wide as it is.
- **Censoring.** Values outside a plausible 3–12 pActivity window are dropped
  as likely unit errors. Inexact relations (`>`, `<`) are excluded, which
  removes genuinely inactive compounds and makes the retained set
  further biased towards actives.

## Ethical and safety considerations

This is a retrospective study on public data for method development. No
compound prioritised by this model has been synthesised or tested, and nothing
here supports a clinical or safety claim. Adenosine A2A antagonists are an
active immuno-oncology area; a prediction from this model is a hypothesis for
an assay, not evidence of activity.

## Reproducing

```bash
pip install -e ".[dev]"
a2a-qsar curate
a2a-qsar assay-noise
a2a-qsar benchmark
a2a-qsar campaign --experiment both
```

All randomness is seeded. CI regenerates the headline numbers from the
committed snapshots on every change.

## Maintainer

Jingjing Wang — [github.com/wang3ji](https://github.com/wang3ji)
