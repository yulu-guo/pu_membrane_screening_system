# PU Membrane Property Prediction and Screening

This system predicts the membrane properties requested for polyurethane design:

- Glass transition temperature (`Tg`, deg C)
- Young's modulus (`MPa`)
- Tensile strength (`MPa`)
- Elongation at break (`%`)
- Initial decomposition temperature (`Td5%`, deg C)
- Maximum decomposition temperature (deg C)

It accepts an arbitrary formulation assembled from isocyanates, polyols, chain
extenders, and crosslinkers. Components can be repeated, and the build order can
be alternating, random, block, or explicitly supplied.

The system covers one reaction family only: `polyurethane`, assembled from
isocyanates, polyols, chain extenders, and crosslinkers. The earlier
`polyimide` family and the standalone **Known Structure** mode (repeat-unit
SMILES, TDTPA diamines, aromatic dianhydrides) were removed on 2026-10-16
along with their dataset `data/structures/polyimide_components.csv`, the
`/api/known-structures` and `/api/predict-polymer` endpoints, and their
interface controls.

## Important Scope

The public `Tg` model is trained on 11,964 unique polymer structures: the
original 7,174-row NeurIPS set plus 4,790 structures sourced later from public
repositories (see `Supplementary Tg Data` below). It includes 516
polyurethane/polyurea-containing structures.

The mechanical model is trained on 643 experimental polyurethane elastomers
released with:

> Ding et al., "Predicting the Mechanical Properties of Polyurethane Elastomers
> Using Machine Learning", Chinese Journal of Polymer Science, 2023.  
> Dataset DOI: `10.57760/sciencedb.14957`

The 643-row dataset uses 20 composition, processing, and measurement features.
It does not contain explicit monomer sequences. Therefore:

- `Tg` is predicted from the generated polymer structure and is
  arrangement-sensitive.
- Mechanical properties use a molecule-to-feature compiler and are
  composition-sensitive. Arrangement changes the generated structure, but
  arrangement-sensitive mechanical accuracy is limited by the public labels.
- The interface marks these predictions as screening estimates.

Additional public data are collected by `collect_public_data.py`:

- 1,023 polymer Td/Tg/Tm/mechanical records from an open GitHub Polymer-Genome
  derived dataset.
- 439 curated records covering 60 polymers and ten thermal properties,
  including `Td5_C`, `Tg_C`, and modulus.

`Td5%` and the maximum decomposition temperature come from the broad
public-data thermal models for every formulation. The public `Td5` training set
holds only 58 rows, so both numbers are qualitative screening values rather
than calibrated estimates.

## Supplementary Tg Data

`build_tg_supplement.py` assembles `data/training/tg_polymers_extended.csv`
from public sources that are none of them the removed polyurethane paper:

- Polymer Tg prediction dataset, `chem-data-extraction/polymer-Tg-prediction-dataset`,
  `data/processed/dataset.csv`, **CC BY 4.0**, 7 372 records of experimentally
  measured Tg assembled from open-access articles and a public Zenodo deposit.
  Archived as `data/raw/public/polymer_tg_dataset_ccby4.csv`.
  Canonical-SMILES comparison shows 2 644 structures shared with the existing
  training file and **4 723 genuinely new ones**.
- NeurIPS Open Polymer Prediction supplementary Tg table (45 new records).
- `Tg_C` rows of the curated ten-property table that were not used before
  (22 new records).

The script keeps a structure only when its canonical form is absent from
`tg_polymers.csv`, so merging cannot double-count a polymer spelled two ways.
`train_models.py` concatenates the supplement when it is present.

Elongation at break is displayed in percent: the stored model target is
`log10(strain ratio)`, so the interface applies a factor of 100 after
converting from log10.

## PLA-OH / PTMC-OH Soft Segment

When the soft segment is built only from `POL-PLA*` and `POL-PTMC*` polyols, the
system switches on a dedicated composition model
(`pu_v2/pla_ptmc.py`, reference table `data/reference/pla_ptmc_family.csv`).
Both sources are independent open-access papers, not the removed dataset:

- `Tg` follows a Gordon-Taylor mixing law between PLA and PTMC. The mixing
  parameter `k` is fitted by least squares to published DSC anchors (neat PLA
  about 58 C, PLA/PTMC 7:3 = 38.8 C, PTMC about -20 C) from Int. J. Mol. Sci.
  18(10), 2041 (2017), DOI `10.3390/ijms18102041`.
- Chain ends matter: these soft segments are oligomeric diols of a few thousand
  g/mol, not high polymers, so a plain blend model reads far too high. The
  Flory-Fox term `Tg = Tg_inf - K / Mn` is applied to the lactide side with `K`
  fitted to a published molecular-weight series for a lactide-based polyester
  (8000 g/mol -> 42.17 C, 110000 g/mol -> 52.62 C) from Pharmaceutics 14(5),
  993 (2022), DOI `10.3390/polym14050993`. No published chain-end parameter for
  PTMC was found, so that side keeps its measured anchor and the family Tg can
  still sit slightly high.
- The soft-segment Mn comes from the `软段多元醇 Mn` field; leave it blank and
  the catalog polyol weight is used, which is well below a commercial
  PLA-OH/PTMC-OH diol. The response warns when that happens.
- `Td5%` and the maximum decomposition temperature are interpolated over the
  measured PLA/PTMC blend series in Table 4 of Int. J. Mol. Sci. 15(2), 2608
  (2014), DOI `10.3390/ijms15022608`. That series covers 0-50 mass% PTMC;
  compositions outside it are clipped to the nearest measured point and the
  response says so.
- Mechanical properties still come from the general 643-row model. No
  independent mechanical dataset for this family is available, so tensile
  strength, modulus, and elongation there stay screening estimates.

The switch only fires when both polyols are present. A soft segment with PLA or
PTMC alone, or with any third polyol family, uses the general models instead,
so predictions can step across that boundary.

## Accuracy Against Published Tg Models

The broad `Tg` model and the published interpretable-ML framework of
Int. J. Mol. Sci. / Polymers 16(8), 1049 (2024), DOI `10.3390/polym16081049`,
are trained on the same 7174-polymer Tg dataset:

| Model | Feature set | Reported error |
|---|---|---:|
| This system, original 7 174 rows | Morgan fingerprint, extra trees | R2 0.886, MAE 26.08 C (5-fold CV) |
| This system, after supplementation | Morgan fingerprint, extra trees, 11 964 rows | R2 0.906, MAE 23.05 C (5-fold CV) |
| This system, single holdout (shipped metric) | same, 11 964 rows | R2 0.914, MAE 22.44 C |
| Molnar-type framework, best tier | molecular descriptors, extra trees | R2 0.869, MAE 27.90 (10-fold CV) |
| Molnar-type framework, second tier | molecular descriptors, SVR / HGB / XGB tuned | R2 0.859-0.862, MAE 28.06-28.94 (10-fold CV) |
| Molnar-type framework, Morgan only | Morgan fingerprint, SVR | R2 0.815, MAE 32.98 (10-fold CV) |
| Molnar-type framework, untuned baselines | decision tree / SVR | R2 0.686 / 0.684, MAE 40.95 / 47.60 (10-fold CV) |

Single-holdout numbers are usually more pessimistic than 10-fold CV on the same
data, so the broad `Tg` model is at least in the same class and slightly ahead
on error even before supplementation, and the supplement widens that margin.
Measured under the same 5-fold protocol, the model beats the published best on
the identical 7 174 rows by 6.5% MAE (26.08 vs 27.90 C) and, after
supplementation, by 17.4% (23.05 vs 27.90 C). On the polyurethane subset alone
the supplement moves the MAE from 24.82 C to 19.23 C, a 22.5% reduction. Note
that the shipped single-holdout figure of 22.44 C is slightly optimistic
compared with cross-validation, and that 10-fold CV is marginally more
generous than 5-fold, so the true margin is if anything a little smaller than
the numbers above.
The polyurethane subset of the training data grew from 316 to 516 structures in
the same step. That is still not enough to resolve the PLA/PTMC family, which is
why that family keeps its own module: the structure set holds only a handful of
lactide and carbonate repeats, so a fingerprint model cannot resolve the
soft-segment composition or its chain length.

## Accuracy of the Mechanical Model

The 643-row dataset comes from Ding et al. (DOI `10.1007/s10118-022-2838-6`),
which reports R2 of 0.73-0.78 on the raw dataset for Young's modulus, tensile
strength, and elongation at break. Directly comparable numbers require scoring
this model on the original units rather than on log10, because R2 is not
invariant to a target transform:

| Target | 5-fold CV, R2 on raw units | MAE on raw units | Ding et al. raw |
|---|---:|---:|---:|
| Young's modulus | 0.542 | 28.4 MPa | 0.73-0.78 |
| Tensile strength | 0.602 | 6.74 MPa | 0.73-0.78 |
| Elongation at break | 0.719 | 126.6 % | 0.73-0.78 |

So this model sits roughly 0.1-0.2 below the source paper on the same rows. Part
of that is the training objective: the shipped model is fitted on log10 targets,
which favours relative accuracy (and therefore the small values) over raw-unit
R2. Fitting the same trees on the raw units instead gives R2 of 0.695 / 0.638 /
0.720 with essentially unchanged MAE, which recovers most of the Young's modulus
gap. The shipped default keeps the log objective because a screening tool cares
about relative error across orders of magnitude.

One further caveat: those numbers use the 20 numeric features exactly as the
source dataset ships them. When a user enters monomers, the features are
compiled from structure descriptors first, which adds error on top.

No additional public polyurethane mechanical dataset was found, so this model
still trains on the same 643 rows. Searches covered GitHub, DataCite,
figshare, Mendeley Data, Harvard Dataverse, Europe PMC and Crossref. The
candidates that exist are far too small or measure something else: a 178-polymer
transfer-learning table whose mechanical label is shear modulus with 13-18 rows
per property, a 51-row UV-cured network set, and a raw stress-strain trace
archive covering four commodity polymers. The only local extra is an 18-row
polyurethane table (`data/training/frontiers_pue_18.csv`) carrying raw
formulation data rather than the dataset's 20 numeric features; rebuilding those
features through the approximation layer would inject more noise than 18 rows
can pay back, so it is left out.

What could be improved is how the same rows are fitted. `model_tuning.py`
benchmarks four learners against two objectives, scored on raw units with 5-fold
cross-validation:

| Configuration | YM R2 / MAPE | TS R2 / MAPE | EB R2 / MAPE |
|---|---|---|---|
| extra trees, log targets (previous) | 0.543 / 87.5% | 0.601 / 41.4% | 0.719 / 26.4% |
| extra trees, raw targets | 0.693 / 211.8% | 0.636 / 50.8% | 0.721 / 29.2% |
| **extra trees, unrestricted depth, log targets (shipped)** | **0.624 / 86.9%** | **0.608 / 40.1%** | **0.710 / 25.8%** |
| random forest, raw targets | 0.607 / 246.0% | 0.615 / 55.6% | 0.683 / 33.4% |
| histogram gradient boosting, raw targets | 0.560 / 247.1% | 0.578 / 49.7% | 0.661 / 31.2% |

Fitting on raw units buys a large R2 gain and destroys relative accuracy: the
modulus MAPE goes from 87% to 212%, which is useless for screening. The shipped
configuration therefore keeps log10 targets and removes the depth and
feature-subsampling limits, which improves every relative-error figure and every
modulus/strength accuracy figure at once.

`Td5%` is unchanged and remains data-limited: with 58 rows, the shipped extra
trees configuration is already the best of the four learners tried (R2 0.489,
MAE 53.9 C, 5-fold CV), and gradient boosting is markedly worse.

The system is also ready for lab calibration. Measured films can be imported,
converted into the same feature space, and used to train a residual/direct
calibration model without discarding the public-data model.

## Removed Training Data

On 2026-10-14 the 16-row PLA-OH/PTMC-OH + LDI/BDO table from DOI
`10.1002/app.57238` was removed from this system at the user's request, together
with the bilinear interpolation model and the interface controls that existed
only for it. That paper now contributes nothing to any model weight, lookup
table, or interface default; the remaining thermal output comes solely from the
public datasets listed above.

The PLA-OH/PTMC-OH capability was then rebuilt from the two independent
open-access sources listed under `PLA-OH / PTMC-OH Soft Segment` above, so the
family input still exists and still drives the prediction.

## Current Model Metrics

Held-out test metrics from the included training run:

| Property | R2 | MAE |
|---|---:|---:|
| Tg | 0.891 | 25.86 deg C |
| log10 Young's modulus | 0.647 | 0.304 |
| log10 tensile strength | 0.715 | 0.161 |
| log10 elongation at break | 0.646 | 0.115 |

See `data/models/metrics.json` for the exact generated values.

## Run

### Setup from a fresh clone

The trained model bundles (`data/models/*.joblib`) are **not** in the
repository: `tg_model.joblib` is about 120 MB, above GitHub's 100 MB per-file
limit. Rebuild them once after cloning:

```powershell
pip install -r requirements.txt
python train_models.py --trees 500
python train_public_thermal_models.py
```

`train_models.py` also reads `data/training/tg_polymers_extended.csv`, which is
already in the repository, so no network access is needed for those two steps
unless you want to refresh the raw downloads with `build_data.py`.

### Start the server

Double-click:

```text
start_pu_system.bat
```

This starts the server and opens `http://127.0.0.1:8766`. Keep the terminal
window open while using the website.

Command-line alternative:

```powershell
cd C:\Users\Lenovo\Documents\Codex\2026-10-09\wo\outputs\pu_membrane_screening_system
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Open:

```text
http://127.0.0.1:8766
```

## Rebuild Public Data

The generated data are already included. Rebuilding downloads or reuses:

- `TgSS_enriched_cleaned.csv` from the public NeurIPS polymer dataset mirror.
- `PUE643 YM-TS-EB.csv` from Science Data Bank.
- Frontiers supplementary Table 2 for an additional 18-sample PU elastomer
  dataset.
- PubChem name lookups for common chain extenders and crosslinkers.

```powershell
python build_data.py
python train_models.py --trees 500
```

## Add Lab Measurements

Use `data/lab_template.csv` as the schema. `monomer_ids` and `counts` are
pipe-separated lists:

```csv
sample_id,monomer_ids,counts,arrangement,nco_oh_index,synthesis_step,forming_method,reaction_temperature_1_c,reaction_temperature_2_c,cross_section_area_mm2,strain_rate_min_inv,Tg_C,Youngs_modulus_MPa,Tensile_strength_MPa,Elongation_at_break_percent
M01,ISO-MDI|POL-PEG400|CE-BDO,2|1|1,alternating,1.05,2,solution_casting,80,80,30,0.1,55,80,25,450
```

Import measured rows:

```powershell
python import_lab_data.py --input .\data\lab_template.csv
```

Train calibration after at least eight rows are available for a target:

```powershell
python train_lab_calibration.py --min-samples 8
```

The application automatically uses
`data/models/lab_calibration.joblib` when it exists.

## API

```text
GET  /api/health
GET  /api/catalog
GET  /api/properties
POST /api/predict
POST /api/screen
```

Example prediction:

```json
{
  "components": [
    {"monomer_id": "ISO-MDI", "count": 2},
    {"monomer_id": "POL-PEG400", "count": 1},
    {"monomer_id": "CE-BDO", "count": 1}
  ],
  "arrangement": "alternating",
  "processing": {
    "nco_oh_index": 1.05,
    "synthesis_step": 2,
    "forming_method": "solution_casting",
    "reaction_temperature_1_c": 80,
    "reaction_temperature_2_c": 80,
    "cross_section_area_mm2": 30,
    "strain_rate_min_inv": 0.1
  }
}
```

## Main Files

```text
app.py                     local web server and API
build_data.py              public data and monomer library build
train_models.py            Tg and mechanical model training
import_lab_data.py         import measured lab films
train_lab_calibration.py   train lab-specific calibration models
pu_v2/chemistry.py         arbitrary monomer sequence and PU construction
pu_v2/features.py          mechanical feature compiler
pu_v2/model.py             structure and property models
pu_v2/screening.py         target-first inverse screening
ui/                        browser interface
```

## Scientific Limitations

- The public mechanical data contain 643 samples, so predictions for untested
  formulations remain uncertain.
- Explicit sequence/blockiness labels are not present in the 643-row dataset.
  The system therefore calculates sequence-sensitive structures but does not
  claim a fully trained sequence-effect model.
- Curing kinetics, solvent removal, chain orientation, annealing, fillers,
  plasticizers, and post-treatment can substantially change real film
  properties. They should be included in lab data as additional columns and
  incorporated during future calibration.

## Excluded Papers

`data/holdout/validation_only_papers.csv` contains papers reserved for external
validation, plus papers removed from training, and it is checked so that none of
them can be re-imported by accident.

External validation entries:

> Zheng et al., "Multi-Scenario Adaptive High-Fidelity Self-Healing Elastomer
> Based on Poly(Ionic Liquid) Containing Polyurethane for Temperature-Pressure
> Sensing and Human-Machine Interaction", Small, DOI `10.1002/smll.76141`,
> published 2026-10-07.

Reported results include 10.93 MPa tensile strength, 1020.23% elongation at
break, 98.3% self-healing efficiency in 3 h at 33 C, and ionic conductivity up
to 0.45 mS/cm.

> Ponomareva et al., "Furan-Urethane Monomers for Self-Healing Polyurethanes",
> Polymers, DOI `10.3390/polym17141951`. Chemical structures only.

Removed entry:

> Zhao et al., "Fully Bio-Based TPUs From PLA/PTMC: Tunable Strength and
> Degradation Profiles", Journal of Applied Polymer Science, DOI
> `10.1002/app.57238`. Deleted from `data/training` on 2026-10-14; the ledger
> entry exists only so the paper cannot re-enter the training set unnoticed.

Run:

```powershell
python check_holdout_papers.py
```

The script verifies that the DOI, title, and key metric do not occur in the
current CSV training files.
