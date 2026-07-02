# Random Forest Experiment Pipeline (`rf_experiments.py`)

Alternative, single-model branch of the ML track. Everything up to and including
`data_transform.py` is **unchanged** — this pipeline only replaces the wide
multi-model grid (`train_modeltests.py`) with a focused study of one estimator:
`RandomForestClassifier`.

Locked decisions for this branch:

- **One model:** Random Forest only.
- **One delay threshold:** `threshold_minutes = 15` (`czy_opozniony = delay > 15 min`).
- **Time validation:** a **single chronological cutoff** (no expanding-window folds).

---

## 1. Pipeline overview

```
baza_lotow.db
     │  (CSV export — unchanged)
     ▼
dataset_loty_krakow_*.csv  ──────────────┐  (raw CSV also re-read here
     │                                    │   for czas_planowany → time order)
     ▼                                    │
FlightsTransform.load_xy(                 │
    threshold_minutes=15,                 │
    encoding='onehot'|'label')            │
     │  → X, y   (NO scale, NO resample)  │
     ▼                                    ▼
┌──────────────────────────────────────────────────────────┐
│ rf_experiments.py                                          │
│                                                            │
│  imblearn Pipeline:                                        │
│    scaler (ft.get_scaler)  → resampler|passthrough → RF    │
│                                                            │
│  Experiment A — Hyperparameter search (StratifiedKFold)    │
│  Experiment B — Imbalance strategy study                   │
│  Experiment C — Validation scheme (KFold vs time cutoff)   │
└──────────────────────────────────────────────────────────┘
     │
     ▼
rf_logs/*.json  ──►  plot_mean_combinations.py / find_best_f1_threshold15.py
                     (schema-compatible, RF is the only model present)
```

---

## 2. Fixed components (reused, untouched)

| Component | Source | Role here |
|---|---|---|
| `FlightsTransform.load_xy(15, encoding)` | `data_transform.py` | Returns leakage-safe `X, y`. No scaling, no resampling. |
| `ft.get_scaler()` | `data_transform.py` | Fresh `ColumnTransformer`; becomes the **first Pipeline step** (fits per fold). |
| `ft.get_resampler(method, random_state)` | `data_transform.py` | imblearn sampler; becomes the **resampler Pipeline step**. |

### The leakage-safe estimator

All experiments operate on a single `imblearn.pipeline.Pipeline`:

```
Pipeline([
    ("scaler",    ft.get_scaler()),                 # ColumnTransformer
    ("resampler", <sampler> | "passthrough"),       # imblearn: TRAIN fold only
    ("rf",        RandomForestClassifier(...)),
])
```

The imblearn `Pipeline` applies the `resampler` step **only to the training fold**,
so any CV / search wrapped around this object is leakage-safe *without* the manual
per-fold loop used in `train_modeltests.py`. The `ColumnTransformer` scaler is also
re-fit on each training fold by virtue of being inside the pipeline.

- Studying `class_weight` → `resampler = "passthrough"`, vary `rf__class_weight`.
- Studying resampling → `resampler = <sampler>`, `rf__class_weight = None`.

---

## 3. Experiment part (precise specification)

All three experiments use:
- `threshold_minutes = 15`
- estimator = Random Forest inside the imblearn Pipeline above
- primary metric = **F1**, secondary = **balanced accuracy**
- base seed configurable (`--base-seed`, default 42)

### Experiment A — Hyperparameter search

**Purpose:** find the best RF configuration; its result feeds Experiments B and C.

**Procedure**
1. Build the Pipeline with `resampler = "passthrough"` (imbalance handled by
   `class_weight` *inside the search space*, so HPO and imbalance choice are tuned
   jointly).
2. Wrap it in **`RandomizedSearchCV`** (default) or `GridSearchCV` (`--search grid`).
   - `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)`
   - `scoring = {"f1": "f1", "balanced_accuracy": "balanced_accuracy"}`
   - `refit = "f1"`
   - `n_iter` configurable (`--n-iter`, default 40) for the randomized search.
3. Run **once at threshold 15**, for each requested `encoding` (`onehot`, `label`).

**Search space** (all prefixed `rf__`)

| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | 10, 15, 20, 25, None |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | "sqrt", "log2", 0.5 |
| `criterion` | "gini", "entropy" |
| `class_weight` | None, "balanced", "balanced_subsample" |

**Output:** `best_params`, best CV F1 + balanced accuracy, and a trimmed
`cv_results_` (params + mean/std of both metrics per candidate) → one JSON log.

### Experiment B — Imbalance strategy study

**Purpose:** for the *tuned* RF, isolate how class imbalance is best handled.

**Procedure**
1. Fix RF at **Experiment A's `best_params`** (minus the `class_weight` field, which
   becomes a variable below). If A was not run, fall back to the documented default
   (`n_estimators=300, max_depth=15, min_samples_leaf=2`).
2. Evaluate the following **7 strategies** under identical CV folds
   (`StratifiedKFold(5, shuffle=True, random_state=base_seed)`):

   | # | resampler step | `rf__class_weight` |
   |---|---|---|
   | 1 | passthrough | None (baseline) |
   | 2 | passthrough | "balanced" |
   | 3 | passthrough | "balanced_subsample" |
   | 4 | smote | None |
   | 5 | undersample | None |
   | 6 | smoteenn | None |
   | 7 | smotetomek | None |

3. For each strategy record per-fold F1 + balanced accuracy, then mean and std.

**Output:** one JSON log per encoding, holding all 7 strategies (each written as a
`"models"`-style entry so the existing plotters can render them side by side).

### Experiment C — Validation scheme (single chronological cutoff)

**Purpose:** contrast the optimistic stratified estimate against an honest
forecasting estimate.

**Two schemes, same 7 strategies as Experiment B:**

- **(i) StratifiedKFold** — identical to Experiment B (the optimistic reference).
- **(ii) Time-based single cutoff (OOT):**
  1. Re-read the **raw CSV separately** and parse `czas_planowany`
     (`load_xy` drops it, and `data_transform.py` must stay untouched).
  2. Align the parsed datetime to `X.index` — `preprocess()` preserves the original
     row index through column selection and `dropna`, so the datetime Series can be
     reindexed onto `X` exactly.
  3. Sort rows chronologically; the **earliest 80%** is the training set, the
     **latest 20%** is the test set (single cutoff, no folds).
  4. Fit the Pipeline on the train slice (scaler + resampler fit on train only),
     evaluate on the held-out future slice.

> **Caveat (must appear in the report):** `note.txt` flags several synthetic dates
> (2026-04-{18,19,21–23}). The time-cutoff result is reported with this caveat, as
> the synthetic days distort the true chronological signal.

**Output:** one JSON log per encoding holding both schemes; each scheme reports the
7 strategies' F1 + balanced accuracy (CV: mean/std over folds; OOT: single test
score).

---

## 4. Output format

One JSON file per (encoding × experiment), written to `rf_logs/`. Schema mirrors
`train_modeltests.py` so the existing analysis/plot scripts work when pointed at
`rf_logs/`:

```json
{
  "source_file": "dataset_loty_krakow_*.csv",
  "dataset": {
    "encoding": "onehot",
    "resampling": "smote",
    "threshold": 15,
    "row_count": 0,
    "feature_count": 0,
    "run_index": 1,
    "cv_random_state": 42
  },
  "experiment": "hpo | imbalance | validation",
  "validation_scheme": "stratified_kfold | time_oot",
  "best_params": { "rf__n_estimators": 300 },
  "models": {
    "Random Forest": {
      "folds": [ { "fold": 1, "f1": 0.0, "balanced_accuracy": 0.0 } ],
      "mean_f1": 0.0,
      "mean_balanced_accuracy": 0.0,
      "std_f1": 0.0,
      "std_balanced_accuracy": 0.0
    }
  },
  "winner": { "model": "Random Forest", "mean_f1": 0.0, "mean_balanced_accuracy": 0.0 }
}
```

Because `dataset.*` and `models.{Random Forest}` keep the original shape,
**`plot_mean_combinations.py`, `plot_traininglogs.py`, and
`find_best_f1_threshold15.py` run unchanged** against `rf_logs/` — Random Forest is
simply the only model present.

---

## 5. CLI

`argparse`, mirroring `train_modeltests.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--data-path` | `dataset_loty_krakow_*.csv` | Raw flights CSV. |
| `--output-dir` | `rf_logs` | JSON log directory. |
| `--encodings` | `onehot label` | Airline encodings to evaluate. |
| `--experiment` | `all` | `hpo` / `imbalance` / `validation` / `all`. |
| `--search` | `random` | `random` (RandomizedSearchCV) or `grid`. |
| `--n-iter` | `40` | Candidates for the randomized search. |
| `--validation` | `both` | `kfold` / `oot` / `both` (Experiment C). |
| `--base-seed` | `42` | Seed for CV splits and samplers. |

> Threshold is fixed at 15 in this branch and is not exposed as a flag.

---

## 6. Runtime notes

- HPO with resampling is the expensive part; running it at the **single threshold
  15** with `RandomizedSearchCV` keeps it tractable. The cheaper B/C comparisons
  reuse A's best params.
- The estimator is constructed via a small factory step so a future swap (e.g.
  XGBoost) is a one-line change — but this branch ships **Random Forest only**.

---

## 7. A better data source (closing recommendation)

### Why our results plateau (~0.57 F1)
Diagnostics on the current dataset show the ceiling is set by the **input features**,
not the model or the tuning:

- Honest 5-fold **ROC-AUC = 0.74** — the schedule/route/airline features carry a
  *real but moderate* delay signal.
- **Best F1 = 0.568** across *any* probability threshold; HPO's only effective levers
  were `class_weight` and `max_features` (more trees / depth did nothing).
- Feature importances are **flat** (top feature ≈ 0.15, airline only ≈ 0.06), and the
  `miesiac_*` (month) features are near-constant because the scrape spans only
  ~April–May 2026.

Individual-flight delay is largely driven by **same-day operational state** that the
Kraków departure board simply does not record: the late arrival of the inbound
aircraft (rotational delay — *the* strongest predictor in the literature), weather,
and air-traffic-flow (ATFM) restrictions.

### Recommended source: U.S. DOT BTS — *Airline On-Time Performance / Delay Causes*
A free, public, well-documented dataset (Bureau of Transportation Statistics,
TranStats; mirrored on Kaggle). It covers all domestic U.S. flights from 2003 to the
present, with **per-flight delay-cause minutes** — exactly the variables our diagnostics
flagged as missing.

| Gap proven in our data | What BTS adds |
|---|---|
| No upstream/rotational signal (top real predictor) | **`LateAircraftDelay`** (minutes the late inbound aircraft contributed) |
| No weather information | **`WeatherDelay`** |
| No ATC / airspace congestion | **`NASDelay`** (National Airspace System) |
| No carrier-side operational cause | **`CarrierDelay`**, **`SecurityDelay`** |
| Dead `miesiac_*` (single 2-month window) | 20+ years → real **seasonal** signal |
| Only ~5k rows, sparse per-route/airline cells | **Millions** of flights/year → dense cells, stable per-(carrier × route × hour) rates |

Because BTS exposes the causal-delay columns, it supports both (a) far stronger
**features** (e.g. lagged/rolling `LateAircraftDelay` per tail or per route as a
congestion proxy) and (b) a cleaner **label**, lifting the achievable F1 well above the
~0.57 ceiling our feature set imposes. The same `FlightsTransform` → RF pipeline transfers
directly; only the column mapping changes.

> **European alternative (geographically closer to KRK):** the **EUROCONTROL Aviation
> Data Repository for Research (ADRR / R&D Data Archive)** — ~12M European commercial
> flights (sample months 2015–2018), with flight plans and ATFM delay, free via the
> OneSky portal. More relevant to Polish/European traffic, but trajectory-oriented and
> needs more assembly than the ready-to-model BTS tables.

**Takeaway for the report:** our model is *not* under-built — it extracts ~0.74 AUC from
the only signal the departure-board data contains. To predict individual delays
materially better, the lever is **a richer source, not a bigger model**; BTS is the
natural upgrade because it ships the exact causal features (led by late-aircraft /
rotational delay) that the Kraków board omits.

*Sources:* [BTS On-Time Performance & Delay Causes (TranStats)](https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp) ·
[BTS On-Time Data overview](https://www.bts.gov/explore-topics-and-geography/topics/time-data) ·
[Airline On-Time Statistics and Delay Causes — Kaggle](https://www.kaggle.com/datasets/daryaheyko/airline-on-time-statistics-and-delay-causes-bts) ·
[EUROCONTROL Aviation Data Repository for Research](https://www.eurocontrol.int/dashboard/aviation-data-research)