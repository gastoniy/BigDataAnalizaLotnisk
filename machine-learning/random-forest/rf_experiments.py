from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)

import sys

# Ten skrypt leży w machine-learning/random-forest/, a wspólny data_transform.py
# w machine-learning/ — dokładamy ten katalog do sys.path, żeby import był
# niezależny od katalogu roboczego (CWD).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data_transform import FlightsTransform


# Constants
DATA_DIR = HERE.parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "dataset_loty_krakow_20260610_165348.csv"
OUTPUT_DIR = HERE / "rf_logs"
THRESHOLD_MINUTES = 15
OOT_TEST_FRACTION = 0.20

# RF defaults used when no HPO best-params are available - if B or C has to be run separately
RF_DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 2}

# Experiment A search space
HPO_SEARCH_SPACE = {
    "rf__n_estimators": [100, 200, 300, 500],
    "rf__max_depth": [10, 15, 20, 25, None],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", 0.5],
    "rf__criterion": ["gini", "entropy"],
    "rf__class_weight": [None, "balanced", "balanced_subsample"],
}

# Experiment B/C strategies
CLASS_WEIGHT_STRATEGIES = [
    ("cw_none", None),
    ("cw_balanced", "balanced"),
    ("cw_balanced_subsample", "balanced_subsample"),
]
RESAMPLING_STRATEGIES = ["smote", "undersample", "smoteenn", "smotetomek"]

SCORING = {"f1": "f1", "balanced_accuracy": "balanced_accuracy"}


# Pipeline / model construction
def make_rf(rf_params: dict | None, base_seed: int) -> RandomForestClassifier:
    """RandomForest with project defaults, overridden by rf_params"""

    params = dict(RF_DEFAULT_PARAMS)
    params.update(rf_params or {})
    params.setdefault("random_state", base_seed)
    params["n_jobs"] = 1    # -1 will be set in next steps for better execution
    return RandomForestClassifier(**params)


def build_pipeline(ft: FlightsTransform, rf_params: dict | None, base_seed: int, resampler=None) -> ImbPipeline:
    """Leakage-safe pipeline: scaler -> [resampler] -> RandomForest"""

    steps = [("scaler", ft.get_scaler())]
    if resampler is not None:
        steps.append(("resampler", resampler))
    steps.append(("rf", make_rf(rf_params, base_seed)))
    return ImbPipeline(steps)


def rf_base_from_best(best_params: dict | None) -> dict:
    """Strip the ``rf__`` prefix from search best-params and drop class_weight"""

    if not best_params:
        return dict(RF_DEFAULT_PARAMS)
    out: dict = {}
    for key, value in best_params.items():
        if not key.startswith("rf__"):
            continue
        name = key[len("rf__"):]
        if name == "class_weight":
            continue
        out[name] = value
    return out or dict(RF_DEFAULT_PARAMS)

def _to_json_value(value):
    """Make numpy scalars / None JSON-serializable."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def evaluate_kfold(pipeline: ImbPipeline, X, y, splitter: StratifiedKFold) -> dict:
    """5-fold stratified CV; returns the per-model log block."""

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    f1_values = [float(v) for v in results["test_f1"]]
    bal_values = [float(v) for v in results["test_balanced_accuracy"]]
    folds = [
        {"fold": i + 1, "f1": f1_values[i], "balanced_accuracy": bal_values[i]}
        for i in range(len(f1_values))
    ]
    return {
        "folds": folds,
        "mean_f1": float(np.mean(f1_values)),
        "mean_balanced_accuracy": float(np.mean(bal_values)),
        "std_f1": float(np.std(f1_values, ddof=0)),
        "std_balanced_accuracy": float(np.std(bal_values, ddof=0)),
    }


def evaluate_oot(pipeline: ImbPipeline, X, y, train_idx, test_idx) -> dict:
    """Single chronological train/test split. Scaler + resampler fit on train only."""

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    fold_f1 = float(f1_score(y_test, y_pred))
    fold_bal = float(balanced_accuracy_score(y_test, y_pred))
    
    return {
        "folds": [{"fold": 1, "f1": fold_f1, "balanced_accuracy": fold_bal}],
        "mean_f1": fold_f1,
        "mean_balanced_accuracy": fold_bal,
        "std_f1": 0.0,
        "std_balanced_accuracy": 0.0,
    }


def time_order_split(data_path: Path, X: pd.DataFrame, test_fraction: float):
    """Recover czas_planowany and build a chronological split"""

    raw = pd.read_csv(data_path)
    planned = pd.to_datetime(raw["czas_planowany"], errors="coerce")
    planned = planned.reindex(X.index).dropna()

    ordered = planned.sort_values(kind="stable").index
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    train_idx = ordered[:-n_test]
    test_idx = ordered[-n_test:]
    return train_idx, test_idx


# Logging
def base_dataset_block(encoding: str, resampling: str, X, y) -> dict:
    return {
        "encoding": encoding,
        "resampling": resampling,
        "threshold": THRESHOLD_MINUTES,
        "row_count": int(len(y)),
        "feature_count": int(X.shape[1]),
        "run_index": 1,
    }


def write_log(payload: dict, output_dir: Path, name_parts: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / ("_".join([timestamp, *name_parts]) + ".json")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


# Experiment A — hyperparameter search
def run_hpo(data_path: Path, encoding: str, search_kind: str, n_iter: int, base_seed: int, output_dir: Path,) -> dict:
    ft = FlightsTransform(str(data_path))
    X, y = ft.load_xy(threshold_minutes=THRESHOLD_MINUTES, encoding=encoding)

    # passthrough resampler: imbalance handled via class_weight in the search space
    pipeline = build_pipeline(ft, rf_params=None, base_seed=base_seed, resampler=None)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)

    if search_kind == "grid":
        search = GridSearchCV(
            pipeline,
            param_grid=HPO_SEARCH_SPACE,
            scoring=SCORING,
            refit="f1",
            cv=splitter,
            n_jobs=-1,
            error_score="raise",
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=HPO_SEARCH_SPACE,
            n_iter=n_iter,
            scoring=SCORING,
            refit="f1",
            cv=splitter,
            n_jobs=-1,
            random_state=base_seed,
            error_score="raise",
        )

    search.fit(X, y)

    best_params = {k: _to_json_value(v) for k, v in search.best_params_.items()}
    cv_results = search.cv_results_
    best_index = int(search.best_index_)
    best_mean_f1 = float(cv_results["mean_test_f1"][best_index])
    best_mean_bal = float(cv_results["mean_test_balanced_accuracy"][best_index])
    best_std_f1 = float(cv_results["std_test_f1"][best_index])
    best_std_bal = float(cv_results["std_test_balanced_accuracy"][best_index])

    candidates = []
    for i in range(len(cv_results["params"])):
        candidates.append(
            {
                "params": {k: _to_json_value(v) for k, v in cv_results["params"][i].items()},
                "mean_f1": float(cv_results["mean_test_f1"][i]),
                "std_f1": float(cv_results["std_test_f1"][i]),
                "mean_balanced_accuracy": float(cv_results["mean_test_balanced_accuracy"][i]),
                "std_balanced_accuracy": float(cv_results["std_test_balanced_accuracy"][i]),
                "rank_f1": int(cv_results["rank_test_f1"][i]),
            }
        )
    candidates.sort(key=lambda c: c["rank_f1"])

    dataset = base_dataset_block(encoding, resampling="hpo", X=X, y=y)
    dataset["cv_random_state"] = base_seed
    payload = {
        "source_file": data_path.name,
        "dataset": dataset,
        "experiment": "hpo",
        "validation_scheme": "stratified_kfold",
        "search_kind": search_kind,
        "best_params": best_params,
        "search_results": candidates,
        "models": {
            "Random Forest": {
                "folds": [],
                "mean_f1": best_mean_f1,
                "mean_balanced_accuracy": best_mean_bal,
                "std_f1": best_std_f1,
                "std_balanced_accuracy": best_std_bal,
            }
        },
        "winner": {
            "model": "Random Forest",
            "mean_f1": best_mean_f1,
            "mean_balanced_accuracy": best_mean_bal,
        },
    }

    log_path = write_log(payload, output_dir, [encoding, "hpo", str(THRESHOLD_MINUTES)])
    print(
        f"[HPO] encoding={encoding} | best_f1={best_mean_f1:.4f} | "
        f"bal_acc={best_mean_bal:.4f} | params={best_params} -> {log_path.name}"
    )
    return best_params


# Experiments B / C — strategy sweep under a given validation scheme
def run_strategy_sweep(data_path: Path, encoding: str, scheme: str, rf_base_params: dict, base_seed: int, output_dir: Path) -> list[dict]:
    """Evaluate all 7 imbalance strategies under one validation scheme"""

    ft = FlightsTransform(str(data_path))
    X, y = ft.load_xy(threshold_minutes=THRESHOLD_MINUTES, encoding=encoding)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)
    train_idx = test_idx = None
    if scheme == "time_oot":
        train_idx, test_idx = time_order_split(data_path, X, OOT_TEST_FRACTION)

    # (strategy_label, rf_params, resampler-factory) for all 7 strategies
    strategies: list[tuple[str, dict, object]] = []
    for label, class_weight in CLASS_WEIGHT_STRATEGIES:
        params = dict(rf_base_params)
        params["class_weight"] = class_weight
        strategies.append((label, params, None))
    for method in RESAMPLING_STRATEGIES:
        params = dict(rf_base_params)
        params["class_weight"] = None
        strategies.append((method, params, method))

    summaries: list[dict] = []
    for label, rf_params, resampler_method in strategies:
        resampler = (
            ft.get_resampler(resampler_method, random_state=base_seed)
            if resampler_method is not None
            else None
        )
        pipeline = build_pipeline(ft, rf_params, base_seed, resampler=resampler)

        if scheme == "time_oot":
            model_block = evaluate_oot(pipeline, X, y, train_idx, test_idx)
        else:
            model_block = evaluate_kfold(pipeline, X, y, splitter)

        dataset = base_dataset_block(encoding, resampling=label, X=X, y=y)
        dataset["cv_random_state"] = base_seed
        if scheme == "time_oot":
            dataset["oot_train_rows"] = int(len(train_idx))
            dataset["oot_test_rows"] = int(len(test_idx))

        payload = {
            "source_file": data_path.name,
            "dataset": dataset,
            "experiment": "imbalance" if scheme == "stratified_kfold" else "validation",
            "validation_scheme": scheme,
            "strategy": {
                "label": label,
                "class_weight": rf_params.get("class_weight"),
                "resampler": resampler_method,
            },
            "rf_params": {k: _to_json_value(v) for k, v in rf_params.items()},
            "models": {"Random Forest": model_block},
            "winner": {
                "model": "Random Forest",
                "mean_f1": model_block["mean_f1"],
                "mean_balanced_accuracy": model_block["mean_balanced_accuracy"],
            },
        }

        write_log(payload, output_dir, [encoding, label, str(THRESHOLD_MINUTES), scheme])
        summaries.append(
            {
                "label": label,
                "mean_f1": model_block["mean_f1"],
                "mean_balanced_accuracy": model_block["mean_balanced_accuracy"],
            }
        )

    summaries.sort(key=lambda s: s["mean_f1"], reverse=True)
    print(f"\n[{scheme}] encoding={encoding} — strategy ranking by F1:")
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"  {rank}. {summary['label']:<22} | f1={summary['mean_f1']:.4f} | "
            f"bal_acc={summary['mean_balanced_accuracy']:.4f}"
        )
    return summaries


# CLI / orchestration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random Forest single-model experiments (threshold fixed at 15 min)."
    )
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH, help="Raw flights CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="JSON log directory.")
    parser.add_argument(
        "--encodings",
        choices=["onehot", "label"],
        nargs="+",
        default=["onehot", "label"],
        help="Airline encodings to evaluate.",
    )
    parser.add_argument(
        "--experiment",
        choices=["hpo", "imbalance", "validation", "all"],
        default="all",
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="Hyperparameter search strategy (Experiment A).",
    )
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV candidates.")
    parser.add_argument(
        "--validation",
        choices=["kfold", "oot", "both"],
        default="both",
        help="Validation scheme(s) for standalone Experiment C.",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Seed for CV splits and samplers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hpo_exp = args.experiment in ("hpo", "all")
    run_imbalance_exp = args.experiment in ("imbalance", "all")
    run_validation_exp = args.experiment in ("validation", "all")

    for encoding in args.encodings:
        print("=" * 70)
        print(f"Encoding: {encoding} | threshold: {THRESHOLD_MINUTES} min")
        print("=" * 70)

        # Experiment A: HPO -> best params reused downstream
        best_params = None
        if run_hpo_exp:
            best_params = run_hpo(
                args.data_path, encoding, args.search, args.n_iter, args.base_seed, args.output_dir
            )
        rf_base_params = rf_base_from_best(best_params)

        # Experiment B: imbalance study under StratifiedKFold
        if run_imbalance_exp:
            run_strategy_sweep(
                args.data_path, encoding, "stratified_kfold", rf_base_params, args.base_seed, args.output_dir
            )

        # Experiment C: validation scheme study
        if run_validation_exp:
            if args.experiment == "all":
                schemes = ["time_oot"]
            else:
                schemes = {
                    "kfold": ["stratified_kfold"],
                    "oot": ["time_oot"],
                    "both": ["stratified_kfold", "time_oot"],
                }[args.validation]
            for scheme in schemes:
                run_strategy_sweep(
                    args.data_path, encoding, scheme, rf_base_params, args.base_seed, args.output_dir
                )

    print(f"\nDone. Logs written to {args.output_dir}/")


if __name__ == "__main__":
    main()
