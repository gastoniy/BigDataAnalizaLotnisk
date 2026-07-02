from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import sys

# Ten skrypt leży w machine-learning/model-comparison/, a wspólny data_transform.py
# w machine-learning/ — dokładamy ten katalog do sys.path, żeby import działał
# niezależnie od katalogu roboczego (CWD).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data_transform import FlightsTransform


DATA_DIR = HERE.parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "dataset_loty_krakow_20260521_213240.csv"
OUTPUT_DIR = HERE / "traininglogs"
DEFAULT_THRESHOLDS = [5, 10, 15, 20, 25]
DEFAULT_ENCODINGS = ["onehot", "label"]
DEFAULT_RESAMPLING = ["smote", "undersample", "smoteenn", "smotetomek"]
DEFAULT_RUNS = 3
DEFAULT_BASE_SEED = 42


def build_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=150,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
        "Gaussian NB (GNB)": GaussianNB(),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }


def evaluate_configuration(
    data_path: Path,
    threshold: int,
    encoding: str,
    resampling: str,
    run_index: int,
    base_seed: int,
) -> dict[str, object]:
    ft = FlightsTransform(str(data_path))
    X, y = ft.load_xy(threshold_minutes=threshold, encoding=encoding)

    splitter_seed = base_seed + run_index
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=splitter_seed)
    models = build_models()
    model_results: dict[str, object] = {}

    for model_name, model in models.items():
        f1_values: list[float] = []
        balanced_values: list[float] = []
        fold_metrics: list[dict[str, float]] = []

        if "Random Forest" in model_name:
            model = clone(model).set_params(class_weight="balanced")
        elif "XGBoost" in model_name:
            model = clone(model).set_params(scale_pos_weight=2.4)

        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = ft.get_scaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=scaler.get_feature_names_out(),
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=scaler.get_feature_names_out(),
            )

            resampler = ft.get_resampler(resampling, random_state=splitter_seed)
            X_train, y_train = resampler.fit_resample(X_train, y_train)

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            fold_f1 = float(f1_score(y_test, y_pred))
            fold_balanced = float(balanced_accuracy_score(y_test, y_pred))
            f1_values.append(fold_f1)
            balanced_values.append(fold_balanced)
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "f1": fold_f1,
                    "balanced_accuracy": fold_balanced,
                }
            )

        model_results[model_name] = {
            "folds": fold_metrics,
            "mean_f1": float(np.mean(f1_values)),
            "mean_balanced_accuracy": float(np.mean(balanced_values)),
            "std_f1": float(np.std(f1_values, ddof=0)),
            "std_balanced_accuracy": float(np.std(balanced_values, ddof=0)),
        }

    winner_name = max(model_results, key=lambda name: model_results[name]["mean_f1"])
    winner_metrics = model_results[winner_name]

    return {
        "source_file": data_path.name,
        "dataset": {
            "encoding": encoding,
            "resampling": resampling,
            "threshold": threshold,
            "row_count": int(len(y)),
            "feature_count": int(X.shape[1]),
            "run_index": run_index + 1,
            "cv_random_state": splitter_seed,
        },
        "models": model_results,
        "winner": {
            "model": winner_name,
            "mean_f1": winner_metrics["mean_f1"],
            "mean_balanced_accuracy": winner_metrics["mean_balanced_accuracy"],
        },
    }


def write_log(payload: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = payload["dataset"]
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = (
        f"{timestamp}_{dataset['encoding']}_{dataset['resampling']}_{dataset['threshold']}_run{dataset['run_index']}.json"
    )
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models with fold-local scaling and resampling on the raw flights dataset."
    )
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH, help="Raw flights CSV to evaluate")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for JSON logs")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Delay thresholds in minutes to evaluate",
    )
    parser.add_argument(
        "--encodings",
        choices=["onehot", "label"],
        nargs="+",
        default=DEFAULT_ENCODINGS,
        help="Airline encoding schemes to evaluate",
    )
    parser.add_argument(
        "--resamplings",
        choices=["smote", "undersample", "smoteenn", "smotetomek"],
        nargs="+",
        default=DEFAULT_RESAMPLING,
        help="Resampling strategies to evaluate",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of reruns per combination")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for CV splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_logs: list[Path] = []

    for threshold in args.thresholds:
        for encoding in args.encodings:
            for resampling in args.resamplings:
                for run_index in range(args.runs):
                    payload = evaluate_configuration(
                        data_path=args.data_path,
                        threshold=threshold,
                        encoding=encoding,
                        resampling=resampling,
                        run_index=run_index,
                        base_seed=args.base_seed,
                    )
                    log_path = write_log(payload, args.output_dir)
                    written_logs.append(log_path)
                    winner = payload["winner"]
                    print(
                        f"Logged threshold={threshold} encoding={encoding} resampling={resampling} run={run_index + 1} -> "
                        f"winner={winner['model']} | f1={winner['mean_f1']:.4f} | bal_acc={winner['mean_balanced_accuracy']:.4f}"
                    )

    print(f"Finished writing {len(written_logs)} log files to {args.output_dir}")


if __name__ == "__main__":
    main()