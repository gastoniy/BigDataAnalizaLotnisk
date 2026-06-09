from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LOG_DIR = Path("traininglogs")
DEFAULT_OUTPUT_DIR = Path("trainingplots") / "mean_combinations"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "mean_combinations_summary.csv"

ENCODING_ORDER = ["label", "onehot"]
RESAMPLING_ORDER = ["smote", "undersample", "smoteenn", "smotetomek"]
MODEL_ORDER = [
    "Random Forest",
    "AdaBoost",
    "XGBoost",
    "Gaussian NB (GNB)",
    "Neural Network (MLP)",
]


def load_logs(log_dir: Path) -> list[dict[str, object]]:
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No JSON logs found in {log_dir}")

    logs: list[dict[str, object]] = []
    for log_file in log_files:
        logs.append(json.loads(log_file.read_text(encoding="utf-8")))
    return logs


def build_summary_frame(logs: list[dict[str, object]]) -> pd.DataFrame:
    aggregated: dict[tuple[int, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "balanced_accuracy": []}
    )

    for log in logs:
        dataset = log["dataset"]
        threshold = int(dataset["threshold"])
        encoding = str(dataset["encoding"])
        resampling = str(dataset["resampling"])

        for model_name, metrics in log["models"].items():
            key = (threshold, encoding, resampling, model_name)
            aggregated[key]["f1"].append(float(metrics["mean_f1"]))
            aggregated[key]["balanced_accuracy"].append(float(metrics["mean_balanced_accuracy"]))

    rows: list[dict[str, object]] = []
    for (threshold, encoding, resampling, model_name), values in aggregated.items():
        f1_scores = np.asarray(values["f1"], dtype=float)
        balanced_scores = np.asarray(values["balanced_accuracy"], dtype=float)
        rows.append(
            {
                "threshold": threshold,
                "encoding": encoding,
                "resampling": resampling,
                "model": model_name,
                "mean_f1": float(f1_scores.mean()),
                "std_f1": float(f1_scores.std(ddof=0)),
                "mean_balanced_accuracy": float(balanced_scores.mean()),
                "std_balanced_accuracy": float(balanced_scores.std(ddof=0)),
                "mean_of_means": float((f1_scores.mean() + balanced_scores.mean()) / 2.0),
                "runs": int(len(f1_scores)),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No model metrics were found in the provided logs.")

    frame["encoding"] = pd.Categorical(frame["encoding"], categories=ENCODING_ORDER, ordered=True)
    frame["resampling"] = pd.Categorical(frame["resampling"], categories=RESAMPLING_ORDER, ordered=True)
    frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)

    return frame.sort_values(["threshold", "encoding", "resampling", "model"], kind="stable")


def plot_threshold_group(
    frame: pd.DataFrame,
    threshold: int,
    output_dir: Path,
    show_error_bars: bool,
) -> Path:
    threshold_frame = frame[frame["threshold"] == threshold].copy()
    if threshold_frame.empty:
        raise ValueError(f"No rows available for threshold={threshold}")

    threshold_frame["scenario"] = (
        threshold_frame["encoding"].astype(str) + " / " + threshold_frame["resampling"].astype(str)
    )

    scenario_order = [f"{encoding} / {resampling}" for encoding in ENCODING_ORDER for resampling in RESAMPLING_ORDER]
    threshold_frame["scenario"] = pd.Categorical(
        threshold_frame["scenario"], categories=scenario_order, ordered=True
    )

    scenarios = [scenario for scenario in scenario_order if scenario in set(threshold_frame["scenario"].astype(str))]
    models = [model for model in MODEL_ORDER if model in set(threshold_frame["model"].astype(str))]

    pivot_f1 = threshold_frame.pivot_table(
        index="scenario",
        columns="model",
        values="mean_f1",
        aggfunc="mean",
    ).reindex(index=scenarios, columns=models)
    pivot_f1_std = threshold_frame.pivot_table(
        index="scenario",
        columns="model",
        values="std_f1",
        aggfunc="mean",
    ).reindex(index=scenarios, columns=models)
    pivot_balanced = threshold_frame.pivot_table(
        index="scenario",
        columns="model",
        values="mean_balanced_accuracy",
        aggfunc="mean",
    ).reindex(index=scenarios, columns=models)
    pivot_balanced_std = threshold_frame.pivot_table(
        index="scenario",
        columns="model",
        values="std_balanced_accuracy",
        aggfunc="mean",
    ).reindex(index=scenarios, columns=models)

    metrics = [
        ("mean_f1", "Mean F1", pivot_f1, pivot_f1_std),
        ("mean_balanced_accuracy", "Mean balanced accuracy", pivot_balanced, pivot_balanced_std),
    ]

    x = np.arange(len(scenarios), dtype=float)
    bar_width = 0.8 / max(len(models), 1)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * bar_width

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(14, len(scenarios) * 0.95), 10), dpi=160, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, (_, metric_label, pivot_values, pivot_std) in zip(axes, metrics, strict=True):
        for model_index, model_name in enumerate(models):
            values = pivot_values[model_name].to_numpy(dtype=float)
            yerr = pivot_std[model_name].to_numpy(dtype=float) if show_error_bars else None
            axis.bar(
                x + offsets[model_index],
                values,
                width=bar_width,
                label=model_name if axis is axes[0] else None,
                yerr=yerr,
                capsize=3 if show_error_bars else 0,
            )

        axis.set_ylabel(metric_label)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.2)
        axis.set_title(f"Threshold {threshold}")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(scenarios, rotation=25, ha="right")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    fig.suptitle(
        f"Mean model performance for threshold {threshold} across label/onehot and resampling settings",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"threshold_{threshold}_mean_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_best_rows(frame: pd.DataFrame) -> None:
    best_f1 = frame.sort_values(["mean_f1", "mean_balanced_accuracy"], ascending=False).iloc[0]
    best_balanced = frame.sort_values(["mean_balanced_accuracy", "mean_f1"], ascending=False).iloc[0]
    best_mean = frame.sort_values(["mean_of_means", "mean_f1"], ascending=False).iloc[0]

    print("Best by mean F1:")
    print(
        f"  threshold={int(best_f1['threshold'])}, encoding={best_f1['encoding']}, resampling={best_f1['resampling']}, "
        f"model={best_f1['model']}, mean_f1={best_f1['mean_f1']:.4f}, mean_balanced_accuracy={best_f1['mean_balanced_accuracy']:.4f}"
    )
    print("Best by mean balanced accuracy:")
    print(
        f"  threshold={int(best_balanced['threshold'])}, encoding={best_balanced['encoding']}, resampling={best_balanced['resampling']}, "
        f"model={best_balanced['model']}, mean_f1={best_balanced['mean_f1']:.4f}, mean_balanced_accuracy={best_balanced['mean_balanced_accuracy']:.4f}"
    )
    print("Best by average of the two means:")
    print(
        f"  threshold={int(best_mean['threshold'])}, encoding={best_mean['encoding']}, resampling={best_mean['resampling']}, "
        f"model={best_mean['model']}, mean_of_means={best_mean['mean_of_means']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all training logs and create column plots for the mean scores of each combination."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory with JSON training logs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and the summary CSV will be written",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to the aggregated summary CSV",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Disable standard deviation error bars in the plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs = load_logs(args.log_dir)
    summary = build_summary_frame(logs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False, encoding="utf-8")

    generated_plots: list[Path] = []
    for threshold in sorted(summary["threshold"].unique()):
        generated_plots.append(
            plot_threshold_group(summary, int(threshold), args.output_dir, show_error_bars=not args.no_error_bars)
        )

    print(f"Wrote summary table to {args.summary_csv}")
    print(f"Wrote {len(generated_plots)} plots to {args.output_dir}")
    print_best_rows(summary)


if __name__ == "__main__":
    main()