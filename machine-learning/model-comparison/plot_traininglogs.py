from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "traininglogs"
OUTPUT_DIR = HERE / "trainingplots"
SUMMARY_CSV = OUTPUT_DIR / "aggregated_metrics.csv"
LOG_NAME_PATTERN = re.compile(
    r"^sanitized_pandas_\d{8}_\d{6}_(?P<encoding>onehot|label)_(?P<resampling>smote|undersample|smoteenn|smotetomek)_(?P<threshold>\d+)\.json$"
)
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

    loaded_logs: list[dict[str, object]] = []
    for log_file in log_files:
        loaded_logs.append(json.loads(log_file.read_text(encoding="utf-8")))
    return loaded_logs


def build_summary_frame(logs: list[dict[str, object]]) -> pd.DataFrame:
    aggregated: dict[tuple[int, str, str, str], dict[str, list[float]]] = defaultdict(lambda: {"f1": [], "balanced_accuracy": []})

    for log in logs:
        dataset = log["dataset"]
        threshold = int(dataset["threshold"])
        encoding = str(dataset["encoding"])
        resampling = str(dataset["resampling"])

        for model_name, metrics in log["models"].items():
            key = (threshold, encoding, resampling, model_name)
            aggregated[key]["f1"].append(float(metrics["mean_f1"]))
            aggregated[key]["balanced_accuracy"].append(float(metrics["mean_balanced_accuracy"]))

    rows = []
    for (threshold, encoding, resampling, model_name), values in aggregated.items():
        rows.append(
            {
                "threshold": threshold,
                "encoding": encoding,
                "resampling": resampling,
                "model": model_name,
                "mean_f1": float(np.mean(values["f1"])),
                "mean_balanced_accuracy": float(np.mean(values["balanced_accuracy"])),
                "runs": len(values["f1"]),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No model metrics were found in the provided logs.")

    return frame.sort_values(["threshold", "encoding", "resampling", "model"], kind="stable")


def plot_group(frame: pd.DataFrame, threshold: int, encoding: str, resampling: str, output_dir: Path) -> Path:
    group = frame[
        (frame["threshold"] == threshold)
        & (frame["encoding"] == encoding)
        & (frame["resampling"] == resampling)
    ].copy()

    if group.empty:
        raise ValueError(f"No rows available for threshold={threshold}, encoding={encoding}, resampling={resampling}")

    ordered_models = [model for model in MODEL_ORDER if model in set(group["model"])]
    group["model"] = pd.Categorical(group["model"], categories=ordered_models, ordered=True)
    group = group.sort_values("model")

    x = np.arange(len(group))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.bar(x - width / 2, group["mean_f1"], width=width, label="Mean F1", color="#1f77b4")
    ax.bar(x + width / 2, group["mean_balanced_accuracy"], width=width, label="Mean balanced accuracy", color="#ff7f0e")

    ax.set_title(f"Threshold {threshold} | {encoding} | {resampling}")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(group["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"threshold_{threshold}_{encoding}_{resampling}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    logs = load_logs(LOG_DIR)
    summary = build_summary_frame(logs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    plotted_files: list[Path] = []
    for (threshold, encoding, resampling), _ in summary.groupby(["threshold", "encoding", "resampling"], sort=True):
        plotted_files.append(plot_group(summary, int(threshold), str(encoding), str(resampling), OUTPUT_DIR))

    print(f"Wrote summary table to {SUMMARY_CSV}")
    print(f"Wrote {len(plotted_files)} plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()