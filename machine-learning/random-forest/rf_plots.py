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


HERE = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = HERE / "rf_logs"
DEFAULT_OUTPUT_DIR = HERE / "rf_plots"

# 7 strategies in display order; the first 3 are class-weight, the last 4 resampling.
STRATEGY_ORDER = [
    "cw_none",
    "cw_balanced",
    "cw_balanced_subsample",
    "smote",
    "undersample",
    "smoteenn",
    "smotetomek",
]
CLASS_WEIGHT_STRATEGIES = {"cw_none", "cw_balanced", "cw_balanced_subsample"}

# group colors: class-weight (blue) vs resampling (orange)
COLOR_CLASS_WEIGHT = "#1f77b4"
COLOR_RESAMPLING = "#ff7f0e"
COLOR_KFOLD = "#1f77b4"
COLOR_OOT = "#d62728"

# hyperparameters searched in Experiment A
HPO_PARAMS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "criterion",
    "class_weight",
]
# compact labels used when summarizing a candidate's params on one line
PARAM_SHORT = {
    "n_estimators": "n",
    "max_depth": "d",
    "min_samples_leaf": "leaf",
    "max_features": "feat",
    "criterion": "crit",
    "class_weight": "cw",
}


def load_logs(log_dir: Path) -> list[dict]:
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No JSON logs found in {log_dir}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in log_files]


def strategy_color(label: str) -> str:
    return COLOR_CLASS_WEIGHT if label in CLASS_WEIGHT_STRATEGIES else COLOR_RESAMPLING


def build_summary_frame(logs: list[dict]) -> pd.DataFrame:
    """One row per (encoding, scheme, strategy) for the two sweep experiments."""
    rows: list[dict] = []
    for log in logs:
        if log.get("experiment") not in ("imbalance", "validation"):
            continue
        dataset = log["dataset"]
        block = log["models"]["Random Forest"]
        rows.append(
            {
                "encoding": dataset["encoding"],
                "scheme": log["validation_scheme"],
                "strategy": dataset["resampling"],
                "mean_f1": float(block["mean_f1"]),
                "std_f1": float(block["std_f1"]),
                "mean_balanced_accuracy": float(block["mean_balanced_accuracy"]),
                "std_balanced_accuracy": float(block["std_balanced_accuracy"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No imbalance/validation logs found to summarize.")
    frame["strategy"] = pd.Categorical(frame["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return frame.sort_values(["encoding", "scheme", "strategy"], kind="stable")


def collect_hpo(logs: list[dict]) -> dict[str, dict]:
    """encoding -> {'best_params':..., 'candidates': [...]}."""
    out: dict[str, dict] = {}
    for log in logs:
        if log.get("experiment") != "hpo":
            continue
        out[log["dataset"]["encoding"]] = {
            "best_params": log.get("best_params", {}),
            "candidates": log.get("search_results", []),
        }
    return out


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    present = [s for s in STRATEGY_ORDER if s in set(frame["strategy"].astype(str))]
    frame = frame.copy()
    frame["strategy"] = pd.Categorical(frame["strategy"], categories=present, ordered=True)
    return frame.sort_values("strategy")


# Figure 1 — strategy comparison (kfold)
def plot_strategy_comparison(frame: pd.DataFrame, encoding: str, output_dir: Path, error_bars: bool) -> Path:
    group = _ordered(frame[(frame["encoding"] == encoding) & (frame["scheme"] == "stratified_kfold")])
    if group.empty:
        raise ValueError(f"No stratified_kfold rows for encoding={encoding}")

    strategies = list(group["strategy"].astype(str))
    x = np.arange(len(strategies), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.bar(
        x - width / 2,
        group["mean_f1"],
        width=width,
        label="Mean F1",
        color="#1f77b4",
        yerr=group["std_f1"] if error_bars else None,
        capsize=3 if error_bars else 0,
    )
    ax.bar(
        x + width / 2,
        group["mean_balanced_accuracy"],
        width=width,
        label="Mean balanced accuracy",
        color="#ff7f0e",
        yerr=group["std_balanced_accuracy"] if error_bars else None,
        capsize=3 if error_bars else 0,
    )

    # annotate F1 value on top of each F1 bar
    for xi, value in zip(x - width / 2, group["mean_f1"]):
        ax.text(xi, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axvline(2.5, color="#999999", linestyle=":", linewidth=1)
    ax.text(1.0, 1.02, "class_weight", ha="center", fontsize=9, color="#555555")
    ax.text(4.5, 1.02, "resampling", ha="center", fontsize=9, color="#555555")

    ax.set_title(f"RF imbalance strategies — 5-fold CV — {encoding} (threshold 15 min)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"strategy_comparison_kfold_{encoding}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# Figure 2 — kfold vs OOT gap
def plot_kfold_vs_oot(frame: pd.DataFrame, encoding: str, output_dir: Path, error_bars: bool) -> Path:
    enc = frame[frame["encoding"] == encoding]
    kfold = _ordered(enc[enc["scheme"] == "stratified_kfold"]).set_index("strategy")
    oot = _ordered(enc[enc["scheme"] == "time_oot"]).set_index("strategy")
    if kfold.empty or oot.empty:
        raise ValueError(f"Need both schemes for encoding={encoding}")

    strategies = [s for s in STRATEGY_ORDER if s in kfold.index and s in oot.index]
    x = np.arange(len(strategies), dtype=float)
    width = 0.38

    metrics = [
        ("mean_f1", "std_f1", "F1"),
        ("mean_balanced_accuracy", "std_balanced_accuracy", "Balanced accuracy"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=160, sharex=True)

    for ax, (mean_col, std_col, label) in zip(axes, metrics):
        k_vals = kfold.loc[strategies, mean_col].to_numpy(dtype=float)
        o_vals = oot.loc[strategies, mean_col].to_numpy(dtype=float)
        k_err = kfold.loc[strategies, std_col].to_numpy(dtype=float) if error_bars else None

        ax.bar(x - width / 2, k_vals, width=width, label="StratifiedKFold", color=COLOR_KFOLD,
               yerr=k_err, capsize=3 if error_bars else 0)
        ax.bar(x + width / 2, o_vals, width=width, label="Time OOT", color=COLOR_OOT)

        for xi, kv, ov in zip(x, k_vals, o_vals):
            gap = kv - ov
            ax.text(xi, max(kv, ov) + 0.02, f"Δ{gap:+.3f}", ha="center", va="bottom",
                    fontsize=8, color="#333333")

        ax.set_ylabel(label)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.2)

    axes[0].axvline(2.5, color="#999999", linestyle=":", linewidth=1)
    axes[1].axvline(2.5, color="#999999", linestyle=":", linewidth=1)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title(f"kfold vs out-of-time — {encoding} (threshold 15 min)  ·  Δ = kfold − OOT")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(strategies, rotation=20, ha="right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"kfold_vs_oot_{encoding}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# Figure 3 — HPO overview
def _short_params(params: dict) -> str:
    parts = []
    for key in HPO_PARAMS:
        full = f"rf__{key}"
        if full in params:
            parts.append(f"{PARAM_SHORT[key]}={params[full]}")
    return ", ".join(parts)


def plot_hpo_overview(hpo: dict, encoding: str, output_dir: Path, top_n: int, error_bars: bool) -> Path:
    candidates = hpo[encoding]["candidates"]
    if not candidates:
        raise ValueError(f"No HPO candidates for encoding={encoding}")

    cand_df = pd.DataFrame(
        [
            {
                "mean_f1": float(c["mean_f1"]),
                "std_f1": float(c["std_f1"]),
                **{f"rf__{p}": c["params"].get(f"rf__{p}") for p in HPO_PARAMS},
                "label": _short_params(c["params"]),
            }
            for c in candidates
        ]
    ).sort_values("mean_f1", ascending=False).reset_index(drop=True)

    n_panels = 1 + len(HPO_PARAMS)
    fig = plt.figure(figsize=(15, 11), dpi=160)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1])

    # Panel A — top-N candidates
    ax_top = fig.add_subplot(gs[0, :])
    top = cand_df.head(top_n).iloc[::-1]  # best at top
    y = np.arange(len(top))
    ax_top.barh(y, top["mean_f1"], color="#4c78a8",
                xerr=top["std_f1"] if error_bars else None,
                capsize=3 if error_bars else 0)
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(top["label"], fontsize=8)
    ax_top.set_xlabel("Mean F1 (5-fold CV)")
    ax_top.set_xlim(0, max(0.05, cand_df["mean_f1"].max() * 1.15))
    ax_top.patches[-1].set_color("#e45756")
    ax_top.set_title(f"HPO top {len(top)} candidates by F1 — {encoding}")
    ax_top.grid(axis="x", alpha=0.2)

    # Panels B — per-hyperparameter effect (mean ± std of candidate mean_f1)
    positions = [gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0], gs[2, 1], gs[2, 2]]
    for param, pos in zip(HPO_PARAMS, positions):
        ax = fig.add_subplot(pos)
        col = f"rf__{param}"
        grouped = cand_df.groupby(cand_df[col].astype(str))["mean_f1"]
        stats = grouped.agg(["mean", "std", "count"]).sort_index()
        labels = list(stats.index)
        xpos = np.arange(len(labels))
        ax.bar(xpos, stats["mean"], yerr=stats["std"].fillna(0) if error_bars else None,
               capsize=3 if error_bars else 0, color="#72b7b2")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(param, fontsize=10)
        ax.set_ylim(0, max(0.05, cand_df["mean_f1"].max() * 1.15))
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Random Forest hyperparameter search — {encoding} (threshold 15 min)", y=0.995, fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hpo_overview_{encoding}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plots adapted to the RF experiment logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory with rf_logs JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write plots/CSV.")
    parser.add_argument("--encodings", nargs="+", default=None, help="Encodings to plot (default: all detected).")
    parser.add_argument("--top-n", type=int, default=10, help="Top HPO candidates in Figure 3 panel A.")
    parser.add_argument("--no-error-bars", action="store_true", help="Disable std error bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_bars = not args.no_error_bars

    logs = load_logs(args.log_dir)
    summary = build_summary_frame(logs)
    hpo = collect_hpo(logs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "rf_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    encodings = args.encodings or sorted(summary["encoding"].unique())
    generated: list[Path] = []

    for encoding in encodings:
        generated.append(plot_strategy_comparison(summary, encoding, args.output_dir, error_bars))
        generated.append(plot_kfold_vs_oot(summary, encoding, args.output_dir, error_bars))
        if encoding in hpo:
            generated.append(plot_hpo_overview(hpo, encoding, args.output_dir, args.top_n, error_bars))
        else:
            print(f"[warn] no HPO log for encoding={encoding}; skipping Figure 3")

    print(f"Wrote summary table to {summary_csv}")
    print(f"Wrote {len(generated)} plots to {args.output_dir}:")
    for path in generated:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
