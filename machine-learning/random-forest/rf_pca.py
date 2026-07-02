from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys

# data_transform.py leży w machine-learning/ (o poziom wyżej) — dokładamy do sys.path.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data_transform import FlightsTransform

DATA_DIR = HERE.parent.parent / "data"
RAW_DATA_PATH = str(DATA_DIR / "dataset_loty_krakow_20260610_165348.csv")
COLORS = {0: "#1f77b4", 1: "#d62728"}
LABELS = {0: "Na czas", 1: "Opóźniony (> 15 min)"}


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA 2D wizualizacja rozdzielności klas.")
    ap.add_argument("--data-path", default=RAW_DATA_PATH)
    ap.add_argument("--encoding", choices=["label", "onehot"], default="label")
    ap.add_argument("--output-dir", type=Path, default=HERE / "rf_plots")
    args = ap.parse_args()

    ft = FlightsTransform(args.data_path)
    X, y = ft.load_xy(threshold_minutes=15, encoding=args.encoding)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    # plot majority class first so the minority isn't hidden underneath
    for cls in (0, 1):
        mask = (y.to_numpy() == cls)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.45, c=COLORS[cls], edgecolors="none",
            label=f"{LABELS[cls]} (n={int(mask.sum())})",
        )

    ax.set_title(
        f"PCA cech lotów — klasy nie tworzą skupisk ({args.encoding})\n"
        f"PC1 + PC2 wyjaśniają tylko {evr.sum() * 100:.1f}% wariancji"
    )
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"pca_classes_{args.encoding}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out}  |  wariancja PC1+PC2 = {evr.sum() * 100:.1f}%")


if __name__ == "__main__":
    main()
