from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


TARGET_COLUMN = "czy_opozniony"
DEFAULT_INPUT_DIR = Path("modeltests")
DEFAULT_OUTPUT_DIR = Path("tsnes")
MAX_PCA_COMPONENTS = 50
CSV_NAME_PATTERN = re.compile(
    r"^sanitized_pandas_\d{8}_\d{6}_(?P<encoding>onehot|label)_(?P<method>smote|undersample|smoteenn|smotetomek)_(?P<threshold>\d+)\.csv$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create t-SNE plots for all sanitized CSV files in modeltests."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the CSV datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper limit on rows per dataset. If set, samples are drawn stratified by class.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. The script will lower it automatically if a dataset is small.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling and t-SNE.",
    )
    return parser


def stratified_sample(frame: pd.DataFrame, target_column: str, max_samples: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_samples:
        return frame

    sample_fraction = max_samples / len(frame)
    sampled_groups = []

    for _, group in frame.groupby(target_column, sort=False):
        group_size = max(1, round(len(group) * sample_fraction))
        sampled_groups.append(group.sample(n=min(group_size, len(group)), random_state=random_state))

    sampled_frame = pd.concat(sampled_groups, axis=0)
    if len(sampled_frame) > max_samples:
        sampled_frame = sampled_frame.sample(n=max_samples, random_state=random_state)

    return sampled_frame.sample(frac=1.0, random_state=random_state)


def select_representative_files(csv_files: list[Path]) -> list[Path]:
    selected_files: dict[tuple[str, str, int], Path] = {}

    for csv_file in csv_files:
        match = CSV_NAME_PATTERN.match(csv_file.name)
        if match is None:
            continue

        encoding = match.group("encoding")
        method = match.group("method")
        threshold = int(match.group("threshold"))
        combo_key = (encoding, method, threshold)

        if combo_key not in selected_files:
            selected_files[combo_key] = csv_file

    return [
        selected_files[key]
        for key in sorted(selected_files, key=lambda item: (item[2], item[0], item[1]))
    ]


def compute_embedding(features: pd.DataFrame, perplexity: float, random_state: int) -> pd.DataFrame:
    if features.shape[0] < 3:
        raise ValueError("t-SNE needs at least 3 rows.")

    n_components = min(MAX_PCA_COMPONENTS, features.shape[1], features.shape[0] - 1)
    reduced = features

    if n_components >= 2 and features.shape[1] > n_components:
        reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(features)
    elif n_components >= 2:
        reduced = features.to_numpy()
    else:
        reduced = features.to_numpy()

    effective_perplexity = min(perplexity, max(2.0, (features.shape[0] - 1) / 3.0))
    effective_perplexity = min(effective_perplexity, features.shape[0] - 1)

    if effective_perplexity < 1:
        raise ValueError("Dataset is too small for t-SNE after preprocessing.")

    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(reduced)

    return pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"])


def plot_embedding(frame: pd.DataFrame, title: str, output_path: Path) -> None:
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "0 = not delayed", 1: "1 = delayed"}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)

    for class_value, group in frame.groupby(TARGET_COLUMN, sort=True):
        ax.scatter(
            group["tsne_1"],
            group["tsne_2"],
            s=12,
            alpha=0.7,
            c=colors.get(int(class_value), "#666666"),
            label=f"{labels.get(int(class_value), str(class_value))} (n={len(group)})",
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_file(csv_path: Path, output_dir: Path, max_samples: int | None, perplexity: float, random_state: int) -> None:
    frame = pd.read_csv(csv_path)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {csv_path.name}")

    frame = frame.dropna().copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)

    if max_samples is not None:
        frame = stratified_sample(frame, TARGET_COLUMN, max_samples=max_samples, random_state=random_state)

    features = frame.drop(columns=[TARGET_COLUMN])
    numeric_features = features.select_dtypes(include="number")

    if numeric_features.shape[1] != features.shape[1]:
        dropped_columns = sorted(set(features.columns) - set(numeric_features.columns))
        raise ValueError(
            f"{csv_path.name} contains non-numeric columns that cannot be used for t-SNE: {dropped_columns}"
        )

    embedding = compute_embedding(numeric_features, perplexity=perplexity, random_state=random_state)
    plot_frame = pd.concat([embedding, frame[[TARGET_COLUMN]].reset_index(drop=True)], axis=1)

    output_path = output_dir / f"{csv_path.stem}.png"
    plot_embedding(plot_frame, title=csv_path.stem, output_path=output_path)
    print(f"Saved {output_path}")


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    csv_files = select_representative_files(csv_files)
    if not csv_files:
        raise FileNotFoundError(
            f"No sanitized CSV files matching the expected naming pattern were found in {input_dir}"
        )

    print(f"Processing {len(csv_files)} representative CSV files: {[csv_file.name for csv_file in csv_files]}")

    for csv_file in csv_files:
        process_file(
            csv_file,
            output_dir=output_dir,
            max_samples=args.max_samples,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()