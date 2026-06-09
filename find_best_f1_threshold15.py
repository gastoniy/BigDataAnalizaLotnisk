#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List


def find_best(log_dir: Path, threshold: int = 15, top: int = 1) -> List[dict]:
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    grouped: dict[tuple[str | None, str | None, str], list[dict]] = defaultdict(list)
    for p in sorted(log_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        ds = payload.get("dataset", {})
        if ds.get("threshold") != threshold:
            continue

        models = payload.get("models", {})
        for mname, mdata in models.items():
            mean_f1 = mdata.get("mean_f1")
            if mean_f1 is None:
                continue
            key = (ds.get("encoding"), ds.get("resampling"), mname)
            grouped[key].append(
                {
                    "file": str(p),
                    "mean_f1": float(mean_f1),
                    "mean_balanced_accuracy": float(mdata.get("mean_balanced_accuracy", 0)),
                    "run_index": ds.get("run_index"),
                }
            )

    candidates: List[dict] = []
    for (encoding, resampling, model), entries in grouped.items():
        mean_f1_values = [entry["mean_f1"] for entry in entries]
        mean_balanced_values = [entry["mean_balanced_accuracy"] for entry in entries]
        candidates.append(
            {
                "encoding": encoding,
                "resampling": resampling,
                "model": model,
                "run_count": len(entries),
                "mean_f1": float(sum(mean_f1_values) / len(mean_f1_values)),
                "std_f1": float((sum((value - (sum(mean_f1_values) / len(mean_f1_values))) ** 2 for value in mean_f1_values) / len(mean_f1_values)) ** 0.5),
                "mean_balanced_accuracy": float(sum(mean_balanced_values) / len(mean_balanced_values)),
                "std_balanced_accuracy": float((sum((value - (sum(mean_balanced_values) / len(mean_balanced_values))) ** 2 for value in mean_balanced_values) / len(mean_balanced_values)) ** 0.5),
                "runs": sorted(entry["run_index"] for entry in entries),
                "files": sorted(entry["file"] for entry in entries),
            }
        )

    candidates.sort(key=lambda x: x["mean_f1"], reverse=True)
    return candidates[:top]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best mean F1 for threshold in JSON logs")
    parser.add_argument("--logs-dir", type=Path, default=Path("traininglogs"), help="Directory with JSON logs")
    parser.add_argument("--threshold", type=int, default=15, help="Threshold in minutes to filter")
    parser.add_argument("--top", type=int, default=1, help="How many top results to show")
    parser.add_argument("--save", type=Path, help="Optional path to save results as JSON")
    args = parser.parse_args()

    results = find_best(args.logs_dir, threshold=args.threshold, top=args.top)
    if not results:
        print(f"No logs found for threshold={args.threshold} in {args.logs_dir}")
        return

    for i, r in enumerate(results, start=1):
        print(
            f"{i}. model={r['model']} | mean_f1={r['mean_f1']:.4f} | bal_acc={r['mean_balanced_accuracy']:.4f} | "
            f"encoding={r['encoding']} | resampling={r['resampling']} | runs={r['run_count']} | run_ids={r['runs']}"
        )

    if args.save:
        args.save.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
