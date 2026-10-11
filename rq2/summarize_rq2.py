"""Summarize RQ2 runs: mean +/- std test accuracy and F1 per model and setting.

Usage:
    python rq2/summarize_rq2.py              # seeds 42 43 44
    python rq2/summarize_rq2.py --seeds 42 43

Reads rq2/results/<model>_<setting>_seed<seed>/metrics.json, prints a markdown
table, and writes rq2/results/rq2_summary.json.
"""

import argparse
import json
import os

import numpy as np

from rq2_data import SETTINGS  # also puts repo root on sys.path

from dataset import LABEL_MAP  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SETTING_NAMES = {"input": "Input only", "output": "Output only", "both": "Input + Output"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    labels = list(LABEL_MAP)
    summary = []
    print("| Model | Setting | Test acc (%) | Per seed | Macro-F1 | "
          + " | ".join(f"{l} F1" for l in labels) + " |")
    print("|---|---|---|---|---|" + "---|" * len(labels))

    for model in ["cnn", "rnn"]:
        for setting in SETTINGS:
            runs = []
            for seed in args.seeds:
                path = os.path.join(RESULTS_DIR, f"{model}_{setting}_seed{seed}", "metrics.json")
                if os.path.exists(path):
                    with open(path) as f:
                        runs.append((seed, json.load(f)))
            if not runs:
                continue

            acc = np.array([r["test_acc"] for _, r in runs])
            f1 = np.array([r["classification_report"]["macro avg"]["f1-score"] for _, r in runs])
            per_class = {l: float(np.mean([r["classification_report"][l]["f1-score"] for _, r in runs]))
                         for l in labels}
            # Confusion matrices summed over seeds (rows = true, cols = predicted)
            cm = np.sum([r["confusion_matrix"] for _, r in runs], axis=0)
            std = acc.std(ddof=1) if len(acc) > 1 else 0.0

            summary.append({
                "model": model.upper(), "setting": SETTING_NAMES[setting],
                "seeds": [s for s, _ in runs],
                "test_acc_mean": float(acc.mean()), "test_acc_std": float(std),
                "test_acc_per_seed": acc.tolist(),
                "macro_f1_mean": float(f1.mean()),
                "per_class_f1_mean": per_class,
                "confusion_matrix_sum": cm.tolist(),
            })
            print(f"| {model.upper()} | {SETTING_NAMES[setting]} | {acc.mean():.1f} ± {std:.1f} | "
                  + " / ".join(f"{a:.1f}" for a in acc) + f" | {f1.mean():.3f} | "
                  + " | ".join(f"{per_class[l]:.3f}" for l in labels) + " |")

    with open(os.path.join(RESULTS_DIR, "rq2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
