# src/visualize_results.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", type=str, default=str(TAB_DIR))
    args = parser.parse_args()

    tdir = Path(args.tables_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(tdir / "metrics.csv")
    dec_train = pd.read_csv(tdir / "deciles_train.csv")
    dec_test = pd.read_csv(tdir / "deciles_test.csv")

    plt.figure()
    plt.plot(dec_test["decile"], dec_test["cum_recall"], marker="o")
    plt.xlabel("Top percent of claims reviewed")
    plt.ylabel("Cumulative recall")
    plt.title("Cumulative Recall Curve on Test")
    plt.savefig(FIG_DIR / "cumulative_recall_test.png", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(dec_test["decile"], dec_test["lift"], marker="o")
    plt.xlabel("Top percent of claims reviewed")
    plt.ylabel("Lift")
    plt.title("Lift Curve on Test")
    plt.savefig(FIG_DIR / "lift_curve_test.png", bbox_inches="tight")
    plt.show()

    print("Saved figures to outputs/figures")
    print(metrics)

if __name__ == "__main__":
    main()
