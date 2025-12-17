# src/run_analysis.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

TABLES_DIR = Path("outputs/tables")

def make_decile_table(y_true, y_prob, n_bins=10) -> pd.DataFrame:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_prob = pd.Series(np.asarray(y_prob).astype(float)).reset_index(drop=True)

    tmp = pd.DataFrame({"y_true": y_true, "score": y_prob})
    tmp["rank"] = tmp["score"].rank(method="first", ascending=False)
    tmp["decile_bin"] = pd.qcut(tmp["rank"], q=n_bins, labels=False).astype(int)

    overall_rate = tmp["y_true"].mean()
    total_pos = tmp["y_true"].sum()

    rows = []
    for k in range(1, n_bins + 1):
        sub = tmp[tmp["decile_bin"] < k]
        cnt = len(sub)
        pos = int(sub["y_true"].sum())
        rate = pos / cnt if cnt else np.nan
        recall = pos / total_pos if total_pos else np.nan
        lift = rate / overall_rate if overall_rate else np.nan
        rows.append({
            "decile": k * 10,
            "target": pos,
            "claims": cnt,
            "cum_recall": recall,
            "cum_rate": rate,
            "overall_rate": overall_rate,
            "lift": lift
        })
    return pd.DataFrame(rows)

def choose_threshold_for_recall(y_true, y_prob, low=0.80, high=0.90):
    thresholds = np.linspace(0.01, 0.99, 199)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "flag_rate": pred.mean()
        })
    df = pd.DataFrame(rows)
    band = df[(df["recall"] >= low) & (df["recall"] <= high)].copy()
    if band.empty:
        df["gap"] = (df["recall"] - 0.85).abs()
        band = df.sort_values(["gap", "precision", "f1"], ascending=[True, False, False])
    best = band.sort_values(["precision", "f1"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), df, band

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True, help="Processed CSV path")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    if "fraud_binary" not in df.columns:
        raise KeyError("fraud_binary not found. Run clean_data.py first.")

    y = df["fraud_binary"].astype(int)
    X = df.drop(columns=["fraud_binary"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_d = pd.get_dummies(X_train, dummy_na=True)
    X_test_d = pd.get_dummies(X_test, dummy_na=True).reindex(columns=X_train_d.columns, fill_value=0)

    X_train_d = X_train_d.fillna(0)
    X_test_d = X_test_d.fillna(0)

    lr = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1, tol=1e-3)
    lr.fit(X_train_d, y_train)

    train_prob = lr.predict_proba(X_train_d)[:, 1]
    test_prob = lr.predict_proba(X_test_d)[:, 1]

    t_star, thr_all, thr_band = choose_threshold_for_recall(y_train, train_prob, low=0.80, high=0.90)

    train_pred = (train_prob >= t_star).astype(int)
    test_pred = (test_prob >= t_star).astype(int)

    results = {
        "t_star": t_star,
        "train_precision": precision_score(y_train, train_pred, zero_division=0),
        "train_recall": recall_score(y_train, train_pred, zero_division=0),
        "train_f1": f1_score(y_train, train_pred, zero_division=0),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
    }

    pd.DataFrame([results]).to_csv(TABLES_DIR / "metrics.csv", index=False)

    pd.DataFrame(confusion_matrix(y_train, train_pred), columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(
        TABLES_DIR / "confusion_train.csv"
    )
    pd.DataFrame(confusion_matrix(y_test, test_pred), columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(
        TABLES_DIR / "confusion_test.csv"
    )

    make_decile_table(y_train, train_prob).to_csv(TABLES_DIR / "deciles_train.csv", index=False)
    make_decile_table(y_test, test_prob).to_csv(TABLES_DIR / "deciles_test.csv", index=False)

    thr_all.to_csv(TABLES_DIR / "threshold_sweep_train.csv", index=False)
    thr_band.to_csv(TABLES_DIR / "threshold_band_train.csv", index=False)

    print("Saved tables to outputs/tables")
    print("Chosen t_star:", round(t_star, 4))
    print("Test recall:", round(results["test_recall"], 4), "Test precision:", round(results["test_precision"], 4))

if __name__ == "__main__":
    main()
