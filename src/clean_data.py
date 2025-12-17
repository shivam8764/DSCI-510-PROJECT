# src/clean_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def cap_top_k(df: pd.DataFrame, col: str, k: int = 30) -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col].astype("string").str.strip().str.lower().fillna("missing")
    top = s.value_counts().nlargest(k).index
    df[col] = s.where(s.isin(top), "other")
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default=str(RAW_DIR / "claims_raw.csv"))
    parser.add_argument("--out_csv", type=str, default=str(PROCESSED_DIR / "claims_processed.csv"))
    parser.add_argument("--top_k_diagnosis", type=int, default=30)
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input raw CSV not found: {in_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    df.columns = df.columns.astype(str).str.strip().str.lower()

    if "fraud_type" not in df.columns:
        raise KeyError("Expected column fraud_type not found")

    s = df["fraud_type"].astype(str).str.strip().str.lower()
    df["fraud_binary"] = np.where(
        s.isin(["phantom billing", "ghost enrolee", "ghost enrollee"]),
        1,
        0
    ).astype(int)

    for c in ["date of encounter", "date of discharge"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "date of encounter" in df.columns and "date of discharge" in df.columns:
        df["length_of_stay_days"] = (df["date of discharge"] - df["date of encounter"]).dt.days
        df["admission_month"] = df["date of encounter"].dt.month
        df["admission_dayofweek"] = df["date of encounter"].dt.dayofweek

    df = cap_top_k(df, "diagnosis", k=args.top_k_diagnosis)

    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to: {out_path.resolve()}")
    print("Rows:", len(df), "Cols:", df.shape[1])
    print("Fraud rate:", round(df["fraud_binary"].mean(), 4))

if __name__ == "__main__":
    main()
