# src/get_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

RAW_DIR = Path("data/raw") #python src/get_data.py --source_csv "/content/drive/MyDrive/Dataset DSCI 510/combined_nhis_dataset_with_fraud_types (1).csv" --out_name "combined_nhis.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, required=True, help="Path to the input CSV on your machine or Drive mount")
    parser.add_argument("--out_name", type=str, default="claims_raw.csv", help="Output filename inside data/raw")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    src = Path(args.source_csv)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst = RAW_DIR / args.out_name
    shutil.copy2(src, dst)
    print(f"Saved raw data to: {dst.resolve()}")

if __name__ == "__main__":
    main()
