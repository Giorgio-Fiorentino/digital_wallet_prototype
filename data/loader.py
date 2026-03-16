"""
data/loader.py

Adapter: reads Kaggle fraud-detection dataset, normalises to internal schema.
Categories extracted DYNAMICALLY from the dataset — no hardcoding.
Reproducible with any future card dataset that has a category column.

Dataset: kaggle.com/datasets/kartik2112/fraud-detection
Usage: python3 data/loader.py
"""

import pandas as pd
import numpy as np
import os
import re

np.random.seed(42)

SOURCES = ["Visa Classic", "Mastercard Gold", "Revolut", "PayPal", "Trade Republic"]

SOURCE_TYPES = {
    "Visa Classic":     "credit_card",
    "Mastercard Gold":  "credit_card",
    "Revolut":          "digital_wallet",
    "PayPal":           "digital_wallet",
    "Trade Republic":   "investment_account",
}


def assign_source(cc_num: int) -> str:
    return SOURCES[int(str(cc_num)[-2:]) % len(SOURCES)]


def clean_category(raw_cat: str) -> str:
    cleaned = re.sub(r'_(pos|net)$', '', raw_cat)
    cleaned = cleaned.replace('_', ' ')
    cleaned = cleaned.strip().title()
    cleaned = cleaned.replace('And', '&')
    return cleaned


def clean_merchant_name(raw: str) -> str:
    name = raw.replace("fraud_", "").strip()
    return " ".join(w.capitalize() for w in name.split(",")[0].split())


def load_and_process(
    train_path:  str = "data/raw/fraudTrain.csv",
    test_path:   str = "data/raw/fraudTest.csv",
    sample_size: int = 5000,
    output_path: str = "data/processed/transactions.csv",
) -> pd.DataFrame:

    print("Loading Kaggle dataset...")
    dfs = []
    for path in [train_path, test_path]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            print(f"  Loaded {path}: {len(dfs[-1]):,} rows")
        else:
            print(f"  Warning: {path} not found, skipping.")

    if not dfs:
        raise FileNotFoundError(
            "No Kaggle data found.\n"
            "Run: python3 -m kaggle datasets download "
            "-d kartik2112/fraud-detection -p data/raw --unzip"
        )

    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Total raw rows: {len(raw):,}")

    unique_raw_cats = raw["category"].unique()
    CATEGORY_MAP = {cat: clean_category(cat) for cat in unique_raw_cats}
    print(f"\n  Categories found in dataset ({len(CATEGORY_MAP)}):")
    for raw_cat, clean_cat in sorted(CATEGORY_MAP.items()):
        print(f"    {raw_cat:<25} → {clean_cat}")

    raw_sampled = (
        raw.groupby("category", group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), max(1, int(sample_size * len(x) / len(raw)))),
            random_state=42
        ))
        .reset_index(drop=True)
    )
    print(f"\n  Sampled: {len(raw_sampled):,} rows (stratified by category)")

    df = pd.DataFrame()
    df["Transaction_ID"]  = [f"TXN{str(i).zfill(5)}" for i in range(len(raw_sampled))]
    df["Date"]            = pd.to_datetime(raw_sampled["trans_date_trans_time"])
    df["Raw_Description"] = raw_sampled["merchant"].apply(clean_merchant_name)
    df["Category"]        = raw_sampled["category"].map(CATEGORY_MAP)
    df["Amount"]          = raw_sampled["amt"].round(2)
    df["Card"]            = raw_sampled["cc_num"].apply(assign_source)
    df["Source_Type"]     = df["Card"].map(SOURCE_TYPES)
    df["Currency"]        = "USD"
    df["Is_Fraud"]        = raw_sampled["is_fraud"].astype(bool)

    merchant_counts    = df.groupby("Raw_Description")["Transaction_ID"].transform("count")
    df["Is_Recurring"] = merchant_counts >= 3

    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\n  Category breakdown:")
    print(df.groupby("Category")["Amount"].agg(["count", "sum"]).round(2).to_string())
    print(f"\n  Source breakdown:")
    print(df.groupby("Card")["Amount"].agg(["count", "sum"]).round(2).to_string())

    return df


if __name__ == "__main__":
    load_and_process()
