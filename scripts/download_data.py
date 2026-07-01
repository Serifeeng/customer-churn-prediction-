"""Download the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_data():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    download_data()
