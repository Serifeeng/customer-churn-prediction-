import pandas as pd
from src.config import RAW_DATA_PATH


def load_data() -> pd.DataFrame:
    """Load raw Telco Customer Churn dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Download it from the link in README.md and place it there."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    return df
