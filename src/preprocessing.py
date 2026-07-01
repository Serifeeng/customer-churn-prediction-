import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    USE_SMOTE,
)


def preprocess_data(df: pd.DataFrame):
    """Clean, encode, and split the dataset for model training."""
    df = df.copy()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].fillna(
        df[NUMERICAL_FEATURES].median()
    )

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if USE_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def compute_sample_weights(y_train: pd.Series) -> np.ndarray:
    """Compute balanced sample weights for models without class_weight support."""
    class_counts = y_train.value_counts()
    total = len(y_train)
    weight_map = {label: total / (len(class_counts) * count) for label, count in class_counts.items()}
    return y_train.map(weight_map).to_numpy()
