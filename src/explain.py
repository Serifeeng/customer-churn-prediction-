from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR


def get_feature_names(preprocessor):
    """Extract feature names after ColumnTransformer."""
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = (
        preprocessor.transformers_[1][1]
        .named_steps["encoder"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )
    return list(numeric_features) + list(categorical_features)


def explain_tree_model(model, preprocessor, model_name: str):
    """Plot and return feature importance for tree-based models."""
    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    _save_feature_plot(
        importance_df.head(10),
        value_column="Importance",
        title=f"Top Features — {model_name}",
        filename=f"feature_importance_{model_name.lower().replace(' ', '_')}.png",
    )

    print(f"\nTop Important Features ({model_name}):")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def explain_logistic_regression(model, preprocessor):
    """Plot and return coefficient importance for logistic regression."""
    feature_names = get_feature_names(preprocessor)
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "AbsCoefficient": np.abs(coefficients),
        }
    ).sort_values(by="AbsCoefficient", ascending=False)

    _save_feature_plot(
        coef_df.head(10),
        value_column="AbsCoefficient",
        title="Top Features — Logistic Regression",
        filename="feature_importance_logistic_regression.png",
    )

    print("\nTop Influential Features (Logistic Regression):")
    print(coef_df.head(10).to_string(index=False))

    return coef_df


def _save_feature_plot(df: pd.DataFrame, value_column: str, title: str, filename: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values(by=value_column, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["Feature"], plot_df[value_column], color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
