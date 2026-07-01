from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BEST_MODEL_METRIC, FIGURES_DIR, RESULTS_DIR


def _ensure_output_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models(models: dict, X_test, y_test):
    """Evaluate trained models and save comparison artifacts."""
    _ensure_output_dirs()
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        _save_confusion_matrix(name, cm)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }
        )

        print(f"\n{name}")
        print("Confusion Matrix:")
        print(cm)

    results_df = pd.DataFrame(results).sort_values(
        by=BEST_MODEL_METRIC, ascending=False
    )

    _save_roc_curves(models, X_test, y_test)
    results_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {results_path}")

    return results_df


def save_cv_results(cv_results: list):
    """Persist cross-validation summary to CSV."""
    _ensure_output_dirs()
    cv_df = pd.DataFrame(cv_results).sort_values(by="CV ROC-AUC", ascending=False)
    output_path = RESULTS_DIR / "cross_validation_results.csv"
    cv_df.to_csv(output_path, index=False)

    print("\nCross-Validation Summary:")
    print(cv_df.to_string(index=False))
    print(f"\nCross-validation results saved to {output_path}")

    return cv_df


def _save_confusion_matrix(model_name: str, cm):
    filename = model_name.lower().replace(" ", "_")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{filename}.png", dpi=150)
    plt.close(fig)


def _save_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)


def select_best_model(results_df: pd.DataFrame, models: dict):
    """Return the best model name and fitted estimator."""
    best_name = results_df.iloc[0]["Model"]
    return best_name, models[best_name]
