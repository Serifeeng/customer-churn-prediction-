from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CV_FOLDS,
    LOGISTIC_MAX_ITER,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    USE_CLASS_WEIGHT,
)
from src.preprocessing import compute_sample_weights


def _build_models():
    class_weight = "balanced" if USE_CLASS_WEIGHT else None

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=LOGISTIC_MAX_ITER,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def train_models(X_train, y_train):
    """Train multiple classifiers and return a dictionary of fitted models."""
    sample_weight = compute_sample_weights(y_train) if USE_CLASS_WEIGHT else None
    models = _build_models()

    for name, model in models.items():
        if name == "Gradient Boosting" and sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

    return models


def cross_validate_models(X_train, y_train):
    """Run stratified cross-validation for all models."""
    sample_weight = compute_sample_weights(y_train) if USE_CLASS_WEIGHT else None
    models = _build_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for name, model in models.items():
        fold_scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
        }

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            fold_model = _build_models()[name]

            if name == "Gradient Boosting" and sample_weight is not None:
                fold_weight = sample_weight[train_idx]
                fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weight)
            else:
                fold_model.fit(X_fold_train, y_fold_train)

            y_pred = fold_model.predict(X_fold_val)
            y_proba = fold_model.predict_proba(X_fold_val)[:, 1]

            fold_scores["accuracy"].append(accuracy_score(y_fold_val, y_pred))
            fold_scores["precision"].append(precision_score(y_fold_val, y_pred))
            fold_scores["recall"].append(recall_score(y_fold_val, y_pred))
            fold_scores["f1"].append(f1_score(y_fold_val, y_pred))
            fold_scores["roc_auc"].append(roc_auc_score(y_fold_val, y_proba))

        cv_results.append(
            {
                "Model": name,
                "CV Accuracy": sum(fold_scores["accuracy"]) / CV_FOLDS,
                "CV Precision": sum(fold_scores["precision"]) / CV_FOLDS,
                "CV Recall": sum(fold_scores["recall"]) / CV_FOLDS,
                "CV F1": sum(fold_scores["f1"]) / CV_FOLDS,
                "CV ROC-AUC": sum(fold_scores["roc_auc"]) / CV_FOLDS,
            }
        )

    return cv_results
