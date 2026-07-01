import argparse

from src.evaluate import evaluate_models, save_cv_results, select_best_model
from src.explain import explain_logistic_regression, explain_tree_model
from src.load_data import load_data
from src.preprocessing import preprocess_data
from src.save_model import save_artifacts
from src.train import cross_validate_models, train_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate customer churn prediction models."
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("[1/6] Loading data...")
    df = load_data()
    print(f"      Loaded {len(df):,} rows.")

    print("[2/6] Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print(f"      Train size: {len(y_train):,} | Test size: {len(y_test):,}")

    if not args.skip_cv:
        print("[3/6] Running cross-validation...")
        cv_results = cross_validate_models(X_train, y_train)
        save_cv_results(cv_results)
    else:
        print("[3/6] Skipping cross-validation.")

    print("[4/6] Training models...")
    models = train_models(X_train, y_train)

    print("[5/6] Evaluating models...")
    results_df = evaluate_models(models, X_test, y_test)

    print("[6/6] Generating explanations...")
    for name, model in models.items():
        if name == "Logistic Regression":
            explain_logistic_regression(model, preprocessor)
        elif name in ["Random Forest", "Gradient Boosting"]:
            explain_tree_model(model, preprocessor, name)

    best_name, best_model = select_best_model(results_df, models)
    save_artifacts(best_model, preprocessor, best_name)

    print("\nPipeline completed successfully.")
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
