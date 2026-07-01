from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
]

LOGISTIC_MAX_ITER = 1000
RF_N_ESTIMATORS = 200

USE_CLASS_WEIGHT = True
USE_SMOTE = False

BEST_MODEL_METRIC = "ROC-AUC"
