from pathlib import Path



# Корінь проєкту 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Директорії та файли даних
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "churn_features.csv"
PREDICTIONS_PATH = DATA_DIR / "processed" / "churn_predictions.csv"
DB_PATH = DATA_DIR / "churn.db"

# SQL-скрипти
SQL_DIR = PROJECT_ROOT / "sql"
EXTRACT_SQL_PATH = SQL_DIR / "extract_features.sql"

# Збережені моделі та метадані
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Графіки 
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PR_ROC_CURVE_PATH = FIGURES_DIR / "roc_pr_curves.png"
THRESHOLD_PLOT_PATH = FIGURES_DIR / "threshold_cost.png"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"

# Колонки та ознаки
ID_COL = "customerID"
TARGET_COL = "Churn"

NUM_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "active_addons_count",
    "MonthlyCharges",
    "TotalCharges",
    "charge_to_contract_avg_ratio",
    "diff_from_internet_avg",
    "charge_pct_rank_in_contract",
]

CAT_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "tenure_group",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Параметри моделювання та валідації
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Матриця бізнес-витрат ($)
COST_RETENTION = 20   # Вартість пропозиції на утримання (знижка / бонус)
COST_CHURN = 100      # Втрата від відтоку клієнта (CAC / втрачений LTV)