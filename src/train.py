import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CAT_FEATURES,
    COST_CHURN,
    COST_RETENTION,
    CV_FOLDS,
    ID_COL,
    METADATA_PATH,
    MODEL_PATH,
    NUM_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RF_PARAMS,
    TARGET_COL,
    TEST_SIZE,
)



def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(**RF_PARAMS)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_cost(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> int:
    y_pred = (y_proba >= threshold).astype(int)
    _, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int((tp + fp) * COST_RETENTION + fn * COST_CHURN)


def find_optimal_threshold(y_true: pd.Series, oof_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = [calculate_cost(y_true, oof_proba, float(t)) for t in thresholds]
    best_idx = int(np.argmin(costs))
    return round(float(thresholds[best_idx]), 3)


def train() -> None:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Вітрину даних не знайдено: {PROCESSED_DATA_PATH}. "
            "Спочатку запусти python -m src.make_dataset"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Стратифікована крос-валідація на Train
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=-1,
    )

    # Підбір оптимального порогу на out-of-fold ймовірностях Train
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    optimal_threshold = find_optimal_threshold(y_train, oof_proba)

    # Фінальне навчання на X_train та перевірка на відкладеному X_test
    pipeline.fit(X_train, y_train)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    test_roc_auc = round(float(roc_auc_score(y_test, y_test_proba)), 4)
    test_pr_auc = round(float(average_precision_score(y_test, y_test_proba)), 4)

    baseline_cost = int(y_test.sum() * COST_CHURN)
    default_cost = calculate_cost(y_test, y_test_proba, 0.50)
    optimal_cost = calculate_cost(y_test, y_test_proba, optimal_threshold)

    # Збереження артефактів
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cv_roc_auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
        "cv_pr_auc_mean": round(float(cv_scores["test_pr_auc"].mean()), 4),
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "optimal_threshold": optimal_threshold,
        "business_metrics_test": {
            "cost_retention": COST_RETENTION,
            "cost_churn": COST_CHURN,
            "cost_no_model": baseline_cost,
            "cost_default_threshold_0_5": default_cost,
            "cost_optimal_threshold": optimal_cost,
            "savings_vs_default_threshold": default_cost - optimal_cost,
            "savings_vs_no_model": baseline_cost - optimal_cost,
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    print("Навчання завершено успішно!")
    print(f"CV ROC-AUC:   {metadata['cv_roc_auc_mean']} | CV PR-AUC:   {metadata['cv_pr_auc_mean']}")
    print(f"Test ROC-AUC: {test_roc_auc} | Test PR-AUC: {test_pr_auc}")
    print(f"Оптимальний поріг (t*): {optimal_threshold}")
    print(
        f"Економія на Test vs поріг 0.50: "
        f"${metadata['business_metrics_test']['savings_vs_default_threshold']:,}"
    )
    print(f"Модель збережено:    {MODEL_PATH}")
    print(f"Метадані збережено:  {METADATA_PATH}")


if __name__ == "__main__":
    train()