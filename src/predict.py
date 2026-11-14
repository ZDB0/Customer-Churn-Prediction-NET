import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    COST_RETENTION,
    ID_COL,
    METADATA_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COL,
)



def assign_risk_segment(proba: float, threshold: float) -> str:
    if proba >= max(threshold * 1.8, 0.50):
        return "High Risk"
    if proba >= threshold:
        return "Medium Risk (Target)"
    return "Low Risk"


def predict(input_path: Path = PROCESSED_DATA_PATH, output_path: Path = PREDICTIONS_PATH) -> None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Модель або метадані не знайдено. Спочатку запусти python -m src.train"
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Вхідний файл не знайдено: {input_path}")

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    threshold = float(metadata["optimal_threshold"])

    df = pd.read_csv(input_path)
    drop_cols = [col for col in [ID_COL, TARGET_COL] if col in df.columns]
    X = df.drop(columns=drop_cols)

    probas = pipeline.predict_proba(X)[:, 1]
    preds = (probas >= threshold).astype(int)

    results = pd.DataFrame(
        {
            ID_COL: df[ID_COL] if ID_COL in df.columns else np.arange(len(df)),
            "churn_probability": np.round(probas, 4),
            "applied_threshold": threshold,
            "churn_prediction": preds,
            "risk_segment": [assign_risk_segment(p, threshold) for p in probas],
            "recommended_action": np.where(
                preds == 1,
                f"Offer retention bonus (${COST_RETENTION})",
                "No action needed",
            ),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    targeted_cnt = int(preds.sum())
    print(f"Інференс завершено для {len(results)} клієнтів (поріг t* = {threshold}).")
    print(
        f"Рекомендовано до утримання: {targeted_cnt} клієнтів "
        f"({targeted_cnt / len(results) * 100:.1f}% бази)."
    )
    print(f"Прогнози збережено у: {output_path}\n")
    print("Топ-5 клієнтів із найвищим ризиком відтоку:")
    print(results.sort_values("churn_probability", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Customer Churn Prediction")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH, help="Шлях до CSV з ознаками")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH, help="Шлях для збереження прогнозів")
    args = parser.parse_args()

    predict(input_path=args.input, output_path=args.output)