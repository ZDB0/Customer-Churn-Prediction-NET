import sqlite3
import pandas as pd
from src.config import DB_PATH, RAW_DATA_PATH



def load_raw_to_sqlite() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Файл не знайдено за шляхом: {RAW_DATA_PATH}. "
            "Переконайся, що датасет лежить у data/raw/telco_churn.csv"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Замінюємо пробіли в TotalCharges на 0.0 для нових клієнтів (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Бінаризуємо цільову змінну Churn: 'Yes' -> 1, 'No' -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    customers_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"
    ]
    services_cols = [
        "customerID", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    billing_cols = [
        "customerID", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn"
    ]

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df[customers_cols].to_sql("customers", conn, if_exists="replace", index=False)
        df[services_cols].to_sql("services", conn, if_exists="replace", index=False)
        df[billing_cols].to_sql("billing", conn, if_exists="replace", index=False)

    print(f"Базу даних створено: {DB_PATH}. Завантажено {len(df)} записів у 3 таблиці.")


if __name__ == "__main__":
    load_raw_to_sqlite()