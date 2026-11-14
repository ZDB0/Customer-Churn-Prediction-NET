import sqlite3
import pandas as pd
from src.config import DB_PATH, EXTRACT_SQL_PATH, PROCESSED_DATA_PATH



def build_feature_mart() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Базу даних не знайдено: {DB_PATH}. Спочатку запусти python -m src.ingest_data"
        )
    if not EXTRACT_SQL_PATH.exists():
        raise FileNotFoundError(f"SQL-файл не знайдено: {EXTRACT_SQL_PATH}")

    query = EXTRACT_SQL_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        df_features = pd.read_sql_query(query, conn)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(PROCESSED_DATA_PATH, index=False)

    print(
        f"Вітрину ознак збережено: {PROCESSED_DATA_PATH}. "
        f"Розмірність таблиці: {df_features.shape}"
    )


if __name__ == "__main__":
    build_feature_mart()