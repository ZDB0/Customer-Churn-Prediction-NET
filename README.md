# Customer Churn Prediction Pipeline (SQL + ML + Business Threshold Optimization)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-CTE%20%26%20Window%20Functions-0F80CC?style=flat&logo=sqlite&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline%20%26%20CV-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-ROC--AUC%200.852-2EA44F?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-6A5ACD?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Tests-8CAAE6?style=flat&logo=scipy&logoColor=white)

End-to-end проєкт машинного навчання для прогнозування відтоку клієнтів телеком-оператора, побудований на базі локального сховища **SQLite**, інженерії ознак через **SQL (CTE, CASE WHEN, віконні функції)**, пайплайнів **Scikit-Learn** без витоку даних та оптимізації порогу класифікації під матрицю бізнес-витрат.

---

## 1. Бізнес-задача та архітектура рішення

Бізнес хоче зарабляти грошики, але не завджи можна найти "нові гаманці", які б з радістю відавали б свої шекеля компанії. У таких умовах постає нова задача не втратити поточного клієнта компанії. У телеком-індустрії залучення нового клієнта коштує у 4–5 разів дорожче, ніж утримання існуючого. 

### Архітектура пайплайна
1. **SQL Ingestion (`src/ingest_data.py`):** нормалізація сирого датасету Telco Customer Churn (7 043 клієнти) у 3 реляційні таблиці в SQLite (`customers`, `services`, `billing`).
2. **SQL Feature Engineering (`sql/extract_features.sql`):** агрегація аналітичної вітрини за допомогою `INNER/LEFT JOIN`, `CTE`, `CASE WHEN` та віконних функцій (`AVG() OVER`, `PERCENT_RANK() OVER`) для розрахунку відносних тарифних показників.
3. **EDA (`notebooks/01_eda_and_stats.ipynb`):** аналіз кількості клієнтів, яких компанія втратила (26.5% відтоку), підрахунок p-value за допомогою критерію Хі-квадрат ($\chi^2$) для категоріальних ознак та U-критерію Манна — Уїтні для числових показників.
4. **Обираємо кращу модель (`notebooks/02_modeling_and_cv.ipynb`):** досліджували три моделі: логістичну регресію, випадковий ліс та LightGBM. За сукупністю показників обрали випадковий ліс.
5. **ML Pipeline (`src/train.py`):** готовий .py код для тренування иоделі.
6. **ML Pipeline (`src/predict.py`)** формуємо .csv файл з передбаченнями по клієнтах(треба давати знижку, нічого не потрібно робити).


---

## 2. Порівняння моделей (Test Set)

Основними метриками якості обрано **ROC-AUC** та **PR-AUC** (*Average Precision*).

| Модель | Test ROC-AUC | Test PR-AUC | Особливості |
| :--- | :---: | :---: | :--- |
| **Random Forest** *(фінальна модель)* | **0.8521** | **0.6788** | Найкращий баланс узагальнення та ранжування ризиків |
| **Logistic Regression** | 0.8515 | 0.6634 | Сильний лінійний бейзлайн завдяки SQL-ознакам |
| **LightGBM** | 0.8404 | 0.6479 | Базова конфігурація градієнтного бустингу |

### Криві якості класифікації (ROC та Precision-Recall)
![ROC and PR Curves](reports/figures/roc_pr_curves.png)

---

## 3. Бізнес-оптимізація порогу та SHAP-інтерпретація

### Оптимізація порогу під матрицю витрат
Замість стандартного порогу `0.50` виконано пошук порогу $t^*$, що мінімізує функцію витрат:

$$\text{Total Cost}(t) = (TP(t) + FP(t)) \cdot \$20 + FN(t) \cdot \$100$$

Зниження порогу класифікації дозволяє виявити більшу частку клієнтів у зоні ризику та суттєво скоротити фінансові втрати бізнесу порівняно зі стандартним порогом `0.50` (точні параметри збережено у `models/model_metadata.json`).

![Business Threshold Optimization](reports/figures/threshold_cost.png)

### Ключові драйвери відтоку (SHAP Summary Plot)
![SHAP Summary Plot](reports/figures/shap_summary.png)

**Ключові інсайти для бізнесу:**
* **Тип контракту (`Contract_Month-to-month`):** найсильніший тригер відтоку — клієнти з помісячною оплатою мають найвищий ризик уходу, тоді як довгострокові контракти (1–2 роки) надійно утримують аудиторію.
* **Стаж клієнта (`tenure`):** найбільший ризик припадає на перші 12 місяців користування послугами.
* **Технологія підключення та чек (`InternetService_Fiber optic`, `charge_to_contract_avg_ratio`):** користувачі оптоволокна та клієнти, чий рахунок перевищує середній чек у їхньому сегменті, частіше схильні до зміни оператора.

---

## 4. Структура проєкту

```text
customer-churn-prediction/
├── data/                    # Локальна БД SQLite та сирі/оброблені дані (ігноруються Git)
├── sql/
│   └── extract_features.sql # SQL-запит з JOIN, CASE WHEN та віконними функціями
├── notebooks/
│   ├── 01_eda_and_stats.ipynb    # EDA та статистичні тести (Хі-квадрат, Манна — Уїтні)
│   └── 02_modeling_and_cv.ipynb  # Порівняння моделей, оптимізація порогу та SHAP
├── src/
│   ├── config.py            # Єдина конфігурація шляхів, ознак та бізнес-параметрів
│   ├── ingest_data.py       # Завантаження CSV у нормалізовані таблиці SQLite
│   ├── make_dataset.py      # Збірка аналітичної вітрини з SQLite
│   ├── train.py             # Навчання пайплайна, валідація та збереження моделі
│   └── predict.py           # Пакетний інференс та сегментація ризиків утримання
├── models/
│   └── model_metadata.json  # Зафіксовані метрики та оптимальний бізнес-поріг
├── reports/figures/         # Візуалізації для аналізу та документації
└── requirements.txt         # Залежності проєкту
```

---
## 5. Інструкція запуску

```text
1. Клонування репозиторію та налаштування середовища:

Bash
    git clone [https://github.com/](https://github.com/)<твій-нікнейм>/customer-churn-prediction.git
    cd customer-churn-prediction
    python -m venv .venv
    source .venv/bin/activate  # Для Windows PowerShell: .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

3. Підготовка даних (покласти telco_churn.csv у data/raw/):
 
  Bash
    python -m src.ingest_data
    python -m src.make_dataset

4. Навчання моделі та оптимізація бізнес-порогу:

  Bash
    python -m src.train

5. Генерація списку клієнтів для утримання:

   Bash
     python -m src.predict
   
