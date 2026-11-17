WITH contract_stats AS (
    SELECT 
        Contract,
        COUNT(*) AS contract_customers_cnt,
        ROUND(AVG(MonthlyCharges), 2) AS avg_contract_charge
    FROM billing
    GROUP BY Contract
)
SELECT 
    c.customerID,
    c.gender,
    c.SeniorCitizen,
    c.Partner,
    c.Dependents,
    c.tenure,

    -- Сегментація клієнта за тривалістю користування послугами
    CASE 
        WHEN c.tenure = 0 THEN 'New_0m'
        WHEN c.tenure <= 12 THEN '0-12m'
        WHEN c.tenure <= 24 THEN '12-24m'
        WHEN c.tenure <= 48 THEN '24-48m'
        ELSE '48m+'
    END AS tenure_group,

    -- Кількість підключених додаткових сервісів
    (
        CASE WHEN s.OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.TechSupport = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.StreamingTV = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.StreamingMovies = 'Yes' THEN 1 ELSE 0 END
    ) AS active_addons_count,

    s.PhoneService,
    s.MultipleLines,
    s.InternetService,
    s.OnlineSecurity,
    s.OnlineBackup,
    s.DeviceProtection,
    s.TechSupport,
    s.StreamingTV,
    s.StreamingMovies,

    b.Contract,
    b.PaperlessBilling,
    b.PaymentMethod,
    b.MonthlyCharges,
    b.TotalCharges,

    -- Відношення платежу клієнта до середнього платежу за його типом контракту
    ROUND(b.MonthlyCharges / cs.avg_contract_charge, 3) AS charge_to_contract_avg_ratio,

    -- Відхилення платежу від середнього по типу інтернету 
    ROUND(
        b.MonthlyCharges - AVG(b.MonthlyCharges) OVER (PARTITION BY s.InternetService), 
        2
    ) AS diff_from_internet_avg,

    -- Процентильний ранг платежу всередині контракту 
    ROUND(
        PERCENT_RANK() OVER (PARTITION BY b.Contract ORDER BY b.MonthlyCharges), 
        3
    ) AS charge_pct_rank_in_contract,

    b.Churn
FROM customers c
INNER JOIN services s ON c.customerID = s.customerID
INNER JOIN billing b ON c.customerID = b.customerID
LEFT JOIN contract_stats cs ON b.Contract = cs.Contract;