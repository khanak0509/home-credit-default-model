WITH per_installment AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_VERSION,
        NUM_INSTALMENT_NUMBER,
        min(DAYS_INSTALMENT) AS days_due,
        max(DAYS_ENTRY_PAYMENT) AS days_paid,
        max(AMT_INSTALMENT) AS amt_due,
        sum(AMT_PAYMENT) AS amt_paid
    FROM installments_valid
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER
),
scored AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        days_due,
        amt_paid,
        CASE WHEN days_paid IS NULL THEN NULL ELSE greatest(days_paid - days_due, 0) END AS dpd,
        amt_paid / nullif(amt_due, 0) AS payment_ratio,
        greatest(amt_due - coalesce(amt_paid, 0), 0) AS underpayment
    FROM per_installment
)
SELECT
    SK_ID_CURR,
    count(*) FILTER (WHERE days_due >= -90) AS INST_COUNT_3M,
    count(*) FILTER (WHERE days_due >= -90 AND dpd > 0) AS INST_LATE_COUNT_3M,
    avg(CAST(dpd > 0 AS DOUBLE)) FILTER (WHERE days_due >= -90) AS INST_LATE_SHARE_3M,
    max(dpd) FILTER (WHERE days_due >= -90) AS INST_MAX_DPD_3M,
    avg(dpd) FILTER (WHERE days_due >= -90) AS INST_MEAN_DPD_3M,
    avg(payment_ratio) FILTER (WHERE days_due >= -90) AS INST_MEAN_PAYMENT_RATIO_3M,
    sum(underpayment) FILTER (WHERE days_due >= -90) AS INST_TOTAL_UNDERPAYMENT_3M,
    count(*) FILTER (WHERE days_due >= -90 AND dpd > 30) AS INST_DPD30_COUNT_3M,
    count(*) FILTER (WHERE days_due >= -180) AS INST_COUNT_6M,
    count(*) FILTER (WHERE days_due >= -180 AND dpd > 0) AS INST_LATE_COUNT_6M,
    avg(CAST(dpd > 0 AS DOUBLE)) FILTER (WHERE days_due >= -180) AS INST_LATE_SHARE_6M,
    max(dpd) FILTER (WHERE days_due >= -180) AS INST_MAX_DPD_6M,
    avg(dpd) FILTER (WHERE days_due >= -180) AS INST_MEAN_DPD_6M,
    avg(payment_ratio) FILTER (WHERE days_due >= -180) AS INST_MEAN_PAYMENT_RATIO_6M,
    sum(underpayment) FILTER (WHERE days_due >= -180) AS INST_TOTAL_UNDERPAYMENT_6M,
    count(*) FILTER (WHERE days_due >= -180 AND dpd > 30) AS INST_DPD30_COUNT_6M,
    count(*) FILTER (WHERE days_due >= -365) AS INST_COUNT_12M,
    count(*) FILTER (WHERE days_due >= -365 AND dpd > 0) AS INST_LATE_COUNT_12M,
    avg(CAST(dpd > 0 AS DOUBLE)) FILTER (WHERE days_due >= -365) AS INST_LATE_SHARE_12M,
    max(dpd) FILTER (WHERE days_due >= -365) AS INST_MAX_DPD_12M,
    avg(dpd) FILTER (WHERE days_due >= -365) AS INST_MEAN_DPD_12M,
    avg(payment_ratio) FILTER (WHERE days_due >= -365) AS INST_MEAN_PAYMENT_RATIO_12M,
    sum(underpayment) FILTER (WHERE days_due >= -365) AS INST_TOTAL_UNDERPAYMENT_12M,
    count(*) FILTER (WHERE days_due >= -365 AND dpd > 30) AS INST_DPD30_COUNT_12M,
    count(*) FILTER (WHERE days_due >= -730) AS INST_COUNT_24M,
    count(*) FILTER (WHERE days_due >= -730 AND dpd > 0) AS INST_LATE_COUNT_24M,
    avg(CAST(dpd > 0 AS DOUBLE)) FILTER (WHERE days_due >= -730) AS INST_LATE_SHARE_24M,
    max(dpd) FILTER (WHERE days_due >= -730) AS INST_MAX_DPD_24M,
    avg(dpd) FILTER (WHERE days_due >= -730) AS INST_MEAN_DPD_24M,
    avg(payment_ratio) FILTER (WHERE days_due >= -730) AS INST_MEAN_PAYMENT_RATIO_24M,
    sum(underpayment) FILTER (WHERE days_due >= -730) AS INST_TOTAL_UNDERPAYMENT_24M,
    count(*) FILTER (WHERE days_due >= -730 AND dpd > 30) AS INST_DPD30_COUNT_24M,
    count(*) AS INST_COUNT_ALL,
    count(*) FILTER (WHERE dpd > 0) AS INST_LATE_COUNT_ALL,
    avg(CAST(dpd > 0 AS DOUBLE)) AS INST_LATE_SHARE_ALL,
    max(dpd) AS INST_MAX_DPD_ALL,
    avg(dpd) AS INST_MEAN_DPD_ALL,
    avg(payment_ratio) AS INST_MEAN_PAYMENT_RATIO_ALL,
    sum(underpayment) AS INST_TOTAL_UNDERPAYMENT_ALL,
    count(*) FILTER (WHERE dpd > 30) AS INST_DPD30_COUNT_ALL,
    count(*) FILTER (WHERE amt_paid IS NULL) AS INST_UNPAID_COUNT,
    count(DISTINCT SK_ID_PREV) AS INST_PREVIOUS_LOAN_COUNT,
    -max(days_due) FILTER (WHERE dpd > 0) AS INST_DAYS_SINCE_LAST_LATE,
    -max(days_due) AS INST_DAYS_SINCE_LAST_INSTALMENT
FROM scored
GROUP BY SK_ID_CURR
