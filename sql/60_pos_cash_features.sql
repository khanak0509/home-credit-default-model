WITH latest AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        arg_max(NAME_CONTRACT_STATUS, MONTHS_BALANCE) AS status,
        arg_max(CNT_INSTALMENT_FUTURE, MONTHS_BALANCE) AS remaining,
        arg_max(CNT_INSTALMENT, MONTHS_BALANCE) AS term
    FROM pos_cash_valid
    GROUP BY SK_ID_CURR, SK_ID_PREV
),
loans AS (
    SELECT
        SK_ID_CURR,
        count(*) AS POS_LOAN_COUNT,
        count(*) FILTER (WHERE status = 'Active') AS POS_ACTIVE_LOAN_COUNT,
        count(*) FILTER (WHERE status = 'Completed') AS POS_COMPLETED_LOAN_COUNT,
        sum(remaining) FILTER (WHERE status = 'Active') AS POS_REMAINING_INSTALMENTS,
        avg(remaining / nullif(term, 0)) FILTER (WHERE status = 'Active') AS POS_MEAN_REMAINING_SHARE
    FROM latest
    GROUP BY SK_ID_CURR
),
monthly AS (
    SELECT
        SK_ID_CURR,
        count(*) AS POS_MONTH_COUNT,
        max(SK_DPD) AS POS_MAX_DPD,
        max(SK_DPD_DEF) AS POS_MAX_DPD_DEF,
        max(SK_DPD) FILTER (WHERE MONTHS_BALANCE >= -12) AS POS_MAX_DPD_12M,
        avg(CAST(SK_DPD > 0 AS DOUBLE)) AS POS_DPD_MONTH_SHARE,
        avg(CAST(SK_DPD > 0 AS DOUBLE)) FILTER (WHERE MONTHS_BALANCE >= -12) AS POS_DPD_MONTH_SHARE_12M,
        avg(CAST(SK_DPD_DEF > 0 AS DOUBLE)) AS POS_DPD_DEF_MONTH_SHARE,
        -max(MONTHS_BALANCE) AS POS_MONTHS_SINCE_LAST_RECORD,
        -max(MONTHS_BALANCE) FILTER (WHERE SK_DPD > 0) AS POS_MONTHS_SINCE_LAST_DPD
    FROM pos_cash_valid
    GROUP BY SK_ID_CURR
)
SELECT *
FROM monthly
LEFT JOIN loans USING (SK_ID_CURR)
