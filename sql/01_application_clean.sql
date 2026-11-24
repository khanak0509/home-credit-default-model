CREATE OR REPLACE VIEW application_clean AS
SELECT
    * REPLACE (
        CASE WHEN DAYS_EMPLOYED = ${days_employed_sentinel} THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        CASE WHEN CODE_GENDER = '${gender_unknown_value}' THEN NULL ELSE CODE_GENDER END AS CODE_GENDER
    ),
    CAST(DAYS_EMPLOYED = ${days_employed_sentinel} AS INTEGER) AS DAYS_EMPLOYED_ANOMALY
FROM application_train;
