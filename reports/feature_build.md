# Feature build

## Leakage filter

Behavioral rows with a positive day or month offset (after the application date) are removed before any aggregation. Bureau balance rows are also removed when their parent bureau record was removed by that filter, or when no bureau record exists for them at all (an orphan, not a leakage case).

| table | raw rows | future-dated rows | parent record filtered | no bureau record | kept rows |
|---|---|---|---|---|---|
| installments_payments | 13,605,401 | 0 | 0 | 0 | 13,605,401 |
| bureau | 1,716,428 | 17 | 0 | 0 | 1,716,411 |
| bureau_balance | 27,299,925 | 0 | 102 | 3,120,184 | 24,179,639 |
| previous_application | 1,670,214 | 0 | 0 | 0 | 1,670,214 |
| pos_cash_balance | 10,001,358 | 0 | 0 | 0 | 10,001,358 |
| credit_card_balance | 3,840,312 | 0 | 0 | 0 | 3,840,312 |

Previous applications keep their row when the decision date is on or before the application, but the five schedule dates (DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE, DAYS_TERMINATION) are set to null when positive: 224,392 rows carry a planned date after the application and 997,144 carry the 365243 placeholder for an event that never happened.

## Feature groups

| group | features |
|---|---|
| application | 138 |
| installments | 44 |
| bureau | 20 |
| bureau_balance | 29 |
| previous_application | 24 |
| pos_cash | 14 |
| credit_card | 18 |

Output: one row per SK_ID_CURR, 307,511 rows.
