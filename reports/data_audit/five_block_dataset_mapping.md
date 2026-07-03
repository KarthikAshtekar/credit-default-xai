# Five-Block Dataset Mapping

Primary dataset: UCI Default of Credit Card Clients / Taiwan credit-card default

| Block | Columns | Coverage |
| --- | --- | ---: |
| Borrower Profile | `SEX, EDUCATION, MARRIAGE, AGE` | 4/4 |
| Credit History | `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` | 6/6 |
| Loan / Exposure | `LIMIT_BAL` | 1/1 |
| Financial Health | `BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, BillToLimitRatio_1, BillToLimitRatio_2, BillToLimitRatio_3, BillToLimitRatio_4, BillToLimitRatio_5, BillToLimitRatio_6, AvgBillToLimitRatio, AvgPaymentToBillRatio, RecentPaymentDelay, MaxPaymentDelay, NumDelayedMonths, AvgBillAmount, AvgPaymentAmount, PaymentToLimitRatio` | 26/26 |
| Target | `Default_Flag` | 1/1 |

`Default_Flag` is the project-standard target name and equals 1 for next-month default.
