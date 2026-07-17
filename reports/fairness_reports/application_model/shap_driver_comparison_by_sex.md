# SHAP Driver Comparison by SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

SHAP comparison is diagnostic and approximate. It helps identify whether risk explanations differ across groups, but it is not causal proof.

This artifact compares mean absolute SHAP values by `SEX` for a bounded held-out test sample.

| sex_code | sex_group | group | feature | mean_abs_shap | mean_shap | n | mean_abs_shap_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Male | Male (SEX=1) | MaxPaymentDelay | 0.4669584631919861 | -0.01674574241042137 | 600 | 0.012964516878128052 |
| 2 | Female | Female (SEX=2) | MaxPaymentDelay | 0.45399394631385803 | -0.06456013023853302 | 600 | 0.012964516878128052 |
| 2 | Female | Female (SEX=2) | PAY_0 | 0.23529283702373505 | -0.08919865638017654 | 600 | -0.003527820110321045 |
| 1 | Male | Male (SEX=1) | PAY_0 | 0.231765016913414 | -0.08646383881568909 | 600 | -0.003527820110321045 |
| 2 | Female | Female (SEX=2) | BILL_AMT1 | 0.11758984625339508 | -0.026201795786619186 | 600 | -0.0011207833886146545 |
| 1 | Male | Male (SEX=1) | BILL_AMT1 | 0.11646906286478043 | -0.019912192597985268 | 600 | -0.0011207833886146545 |
| 1 | Male | Male (SEX=1) | NumDelayedMonths | 0.09597199410200119 | -0.023861385881900787 | 600 | 0.005298733711242676 |
| 2 | Female | Female (SEX=2) | NumDelayedMonths | 0.09067326039075851 | -0.03980674222111702 | 600 | 0.005298733711242676 |
| 1 | Male | Male (SEX=1) | AvgPaymentAmount | 0.08097777515649796 | -0.01801471598446369 | 600 | 0.00762581080198288 |
| 1 | Male | Male (SEX=1) | PAY_AMT2 | 0.0762077048420906 | -0.018435148522257805 | 600 | 0.004230372607707977 |
| 1 | Male | Male (SEX=1) | LIMIT_BAL | 0.07487637549638748 | -0.00018633146828506142 | 600 | 0.005830980837345123 |
| 2 | Female | Female (SEX=2) | AvgPaymentAmount | 0.07335196435451508 | -0.013689244166016579 | 600 | 0.00762581080198288 |
| 2 | Female | Female (SEX=2) | RecentPaymentDelay | 0.07291384041309357 | -0.02357575111091137 | 600 | -0.00012221187353134155 |
| 1 | Male | Male (SEX=1) | RecentPaymentDelay | 0.07279162853956223 | -0.023902367800474167 | 600 | -0.00012221187353134155 |
| 2 | Female | Female (SEX=2) | PAY_AMT2 | 0.07197733223438263 | -0.01474485732614994 | 600 | 0.004230372607707977 |
| 2 | Female | Female (SEX=2) | LIMIT_BAL | 0.06904539465904236 | -0.008774343878030777 | 600 | 0.005830980837345123 |
| 1 | Male | Male (SEX=1) | AvgBillAmount | 0.06814739108085632 | -0.004773986525833607 | 600 | 0.0029672980308532715 |
| 2 | Female | Female (SEX=2) | AvgBillAmount | 0.06518009305000305 | -0.009328151121735573 | 600 | 0.0029672980308532715 |
| 1 | Male | Male (SEX=1) | BillToLimitRatio_1 | 0.0649140328168869 | -0.0011966212186962366 | 600 | 0.0005067586898803711 |
| 2 | Female | Female (SEX=2) | BillToLimitRatio_1 | 0.06440727412700653 | -0.012080920860171318 | 600 | 0.0005067586898803711 |
