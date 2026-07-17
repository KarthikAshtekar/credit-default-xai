# Proxy Predictability of SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

This diagnostic asks whether `SEX` can be predicted from non-sensitive credit variables.

Excluding `SEX` from the credit model does not automatically remove group-related signal. If `SEX` is predictable from other variables, those variables may carry proxy information.

Proxy predictability is not proof of legal discrimination. It indicates that direct removal of the protected attribute is insufficient as a complete fairness strategy.

Best proxy ROC-AUC observed: `0.6476`.

## Proxy model metrics

| model | status | positive_class_sex_code | positive_class_sex_group | positive_class | accuracy | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proxy_logistic_regression | completed | 2 | Female | Female (SEX=2) | 0.6191269576807731 | 0.5923933602384389 | 0.6781187232178182 |
| proxy_random_forest | completed | 2 | Female | Female (SEX=2) | 0.6089636787737421 | 0.6475617583081775 | 0.7306064536493715 |

## Top proxy-associated features

| model | feature | coefficient | absolute_score | feature_importance |
| --- | --- | --- | --- | --- |
| proxy_logistic_regression | AGE | -0.28256814595256813 | 0.28256814595256813 |  |
| proxy_logistic_regression | BillToLimitRatio_2 | -0.17453988043155622 | 0.17453988043155622 |  |
| proxy_logistic_regression | BILL_AMT5 | 0.16792416662206802 | 0.16792416662206802 |  |
| proxy_logistic_regression | BillToLimitRatio_6 | 0.14864407622024234 | 0.14864407622024234 |  |
| proxy_logistic_regression | MARRIAGE | -0.1474544521823949 | 0.1474544521823949 |  |
| proxy_logistic_regression | BILL_AMT6 | -0.14153696442957284 | 0.14153696442957284 |  |
| proxy_logistic_regression | PaymentToLimitRatio | -0.10730055149382016 | 0.10730055149382016 |  |
| proxy_logistic_regression | EDUCATION | 0.09938888919708852 | 0.09938888919708852 |  |
| proxy_logistic_regression | BILL_AMT2 | 0.08609343164485724 | 0.08609343164485724 |  |
| proxy_logistic_regression | BillToLimitRatio_3 | 0.079936293835554 | 0.079936293835554 |  |
| proxy_logistic_regression | BillToLimitRatio_1 | -0.0716680745699048 | 0.0716680745699048 |  |
| proxy_logistic_regression | BILL_AMT1 | -0.06492254383673823 | 0.06492254383673823 |  |
