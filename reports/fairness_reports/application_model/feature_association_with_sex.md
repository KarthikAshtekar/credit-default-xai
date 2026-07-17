# Feature Association with SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

This diagnostic compares feature distributions across `SEX` groups using group means, medians, missing rates, standardized mean differences, correlation, and mutual information.

Group differences in feature distributions can help explain outcome disparities. They may reflect portfolio composition, historical access to credit, socioeconomic patterns, or dataset artifacts. They do not by themselves prove unfair treatment.

## Largest absolute standardized mean differences

| feature | group_a_sex_code | group_a_sex_group | group_a | group_b_sex_code | group_b_sex_group | group_b | group_a_mean | group_b_mean | group_a_median | group_b_median | group_a_missing_rate | group_b_missing_rate | standardized_mean_difference | absolute_standardized_mean_difference | correlation_with_sex_binary | mutual_information_with_sex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BillToLimitRatio_1 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.46308026448272926 | 0.39797070205261414 | 0.41313216783216783 | 0.2539072413793103 | 0.0 | 0.0 | -0.1582928301283148 | 0.1582928301283148 | -0.07739950783228618 | 0.0014413568685744949 |
| BillToLimitRatio_2 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.44945653385512907 | 0.38597086430447175 | 0.387665 | 0.23671923076923077 | 0.0 | 0.0 | -0.15695848969062778 | 0.15695848969062778 | -0.07675770568861859 | 0.007119602689666715 |
| PAY_2 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | -0.029189098250336474 | -0.2024072438162544 | 0.0 | 0.0 | 0.0 | 0.0 | -0.14465913580483566 | 0.14465913580483566 | -0.07077100316682187 | 0.003948685086996084 |
| AvgBillToLimitRatio | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.4029499987793695 | 0.35342142154876816 | 0.3600641762452107 | 0.23259652777777778 | 0.0 | 0.0 | -0.1410340659408664 | 0.1410340659408664 | -0.06884502950487878 | 0.0028758922935054887 |
| BillToLimitRatio_3 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.42469513611527737 | 0.37085876033477255 | 0.3505626041666666 | 0.22286846153846154 | 0.0 | 0.0 | -0.13591094509424606 | 0.13591094509424606 | -0.0664220571228798 | 0.0038348637040381828 |
| PAY_3 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | -0.06855652759084792 | -0.2302893109540636 | 0.0 | 0.0 | 0.0 | 0.0 | -0.13506514434965924 | 0.13506514434965924 | -0.06609605640042963 | 0.0038815753656717966 |
| BillToLimitRatio_4 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.3868580217135952 | 0.34154807127481523 | 0.31105 | 0.19672436974789914 | 0.0 | 0.0 | -0.12274262523467562 | 0.12274262523467562 | -0.060111938556981076 | 0.0033450373365964126 |
| PAY_4 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | -0.13383243606998654 | -0.27766121908127206 | 0.0 | 0.0 | 0.0 | 0.0 | -0.12267205780903591 | 0.12267205780903591 | -0.060173238366203495 | 0.0038499820444120036 |
| MaxPaymentDelay | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.5352456258411844 | 0.37538648409893993 | 0.0 | 0.0 | 0.0 | 0.0 | -0.11852887960903198 | 0.11852887960903198 | -0.058128479765487136 | 0.00026052585314917387 |
| RecentPaymentDelay | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.063257065948856 | -0.06918065371024736 | 0.0 | 0.0 | 0.0 | 0.0 | -0.11786417178036686 | 0.11786417178036686 | -0.05764287886698632 | 0.007580626480706121 |
| PAY_0 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.063257065948856 | -0.06918065371024736 | 0.0 | 0.0 | 0.0 | 0.0 | -0.11786417178036686 | 0.11786417178036686 | -0.05764287886698632 | 0.0 |
| PAY_5 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | -0.18918236877523553 | -0.3167513250883392 | 0.0 | 0.0 | 0.0 | 0.0 | -0.11214276468213248 | 0.11214276468213248 | -0.05506388503522643 | 0.005505943138581326 |
| BillToLimitRatio_5 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.35579616142595166 | 0.3182162896399819 | 0.26897499999999996 | 0.1776329365079365 | 0.0 | 0.0 | -0.10694058779345907 | 0.10694058779345907 | -0.05243711056361324 | 0.005478830910216281 |
| BillToLimitRatio_6 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | 0.3378138750835348 | 0.30596384168595353 | 0.23349333333333333 | 0.15705285714285713 | 0.0 | 0.0 | -0.09204940327908155 | 0.09204940327908155 | -0.04511655513462472 | 0.012901807325744752 |
| PAY_6 | 1 | Male | Male (SEX=1) | 2 | Female | Female (SEX=2) | -0.22863391655450874 | -0.3321002650176678 | 0.0 | 0.0 | 0.0 | 0.0 | -0.08960704706547055 | 0.08960704706547055 | -0.04400778818765756 | 0.00040742278691152656 |
