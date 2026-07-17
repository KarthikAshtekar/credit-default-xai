# Group Calibration by SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

Calibration checks whether the same predicted default-risk score corresponds to similar observed default rates across groups.

Miscalibration across groups is a governance concern because the same score may imply different realized default risk for different groups. It does not prove causal bias.

Largest absolute bin-level calibration gap in this artifact: `0.0863`.

| sex_group | sex_code | group | bin | n | mean_predicted_probability | observed_default_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 1 | Male (SEX=1) | 0.00-0.10 | 623 | 0.06600187740449438 | 0.07223113964686999 | 0.006229262242375605 |
| Male | 1 | Male (SEX=1) | 0.10-0.20 | 803 | 0.14489681826027398 | 0.1643835616438356 | 0.019486743383561628 |
| Male | 1 | Male (SEX=1) | 0.20-0.30 | 315 | 0.2400682561904762 | 0.23174603174603176 | -0.008322224444444443 |
| Male | 1 | Male (SEX=1) | 0.30-0.40 | 202 | 0.3467757544059406 | 0.3415841584158416 | -0.005191595990098996 |
| Male | 1 | Male (SEX=1) | 0.40-0.50 | 107 | 0.4482239881308411 | 0.514018691588785 | 0.06579470345794391 |
| Male | 1 | Male (SEX=1) | 0.50-0.60 | 59 | 0.541042810338983 | 0.5932203389830508 | 0.05217752864406777 |
| Male | 1 | Male (SEX=1) | 0.60-0.70 | 94 | 0.6540819509574468 | 0.648936170212766 | -0.005145780744680861 |
| Male | 1 | Male (SEX=1) | 0.70-0.80 | 115 | 0.7483096026086956 | 0.7130434782608696 | -0.035266124347826056 |
| Male | 1 | Male (SEX=1) | 0.80-0.90 | 33 | 0.8332970254545454 | 0.8181818181818182 | -0.015115207272727194 |
| Male | 1 | Male (SEX=1) | 0.90-1.00 | 0 |  |  |  |
| Female | 2 | Female (SEX=2) | 0.00-0.10 | 1182 | 0.06630628605922165 | 0.0583756345177665 | -0.007930651541455148 |
| Female | 2 | Female (SEX=2) | 0.10-0.20 | 1294 | 0.14356787461128284 | 0.1553323029366306 | 0.01176442832534777 |
| Female | 2 | Female (SEX=2) | 0.20-0.30 | 447 | 0.24056601702460848 | 0.22595078299776286 | -0.014615234026845614 |
| Female | 2 | Female (SEX=2) | 0.30-0.40 | 225 | 0.3438005252888889 | 0.3333333333333333 | -0.010467191955555566 |
| Female | 2 | Female (SEX=2) | 0.40-0.50 | 116 | 0.45222193439655173 | 0.46551724137931033 | 0.013295306982758603 |
| Female | 2 | Female (SEX=2) | 0.50-0.60 | 70 | 0.5413383828571429 | 0.5714285714285714 | 0.030090188571428533 |
| Female | 2 | Female (SEX=2) | 0.60-0.70 | 129 | 0.6599458210077519 | 0.5736434108527132 | -0.0863024101550387 |
| Female | 2 | Female (SEX=2) | 0.70-0.80 | 160 | 0.7439342405 | 0.69375 | -0.05018424050000003 |
| Female | 2 | Female (SEX=2) | 0.80-0.90 | 28 | 0.8343002582142857 | 0.8214285714285714 | -0.012871686785714265 |
| Female | 2 | Female (SEX=2) | 0.90-1.00 | 0 |  |  |  |
