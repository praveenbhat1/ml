Wilcoxon signed-rank on absolute errors (averaged over 5 seeds). Record-level pairs every test record; series-level pairs the mean error of each of the 886 State-Crop series (reported in the paper).

| Model A                 | Model B                           |   Median |err| A |   Median |err| B |   p (record-level) | Series where A better   |   p (series-level) |
|:------------------------|:----------------------------------|-----------------:|-----------------:|-------------------:|:------------------------|-------------------:|
| XGBoost + yield history | Last observed yield (persistence) |           0.1648 |           0.144  |           3.6e-28  | 347/886                 |            6.3e-10 |
| XGBoost + yield history | XGBoost                           |           0.1648 |           0.1924 |           1e-22    | 546/886                 |            9.8e-08 |
| XGBoost                 | LightGBM                          |           0.1924 |           0.2093 |           3.5e-23  | 511/886                 |            9.5e-09 |
| XGBoost                 | Random Forest                     |           0.1924 |           0.1978 |           0.021    | 428/886                 |            0.13    |
| XGBoost                 | Ridge                             |           0.1924 |           0.3037 |           1.9e-142 | 565/886                 |            3.8e-30 |
| XGBoost                 | State + Crop mean                 |           0.1924 |           0.2354 |           1.5e-20  | 488/886                 |            0.00094 |
| XGBoost                 | Last observed yield (persistence) |           0.1924 |           0.144  |           1.8e-47  | 322/886                 |            6e-17   |
