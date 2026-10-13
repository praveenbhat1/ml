Wilcoxon signed-rank on series-level mean absolute errors.

| Model A                 | Model B                           |   Median |err| A |   Median |err| B | Series where A better   |   p (series-level) |
|:------------------------|:----------------------------------|-----------------:|-----------------:|:------------------------|-------------------:|
| XGBoost + yield history | Last observed yield (persistence) |           0.2425 |           0.2407 | 9128/18616              |           8e-11    |
| XGBoost + yield history | XGBoost                           |           0.2425 |           0.279  | 10510/18616             |           2.1e-57  |
| XGBoost                 | LightGBM                          |           0.279  |           0.2931 | 9667/18616              |           1.1e-20  |
| XGBoost                 | Random Forest                     |           0.279  |           0.2438 | 8078/18616              |           6.2e-57  |
| XGBoost                 | Ridge                             |           0.279  |           0.3497 | 10721/18616             |           3.3e-215 |
| XGBoost                 | District + Crop mean              |           0.279  |           0.3011 | 9656/18616              |           3.9e-29  |
| XGBoost                 | Last observed yield (persistence) |           0.279  |           0.2407 | 8393/18616              |           6e-09    |
