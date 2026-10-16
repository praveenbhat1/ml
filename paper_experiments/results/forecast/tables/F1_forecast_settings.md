Test 2015-2020; models trained once on 1997-2014; R2 mean ± s.d. over 5 seeds; 95% CI: cluster bootstrap over region-crop series (1000 resamples).

| Level    | Setting      | Method                   | R2            |   Within-crop R2 |   MAE |   Median |err| | R2 95% CI      |
|:---------|:-------------|:-------------------------|:--------------|-----------------:|------:|---------------:|:---------------|
| State    | Fixed origin | Persistence              | 0.908 ± 0.000 |            0.546 | 0.26  |         0.144  | [0.882, 0.928] |
| State    | Fixed origin | XGBoost + history        | 0.904 ± 0.001 |            0.526 | 0.281 |         0.1648 | [0.878, 0.923] |
| State    | Fixed origin | Persistence + correction | 0.908 ± 0.001 |            0.547 | 0.272 |         0.1597 | [0.883, 0.927] |
| State    | Rolling      | Persistence              | 0.932 ± 0.000 |            0.663 | 0.2   |         0.0876 | [0.912, 0.947] |
| State    | Rolling      | XGBoost + history        | 0.931 ± 0.001 |            0.66  | 0.23  |         0.128  | [0.913, 0.945] |
| State    | Rolling      | Persistence + correction | 0.935 ± 0.000 |            0.677 | 0.221 |         0.1205 | [0.917, 0.948] |
| District | Fixed origin | Persistence              | 0.817 ± 0.000 |            0.191 | 0.403 |         0.2407 | [0.807, 0.826] |
| District | Fixed origin | XGBoost + history        | 0.844 ± 0.002 |            0.308 | 0.383 |         0.2425 | [0.836, 0.852] |
| District | Fixed origin | Persistence + correction | 0.837 ± 0.001 |            0.277 | 0.392 |         0.2466 | [0.828, 0.845] |
| District | Rolling      | Persistence              | 0.866 ± 0.000 |            0.405 | 0.298 |         0.144  | [0.856, 0.873] |
| District | Rolling      | XGBoost + history        | 0.886 ± 0.001 |            0.497 | 0.314 |         0.1938 | [0.880, 0.892] |
| District | Rolling      | Persistence + correction | 0.883 ± 0.001 |            0.481 | 0.313 |         0.1919 | [0.876, 0.889] |
