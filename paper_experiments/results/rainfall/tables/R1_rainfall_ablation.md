District level, train 2009-2014 (n=92347), test 2015-2020 (n=106535); 5 seeds; cluster bootstrap over district-crop series.

| Model                    | Rainfall input             | R2            | R2 95% CI      |   Within-crop R2 |
|:-------------------------|:---------------------------|:--------------|:---------------|-----------------:|
| XGBoost                  | No rainfall                | 0.850 ± 0.000 | [0.842, 0.858] |            0.33  |
| XGBoost                  | State annual rainfall      | 0.844 ± 0.001 | [0.836, 0.852] |            0.305 |
| XGBoost                  | District seasonal rainfall | 0.851 ± 0.001 | [0.842, 0.858] |            0.333 |
| Persistence + correction | No rainfall                | 0.883 ± 0.001 | [0.875, 0.889] |            0.476 |
| Persistence + correction | State annual rainfall      | 0.883 ± 0.000 | [0.876, 0.889] |            0.478 |
| Persistence + correction | District seasonal rainfall | 0.885 ± 0.000 | [0.878, 0.891] |            0.487 |
| Persistence (rolling)    | —                          | 0.868 ± 0.000 | [0.859, 0.875] |            0.41  |
