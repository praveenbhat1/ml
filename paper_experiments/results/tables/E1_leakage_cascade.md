Random Forest High/Low classification, mean ± std over 5 seeds.

| Setting                                                     | RF accuracy   | RF F1         | Crop-only baseline acc.   |
|:------------------------------------------------------------|:--------------|:--------------|:--------------------------|
| 1. Original: global-median label, +Production, random split | 0.957 ± 0.002 | 0.957 ± 0.002 | 0.823 ± 0.003             |
| 2. Remove Production                                        | 0.898 ± 0.002 | 0.897 ± 0.002 | 0.823 ± 0.003             |
| 3. + Time-based split (train<=2014, test>=2015)             | 0.858 ± 0.001 | 0.875 ± 0.001 | 0.809 ± 0.000             |
| 4. + Per-crop median label                                  | 0.807 ± 0.002 | 0.854 ± 0.002 | 0.544 ± 0.000             |
