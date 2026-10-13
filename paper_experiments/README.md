# Supplementary code: leakage-aware crop-yield benchmark

Regenerates every number, table and figure in the paper from the raw data.

```bash
pip install -r requirements.txt     # Python 3.10+; macOS also needs `brew install libomp` for XGBoost
python run_all.py                   # state-level study, ~5 min  -> results/tables, results/figures
python run_district.py              # district-level study, ~12 min -> results/district/tables (needs step below)
```

## Data
| File | Source | SHA-256 |
|---|---|---|
| `crop_yield.csv` (included) | "Crop Yield in Indian States" dataset, Kaggle | `ab9bc356b1f8107d490376cec24450a0e2906322ad25788ab22fb32537ab1f8f` |
| `apy_district.csv` (download, ~56 MB) | Crop-wise Area, Production and Yield (APY), Directorate of Economics and Statistics, Ministry of Agriculture and Farmers Welfare, via India Data Portal | `fe10fbf129b3501bbb3d282c7600402a8ed7248c116f2cc9428bc2b573d1d4a9` |

Download the district file into this folder:
```bash
curl -L -o apy_district.csv "https://ckandev.indiadataportal.com/dataset/f2bbc28c-6c7c-462b-9064-ea4c4213d466/resource/f980409d-49a2-42ae-9eb0-182365005c04/download/crop-wise-area-production-yield.csv"
shasum -a 256 apy_district.csv   # should match the checksum above
```
If the portal has updated the file, the checksum will differ and district numbers may change slightly.

## Where each paper item comes from
| Paper item | Output file(s) |
|---|---|
| Sec. III audit (state) | `results/tables/E0_audit.json`, `E0_fertilizer_pesticide_rate_by_year.*` |
| Sec. III audit (district) | `results/district/tables/D0_audit.json` |
| Fig. 1, Table I | `E1_leakage_cascade.*`, `fig1_leakage_cascade.*` |
| Table II | `E2_regression_time_split.*` (mean ± s.d.), `E2d_bootstrap_ci.*` (95% CI) |
| Wilcoxon tests, paired bootstrap (Sec. V-B) | `E2b_wilcoxon.*`, `E2e_bootstrap_difference.*` |
| Table III | `E2f_yearwise.*` |
| Random vs. temporal split (Sec. V-B) | `E2c_random_vs_time_split.*` |
| Table IV, Fig. 2 | `E3_ablation.*`, `E3b_shap_importance.*`, `fig2_shap_importance.*` |
| Unseen states (Sec. V-D) | `E4_unseen_states.*` |
| Table V, Sec. V-E (district) | `D1_leakage_cascade.*`, `D2_regression_time_split.*`, `D2c_random_vs_time_split.*`, `D2d_bootstrap_ci.*`, `D2e_bootstrap_difference.*`, `D2b_wilcoxon.*`, `D2f_yearwise.*`, `D3_ablation.*`, `D4_unseen_states.*` (state column of Table V: the `E*` files above) |

## Protocol
Seeds 0–4. Train 1997–2014, test 2015–2020. Target log(yield). Cluster bootstrap with 1000 resamples
over State–Crop (state data) or District–Crop (district data) series; Wilcoxon signed-rank tests on
series-level mean absolute errors. Hyperparameters are in `regressors()` in each script.
District cleaning: drop "Total" season rows, missing/zero area, production or yield, crops with < 100
records and years after 2020; state-year rainfall is joined from `crop_yield.csv`.
