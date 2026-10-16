# Supplementary code: leakage-aware crop-yield benchmark

Regenerates every number, table and figure in the paper from the raw data.

```bash
pip install -r requirements.txt     # Python 3.10+; macOS also needs `brew install libomp` for XGBoost
python run_all.py                   # state-level study, ~5 min      -> results/tables, results/figures
python run_district.py              # district-level study, ~12 min  -> results/district/tables
python run_forecast.py              # rolling origin + persistence + correction, ~3 min -> results/forecast/tables
python run_rainfall.py              # IMD seasonal rainfall test, ~2 min -> results/rainfall/tables
```
The last three scripts need the two downloaded files below.

## Data
| File | Source | SHA-256 |
|---|---|---|
| `crop_yield.csv` (included) | "Crop Yield in Indian States" dataset, Kaggle | `ab9bc356b1f8107d490376cec24450a0e2906322ad25788ab22fb32537ab1f8f` |
| `apy_district.csv` (download, ~56 MB) | Crop-wise Area, Production and Yield (APY), Directorate of Economics and Statistics, Ministry of Agriculture and Farmers Welfare, via India Data Portal | `fe10fbf129b3501bbb3d282c7600402a8ed7248c116f2cc9428bc2b573d1d4a9` |
| `imd_daily_district.csv` (download, ~280 MB) | India Meteorological Department, "Daily Rainfall Data - District Level", Climate Data collection, India Data Portal (ODC-By licence) | `7769b6b9872857e2351f025fc8256869f2f89c5eec5af48ae7f175601fe4e9eb` |

Download both into this folder:
```bash
curl -L -o apy_district.csv "https://ckandev.indiadataportal.com/dataset/f2bbc28c-6c7c-462b-9064-ea4c4213d466/resource/f980409d-49a2-42ae-9eb0-182365005c04/download/crop-wise-area-production-yield.csv"
curl -L -o imd_daily_district.csv "https://ckandev.indiadataportal.com/dataset/25664936-d131-4b07-845e-4bd2c0d794aa/resource/d97707e6-9309-46a5-ac31-e8482ce8f9eb/download/daily-rainfall-data-district-level.csv"
shasum -a 256 apy_district.csv imd_daily_district.csv   # compare with the table above
```
If the portal has updated a file, its checksum will differ and the numbers may change slightly.

## Where each paper item comes from
| Paper item | Output file(s) |
|---|---|
| Sec. III audit | `results/tables/E0_*`, `results/district/tables/D0_audit.json`, `results/rainfall/tables/R0_audit.json` |
| Fig. 1, Table I | `E1_leakage_cascade.*`, `fig1_leakage_cascade.*` |
| Table II | `E2_regression_time_split.*` (mean ± s.d.), `E2d_bootstrap_ci.*` (95% CI) |
| Sec. V-B tests, year-wise sentence, random split | `E2b_wilcoxon.*`, `E2e_bootstrap_difference.*`, `E2f_yearwise.*`, `E2c_random_vs_time_split.*` |
| Table III, Fig. 2 | `E3_ablation.*`, `E3b_shap_importance.*`, `fig2_shap_importance.*` |
| Unseen states (Sec. V-D) | `E4_unseen_states.*`, `D4_unseen_states.*` |
| Table IV, Sec. V-E | `D1_*`, `D2_*`, `D2b_*`, `D2c_*`, `D2d_*`, `D2e_*`, `D2f_*`, `D3_ablation.*` (district column) and the `E*` files (state column) |
| Table V, Sec. V-F | `results/forecast/tables/F1_forecast_settings.*`, `F2_vs_persistence.*`, `F3_rolling_yearwise.*` |
| Table VI, Sec. V-G | `results/rainfall/tables/R1_rainfall_ablation.*`, `R2_rainfall_differences.*` |

## Protocol
Seeds 0–4. Train 1997–2014, test 2015–2020 (rainfall test: train 2009–2014). Target log(yield).
Cluster bootstrap with 1000 resamples over State–Crop (state data) or District–Crop (district data) series;
Wilcoxon signed-rank tests on series-level mean absolute errors. Hyperparameters are in `regressors()`.
District cleaning: drop "Total" season rows, missing/zero area, production or yield, crops with < 100
records and years after 2020; state-year rainfall is joined from `crop_yield.csv`.
Rolling origin: history = most recent earlier observation of the series (may be in the test period);
models are trained once on the training years and never refitted.
