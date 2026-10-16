"""Does better rainfall data help? District-level seasonal rainfall ablation.

Rainfall: IMD daily district rainfall (India Data Portal, "Daily Rainfall Data - District Level", 2009-),
aggregated to the growing season of each record:
    Kharif, Autumn : Jun-Sep of crop year t
    Rabi, Winter   : Oct of t - Feb of t+1
    Summer         : Mar-May of t+1
    Whole Year     : Jun of t - May of t+1
For each district-season-year we use the total (missing days filled with the season's mean daily value,
season dropped if < 80% of days are observed) and its ratio to that district-season's 2009-2014 mean.
This is the rainfall that actually fell, i.e. an upper bound on what a rainfall forecast could provide.

Window 2009-2020 (rainfall starts in 2009); train 2009-2014, test 2015-2020. Compared feature sets:
    none / state annual rainfall / district seasonal rainfall
for XGBoost (no yield history) and for persistence + correction (rolling history).
Outputs: results/rainfall/tables. Usage (needs apy_district.csv and imd_daily_district.csv, see README):

    python run_rainfall.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import run_district as dm

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RAIN = HERE / "imd_daily_district.csv"  # download: see README
TABLES = HERE / "results" / "rainfall" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
FIRST, CUTOFF, LAST = 2009, 2014, 2020
MIN_COVERAGE = 0.8
SEASON_WINDOW = {"Kharif": "kharif", "Autumn": "kharif", "Rabi": "rabi", "Winter": "rabi",
                 "Summer": "summer", "Whole Year": "whole"}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def save_table(df, name, note=""):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (TABLES / f"{name}.md").write_text((note + "\n\n" if note else "") + md + "\n")
    print(f"\n== {name} ==\n{md}", flush=True)


def seasonal_rainfall():
    r = pd.read_csv(RAIN, usecols=["date", "district_code", "actual"], low_memory=False)
    r["date"] = pd.to_datetime(r.date)
    y, m = r.date.dt.year, r.date.dt.month
    windows = {
        "kharif": (m.between(6, 9), y),
        "rabi": (m >= 10, y),
        "rabi_b": (m <= 2, y - 1),
        "summer": (m.between(3, 5), y - 1),
        "whole_a": (m >= 6, y),
        "whole_b": (m <= 5, y - 1),
    }
    parts = []
    for name, (mask, crop_year) in windows.items():
        p = r[mask].assign(window=name.split("_")[0], Crop_Year=crop_year[mask])
        parts.append(p)
    r = pd.concat(parts)
    g = r.groupby(["district_code", "window", "Crop_Year"]).actual
    s = pd.DataFrame({"mean_daily": g.mean(), "observed": g.count(), "days": g.size()}).reset_index()
    full_len = {"kharif": 122, "rabi": 151, "summer": 92, "whole": 365}
    s["coverage"] = s.observed / s.window.map(full_len)
    s = s[s.coverage >= MIN_COVERAGE]
    s["season_rain"] = s.mean_daily * s.window.map(full_len)
    base = s[s.Crop_Year.between(FIRST, CUTOFF)].groupby(["district_code", "window"]).season_rain.mean()
    s = s.join(base.rename("base"), on=["district_code", "window"])
    s["season_rain_ratio"] = s.season_rain / s.base
    s["District"] = s.district_code.astype(str)
    return s[["District", "window", "Crop_Year", "season_rain", "season_rain_ratio"]]


def main():
    d = dm.load()
    d = d[d.Crop_Year.between(FIRST, LAST)].copy()
    d["window"] = d.Season.map(SEASON_WINDOW)
    rain = seasonal_rainfall()
    d = d.merge(rain, on=["District", "window", "Crop_Year"], how="left")
    audit = {"rows": len(d), "train_rows": int((d.Crop_Year <= CUTOFF).sum()),
             "test_rows": int((d.Crop_Year > CUTOFF).sum()),
             "share_with_district_seasonal_rain": round(float(d.season_rain.notna().mean()), 3),
             "share_with_state_annual_rain": round(float(d.Annual_Rainfall.notna().mean()), 3),
             "districts_matched": int(d.loc[d.season_rain.notna(), "District"].nunique()),
             "districts_total": int(d.District.nunique())}
    (TABLES / "R0_audit.json").write_text(json.dumps(audit, indent=2))
    log("audit " + json.dumps(audit))

    # rolling history (one-year-ahead), fallback = training district-crop mean
    series = ["District", "Crop", "Season"]
    d = d.assign(_o=np.arange(len(d))).sort_values("Crop_Year")
    d["hist"] = d.groupby(series).log_Yield.shift(1)
    d = d.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    tr, te = d[d.Crop_Year <= CUTOFF].copy(), d[d.Crop_Year > CUTOFF].copy()
    h_tr = tr["hist"].fillna(pd.Series(dm.group_mean_predict(tr, tr, ["District", "Crop"]), index=tr.index)).to_numpy()
    h_te = te["hist"].fillna(pd.Series(dm.group_mean_predict(tr, te, ["District", "Crop"]), index=te.index)).to_numpy()
    tr["has_hist"], te["has_hist"] = tr["hist"].notna().astype(int), te["hist"].notna().astype(int)
    tr["hist"], te["hist"] = h_tr, h_te
    y = te.log_Yield.to_numpy()
    crop_mu = dm.group_mean_predict(tr, te, ["Crop"])

    base_feats = ["Crop", "Season", "State", "District", "Crop_Year", "log_Area"]
    feature_sets = {"No rainfall": [],
                    "State annual rainfall": ["Annual_Rainfall"],
                    "District seasonal rainfall": ["season_rain", "season_rain_ratio"]}
    preds = {}
    for model in ["XGBoost", "Persistence + correction"]:
        for fs, extra in feature_sets.items():
            cols = base_feats + extra + (["hist", "has_hist"] if model != "XGBoost" else [])
            A, B = dm.encode(tr, te, cols)
            for seed in SEEDS:
                m = dm.regressors(seed)["XGBoost"]
                if model == "XGBoost":
                    p = m.fit(A, tr.log_Yield).predict(B)
                else:
                    p = h_te + m.fit(A, tr.log_Yield.to_numpy() - h_tr).predict(B)
                preds.setdefault((model, fs), []).append(p)
            log(f"{model} / {fs} done")
    preds[("Persistence (rolling)", "—")] = [h_te]

    cluster = te.District.str.cat(te.Crop, sep="|").to_numpy()
    codes, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([order[bounds[g]:bounds[g + 1]] for g in rng.integers(0, len(codes), len(codes))])
                for _ in range(N_BOOT)]

    def r2(ps, i):
        return np.mean([r2_score(y[i], p[i]) for p in ps])

    boot = {k: np.array([r2(ps, i) for i in boot_idx]) for k, ps in preds.items()}
    rows = []
    for (model, fs), ps in preds.items():
        r2s = [r2_score(y, p) for p in ps]
        wc = np.mean([r2_score(y - crop_mu, p - crop_mu) for p in ps])
        b = boot[(model, fs)]
        rows.append({"Model": model, "Rainfall input": fs, "R2": f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}",
                     "R2 95% CI": f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]",
                     "Within-crop R2": f"{wc:.3f}"})
    save_table(pd.DataFrame(rows), "R1_rainfall_ablation",
               f"District level, train {FIRST}-{CUTOFF} (n={len(tr)}), test {CUTOFF + 1}-{LAST} (n={len(te)}); "
               "5 seeds; cluster bootstrap over district-crop series.")
    comp = []
    for model in ["XGBoost", "Persistence + correction"]:
        for fs in ["State annual rainfall", "District seasonal rainfall"]:
            diff = boot[(model, fs)] - boot[(model, "No rainfall")]
            comp.append({"Model": model, "Comparison": f"{fs} − no rainfall",
                         "ΔR2": f"{r2(preds[(model, fs)], np.arange(len(y))) - r2(preds[(model, 'No rainfall')], np.arange(len(y))):+.4f}",
                         "95% CI": f"[{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]"})
    save_table(pd.DataFrame(comp), "R2_rainfall_differences", "Paired cluster bootstrap (1000 resamples).")
    log("Done.")


if __name__ == "__main__":
    main()
