"""Fair-baseline and persistence-anchored experiments on both datasets.

Two forecast settings, both with models trained only on 1997-2014 and evaluated on 2015-2020:
  * fixed origin  - the forecaster knows observations up to 2014 only (as in run_all.py / run_district.py);
  * rolling       - one-year-ahead: when forecasting year t, all observations before t are known
                    (models are still trained once on 1997-2014, never refitted).
In each setting we compare three methods that use the same history:
  * persistence                   - last known log-yield of the same region-crop-season series;
  * XGBoost + history             - XGBoost with the last known log-yield as an extra feature;
  * persistence + correction      - XGBoost trained to predict (log-yield - last known log-yield),
                                    added back to persistence.
Missing history (new series) falls back to the training region-crop mean in all three methods.

Outputs: results/forecast/tables. Usage (needs apy_district.csv, see README):

    python run_forecast.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

import run_all as state_mod
import run_district as district_mod

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
TABLES = HERE / "results" / "forecast" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
CUTOFF = 2014
t0 = time.time()

LEVELS = {
    "State": dict(mod=state_mod, region="State"),
    "District": dict(mod=district_mod, region="District"),
}


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def save_table(df, name, note=""):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (TABLES / f"{name}.md").write_text((note + "\n\n" if note else "") + md + "\n")
    print(f"\n== {name} ==\n{md}", flush=True)


def histories(d, series):
    """Previous observed log-yield of the same series (rolling), and the fixed-origin version for test rows."""
    d = d.assign(_order=np.arange(len(d))).sort_values("Crop_Year")
    d["hist_roll"] = d.groupby(series).log_Yield.shift(1)  # most recent earlier observation, any year
    train = d.Crop_Year <= CUTOFF
    last_train = d[train].groupby(series).log_Yield.last().rename("last_train").reset_index()
    # restore the original row order so model fits match run_all.py / run_district.py exactly
    d = d.merge(last_train, on=series, how="left").sort_values("_order").drop(columns="_order").reset_index(drop=True)
    # fixed origin: train rows see their previous (training-period) observation; test rows see the 2014 origin
    d["hist_fixed"] = np.where(d.Crop_Year <= CUTOFF, d.hist_roll, d.last_train)
    return d.drop(columns="last_train")


def run_level(name, cfg):
    mod, region = cfg["mod"], cfg["region"]
    series = [region, "Crop", "Season"]
    d = mod.load()[0] if name == "State" else mod.load()
    d = histories(d, series)
    tr, te = d[d.Crop_Year <= CUTOFF].copy(), d[d.Crop_Year > CUTOFF].copy()
    fb_tr = mod.group_mean_predict(tr, tr, [region, "Crop"])
    fb_te = mod.group_mean_predict(tr, te, [region, "Crop"])
    y = te.log_Yield.to_numpy()
    crop_mu = mod.group_mean_predict(tr, te, ["Crop"])
    log(f"{name}: train {len(tr)}, test {len(te)}, "
        f"test rows with rolling history {te.hist_roll.notna().mean():.3f}, fixed {te.hist_fixed.notna().mean():.3f}")

    preds = {}
    for setting, col in [("Fixed origin", "hist_fixed"), ("Rolling", "hist_roll")]:
        h_tr = tr[col].fillna(pd.Series(fb_tr, index=tr.index)).to_numpy()
        h_te = te[col].fillna(pd.Series(fb_te, index=te.index)).to_numpy()
        preds[(setting, "Persistence")] = [h_te]
        trh, teh = tr.assign(hist=tr[col], has_hist=tr[col].notna().astype(int)), \
            te.assign(hist=te[col], has_hist=te[col].notna().astype(int))
        A, B = mod.encode(trh, teh, mod.FEATURES + ["hist"])
        Ac, Bc = mod.encode(trh.assign(hist=h_tr), teh.assign(hist=h_te), mod.FEATURES + ["hist", "has_hist"])
        for seed in SEEDS:
            p = mod.regressors(seed)["XGBoost"].fit(A, tr.log_Yield).predict(B)
            preds.setdefault((setting, "XGBoost + history"), []).append(p)
            r = mod.regressors(seed)["XGBoost"].fit(Ac, tr.log_Yield.to_numpy() - h_tr).predict(Bc)
            preds.setdefault((setting, "Persistence + correction"), []).append(h_te + r)
        log(f"{name} {setting} done")

    def metrics(ps, i=None):
        i = np.arange(len(y)) if i is None else i
        return (np.mean([r2_score(y[i], p[i]) for p in ps]),
                np.mean([r2_score(y[i] - crop_mu[i], p[i] - crop_mu[i]) for p in ps]))

    rows = []
    for (setting, method), ps in preds.items():
        r2s = [r2_score(y, p) for p in ps]
        rows.append({"Level": name, "Setting": setting, "Method": method,
                     "R2": f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}",
                     "Within-crop R2": f"{metrics(ps)[1]:.3f}",
                     "MAE": f"{np.mean([mean_absolute_error(y, p) for p in ps]):.3f}",
                     "Median |err|": f"{np.median(np.mean([np.abs(y - p) for p in ps], axis=0)):.4f}"})

    # cluster bootstrap over region-crop series
    cluster = te[region].astype(str).str.cat(te.Crop, sep="|").to_numpy()
    codes, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(0)
    boot = {k: [] for k in preds}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(codes), len(codes))
        i = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in pick])
        for k, ps in preds.items():
            boot[k].append(metrics(ps, i)[0])
    boot = {k: np.array(v) for k, v in boot.items()}
    for row in rows:
        b = boot[(row["Setting"], row["Method"])]
        row["R2 95% CI"] = f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"

    # paired comparisons against persistence in the same setting
    comp = []
    err = {k: np.mean([np.abs(y - p) for p in ps], axis=0) for k, ps in preds.items()}
    for setting in ["Fixed origin", "Rolling"]:
        base = (setting, "Persistence")
        for method in ["XGBoost + history", "Persistence + correction"]:
            k = (setting, method)
            diff = boot[k] - boot[base]
            ga, gb = pd.Series(err[k]).groupby(cluster).mean(), pd.Series(err[base]).groupby(cluster).mean()
            comp.append({"Level": name, "Setting": setting, "Comparison": f"{method} − Persistence",
                         "ΔR2": f"{metrics(preds[k])[0] - metrics(preds[base])[0]:+.4f}",
                         "ΔR2 95% CI": f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]",
                         "Series where method better": f"{(ga < gb).sum()}/{len(ga)}",
                         "Wilcoxon p (series-level)": f"{wilcoxon(ga, gb).pvalue:.1e}"})
    # does rolling persistence beat fixed-origin persistence?
    k1, k0 = ("Rolling", "Persistence"), ("Fixed origin", "Persistence")
    diff = boot[k1] - boot[k0]
    comp.append({"Level": name, "Setting": "Rolling vs fixed", "Comparison": "Rolling − fixed-origin persistence",
                 "ΔR2": f"{metrics(preds[k1])[0] - metrics(preds[k0])[0]:+.4f}",
                 "ΔR2 95% CI": f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]",
                 "Series where method better": "", "Wilcoxon p (series-level)": ""})

    # year-wise R2 in the rolling setting
    yrs = []
    for yr in sorted(te.Crop_Year.unique()):
        m = (te.Crop_Year == yr).to_numpy()
        row = {"Level": name, "Year": int(yr), "n": int(m.sum())}
        for method in ["Persistence", "XGBoost + history", "Persistence + correction"]:
            row[method] = round(np.mean([r2_score(y[m], p[m]) for p in preds[("Rolling", method)]]), 3)
        yrs.append(row)
    return pd.DataFrame(rows), pd.DataFrame(comp), pd.DataFrame(yrs)


if __name__ == "__main__":
    outs = [run_level(n, c) for n, c in LEVELS.items()]
    save_table(pd.concat([o[0] for o in outs]), "F1_forecast_settings",
               "Test 2015-2020; models trained once on 1997-2014; R2 mean ± s.d. over 5 seeds; "
               "95% CI: cluster bootstrap over region-crop series (1000 resamples).")
    save_table(pd.concat([o[1] for o in outs]), "F2_vs_persistence",
               "Paired cluster bootstrap for ΔR2; Wilcoxon signed-rank on series-level mean absolute errors.")
    save_table(pd.concat([o[2] for o in outs]), "F3_rolling_yearwise", "Rolling setting, R2 by test year.")
    log("Done.")
