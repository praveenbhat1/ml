"""District-level replication of the leakage-aware benchmark.

Data: Crop-wise Area, Production and Yield (APY), Directorate of Economics and Statistics, Ministry of
Agriculture and Farmers Welfare (via India Data Portal). State-year annual rainfall is joined from the
state-level Kaggle file so that the feature set matches the main study.

Same protocol as run_all.py: train 1997-2014, test 2015-2020, log-yield target, five naive baselines,
four ML models + XGBoost with yield history, seeds 0-4, cluster bootstrap and series-level Wilcoxon tests.
Outputs go to ../results/district/{tables,figures}. Usage:

    python run_district.py
"""
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
APY = HERE / "apy_district.csv"  # download: see README
KAGGLE = HERE / "crop_yield.csv"
OUT = HERE / "results" / "district"
TABLES, FIGS = OUT / "tables", OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
TIME_CUTOFF, LAST_YEAR = 2014, 2020  # same split and test period as the state-level study
MIN_CROP_RECORDS = 100
CATS = ["Crop", "Season", "State", "District"]
NUMS = ["Crop_Year", "log_Area", "Annual_Rainfall"]
FEATURES = CATS + NUMS
SERIES = ["District", "Crop", "Season"]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def load():
    d = pd.read_csv(APY, low_memory=False)
    audit = {"rows_raw": len(d)}
    d = d.rename(columns={"state_name": "State", "crop_name": "Crop", "season": "Season", "area": "Area",
                          "production": "Production", "yield": "Yield"})
    d["Crop_Year"] = d.year.str[:4].astype(int)  # "1997-1998" -> 1997, as in the state-level file
    d["District"] = d.district_code.astype(str)
    d = d[d.Season != "Total"]  # "Total" rows are sums of the other seasons
    audit["rows_after_drop_total_season"] = len(d)
    d = d.dropna(subset=["Area", "Production", "Yield"])
    d = d[(d.Area > 0) & (d.Production > 0) & (d.Yield > 0)]
    d = d[d.Crop_Year <= LAST_YEAR]
    keep = d.Crop.value_counts()
    d = d[d.Crop.isin(keep[keep >= MIN_CROP_RECORDS].index)].reset_index(drop=True)

    rain = pd.read_csv(KAGGLE)[["State", "Crop_Year", "Annual_Rainfall"]]
    rain["State"] = rain.State.str.strip().replace({"Jammu and Kashmir": "Jammu And Kashmir"})
    rain = rain.drop_duplicates(["State", "Crop_Year"])
    d = d.merge(rain, on=["State", "Crop_Year"], how="left")

    d["log_Area"] = np.log1p(d.Area)
    d["log_Yield"] = np.log(d.Yield)
    crop_mu = d.groupby("Crop").log_Yield.transform("mean")
    audit.update({
        "rows_used": len(d), "years": [int(d.Crop_Year.min()), int(d.Crop_Year.max())],
        "states": int(d.State.nunique()), "districts": int(d.District.nunique()), "crops": int(d.Crop.nunique()),
        "seasons": int(d.Season.nunique()),
        "share_rows_with_rainfall": round(float(d.Annual_Rainfall.notna().mean()), 3),
        "yield_vs_production_over_area_median_rel_err": float(((d.Production / d.Area - d.Yield).abs() / d.Yield).median()),
        "share_log_yield_variance_explained_by_crop": round(float(
            1 - ((d.log_Yield - crop_mu) ** 2).sum() / ((d.log_Yield - d.log_Yield.mean()) ** 2).sum()), 3),
        "train_rows": int((d.Crop_Year <= TIME_CUTOFF).sum()), "test_rows": int((d.Crop_Year > TIME_CUTOFF).sum()),
    })
    (TABLES / "D0_audit.json").write_text(json.dumps(audit, indent=2))
    log("audit " + json.dumps(audit))
    return d


def encode(train, test, cols):
    Xtr, Xte = train[cols].copy(), test[cols].copy()
    for c in [c for c in cols if c in CATS]:
        cats = pd.Categorical(train[c]).categories
        Xtr[c] = pd.Categorical(train[c], categories=cats).codes
        Xte[c] = pd.Categorical(test[c], categories=cats).codes
    return Xtr, Xte


def time_split(d):
    return d[d.Crop_Year <= TIME_CUTOFF], d[d.Crop_Year > TIME_CUTOFF]


def ridge_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
        ("num", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()), NUMS),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def regressors(seed):
    # identical hyperparameters to the state-level study
    return {
        "Ridge": ridge_model(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }


def group_mean_predict(train, test, keys, target="log_Yield"):
    pred = test[keys].merge(train.groupby(keys)[target].mean().rename("p").reset_index(), on=keys, how="left")["p"]
    crop = test[["Crop"]].merge(train.groupby("Crop")[target].mean().rename("c").reset_index(), on="Crop", how="left")["c"]
    return pred.fillna(crop).fillna(train[target].mean()).to_numpy()


def last_year_predict(train, test):
    last = train.sort_values("Crop_Year").groupby(SERIES)["log_Yield"].last().rename("p").reset_index()
    p = test[SERIES].merge(last, how="left", on=SERIES)["p"].to_numpy()
    return np.where(np.isnan(p), group_mean_predict(train, test, ["District", "Crop"]), p)


def add_history(train, test):
    tr = train.sort_values("Crop_Year").copy()
    tr["hist"] = tr.groupby(SERIES).log_Yield.shift(1)
    last = tr.groupby(SERIES).log_Yield.last().rename("hist").reset_index()
    te = test.merge(last, on=SERIES, how="left")
    te.index = test.index
    return tr.loc[train.index], te


def reg_metrics(test, pred, train):
    y = test.log_Yield.to_numpy()
    crop_mu = group_mean_predict(train, test, ["Crop"])
    return {"R2": r2_score(y, pred), "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "MAE": mean_absolute_error(y, pred), "Within-crop R2": r2_score(y - crop_mu, pred - crop_mu)}


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def save_table(df, name, note=""):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (TABLES / f"{name}.md").write_text((note + "\n\n" if note else "") + md + "\n")
    print(f"\n== {name} ==\n{md}", flush=True)


# ------------------------------------------------- D1: leakage cascade (classification)
def d1_cascade(d):
    rows = []
    steps = [("1. Original: global-median label, +Production, random split", "global", True, "random"),
             ("2. Remove Production", "global", False, "random"),
             ("3. + Time-based split", "global", False, "time"),
             ("4. + Per-crop median label", "crop", False, "time")]
    for name, label, use_prod, split in steps:
        accs, f1s, base = [], [], []
        for seed in SEEDS:
            tr, te = train_test_split(d, test_size=0.2, random_state=seed) if split == "random" else time_split(d)
            if label == "global":
                thr = tr.Yield.median()
                ytr, yte = (tr.Yield >= thr).astype(int), (te.Yield >= thr).astype(int)
            else:
                med = tr.groupby("Crop").Yield.median()
                ytr = (tr.Yield >= tr.Crop.map(med)).astype(int)
                yte = (te.Yield >= te.Crop.map(med).fillna(tr.Yield.median())).astype(int)
            Xtr, Xte = encode(tr, te, FEATURES + (["Production"] if use_prod else []))
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            p = clf.predict(Xte)
            accs.append(accuracy_score(yte, p)); f1s.append(f1_score(yte, p))
            maj = ytr.groupby(tr.Crop.values).mean().round()
            base.append(accuracy_score(yte, te.Crop.map(maj).fillna(round(ytr.mean())).astype(int)))
        rows.append({"Setting": name, "RF accuracy": fmt(np.mean(accs), np.std(accs)),
                     "RF F1": fmt(np.mean(f1s), np.std(f1s)), "Crop-only baseline acc.": fmt(np.mean(base), np.std(base))})
        log(f"D1 {name} done")
    save_table(pd.DataFrame(rows), "D1_leakage_cascade", "District level. Random forest, mean ± std over 5 seeds.")


# --------------------------------------------- D2: regression benchmark (time split)
def d2_benchmark(d):
    tr, te = time_split(d)
    Xtr, Xte = encode(tr, te, FEATURES)
    seed_preds = {
        "Global mean": [np.full(len(te), tr.log_Yield.mean())],
        "Crop mean": [group_mean_predict(tr, te, ["Crop"])],
        "District + Crop mean": [group_mean_predict(tr, te, ["District", "Crop"])],
        "District + Crop + Season mean": [group_mean_predict(tr, te, SERIES)],
        "Last observed yield (persistence)": [last_year_predict(tr, te)],
    }
    trh, teh = add_history(tr, te)
    Ah, Bh = encode(trh, teh, FEATURES + ["hist"])
    for seed in SEEDS:
        for name, m in regressors(seed).items():
            if name == "Ridge":
                if seed:  # deterministic: fit once, reuse
                    seed_preds[name].append(seed_preds[name][0])
                    continue
                p = m.fit(tr[FEATURES], tr.log_Yield).predict(te[FEATURES])
            else:
                p = m.fit(Xtr, tr.log_Yield).predict(Xte)
            seed_preds.setdefault(name, []).append(p)
        p = regressors(seed)["XGBoost"].fit(Ah, trh.log_Yield).predict(Bh)
        seed_preds.setdefault("XGBoost + yield history", []).append(p)
        log(f"D2 seed {seed} done")

    y = te.log_Yield.to_numpy()
    crop_mu = group_mean_predict(tr, te, ["Crop"])
    rows = []
    for name, ps in seed_preds.items():
        ms = pd.DataFrame([reg_metrics(te, p, tr) for p in ps])
        rows.append({"Model": name, **{k: fmt(ms[k].mean(), ms[k].std(ddof=0)) for k in ms.columns},
                     "_r2": ms.R2.mean()})
    tab = pd.DataFrame(rows)
    save_table(tab.drop(columns="_r2"), "D2_regression_time_split",
               f"District level. Target log(yield); train 1997-{TIME_CUTOFF} (n={len(tr)}), test "
               f"{TIME_CUTOFF + 1}-{LAST_YEAR} (n={len(te)}); ML: mean ± std over 5 seeds.")

    # cluster bootstrap over District-Crop series
    cluster = te.District.str.cat(te.Crop, sep="|").to_numpy()
    codes, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(0)
    PERS, HIST = "Last observed yield (persistence)", "XGBoost + yield history"

    def stat(ps, i):
        return (np.mean([r2_score(y[i], p[i]) for p in ps]),
                np.mean([r2_score(y[i] - crop_mu[i], p[i] - crop_mu[i]) for p in ps]))

    boot = {k: [] for k in seed_preds}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(codes), len(codes))
        i = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in pick])
        for k, ps in seed_preds.items():
            boot[k].append(stat(ps, i))
    boot = {k: np.array(v) for k, v in boot.items()}
    point = {k: stat(ps, np.arange(len(y))) for k, ps in seed_preds.items()}
    ci = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"  # noqa: E731
    save_table(pd.DataFrame([{"Model": k, "R2": round(point[k][0], 3), "R2 95% CI": ci(b[:, 0]),
                              "Within-crop R2": round(point[k][1], 3), "Within-crop R2 95% CI": ci(b[:, 1])}
                             for k, b in boot.items()]),
               "D2d_bootstrap_ci", f"Cluster bootstrap, {N_BOOT} resamples of {len(codes)} District-Crop series.")
    log("D2 bootstrap done")

    ml = tab[tab.Model.isin(regressors(0))].sort_values("_r2", ascending=False).Model.tolist()
    best = ml[0]
    diff_rows = []
    for other in [HIST, best]:
        diff = boot[PERS][:, 0] - boot[other][:, 0]
        diff_rows.append({"Comparison": f"Persistence R2 minus {other} R2",
                          "Difference": round(point[PERS][0] - point[other][0], 4), "95% CI": ci(diff),
                          "Share of resamples where persistence is higher": round((diff > 0).mean(), 3)})
    save_table(pd.DataFrame(diff_rows), "D2e_bootstrap_difference", "Paired cluster bootstrap.")

    err = {k: np.mean([np.abs(y - p) for p in ps], axis=0) for k, ps in seed_preds.items()}
    pairs = [(HIST, PERS), (HIST, best)] + [(best, m) for m in ml[1:]] + [(best, "District + Crop mean"), (best, PERS)]
    wt = []
    for a_, b_ in pairs:
        ga, gb = pd.Series(err[a_]).groupby(cluster).mean(), pd.Series(err[b_]).groupby(cluster).mean()
        wt.append({"Model A": a_, "Model B": b_, "Median |err| A": round(np.median(err[a_]), 4),
                   "Median |err| B": round(np.median(err[b_]), 4),
                   "Series where A better": f"{(ga < gb).sum()}/{len(ga)}",
                   "p (series-level)": f"{wilcoxon(ga, gb).pvalue:.1e}"})
    save_table(pd.DataFrame(wt), "D2b_wilcoxon", "Wilcoxon signed-rank on series-level mean absolute errors.")

    yrs = []
    for yr in sorted(te.Crop_Year.unique()):
        m = (te.Crop_Year == yr).to_numpy()
        row = {"Year": int(yr), "n": int(m.sum())}
        for k in ["District + Crop mean", PERS, best, HIST]:
            row[k] = round(np.mean([r2_score(y[m], p[m]) for p in seed_preds[k]]), 3)
        yrs.append(row)
    save_table(pd.DataFrame(yrs), "D2f_yearwise", "R2 per test year (ML: mean over 5 seeds).")

    rr = []
    for seed in SEEDS:
        a, b = train_test_split(d, test_size=0.2, random_state=seed)
        A, B = encode(a, b, FEATURES)
        rr.append(reg_metrics(b, regressors(seed)[best].fit(A, a.log_Yield).predict(B), a))
    rr = pd.DataFrame(rr)
    tt = pd.DataFrame([reg_metrics(te, p, tr) for p in seed_preds[best]])
    save_table(pd.DataFrame([
        {"Split": "Random 80/20", **{k: fmt(rr[k].mean(), rr[k].std(ddof=0)) for k in rr}},
        {"Split": "Time-based", **{k: fmt(tt[k].mean(), tt[k].std(ddof=0)) for k in tt}}]),
        "D2c_random_vs_time_split", f"{best}, mean ± std over 5 seeds.")
    log("D2 done")
    return best


# ------------------------------------------------------ D3: rainfall ablation + SHAP
def d3_ablation(d, best):
    tr, te = time_split(d)
    variants = {"All features": FEATURES,
                "− Rainfall": [f for f in FEATURES if f != "Annual_Rainfall"],
                "− Rainfall, − Area": ["Crop", "Season", "State", "District", "Crop_Year"],
                "Identity only (Crop, Season, State, District)": CATS}
    rows = []
    for name, cols in variants.items():
        ms = []
        for seed in SEEDS:
            Xtr, Xte = encode(tr, te, cols)
            ms.append(reg_metrics(te, regressors(seed)[best].fit(Xtr, tr.log_Yield).predict(Xte), tr))
        ms = pd.DataFrame(ms)
        rows.append({"Feature set": name, "R2": fmt(ms.R2.mean(), ms.R2.std(ddof=0)),
                     "Within-crop R2": fmt(ms["Within-crop R2"].mean(), ms["Within-crop R2"].std(ddof=0))})
    save_table(pd.DataFrame(rows), "D3_ablation", f"{best}, district level, time split, 5 seeds.")
    Xtr, Xte = encode(tr, te, FEATURES)
    model = regressors(0)[best].fit(Xtr, tr.log_Yield)
    sample = Xte.sample(min(20000, len(Xte)), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    save_table(imp.rename("mean |SHAP|").reset_index().rename(columns={"index": "Feature"}).round(4),
               "D3b_shap_importance", f"Mean |SHAP| on 20,000 test records ({best}).")
    log("D3 done")


# ------------------------------------------------- D4: generalisation to unseen states
def d4_unseen_states(d):
    rows = {}
    for k, (itr, ite) in enumerate(GroupKFold(n_splits=5).split(d, groups=d.State)):
        tr, te = d.iloc[itr], d.iloc[ite]
        Xtr, Xte = encode(tr, te, FEATURES)
        rows.setdefault("Crop mean", []).append(reg_metrics(te, group_mean_predict(tr, te, ["Crop"]), tr))
        rows.setdefault("Crop + Season mean", []).append(reg_metrics(te, group_mean_predict(tr, te, ["Crop", "Season"]), tr))
        for name in ["XGBoost", "LightGBM"]:
            m = regressors(k)[name]
            rows.setdefault(name, []).append(reg_metrics(te, m.fit(Xtr, tr.log_Yield).predict(Xte), tr))
        log(f"D4 fold {k} done")
    out = []
    for name, ms in rows.items():
        ms = pd.DataFrame(ms)
        out.append({"Model": name, **{c: fmt(ms[c].mean(), ms[c].std(ddof=0)) for c in ["R2", "Within-crop R2", "RMSE"]}})
    save_table(pd.DataFrame(out), "D4_unseen_states", "5-fold GroupKFold by State, district level.")


if __name__ == "__main__":
    data = load()
    best_model = d2_benchmark(data)
    d3_ablation(data, best_model)
    d4_unseen_states(data)
    d1_cascade(data)
    log("Done. Tables in results/district/tables.")
