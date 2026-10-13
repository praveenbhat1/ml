"""Leakage-aware benchmark for Indian crop-yield prediction.

Runs every experiment for the paper and writes tables (CSV + Markdown) and
figures (PDF + PNG) into ./results. Usage:

    pip install -r requirements.txt
    python run_all.py
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
TABLES = HERE / "results" / "tables"
FIGS = HERE / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
TIME_CUTOFF = 2014  # train <= 2014, test >= 2015
CATS = ["Crop", "Season", "State"]
NUMS = ["Crop_Year", "log_Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
FEATURES = CATS + NUMS

BLUE, BLUE_DARK, GRAY, INK, INK2 = "#2a78d6", "#184f95", "#b4b2a9", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- data
def load():
    csv = HERE / "crop_yield.csv"
    d = pd.read_csv(csv if csv.exists() else HERE.parent / "crop_yield.csv")
    for c in CATS:
        d[c] = d[c].str.strip()
    n0 = len(d)
    d = d[(d.Yield > 0) & (d.Area > 0)].reset_index(drop=True)
    d["log_Area"] = np.log1p(d.Area)
    d["log_Yield"] = np.log(d.Yield)
    return d, n0 - len(d)


def encode(train, test, cols):
    """Integer-code categoricals using train categories only (unseen -> -1)."""
    Xtr, Xte = train[cols].copy(), test[cols].copy()
    for c in [c for c in cols if c in CATS]:
        cats = pd.Categorical(train[c]).categories
        Xtr[c] = pd.Categorical(train[c], categories=cats).codes
        Xte[c] = pd.Categorical(test[c], categories=cats).codes
    return Xtr, Xte


def time_split(d):
    return d[d.Crop_Year <= TIME_CUTOFF], d[d.Crop_Year > TIME_CUTOFF]


def regressors(seed):
    return {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }


def group_mean_predict(train, test, keys, target="log_Yield"):
    """Predict the training mean of `target` for each group; back off to crop, then global mean."""
    pred = test[keys].merge(train.groupby(keys)[target].mean().rename("p").reset_index(), on=keys, how="left")["p"]
    crop = test[["Crop"]].merge(train.groupby("Crop")[target].mean().rename("c").reset_index(), on="Crop", how="left")["c"]
    return pred.fillna(crop).fillna(train[target].mean()).to_numpy()


def last_year_predict(train, test):
    """Persistence baseline: most recent training-period yield for the same state-crop-season."""
    last = (train.sort_values("Crop_Year").groupby(["State", "Crop", "Season"])["log_Yield"].last()
            .rename("p").reset_index())
    p = test[["State", "Crop", "Season"]].merge(last, how="left", on=["State", "Crop", "Season"])["p"].to_numpy()
    fallback = group_mean_predict(train, test, ["State", "Crop"])
    return np.where(np.isnan(p), fallback, p)


def add_history(train, test):
    """Most recent yield of the same State-Crop-Season strictly before the row's year,
    never using test-period years (test rows see at most the last training year)."""
    keys = ["State", "Crop", "Season"]
    tr = train.sort_values("Crop_Year").copy()
    tr["hist"] = tr.groupby(keys).log_Yield.shift(1)
    last = tr.groupby(keys).log_Yield.last().rename("hist").reset_index()
    te = test.drop(columns="hist", errors="ignore").merge(last, on=keys, how="left")
    te.index = test.index
    return tr.loc[train.index], te


def reg_metrics(test, pred, train):
    y = test.log_Yield.to_numpy()
    crop_mu = group_mean_predict(train, test, ["Crop"])  # crop mean from train only
    return {
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        # how much of the within-crop variation is explained (crop identity alone scores 0)
        "Within-crop R2": r2_score(y - crop_mu, pred - crop_mu),
    }


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def save_table(df, name, note=""):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (TABLES / f"{name}.md").write_text((note + "\n\n" if note else "") + md + "\n")
    print(f"\n== {name} ==\n{md}")


# ---------------------------------------------------------------- E0: data audit
def e0_audit(d, dropped):
    rate_f = d.Fertilizer / d.Area
    rate_p = d.Pesticide / d.Area
    cv = lambda s: (s.std() / s.mean()) if s.mean() else 0.0  # noqa: E731
    by_year = pd.DataFrame({
        "Year": sorted(d.Crop_Year.unique()),
        "Fertilizer per ha": d.groupby("Crop_Year").apply(lambda x: (x.Fertilizer / x.Area).mean()).values,
        "CV across rows (fert)": d.assign(r=rate_f).groupby("Crop_Year").r.apply(cv).values,
        "Pesticide per ha": d.groupby("Crop_Year").apply(lambda x: (x.Pesticide / x.Area).mean()).values,
        "CV across rows (pest)": d.assign(r=rate_p).groupby("Crop_Year").r.apply(cv).values,
    })
    save_table(by_year.round(6), "E0_fertilizer_pesticide_rate_by_year",
               "Fertilizer/Area and Pesticide/Area are constant within each year (CV ~ 0): "
               "both columns are Area x one national yearly rate.")
    audit = {
        "rows_raw": int(len(d) + dropped), "rows_dropped_nonpositive": int(dropped), "rows_used": int(len(d)),
        "years": [int(d.Crop_Year.min()), int(d.Crop_Year.max())],
        "crops": int(d.Crop.nunique()), "states": int(d.State.nunique()), "seasons": int(d.Season.nunique()),
        "max_rainfall_values_per_state_year": int(d.groupby(["State", "Crop_Year"]).Annual_Rainfall.nunique().max()),
        "max_fert_rate_cv_any_year": float(by_year["CV across rows (fert)"].max()),
        "max_pest_rate_cv_any_year": float(by_year["CV across rows (pest)"].max()),
        "corr_log_fert_vs_log_area": float(np.corrcoef(np.log1p(d.Fertilizer), d.log_Area)[0, 1]),
        "yield_vs_production_over_area_median_rel_err": float(((d.Production / d.Area - d.Yield).abs() / d.Yield).median()),
        "share_log_yield_variance_explained_by_crop": float(
            1 - ((d.log_Yield - d.groupby("Crop").log_Yield.transform("mean")) ** 2).sum()
            / ((d.log_Yield - d.log_Yield.mean()) ** 2).sum()),
        "train_rows_time_split": int((d.Crop_Year <= TIME_CUTOFF).sum()),
        "test_rows_time_split": int((d.Crop_Year > TIME_CUTOFF).sum()),
    }
    (TABLES / "E0_audit.json").write_text(json.dumps(audit, indent=2))
    print("\n== E0 audit ==\n" + json.dumps(audit, indent=2))
    return audit


# ------------------------------------------------- E1: leakage cascade (classification)
def e1_cascade(d):
    """Reproduce the original ~95% High/Low accuracy, then remove each shortcut in turn."""
    rows = []
    steps = [
        ("1. Original: global-median label, +Production, random split", "global", True, "random"),
        ("2. Remove Production", "global", False, "random"),
        ("3. + Time-based split (train<=2014, test>=2015)", "global", False, "time"),
        ("4. + Per-crop median label", "crop", False, "time"),
    ]
    for name, label, use_prod, split in steps:
        accs, f1s, base = [], [], []
        for seed in SEEDS:
            if split == "random":
                tr, te = train_test_split(d, test_size=0.2, random_state=seed)
            else:
                tr, te = time_split(d)
            if label == "global":
                thr = tr.Yield.median()
                ytr, yte = (tr.Yield >= thr).astype(int), (te.Yield >= thr).astype(int)
            else:  # High = above this crop's training-period median
                med = tr.groupby("Crop").Yield.median()
                ytr = (tr.Yield >= tr.Crop.map(med)).astype(int)
                yte = (te.Yield >= te.Crop.map(med).fillna(tr.Yield.median())).astype(int)
            cols = FEATURES + (["Production"] if use_prod else [])
            Xtr, Xte = encode(tr, te, cols)
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            p = clf.predict(Xte)
            accs.append(accuracy_score(yte, p)); f1s.append(f1_score(yte, p))
            # crop-only baseline: majority class of that crop in train
            maj = ytr.groupby(tr.Crop.values).mean().round()
            base.append(accuracy_score(yte, te.Crop.map(maj).fillna(round(ytr.mean())).astype(int)))
        rows.append({"Setting": name, "RF accuracy": fmt(np.mean(accs), np.std(accs)),
                     "RF F1": fmt(np.mean(f1s), np.std(f1s)),
                     "Crop-only baseline acc.": fmt(np.mean(base), np.std(base)),
                     "_acc": np.mean(accs), "_sd": np.std(accs), "_base": np.mean(base)})
    df = pd.DataFrame(rows)
    save_table(df.drop(columns=["_acc", "_sd", "_base"]), "E1_leakage_cascade",
               "Random Forest High/Low classification, mean ± std over 5 seeds.")

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(df))
    ax.bar(x - 0.19, df._acc * 100, 0.36, yerr=df._sd * 100, color=BLUE, label="Random Forest",
           error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK2})
    ax.bar(x + 0.19, df._base * 100, 0.36, color=GRAY, label="Crop-only baseline")
    for xi, v in zip(x, df._acc * 100):
        ax.text(xi - 0.19, v + 1.2, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, ["Original", "−Production", "+Time\nsplit", "+Per-crop\nlabel"])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(40, 102)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(FIGS / "fig1_leakage_cascade.pdf"); fig.savefig(FIGS / "fig1_leakage_cascade.png")
    plt.close(fig)
    return df


# --------------------------------------------- E2: regression benchmark (time split)
def e2_benchmark(d):
    tr, te = time_split(d)
    Xtr, Xte = encode(tr, te, FEATURES)
    results, preds = {}, {}
    baselines = {
        "Global mean": np.full(len(te), tr.log_Yield.mean()),
        "Crop mean": group_mean_predict(tr, te, ["Crop"]),
        "State + Crop mean": group_mean_predict(tr, te, ["State", "Crop"]),
        "State + Crop + Season mean": group_mean_predict(tr, te, ["State", "Crop", "Season"]),
        "Last observed yield (persistence)": last_year_predict(tr, te),
    }
    for name, p in baselines.items():
        results[name] = [reg_metrics(te, p, tr)]
        preds[name] = p
    for seed in SEEDS:
        for name, m in regressors(seed).items():
            Xa, Xb = (pd.get_dummies(pd.concat([Xtr, Xte]).astype({c: "category" for c in CATS}))
                      .pipe(lambda z: (z.iloc[:len(Xtr)], z.iloc[len(Xtr):]))) if name == "Ridge" else (Xtr, Xte)
            p = m.fit(Xa, tr.log_Yield).predict(Xb)
            results.setdefault(name, []).append(reg_metrics(te, p, tr))
            preds.setdefault(name, []).append(p)
    trh, teh = add_history(tr, te)
    for seed in SEEDS:
        A, B = encode(trh, teh, FEATURES + ["hist"])
        p = regressors(seed)["XGBoost"].fit(A, trh.log_Yield).predict(B)
        results.setdefault("XGBoost + yield history", []).append(reg_metrics(te, p, tr))
        preds.setdefault("XGBoost + yield history", []).append(p)
    # every model as a list of per-seed predictions (baselines: one deterministic "seed")
    seed_preds = {k: (v if isinstance(v, list) else [v]) for k, v in preds.items()}
    preds = {k: np.mean(v, axis=0) for k, v in seed_preds.items()}  # 5-seed ensemble, used only for Fig. 3

    rows = []
    for name, ms in results.items():
        df = pd.DataFrame(ms)
        rows.append({"Model": name, **{k: fmt(df[k].mean(), df[k].std(ddof=0)) for k in df.columns},
                     "_r2": df.R2.mean()})
    tab = pd.DataFrame(rows)
    save_table(tab.drop(columns="_r2"), "E2_regression_time_split",
               f"Target log(yield); train {int(tr.Crop_Year.min())}-{TIME_CUTOFF} (n={len(tr)}), "
               f"test {TIME_CUTOFF + 1}-{int(te.Crop_Year.max())} (n={len(te)}); ML models mean ± std over 5 seeds; "
               "baselines are deterministic. Within-crop R2 = R2 after subtracting the training crop mean.")

    y = te.log_Yield.to_numpy()
    crop_mu = group_mean_predict(tr, te, ["Crop"])
    PERS, HIST = "Last observed yield (persistence)", "XGBoost + yield history"

    # Cluster bootstrap: resample whole State-Crop series, because records of the same series are
    # correlated across years and seasons. For ML models the statistic is the mean over seeds of the
    # per-seed R2, so the point estimate equals the Table II value.
    cluster = te.State.str.cat(te.Crop, sep="|").to_numpy()
    groups = {g: np.flatnonzero(cluster == g) for g in np.unique(cluster)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([groups[g] for g in rng.choice(keys, len(keys))]) for _ in range(N_BOOT)]

    def stat(ps, i):
        return (np.mean([r2_score(y[i], p[i]) for p in ps]),
                np.mean([r2_score(y[i] - crop_mu[i], p[i] - crop_mu[i]) for p in ps]))

    boot = {k: np.array([stat(ps, i) for i in boot_idx]) for k, ps in seed_preds.items()}
    point = {k: stat(ps, np.arange(len(y))) for k, ps in seed_preds.items()}
    ci = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"  # noqa: E731
    save_table(pd.DataFrame([{"Model": k, "R2": round(point[k][0], 3), "R2 95% CI": ci(b[:, 0]),
                              "Within-crop R2": round(point[k][1], 3), "Within-crop R2 95% CI": ci(b[:, 1])}
                             for k, b in boot.items()]),
               "E2d_bootstrap_ci", f"Cluster bootstrap: {N_BOOT} resamples of the {len(keys)} State-Crop series "
               f"in the test set ({len(y)} records). ML: mean over 5 seeds of per-seed R2 (matches Table E2).")
    diff_rows = []
    for other in [HIST, "XGBoost"]:
        diff = boot[PERS][:, 0] - boot[other][:, 0]
        diff_rows.append({"Comparison": f"Persistence R2 minus {other} R2",
                          "Difference": round(point[PERS][0] - point[other][0], 4), "95% CI": ci(diff),
                          "Share of resamples where persistence is higher": round((diff > 0).mean(), 3)})
    save_table(pd.DataFrame(diff_rows), "E2e_bootstrap_difference",
               "Paired cluster bootstrap (same resampled State-Crop series for both models).")

    # Wilcoxon signed-rank. Observations are paired by test record; each record's error is its absolute
    # error averaged over the 5 seeds. Because records within a State-Crop series are not independent,
    # we also run the test on series-level mean errors (one pair per series), which is the version we report.
    err = {k: np.mean([np.abs(y - p) for p in ps], axis=0) for k, ps in seed_preds.items()}
    ml = tab[tab.Model.isin(regressors(0))].sort_values("_r2", ascending=False).Model.tolist()
    best = ml[0]  # best model on the dataset's own features (used for ablation / SHAP)
    pairs = ([(HIST, PERS), (HIST, best)] + [(best, m) for m in ml[1:]]
             + [(best, "State + Crop mean"), (best, PERS)])
    wt = []
    for a_, b_ in pairs:
        ga = pd.Series(err[a_]).groupby(cluster).mean()
        gb = pd.Series(err[b_]).groupby(cluster).mean()
        wt.append({"Model A": a_, "Model B": b_,
                   "Median |err| A": round(np.median(err[a_]), 4), "Median |err| B": round(np.median(err[b_]), 4),
                   "p (record-level)": f"{wilcoxon(err[a_], err[b_]).pvalue:.1e}",
                   "Series where A better": f"{(ga < gb).sum()}/{len(ga)}",
                   "p (series-level)": f"{wilcoxon(ga, gb).pvalue:.1e}"})
    save_table(pd.DataFrame(wt), "E2b_wilcoxon",
               "Wilcoxon signed-rank on absolute errors (averaged over 5 seeds). Record-level pairs every test record; "
               f"series-level pairs the mean error of each of the {len(keys)} State-Crop series (reported in the paper).")

    # year-wise robustness: R2 per test year (mean over seeds for ML models)
    yrs = []
    for yr in sorted(te.Crop_Year.unique()):
        m = (te.Crop_Year == yr).to_numpy()
        row = {"Year": int(yr), "n": int(m.sum())}
        for k in ["State + Crop mean", PERS, "XGBoost", HIST]:
            row[k] = round(np.mean([r2_score(y[m], p[m]) for p in seed_preds[k]]), 3)
        yrs.append(row)
    save_table(pd.DataFrame(yrs), "E2f_yearwise", "R2 on log(yield) for each test year (ML: mean over 5 seeds).")

    # random split for contrast (same models, 5 seeds)
    rr = []
    for seed in SEEDS:
        a, b = train_test_split(d, test_size=0.2, random_state=seed)
        A, B = encode(a, b, FEATURES)
        m = regressors(seed)[best]
        rr.append(reg_metrics(b, m.fit(A, a.log_Yield).predict(B), a))
    rr = pd.DataFrame(rr)
    tr_ = pd.DataFrame(results[best])
    save_table(pd.DataFrame([
        {"Split": "Random 80/20", **{k: fmt(rr[k].mean(), rr[k].std(ddof=0)) for k in rr}},
        {"Split": "Time-based (<=2014 / >=2015)", **{k: fmt(tr_[k].mean(), tr_[k].std(ddof=0)) for k in tr_}},
    ]), "E2c_random_vs_time_split", f"{best}, target log(yield), mean ± std over 5 seeds.")

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    top = "XGBoost + yield history"
    ax.scatter(y, preds[top], s=3, color=BLUE, alpha=0.35, linewidths=0)
    lo, hi = y.min() - 0.3, y.max() + 0.3
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual log(yield)"); ax.set_ylabel("Predicted log(yield)")
    ax.set_title(top, loc="left", color=INK)
    ax.text(0.04, 0.94, f"n = {len(y)} test records (2015–2020)", transform=ax.transAxes, va="top",
            fontsize=7, color=INK)
    fig.savefig(FIGS / "fig3_pred_vs_actual.pdf"); fig.savefig(FIGS / "fig3_pred_vs_actual.png")
    plt.close(fig)
    return best


# ------------------------------------------------------ E3: feature ablation + SHAP
def e3_ablation_shap(d, best):
    tr, te = time_split(d)
    variants = {
        "All features": FEATURES,
        "− Rainfall": [f for f in FEATURES if f != "Annual_Rainfall"],
        "− Fertilizer": [f for f in FEATURES if f != "Fertilizer"],
        "− Pesticide": [f for f in FEATURES if f != "Pesticide"],
        "− Rainfall, Fertilizer, Pesticide": ["Crop", "Season", "State", "Crop_Year", "log_Area"],
        "− Area (and Fertilizer, Pesticide)": ["Crop", "Season", "State", "Crop_Year", "Annual_Rainfall"],
        "Crop + State + Season only": ["Crop", "Season", "State"],
    }
    rows = []
    for name, cols in variants.items():
        ms = []
        for seed in SEEDS:
            Xtr, Xte = encode(tr, te, cols)
            ms.append(reg_metrics(te, regressors(seed)[best].fit(Xtr, tr.log_Yield).predict(Xte), tr))
        ms = pd.DataFrame(ms)
        rows.append({"Feature set": name, "R2": fmt(ms.R2.mean(), ms.R2.std(ddof=0)),
                     "Within-crop R2": fmt(ms["Within-crop R2"].mean(), ms["Within-crop R2"].std(ddof=0)),
                     "RMSE": fmt(ms.RMSE.mean(), ms.RMSE.std(ddof=0))})
    save_table(pd.DataFrame(rows), "E3_ablation", f"{best}, time split, target log(yield), mean ± std over 5 seeds.")

    Xtr, Xte = encode(tr, te, FEATURES)
    model = regressors(0)[best].fit(Xtr, tr.log_Yield)
    sv = shap.TreeExplainer(model).shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values()
    save_table(imp[::-1].rename("mean |SHAP|").reset_index().rename(columns={"index": "Feature"}).round(4),
               "E3b_shap_importance", f"Mean absolute SHAP value on the time-split test set ({best}).")
    labels = {"log_Area": "Area (log)", "Annual_Rainfall": "Rainfall", "Crop_Year": "Year"}
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.barh([labels.get(i, i) for i in imp.index], imp.values, color=BLUE, height=0.6)
    for yi, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlabel("Mean |SHAP value| on log(yield)")
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.5); ax.set_axisbelow(True)
    fig.savefig(FIGS / "fig2_shap_importance.pdf"); fig.savefig(FIGS / "fig2_shap_importance.png")
    plt.close(fig)


# ------------------------------------------------- E4: generalization to unseen states
def e4_unseen_states(d):
    rows = {}
    for k, (itr, ite) in enumerate(GroupKFold(n_splits=5).split(d, groups=d.State)):
        tr, te = d.iloc[itr], d.iloc[ite]
        Xtr, Xte = encode(tr, te, FEATURES)
        rows.setdefault("Crop mean", []).append(reg_metrics(te, group_mean_predict(tr, te, ["Crop"]), tr))
        rows.setdefault("Crop + Season mean", []).append(
            reg_metrics(te, group_mean_predict(tr, te, ["Crop", "Season"]), tr))
        for name, m in regressors(k).items():
            if name == "Ridge":
                continue
            rows.setdefault(name, []).append(reg_metrics(te, m.fit(Xtr, tr.log_Yield).predict(Xte), tr))
    out = []
    for name, ms in rows.items():
        ms = pd.DataFrame(ms)
        out.append({"Model": name, **{k: fmt(ms[k].mean(), ms[k].std(ddof=0)) for k in ["R2", "Within-crop R2", "RMSE"]}})
    save_table(pd.DataFrame(out), "E4_unseen_states",
               "5-fold GroupKFold by State (test states never seen in training); mean ± std over folds.")


if __name__ == "__main__":
    data, dropped = load()
    e0_audit(data, dropped)
    e1_cascade(data)
    best_model = e2_benchmark(data)
    e3_ablation_shap(data, best_model)
    e4_unseen_states(data)
    print("\nDone. Tables in results/tables, figures in results/figures.")
