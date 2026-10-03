"""Assumption checks for the structured linear model + report figures.

    python -m src.diagnostics      # writes outputs/assumption_tests.csv,
                                   # outputs/linear_coefficients.csv, outputs/figures/*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .features import COOLING_BASE, ROOT, SENSORS, TARGET, load_test, load_train
from .models import StructuredLinearModel

OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Reference categorical palette (fixed slot order) and chart chrome.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2,
    "font.family": "sans-serif",
})


def fit_ols(train: pd.DataFrame):
    """The full linear model on complete rows, via statsmodels for inference."""
    cc = train.dropna(subset=SENSORS)
    lin = StructuredLinearModel()
    lin._levels(cc)
    X = lin.design(cc)
    ols = sm.OLS(cc[TARGET].to_numpy(), X).fit()
    robust = ols.get_robustcov_results(cov_type="HC3")
    return cc, X, ols, robust


def assumption_tests(train, cc, X, ols) -> pd.DataFrame:
    resid, fitted = ols.resid, ols.fittedvalues
    rows = []

    # 1. Linearity / correct functional form.
    reset = linear_reset(ols, power=2, use_f=True)
    rows.append(("Linearity", "Ramsey RESET (fitted^2 added)", f"F={reset.fvalue:.2f}",
                 f"{reset.pvalue:.3f}"))

    # 2. Independence of errors (rows ordered by id, which is not time-ordered).
    order = np.argsort(cc["id"].str[2:].astype(int).to_numpy())
    rows.append(("Independence", "Durbin-Watson (rows in id order)",
                 f"{durbin_watson(np.asarray(resid)[order]):.3f}", "~2 means none"))

    # 3. Constant variance.
    bp = het_breuschpagan(resid, X)
    dec = pd.qcut(fitted, 10, labels=False)
    sd = pd.Series(np.asarray(resid)).groupby(np.asarray(dec)).std()
    rows.append(("Homoscedasticity", "Breusch-Pagan LM", f"{bp[0]:.1f}", f"{bp[1]:.2g}"))
    rows.append(("Homoscedasticity", "Residual SD, top / bottom fitted decile",
                 f"{sd.iloc[-1]:.2f} / {sd.iloc[0]:.2f}", "ratio %.2f" % (sd.iloc[-1] / sd.iloc[0])))

    # 4. Normality of errors.
    jb, jbp, skew, kurt = jarque_bera(resid)
    rows.append(("Normality", "Jarque-Bera", f"{jb:.1f}", f"{jbp:.2g}"))
    rows.append(("Normality", "Skew / kurtosis", f"{skew:.2f} / {kurt:.2f}", "normal: 0 / 3"))

    # 5. Multicollinearity among the continuous drivers (main-effects design).
    me = pd.get_dummies(cc[["building_id"]].astype(str), drop_first=True).astype(float)
    me = pd.concat([me, pd.get_dummies(cc["hour"].astype(str), prefix="h", drop_first=True).astype(float),
                    cc[SENSORS + ["cooling_deg"]]], axis=1)
    me = sm.add_constant(me)
    for s in SENSORS + ["cooling_deg"]:
        rows.append(("No multicollinearity", f"VIF {s}",
                     f"{variance_inflation_factor(me.to_numpy(), me.columns.get_loc(s)):.2f}", "<5 fine, >10 severe"))

    # 6. No dominant influential points.
    cooks = ols.get_influence().cooks_distance[0]
    rows.append(("No influential outliers", "max Cook's D / # D>4/n",
                 f"{cooks.max():.3f} / {(cooks > 4 / len(cooks)).sum()}", "D>1 would be a concern"))

    # 7. Missing completely at random: can the observed data predict missingness?
    any_miss = train[SENSORS].isna().any(axis=1).astype(int).to_numpy()
    Z = pd.get_dummies(train[["building_id"]], drop_first=True).astype(float)
    Z[["hour", "month", "weekend", TARGET]] = train[["hour", "month", "weekend", TARGET]]
    Z = (Z - Z.mean()) / Z.std()
    p = cross_val_predict(LogisticRegression(max_iter=2000), Z, any_miss, cv=5, method="predict_proba")[:, 1]
    t = stats.ttest_ind(train.loc[any_miss == 1, TARGET], train.loc[any_miss == 0, TARGET], equal_var=False)
    rows.append(("Missing completely at random", "AUC predicting 'row has a gap' from building/time/target",
                 f"{roc_auc_score(any_miss, p):.3f}", "0.5 = unpredictable"))
    rows.append(("Missing completely at random", "Welch t-test, usage of gappy vs complete rows",
                 f"t={t.statistic:.2f}", f"{t.pvalue:.3f}"))

    return pd.DataFrame(rows, columns=["assumption", "test", "statistic", "p-value / reference"])


def coefficient_table(robust, X) -> pd.DataFrame:
    names = list(X.columns)
    ci = robust.conf_int()
    tab = pd.DataFrame({"term": names, "coef": robust.params, "se_HC3": robust.bse,
                        "ci_low": ci[:, 0], "ci_high": ci[:, 1], "p": robust.pvalues})
    keep = tab["term"].str.contains("temperature|occupancy|previous_usage|humidity|cooling|weekend|month|bld")
    return tab[keep].round(4)


def slope_stability(cc: pd.DataFrame) -> pd.DataFrame:
    """Does usage keep responding the same way in the tails? (covariate-shift check)

    Residualise usage, previous usage and occupancy on building/hour/calendar,
    then compare the usage response in normal rows vs extreme rows.
    """
    import statsmodels.formula.api as smf
    ctx = StructuredLinearModel(sensors=[])  # building + type x hour + type x weekend + month
    ctx._levels(cc)
    C = ctx.design(cc)
    d = cc.copy()
    for v in [TARGET, "previous_usage", "occupancy", "temperature"]:
        d[v + "_r"] = sm.OLS(cc[v].to_numpy(), C).fit().resid
    out = []
    for v, cut in [("previous_usage", 12), ("occupancy", 60)]:
        ext = d[v + "_r"].abs() > cut
        for name, m in [("normal", ~ext), ("extreme", ext)]:
            f = smf.ols(f"{TARGET}_r ~ previous_usage_r + occupancy_r + temperature_r", data=d[m]).fit()
            out.append({"driver": v, "rows": f"{name} (|dev|{'>' if name == 'extreme' else '<='}{cut})",
                        "n": int(m.sum()), "slope": f.params[v + "_r"], "se": f.bse[v + "_r"]})
    return pd.DataFrame(out).round(4)


# ------------------------------------------------------------------ figures
def fig_profiles(train):
    types = sorted(train["building_type"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    g = train.groupby(["building_type", "weekend", "hour"])[TARGET].mean()
    for ax, t in zip(axes.flat, types):
        ax.plot(range(24), g[t][0].reindex(range(24)), color=S1, label="Weekday")
        ax.plot(range(24), g[t][1].reindex(range(24)), color=S2, label="Weekend")
        ax.set_title(t, loc="left")
        ax.set_xticks([0, 6, 12, 18, 23])
    for ax in axes[1]:
        ax.set_xlabel("Hour of day")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean energy usage")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Each building type has its own daily load shape; weekends shift it",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_load_profiles.png", dpi=150)
    plt.close(fig)


def fig_drivers(cc, ols, X):
    resid = np.asarray(ols.resid)
    b = ols.params
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.scatter(cc["previous_usage"], cc[TARGET], s=6, alpha=0.25, color=S1, linewidths=0)
    lim = [0, 150]
    ax.plot(lim, lim, color=MUTED, lw=1)
    ax.text(140, 128, "y = x", color=MUTED, fontsize=8, ha="right")
    ax.set(xlabel="Previous-hour usage", ylabel="Energy usage", xlim=lim, ylim=lim,
           title="Strongest single signal: persistence (r = 0.95)")

    def partial(ax, var, cols, title, xlabel, bins):
        comp = (X[cols].to_numpy() @ b[cols].to_numpy())
        pr = resid + comp
        x = cc[var].to_numpy()
        ax.scatter(x, pr, s=5, alpha=0.15, color=S1, linewidths=0)
        cut = pd.cut(x, bins)
        m = pd.Series(pr).groupby(cut, observed=True).mean()
        mid = [iv.mid for iv in m.index]
        ax.plot(mid, m.to_numpy(), "o", color=S2, ms=5, label="Binned mean of partial residual")
        order = np.argsort(x)
        fitted = pd.Series(comp[order]).rolling(150, center=True, min_periods=20).mean()
        ax.plot(x[order], fitted, color=INK, lw=1.5, label="Model component (avg over building types)")
        ax.set(xlabel=xlabel, ylabel="Partial residual", title=title)

    tcols = [c for c in X.columns if c.endswith(":temperature")] + ["cooling_deg"]
    partial(axes[1], "temperature", tcols, f"Temperature: linear + cooling hinge at {COOLING_BASE:.0f}°C",
            "Outdoor temperature (°C)", np.arange(23, 36.5, 1))
    axes[1].axvline(COOLING_BASE, color=MUTED, lw=1)
    ocols = [c for c in X.columns if c.endswith(":occupancy")]
    partial(axes[2], "occupancy", ocols, "Occupancy: linear, slope varies by type",
            "Occupancy (people)", np.arange(0, 351, 25))
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_drivers.png", dpi=150)
    plt.close(fig)


def fig_shift(train, test):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (v, bins) in zip(axes, [("temperature", np.arange(22.5, 36.5, 0.5)),
                                     ("occupancy", np.arange(0, 350, 12.5)),
                                     ("previous_usage", np.arange(0, 155, 5))]):
        ax.hist(train[v].dropna(), bins=bins, density=True, histtype="step", color=S1, lw=2, label="Train")
        ax.hist(test[v].dropna(), bins=bins, density=True, histtype="step", color=S2, lw=2, label="Test")
        ax.set(title=v.replace("_", " ").capitalize(), ylabel="Density")
        ax.set_yscale("log")
    axes[0].legend(loc="upper left")
    ax = axes[3]
    x = np.arange(len(SENSORS))
    ax.bar(x - 0.2, train[SENSORS].isna().mean() * 100, 0.38, color=S1, label="Train")
    ax.bar(x + 0.2, test[SENSORS].isna().mean() * 100, 0.38, color=S2, label="Test")
    ax.set_xticks(x, ["temp", "humidity", "occupancy", "prev usage"])
    ax.set(title="Missing values (% of rows)", ylabel="%")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Test set is a stress test: heavier tails (log-density) and ~5x more gaps",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_train_test_shift.png", dpi=150)
    plt.close(fig)


def fig_residuals(cc, ols):
    resid, fitted = np.asarray(ols.resid), np.asarray(ols.fittedvalues)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    ax = axes[0]
    ax.scatter(fitted, resid, s=5, alpha=0.2, color=S1, linewidths=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xlabel="Fitted value", ylabel="Residual", title="Residuals vs fitted: no visible curvature")
    ax = axes[1]
    (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=5, color=S1, alpha=0.4, linewidths=0)
    ax.plot(osm, slope * osm + icpt, color=INK, lw=1.2)
    ax.set(xlabel="Theoretical normal quantile", ylabel="Residual quantile", title="Q-Q: close to normal")
    ax = axes[2]
    dec = pd.qcut(fitted, 10)
    sd = pd.Series(resid).groupby(dec, observed=True).std()
    ax.plot([iv.mid for iv in sd.index], sd.to_numpy(), "o-", color=S1, ms=5)
    ax.set(xlabel="Fitted value (decile midpoint)", ylabel="Residual SD", ylim=(0, None),
           title="Spread grows mildly with level")
    ax = axes[3]
    types = sorted(cc["building_type"].unique())
    data = [resid[cc["building_type"].to_numpy() == t] for t in types]
    bp = ax.boxplot(data, orientation="vertical", patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2))
    for patch in bp["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=S1)
    ax.set_xticks(range(1, len(types) + 1), [t[:5] for t in types], rotation=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(ylabel="Residual", title="Unbiased in every building type")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_residual_diagnostics.png", dpi=150)
    plt.close(fig)


def fig_model_comparison():
    path = OUT / "cv_results.csv"
    if not path.exists():
        return
    res = pd.read_csv(path).iloc[::-1]
    views = [("cv_RMSE", "Standard 5-fold CV", S1),
             ("cv_missing_RMSE", "CV + test-like gaps", S2),
             ("test_like_RMSE", "CV + gaps + shift weighting", S3)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(res))
    h = 0.26
    for i, (col, label, color) in enumerate(views):
        ax.barh(y + (1 - i) * h, res[col], h - 0.04, color=color, label=label)
    for yy, v in zip(y, res["test_like_RMSE"]):
        ax.text(v + 0.08, yy - h, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, res["model"])
    ax.set_xlabel("RMSE (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Linear structure + tree correction wins in every view", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_model_comparison.png", dpi=150)
    plt.close(fig)


def fig_pred_vs_actual():
    path = OUT / "oof_predictions.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    ax.scatter(d["actual"], d["oof_pred"], s=5, alpha=0.25, color=S1, linewidths=0)
    ax.plot([0, 150], [0, 150], color=MUTED, lw=1)
    ax.set(xlabel="Actual energy usage", ylabel="Out-of-fold prediction", xlim=(0, 150), ylim=(0, 150),
           title="Final model, out-of-fold (8,000 rows)")
    ax = axes[1]
    g = d.groupby("n_missing_test_like")
    rmse = g.apply(lambda x: np.sqrt(((x["actual"] - x["oof_pred_test_like_gaps"]) ** 2).mean()))
    ax.bar(rmse.index.astype(str), rmse.to_numpy(), 0.6, color=S1)
    for k, v in rmse.items():
        ax.text(str(k), v + 0.05, f"{v:.2f}\n(n={len(g.get_group(k))})", ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.set(xlabel="Sensors missing in the row (test-like injection)", ylabel="RMSE",
           ylim=(0, rmse.max() * 1.3), title="Graceful degradation as sensors drop out")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_final_model_fit.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train, test = load_train(), load_test()
    cc, X, ols, robust = fit_ols(train)
    tests = assumption_tests(train, cc, X, ols)
    tests.to_csv(OUT / "assumption_tests.csv", index=False)
    print(tests.to_string(index=False))
    coefs = coefficient_table(robust, X)
    coefs.to_csv(OUT / "linear_coefficients.csv", index=False)
    print(coefs[~coefs.term.str.startswith(("month", "bld"))].to_string(index=False))
    stab = slope_stability(cc)
    stab.to_csv(OUT / "slope_stability.csv", index=False)
    print(stab.to_string(index=False))
    with open(OUT / "linear_fit_summary.json", "w") as f:
        json.dump({"n": int(ols.nobs), "k": int(ols.df_model + 1), "R2_in_sample": ols.rsquared,
                   "adj_R2": ols.rsquared_adj, "resid_SD": float(np.sqrt(ols.scale))}, f, indent=2)
    fig_profiles(train)
    fig_drivers(cc, ols, X)
    fig_shift(train, test)
    fig_residuals(cc, ols)
    fig_model_comparison()
    fig_pred_vs_actual()


if __name__ == "__main__":
    main()
