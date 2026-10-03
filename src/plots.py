"""Figures for the report and notebook. All use the shared style in fraud_pipeline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_pipeline import CAT, PALETTE as C, TARGET, set_style

set_style()


def _baseline(ax, rate, label=True):
    ax.axhline(rate, color=C["ink2"], lw=1, zorder=3)
    if label:
        ax.annotate(f"overall {rate:.1%}", xy=(0, rate), xycoords=("axes fraction", "data"),
                    xytext=(2, 3), textcoords="offset points", ha="left", va="bottom",
                    fontsize=7, color=C["ink2"])


def fig_risk_drivers(train: pd.DataFrame, path):
    """Fraud rate by binned feature value: the shapes the model has to learn."""
    base = train[TARGET].mean()
    panels = [
        ("transaction_amount", [0, 25, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 9000], "Transaction amount"),
        ("account_age", [0, 7, 30, 90, 180, 365, 730, 1500, 3000, 4000], "Account age (days)"),
        ("transactions_last_24h", [0, 2, 4, 6, 8, 10, 15, 27], "Transactions in last 24h"),
        ("transactions_last_1h", [-1, 0, 1, 2, 3, 4, 9], "Transactions in last 1h"),
        ("transaction_hour", list(range(-1, 24, 3)), "Hour of day"),
        ("new_device", None, "New device"),
        ("merchant_category", None, "Merchant category"),
        ("transaction_channel", None, "Channel"),
        ("country", None, "Country"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
    for ax, (col, bins, title) in zip(axes.flat, panels):
        if bins is not None:
            g = train.groupby(pd.cut(train[col], bins), observed=True)[TARGET].agg(["mean", "size"])
            labels = [f"{int(i.left) + 1 if col != 'transaction_amount' else int(i.left)}–{int(i.right)}" for i in g.index]
            if col == "transactions_last_1h":
                labels = [str(int(i.right)) if i.right - i.left == 1 else f"{int(i.left) + 1}+" for i in g.index]
        else:
            g = train.groupby(col)[TARGET].agg(["mean", "size"]).sort_values("mean")
            labels = [("yes" if v == 1 else "no") if col == "new_device" else str(v) for v in g.index]
        x = np.arange(len(g))
        ax.bar(x, g["mean"], width=0.7, color=C["blue"], zorder=2)
        _baseline(ax, base, label=(col == "transaction_amount"))
        ax.set_xticks(x, labels, rotation=45 if len(g) > 4 else 0, ha="right" if len(g) > 4 else "center")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=None))
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Fraud rate by feature value (training set)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_drift(train: pd.DataFrame, test: pd.DataFrame, adv_auc: float, adv_imp: pd.Series, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.7, 1]})

    ax = axes[0]
    bins = np.logspace(np.log10(2), np.log10(9000), 40)
    ax.hist(train["transaction_amount"], bins=bins, density=True, histtype="step", lw=2, color=C["blue"], label="Train")
    ax.hist(test["transaction_amount"], bins=bins, density=True, histtype="step", lw=2, color=C["orange"], label="Test")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Transaction amount"); ax.set_xlabel("amount (log scale)"); ax.set_ylabel("density (log)")
    ax.legend(loc="lower left")

    ax = axes[1]
    for d, color, label in [(train, C["blue"], "Train"), (test, C["orange"], "Test")]:
        v = np.sort(d["account_age"].dropna().to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=2, color=color, label=label)
    ax.set_xscale("log")
    ax.set_title("Account age (cumulative)"); ax.set_xlabel("days (log scale)")
    ax.set_ylabel("share of transactions ≤ x")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left")

    ax = axes[2]
    stats = pd.DataFrame({
        "Train": [train["new_device"].mean(), train[CAT + ["account_age"]].isna().any(axis=1).mean()],
        "Test": [test["new_device"].mean(), test[CAT + ["account_age"]].isna().any(axis=1).mean()],
    }, index=["new device", "any missing*"])
    x = np.arange(len(stats)); w = 0.34
    ax.bar(x - w / 2 - 0.01, stats["Train"], w, color=C["blue"], label="Train", zorder=2)
    ax.bar(x + w / 2 + 0.01, stats["Test"], w, color=C["orange"], label="Test", zorder=2)
    for i, (a, b) in enumerate(stats.values):
        ax.text(i - w / 2, a, f"{a:.0%}", ha="center", va="bottom", fontsize=7, color=C["ink2"])
        ax.text(i + w / 2, b, f"{b:.0%}", ha="center", va="bottom", fontsize=7, color=C["ink2"])
    ax.set_xticks(x, stats.index); ax.set_title("Rates"); ax.legend(loc="upper right")
    ax.set_ylim(0, stats.values.max() * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)

    ax = axes[3]
    imp = adv_imp.sort_values()
    ax.barh(np.arange(len(imp)), imp.values, height=0.6, color=C["blue"], zorder=2)
    ax.set_yticks(np.arange(len(imp)), imp.index)
    ax.set_title(f"What separates test from train (AUC {adv_auc:.2f})")
    ax.set_xlabel("share of adversarial-model gain")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, "*any of merchant_category, country, channel or account_age missing",
             fontsize=7, color=C["muted"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_model_comparison(cv: pd.DataFrame, prevalence: float, highlight: str, path):
    d = cv.sort_values("pr_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(d))
    colors = [C["blue"] if m == highlight else C["deemph"] for m in d["model"]]
    ax.barh(y, d["pr_auc_mean"], height=0.6, color=colors, zorder=2)
    ax.errorbar(d["pr_auc_mean"], y, xerr=d["pr_auc_std"], fmt="none", ecolor=C["ink2"], elinewidth=1, capsize=0, zorder=3)
    for yi, v in zip(y, d["pr_auc_mean"]):
        ax.text(0.004, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.5,
                color="white" if d["model"].iloc[yi] == highlight else C["ink"], zorder=4)
    ax.axvline(prevalence, color=C["ink2"], lw=1)
    ax.annotate(f"random classifier = {prevalence:.3f}", xy=(prevalence, len(d) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=C["ink2"], va="center")
    ax.set_yticks(y, d["model"])
    ax.set_xlabel("PR-AUC (mean ± 1 sd over 15 folds: 5-fold × 3 repeats)")
    ax.set_title("Cross-validated PR-AUC by model")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pr_curve(y, oof_mean, threshold, path):
    pr, rc, th = precision_recall_curve(y, oof_mean)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(rc, pr, color=C["blue"], lw=2, label="EBM (out-of-fold)")
    ax.axhline(y.mean(), color=C["ink2"], lw=1)
    ax.annotate(f"no-skill = {y.mean():.3f}", xy=(1, y.mean()), xytext=(-2, 4), textcoords="offset points",
                ha="right", fontsize=7, color=C["ink2"])
    k = np.searchsorted(th, threshold)
    ax.scatter([rc[k]], [pr[k]], s=60, color=C["orange"], edgecolor=C["surface"], linewidth=2, zorder=5)
    ax.annotate(f"operating point t={threshold:.3f}\nP={pr[k]:.2f}, R={rc[k]:.2f}", xy=(rc[k], pr[k]),
                xytext=(14, 18), textcoords="offset points", fontsize=8, color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.8))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_threshold(table: pd.DataFrame, threshold: float, beta: float, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    t = table["threshold"]
    series = [("precision", "Precision", C["deemph"]), ("recall", "Recall", C["aqua"]),
              ("f1", "F1", C["orange"]), ("f_beta", f"F-beta (beta=sqrt 3)", C["blue"])]
    for col, label, color in series:
        ax.plot(t, table[col], color=color, lw=2, label=label)
    ax.axvline(threshold, color=C["ink2"], lw=1)
    ax.annotate(f"chosen t = {threshold:.3f}", xy=(threshold, 0.97), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=C["ink2"], va="top")
    f1_best = table.loc[table["f1"].idxmax(), "threshold"]
    ax.axvline(f1_best, color=C["grid"], lw=1)
    ax.annotate(f"F1-optimal t = {f1_best:.3f}", xy=(f1_best, 0.88), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=C["muted"], va="top")
    ax.set_xlim(0, 0.4); ax.set_ylim(0, 1)
    ax.set_xlabel("Decision threshold on predicted fraud probability")
    ax.set_title("Metric trade-off vs threshold (out-of-fold, averaged over 3 repeats)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(y, oof_mean, path, n_bins=10):
    q = pd.qcut(oof_mean, n_bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": oof_mean, "y": y}).groupby(q).mean()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lo = min(g["p"].min(), g["y"].min()) * 0.7
    hi = max(g["p"].max(), g["y"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color=C["axis"], lw=1)
    ax.plot(g["p"], g["y"], color=C["blue"], lw=2, marker="o", ms=6, mec=C["surface"], mew=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean predicted probability (decile, log scale)")
    ax.set_ylabel("Observed fraud rate (log scale)")
    ax.set_title("Calibration (out-of-fold)")
    ax.annotate("perfect calibration", xy=(hi * 0.5, hi * 0.5), xytext=(4, -10), textcoords="offset points",
                fontsize=7, color=C["muted"], rotation=40)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ebm_importance(ebm, path, top=15):
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = [C["blue"] if " & " not in n else C["violet"] for n in imp.index]
    ax.barh(np.arange(len(imp)), imp.values, height=0.6, color=colors, zorder=2)
    ax.set_yticks(np.arange(len(imp)), imp.index)
    ax.set_xlabel("Mean |contribution| to log-odds")
    ax.set_title("EBM global importance (blue = main effect, violet = pairwise interaction)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ebm_shapes(ebm, path):
    """Shape function of every main effect: contribution to log-odds by feature value."""
    glob = ebm.explain_global()
    mains = [i for i, n in enumerate(ebm.term_names_) if " & " not in n]
    ncols = 4
    nrows = int(np.ceil(len(mains) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows))
    for ax in axes.flat[len(mains):]:
        ax.set_visible(False)
    for ax, i in zip(axes.flat, mains):
        d = glob.data(i)
        name = ebm.term_names_[i]
        scores = np.asarray(d["scores"])
        if ebm.feature_types_in_[ebm.term_features_[i][0]] == "continuous":
            edges = np.asarray(d["names"], dtype=float)
            ax.stairs(scores, edges, color=C["blue"], lw=2, baseline=None)
            if "upper_bounds" in d and d["upper_bounds"] is not None:
                ax.stairs(np.asarray(d["upper_bounds"]), edges, color=C["blue"], alpha=0.25, lw=1, baseline=None)
                ax.stairs(np.asarray(d["lower_bounds"]), edges, color=C["blue"], alpha=0.25, lw=1, baseline=None)
            if name in ("transaction_amount", "spend_last_24h"):
                ax.set_xscale("symlog", linthresh=10)
        else:
            labels = [str(n) for n in d["names"]]
            order = np.argsort(scores)
            ax.bar(np.arange(len(scores)), scores[order], width=0.7, color=C["blue"], zorder=2)
            ax.set_xticks(np.arange(len(scores)), [labels[k] for k in order], rotation=45, ha="right")
            ax.grid(axis="x", visible=False)
        ax.axhline(0, color=C["axis"], lw=1)
        ax.set_title(name)
    axes.flat[0].set_ylabel("log-odds contribution")
    fig.suptitle("EBM shape functions: how each feature moves the fraud log-odds", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_local_explanation(ebm, x_row: pd.DataFrame, title, path, top=10):
    loc = ebm.explain_local(x_row).data(0)
    contrib = pd.Series(loc["scores"], index=loc["names"])
    vals = pd.Series(loc["values"], index=loc["names"])
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)[::-1]
    labels = []
    for n in contrib.index:
        v = vals[n]
        if isinstance(v, (tuple, list)):
            v = ", ".join(f"{x:,.0f}" if isinstance(x, float) and abs(x) >= 10 else str(x) for x in v)
        elif isinstance(v, float):
            v = f"{v:,.0f}" if abs(v) >= 10 else f"{v:g}"
        labels.append(f"{n} = {v}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [C["orange"] if s > 0 else C["blue"] for s in contrib.values]
    ax.barh(np.arange(len(contrib)), contrib.values, height=0.6, color=colors, zorder=2)
    ax.axvline(0, color=C["axis"], lw=1)
    ax.set_yticks(np.arange(len(contrib)), labels)
    ax.set_xlabel("Contribution to log-odds (orange raises risk, blue lowers it)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
