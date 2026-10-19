import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CONCERN = "#ED8B00"
IMPROVE = "#005EB8"
COMMON = "#768692"
INK = "#212B32"
GRID = "#D8DDE0"
BASELINE_FILL = "#E8EDEE"
N_BASE = 12


def run_flags(cond, min_len):
    cond = np.asarray(cond, dtype=bool)
    out = np.zeros(len(cond), dtype=bool)
    start = None
    for i in range(len(cond) + 1):
        on = i < len(cond) and cond[i]
        if on and start is None:
            start = i
        if not on and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    return out


def xmr_signals(x, n_base=N_BASE, run_len=7, mask_baseline=True):
    x = np.asarray(x, dtype=float)
    base = x[:n_base]
    mean = base.mean()
    mr_bar = np.abs(np.diff(base)).mean()
    ucl = mean + 2.66 * mr_bar
    lcl = mean - 2.66 * mr_bar
    sigma = mr_bar / 1.128
    upper2 = mean + 2 * sigma
    lower2 = mean - 2 * sigma
    n = len(x)

    high = np.zeros(n, dtype=bool)
    low = np.zeros(n, dtype=bool)

    high |= x > ucl
    low |= x < lcl

    high |= run_flags(x > mean, run_len)
    low |= run_flags(x < mean, run_len)

    steps = np.diff(x)
    up = run_flags(steps > 0, run_len - 1)
    down = run_flags(steps < 0, run_len - 1)
    up_pts = np.zeros(n, dtype=bool)
    down_pts = np.zeros(n, dtype=bool)
    up_pts[:-1] |= up
    up_pts[1:] |= up
    down_pts[:-1] |= down
    down_pts[1:] |= down
    high |= up_pts
    low |= down_pts

    for i in range(2, n):
        w = slice(i - 2, i + 1)
        if (x[w] > upper2).sum() >= 2:
            high[w] |= x[w] > upper2
        if (x[w] < lower2).sum() >= 2:
            low[w] |= x[w] < lower2

    if mask_baseline:
        high[:n_base] = False
        low[:n_base] = False

    return mean, ucl, lcl, high, low


def load_flags(path):
    flags = pd.read_csv(path, parse_dates=["month"])
    return flags.sort_values(["Org Code", "month"]).reset_index(drop=True)


def build_summary(flags, recent_months=3):
    months = sorted(flags["month"].unique())
    recent = flags[flags["month"].isin(months[-recent_months:])]

    summary = (
        flags.groupby("Org Code")
        .agg(
            org_name=("Org name", "last"),
            names_used=("Org name", "nunique"),
            baseline_mean=("mean", "first"),
            latest_perf=("perf", "last"),
            worse_months=("low_signal", "sum"),
            better_months=("high_signal", "sum"),
        )
        .reset_index()
    )

    recent_flags = (
        recent.groupby("Org Code")
        .agg(worse_recent=("low_signal", "any"), better_recent=("high_signal", "any"))
        .reset_index()
    )
    summary = summary.merge(recent_flags, on="Org Code")

    summary["status"] = np.where(
        summary["worse_recent"],
        "Deteriorating",
        np.where(summary["better_recent"], "Improving", "Normal variation"),
    )
    return summary


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_trust(g, title, fc=None):
    g = g.sort_values("month").reset_index(drop=True)
    concern = g["low_signal"].values
    improve = g["high_signal"].values & ~concern
    normal = ~(concern | improve)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(
        g["month"].iloc[0],
        g["month"].iloc[N_BASE - 1],
        color=BASELINE_FILL,
        label="Baseline: limits calculated from these 12 months",
    )
    ax.plot(g["month"], g["perf"], color=COMMON, linewidth=1.4, zorder=2)
    ax.scatter(g.loc[normal, "month"], g.loc[normal, "perf"], s=22, color=COMMON, zorder=3, label="Normal variation")
    ax.scatter(g.loc[improve, "month"], g.loc[improve, "perf"], s=46, color=IMPROVE, zorder=4, label="Signal: better")
    ax.scatter(g.loc[concern, "month"], g.loc[concern, "perf"], s=52, marker="D", color=CONCERN, zorder=4, label="Signal: worse")

    if fc is not None:
        last_x, last_y = g["month"].iloc[-1], g["perf"].iloc[-1]
        ax.plot([last_x, fc["forecast_month"]], [last_y, fc["forecast"]], color=INK, linewidth=1, linestyle=":", zorder=2)
        ax.errorbar(
            [fc["forecast_month"]],
            [fc["forecast"]],
            yerr=[[fc["forecast"] - fc["forecast_low"]], [fc["forecast_high"] - fc["forecast"]]],
            fmt="s",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=INK,
            ecolor=INK,
            capsize=4,
            zorder=5,
            label="Forecast, next month, with 80% range",
        )

    ax.axhline(g["mean"].iloc[0], color=INK, linewidth=1.2, label="Baseline average")
    ax.axhline(g["ucl"].iloc[0], color=INK, linewidth=1, linestyle="--", alpha=0.7, label="Control limits")
    ax.axhline(g["lcl"].iloc[0], color=INK, linewidth=1, linestyle="--", alpha=0.7)

    ax.set_ylabel("Seen within 4 hours (%)", color=INK)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_monthly_counts(flags):
    months = sorted(flags["month"].unique())[N_BASE:]
    monitored = flags[flags["month"].isin(months)]
    counts = monitored.groupby("month").agg(
        worse=("low_signal", "sum"), better=("high_signal", "sum")
    )

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(counts.index, counts["better"], color=IMPROVE, linewidth=2, label="Trusts with a better signal")
    ax.plot(counts.index, counts["worse"], color=CONCERN, linewidth=2, label="Trusts with a worse signal")
    ax.set_ylabel("Number of trusts", color=INK)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style_axis(ax)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
