import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

FEATURES = ["lag1", "lag2", "lag3", "roll3", "diff1"]
FIRST_TEST_MONTH = pd.Timestamp("2025-04-01")
INTERVAL = (10, 90)


def add_lags(flags):
    f = flags.sort_values(["Org Code", "month"]).copy()
    g = f.groupby("Org Code")["perf"]
    for k in (1, 2, 3):
        f[f"lag{k}"] = g.shift(k)
    f["roll3"] = f[["lag1", "lag2", "lag3"]].mean(axis=1)
    f["diff1"] = f["lag1"] - f["lag2"]
    return f


def backtest(flags, first_test_month=FIRST_TEST_MONTH):
    data = add_lags(flags).dropna(subset=["lag3"])
    months = sorted(m for m in data["month"].unique() if m >= first_test_month)

    rows = []
    signed = []
    for m in months:
        train = data[data["month"] < m]
        test = data[data["month"] == m]
        model = Ridge(alpha=1.0).fit(train[FEATURES], train["perf"])
        pred = model.predict(test[FEATURES])
        err_model = test["perf"].values - pred
        err_naive = test["perf"].values - test["lag1"].values
        rows.append(
            {
                "month": m,
                "naive_mae": np.abs(err_naive).mean(),
                "model_mae": np.abs(err_model).mean(),
            }
        )
        signed.append(err_model)

    table = pd.DataFrame(rows)
    signed = np.concatenate(signed)
    lo, hi = np.percentile(signed, INTERVAL)
    return {
        "table": table,
        "n_months": len(table),
        "naive_mae": float(table["naive_mae"].mean()),
        "model_mae": float(table["model_mae"].mean()),
        "months_model_better": int((table["model_mae"] < table["naive_mae"]).sum()),
        "interval_low": float(lo),
        "interval_high": float(hi),
    }


def forecast_next(flags, bt=None):
    bt = bt or backtest(flags)
    data = add_lags(flags).dropna(subset=["lag3"])
    model = Ridge(alpha=1.0).fit(data[FEATURES], data["perf"])

    last_month = flags["month"].max()
    next_month = last_month + pd.offsets.MonthBegin(1)

    recent = (
        flags.sort_values(["Org Code", "month"])
        .groupby("Org Code")["perf"]
        .apply(lambda s: s.tail(3).tolist())
    )
    latest = pd.DataFrame(recent.tolist(), index=recent.index, columns=["lag3", "lag2", "lag1"])
    latest["roll3"] = latest[["lag1", "lag2", "lag3"]].mean(axis=1)
    latest["diff1"] = latest["lag1"] - latest["lag2"]

    out = pd.DataFrame(index=latest.index)
    out["forecast"] = model.predict(latest[FEATURES])
    out["forecast_low"] = out["forecast"] + bt["interval_low"]
    out["forecast_high"] = out["forecast"] + bt["interval_high"]
    out["naive_forecast"] = latest["lag1"]
    out["forecast_month"] = next_month
    return out.reset_index()
