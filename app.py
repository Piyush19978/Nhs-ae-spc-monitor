import json
import os
from pathlib import Path

import streamlit as st

from forecast import backtest, forecast_next
from llm import DEFAULT_MODEL, build_facts, explain
from spc import N_BASE, build_summary, load_flags, plot_monthly_counts, plot_trust

DATA_PATH = Path(__file__).parent / "spc_flags_all_trusts.csv""
STATUS_ORDER = {"Deteriorating": 0, "Improving": 1, "Normal variation": 2}

st.set_page_config(page_title="A&E 4-hour performance by trust", layout="wide")


@st.cache_data
def get_data():
    flags = load_flags(DATA_PATH)
    bt = backtest(flags)
    fc = forecast_next(flags, bt)
    limits = flags.groupby("Org Code")["lcl"].first().reset_index()
    summary = build_summary(flags).merge(fc, on="Org Code").merge(limits, on="Org Code")
    summary["forecast_below_lcl"] = summary["forecast"] < summary["lcl"]
    return flags, summary, bt


def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


flags, summary, bt = get_data()
months = sorted(flags["month"].unique())
monitored_months = months[N_BASE:]
n_monitored = len(monitored_months)
baseline_label = f"{months[0]:%b %Y} to {months[N_BASE - 1]:%b %Y}"
recent_label = f"{months[-3]:%b %Y} to {months[-1]:%b %Y}"
next_label = f"{summary['forecast_month'].iloc[0]:%b %Y}"
api_key = get_api_key()
model_name = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

st.title("A&E 4-hour performance by trust")
st.caption(
    f"Monthly NHS England A&E data, {months[0]:%b %Y} to {months[-1]:%b %Y}. "
    f"Each trust is compared with its own first 12 months ({baseline_label})."
)

counts = summary["status"].value_counts()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trusts monitored", len(summary), help="Trusts with a major A&E department and a full monthly record.")
k2.metric(
    "Deteriorating",
    int(counts.get("Deteriorating", 0)),
    help=f"At least one worse signal in the latest 3 months ({recent_label}).",
)
k3.metric("Improving", int(counts.get("Improving", 0)), help="A better signal in the latest 3 months and no worse signal.")
k4.metric("Normal variation", int(counts.get("Normal variation", 0)), help="No signal in the latest 3 months.")
k5.metric(
    f"Forecast below limit, {next_label}",
    int(summary["forecast_below_lcl"].sum()),
    help="Trusts whose forecast for next month is under their own lower control limit.",
)

tab_watch, tab_trust, tab_method = st.tabs(["Watchlist", "Trust chart", "How to read this"])

with tab_watch:
    f1, f2 = st.columns([2, 1])
    chosen = f1.multiselect("Show trusts that are", list(STATUS_ORDER), default=["Deteriorating"])
    query = f2.text_input("Search by trust name")
    only_low = st.checkbox(f"Only trusts forecast to fall below their lower limit in {next_label}")

    table = summary[summary["status"].isin(chosen)].copy()
    if query:
        table = table[table["org_name"].str.contains(query, case=False, na=False)]
    if only_low:
        table = table[table["forecast_below_lcl"]]
    table["order"] = table["status"].map(STATUS_ORDER)
    table = table.sort_values(["order", "worse_months"], ascending=[True, False])

    latest_col = f"Latest, {months[-1]:%b %Y} (%)"
    forecast_col = f"Forecast, {next_label} (%)"
    view = table[
        ["org_name", "Org Code", "status", "worse_months", "better_months", "baseline_mean", "latest_perf", "forecast", "forecast_below_lcl"]
    ].rename(
        columns={
            "org_name": "Trust",
            "Org Code": "Code",
            "status": "Status",
            "worse_months": "Months worse",
            "better_months": "Months better",
            "baseline_mean": "Baseline average (%)",
            "latest_perf": latest_col,
            "forecast": forecast_col,
            "forecast_below_lcl": "Forecast below lower limit",
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "Baseline average (%)": st.column_config.NumberColumn(format="%.1f"),
            latest_col: st.column_config.NumberColumn(format="%.1f"),
            forecast_col: st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        f"Months worse and months better count the {n_monitored} months after the baseline "
        f"({monitored_months[0]:%b %Y} to {months[-1]:%b %Y}) flagged by at least one rule. "
        "A trust with a high count has been away from its early level for a long time, "
        "not just for the latest month."
    )
    st.download_button(
        "Download this table",
        view.to_csv(index=False).encode("utf-8"),
        file_name="ae_spc_watchlist.csv",
        mime="text/csv",
    )

    st.subheader("Trusts flagged each month")
    st.pyplot(plot_monthly_counts(flags))

with tab_trust:
    by_name = summary.sort_values("org_name")
    codes = by_name["Org Code"].tolist()
    names = dict(zip(by_name["Org Code"], by_name["org_name"]))
    worst = summary[summary["status"] == "Deteriorating"].sort_values("worse_months", ascending=False)
    default_code = worst["Org Code"].iloc[0] if len(worst) else codes[0]

    code = st.selectbox(
        "Trust",
        codes,
        index=codes.index(default_code),
        format_func=lambda c: f"{names[c]} ({c})",
    )
    row = summary[summary["Org Code"] == code].iloc[0]
    trust_flags = flags[flags["Org Code"] == code]

    st.pyplot(plot_trust(trust_flags, names[code], fc=row))

    st.markdown(
        f"**Status for {recent_label}: {row['status']}.** "
        f"Since {monitored_months[0]:%b %Y}, {int(row['worse_months'])} of {n_monitored} months show a worse signal "
        f"and {int(row['better_months'])} show a better signal. "
        f"The latest month ({months[-1]:%b %Y}) was {row['latest_perf']:.1f}%, "
        f"against a baseline average of {row['baseline_mean']:.1f}%. "
        f"Forecast for {next_label}: {row['forecast']:.1f}% "
        f"(80% range {row['forecast_low']:.1f}% to {row['forecast_high']:.1f}%)."
    )
    if row["forecast_below_lcl"]:
        st.warning(f"The forecast for {next_label} is below this trust's lower control limit ({row['lcl']:.1f}%).")
    if row["worse_months"] >= 0.7 * n_monitored:
        st.info(
            "Most monitored months are flagged as worse. This is a sustained move away from the "
            "trust's early level, not a recent dip."
        )
    if row["names_used"] > 1:
        st.warning(
            "This trust changed name during the period, which can follow a merger. "
            "A shift in its chart may reflect a change in what the trust covers, not a change in performance."
        )

    st.subheader("Plain-English summary")
    question = st.text_input("Ask a question about this trust (optional)", max_chars=300, key=f"question_{code}")
    if question and not api_key:
        st.caption("Questions need an API key. Without one, you get the template summary below.")
    if st.button("Explain this trust"):
        facts = build_facts(row, trust_flags, row, bt, baseline_label, n_monitored)
        cache_key = (json.dumps(facts, sort_keys=True), question.strip())
        store = st.session_state.setdefault("llm_cache", {})
        if cache_key in store:
            text, source = store[cache_key], "llm"
        else:
            with st.spinner("Writing the summary"):
                text, source = explain(facts, question, api_key=api_key, model=model_name)
            if source == "llm":
                store[cache_key] = text
        st.markdown(text)
        if source == "llm":
            st.caption(f"Written by {model_name} from the figures on this page. Check it against the chart.")
        elif source == "template":
            st.caption(
                "Template summary built from the same figures. Set ANTHROPIC_API_KEY to get a language-model "
                "summary and to ask questions."
            )
        else:
            st.caption(f"The language model could not be reached ({source}). Showing the template summary instead.")

with tab_method:
    st.markdown(
        f"""
**What the chart shows.** Each dot is one month of A&E performance: the share of attendances seen,
admitted or discharged within 4 hours. The solid line is the trust's average over its first 12 months
({baseline_label}). The dashed lines are control limits, set at the average plus or minus 2.66 times the
average month-to-month change (an XmR chart, the method taught in NHS England's Making Data Count programme).
Months inside the limits are ordinary variation.

**When a month is flagged.** After the baseline, a month is flagged if any of these happen:

- a point falls outside the control limits
- 7 or more points in a row sit on the same side of the average
- 7 or more points in a row rise, or 7 or more in a row fall
- 2 of 3 points in a row fall beyond two sigma on the same side

Different NHS guides use 6 or 7 points for the run and trend rules. This dashboard uses 7.
Orange diamonds mark worse signals. Blue circles mark better signals. A higher percentage is better.

**Status.** A trust is Deteriorating if any of the latest 3 months has a worse signal ({recent_label}).
It is Improving if it has a better signal and no worse one, and Normal variation if it has neither.

**Forecast.** The square on the chart is a one-month-ahead forecast for {next_label}. It comes from a ridge
regression on each trust's last three months, fitted on all 120 trusts together. It was tested by repeatedly
forecasting one month ahead over {bt['n_months']} past months using only earlier data. Its average error was
{bt['model_mae']:.2f} percentage points, against {bt['naive_mae']:.2f} for simply repeating last month, and it was
closer in {bt['months_model_better']} of {bt['n_months']} months. That is a small gain. The vertical bar shows the range
that held 80% of the time in those tests, about {bt['interval_low']:.1f} to +{bt['interval_high']:.1f} points, and it is the same
width for every trust. The model has no explicit winter term, because there are only three winters in the data.

**Plain-English summary.** The button writes a short explanation from the figures shown on this page. With an
Anthropic API key set, a language model writes it and can answer a question. It is told to use only these figures and not to
suggest causes. Without a key, a fixed template writes the summary. Read the summary against the chart.

**Data.** NHS England monthly A&E attendances and emergency admissions, trust level. The analysis covers
120 trusts with a major A&E department (Type 1) and a full record from June 2023 to September 2026.
April and May 2023 were dropped because 14 trusts reported zero patients waiting over 4 hours,
which reflects a missing field, not real performance. Three trusts with incomplete histories were excluded.
Booked-appointment attendances are included in both the attendance and the over-4-hour counts.

**Limits of this analysis.**

- A trust is compared with itself, not with other trusts or the 4-hour target.
- Nationally, performance rose over the period, so many trusts show better signals.
- The baseline year includes the 2023-24 winter, so some baselines are not fully stable.
- Running four rules across 120 trusts will produce some false alarms. Treat a flag as a reason to look, not a verdict.
- Trusts that merged or renamed can show shifts caused by a change in what they cover.
- The forecast is only slightly better than repeating last month, so it mostly says a trust will stay near where it is.
"""
    )
