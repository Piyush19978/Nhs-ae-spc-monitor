# A&E 4-hour performance by trust: SPC monitor with forecast

An interactive dashboard that applies statistical process control (XmR charts) to NHS England's monthly
A&E data, flags which trusts have moved away from their own normal level of performance, forecasts the next
month, and writes a plain-English summary of each trust.

Question it answers: **across England, which A&E departments have changed, in which direction, and where are they heading next month?**

## What it shows

- **Watchlist**: every trust with a status (Deteriorating, Improving, Normal variation) and a next-month forecast, searchable and downloadable.
- **Trust chart**: an XmR chart for one trust with baseline, control limits, flagged months and the forecast with its range.
- **Plain-English summary**: a button that explains the chart. A language model writes it when an API key is set, and a fixed template writes it otherwise.
- **How to read this**: method, rules and limits, in plain language.

## Findings (120 trusts, June 2023 to September 2026)

- Latest 3 months (Jul to Sep 2026): **18 trusts Deteriorating, 77 Improving, 25 Normal variation.**
- Performance rose across the system over the period, so better signals outnumber worse ones.
- Worse signals peak every winter: 34 trusts had one in Dec 2024 and 25 in Dec 2025, against 16 in both June 2025 and June 2026.
- Several trusts are flagged worse in more than 20 of 28 monitored months. That is a sustained move away from their early level, not a new dip.
- **Forecast**: 10 trusts are forecast to fall below their own lower control limit in Oct 2026. All 10 are already Deteriorating.

A flag means "look at this trust", not "this trust is failing".

## Method

- **Measure**: share of A&E attendances seen within 4 hours = 1 - (attendances over 4 hours / all attendances), with booked-appointment attendances included in both counts.
- **Chart**: XmR (individuals and moving range). Centre line = mean of the first 12 months. Control limits = mean +/- 2.66 x average moving range.
- **Baseline**: each trust's first 12 months (Jun 2023 to May 2024). A single average over all 40 months flagged 117 of 120 trusts, because the whole system improved.
- **Rules** (applied after the baseline): a point outside the limits; 7 or more points on one side of the mean; 7 or more points rising or falling; 2 of 3 points beyond two sigma. NHS guides vary between 6 and 7 points for the run and trend rules. This project uses 7.
- **Status**: Deteriorating if any of the latest 3 months has a worse signal; otherwise Improving if it has a better signal; otherwise Normal variation.

## Forecast

- **Model**: ridge regression on each trust's last three months (plus their mean and latest change), fitted on all 120 trusts together. One month ahead.
- **Evaluation**: rolling-origin backtest over 18 months (Apr 2025 to Sep 2026, including winter 2025-26). Each month was forecast using only earlier data, then compared with repeating last month.
- **Result**: average error 1.99 percentage points for the model against 2.04 for repeating last month. The model was closer in 14 of 18 months. That is a small gain.
- **Range**: the 10th to 90th percentile of the backtest errors, about -3.0 to +3.1 points. It is the same width for every trust.
- **Tried and dropped**: seasonal terms (same month last year, month-of-year) and gradient boosting did not beat the simple model on 27 usable months, so they were left out.
- **Not tested**: the forecast has no explicit winter term, and autumn into winter is where it matters most.

## Plain-English summary (language model)

The app sends a language model a JSON of figures already computed on the page (baseline, limits, flagged months, latest value, forecast and its range, whether the trust changed name). The system prompt tells it to use only those figures, not to invent numbers, and not to suggest causes such as staffing or demand. The user can also type a question, limited to 300 characters. If there is no key, or the call fails, the app shows a template summary built from the same figures.

Set a key (never commit it):

```bash
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_MODEL="claude-haiku-5-5"
```

or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. On Streamlit Community Cloud, add the key under the app's Secrets. Check the model name against the current Anthropic documentation, because model names change.

The language-model call has been tested only with a stand-in client, not against the live API. Responses can still be wrong, so the app tells the reader to check the summary against the chart.

## Data

NHS England, A&E Attendances and Emergency Admissions, monthly provider-level files, April 2023 to September 2026 (42 files): https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/

| Issue | Decision |
|---|---|
| Each file ends with a TOTAL row | Removed, otherwise England is counted twice |
| September 2024 file has 6 empty extra columns | Dropped |
| 14 trusts report zero patients waiting over 4 hours in Apr and May 2023, then 60-70% from June | Treated as a missing field, not 100% performance. April and May 2023 excluded |
| 3 trusts with incomplete histories (North Middlesex, North Bristol, Southport and Ormskirk) | Excluded |
| Walk-in centres and units without a major A&E | Excluded; only trusts with Type 1 attendances kept |
| Some trusts changed name part-way through | Trusts matched on organisation code, not name |

Final sample: 120 trusts x 40 months = 4,800 trust-months.

## Limitations

- A trust is compared with itself, not with other trusts or with the 4-hour standard.
- The baseline year includes the 2023-24 winter, so some baselines are not fully stable.
- Four rules across 120 trusts will produce some false alarms.
- Trusts that merged or renamed can show shifts caused by a change in what they cover. This has not been checked trust by trust.
- 40 months is a short series for judging seasonality.
- The forecast is only slightly better than repeating last month, so it mostly says a trust will stay near where it is.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

With Docker:

```bash
docker build -t ae-spc-monitor .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY="your-key" ae-spc-monitor
```

The app works without a key.

## Files

```
app.py                         Streamlit dashboard
spc.py                         XmR limits, signal rules, summary table, charts
forecast.py                    Ridge forecast, rolling backtest, prediction range
llm.py                         Facts, template summary, language-model call with fallback
data/spc_flags_all_trusts.csv  One row per trust per month: performance, limits and signals
.streamlit/config.toml         Theme
.streamlit/secrets.toml.example  Where the API key goes
Dockerfile, requirements.txt, .gitignore
```
