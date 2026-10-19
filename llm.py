import json

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_QUESTION_CHARS = 300

SYSTEM_PROMPT = (
    "You explain statistical process control (XmR) charts of NHS A&E 4-hour performance to a manager "
    "who is not a statistician. Use only the facts in the JSON the user sends. "
    "Do not state any number that is not in the facts. "
    "Do not suggest causes such as staffing, demand, winter pressure or a merger. "
    "The only exception: if name_changed is true, you may say a name change can follow a merger and may "
    "change what the trust covers. "
    "Write 4 to 6 plain sentences: what the chart shows, whether the trust is moving away from its usual level "
    "and in which direction, what the forecast says and how uncertain it is, and one thing worth checking next. "
    "A higher percentage is better. A flag is a reason to look, not a verdict. "
    "If the user asks a question, answer it from the facts only. If the facts cannot answer it, say so."
)


def build_facts(row, trust_flags, forecast_row, bt, baseline_label, n_monitored):
    g = trust_flags.sort_values("month")
    last3 = g.tail(3)
    return {
        "trust": row["org_name"],
        "code": row["Org Code"],
        "status_latest_3_months": row["status"],
        "baseline_period": baseline_label,
        "baseline_average_pct": round(float(row["baseline_mean"]), 1),
        "lower_control_limit_pct": round(float(g["lcl"].iloc[0]), 1),
        "upper_control_limit_pct": round(float(g["ucl"].iloc[0]), 1),
        "latest_month": f"{g['month'].iloc[-1]:%b %Y}",
        "latest_pct": round(float(row["latest_perf"]), 1),
        "last_3_months_pct": [round(float(v), 1) for v in last3["perf"]],
        "months_monitored": int(n_monitored),
        "months_with_worse_signal": int(row["worse_months"]),
        "months_with_better_signal": int(row["better_months"]),
        "forecast_month": f"{forecast_row['forecast_month']:%b %Y}",
        "forecast_pct": round(float(forecast_row["forecast"]), 1),
        "forecast_80pct_range": [round(float(forecast_row["forecast_low"]), 1), round(float(forecast_row["forecast_high"]), 1)],
        "forecast_below_lower_limit": bool(forecast_row["forecast"] < g["lcl"].iloc[0]),
        "forecast_typical_error_points": round(float(bt["model_mae"]), 1),
        "forecast_beats_last_month_guess_by_points": round(float(bt["naive_mae"] - bt["model_mae"]), 2),
        "name_changed": bool(row["names_used"] > 1),
        "baseline_includes_a_winter": True,
    }


def template_summary(facts):
    direction = "below" if facts["latest_pct"] < facts["baseline_average_pct"] else "above"
    parts = [
        f"{facts['trust']} was at {facts['latest_pct']}% in {facts['latest_month']}, {direction} its baseline average of "
        f"{facts['baseline_average_pct']}% ({facts['baseline_period']}). Status for the latest 3 months: {facts['status_latest_3_months']}.",
        f"Since the baseline, {facts['months_with_worse_signal']} of {facts['months_monitored']} months show a worse signal "
        f"and {facts['months_with_better_signal']} show a better one.",
        f"The forecast for {facts['forecast_month']} is {facts['forecast_pct']}%, with an 80% range of "
        f"{facts['forecast_80pct_range'][0]}% to {facts['forecast_80pct_range'][1]}%.",
    ]
    if facts["forecast_below_lower_limit"]:
        parts.append("That forecast is below the trust's lower control limit, so the trust is expected to stay outside its normal range.")
    if facts["name_changed"]:
        parts.append("The trust changed name during the period, so check whether its coverage changed before reading a shift as performance.")
    return " ".join(parts)


def explain(facts, question=None, api_key=None, model=None, client=None):
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not api_key and client is None:
        return template_summary(facts), "template"

    payload = {"facts": facts}
    if question:
        payload["question"] = question

    try:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
        response = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text").strip()
        if not text:
            raise ValueError("empty response")
        return text, "llm"
    except Exception as exc:
        return template_summary(facts), f"fallback ({type(exc).__name__})"
