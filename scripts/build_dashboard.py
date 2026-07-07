#!/usr/bin/env python3
"""Build a static HTML dashboard for the gold model outputs."""

import json
import re
from pathlib import Path

import pandas as pd
from jinja2 import Template


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def to_records(df, columns):
    records = []
    for row in df[columns].to_dict(orient="records"):
        records.append({key: (None if pd.isna(value) else value) for key, value in row.items()})
    return records


def metric(value, decimals=0, prefix="", suffix=""):
    return "{}{:,.{}f}{}".format(prefix, value, decimals, suffix)


def pct_metric(value, decimals=1):
    return "{:+,.{}f}%".format(value * 100, decimals)


def build_short_term_payload():
    signal_path = OUTPUT_DIR / "short_term_signal_latest.csv"
    weekly_path = OUTPUT_DIR / "short_term_regression_weekly.csv"
    summary_path = OUTPUT_DIR / "short_term_model_summary.txt"
    if not signal_path.exists() or not weekly_path.exists() or not summary_path.exists():
        return None, None

    signal = pd.read_csv(signal_path)
    weekly = pd.read_csv(weekly_path)
    summary = summary_path.read_text()
    latest_rows = weekly.dropna(subset=["predicted_4w_return", "gold_usd"])
    if latest_rows.empty:
        return None, None
    latest = latest_rows.iloc[-1]

    label_map = {
        "real_yield_pressure": "Real-rate pressure",
        "dollar_pressure": "Dollar pressure",
        "etf_flow_pressure": "ETF flows",
        "futures_position_pressure": "Futures positioning",
        "momentum_pressure": "Momentum",
        "market_risk_pressure": "Market risk",
    }
    weights = []
    for row in signal.to_dict(orient="records"):
        weights.append(
            {
                "label": label_map.get(row["factor"], row["factor"]),
                "factor": row["factor"],
                "weight": row["weight_abs_pct"],
                "coefficient": row["coefficient"],
                "z": row["current_z"],
                "contribution": row["current_contribution"],
            }
        )

    r2_match = re.search(r"R2:\s*([0-9.]+)", summary)
    obs_match = re.search(r"Observations:\s*(\d+)", summary)
    sample_match = re.search(r"Sample:\s*(.*?)\n", summary)
    latest_complete = latest["week"]
    cards = [
        ("Short-term signal", "Neutral" if abs(latest["predicted_4w_return"]) < 0.02 else ("Bullish" if latest["predicted_4w_return"] > 0 else "Bearish")),
        ("Predicted 4-week move", pct_metric(latest["predicted_4w_return"], 1)),
        ("Signal date", latest_complete),
        ("Regression sample", "{} weeks".format(obs_match.group(1) if obs_match else "n/a")),
        ("Backtest R2", "{:.1f}%".format(float(r2_match.group(1)) * 100) if r2_match else "n/a"),
    ]
    history = weekly.dropna(subset=["predicted_4w_return"]).tail(52)
    payload = {
        "history": to_records(history, ["week", "predicted_4w_return", "actual_4w_forward_return"]),
    }
    view = {
        "cards": cards,
        "weights": weights,
        "sample": sample_match.group(1) if sample_match else "2024 onward",
        "latestGold": metric(latest["gold_usd"], 0, "$", "/oz"),
        "latestWeek": latest_complete,
        "note": (
            "This is a 2024-onward historical regression for the next 4 weeks. "
            "It is a short-term pressure gauge, not the long-term fair-value model."
        ),
    }
    return payload, view


def build_dashboard():
    upgraded = pd.read_csv(OUTPUT_DIR / "upgraded_real_gold_fit.csv")
    legacy = pd.read_csv(OUTPUT_DIR / "legacy_four_factor_fit.csv")
    latest_path = OUTPUT_DIR / "latest_available_nowcast.csv"
    latest_nowcast = pd.read_csv(latest_path) if latest_path.exists() else None
    df = upgraded.merge(
        legacy[["quarter", "fitted_gold_usd", "residual_pct_gold_usd"]],
        on="quarter",
        how="left",
        suffixes=("", "_legacy"),
    )
    latest_complete = df.iloc[-1]

    chart_df = df.copy()
    nowcast_note = ""
    latest_label = latest_complete["quarter"]
    latest = latest_complete
    if latest_nowcast is not None and not latest_nowcast.empty:
        latest = latest_nowcast.iloc[-1]
        latest_label = "Latest*"
        nowcast_note = (
            "Latest* uses spot gold from {date}, quarter-to-date price data, and the latest reported "
            "central-bank buying since Q1. It is not a complete quarterly observation."
        ).format(date=latest["latest_spot_date"])
        append_row = {
            "quarter": latest_label,
            "gold_usd": latest["spot_gold_usd"],
            "upgraded_fitted_gold_usd": latest["upgraded_fitted_gold_usd"],
            "fitted_gold_usd": None,
            "upgraded_residual_pct": latest["spot_residual_pct"],
            "residual_pct_gold_usd": None,
            "real_gold_usd": latest["real_gold_usd"],
            "fitted_real_gold_usd": latest["fitted_real_gold_usd"],
            "cumulative_central_bank_purchase_tonnes": latest["cumulative_central_bank_purchase_tonnes"],
            "debt_to_gdp": latest["debt_to_gdp"],
            "broad_dollar_index": latest["broad_dollar_index"],
            "consumer_sentiment": latest["consumer_sentiment"],
        }
        chart_df = pd.concat([chart_df, pd.DataFrame([append_row])], ignore_index=True)

    short_payload, short_term = build_short_term_payload()
    payload = {
        "quarters": chart_df["quarter"].tolist(),
        "price": to_records(
            chart_df,
            ["quarter", "gold_usd", "upgraded_fitted_gold_usd", "fitted_gold_usd"],
        ),
        "gap": to_records(
            chart_df,
            ["quarter", "upgraded_residual_pct", "residual_pct_gold_usd"],
        ),
        "real_price": to_records(
            chart_df,
            ["quarter", "real_gold_usd", "fitted_real_gold_usd"],
        ),
        "drivers": to_records(
            chart_df,
            [
                "quarter",
                "cumulative_central_bank_purchase_tonnes",
                "debt_to_gdp",
                "broad_dollar_index",
                "consumer_sentiment",
            ],
        ),
    }
    if short_payload is not None:
        payload["short"] = short_payload

    if latest_nowcast is not None and not latest_nowcast.empty:
        cards = [
            ("Spot gold", metric(latest["spot_gold_usd"], 0, "$", "/oz")),
            ("Nowcast fit", metric(latest["upgraded_fitted_gold_usd"], 0, "$", "/oz")),
            ("Spot premium", metric(latest["spot_residual_pct"] * 100, 1, "", "%")),
            ("CB reported since Q1", metric(latest["central_bank_net_purchase_tonnes"], 0, "", "t")),
            ("US debt/GDP", metric(latest["debt_to_gdp"] * 100, 1, "", "%")),
        ]
    else:
        cards = [
            ("Actual gold", metric(latest["gold_usd"], 0, "$", "/oz")),
            ("Upgraded fit", metric(latest["upgraded_fitted_gold_usd"], 0, "$", "/oz")),
            ("Premium", metric(latest["upgraded_residual_pct"] * 100, 1, "", "%")),
            ("CB cumulative buying", metric(latest["cumulative_central_bank_purchase_tonnes"], 0, "", "t")),
            ("US debt/GDP", metric(latest["debt_to_gdp"] * 100, 1, "", "%")),
        ]
    complete_cards = [
        ("Latest complete quarter", latest_complete["quarter"]),
        ("Complete-quarter gold", metric(latest_complete["gold_usd"], 0, "$", "/oz")),
        ("Complete-quarter fit", metric(latest_complete["upgraded_fitted_gold_usd"], 0, "$", "/oz")),
        ("Complete-quarter premium", metric(latest_complete["upgraded_residual_pct"] * 100, 1, "", "%")),
        ("Latest cumulative buying", metric(latest["cumulative_central_bank_purchase_tonnes"], 0, "", "t")),
    ]
    model_formula = {
        "realGoldFormula": "Real gold = Nominal gold × CPI base / Current CPI",
        "fitFormula": "ln(Real gold) = 7.6642 + 0.2585×Z(CB cumulative buying) + 0.0438×Z(US debt/GDP) + 0.0143×Z(Weak consumer sentiment) - 0.1028×Z(Broad dollar)",
        "nominalFormula": "Nominal fair value = exp(fitted ln real gold) × Current CPI / CPI base",
        "premiumFormula": "Premium = Spot gold / Nominal fair value - 1",
        "latestFit": metric(latest["upgraded_fitted_gold_usd"], 0, "$", "/oz"),
        "latestSpot": metric(latest["spot_gold_usd"], 0, "$", "/oz") if "spot_gold_usd" in latest else metric(latest["gold_usd"], 0, "$", "/oz"),
        "latestPremium": metric((latest["spot_residual_pct"] if "spot_residual_pct" in latest else latest["upgraded_residual_pct"]) * 100, 1, "", "%"),
    }

    html = Template(TEMPLATE).render(
        data=json.dumps(payload),
        cards=cards,
        complete_cards=complete_cards,
        latest_quarter=latest_label,
        nowcast_note=nowcast_note,
        model_formula=model_formula,
        short_term=short_term,
    )
    out = OUTPUT_DIR / "gold_model_dashboard.html"
    out.write_text(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gold Model Dashboard</title>
  <style>
    :root {
      --bg: #f7f7f3;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #687184;
      --line: #d9dddf;
      --gold: #c6922e;
      --blue: #315f8c;
      --green: #28765a;
      --red: #b8504b;
      --ink: #293241;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, calc(100% - 36px));
      margin: 28px auto 44px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 22px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 14px 12px;
      min-height: 82px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .value {
      font-size: 24px;
      font-weight: 650;
      line-height: 1.1;
    }
    .note {
      background: #fff8e7;
      border: 1px solid #ead9ad;
      border-radius: 8px;
      padding: 12px 14px;
      margin: -4px 0 18px;
      color: #5b4a1f;
      font-size: 13px;
      line-height: 1.45;
    }
    .subcards {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin: -6px 0 18px;
    }
    .subcard {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: rgba(255,255,255,.65);
    }
    .subcard .value { font-size: 17px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .model-structure {
      display: grid;
      grid-template-columns: 0.95fr 1.05fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    .flow {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .flow-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .flow-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 74px;
      background: #fbfbf8;
    }
    .flow-box strong {
      display: block;
      font-size: 13px;
      margin-bottom: 5px;
    }
    .flow-box span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .arrow {
      text-align: center;
      color: var(--muted);
      font-size: 18px;
      line-height: 1;
    }
    .formula-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .formula {
      border-left: 3px solid var(--green);
      background: #f7faf7;
      padding: 10px 12px;
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #253226;
      overflow-wrap: anywhere;
    }
    .formula-result {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .formula-result div {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
    }
    .short-grid {
      display: grid;
      grid-template-columns: 1.05fr .95fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    .mini-cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .mini-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfbf8;
    }
    .mini-card .value {
      font-size: 17px;
    }
    .weights {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .weight-row {
      display: grid;
      grid-template-columns: 150px 1fr 72px 82px;
      align-items: center;
      gap: 10px;
      font-size: 12px;
    }
    .bar-track {
      height: 8px;
      border-radius: 999px;
      background: #edf0ee;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: var(--green);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    section.wide {
      grid-column: 1 / -1;
    }
    h2 {
      font-size: 16px;
      line-height: 1.2;
      margin: 0 0 4px;
      letter-spacing: 0;
    }
    .chart {
      width: 100%;
      height: 330px;
      margin-top: 12px;
    }
    .chart.small {
      height: 280px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
    svg { width: 100%; height: 100%; display: block; }
    .axis text { fill: var(--muted); font-size: 11px; }
    .axis line, .gridline { stroke: var(--line); stroke-width: 1; }
    .axis path { stroke: var(--line); }
    .tooltip {
      position: fixed;
      pointer-events: none;
      background: #202733;
      color: white;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,.18);
      display: none;
      z-index: 4;
      max-width: 260px;
    }
    @media (max-width: 900px) {
      header { display: block; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .subcards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .model-structure { grid-template-columns: 1fr; }
      .short-grid { grid-template-columns: 1fr; }
      .mini-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .weight-row { grid-template-columns: 1fr 1fr 56px 72px; }
      .flow-row { grid-template-columns: 1fr; }
      .formula-result { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .chart { height: 290px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Gold Model Dashboard</h1>
      <p>Quarterly view through {{ latest_quarter }}. The upgraded model explains inflation-adjusted gold, then converts the estimate back to nominal dollars.</p>
    </div>
  </header>

  <div class="cards">
    {% for label, value in cards %}
    <div class="card">
      <div class="label">{{ label }}</div>
      <div class="value">{{ value }}</div>
    </div>
    {% endfor %}
  </div>
  {% if nowcast_note %}
  <div class="note">{{ nowcast_note }}</div>
  <div class="subcards">
    {% for label, value in complete_cards %}
    <div class="subcard">
      <div class="label">{{ label }}</div>
      <div class="value">{{ value }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <div class="model-structure">
    <section>
      <h2>Model Calculation Structure</h2>
      <p>The upgraded model first removes inflation, explains real gold with reserve and macro drivers, then converts the fitted value back to nominal dollars.</p>
      <div class="flow">
        <div class="flow-row">
          <div class="flow-box"><strong>1. Inputs</strong><span>Gold, CPI, central-bank buying, US debt/GDP, consumer sentiment, broad dollar</span></div>
          <div class="flow-box"><strong>2. Deflate</strong><span>Turn nominal gold into real gold using CPI</span></div>
          <div class="flow-box"><strong>3. Fit</strong><span>Estimate the real-gold center from standardized drivers</span></div>
          <div class="flow-box"><strong>4. Reflate</strong><span>Convert fitted real gold back into nominal dollars</span></div>
        </div>
        <div class="arrow">↓</div>
        <div class="flow-row">
          <div class="flow-box"><strong>Reserve demand</strong><span>Cumulative central-bank gold buying</span></div>
          <div class="flow-box"><strong>Fiscal pressure</strong><span>US debt divided by US GDP</span></div>
          <div class="flow-box"><strong>Risk pressure</strong><span>Weak consumer sentiment</span></div>
          <div class="flow-box"><strong>Dollar pressure</strong><span>Broad dollar index</span></div>
        </div>
      </div>
    </section>

    <section>
      <h2>Key Formulas</h2>
      <p>Z(x) means the variable is standardized against the historical model sample.</p>
      <div class="formula-list">
        <div class="formula">{{ model_formula.realGoldFormula }}</div>
        <div class="formula">{{ model_formula.fitFormula }}</div>
        <div class="formula">{{ model_formula.nominalFormula }}</div>
        <div class="formula">{{ model_formula.premiumFormula }}</div>
      </div>
      <div class="formula-result">
        <div><div class="label">Latest model price</div><div class="value">{{ model_formula.latestFit }}</div></div>
        <div><div class="label">Latest spot</div><div class="value">{{ model_formula.latestSpot }}</div></div>
        <div><div class="label">Latest premium</div><div class="value">{{ model_formula.latestPremium }}</div></div>
      </div>
    </section>
  </div>

  {% if short_term %}
  <div class="short-grid">
    <section>
      <h2>Short-Term Regression Signal</h2>
      <p>{{ short_term.note }} Latest signal uses data available through {{ short_term.latestWeek }}.</p>
      <div class="mini-cards">
        {% for label, value in short_term.cards %}
        <div class="mini-card">
          <div class="label">{{ label }}</div>
          <div class="value">{{ value }}</div>
        </div>
        {% endfor %}
      </div>
      <div id="short-return-chart" class="chart small"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--green)"></i>Predicted 4-week move</span>
        <span><i class="dot" style="background:var(--gold)"></i>Actual next 4-week move</span>
      </div>
    </section>
    <section>
      <h2>Historical Regression Weights</h2>
      <p>Weights are based on absolute standardized coefficients from the 2024-onward sample. The sign of the coefficient still matters for direction.</p>
      <div class="weights">
        {% for item in short_term.weights %}
        <div class="weight-row">
          <div>{{ item.label }}</div>
          <div class="bar-track"><div class="bar-fill" style="width: {{ '%.1f'|format(item.weight) }}%"></div></div>
          <div>{{ '%.1f'|format(item.weight) }}%</div>
          <div>{{ '%+.2f'|format(item.contribution * 100) }}%</div>
        </div>
        {% endfor %}
      </div>
    </section>
  </div>
  {% endif %}

  <div class="grid">
    <section class="wide">
      <h2>Actual Gold vs Model Fair Value</h2>
      <p>The gap between actual price and the upgraded model is the premium investors are paying above the long-term drivers.</p>
      <div id="price-chart" class="chart"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--gold)"></i>Actual gold</span>
        <span><i class="dot" style="background:var(--green)"></i>Upgraded fit</span>
        <span><i class="dot" style="background:var(--blue)"></i>Legacy fit</span>
      </div>
    </section>

    <section>
      <h2>Premium / Discount</h2>
      <p>Positive bars mean gold trades above the model estimate.</p>
      <div id="gap-chart" class="chart small"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--green)"></i>Upgraded model</span>
        <span><i class="dot" style="background:var(--blue)"></i>Legacy model</span>
      </div>
    </section>

    <section>
      <h2>Real Gold Price</h2>
      <p>Gold price after removing CPI inflation, compared with the upgraded model.</p>
      <div id="real-chart" class="chart small"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--gold)"></i>Real gold</span>
        <span><i class="dot" style="background:var(--green)"></i>Real fitted value</span>
      </div>
    </section>

    <section>
      <h2>Central Bank Accumulation</h2>
      <p>Long-term reserve demand is shown as cumulative net buying.</p>
      <div id="cb-chart" class="chart small"></div>
    </section>

    <section>
      <h2>Macro Drivers</h2>
      <p>Debt pressure, dollar strength, and consumer confidence are normalized for comparison.</p>
      <div id="drivers-chart" class="chart small"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--red)"></i>Debt/GDP</span>
        <span><i class="dot" style="background:var(--blue)"></i>Dollar</span>
        <span><i class="dot" style="background:var(--ink)"></i>Consumer sentiment</span>
      </div>
    </section>
  </div>
</main>
<div class="tooltip" id="tooltip"></div>

<script>
const data = {{ data }};
const colors = {
  gold: "#c6922e",
  blue: "#315f8c",
  green: "#28765a",
  red: "#b8504b",
  ink: "#293241",
  grid: "#d9dddf",
  muted: "#687184"
};

function fmtMoney(v) { return "$" + Math.round(v).toLocaleString(); }
function fmtPct(v) { return (v * 100).toFixed(1) + "%"; }
function fmtNum(v, d=0) { return Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}); }

function extent(seriesList) {
  let values = [];
  seriesList.forEach(s => s.values.forEach(p => {
    const v = Number(p.y);
    if (Number.isFinite(v)) values.push(v);
  }));
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function lineChart(elId, rows, series, opts={}) {
  const el = document.getElementById(elId);
  const width = el.clientWidth || 800;
  const height = el.clientHeight || 300;
  const margin = {top: 18, right: 16, bottom: 34, left: opts.left || 58};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const prepared = series.map(s => ({
    ...s,
    values: rows.map((r, i) => ({x: i, y: r[s.key] === null ? NaN : Number(r[s.key]), row: r}))
  }));
  const [minY, maxY] = opts.yDomain || extent(prepared);
  const x = i => margin.left + (rows.length === 1 ? 0 : i / (rows.length - 1) * innerW);
  const y = v => margin.top + (maxY - v) / (maxY - minY) * innerH;
  const ticks = 5;
  let svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">`;
  for (let t = 0; t <= ticks; t++) {
    const val = minY + (maxY - minY) * t / ticks;
    const yy = y(val);
    svg += `<line class="gridline" x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}"></line>`;
    svg += `<text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end" fill="${colors.muted}" font-size="11">${opts.format ? opts.format(val) : fmtNum(val)}</text>`;
  }
  const labelEvery = Math.max(1, Math.ceil(rows.length / 8));
  rows.forEach((r, i) => {
    if (i % labelEvery === 0 || i === rows.length - 1) {
      const label = r.quarter || r.week || "";
      svg += `<text x="${x(i)}" y="${height - 10}" text-anchor="middle" fill="${colors.muted}" font-size="11">${label}</text>`;
    }
  });
  prepared.forEach(s => {
    let open = false;
    const d = s.values.map((p) => {
      if (!Number.isFinite(p.y)) { open = false; return ""; }
      const cmd = open ? "L" : "M";
      open = true;
      return `${cmd} ${x(p.x).toFixed(2)} ${y(p.y).toFixed(2)}`;
    }).filter(Boolean).join(" ");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.4" vector-effect="non-scaling-stroke"></path>`;
    s.values.forEach((p, i) => {
      if (!Number.isFinite(p.y)) return;
      const label = p.row.quarter || p.row.week || "";
      svg += `<circle cx="${x(i)}" cy="${y(p.y)}" r="3.2" fill="${s.color}" data-tip="${s.name}: ${opts.format ? opts.format(p.y) : fmtNum(p.y)}<br>${label}"></circle>`;
    });
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  attachTooltips(el);
}

function barChart(elId, rows, series, opts={}) {
  const el = document.getElementById(elId);
  const width = el.clientWidth || 800;
  const height = el.clientHeight || 300;
  const margin = {top: 18, right: 16, bottom: 34, left: opts.left || 52};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const all = [];
  rows.forEach(r => series.forEach(s => {
    const v = r[s.key] === null ? NaN : Number(r[s.key]);
    if (Number.isFinite(v)) all.push(v);
  }));
  const minY = Math.min(0, Math.min(...all) * 1.15);
  const maxY = Math.max(0, Math.max(...all) * 1.15);
  const xBand = innerW / rows.length;
  const barW = Math.max(2, xBand / (series.length + 1));
  const y = v => margin.top + (maxY - v) / (maxY - minY) * innerH;
  let svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">`;
  for (let t = 0; t <= 5; t++) {
    const val = minY + (maxY - minY) * t / 5;
    const yy = y(val);
    svg += `<line class="gridline" x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}"></line>`;
    svg += `<text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end" fill="${colors.muted}" font-size="11">${opts.format ? opts.format(val) : fmtNum(val)}</text>`;
  }
  const zeroY = y(0);
  rows.forEach((r, i) => {
    series.forEach((s, j) => {
      const v = Number(r[s.key]);
      if (!Number.isFinite(v)) return;
      const xx = margin.left + i * xBand + (xBand - barW * series.length) / 2 + j * barW;
      const yy = Math.min(y(v), zeroY);
      const hh = Math.max(1, Math.abs(y(v) - zeroY));
      svg += `<rect x="${xx}" y="${yy}" width="${barW * .82}" height="${hh}" fill="${s.color}" data-tip="${s.name}: ${opts.format ? opts.format(v) : fmtNum(v)}<br>${r.quarter}"></rect>`;
    });
  });
  const labelEvery = Math.max(1, Math.ceil(rows.length / 8));
  rows.forEach((r, i) => {
    if (i % labelEvery === 0 || i === rows.length - 1) {
      svg += `<text x="${margin.left + i * xBand + xBand/2}" y="${height - 10}" text-anchor="middle" fill="${colors.muted}" font-size="11">${r.quarter}</text>`;
    }
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  attachTooltips(el);
}

function normalizedRows(rows, keys) {
  const stats = {};
  keys.forEach(k => {
    const values = rows.map(r => Number(r[k]));
    const min = Math.min(...values), max = Math.max(...values);
    stats[k] = {min, max};
  });
  return rows.map(r => {
    const out = {quarter: r.quarter};
    keys.forEach(k => {
      const {min, max} = stats[k];
      out[k] = max === min ? 0 : (Number(r[k]) - min) / (max - min);
    });
    return out;
  });
}

function attachTooltips(el) {
  const tt = document.getElementById("tooltip");
  el.querySelectorAll("[data-tip]").forEach(node => {
    node.addEventListener("mousemove", e => {
      tt.innerHTML = node.getAttribute("data-tip");
      tt.style.display = "block";
      tt.style.left = (e.clientX + 12) + "px";
      tt.style.top = (e.clientY + 12) + "px";
    });
    node.addEventListener("mouseleave", () => { tt.style.display = "none"; });
  });
}

function render() {
  lineChart("price-chart", data.price, [
    {key: "gold_usd", name: "Actual gold", color: colors.gold},
    {key: "upgraded_fitted_gold_usd", name: "Upgraded fit", color: colors.green},
    {key: "fitted_gold_usd", name: "Legacy fit", color: colors.blue}
  ], {format: fmtMoney, left: 66});
  barChart("gap-chart", data.gap, [
    {key: "upgraded_residual_pct", name: "Upgraded premium", color: colors.green},
    {key: "residual_pct_gold_usd", name: "Legacy premium", color: colors.blue}
  ], {format: fmtPct, left: 58});
  lineChart("real-chart", data.real_price, [
    {key: "real_gold_usd", name: "Real gold", color: colors.gold},
    {key: "fitted_real_gold_usd", name: "Real fitted value", color: colors.green}
  ], {format: fmtMoney, left: 66});
  lineChart("cb-chart", data.drivers, [
    {key: "cumulative_central_bank_purchase_tonnes", name: "Cumulative CB buying", color: colors.green}
  ], {format: v => fmtNum(v, 0) + "t", left: 58});
  lineChart("drivers-chart", normalizedRows(data.drivers, ["debt_to_gdp", "broad_dollar_index", "consumer_sentiment"]), [
    {key: "debt_to_gdp", name: "Debt/GDP", color: colors.red},
    {key: "broad_dollar_index", name: "Dollar", color: colors.blue},
    {key: "consumer_sentiment", name: "Consumer sentiment", color: colors.ink}
  ], {format: v => (v * 100).toFixed(0), left: 42, yDomain: [-.05, 1.05]});
  if (data.short) {
    lineChart("short-return-chart", data.short.history, [
      {key: "predicted_4w_return", name: "Predicted 4-week move", color: colors.green},
      {key: "actual_4w_forward_return", name: "Actual next 4-week move", color: colors.gold}
    ], {format: fmtPct, left: 58});
  }
}

render();
window.addEventListener("resize", () => { clearTimeout(window.__resizeTimer); window.__resizeTimer = setTimeout(render, 120); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(build_dashboard())
