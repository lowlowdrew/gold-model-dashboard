#!/usr/bin/env python3
"""Build a short-term gold signal from 2024-onward historical regressions."""

import hashlib
import io
import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"

START_DATE = "2024-01-01"
END_DATE = "2026-07-07"
FORECAST_WEEKS = 4

WGC_PRICE_URL = "https://fsapi.gold.org/api/goldprice/v13/chart/main?cache09092024"
WGC_ETF_WEEKLY_FLOW_URL = "https://fsapi.gold.org/api/v11/charts/etfv2/revised/flows-chart2?break-cache=29Jun26"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?cosd={start}&coed={end}&id={series_id}"
CFTC_HISTORY_ZIP_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"


def fetch_bytes(url):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".bin")
    refresh = os.environ.get("GOLD_MODEL_REFRESH", "").lower() in ("1", "true", "yes")
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    base_cmd = [
        "curl",
        "-L",
        "--connect-timeout",
        "5",
        "--max-time",
        "90",
        "-sS",
        url,
    ]
    variants = [
        base_cmd,
        ["curl", "-k"] + base_cmd[1:],
        ["curl", "--http1.1"] + base_cmd[1:],
        ["curl", "--ipv4"] + base_cmd[1:],
    ]
    last_error = None
    for cmd in variants:
        for _ in range(2):
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=95,
                )
                if completed.stdout:
                    cache_path.write_bytes(completed.stdout)
                    return completed.stdout
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.decode("utf-8", errors="ignore").strip())
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)
    raise RuntimeError("Could not fetch {}: {}".format(url, last_error))


def fetch_text(url):
    return fetch_bytes(url).decode("utf-8", errors="replace")


def fetch_json(url):
    return json.loads(fetch_text(url))


def weekly_last(df, value_col):
    tmp = df[["date", value_col]].dropna().set_index("date").sort_index()
    return tmp[value_col].resample("W-FRI").last()


def weekly_sum(df, value_col):
    tmp = df[["date", value_col]].dropna().set_index("date").sort_index()
    return tmp[value_col].resample("W-FRI").sum(min_count=1)


def fetch_wgc_gold_price_weekly():
    payload = fetch_json(WGC_PRICE_URL)
    points = payload["chartData"]["lbma_pm_usd"]
    df = pd.DataFrame(points, columns=["timestamp_ms", "gold_usd"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df["gold_usd"] = pd.to_numeric(df["gold_usd"], errors="coerce")
    df = df[df["date"] >= pd.Timestamp(START_DATE)]
    return weekly_last(df, "gold_usd")


def fetch_fred_weekly(series_id, value_col):
    raw = fetch_text(FRED_CSV_URL.format(series_id=series_id, start=START_DATE, end=END_DATE))
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return weekly_last(df, value_col)


def fetch_wgc_etf_weekly_flows():
    payload = fetch_json(WGC_ETF_WEEKLY_FLOW_URL)
    series = payload["chartData"]["data"]["Weekly"]["series"]["tonnes"]
    frames = []
    for item in series:
        if item.get("name") == "Gold Price (rhs)" or item.get("yAxis") == 1:
            continue
        frame = pd.DataFrame(item["data"], columns=["timestamp_ms", "flow_tonnes"])
        frame["date"] = pd.to_datetime(frame["timestamp_ms"], unit="ms")
        frame["flow_tonnes"] = pd.to_numeric(frame["flow_tonnes"], errors="coerce")
        frames.append(frame[["date", "flow_tonnes"]])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= pd.Timestamp(START_DATE)]
    by_day = df.groupby("date", as_index=False)["flow_tonnes"].sum()
    return weekly_sum(by_day, "flow_tonnes")


def read_cftc_year(year):
    raw_zip = fetch_bytes(CFTC_HISTORY_ZIP_URL.format(year=year))
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fp:
            return pd.read_csv(fp, na_values=["."])


def fetch_cftc_gold_weekly():
    frames = []
    for year in range(2024, 2027):
        frame = read_cftc_year(year)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Market_and_Exchange_Names"].str.contains("GOLD - COMMODITY EXCHANGE INC.", na=False)].copy()
    df["date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"])
    for col in ["M_Money_Positions_Long_All", "M_Money_Positions_Short_All", "Open_Interest_All"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["managed_money_net_pct_oi"] = (
        (df["M_Money_Positions_Long_All"] - df["M_Money_Positions_Short_All"])
        / df["Open_Interest_All"]
    )
    df = df[df["date"] >= pd.Timestamp(START_DATE)]
    return weekly_last(df, "managed_money_net_pct_oi")


def build_weekly_dataset():
    gold = fetch_wgc_gold_price_weekly()
    real_yield = fetch_fred_weekly("DFII10", "real_yield_10y")
    dollar = fetch_fred_weekly("DTWEXBGS", "broad_dollar_index")
    vix = fetch_fred_weekly("VIXCLS", "vix")
    etf_flow = fetch_wgc_etf_weekly_flows()
    futures = fetch_cftc_gold_weekly()

    df = pd.concat(
        [
            gold.rename("gold_usd"),
            real_yield.rename("real_yield_10y"),
            dollar.rename("broad_dollar_index"),
            etf_flow.rename("etf_flow_tonnes"),
            futures.rename("managed_money_net_pct_oi"),
            vix.rename("vix"),
        ],
        axis=1,
    ).sort_index()
    df = df.loc[df.index >= pd.Timestamp(START_DATE)]
    df[["real_yield_10y", "broad_dollar_index", "managed_money_net_pct_oi", "vix"]] = df[
        ["real_yield_10y", "broad_dollar_index", "managed_money_net_pct_oi", "vix"]
    ].ffill(limit=2)

    df["target_4w_forward_return"] = np.log(df["gold_usd"].shift(-FORECAST_WEEKS) / df["gold_usd"])
    df["real_yield_pressure"] = -(df["real_yield_10y"] - df["real_yield_10y"].shift(FORECAST_WEEKS))
    df["dollar_pressure"] = -np.log(df["broad_dollar_index"] / df["broad_dollar_index"].shift(FORECAST_WEEKS))
    df["etf_flow_pressure"] = df["etf_flow_tonnes"].rolling(FORECAST_WEEKS, min_periods=FORECAST_WEEKS).sum()
    df["futures_position_pressure"] = df["managed_money_net_pct_oi"] - df["managed_money_net_pct_oi"].shift(FORECAST_WEEKS)
    df["momentum_pressure"] = np.log(df["gold_usd"] / df["gold_usd"].shift(FORECAST_WEEKS))
    df["market_risk_pressure"] = df["vix"] - df["vix"].shift(FORECAST_WEEKS)
    df.index.name = "week"
    return df


def fit_short_term_model(df):
    features = [
        "real_yield_pressure",
        "dollar_pressure",
        "etf_flow_pressure",
        "futures_position_pressure",
        "momentum_pressure",
        "market_risk_pressure",
    ]
    fit_df = df.dropna(subset=features + ["target_4w_forward_return"]).copy()
    if len(fit_df) < 30:
        raise RuntimeError("Not enough weekly observations for short-term regression: {}".format(len(fit_df)))

    means = fit_df[features].mean()
    stds = fit_df[features].std(ddof=0).replace(0, np.nan)
    z = (fit_df[features] - means) / stds
    x = np.column_stack([np.ones(len(z)), z.values])
    y = fit_df["target_4w_forward_return"].values
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x.dot(beta)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else math.nan

    scored = df.copy()
    scored_z = (scored[features] - means) / stds
    scored["predicted_4w_return"] = beta[0] + scored_z.values.dot(beta[1:])
    scored["fitted_4w_return"] = np.nan
    scored.loc[fit_df.index, "fitted_4w_return"] = fitted
    scored["actual_4w_forward_return"] = scored["target_4w_forward_return"]
    for feature, coef in zip(features, beta[1:]):
        scored["contribution_{}".format(feature)] = scored_z[feature] * coef
        scored["z_{}".format(feature)] = scored_z[feature]

    abs_beta = np.abs(beta[1:])
    weights = abs_beta / abs_beta.sum()
    coefficients = pd.DataFrame(
        {
            "factor": features,
            "coefficient": beta[1:],
            "weight_abs_pct": weights * 100,
            "feature_mean": means.values,
            "feature_std": stds.values,
        }
    )
    coefficients = coefficients.sort_values("weight_abs_pct", ascending=False)
    return {
        "features": features,
        "beta": beta,
        "means": means,
        "stds": stds,
        "r2": r2,
        "fit_df": fit_df,
        "scored": scored,
        "coefficients": coefficients,
    }


def signal_label(value):
    if value >= 0.02:
        return "bullish"
    if value <= -0.02:
        return "bearish"
    return "neutral"


def write_outputs(model):
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    scored = model["scored"].reset_index().copy()
    scored["week"] = scored["week"].dt.date.astype(str)
    scored.to_csv(OUTPUT_DIR / "short_term_regression_weekly.csv", index=False)
    model["coefficients"].to_csv(OUTPUT_DIR / "short_term_regression_coefficients.csv", index=False)

    latest_candidates = model["scored"].dropna(subset=["predicted_4w_return", "gold_usd"])
    latest = latest_candidates.iloc[-1]
    latest_date = latest.name.date().isoformat()
    latest_rows = []
    for row in model["coefficients"].itertuples(index=False):
        factor = row.factor
        latest_rows.append(
            {
                "week": latest_date,
                "factor": factor,
                "coefficient": row.coefficient,
                "weight_abs_pct": row.weight_abs_pct,
                "current_z": latest["z_{}".format(factor)],
                "current_contribution": latest["contribution_{}".format(factor)],
            }
        )
    latest_df = pd.DataFrame(latest_rows)
    latest_df.to_csv(OUTPUT_DIR / "short_term_signal_latest.csv", index=False)

    last_actual = model["fit_df"].index[-1].date().isoformat()
    lines = [
        "Short-term historical regression model",
        "Sample: 2024-01-01 to {}".format(last_actual),
        "Observations: {}".format(len(model["fit_df"])),
        "Target: future 4-week log return in gold",
        "Latest signal week: {}".format(latest_date),
        "Latest gold price in weekly data: ${:,.2f}/oz".format(latest["gold_usd"]),
        "Predicted 4-week return: {:+.2%}".format(latest["predicted_4w_return"]),
        "Signal: {}".format(signal_label(latest["predicted_4w_return"])),
        "R2: {:.3f}".format(model["r2"]),
        "Intercept: {:+.4f}".format(model["beta"][0]),
        "",
        "Weights by absolute standardized coefficient:",
    ]
    for row in model["coefficients"].itertuples(index=False):
        lines.append("{}: {:+.4f}; weight {:.1f}%".format(row.factor, row.coefficient, row.weight_abs_pct))
    lines.extend(
        [
            "",
            "Data sources: World Gold Council daily gold price and ETF weekly flows; FRED 10-year TIPS real yield, broad dollar index, and VIX; CFTC Disaggregated Commitments of Traders futures-only reports for COMEX gold.",
            "Risk factor note: market_risk_pressure uses VIX as a tradable market-risk proxy, not a pure geopolitical-war index.",
        ]
    )
    (OUTPUT_DIR / "short_term_model_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    df = build_weekly_dataset()
    model = fit_short_term_model(df)
    write_outputs(model)


if __name__ == "__main__":
    main()
