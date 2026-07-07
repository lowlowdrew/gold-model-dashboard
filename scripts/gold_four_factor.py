#!/usr/bin/env python3
"""Build public-data versions of CICC-style gold valuation models."""

import io
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"

WGC_PRICE_URL = "https://fsapi.gold.org/api/goldprice/v13/chart/price/?cache09092024"
WGC_SPOT_URL = "https://fsapi.gold.org/api/goldprice/v13/charts/spotprice"
WGC_CB_QUARTERLY_URL = "https://fsapi.gold.org/api/v12/charts/js/gdt-q1-2026-zym8u/3301"
FED_DOLLAR_URL = "https://www.federalreserve.gov/releases/h10/summary/jrxwtfb_nb.htm"
FED_REAL_RATE_MONTHLY_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?filetype=csv&from=&label=include"
    "&lastObs=&layout=seriescolumn&rel=H15&series=d12d70db9d7c91efdbf42f0f2a2bf083&to="
)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?cosd={start}&coed={end}&id={series_id}"
TREASURY_DEBT_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/"
    "debt_to_penny?fields=record_date,tot_pub_debt_out_amt&filter=record_date:gte:2016-01-01"
    "&sort=record_date&page[size]=10000&format=csv"
)
LATEST_REPORTED_CB_MONTHS_2026Q2 = {
    "2026-04": 19.0,
    "2026-05": 41.0,
    "2026-06 China reported": 14.929668864,
}
LATEST_REPORTED_CB_SOURCE = (
    "World Gold Council monthly central-bank updates: April 2026 net buying 19t; "
    "May 2026 net buying 41t. People's Bank of China June 2026 official reserves "
    "rose by 0.48mn oz, equivalent to 14.9t. June is China-only reported data; "
    "full global June and Q2 Gold Demand Trends data are not yet published."
)


def fetch_text(url):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")
    refresh = os.environ.get("GOLD_MODEL_REFRESH", "").lower() in ("1", "true", "yes")
    if cache_path.exists() and not refresh:
        return cache_path.read_text()

    base_cmd = [
        "curl",
        "-L",
        "--connect-timeout",
        "5",
        "--max-time",
        "60",
        "-sS",
    ]
    if "[" in url or "]" in url:
        base_cmd.insert(1, "--globoff")
    variants = [
        base_cmd + [url],
        base_cmd + ["-A", "Mozilla/5.0", url],
        ["curl", "--http1.1"] + base_cmd[1:] + [url],
        ["curl", "--ipv4"] + base_cmd[1:] + [url],
        ["curl", "-k"] + base_cmd[1:] + [url],
    ]
    last_error = None
    for cmd in variants:
        for _ in range(2):
            try:
                completed = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=65,
                )
                text = completed.stdout
                if text.strip():
                    cache_path.write_text(text)
                    return text
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip())
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)
    raise RuntimeError("Could not fetch {}: {}".format(url, last_error))


def fetch_json(url):
    return json.loads(fetch_text(url))


def fetch_fred_series(series_id, value_name):
    try:
        raw = fetch_text(FRED_CSV_URL.format(series_id=series_id, start="2016-01-01", end="2026-06-30"))
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", value_name]
        df["date"] = pd.to_datetime(df["date"])
        df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
        return df.dropna()
    except Exception:
        pass

    frames = []
    for year in range(2016, 2027):
        start = "{}-01-01".format(year)
        end = "{}-12-31".format(year) if year < 2026 else "2026-06-30"
        raw = fetch_text(FRED_CSV_URL.format(series_id=series_id, start=start, end=end))
        frame = pd.read_csv(io.StringIO(raw))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna()


def quarter_label_to_period(label):
    match = re.fullmatch(r"Q([1-4])'(\d{2})", label)
    if not match:
        raise ValueError("Bad quarter label: {}".format(label))
    quarter = int(match.group(1))
    year = 2000 + int(match.group(2))
    return pd.Period(year=year, quarter=quarter, freq="Q")


def fetch_wgc_central_bank_quarterly():
    js = fetch_text(WGC_CB_QUARTERLY_URL)
    data_match = re.search(r'"name":"Net purchase".*?"data":\[(.*?)\]', js, re.S)
    categories_match = re.search(r'"categories":\[(.*?)\]', js, re.S)
    if not data_match or not categories_match:
        raise RuntimeError("Could not parse WGC central bank chart data")

    purchases = json.loads("[" + data_match.group(1) + "]")
    categories = json.loads("[" + categories_match.group(1) + "]")
    df = pd.DataFrame(
        {
            "quarter": [quarter_label_to_period(x) for x in categories],
            "central_bank_net_purchase_tonnes": purchases,
        }
    )
    return df


def fetch_wgc_gold_price_quarterly():
    payload = fetch_json(WGC_PRICE_URL)
    points = payload["chartData"]["USD"]
    df = pd.DataFrame(points, columns=["timestamp_ms", "gold_usd"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df["gold_usd"] = pd.to_numeric(df["gold_usd"], errors="coerce")
    df = df.dropna(subset=["gold_usd"]).set_index("date").sort_index()
    quarterly = df["gold_usd"].resample("Q").mean().dropna().reset_index()
    quarterly["quarter"] = quarterly["date"].dt.to_period("Q")
    return quarterly[["quarter", "gold_usd"]]


def fetch_wgc_spot_price():
    payload = fetch_json(WGC_SPOT_URL)
    chart_data = payload["chartData"]
    usd_mid = chart_data["usd"]["mid"]
    return {
        "date": pd.to_datetime(chart_data["asOfDate"]),
        "gold_usd": float(str(usd_mid["price"]).replace(",", "")),
        "timestamp": chart_data.get("timestamp"),
    }


def fetch_treasury_real_rate():
    raw = fetch_text(FED_REAL_RATE_MONTHLY_URL)
    df = pd.read_csv(io.StringIO(raw), skiprows=5)
    df = df[["Time Period", "RIFLGFCY10_XII_N.M"]].copy()
    df.columns = ["date", "real_rate_10y_tips"]
    df["date"] = pd.to_datetime(df["date"] + "-01")
    df["real_rate_10y_tips"] = pd.to_numeric(df["real_rate_10y_tips"], errors="coerce")
    return df.dropna()


def fetch_fed_dollar_index():
    html = fetch_text(FED_DOLLAR_URL)
    rows = re.findall(
        r">\s*(\d{1,2}-[A-Z]{3}-\d{2})\s*</th>\s*<td[^>]*>\s*([0-9]+(?:\.[0-9]+)?)",
        html,
        re.S,
    )
    if not rows:
        raise RuntimeError("Could not parse Federal Reserve dollar index page")
    df = pd.DataFrame(rows, columns=["date", "broad_dollar_index"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")
    df["broad_dollar_index"] = pd.to_numeric(df["broad_dollar_index"], errors="coerce")
    return df.dropna()


def fetch_treasury_debt():
    df = pd.read_csv(io.StringIO(fetch_text(TREASURY_DEBT_URL)))
    df = df[["record_date", "tot_pub_debt_out_amt"]].copy()
    df.columns = ["date", "total_public_debt_usd"]
    df["date"] = pd.to_datetime(df["date"])
    df["total_public_debt_usd"] = pd.to_numeric(df["total_public_debt_usd"], errors="coerce")
    df["federal_debt_trillion_usd"] = df["total_public_debt_usd"] / 1_000_000_000_000
    return df[["date", "federal_debt_trillion_usd"]].dropna()


def to_quarterly_mean(df, date_col, value_col):
    tmp = df[[date_col, value_col]].dropna().set_index(date_col).sort_index()
    out = tmp[value_col].resample("Q").mean().dropna().reset_index()
    out["quarter"] = out[date_col].dt.to_period("Q")
    return out[["quarter", value_col]]


def to_quarterly_last(df, date_col, value_col):
    tmp = df[[date_col, value_col]].dropna().set_index(date_col).sort_index()
    out = tmp[value_col].resample("Q").last().dropna().reset_index()
    out["quarter"] = out[date_col].dt.to_period("Q")
    return out[["quarter", value_col]]


def add_derived_columns(df, cpi_base):
    df = df.sort_values("quarter").reset_index(drop=True)
    df["debt_to_gdp"] = df["federal_debt_trillion_usd"] / (df["nominal_gdp_billion_usd"] / 1000.0)
    df["weak_consumer_sentiment"] = -df["consumer_sentiment"]
    df["cumulative_central_bank_purchase_tonnes"] = df["central_bank_net_purchase_tonnes"].cumsum()
    df["central_bank_purchase_4q_tonnes"] = df["central_bank_net_purchase_tonnes"].rolling(4, min_periods=1).sum()
    df["real_gold_usd"] = df["gold_usd"] * cpi_base / df["us_cpi"]
    return df


def build_dataset():
    gold = fetch_wgc_gold_price_quarterly()
    central_bank = fetch_wgc_central_bank_quarterly()
    real_rate = to_quarterly_mean(fetch_treasury_real_rate(), "date", "real_rate_10y_tips")
    dollar = to_quarterly_mean(fetch_fed_dollar_index(), "date", "broad_dollar_index")
    debt = to_quarterly_last(fetch_treasury_debt(), "date", "federal_debt_trillion_usd")
    cpi = to_quarterly_mean(fetch_fred_series("CPIAUCSL", "us_cpi"), "date", "us_cpi")
    gdp = to_quarterly_last(fetch_fred_series("GDP", "nominal_gdp_billion_usd"), "date", "nominal_gdp_billion_usd")
    sentiment = to_quarterly_mean(fetch_fred_series("UMCSENT", "consumer_sentiment"), "date", "consumer_sentiment")

    df = gold.merge(central_bank, on="quarter", how="inner")
    for factor in [real_rate, dollar, debt, cpi, gdp, sentiment]:
        df = df.merge(factor, on="quarter", how="inner")
    latest_cpi = df["us_cpi"].iloc[-1]
    return add_derived_columns(df, latest_cpi)


def build_latest_observation(base_df):
    latest_spot = fetch_wgc_spot_price()
    quarter = pd.Period(latest_spot["date"], freq="Q")
    if quarter <= base_df["quarter"].iloc[-1]:
        return None

    gold = fetch_wgc_gold_price_quarterly()
    real_rate = to_quarterly_mean(fetch_treasury_real_rate(), "date", "real_rate_10y_tips")
    dollar = to_quarterly_mean(fetch_fed_dollar_index(), "date", "broad_dollar_index")
    debt = to_quarterly_last(fetch_treasury_debt(), "date", "federal_debt_trillion_usd")
    cpi = to_quarterly_mean(fetch_fred_series("CPIAUCSL", "us_cpi"), "date", "us_cpi")
    gdp = to_quarterly_last(fetch_fred_series("GDP", "nominal_gdp_billion_usd"), "date", "nominal_gdp_billion_usd")
    sentiment = to_quarterly_mean(fetch_fred_series("UMCSENT", "consumer_sentiment"), "date", "consumer_sentiment")

    def value_for(frame, column):
        matched = frame.loc[frame["quarter"] == quarter, column]
        if not matched.empty:
            return float(matched.iloc[-1])
        return float(frame[column].iloc[-1])

    partial_cb = sum(LATEST_REPORTED_CB_MONTHS_2026Q2.values())
    latest_row = pd.DataFrame(
        [
            {
                "quarter": quarter,
                "gold_usd": value_for(gold, "gold_usd"),
                "central_bank_net_purchase_tonnes": partial_cb,
                "real_rate_10y_tips": value_for(real_rate, "real_rate_10y_tips"),
                "broad_dollar_index": value_for(dollar, "broad_dollar_index"),
                "federal_debt_trillion_usd": value_for(debt, "federal_debt_trillion_usd"),
                "us_cpi": value_for(cpi, "us_cpi"),
                "nominal_gdp_billion_usd": value_for(gdp, "nominal_gdp_billion_usd"),
                "consumer_sentiment": value_for(sentiment, "consumer_sentiment"),
            }
        ]
    )
    combined = pd.concat([base_df[latest_row.columns], latest_row], ignore_index=True)
    combined = add_derived_columns(combined, base_df["us_cpi"].iloc[-1])
    latest_full = combined.iloc[[-1]].copy()
    latest_full["period_status"] = "partial_latest_available"
    latest_full["spot_gold_usd"] = latest_spot["gold_usd"]
    latest_full["latest_spot_date"] = latest_spot["date"].date().isoformat()
    latest_full["latest_spot_timestamp"] = latest_spot["timestamp"]
    latest_full["central_bank_source_note"] = LATEST_REPORTED_CB_SOURCE
    return latest_full


def standardize(frame, columns):
    means = frame[columns].mean()
    stds = frame[columns].std(ddof=0)
    return (frame[columns] - means) / stds, means, stds


def fit_log_model(df, target, features):
    x_std, means, stds = standardize(df, features)
    x = np.column_stack([np.ones(len(x_std)), x_std.values])
    y = np.log(df[target].values)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted_log = x.dot(beta)
    fitted = np.exp(fitted_log)
    residual = df[target].values - fitted
    ss_res = float(np.sum((y - fitted_log) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else math.nan

    result = df.copy()
    result["fitted_{}".format(target)] = fitted
    result["residual_{}".format(target)] = residual
    result["residual_pct_{}".format(target)] = residual / fitted
    return {
        "target": target,
        "features": features,
        "beta": beta,
        "means": means,
        "stds": stds,
        "r2_log": r2,
        "result": result,
    }


def score_log_model(fit, df):
    features = fit["features"]
    x_std = (df[features] - fit["means"]) / fit["stds"]
    x = np.column_stack([np.ones(len(x_std)), x_std.values])
    fitted = np.exp(x.dot(fit["beta"]))
    result = df.copy()
    target = fit["target"]
    result["fitted_{}".format(target)] = fitted
    result["residual_{}".format(target)] = result[target] - fitted
    result["residual_pct_{}".format(target)] = result["residual_{}".format(target)] / fitted
    return result


def fit_legacy_nominal_model(df):
    return fit_log_model(
        df,
        "gold_usd",
        [
            "real_rate_10y_tips",
            "broad_dollar_index",
            "central_bank_net_purchase_tonnes",
            "federal_debt_trillion_usd",
        ],
    )


def fit_upgraded_real_gold_model(df):
    fit = fit_log_model(
        df,
        "real_gold_usd",
        [
            "cumulative_central_bank_purchase_tonnes",
            "debt_to_gdp",
            "weak_consumer_sentiment",
            "broad_dollar_index",
        ],
    )
    result = fit["result"].copy()
    cpi_base = df["us_cpi"].iloc[-1]
    result["upgraded_fitted_gold_usd"] = result["fitted_real_gold_usd"] * result["us_cpi"] / cpi_base
    result["upgraded_residual_usd"] = result["gold_usd"] - result["upgraded_fitted_gold_usd"]
    result["upgraded_residual_pct"] = result["upgraded_residual_usd"] / result["upgraded_fitted_gold_usd"]
    fit["result"] = result
    return fit


def score_upgraded_real_gold_model(fit, latest_df, cpi_base):
    result = score_log_model(fit, latest_df)
    result["upgraded_fitted_gold_usd"] = result["fitted_real_gold_usd"] * result["us_cpi"] / cpi_base
    result["upgraded_residual_usd"] = result["gold_usd"] - result["upgraded_fitted_gold_usd"]
    result["upgraded_residual_pct"] = result["upgraded_residual_usd"] / result["upgraded_fitted_gold_usd"]
    result["spot_residual_usd"] = result["spot_gold_usd"] - result["upgraded_fitted_gold_usd"]
    result["spot_residual_pct"] = result["spot_residual_usd"] / result["upgraded_fitted_gold_usd"]
    return result


def append_model_summary(lines, title, fit, actual_col, fitted_col, residual_col, residual_pct_col):
    result = fit["result"]
    latest = result.iloc[-1]
    lines.append(title)
    lines.append("Sample: {} to {}".format(result["quarter"].iloc[0], result["quarter"].iloc[-1]))
    lines.append("Observations: {}".format(len(result)))
    lines.append("R2 on log {}: {:.3f}".format(fit["target"], fit["r2_log"]))
    lines.append("Latest quarter: {}".format(latest["quarter"]))
    lines.append("Actual gold price: ${:,.2f}/oz".format(latest[actual_col]))
    lines.append("Model fitted price: ${:,.2f}/oz".format(latest[fitted_col]))
    lines.append("Residual: ${:,.2f}/oz ({:+.1%})".format(latest[residual_col], latest[residual_pct_col]))
    lines.append("Standardized coefficients:")
    lines.append("Intercept: {:.4f}".format(fit["beta"][0]))
    for name, coef in zip(fit["features"], fit["beta"][1:]):
        lines.append("{}: {:+.4f}".format(name, coef))
    lines.append("")


def write_outputs(df, legacy_fit, upgraded_fit, latest_score=None):
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_out = df.copy()
    df_out["quarter"] = df_out["quarter"].astype(str)
    df_out.to_csv(DATA_DIR / "model_input_quarterly.csv", index=False)

    legacy_result = legacy_fit["result"].copy()
    legacy_result["quarter"] = legacy_result["quarter"].astype(str)
    legacy_result.to_csv(OUTPUT_DIR / "four_factor_fit.csv", index=False)
    legacy_result.to_csv(OUTPUT_DIR / "legacy_four_factor_fit.csv", index=False)

    upgraded_result = upgraded_fit["result"].copy()
    upgraded_result["quarter"] = upgraded_result["quarter"].astype(str)
    upgraded_result.to_csv(OUTPUT_DIR / "upgraded_real_gold_fit.csv", index=False)

    if latest_score is not None:
        latest_out = latest_score.copy()
        latest_out["quarter"] = latest_out["quarter"].astype(str)
        latest_out.to_csv(OUTPUT_DIR / "latest_available_nowcast.csv", index=False)

    lines = []
    append_model_summary(
        lines,
        "Legacy nominal four-factor model",
        legacy_fit,
        "gold_usd",
        "fitted_gold_usd",
        "residual_gold_usd",
        "residual_pct_gold_usd",
    )
    append_model_summary(
        lines,
        "Upgraded real-gold model",
        upgraded_fit,
        "gold_usd",
        "upgraded_fitted_gold_usd",
        "upgraded_residual_usd",
        "upgraded_residual_pct",
    )
    if latest_score is not None:
        latest = latest_score.iloc[-1]
        lines.append("Latest available nowcast")
        lines.append("Quarter: {} ({})".format(latest["quarter"], latest["period_status"]))
        lines.append("Latest spot date: {}".format(latest["latest_spot_date"]))
        lines.append("Spot gold price: ${:,.2f}/oz".format(latest["spot_gold_usd"]))
        lines.append("Quarter-to-date gold average: ${:,.2f}/oz".format(latest["gold_usd"]))
        lines.append("Model fitted price: ${:,.2f}/oz".format(latest["upgraded_fitted_gold_usd"]))
        lines.append("Spot residual: ${:,.2f}/oz ({:+.1%})".format(latest["spot_residual_usd"], latest["spot_residual_pct"]))
        lines.append("Central bank input: {:.1f}t reported since Q1 2026; June is China-only".format(sum(LATEST_REPORTED_CB_MONTHS_2026Q2.values())))
        lines.append("")

    (OUTPUT_DIR / "model_summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    df = build_dataset()
    if len(df) < 20:
        raise RuntimeError("Not enough observations after merging data: {}".format(len(df)))
    legacy_fit = fit_legacy_nominal_model(df)
    upgraded_fit = fit_upgraded_real_gold_model(df)
    latest_df = build_latest_observation(df)
    latest_score = None
    if latest_df is not None:
        latest_score = score_upgraded_real_gold_model(upgraded_fit, latest_df, df["us_cpi"].iloc[-1])
    summary = write_outputs(df, legacy_fit, upgraded_fit, latest_score)
    print(summary)


if __name__ == "__main__":
    main()
