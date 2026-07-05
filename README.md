# Gold Four-Factor Model

This workspace contains public-data reconstructions of CICC-style gold valuation frameworks.

## What CICC Published

Publicly available CICC writeups describe the 2024 version as a four-factor model using:

- US real interest rate
- US dollar
- Central bank gold purchases
- US public debt

The 2025 update says CICC moved toward a longer-horizon real-gold-price framework and reduced reliance on US real rates for long-run valuation.

## Local Reconstruction

The script in `scripts/gold_four_factor.py` builds two quarterly models with public data:

1. Legacy nominal four-factor model: explains nominal USD gold price directly.
2. Upgraded real-gold model: deflates gold with US CPI, explains real gold, then converts the fitted value back to nominal gold. It uses cumulative central bank purchases, US debt-to-GDP, weak consumer sentiment, and the broad dollar index.

Data inputs:

- Gold price: World Gold Council public gold-price endpoint, USD series
- US real interest rate: Federal Reserve H.15 10-year TIPS series
- US dollar: Federal Reserve H.10 broad dollar index
- US public debt: US Treasury Fiscal Data
- US CPI: FRED `CPIAUCSL`
- US nominal GDP: FRED `GDP`
- US consumer sentiment: FRED `UMCSENT`
- Central bank net purchases: World Gold Council chart data from Gold Demand Trends

This is not a licensed Wind/Bloomberg replication. It is a transparent public-data version that lets us test the framework and iterate on data choices.

## Run

```bash
python3 scripts/gold_four_factor.py
```

Outputs are written to:

- `data/model_input_quarterly.csv`
- `outputs/legacy_four_factor_fit.csv`
- `outputs/upgraded_real_gold_fit.csv`
- `outputs/model_summary.txt`

## Dashboard

Generate the static dashboard:

```bash
python3 scripts/build_dashboard.py
```

Open `outputs/gold_model_dashboard.html` in a browser. It visualizes actual gold, upgraded fair value, the premium/discount, real gold, central bank accumulation, and macro drivers.

Refresh public data and rebuild all outputs:

```bash
GOLD_MODEL_REFRESH=1 python3 scripts/gold_four_factor.py
python3 scripts/build_dashboard.py
```

When full quarterly central-bank data is not yet published, `outputs/latest_available_nowcast.csv` adds a clearly marked latest observation using spot gold and the latest reported monthly central-bank updates.
