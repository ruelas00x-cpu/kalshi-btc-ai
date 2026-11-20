# Kalshi BTC Intelligence v4 — Research Training / iPhone Dashboard

**Purpose:** Run the previous v3.7 directional monitor unchanged while collecting one-minute historical feature snapshots and **official Kalshi yes/no settlement results**. Train a candidate model only in observation mode. **It does not make orders, and the trained model does not influence the UP/DOWN/PASS decision.**

## What is new

- Background monitor records completed BTC-USD 1-minute candle features for the current Kalshi BTC 15-minute window, even when your iPhone is closed (as long as Render is running).
- Preserves timestamped range position, EMA spread, momentum, RSI, price distance from Kalshi strike, quotes and available Coinbase trade-side volume. Missing reference/quotes/data leads to **no training sample**, not invented data.
- Background settlement matching reads Kalshi market detail. Records **only** an official settled `yes` or `no` result (void, unsettled, and ambiguous markets excluded); with archive fallback.
- **Research candidate model**: after at least 250 eligible labeled markets, daily retraining of logistic regression with one minute-3-like sample per contract and a chronological 20% / minimum 50-market holdout. Shows Brier score against a historical-base-rate baseline. This is an exploration metric, not proof of future accuracy.
- Persistent Render disk keeps collected data across restarts and deploys. **Disk storage and a continuously running paid instance may incur monthly charges**; review Render pricing before confirming.
- Phone view includes a research training panel: observed markets, official settlements matched, usable labels and experimental model metrics.

## iPhone setup (requires GitHub + Render + TradingView)

1. Unzip and upload the **contents** of this folder to a new **private GitHub repository** (not the ZIP itself). Keep secrets and your database out of Git. The included `.gitignore` excludes `data/`, SQLite, model files, and environment files.
2. In [Render](https://dashboard.render.com/), choose **New → Blueprint** and connect the repository. Review the `render.yaml` paid Starter service plus **1 GB persistent disk** and the latest fees; approve deployment. Do not attempt to run multiple instances on the same SQLite disk.
3. In Render, set/review `DASHBOARD_PASSWORD` and `WEBHOOK_SECRET` (separate long random values). The Blueprint's randomly generated secret values can be read from Render's service settings. They must match the TradingView indicator input for the webhook. **Never put login or webhook secrets in your GitHub repository.**
4. On a desktop or TradingView's full Pine editor, load `tradingview_btc_v37.pine` on a **BTCUSD 1-minute chart**; configure its webhook secret, and create an **Any alert() function call** alert to `https://YOUR-RENDER-URL.onrender.com/webhook`. Enable TradingView account two-factor authentication for outgoing webhooks.
5. In Safari on iPhone open your deployed HTTPS site, log in with user `btc` and your dashboard password, then **Share → Add to Home Screen**. Data collection runs on the server; you can close Safari.
6. Watch **Automatic learning · Research mode**. Labels must match official Kalshi settlement; incomplete feed days reduce coverage. The trained model remains a **candidate** and does not make the trading call.

## Important guardrails

- **No training occurs before deployment and data accumulation.** Neither the app nor this release is hosted by ChatGPT. There is no historical dataset bundled or fabricated win-rate statistic.
- The model uses **Coinbase spot** as a proxy, *not* Kalshi's official settlement index. It learns against official market results; any feed divergence should be separately studied.
- Kalshi's `yes` / `no` is used as the settlement label; verify the market's YES rule maps to UP for the precise series before relying on any result. Missing/void labels are discarded. Historical archive endpoints or API field names could change.
- No guaranteed prediction accuracy. 250 markets is a **minimum experiment size**, not enough to certify profitability. A 20%-holdout used repeatedly for model development stops being a truly untouched test set. Future versions should use walk-forward evaluation, calibration, quote-fill simulation and a final untouched test period before enabling any trained prediction.
- Prices, spreads, order delays, transaction fees and Kalshi liquidity make **directional accuracy distinct from profitability**. Never interpret a positive Brier score comparison as evidence of positive expected value.
- Do not expose the dashboard to the public; it uses HTTP Basic Auth over HTTPS. This is a prototype, not hardened production infrastructure. Back up the mounted data disk periodically.

## Local tests (no network required)

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_intelligence test_training
```

Run without a webhook/market connection to inspect the protected UI locally:

```sh
export DASHBOARD_PASSWORD='a-long-local-password'
export WEBHOOK_SECRET='a-different-long-random-value'
export ENABLE_RESEARCH_POLLER=false
export ENABLE_COINBASE_TRADE_FLOW=false
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Navigate to http://127.0.0.1:8000. Username: `btc`.

To manually rerun candidate training after labels have accumulated: `python training.py`. Data is stored under `DATA_DIR` (default `./data`). The model is persisted only for offline review; it cannot place trades and is not called by `server.py` for any decision.
