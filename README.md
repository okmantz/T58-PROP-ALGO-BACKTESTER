# T58 Trading — Prop Algo Backtester (MVP)

Answers one question:

> **"If I trade this strategy under these prop-firm rules, what is the probability that I pass the evaluation and reach my first payout?"**

This is a prop-firm-first backtester, not a traditional long-term investment backtester. The core workflow:

```
Market Data + Strategy + Risk + Prop Rules
        -> Historical Backtest
        -> Prop-Firm Simulation
        -> Monte Carlo Simulation (thousands of simulated accounts)
        -> Probability of Success
        -> Comprehensive Report
```

Three ways to run it: a **Windows desktop app (.exe)**, a **local Python app** (any OS), or a **mobile-friendly web app** you open in a phone browser.

## 1. Windows `.exe`

Every push to `main` (and every `vX.Y.Z` tag) triggers `.github/workflows/build-exe.yml`,
which builds a single-file Windows executable with PyInstaller and uploads it
as a workflow artifact — no local Python install needed to get the `.exe`:

1. Push this repo to GitHub.
2. Go to **Actions → build-exe → (latest run)**.
3. Download the **T58-Prop-Algo-Backtester-windows-exe** artifact — it contains `T58-Prop-Algo-Backtester.exe`.
4. Tagging a release (`git tag v0.1.0 && git push --tags`) also attaches the `.exe` directly to a GitHub Release.

To build it locally on Windows instead:

```bat
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --windowed --name T58-Prop-Algo-Backtester ^
  --add-data "data/examples;data/examples" app\main.py
```

The `.exe` launches the same Tkinter desktop GUI described below.

## 2. Local Python app (any OS)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.main                 # desktop GUI
python -m app.main --cli --csv data/examples/EURUSD_5M_sample.csv --sims 10000   # headless
```

CLI output is written to `reports/report.{json,html}` plus `report_summary.csv`
and `report_trades.csv`. Open `report.html` in a browser — it's self-contained
and print-to-PDF friendly.

## 3. Mobile app (web / installable PWA)

Tkinter (the desktop GUI toolkit) can't run on a phone, so mobile access is
provided as a lightweight **Flask web app that reuses the exact same
engine** — no logic is duplicated between the desktop and mobile versions.

```bash
pip install -r requirements.txt
python -m app.web.server
```

This serves on `http://0.0.0.0:5000`. On your phone, connect to the **same
Wi-Fi** as the computer running the server and open
`http://<your-computer's-LAN-IP>:5000` (find your LAN IP with `ipconfig`
on Windows or `ifconfig`/`ip addr` on Mac/Linux). From there:

- The page is mobile-responsive with the full 5-step workflow (upload CSV,
  pick a strategy, set prop rules/risk, run).
- Tap **Share → Add to Home Screen** (iOS Safari) or the browser's **Install
  app** prompt (Android Chrome) to install it as a standalone PWA with its
  own icon (`app/web/static/manifest.json` + `sw.js`) — it opens without
  browser chrome, like a native app.
- To access it from outside your home Wi-Fi (not just locally), deploy
  `app/web/server.py` to any small host that runs a Python/Flask app
  (Render, Railway, Fly.io, a VPS, etc.) and open that host's URL on your
  phone instead. Running the Python server process directly *on* the phone
  itself is out of scope for this MVP.

## Workflow (matches the product spec 1:1)

1. **Upload Market Data** — CSV import with auto column-mapping, timestamp/OHLC
   validation, duplicate & gap detection (`app/data/importer.py`). A sample
   dataset is included at `data/examples/EURUSD_5M_sample.csv`. Historical
   backtesting datasets for instruments such as XAUUSD, EURUSD, GBPUSD, S&P500,
   NASDAQ, etc. are also included (`data/raw`).
3. **Import/Create a Strategy** — four adapters, all reduced to the same
   standardized `-1/0/1` signal series before hitting the backtest engine
   (`app/strategy/`):
   - **Manual Builder** — entry/exit rules, SL/TP, built-in SMA/EMA/WMA/RSI
     indicators, expressions evaluated safely via `pandas.eval` (no arbitrary
     code execution).
   - **Python** — upload/paste a `.py` file exposing `generate_signals(df)`.
   - **PineScript** (`app/strategy/pinescript.py`) and **MQL5**
     (`app/strategy/mql5.py`) — real parsers supporting a common subset of
     each language (see below), not full language implementations.
     Anything outside the supported subset raises a clear, specific
     `StrategyError` instead of silently producing an inaccurate backtest.
4. **Enter Prop-Firm Rules** — account size, eval profit target, daily loss
   limit, max drawdown (trailing or static), consistency rule, minimum
   trading days, payout threshold/cap/frequency, required buffer, max
   position size (`app/prop/simulator.py::PropRules`).
5. **Configure Risk & Execution** — fixed-$ or %-of-equity risk per trade,
   max trades/day, commission, slippage, spread, pip size
   (`app/backtest/risk.py::RiskConfig`).
6. **Backtest -> Prop Simulation -> Monte Carlo -> Report** — one click in
   the GUI/web app's run step, or the `--cli` flag.

### PineScript support (subset)

Supported: `open/high/low/close/hl2/hlc3/ohlc4`, `input.int`/`input.float`,
`ta.sma`/`ta.ema`/`ta.wma`/`ta.rsi`, `ta.crossover`/`ta.crossunder`, boolean
rule variables (`and`/`or`/comparisons`), `strategy.entry(..., when=...)` and
`strategy.close(..., when=...)` either inline or inside an `if` block, and
`// T58_SL_PIPS=20` / `// T58_TP_PIPS=40` directive comments for stop-loss/
take-profit (Pine's own `strategy.exit()` uses absolute price offsets, which
aren't a portable "pips" concept across instruments).
Not supported: custom functions, arrays/matrices, `security()`/multi-timeframe
requests, plotting/alerts, and any `ta.*` function beyond the list above.

### MQL5 support (subset)

Supported: direct-value `iMA(...)` (`MODE_SMA`/`MODE_EMA`/`MODE_LWMA`) and
`iRSI(...)` calls, C-style boolean conditions (`&& || ! > < >= <= == !=`),
`if (cond) { ... }` in both Allman and K&R brace styles plus single-statement
`if (cond) stmt;`, `trade.Buy`/`trade.Sell`/`OrderSend(..., ORDER_TYPE_BUY/SELL
or OP_BUY/OP_SELL, ...)` for entries, `trade.PositionClose`/`OrderClose` for
exits, and the same `// T58_SL_PIPS=` / `// T58_TP_PIPS=` directive comments.
Not supported: `CopyBuffer()`-based indicator handles, custom indicators,
arrays/structs, multi-symbol/multi-timeframe logic, and trailing stops.

## Engines

- **Backtest engine** (`app/backtest/`): bar-by-bar execution with
  intrabar stop-loss/take-profit checks, producing a trade list, equity
  curve, and the full statistics set from the spec (returns, win/loss,
  risk, strategy-quality, risk-adjusted ratios).
- **Prop-firm simulator** (`app/prop/simulator.py`): walks a chronological
  trade P&L sequence through the configured rules, determining pass/fail,
  days to pass, payout events, and failure cause. This exact function is
  reused for both the single historical run and every Monte Carlo
  iteration, so results are directly comparable.
- **Monte Carlo engine** (`app/monte_carlo/engine.py`): resamples the
  historical trade sequence (bootstrap / shuffle / block-bootstrap for
  loss-streak stress, plus optional slippage stress) thousands of times and
  re-runs the prop simulator on each, producing the probability
  distributions that are the primary feature of the product — pass
  probability, first-payout probability, failure-before-payout, speed
  (days to pass/payout), financial outcome (expected/median payout), and
  risk (drawdown percentiles, risk of ruin, losing streaks).
- **Report generator** (`app/reports/generator.py`): combines everything
  into one report, exported as JSON, a flattened summary CSV, a trades CSV,
  and a self-contained HTML report. HTML was chosen over a PDF library
  dependency for the MVP — any browser can print it to PDF with zero extra
  install burden; a dedicated PDF export can be added later without
  changing the report data model.

## MVP scope decisions

- PineScript/MQL5 support a real, tested *subset* of each language (see
  above) rather than a full parser/runtime for either — both are large
  languages, and reproducing them completely is out of scope for an MVP.
  Anything unsupported fails loudly and clearly instead of producing a
  silently inaccurate backtest.
- Report export is JSON + CSV + HTML instead of JSON + CSV + PDF, to avoid a
  heavy PDF rendering dependency in v1.
- The desktop GUI is built with Tkinter (Python's standard library) so it
  has zero extra GUI-framework install burden and packages into a Windows
  `.exe` with PyInstaller without any application code changes.
- Mobile access is a web app (Flask + installable PWA) rather than a native
  iOS/Android build — this reuses the engine with zero duplication and
  needs no App Store/Play Store submission; it does need the Flask server
  running somewhere reachable (your own machine on Wi-Fi, or any small
  cloud host).
- One open position at a time (consistent with the standardized long/flat/
  short signal model); no partial fills or multi-leg positions in v1.

## Project layout

```
T58-Prop-Algo-Backtester/
├── app/
│   ├── main.py                # entry point (GUI, or --cli headless run)
│   ├── ui/main_window.py      # Tkinter desktop GUI (step wizard)
│   ├── web/                   # Flask mobile/web app (same engine, new front end)
│   │   ├── server.py
│   │   ├── templates/index.html
│   │   └── static/             # manifest.json, service worker, icons
│   ├── data/importer.py       # CSV import + validation
│   ├── strategy/               # manual / python / pinescript / mql5 adapters
│   │   ├── indicators.py       # shared SMA/EMA/WMA/RSI/crossover math
│   │   ├── expr.py             # shared safe boolean-expression evaluator
│   │   ├── manual.py / python.py / pinescript.py / mql5.py
│   ├── backtest/                # execution engine, risk sizing, statistics
│   ├── prop/simulator.py        # prop-firm rules + account simulator
│   ├── monte_carlo/engine.py
│   └── reports/generator.py     # JSON / CSV / HTML report export
├── data/examples/                # sample OHLCV dataset for immediate testing
├── data/raw/                     # dataset for common forex pairs (1min, 5min, 15min, 1hr, 4hr, and daily timeframes)
├── tests/                        # pytest unit tests for every engine
└── .github/workflows/
    ├── build.yml                 # runs pytest on push/PR
    └── build-exe.yml             # builds & uploads the Windows .exe
```

## Tests

```bash
pytest -q
```

## Disclaimer

Simulated results are estimates derived from historical data and
resampling. Past performance and simulated outcomes do not guarantee future
results.
