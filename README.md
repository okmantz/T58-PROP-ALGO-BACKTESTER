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
  --paths . --add-data "data/examples;data/examples" run_app.py
```

**Important:** the PyInstaller entry point is `run_app.py`, at the repo root
— **not** `app/main.py`. `run_app.py` exists specifically so PyInstaller's
import analysis can resolve the `app` package correctly; pointing it at
`app/main.py` directly (a script that lives *inside* the `app` package)
produces a broken `.exe` that crashes on launch with
`ModuleNotFoundError: No module named 'app.ui.main_window'`. If you ever
rebuild the workflow or the local build command by hand, keep the entry
point as `run_app.py`.

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
(charts included, see below) and print-to-PDF friendly.

## 3. Mobile app (web / installable PWA)

Tkinter (the desktop GUI toolkit) can't run on a phone, so mobile access is
provided as a lightweight **Flask web app that reuses the exact same
engine** — no logic is duplicated between the desktop and mobile versions.

### Easiest: download `T58-Web-App.exe` (no Python install, no terminal)

Same idea as the desktop `.exe`: grab `T58-Web-App-Windows.zip` from
[GitHub Releases](../../releases) (built by `.github/workflows/build-web-exe.yml`),
extract it, and double-click `T58-Web-App.exe`. It finds your PC's Wi-Fi
address for you and pops up a **QR code** — scan it with your phone's
camera (same Wi-Fi network) to open the backtester, then use your
browser's **"Add to Home Screen"** to get a real app icon. Full
step-by-step with screenshots-in-words: see
[`HOW_TO_OPEN_ON_YOUR_PHONE.md`](HOW_TO_OPEN_ON_YOUR_PHONE.md).

Your phone is a remote screen for the app running on your PC — no
hosting account, no Play Store, no Termux — but your PC does need to
stay on while you use it from your phone.

### Alternative: run it from source

```bash
pip install -r requirements.txt
python run_web.py
```

This does the same thing as the exe (prints your LAN address, opens a
QR code, serves on `http://0.0.0.0:5000`). You can also run the plainer
`python -m app.web.server` if you'd rather find your LAN IP manually
(`ipconfig` on Windows, `ifconfig`/`ip addr` on Mac/Linux).

From your phone's browser:

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

   The dataset picker supports selecting **more than one file at once** for
   multi-timeframe analysis — e.g. select a 60-minute file for bias, a
   15-minute file for zone, and a 5-minute file for entry, all in the same
   run (Ctrl/Cmd-click or Shift-click to multi-select). The finest
   (smallest-interval) file selected becomes the base/entry timeframe; every
   coarser file is merged onto it (`app/data/multi_timeframe.py`, as-of /
   backward merge — no lookahead) as `tfNN_open/high/low/close/volume`
   columns, e.g. `tf60_close`, `tf15_high`. Those columns are directly usable
   as a condition source in the strategy builder below.

2. **Import/Create a Strategy** — four adapters, all reduced to the same
   standardized `-1/0/1` signal series before hitting the backtest engine
   (`app/strategy/`):

   - **Manual Builder** — a full, no-code visual strategy builder
     (`app/ui/condition_builder.py` + `app/strategy/manual.py`):
     - **Strategy information**: name, description, author, version,
       instrument, timeframe, trading session (start/end), and direction
       (Long / Short / Both).
     - **Entry conditions**: build any number of rules, chained with AND/OR,
       from Price, EMA, SMA, WMA, VWAP, RSI, MACD (line/signal/histogram),
       ATR, Bollinger Bands (upper/mid/lower), Highest High, Lowest Low,
       Volume, Average Volume, Candle Direction, Candle Range, Percentage
       Change, Cross Above/Below, and the standard comparison operators
       (Greater/Less Than, Equal To, etc.) — e.g. `Close > EMA(50) AND
       RSI(14) > 55`. Every field not relevant to the chosen source (a
       period, a price column, a direction) hides itself automatically, so
       nothing has to be filled in unless it's actually needed.
     - **Advanced / market-structure conditions**: Swing High, Swing Low,
       Liquidity Sweep, Break of Structure, Change of Character, Fair Value
       Gap, Order Block, Session High/Low, Previous Day High/Low/Close,
       Opening Range High/Low, ATR Regime, and Volatility Regime. These are
       true/false conditions — no operator or comparison value is shown for
       them, since there's nothing to compare.
     - **Exit conditions**: Take Profit and Stop Loss (fixed pips or an ATR
       multiple), a fully dynamic ATR-based Trailing Stop, Break-Even (move
       the stop to entry once profit reaches a configurable multiple of
       initial risk, e.g. "+1R"), a Time-Based Exit, Maximum Bars in Trade,
       an Opposite-Signal-Exit toggle, and Indicator Exit conditions (built
       the same way as entry conditions).
   - **Python** — upload/paste a `.py` file exposing `generate_signals(df)`.
   - **PineScript** (`app/strategy/pinescript.py`) and **MQL5**
     (`app/strategy/mql5.py`) — real parsers supporting a common subset of
     each language (see below), not full language implementations.
     Anything outside the supported subset raises a clear, specific
     `StrategyError` instead of silently producing an inaccurate backtest.

3. **Enter Prop-Firm Rules** — account size, eval profit target, daily loss
   limit, max drawdown (trailing or static), consistency rule, minimum
   trading days, payout threshold/cap/frequency, required buffer, max
   position size (`app/prop/simulator.py::PropRules`).

4. **Configure Risk & Execution** — fixed-$ or %-of-equity risk per trade,
   max trades/day, commission, slippage, spread, pip size
   (`app/backtest/risk.py::RiskConfig`).

5. **Backtest -> Prop Simulation -> Monte Carlo -> Report** — one click in
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
(The Manual Builder's own trailing stop/break-even support, described above,
is not subject to this limitation.)

## Engines

- **Backtest engine** (`app/backtest/`): bar-by-bar execution with
  intrabar stop-loss/take-profit checks — including ATR-based dynamic
  stop/target distances, a ratcheting ATR-based trailing stop, and
  break-even stop management — producing a trade list, equity curve, and
  the full statistics set from the spec (returns, win/loss, risk,
  strategy-quality, risk-adjusted ratios).
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
- **Report generator** (`app/reports/generator.py` + `app/reports/charts.py`):
  combines everything into one report, exported as JSON, a flattened
  summary CSV, a trades CSV, and a self-contained HTML report. The HTML
  report includes inline SVG charts — no extra plotting dependency, no
  external image files — covering the historical equity curve and, for the
  Monte Carlo results, a histogram of simulated account returns and a
  histogram of simulated max drawdown, each with median/P95 markers. HTML
  was chosen over a PDF library dependency for the MVP — any browser can
  print it to PDF with zero extra install burden; a dedicated PDF export
  can be added later without changing the report data model.

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
  `.exe` with PyInstaller without any application code changes. The
  PyInstaller entry point is the repo-root `run_app.py`, not `app/main.py`
  (see the `.exe` section above for why).
- Mobile access is a web app (Flask + installable PWA) rather than a native
  iOS/Android build — this reuses the engine with zero duplication and
  needs no App Store/Play Store submission; it does need the Flask server
  running somewhere reachable (your own machine on Wi-Fi, or any small
  cloud host).
- One open position at a time (consistent with the standardized long/flat/
  short signal model); no partial fills or multi-leg positions in v1.
- Multi-timeframe analysis is implemented as an as-of merge onto the finest
  selected timeframe (see step 1 above) rather than running fully separate
  per-timeframe backtests — this keeps every strategy source (Manual,
  Python, PineScript, MQL5) working against one dataframe unchanged.

## Project layout

```
T58-Prop-Algo-Backtester/
├── run_app.py                  # PyInstaller entry point (repo root — see .exe section)
├── app/
│   ├── main.py                # entry point (GUI, or --cli headless run)
│   ├── ui/
│   │   ├── main_window.py      # Tkinter desktop GUI (step wizard)
│   │   └── condition_builder.py  # visual condition-row widget used by the Manual Builder
│   ├── web/                   # Flask mobile/web app (same engine, new front end)
│   │   ├── server.py
│   │   ├── templates/index.html
│   │   └── static/             # manifest.json, service worker, icons
│   ├── data/
│   │   ├── importer.py         # CSV import + validation
│   │   ├── storage.py          # persists imported CSVs alongside the app/exe
│   │   └── multi_timeframe.py  # merges multiple timeframes onto the finest one
│   ├── strategy/               # manual / python / pinescript / mql5 adapters
│   │   ├── indicators.py       # shared indicator math (SMA/EMA/WMA/RSI/MACD/ATR/Bollinger/etc.)
│   │   ├── expr.py             # shared safe boolean-expression evaluator
│   │   ├── manual.py           # visual-builder condition + risk-management engine
│   │   ├── python.py / pinescript.py / mql5.py
│   ├── backtest/                # execution engine, risk sizing, statistics
│   ├── prop/simulator.py        # prop-firm rules + account simulator
│   ├── monte_carlo/engine.py
│   └── reports/
│       ├── generator.py         # JSON / CSV / HTML report export
│       └── charts.py            # dependency-free SVG chart generation for the HTML report
├── data/
│   ├── examples/                # sample OHLCV dataset for immediate testing
│   ├── raw/                     # dataset for common forex pairs (1min, 5min, 15min, 1hr, 4hr, and daily timeframes)
├── tests/                        # pytest unit tests for every engine
└── .github/workflows/
    ├── build.yml                 # runs pytest on push/PR
    └── build-exe.yml             # builds & uploads the Windows .exe (entry point: run_app.py)
```

## Tests

```bash
pytest -q
```

## Disclaimer

Simulated results are estimates derived from historical data and
resampling. Past performance and simulated outcomes do not guarantee future
results.
