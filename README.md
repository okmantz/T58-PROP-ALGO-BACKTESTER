# T58 Trading — Quant Algo Backtester

The one-stop shop for taking a trading idea from "here's a script" to
"here's a validated, prop-firm-ready strategy" — without leaving one app.
Import or write a strategy in any of four formats, validate it against
real prop-firm rules, search for a version that actually holds up
out-of-sample, optionally let a local AI suggest parameters to try while
that search runs, and walk away with one report that answers the only
question that actually matters:

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

On top of that core loop sits a full research stack for finding, tuning, and
stress-testing strategies before you ever risk a real evaluation fee:
**Iterative Refinement** (single-strategy GA tuning), the **Search Lab**
(multi-strategy discovery across a 5-stage funnel), the **Validation
Lab** (walk-forward optimization, combinatorial purged cross-validation,
parameter sensitivity, multi-asset portfolios, multi-objective search, and
a walk-forward-aware GA), **Full Pipeline** (one button that runs the
entire stack in order and hands back a single READY/MARGINAL/NOT READY
verdict), and **Evolution Lab** (an unattended generate -> filter ->
validate -> mutate loop that runs for hours on its own, checkpointing its
progress so it can be stopped and resumed) — all described below. An
optional local **AI Assist** can participate in that search too, suggesting
parameters for a local Ollama model to try while Full Pipeline runs.

Three ways to run it: a **Windows desktop app (.exe)**, a **local Python app**
(any OS), or a **mobile-friendly web app** you open in a phone browser.

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
pip install -r config/requirements.txt pyinstaller
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

The `.exe` launches the same Tkinter desktop GUI described below. The
9 example strategies under `strategies/` (Python, PineScript, and MQL5)
ship bundled inside the `.exe` itself and self-seed into the Strategy
Library the first time it runs, so the library isn't empty on a fresh
install — no manual file copying required.

## 2. Local Python app (any OS)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r config/requirements.txt

python -m app.main                 # desktop GUI
python -m app.main --cli --csv data/examples/EURUSD_5M_sample.csv --sims 10000   # headless
```

CLI output is written to `reports/report.{json,html}` plus `report_summary.csv`
and `report_trades.csv`. Open `report.html` in a browser — it's self-contained
(charts included, see below) and print-to-PDF friendly. See **CLI reference**
below for every other headless mode (Iterative Refinement, Search Lab, and
all six Validation Lab features).

## 3. Mobile app (web / installable PWA)

Tkinter (the desktop GUI toolkit) can't run on a phone, so mobile access is
provided as a lightweight **Flask web app that reuses the exact same
engine** — no logic is duplicated between the desktop and mobile versions.
(The Validation Lab tabs described below are currently desktop-only; the
web app covers the core Steps 1-5 workflow.)

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
pip install -r config/requirements.txt
python run_web.py
```

This does the same thing as the exe (prints your LAN address, opens a
QR code, serves on `http://0.0.0.0:5000`). You can also run the plainer
`python -m app.web.server` if you'd rather find your LAN IP manually
(`ipconfig` on Windows, `ifconfig`/`ip addr` on Mac/Linux).

From your phone's browser:

- The page is mobile-responsive with the full 5-step workflow (upload CSV,
  pick a strategy, set prop rules/risk, run), plus Search Lab.
- Tap **Share → Add to Home Screen** (iOS Safari) or the browser's **Install
  app** prompt (Android Chrome) to install it as a standalone PWA with its
  own icon (`app/web/static/manifest.json` + `sw.js`) — it opens without
  browser chrome, like a native app.
- This is designed to run entirely on your own home/office Wi-Fi, for
  free, with nothing to sign up for and nothing to pay for: your PC does
  the actual work and your phone is just a screen for it, the same way
  the `T58-Web-App.exe` flow above works. There's no hosted/cloud version
  of this app and no plan to add one — accessing it from outside your own
  Wi-Fi (e.g. over cellular data) isn't supported.

## Workflow (Steps 1-5 — the core loop)

1. **Upload Market Data** — CSV import with auto column-mapping, timestamp/OHLC
   validation, duplicate & gap detection (`app/data/importer.py`). A sample
   dataset is included at `data/examples/EURUSD_5M_sample.csv`. Historical
   backtesting datasets for instruments such as XAUUSD, EURUSD, GBPUSD, S&P500,
   NASDAQ, etc. are also included (`data/raw`).

   **Supported file types**, dispatched automatically by extension:
   - `.csv` / `.tsv` / `.txt` — delimiter (comma/tab/semicolon/pipe) and
     header/no-header are both auto-detected. Headerless 6-column files are
     assumed to be `timestamp, open, high, low, close, volume` in that order.
   - `.parquet` — read directly (requires the `pyarrow` package, already
     listed in `config/requirements.txt` and bundled into both `.exe` builds).
   - `.zip` / `.7z` archives — opened automatically and whichever member
     inside looks like the actual OHLCV file (`.csv`/`.tsv`/`.txt`/`.parquet`)
     is read, skipping folders and OS junk like `__MACOSX/`/`.DS_Store`.
     `.7z` requires the `py7zr` package (also bundled).
   - **Column names**: a wide alias list maps common vendor/broker column
     names to the standard `timestamp/open/high/low/close/volume` schema —
     e.g. `ts`, `time`, `date`, `datetime`, `Gmt time`, `bar_time` all map to
     `timestamp`; `o/h/l/c/v` and `Open Price`/`Tick Volume`-style names are
     recognized too. If a timestamp column uses a name no alias list
     anticipated, a fallback tries every remaining date/time-looking column
     name and, if more than one candidate remains, actually test-parses a
     sample of each as a date and picks whichever one works. Extra columns
     the schema doesn't use (e.g. a vendor's `symbol` column) are simply
     ignored rather than causing an import error.

   The dataset picker supports selecting **more than one file at once** for
   multi-timeframe analysis — e.g. select a 60-minute file for bias, a
   15-minute file for zone, and a 5-minute file for entry, all in the same
   run (Ctrl/Cmd-click or Shift-click to multi-select). The finest
   (smallest-interval) file selected becomes the base/entry timeframe; every
   coarser file is merged onto it (`app/data/multi_timeframe.py`, as-of /
   backward merge — no lookahead) as `tfNN_open/high/low/close/volume`
   columns, e.g. `tf60_close`, `tf15_high`. Those columns are directly usable
   as a condition source in the strategy builder below.

   **Alpaca API fetch** (`app/data/alpaca_source.py`): an alternative to
   picking local files — pulls bars directly from Alpaca (US equities +
   crypto only; no forex/futures/CFD feed) using saved or freshly entered
   API keys (`app/data/alpaca_credentials.py`), and saves them into
   `data/raw/` so they join the normal dataset list.

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
   - **Python** — upload/paste a `.py` file exposing `generate_signals(df)`
     (see `app/strategy/python.py`'s docstring for the full contract,
     including the `.attrs` mechanism for per-trade dynamic stop/target/
     trailing distances, and the multi-timeframe-bias lookahead trap it
     specifically warns about).
   - **PineScript** (`app/strategy/pinescript.py`) and **MQL5**
     (`app/strategy/mql5.py`) — real parsers supporting a common subset of
     each language (see below), not full language implementations.
     Anything outside the supported subset raises a clear, specific
     `StrategyError` instead of silently producing an inaccurate backtest.

   Every strategy — regardless of source — can be checked for **lookahead
   bias** (`app/strategy/lookahead_check.py`) before you trust its numbers:
   it re-runs the strategy's own signal generation on the data truncated
   right after each of several checkpoints (chosen from where the strategy
   actually fired a trade, not arbitrary evenly-spaced points) and diffs
   the result against the full-data run. Any bar whose signal changes
   depending on data that hadn't happened yet is a confirmed leak, named
   with the exact bar/timestamp it first appears at. The same class of bug
   — a naive higher-timeframe filter that leaks the still-forming current
   HTF bar into every bar that isn't exactly on its boundary — was found
   in a real uploaded strategy and flipped its reported result from
   solidly profitable to a clear loser once fixed; see
   `app/strategy/mtf.py`'s docstring for the exact before/after numbers and
   `app/strategy/lookahead_check.py`'s own docstring for how the detector
   itself works.

   **Strategy Library** (`app/strategy/library.py`): any Python/PineScript/
   MQL5 strategy can be saved *inside* the app's own data folder — the same
   persistent, writable location `app.data.storage` uses for market-data
   CSVs — instead of only ever being pulled from wherever it happens to
   live on a particular computer or phone. Once saved, it shows up in the
   library dropdown/listbox on every future run, with per-strategy status
   tags, backtest/lookahead/search result history, and library-wide export.

3. **Enter Prop-Firm Rules** — account size, eval profit target, daily loss
   limit, max drawdown (trailing or static, intrabar or end-of-day check
   mode), consistency rule, minimum trading days, payout threshold/cap/
   frequency, required buffer, max position size
   (`app/prop/simulator.py::PropRules`).

4. **Configure Risk & Execution** — fixed-$ or %-of-equity risk per trade,
   max trades/day, commission, slippage, spread, pip size
   (`app/backtest/risk.py::RiskConfig`). A **"Detect pip size from data"**
   button suggests a starting value from whatever's loaded in Step 1 —
   leaving pip size at its FX default (0.0001) against a non-FX instrument
   (stocks, indices, crypto, JPY pairs) is the single most common cause of
   a fixed-pips stop translating into a nonsensical position size.

5. **Backtest -> Prop Simulation -> Monte Carlo -> Report** — one click in
   the GUI/web app's run step, or the `--cli` flag.

## Step 6 — Iterative Refinement (optional)

A genetic-algorithm-style parameter search: re-runs the current strategy
many times with mutated numeric parameters on the *same* historical data,
keeps the best-performing configurations each generation (elitism +
tournament selection + random immigrants), and converges toward the
best-scoring configuration it can find, judged by a configurable fitness
metric (composite prop score, eval-pass probability, first-payout
probability, expected payout, net profit, profit factor, or Sharpe —
`app/optimize/refinement.py::FITNESS_METRICS`).

Works across all four strategy sources via a shared gene-discovery layer
(`app/optimize/parameter_space.py`, `app/optimize/code_parameter_space.py`):
Manual Builder numeric fields, every top-level `SCREAMING_SNAKE_CASE`
numeric constant in a Python strategy, every `input.int()`/`input.float()`
in PineScript, and every `iMA()`/`iRSI()` period in MQL5 (plus the
`T58_SL_PIPS`/`T58_TP_PIPS` directives for all three). A strategy with no
such parameters says so clearly rather than running a meaningless search.
Produces its own separate report (`app/reports/refinement_report.py`) and
an "apply best configuration back to the Strategy tab" button — the normal
Run & Report pipeline is completely unaffected unless you explicitly
enable this.

## Step 7 — Search Lab

Discovers and validates *many* candidate strategies in one run, instead of
tuning one you already picked. A 5-stage funnel
(`app/search/batch_runner.py`):

1. **Generate** a candidate pool — either a combinatorial grid across one
   of the built-in named-hypothesis families (trend/breakout, multi-
   timeframe pullback, mean-reversion band, volatility breakout, session/
   time-of-day effect, volume imbalance, statistical pairs/relative-value —
   `app/search/strategy_space.py::FAMILIES`), or a grid over an uploaded
   strategy file's own tunable parameters.
2. **Stage 1 — cheap filter**: a fast backtest-only pass over every
   candidate, gated by minimum trade count and profit factor.
3. **Stage 2 — GA refinement**: the same Iterative Refinement engine from
   Step 6, applied to each Stage-1 survivor's own tunable parameters —
   including its cost-stress penalty (see Step 14 below), so the GA itself
   is biased toward candidates whose edge survives worse execution.
4. **Stage 3 — validation gate**: full Monte Carlo + walk-forward holdout +
   parameter-neighborhood robustness + Deflated Sharpe Ratio
   (`app/search/robustness.py`) — candidates that only look good in-sample
   get filtered out here.
5. **Stage 4/5 — leaderboard + champion promotion**: every surviving
   candidate's results are stored in a queryable SQLite database
   (`app/search/results_db.py`) and ranked; the top candidate can be
   promoted to a full, standalone report exactly like a normal single-
   strategy run.

Completely separate from the normal Run & Report pipeline and from Step 6
— running it doesn't touch either.

## Steps 8-13 — Validation Lab

Six additional statistical-rigor tools, each answering a different
"how much should I actually trust this backtest?" question that a single
in-sample run or a single 80/20 holdout split can't answer on its own.
Each has its own desktop tab (sidebar group below Search Lab) and CLI flag;
all six reuse the same gene-discovery/GA machinery as Iterative Refinement,
so they work across Manual/Python/PineScript/MQL5 strategies consistently.

- **08 — Walk-Forward Optimization** (`app/validation/walk_forward_opt.py`):
  a first-class workflow, not just a holdout check. Splits the data into
  rolling or anchored folds, runs a *fresh* GA search on each fold's train
  window only, applies the winning configuration unchanged to that fold's
  held-out test window, and chains every fold's out-of-sample trades into
  ONE continuous equity curve — the number to trust over a single in-
  sample backtest.
- **09 — CPCV / PBO** (`app/validation/cpcv.py`): Combinatorial Purged
  Cross-Validation stress-tests one strategy across many combinatorial
  train/test partitions of the same data (not just one split); the
  Probability of Backtest Overfitting (Bailey/López de Prado) checks a
  *pool* of candidates and reports the probability that whichever one
  looks best in-sample is, out-of-sample, no better than a coin flip.
- **10 — Parameter Sensitivity** (`app/validation/sensitivity.py`): 1D
  sweeps of every tunable parameter (±X%, with automatic "cliff"
  detection for a knife-edge parameter vs. a real stable plateau), plus an
  optional 2D heatmap for a chosen pair of parameters.
- **11 — Multi-Asset Portfolio** (`app/portfolio/portfolio.py`): runs a
  strategy across several instruments, computes their return correlation
  matrix, re-weights each instrument's risk (correlated legs sized down),
  and merges every leg's trades into one shared account equity curve —
  modeling "one prop account trading several instruments," with an
  explicit, documented set of simplifications (see the module's own
  docstring) rather than pretending to be a full multi-position margin
  engine.
- **12 — Multi-Objective Optimization** (`app/optimize/multi_objective.py`):
  a real NSGA-II implementation (non-dominated sorting + crowding
  distance) producing a genuine Pareto front across several objectives at
  once (e.g. Sharpe, max drawdown, eval-pass probability) instead of
  collapsing them into one weighted score the way Step 6's GA does.
  Picking a final winner from the front is left as a judgment call.
- **13 — Walk-Forward-Aware GA** (`app/optimize/walkforward_ga.py`): the
  same GA operators as Step 6, but every candidate's fitness is scored
  *only* on chained out-of-sample fold data — never the training windows,
  never the full dataset — so the search can't just curve-fit harder. Also
  reports an "overfitting gap" (in-sample fitness vs. chained-OOS fitness
  of the winning genome).

Report generation for all six lives in `app/reports/validation_reports.py`
(JSON + a focused, self-contained HTML page per feature, reusing
`app/reports/charts.py`'s SVG chart helpers — including a 2D heatmap chart
added specifically for Sensitivity). See **CLI reference** below for every
flag; the desktop tabs expose the same functionality with live progress
logs.

**Known limitation, now closed for account-state logic (see Step 14)**:
`generate_signals(df)` (and its Manual/PineScript/MQL5 equivalents) is
still called once, statelessly, over the whole dataset before any P&L
exists — no strategy source can implement its own account-state-dependent
logic directly. What changed: that protection no longer has to be a hard
binary breaker only. `app/backtest/adaptive_risk.py` adds a declarative,
engine-level money-management layer (de-risk after N losses, cut size once
a daily loss threshold is hit, coast once X% of the way to a profit
target) that plugs into `run_backtest()` the same way
`RiskConfig.daily_loss_limit_pct` already did — see Step 14 for the full
rule set and an example.

## Step 14 — Finding an Edge (widened Search Lab, cost-stress fitness, adaptive risk, ensembles)

Steps 6-13 are all about validating a strategy rigorously once you already
have one. This step is aimed one level upstream — at actually finding a
real edge in the first place — across four additions. All four are
available on the desktop GUI (the new "14 ENSEMBLE" sidebar tab, a new
"Adaptive risk" section on Step 4/Risk & Execution, and new cost-stress
controls on Step 6/Refinement and Step 7/Search Lab), the CLI, and
directly via the underlying modules.

- **Wider Search Lab hypothesis space.** Four new named families join the
  original three: **Volatility Breakout** (a Donchian breakout gated by
  ATR expanding vs. its own baseline, not by trend direction), **Session /
  Time-of-Day Effect** (an opening-range breakout confined to a specific
  clock-time window, with a forced flat-by time), **Volume Imbalance**
  (trades a rolling signed-volume-pressure oscillator,
  `app/strategy/indicators.py::volume_delta`), and **Statistical Pairs /
  Relative Value** (mean-reverts the primary instrument against a second,
  merged-in instrument's price ratio z-score — see
  `app/data/pairs.py::merge_pair_series()`; only the primary leg is
  actually traded, since the engine stays single-instrument, so treat this
  as a relative-value entry filter, not a full two-leg pairs trade). No
  GUI/CLI changes were needed for the family dropdown itself — it's
  generated from `app.search.strategy_space.list_families()`, so new
  families just appear. A search over `family="all"` automatically skips
  the pairs family unless you've merged in a second instrument first (a
  "Pair instrument" CSV picker on the Search Lab tab, or `--pair-csv` on
  the CLI).
- **Cost-stress-adjusted GA fitness.** Iterative Refinement, Search Lab's
  Stage 2, and the Walk-Forward-Aware GA all now ALSO re-backtest every
  candidate at spread/slippage/commission multiplied by
  `cost_stress_multiplier` (default 2x) and blend that stressed-cost
  result into the fitness the GA actually selects on
  (`app/optimize/refinement.py::apply_cost_stress_penalty`) — on by
  default, tunable via `cost_stress_penalty_weight` (0 = ignore, 1 = full
  penalty), reported statistics/summaries in every report stay nominal
  (un-stressed); only the scalar the GA breeds toward is adjusted. This is
  distinct from Stage 3's cost-ladder check and the Refinement report's
  own cost-ladder table, which only *report* cost sensitivity after the
  fact — this feeds it back into what gets selected in the first place.
  GUI: a "Cost-stress penalty" section on both the Refinement tab (Step 6)
  and the Search Lab tab's Stage 2 section (Step 7).
- **Declarative adaptive risk layer** (`app/backtest/adaptive_risk.py`):
  engine-level money-management rules — `consecutive_losses`,
  `daily_loss_pct`, `daily_profit_pct`, `progress_to_target_pct` — each
  scaling new-entry position size by a configured multiplier once
  triggered; multiple active rules stack multiplicatively. Passed as an
  optional `AdaptiveRiskConfig` into `run_backtest()`; every `Trade` records
  the multiplier and which rule(s) were active when it opened, so a report
  can show exactly when and why sizing was cut. GUI: a new "Adaptive
  risk" section on the Risk & Execution tab (Step 4) — enable, set a
  profit-target %, and add rules via a small dialog (trigger, threshold,
  multiplier). CLI: `--adaptive-risk-rules
  '{"rules": [{"trigger": "consecutive_losses", "threshold": 2,
  "risk_multiplier": 0.5}], "profit_target_amount_pct": 8.0}'`.
- **Multi-strategy ensembles** (`app/ensemble/ensemble.py`): the mirror
  case of Step 11's multi-asset Portfolio — several *different*,
  weakly-correlated strategies combined on the *same* instrument, instead
  of one strategy across several instruments. Two modes: `run_ensemble_blend`
  (each leg keeps trading independently at a correlation-adjusted risk
  weight — reuses `app.portfolio.portfolio.run_portfolio_backtest`
  unmodified, just pointed at one shared `df`) and `run_ensemble_vote`
  (combines every leg's raw signal into one majority/threshold-vote entry,
  run through the ordinary single-position engine; risk management is
  inherited from the first-listed leg only). GUI: new "14 ENSEMBLE" tab
  (add strategy files, pick Blend/Vote). CLI: `--ensemble
  --ensemble-strategy path1.py --ensemble-strategy path2.pine
  --ensemble-mode blend|vote`.

## Step 15 — Full Pipeline (one button, the whole workflow)

Every other feature above is a separate tool: Run & Report backtests one
fixed configuration, Iterative Refinement tunes it in-sample, the
Walk-Forward-Aware GA tunes it against out-of-sample folds, the Validation
Lab checks robustness after the fact. Getting from "here's a strategy
file" to "here's the best, validated version of it, ready for a prop
firm" means running several of those in the right order and carrying the
winner from one into the next by hand. **Full Pipeline**
(`app/orchestration/full_pipeline.py`) does that hand-off automatically,
in six steps: baseline backtest + lookahead check → walk-forward-aware GA
search (optionally AI-assisted, see below) → re-validated final report →
out-of-sample fold check → holdout check → final report with a plain
READY / MARGINAL / NOT READY verdict and the exact reasons behind it. For
Python/PineScript/MQL5 strategies, the winning source is also saved
straight into the Strategy Library, tagged `validated` by default.

The report it produces carries everything needed to trust (or distrust)
the numbers, front and center rather than buried in a console log:

- **Execution-integrity warnings** — a pip-size/instrument-scale mismatch
  (a strategy's fixed-pips stop translating to a nonsensical fraction of
  the instrument's real price), or trades where the market gapped straight
  past a resting stop — surfaced as a banner at the top of the report
  itself, not just a line in a log that scrolled past.
- **The verdict, in the report** — READY/MARGINAL/NOT READY and the exact
  Monte Carlo thresholds it did or didn't clear, so the saved HTML file
  answers "does this pass" on its own, without needing the live run
  console open.
- **The winning parameters, in the report** — the exact tunable values
  (indicator periods, SL/TP, session hours, etc.) the search settled on,
  in a table right next to the metrics they produced.

Available on the desktop GUI (sidebar: **15 FULL PIPELINE**) and headlessly
via `--full-pipeline` (see **CLI reference** below).

## Step 16 — Forward Test (MT5 Demo)

Deploy any Strategy Library strategy to a free MetaTrader 5 demo account
and watch it trade forward against real broker prices, bar by bar, instead
of a CSV. This is the bridge between "the backtest looks good" and "I'd
trust this with real money": spread, slippage, and fills come from the
actual market instead of a cost model, over however long you let it run.

**Why MT5, not TradingView.** TradingView's webhook alerts — the usual way
to wire a chart strategy to automated execution — require a paid plan.
MT5's free, official `MetaTrader5` Python package talks directly to a
locally-running MT5 terminal at no cost, and virtually every prop firm
offers MT5-based demo/eval/funded accounts. That's the whole reason this
was built on MT5 instead.

**Requirements:** Windows, a running MT5 terminal logged into a demo
account (any MT5 broker's website offers a free demo signup), and
`pip install MetaTrader5` (already conditional in `config/requirements.txt` on
Windows). On any other OS, or without the package, the tab explains this
plainly instead of erroring.

**What it does, and doesn't, do:**

- Reuses the *exact* signal engine (`Strategy.generate(df)`) and *exact*
  position-sizing math (`RiskConfig.position_size(...)`) the backtester
  uses — forward-test behavior is never a second, drifting implementation
  of "what should this strategy do."
- Polls for newly-closed bars only (never a still-forming bar), resolves
  each trade's stop/target with the same precedence the backtest engine
  uses (dynamic distance → fixed pips → 1%-of-price fallback), and sizes
  the position from live account equity.
- Enforces a daily-loss circuit breaker (same `daily_loss_limit_pct`
  semantics as a backtest run) that halts new entries for the rest of the
  calendar day once tripped.
- Reconciles with MT5's actual open positions on every start — an app
  restart mid-trade adopts the real position instead of opening a
  duplicate.
- Logs every trade and event to a local SQLite journal
  (`data/forward_test/forward_test.db`) that survives a restart.
- Flags (doesn't auto-stop on) win-rate drift versus a backtest baseline
  you can optionally enter, once enough forward trades have accumulated.
- Ships a **kill switch** — one button closes every open position on the
  symbol immediately and stops the session.
- **Demo accounts only.** There is no live/funded order path anywhere in
  this module. Wiring it to a funded account is a deliberate, separate,
  later decision — not a checkbox here.

Before deploying anything: see `strategies/SCREENING_RESULTS.md` for an
honest read on which of the bundled library strategies currently show any
real edge (as of writing: none of them do — forward-testing one of them
won't turn it profitable).

Available on the desktop GUI (sidebar: **16 FORWARD TEST**). No CLI
equivalent yet — this is an interactive, long-running session by nature.

## Step 17 — Evolution Lab (unattended, run-for-hours strategy discovery)

Every other feature above evaluates or tunes a strategy someone already
picked. **Evolution Lab** (`app/evolution/`) instead runs the whole
generate → filter → validate → keep-the-winners → mutate loop by itself,
unattended, for as long as you leave it running:

```
RESEARCH (knowledge graph -- informs which families/features get weighted
          into GENERATE, based on what has historically scored well)
    v
GENERATE ~N STRATEGIES   (every family app/search/strategy_space.py knows)
    v
PRE-FILTER + BACKTEST    (one cheap backtest: trades / profit factor / DD)
    v
ROBUSTNESS + OOS + MONTE CARLO + PROP SIMULATION
    v
CPCV / PBO                (real combinatorial-purged CV, top candidates only)
    v
STRESS TEST                (re-run at N-x execution costs)
    v
CLUSTER                    (correlation-dedupe so the top 10 aren't 10
                             near-identical variants of the same winner)
    v
KEEP TOP N -> record to knowledge graph -> MUTATE -> repeat
```

Candidates are ranked by **PROP FITNESS** (`app/evolution/prop_fitness.py`)
— pass probability × payout probability × robustness × OOS consistency,
divided by drawdown, minus penalties for thin trade counts, high parameter
sensitivity, high PBO, in/out-of-sample degradation, profit concentrated in
one lucky trade, and long losing streaks — not raw net profit, so the
leaderboard reflects "would actually survive a funded account," not just
"backtested well once."

**Every candidate tested is logged**, not just the winners:
- The **Tested Strategies** panel lists every candidate the PRE-FILTER
  stage has backtested this run, pass or fail, with the specific reason it
  was rejected (`min_trades`, `profit_factor`, `max_drawdown`,
  `unprofitable`, `no_trades`, or a build/backtest error) if it failed, and
  how far it got (plus its PROP FITNESS score) if it passed. This is a
  durable, on-disk log (`data/evolution/tested_candidates.jsonl`), not just
  console scrollback — click REFRESH any time, including after reopening
  the app.
- If a generation produces **zero** PRE-FILTER survivors, the log shows a
  rejection breakdown (e.g. "min_trades: 40, profit_factor: 12,
  unprofitable: 3") right there instead of a bare "0 survived." If that
  happens **3 generations in a row**, the pre-filter thresholds are
  automatically loosened once (min trades reduced, minimum profit factor
  relaxed, drawdown buffer widened) — the same auto-relax idea Search Lab's
  own Stage 1 already uses — so a run doesn't grind for hours with an
  empty leaderboard and no visible reason why.
- The **knowledge graph** (`data/evolution/knowledge_graph.jsonl`) is an
  append-only log of every candidate's structural feature vector (family,
  session/volatility/trend filters used, indicator mix, direction bias)
  paired with its outcome, across every run ever started. Each
  generation's journal entry queries it for similar past candidates, so
  later generations can say "this mechanism has historically worked 86% of
  the time" rather than judging each generation in isolation.

**Progress survives STOP and restarting the app.** Generation number,
current elites (used to seed next generation's mutated children), the
all-time leaderboard, and the hypothesis journal are all saved to disk
(`data/evolution/checkpoint.json`) after every generation. Clicking START
again — even in a new session — resumes exactly where it left off instead
of starting over from scratch, as long as the same market data is loaded;
loading different data is detected automatically and starts a fresh run
instead of silently mixing incompatible runs. Click **RESET** to discard
the saved checkpoint and tested-candidates log and genuinely start over.

**Confidence rating.** Each generation's HYPOTHESIS journal entry rates its
winner LOW / MEDIUM / HIGH based on whether it's stable under
parameter-neighborhood perturbation *and* how many similar historical
candidates the knowledge graph has seen. Treat LOW-confidence winners
(the vast majority, especially early on) as leads worth tracking, not
strategies worth funding.

**Scope, stated plainly:** candidates are Manual Strategy Builder configs
generated from `app.search.strategy_space`'s families — this does not
mutate uploaded Python/PineScript/MQL5 files. It runs single-process; each
generation is currently slower than Search Lab's own multi-worker Stage
1-3 pipeline, since porting that same `ProcessPoolExecutor` parallelism
into Evolution Lab is the natural next optimization once the loop's shape
is validated in practice.

Available on the desktop GUI (sidebar: **EVOLUTION LAB**). Safe to leave
running for hours while working in other tabs — it runs on a background
thread and checkpoints itself automatically.

## AI Assist (optional, local Ollama)

Full Pipeline's walk-forward-aware GA search can optionally ask a local
[Ollama](https://ollama.com) model for candidate parameter values to try
— once per generation, while the search is actually running, not just a
one-off suggestion at the start. Every suggestion still has to pass
through the exact same backtest → prop-simulation → Monte Carlo pipeline
as any other candidate the GA tries: the model only ever proposes numbers
for a strategy's already-discovered tunable parameters (see the gene
discovery described under Step 6 above) — it never writes or edits
strategy code, and can never displace a genuinely better candidate the GA
already found.

**Setup is deliberately minimal:**

1. Install Ollama and pull a model — free, runs entirely on your own
   machine: **[ollama.com/download](https://ollama.com/download)**, then
   `ollama pull llama3.1` (or any model you prefer) from a terminal.
2. On the Full Pipeline tab, open **AI Assist**, check **Enable AI Assist
   for this run**, and hit **Test Connection** to confirm it's reachable.
3. Run Full Pipeline as normal — nothing else changes.

Off by default, everywhere. Leaving it disabled (or never installing
Ollama at all) runs Full Pipeline exactly as if this feature didn't exist.
An unreachable, slow, or misconfigured Ollama degrades the same way: a
couple of failed attempts and the search quietly continues without it,
logged once, never blocking the run. An optional API key field supports
pointing this at a remote/proxied Ollama endpoint behind auth instead of
a local install, for anyone running it that way.

## Step 18 — AI Research Engine (RAG + Research Agent)

AI Assist's numeric-only parameter suggestions and the Strategy Generator's
one-shot code drafts are single request/response calls. The **AI Research
Agent** (Step 18) is a meaningfully bigger step: a local Ollama model
investigates a strategy across several reasoning steps by calling a fixed
toolbox of read-only analysis actions, each of which runs this app's own
already-validated engine — never a guess, never invented numbers.

```
RESEARCH LIBRARY                    T58 RESEARCH MEMORY
research/ papers, books,            every strategy this app has
your own notes                      ever tested (SQLite + semantic index)
        │                                   │
        └───────────────┬───────────────────┘
                         ↓
              local Ollama embeddings
           (e.g. `ollama pull nomic-embed-text`)
                         ↓
                  local vector store
                (data/ai_memory/*.json)
                         ↓
              T58 AI RESEARCH AGENT (Ollama)
                         │
     proposes which tool to call next, reasons over
     the result, repeats up to N steps, then answers
                         ↓
        run_backtest / run_prop_simulation / run_monte_carlo /
        run_walk_forward / run_regime_analysis /
        run_parameter_sensitivity / run_cost_stress /
        compare_strategies / search_research / search_experiments
                         ↓
              T58's real backtest/prop/Monte Carlo engine
                (the same one every other tab uses)
```

**The one rule that matters:** the quantitative engine is the authority,
never the model's own judgment. There is no `edit_strategy_code` or
`apply_parameters` tool — the agent can recommend a next step in plain
language ("test tightening the ATR filter — the sensitivity sweep shows a
cliff there"), but turning that into a tested strategy still goes through
Step 6 Iterative Refinement / Quick Optimize / Step 15 Full Pipeline, same
as a human-typed idea would.

**Three layers, in the order they're worth setting up:**

1. **RAG over your research library** (`research/` folder — unchanged
   location from AI Assist). `app.ai.research_library` now does hybrid
   retrieval: plain keyword-overlap scoring always works with zero setup,
   and blends in real semantic search once you pull a local embedding
   model and hit **EMBED RESEARCH LIBRARY** on the Research Agent tab. No
   cloud API, no vector-database server — embeddings are stored locally
   as plain JSON under `data/ai_memory/`.
2. **T58 Research Memory** (`app.ai.experiment_memory`) — every Full
   Pipeline run, Quick Optimize run, and Batch Test item is automatically
   recorded (strategy, verdict, stats, and any lesson learned) into a
   local SQLite database, searchable semantically the same way as the
   paper library. Click **REFRESH MEMORY SUMMARY** on the tab to see the
   running totals (how many strategies tested, broken down by verdict).
3. **The agent loop itself** — type a research question, hit **RUN
   RESEARCH AGENT**. It automatically uses the strategy/data/prop
   rules/risk already configured in Steps 01-04, exactly like Full
   Pipeline does.

Fine-tuning a model on your own accumulated experiments (Level 2 in the
original research-engine plan) is intentionally not built yet — RAG plus
the growing Research Memory table gets most of the value with none of the
training infrastructure, and the memory table itself is exactly the
dataset a future fine-tune would need.

**Setup**, on top of the AI Assist setup above:

1. `ollama pull nomic-embed-text` (or another embedding model) for
   semantic search — optional; without it, `search_research` and
   `search_experiments` still work via plain keyword matching.
2. Open the **18 Research Agent** tab, confirm **AI Assist** is enabled
   and Test Connection passes, optionally click **EMBED RESEARCH
   LIBRARY**, type a research question, and click **RUN RESEARCH AGENT**.

Off by default, everywhere, and fails exactly the same way AI Assist does:
an unreachable/misconfigured Ollama surfaces a clear error in the
transcript rather than a stack trace, and nothing here ever runs
automatically as part of any other tab's workflow.

## PineScript support (subset)

Supported: `open/high/low/close/hl2/hlc3/ohlc4`, `input.int`/`input.float`,
`ta.sma`/`ta.ema`/`ta.wma`/`ta.rsi`, `ta.crossover`/`ta.crossunder`, boolean
rule variables (`and`/`or`/comparisons`), `strategy.entry(..., when=...)` and
`strategy.close(..., when=...)` either inline or inside an `if` block, and
`// T58_SL_PIPS=20` / `// T58_TP_PIPS=40` directive comments for stop-loss/
take-profit (Pine's own `strategy.exit()` uses absolute price offsets, which
aren't a portable "pips" concept across instruments).
Not supported: custom functions, arrays/matrices, `security()`/multi-timeframe
requests, plotting/alerts, and any `ta.*` function beyond the list above.
`input.int()`/`input.float()` values are usable as the *length* argument of a
`ta.*` call, but not as a general numeric constant inside a comparison
expression (e.g. `rsiVal < rsiThreshold`) — use a literal number there
instead; only the four price columns, indicator outputs, and literal numbers
are guaranteed to resolve inside a boolean expression.

## MQL5 support (subset)

Supported: direct-value `iMA(...)` (`MODE_SMA`/`MODE_EMA`/`MODE_LWMA`) and
`iRSI(...)` calls, C-style boolean conditions (`&& || ! > < >= <= == !=`),
`if (cond) { ... }` in both Allman and K&R brace styles plus single-statement
`if (cond) stmt;`, `trade.Buy`/`trade.Sell`/`OrderSend(..., ORDER_TYPE_BUY/SELL
or OP_BUY/OP_SELL, ...)` for entries, `trade.PositionClose`/`OrderClose` for
exits, and the same `// T58_SL_PIPS=` / `// T58_TP_PIPS=` directive comments.
Not supported: `CopyBuffer()`-based indicator handles, custom indicators,
arrays/structs, multi-symbol/multi-timeframe logic, ATR or any indicator
beyond iMA/iRSI, and trailing stops. `iMA()`'s `shift` argument is parsed
but not used — there is no "previous bar's MA" available in this subset, so
a true crossover *event* isn't expressible here (only sustained-state
comparisons); write around this rather than relying on shift.
(The Manual Builder's own trailing stop/break-even support, described above,
is not subject to any of this limitation.)

## Engines

- **Backtest engine** (`app/backtest/`): bar-by-bar execution with
  intrabar stop-loss/take-profit checks — including ATR-based dynamic
  stop/target distances, a ratcheting ATR-based trailing stop, and
  break-even stop management — producing a trade list, equity curve, and
  the full statistics set from the spec (returns, win/loss, risk,
  strategy-quality, risk-adjusted ratios). Also home to
  `run_holdout_comparison()` (a single chronological in-sample/out-of-
  sample split, distinct from the fold-based Walk-Forward Optimization in
  Step 8) and `app/strategy/lookahead_check.py`.
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
- **Optimization engines** (`app/optimize/`): the shared GA operators
  (crossover/mutation/tournament selection/elitism/random immigrants) that
  power Iterative Refinement, the Search Lab's Stage 2, the Walk-Forward-
  Aware GA, and (via non-dominated sorting instead of scalar tournament)
  Multi-Objective Optimization. `refinement.py` also owns the cost-stress
  penalty (Step 14) shared by all three GA-based searches.
- **Report generator** (`app/reports/generator.py` + `app/reports/charts.py`,
  plus `refinement_report.py` and `validation_reports.py` for the
  research-stack features above): combines everything into a report,
  exported as JSON, a flattened summary CSV, a trades CSV, and a
  self-contained HTML report. The HTML report includes inline SVG charts —
  no extra plotting dependency, no external image files — covering the
  historical equity curve, Monte Carlo return/drawdown histograms with
  median/P95 markers, and (for the Validation Lab reports) chained
  out-of-sample equity curves, Pareto-front convergence, and parameter
  sensitivity heatmaps. It also surfaces execution-integrity warnings
  (pip-size/instrument mismatches, gap-through stop fills) and, for Full
  Pipeline reports, the READY/MARGINAL/NOT READY verdict and winning
  parameter values, as banners/tables at the top rather than only in a
  live run log. HTML was chosen over a PDF library dependency for the MVP
  — any browser can print it to PDF with zero extra install burden.

## CLI reference

Every feature above a plain single-strategy run is also available headless
via `python -m app.main --cli <flag> ...`. The base flags (`--csv`,
`--output`, `--sims`) apply throughout; each feature's own flags are listed
under its own heading. Run `python -m app.main --help` for the full,
current list with defaults — this is a summary, not the source of truth.

| Flag | Runs |
|---|---|
| `--refine` (+ `--refine-population/-generations/-metric/-seed`) | Iterative Refinement (Step 6) as part of the normal pipeline |
| `--search` (+ `--search-mode/-family/-strategy-file/-grid-points/-max-candidates/-workers/-min-trades/-min-profit-factor/-stage1-top-n/-stage2-top-n/-ga-population/-ga-generations/-full-mc-sims/-walk-forward-folds/-robustness-neighbors/-metric/-seed/-db/-no-promote`) | Search Lab (Step 7), Stages 1-5 |
| `--wfo` (+ `--wfo-folds/-window-mode/-train-frac/-population/-generations/-metric/-seed`) | Walk-Forward Optimization (Step 8) |
| `--cpcv` (+ `--cpcv-groups/-test-groups/-metric/-max-paths`) | Combinatorial Purged Cross-Validation (Step 9) |
| `--pbo` (+ `--pbo-groups/-test-groups/-metric/-max-paths/-candidates/-seed`) | Probability of Backtest Overfitting (Step 9) |
| `--sensitivity` (+ `--sensitivity-metric/-pct-range/-steps/-heatmap`) | Parameter Sensitivity (Step 10) |
| `--portfolio` (+ `--portfolio-csv` [repeatable, 2+ required] `/-balance/-correlation-strength`) | Multi-Asset Portfolio (Step 11) |
| `--multi-objective` (+ `--mo-objectives/-population/-generations/-seed`) | Multi-Objective Optimization (Step 12) |
| `--wfga` (+ `--wfga-folds/-window-mode/-population/-generations/-metric/-seed`) | Walk-Forward-Aware GA (Step 13) |
| `--ensemble` (+ `--ensemble-strategy` [repeatable, 2+ required] `/-mode/-min-agreement/-balance/-correlation-strength`) | Multi-strategy ensemble, blend or vote (Step 14) |
| `--full-pipeline` (+ `--fp-folds/-window-mode/-population/-generations/-metric/-final-mc-sims/-seed/-no-save-to-library`) | Full Pipeline (Step 15): baseline → GA → re-validated report → OOS/holdout checks → verdict |

`--refine` and `--search` additionally accept `--refine-no-cost-stress` /
`--refine-cost-stress-multiplier` / `--refine-cost-stress-weight` and
`--search-no-cost-stress` / `--search-cost-stress-multiplier` /
`--search-cost-stress-weight` respectively (Step 14's cost-stress fitness,
on by default). `--search` also accepts `--pair-csv <path>` to merge in a
second instrument so the `stat_pairs` family can be searched. Plain `--cli`
accepts `--adaptive-risk-rules '<json>'` (Step 14's adaptive risk layer).

Each of the ten `--wfo`/`--cpcv`/`--pbo`/`--sensitivity`/`--portfolio`/
`--multi-objective`/`--wfga`/`--ensemble`/`--full-pipeline` runs is mutually
exclusive with the others and with `--search`/plain `--cli`; pick one per
invocation. All write their report(s) under `--output` (default `reports/`).

AI Assist (above) is currently desktop-GUI-only — `--full-pipeline` runs
the same search headlessly without it. The AI Research Agent (Step 18) is
also desktop-GUI-only for now; there's no CLI flag or web route for it yet.

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
  cloud host). The Validation Lab (Steps 8-13) is desktop-only for now —
  it's fully usable headlessly via the CLI in the meantime.
- One open position at a time (consistent with the standardized long/flat/
  short signal model); no partial fills or multi-leg positions in v1.
  Account-state-dependent money management (daily-loss circuit breakers,
  consecutive-loss risk scaling, progress-to-target coasting) still can't
  be expressed inside a strategy's own `generate_signals()` — but is now
  available as a first-class, declarative engine feature; see Step 14's
  adaptive risk layer.
- Multi-timeframe analysis is implemented as an as-of merge onto the finest
  selected timeframe (see step 1 above) rather than running fully separate
  per-timeframe backtests — this keeps every strategy source (Manual,
  Python, PineScript, MQL5) working against one dataframe unchanged.
- Multi-Asset Portfolio backtesting (Step 11) uses a static (whole-window)
  correlation pass and combines legs by chronological trade-close time
  rather than a fully unified multi-position margin engine — the right
  model for "one account, one drawdown floor, several instruments," not
  for a margin-constrained concurrent-position book. See
  `app/portfolio/portfolio.py`'s docstring for the full reasoning.
- AI Assist (Step 15) is desktop-GUI-only for now, same as the Validation
  Lab — `--full-pipeline` runs the identical search headlessly, just
  without the optional AI-suggested candidates. It also only ever
  proposes numeric values for a strategy's already-discovered tunable
  parameters, deliberately never code — keeping the search's safety
  properties (every candidate re-validated through the normal backtest/
  prop-sim/Monte Carlo pipeline) unchanged whether AI Assist is on or off.

## Project layout

```
T58-Prop-Algo-Backtester/
├── run_app.py                  # PyInstaller entry point (must stay at repo root — see .exe section)
├── run_web.py                  # PyInstaller entry point, web/phone edition (must stay at repo root)
├── config/                     # pyproject.toml, requirements.txt
├── docs/                       # WEB_PARITY_ROADMAP.md
├── app/
│   ├── main.py                 # entry point (GUI, or --cli headless run — see CLI reference)
│   ├── ui/
│   │   ├── main_window.py      # Tkinter desktop GUI (Steps 1-7 core, 8-13 Validation Lab, 14 Ensemble, 15 Full Pipeline, 16 Forward Test, 17 Evolution Lab, 18 AI Research Agent)
│   │   └── condition_builder.py  # visual condition-row widget used by the Manual Builder
│   ├── web/                    # Flask mobile/web app (same engine, new front end)
│   │   ├── server.py
│   │   ├── templates/           # index, dashboard, search, search-job, and shared partials
│   │   └── static/             # manifest.json, service worker, icons
│   ├── data/
│   │   ├── importer.py         # CSV import + validation
│   │   ├── storage.py          # persists imported CSVs alongside the app/exe
│   │   ├── multi_timeframe.py  # merges multiple timeframes onto the finest one
│   │   ├── pairs.py            # Step 14: merges a second instrument's close in for pairs/relative-value
│   │   ├── alpaca_source.py    # optional Alpaca API data fetch (US equities/crypto)
│   │   └── alpaca_credentials.py
│   ├── strategy/                # manual / python / pinescript / mql5 adapters
│   │   ├── indicators.py        # shared indicator math (SMA/EMA/WMA/RSI/MACD/ATR/Bollinger/etc.)
│   │   ├── expr.py              # shared safe boolean-expression evaluator
│   │   ├── manual.py            # visual-builder condition + risk-management engine
│   │   ├── python.py / pinescript.py / mql5.py
│   │   ├── mtf.py                # safe "last fully-closed HTF bar" helper (avoids the #1 real lookahead trap)
│   │   ├── lookahead_check.py    # generic, code-agnostic lookahead-bias detector
│   │   └── library.py            # persistent strategy library (save/load python/pinescript/mql5)
│   ├── backtest/                 # execution engine, risk sizing, statistics, holdout comparison
│   │   └── adaptive_risk.py      # Step 14: declarative consecutive-loss/daily-P&L/progress-to-target sizing rules
│   ├── ensemble/ensemble.py       # Step 14: multi-strategy ensembles (blend or vote) on one instrument
│   ├── orchestration/full_pipeline.py  # Step 15: one-button baseline -> GA -> re-validation -> OOS/holdout -> verdict
│   ├── evolution/                 # Step 17: Evolution Lab (unattended generate/filter/mutate loop)
│   │   ├── engine.py              # the generation loop itself (EvolutionRunner)
│   │   ├── checkpoint.py          # on-disk checkpoint (resume) + tested-candidates log
│   │   ├── prop_fitness.py        # composite PROP FITNESS ranking score
│   │   └── knowledge_graph.py     # append-only feature-vector -> outcome log + similarity queries
│   ├── ai/                        # optional local-Ollama AI Assist + AI Research Engine (off by default)
│   │   ├── ollama_client.py       # connection test + per-generation parameter-suggestion requests
│   │   ├── ollama_settings.py     # persisted host/model/API-key settings (keyring-backed)
│   │   ├── strategy_generator.py  # drafts a new strategy file from a plain-language idea (tagged DRAFT)
│   │   ├── research_library.py    # research/ paper library: keyword + (optional) semantic RAG retrieval
│   │   ├── vector_store.py        # local embedding store (Ollama /api/embeddings + cosine similarity, JSON-backed)
│   │   ├── experiment_memory.py   # Step 18: durable + semantically-searchable record of every strategy test
│   │   └── research_agent.py      # Step 18: ReAct tool-calling research agent over the real engine
│   ├── optimize/
│   │   ├── parameter_space.py / code_parameter_space.py   # shared gene discovery (all 4 strategy sources)
│   │   ├── refinement.py         # Step 6: Iterative Refinement GA
│   │   ├── multi_objective.py    # Step 12: NSGA-II Pareto-front optimization
│   │   └── walkforward_ga.py     # Step 13: walk-forward-aware GA
│   ├── validation/
│   │   ├── walk_forward_opt.py   # Step 8: walk-forward optimization + fold splitting
│   │   ├── cpcv.py                # Step 9: Combinatorial Purged CV + Probability of Backtest Overfitting
│   │   └── sensitivity.py         # Step 10: 1D sweeps + 2D heatmaps
│   ├── portfolio/portfolio.py     # Step 11: multi-asset portfolio backtesting
│   ├── search/                    # Step 7: Search Lab (5-stage funnel)
│   │   ├── strategy_space.py      # named-hypothesis families + candidate-spec builder
│   │   ├── batch_runner.py        # Stages 1-5 orchestration
│   │   ├── robustness.py          # walk-forward holdout, parameter-neighborhood robustness, Deflated Sharpe
│   │   ├── results_db.py          # SQLite leaderboard storage
│   │   └── search_report.py
│   ├── prop/simulator.py          # prop-firm rules + account simulator
│   ├── monte_carlo/engine.py
│   └── reports/
│       ├── generator.py           # JSON / CSV / HTML report export (single-strategy runs)
│       ├── refinement_report.py   # Step 6 report
│       ├── validation_reports.py  # Steps 8-13 reports (JSON + focused HTML per feature)
│       └── charts.py              # dependency-free SVG chart generation, incl. 2D heatmaps
├── data/
│   ├── examples/                  # sample OHLCV dataset for immediate testing
│   ├── raw/                       # dataset for common forex pairs (1min, 5min, 15min, 1hr, 4hr, and daily timeframes)
├── strategies/                    # persistent Strategy Library storage (python/pinescript/mql5 + metadata)
├── tests/                         # pytest unit tests for every engine (~30 test files)
└── .github/workflows/
    ├── build.yml                  # runs pytest on push/PR
    └── build-exe.yml              # builds & uploads the Windows .exe (entry point: run_app.py)
```

## Tests

```bash
pytest -q tests
```

## Disclaimer

Simulated results are estimates derived from historical data and
resampling. Past performance and simulated outcomes do not guarantee future
results.
