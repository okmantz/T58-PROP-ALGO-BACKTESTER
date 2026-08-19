"""
T58 Prop Algo Backtester — Desktop GUI (Tkinter, stdlib only).

Implements the MVP step wizard described in the product spec:
  1. Upload Market Data
  2. Import/Create Strategy
  3. Enter Prop-Firm Rules
  4. Configure Risk & Execution
  5. Run Backtest -> Prop Simulation -> Monte Carlo -> Report

Kept intentionally simple (no external GUI framework) so the MVP has zero
extra install burden beyond pandas/numpy.
"""
from __future__ import annotations

import os
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, Text, END, filedialog, messagebox, ttk, Listbox, SINGLE,
)

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.storage import list_stored_datasets, store_csv_path
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

OUTPUT_DIR = Path.cwd() / "reports"

DEFAULT_MANUAL_STRATEGY = {
    "name": "SMA 20/50 Cross",
    "indicators": [
        {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
        {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"},
    ],
    "long_entry": "sma_fast > sma_slow",
    "long_exit": "sma_fast < sma_slow",
    "short_entry": "sma_fast < sma_slow",
    "short_exit": "sma_fast > sma_slow",
    "stop_loss_pips": 20,
    "take_profit_pips": 40,
}


class LabeledEntry(Frame):
    def __init__(self, parent, label, default=""):
        super().__init__(parent)
        Label(self, text=label, width=28, anchor="w").pack(side="left")
        self.var = StringVar(value=str(default))
        Entry(self, textvariable=self.var, width=18).pack(side="left")
        self.pack(fill="x", pady=2, padx=8)

    def get_float(self, fallback=0.0):
        try:
            return float(self.var.get())
        except ValueError:
            return fallback

    def get_int(self, fallback=0):
        try:
            return int(float(self.var.get()))
        except ValueError:
            return fallback

    def get_str(self):
        return self.var.get()


class MainWindow:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("T58 Trading — Prop Algo Backtester (MVP)")
        self.root.geometry("880x720")

        self.csv_path: str | None = None
        self.strategy_py_path: str | None = None
        self.strategy_mode = StringVar(value="manual")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        self.tab_data = Frame(nb)
        self.tab_strategy = Frame(nb)
        self.tab_prop = Frame(nb)
        self.tab_risk = Frame(nb)
        self.tab_run = Frame(nb)

        nb.add(self.tab_data, text="1. Market Data")
        nb.add(self.tab_strategy, text="2. Strategy")
        nb.add(self.tab_prop, text="3. Prop-Firm Rules")
        nb.add(self.tab_risk, text="4. Risk & Execution")
        nb.add(self.tab_run, text="5. Run & Report")

        self._build_data_tab()
        self._build_strategy_tab()
        self._build_prop_tab()
        self._build_risk_tab()
        self._build_run_tab()

    # ---------------------------------------------------------- Tab 1
    def _build_data_tab(self):
        f = self.tab_data
        Label(f, text="Step 1 — Market Data", font=("", 13, "bold")).pack(pady=10, anchor="w", padx=10)

        Label(
            f,
            text="Datasets stored in data/raw/ (auto-loaded every time you open the app):",
            fg="#555",
        ).pack(anchor="w", padx=10)

        list_frame = Frame(f)
        list_frame.pack(fill="x", padx=10, pady=6)
        self.dataset_listbox = Listbox(list_frame, height=8, selectmode=SINGLE, exportselection=False)
        self.dataset_listbox.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, command=self.dataset_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selected)

        btn_row = Frame(f)
        btn_row.pack(anchor="w", padx=10, pady=4)
        Button(btn_row, text="Import CSV(s)...", command=self._browse_csv).pack(side="left")
        Button(btn_row, text="Refresh list", command=self._refresh_dataset_list).pack(side="left", padx=6)

        self.data_status = Label(f, text="No dataset selected.", fg="#555")
        self.data_status.pack(anchor="w", padx=10, pady=6)
        Label(
            f,
            text="Selecting one or more CSVs here copies them into data/raw/ automatically, so they're "
                 "already available the next time you open the app -- no re-uploading. You can also just "
                 "drop CSV files into data/raw/ yourself; click 'Refresh list' to pick them up.",
            fg="#777", wraplength=820, justify="left",
        ).pack(anchor="w", padx=10)

        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END)
        self._stored_datasets = list_stored_datasets()
        for ds in self._stored_datasets:
            self.dataset_listbox.insert(END, ds.name)
        if self._stored_datasets and not self.csv_path:
            self.dataset_listbox.selection_set(0)
            self._select_dataset(self._stored_datasets[0].path)

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        if not sel:
            return
        ds = self._stored_datasets[sel[0]]
        self._select_dataset(ds.path)

    def _select_dataset(self, path: Path):
        result = import_csv(path)
        if not result.is_valid:
            messagebox.showerror("Import failed", "\n".join(result.errors))
            self.data_status.config(text=f"{path.name}: import failed. See error dialog.", fg="red")
            return
        self.csv_path = str(path)
        n = len(result.dataframe)
        warn = f" ({len(result.warnings)} warning(s))" if result.warnings else ""
        self.data_status.config(text=f"Active dataset: {path.name} — {n} bars{warn}", fg="green")

    def _browse_csv(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not paths:
            return

        imported, failed = [], []
        for p in paths:
            result = import_csv(p)
            if not result.is_valid:
                failed.append((os.path.basename(p), "; ".join(result.errors)))
                continue
            stored_path = store_csv_path(p)
            imported.append(stored_path)

        self._refresh_dataset_list()

        if imported:
            # make the most recently imported file the active dataset
            self._select_dataset(imported[-1])
            for i, ds in enumerate(self._stored_datasets):
                if ds.path == imported[-1]:
                    self.dataset_listbox.selection_clear(0, END)
                    self.dataset_listbox.selection_set(i)
                    break

        if failed:
            detail = "\n".join(f"- {name}: {err}" for name, err in failed)
            messagebox.showwarning(
                "Some files failed to import",
                f"{len(imported)} file(s) imported successfully.\n\n"
                f"{len(failed)} file(s) failed:\n{detail}",
            )
        elif imported:
            messagebox.showinfo("Import complete", f"Imported and stored {len(imported)} file(s) in data/raw/.")

    # ---------------------------------------------------------- Tab 2
    def _build_strategy_tab(self):
        f = self.tab_strategy
        Label(f, text="Step 2 — Import/Create Strategy", font=("", 13, "bold")).pack(pady=10, anchor="w", padx=10)

        modes = Frame(f)
        modes.pack(anchor="w", padx=10)
        for val, text in [("manual", "Manual Builder (SMA 20/50 cross default)"),
                           ("python", "Python (.py)"), ("pinescript", "PineScript (.pine)"),
                           ("mql5", "MQL5 (.mq5)")]:
            Button(modes, text=text, command=lambda v=val: self._set_strategy_mode(v)).pack(side="left", padx=4)

        self.strategy_mode_label = Label(f, text="Selected: manual (SMA 20/50 cross)", fg="#333")
        self.strategy_mode_label.pack(anchor="w", padx=10, pady=6)

        Button(f, text="Browse for strategy file...", command=self._browse_strategy_file).pack(anchor="w", padx=10)
        self.strategy_file_status = Label(f, text="(only needed for Python/PineScript/MQL5 modes)", fg="#777")
        self.strategy_file_status.pack(anchor="w", padx=10, pady=6)

        Label(f, text="Manual builder params (used when mode = manual):").pack(anchor="w", padx=10, pady=(12, 0))
        self.sma_fast = LabeledEntry(f, "SMA fast period", 20)
        self.sma_slow = LabeledEntry(f, "SMA slow period", 50)
        self.sl_pips = LabeledEntry(f, "Stop loss (pips)", 20)
        self.tp_pips = LabeledEntry(f, "Take profit (pips)", 40)

    def _set_strategy_mode(self, mode: str):
        self.strategy_mode.set(mode)
        self.strategy_mode_label.config(text=f"Selected: {mode}")

    def _browse_strategy_file(self):
        ext = {"python": "*.py", "pinescript": "*.pine", "mql5": "*.mq5"}.get(self.strategy_mode.get(), "*.*")
        path = filedialog.askopenfilename(filetypes=[("Strategy file", ext)])
        if path:
            self.strategy_py_path = path
            self.strategy_file_status.config(text=f"Selected: {os.path.basename(path)}", fg="green")

    def _build_strategy(self):
        mode = self.strategy_mode.get()
        if mode == "manual":
            cfg = dict(DEFAULT_MANUAL_STRATEGY)
            cfg["indicators"] = [
                {"type": "sma", "period": self.sma_fast.get_int(20), "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": self.sma_slow.get_int(50), "column": "close", "as": "sma_slow"},
            ]
            cfg["stop_loss_pips"] = self.sl_pips.get_float(20)
            cfg["take_profit_pips"] = self.tp_pips.get_float(40)
            return ManualStrategy(cfg)
        if not self.strategy_py_path:
            raise StrategyError(f"No file selected for '{mode}' strategy mode.")
        if mode == "python":
            return PythonStrategy(self.strategy_py_path)
        if mode == "pinescript":
            return PineScriptStrategy(self.strategy_py_path)
        if mode == "mql5":
            return MQL5Strategy(self.strategy_py_path)
        raise StrategyError(f"Unknown strategy mode: {mode}")

    # ---------------------------------------------------------- Tab 3
    def _build_prop_tab(self):
        f = self.tab_prop
        Label(f, text="Step 3 — Prop-Firm Rules", font=("", 13, "bold")).pack(pady=10, anchor="w", padx=10)
        self.p_account_size = LabeledEntry(f, "Account size ($)", 100000)
        self.p_profit_target = LabeledEntry(f, "Evaluation profit target (%)", 8)
        self.p_daily_loss = LabeledEntry(f, "Daily loss limit (%)", 5)
        self.p_max_dd = LabeledEntry(f, "Maximum drawdown (%)", 10)
        self.p_dd_type = LabeledEntry(f, "Drawdown type (trailing/static)", "trailing")
        self.p_consistency = LabeledEntry(f, "Consistency rule (% best day of total profit)", 30)
        self.p_min_days = LabeledEntry(f, "Minimum trading days", 5)
        self.p_payout_threshold = LabeledEntry(f, "Payout threshold (extra % profit)", 0)
        self.p_payout_cap = LabeledEntry(f, "Payout cap (% of profit, blank=100)", 100)
        self.p_payout_freq = LabeledEntry(f, "Payout frequency (days)", 14)
        self.p_buffer = LabeledEntry(f, "Required buffer (%)", 0)
        self.p_max_pos = LabeledEntry(f, "Max position size (units, blank=unlimited)", "")

    def _build_prop_rules(self) -> PropRules:
        cap = self.p_payout_cap.get_str().strip()
        max_pos = self.p_max_pos.get_str().strip()
        return PropRules(
            account_size=self.p_account_size.get_float(100000),
            evaluation_profit_target_pct=self.p_profit_target.get_float(8),
            daily_loss_limit_pct=self.p_daily_loss.get_float(5),
            max_drawdown_pct=self.p_max_dd.get_float(10),
            drawdown_type=self.p_dd_type.get_str().strip() or "trailing",
            consistency_rule_pct=self.p_consistency.get_float(30),
            min_trading_days=self.p_min_days.get_int(5),
            payout_threshold_pct=self.p_payout_threshold.get_float(0),
            payout_cap_pct=float(cap) if cap else None,
            payout_frequency_days=self.p_payout_freq.get_int(14),
            required_buffer_pct=self.p_buffer.get_float(0),
            max_position_size=float(max_pos) if max_pos else None,
        )

    # ---------------------------------------------------------- Tab 4
    def _build_risk_tab(self):
        f = self.tab_risk
        Label(f, text="Step 4 — Configure Risk & Execution", font=("", 13, "bold")).pack(pady=10, anchor="w", padx=10)
        self.r_initial_balance = LabeledEntry(f, "Initial balance ($)", 100000)
        self.r_risk_mode = LabeledEntry(f, "Risk mode (percent/fixed)", "percent")
        self.r_risk_value = LabeledEntry(f, "Risk per trade (% or $)", 1.0)
        self.r_max_trades_day = LabeledEntry(f, "Max trades/day", 10)
        self.r_commission = LabeledEntry(f, "Commission per trade ($)", 0)
        self.r_slippage = LabeledEntry(f, "Slippage (pips)", 0.5)
        self.r_spread = LabeledEntry(f, "Spread (pips)", 1.0)
        self.r_pip_size = LabeledEntry(f, "Pip size (e.g. 0.0001 FX)", 0.0001)

    def _build_risk_config(self) -> RiskConfig:
        return RiskConfig(
            initial_balance=self.r_initial_balance.get_float(100000),
            risk_mode=self.r_risk_mode.get_str().strip() or "percent",
            risk_value=self.r_risk_value.get_float(1.0),
            max_trades_per_day=self.r_max_trades_day.get_int(10),
            commission_per_trade=self.r_commission.get_float(0),
            slippage_pips=self.r_slippage.get_float(0.5),
            spread_pips=self.r_spread.get_float(1.0),
            pip_size=self.r_pip_size.get_float(0.0001),
        )

    # ---------------------------------------------------------- Tab 5
    def _build_run_tab(self):
        f = self.tab_run
        Label(f, text="Step 5 — Run Backtest -> Prop Simulation -> Monte Carlo -> Report",
              font=("", 13, "bold")).pack(pady=10, anchor="w", padx=10)

        opts = Frame(f)
        opts.pack(anchor="w", padx=10)
        self.mc_sims = LabeledEntry(opts, "Monte Carlo simulations", 10000)
        self.mc_method = LabeledEntry(opts, "Method (bootstrap/shuffle/block_bootstrap)", "bootstrap")

        Button(f, text="Run Full Pipeline", command=self._run_clicked, bg="#111", fg="white",
               font=("", 11, "bold")).pack(anchor="w", padx=10, pady=10)

        self.progress = ttk.Progressbar(f, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=4)

        self.output = Text(f, height=24, wrap="word")
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

        self.open_report_btn = Button(f, text="Open HTML Report", command=self._open_report, state="disabled")
        self.open_report_btn.pack(anchor="w", padx=10, pady=(0, 10))
        self._last_html_path: Path | None = None

    def _log(self, msg: str):
        self.output.insert(END, msg + "\n")
        self.output.see(END)
        self.root.update_idletasks()

    def _run_clicked(self):
        if not self.csv_path:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.output.delete("1.0", END)
        self.progress.start(10)
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _open_report(self):
        if self._last_html_path:
            webbrowser.open(f"file://{self._last_html_path.resolve()}")

    def _run_pipeline(self):
        try:
            self._log("Importing market data...")
            import_result = import_csv(self.csv_path)
            if not import_result.is_valid:
                self._log("Import errors:\n" + "\n".join(import_result.errors))
                return
            df = import_result.dataframe
            for w in import_result.warnings:
                self._log(f"  [warning] {w}")
            self._log(f"Loaded {len(df)} bars.")

            self._log("Building strategy...")
            strategy = self._build_strategy()

            self._log("Configuring risk & prop rules...")
            risk = self._build_risk_config()
            rules = self._build_prop_rules()

            self._log("Running historical backtest...")
            bt_result = run_backtest(df, strategy, risk)
            self._log(f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}  "
                       f"Win rate: {bt_result.statistics.win_rate:.1f}%  Max DD: {bt_result.statistics.max_drawdown_pct:.2f}%")

            self._log("Running prop-firm simulation on historical sequence...")
            trade_pnls = [t.pnl for t in bt_result.trades]
            trade_dates = [t.entry_time for t in bt_result.trades]
            single_run = simulate_account(trade_pnls, trade_dates, rules)
            self._log(f"  Passed evaluation: {single_run.passed_evaluation}  Reached payout: {single_run.reached_first_payout}  "
                       f"Failed: {single_run.failed} ({single_run.failure_reason})")

            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            self._log(f"Running Monte Carlo simulation ({n_sims:,} runs, method={method})...")
            mc_cfg = MonteCarloConfig(n_simulations=n_sims, method=method)
            mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)
            self._log(f"  Evaluation pass probability: {mc_result.evaluation_pass_probability:.1f}%")
            self._log(f"  First payout probability: {mc_result.first_payout_probability:.1f}%")
            self._log(f"  Expected payout: ${mc_result.expected_payout:,.2f}")
            self._log(f"  Risk of ruin: {mc_result.risk_of_ruin_pct:.1f}%")

            self._log("Generating report...")
            period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
            paths = generate_full_report(
                output_dir=OUTPUT_DIR,
                strategy_name=bt_result.strategy_name,
                strategy_source_type=strategy.source_type,
                instrument=os.path.basename(self.csv_path),
                timeframe="unknown",
                backtest_period=period,
                backtest_result=bt_result,
                prop_rules=rules,
                prop_single_run=single_run,
                monte_carlo_result=mc_result,
            )
            self._last_html_path = paths["html"]
            self.open_report_btn.config(state="normal")
            self._log("\nDone. Report written to:")
            for k, p in paths.items():
                self._log(f"  {k}: {p}")

        except StrategyError as exc:
            self._log(f"\nStrategy error: {exc}")
        except Exception:  # noqa: BLE001
            self._log("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.progress.stop()


def launch():
    root = Tk()
    MainWindow(root)
    root.mainloop()
