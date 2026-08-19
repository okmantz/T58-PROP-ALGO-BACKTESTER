"""
T58 Prop Algo Backtester — Desktop GUI.

Dark T58 interface with a full visual Manual Strategy Builder. The builder
creates a structured JSON-like strategy configuration consumed by
app.strategy.manual.ManualStrategy; no strategy code is required.

The original data, strategy-file, prop-rules, risk, run, Monte Carlo and
report workflows are preserved.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, Text, END,
    filedialog, messagebox, ttk, Listbox, SINGLE, BooleanVar, Canvas,
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
    "description": "Trend-following moving-average crossover strategy.",
    "author": "",
    "version": "1.0",
    "market": {"instrument": "", "timeframe": "5m", "session": "All", "direction": "Both"},
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

BG = "#080A0D"
PANEL = "#101318"
PANEL_2 = "#15191F"
PANEL_3 = "#1B2027"
BORDER = "#292E36"
BORDER_LIGHT = "#3A414B"
TEXT = "#E5E7EB"
TEXT_MUTED = "#8B929D"
TEXT_DIM = "#626A75"
METAL = "#B8BDC5"
METAL_BRIGHT = "#E1E4E8"
GREEN = "#43D17A"
RED = "#F05B63"
BLUE = "#6FA8FF"
AMBER = "#D9A441"
FONT = "Segoe UI"
MONO = "Consolas"

CONDITION_SOURCES = [
    "Price", "EMA", "SMA", "VWAP", "RSI", "MACD", "MACD Signal", "MACD Histogram",
    "ATR", "Bollinger Bands", "Bollinger Upper", "Bollinger Lower", "Highest High", "Lowest Low", "Volume", "Average Volume",
    "Candle Direction", "Candle Range", "Percentage Change", "Swing High", "Swing Low",
    "Liquidity Sweep", "Break of Structure", "Change of Character", "Fair Value Gap", "Order Block",
    "Session High", "Session Low", "Previous Day High", "Previous Day Low", "Previous Day Close",
    "Opening Range High", "Opening Range Low", "ATR Regime", "Volatility Regime",
]
RIGHT_SOURCES = ["Value"] + CONDITION_SOURCES
OPERATORS = [
    "Greater Than", "Greater Than or Equal", "Less Than", "Less Than or Equal",
    "Equal To", "Not Equal", "Cross Above", "Cross Below", "Is True", "Is False",
]
SOURCE_KIND = {
    "Price": "price", "EMA": "ema", "SMA": "sma", "VWAP": "vwap", "RSI": "rsi",
    "MACD": "macd", "MACD Signal": "macd_signal", "MACD Histogram": "macd_histogram",
    "ATR": "atr", "Bollinger Bands": "bollinger_mid", "Bollinger Upper": "bollinger_upper", "Bollinger Lower": "bollinger_lower", "Highest High": "highest_high",
    "Lowest Low": "lowest_low", "Volume": "volume", "Average Volume": "average_volume",
    "Candle Direction": "candle_direction", "Candle Range": "candle_range",
    "Percentage Change": "percentage_change", "Swing High": "swing_high", "Swing Low": "swing_low",
    "Liquidity Sweep": "liquidity_sweep", "Break of Structure": "break_of_structure",
    "Change of Character": "change_of_character", "Fair Value Gap": "fair_value_gap",
    "Order Block": "order_block", "Session High": "session_high", "Session Low": "session_low",
    "Previous Day High": "previous_day_high", "Previous Day Low": "previous_day_low",
    "Previous Day Close": "previous_day_close", "Opening Range High": "opening_range_high",
    "Opening Range Low": "opening_range_low", "ATR Regime": "atr_regime", "Volatility Regime": "volatility_regime",
}


def _safe_font(size=10, weight="normal"):
    return (FONT, size, weight)


class LabeledEntry(Frame):
    def __init__(self, parent, label, default="", width=20):
        super().__init__(parent, bg=PANEL)
        Label(self, text=label, width=31, anchor="w", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
        self.var = StringVar(value=str(default))
        self.entry = Entry(self, textvariable=self.var, width=width, bg=PANEL_3, fg=TEXT,
                           insertbackground=TEXT, relief="flat", highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=BORDER_LIGHT, font=_safe_font(10))
        self.entry.pack(side="left", ipady=5, padx=(4, 0))
        self.pack(fill="x", pady=3, padx=18)

    def get_float(self, fallback=0.0):
        try:
            return float(self.var.get())
        except (TypeError, ValueError):
            return fallback

    def get_int(self, fallback=0):
        try:
            return int(float(self.var.get()))
        except (TypeError, ValueError):
            return fallback

    def get_str(self):
        return self.var.get()


class ConditionRow:
    """One visual IF condition row."""
    def __init__(self, parent, remove_callback, defaults=None):
        self.parent = parent
        self.remove_callback = remove_callback
        d = defaults or {}
        self.frame = Frame(parent, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
        self.frame.pack(fill="x", padx=12, pady=4)

        self.source = StringVar(value=d.get("source", "Price"))
        self.field = StringVar(value=d.get("field", "close"))
        self.period = StringVar(value=str(d.get("period", 14)))
        self.operator = StringVar(value=d.get("operator", "Greater Than"))
        self.right = StringVar(value=d.get("right_source", "Value"))
        self.right_value = StringVar(value=str(d.get("right_value", 0)))
        self.right_period = StringVar(value=str(d.get("right_period", 14)))
        self.direction = StringVar(value=d.get("direction", "Both"))
        self.lookback = StringVar(value=str(d.get("lookback", 20)))
        self.session_start = StringVar(value=d.get("session_start", "08:30"))
        self.session_end = StringVar(value=d.get("session_end", "15:00"))

        self._combo(self.frame, self.source, CONDITION_SOURCES, 18).pack(side="left", padx=(8, 4), pady=7)
        self._entry(self.frame, self.field, 7).pack(side="left", padx=2)
        self._entry(self.frame, self.period, 5).pack(side="left", padx=2)
        self._combo(self.frame, self.operator, OPERATORS, 18).pack(side="left", padx=2)
        self._combo(self.frame, self.right, RIGHT_SOURCES, 18).pack(side="left", padx=2)
        self._entry(self.frame, self.right_value, 8).pack(side="left", padx=2)
        self._entry(self.frame, self.right_period, 5).pack(side="left", padx=2)
        self._button(self.frame, "×", lambda: remove_callback(self), width=3).pack(side="right", padx=6)

        # Second line contains advanced condition parameters. They are harmless
        # for ordinary indicators and make structure/session rules configurable.
        advanced = Frame(self.frame, bg=PANEL_2)
        advanced.pack(fill="x", padx=8, pady=(0, 7))
        Label(advanced, text="Direction", bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(8)).pack(side="left")
        self._combo(advanced, self.direction, ["Both", "Bullish", "Bearish"], 10).pack(side="left", padx=4)
        Label(advanced, text="Lookback", bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(8)).pack(side="left", padx=(8, 2))
        self._entry(advanced, self.lookback, 5).pack(side="left")
        Label(advanced, text="Session", bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(8)).pack(side="left", padx=(10, 2))
        self._entry(advanced, self.session_start, 6).pack(side="left")
        Label(advanced, text="to", bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(8)).pack(side="left", padx=2)
        self._entry(advanced, self.session_end, 6).pack(side="left")

    def _combo(self, parent, var, values, width):
        return ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width,
                            style="T58.TCombobox", font=_safe_font(8))

    def _entry(self, parent, var, width):
        return Entry(parent, textvariable=var, width=width, bg=PANEL_3, fg=TEXT,
                     insertbackground=TEXT, relief="flat", highlightthickness=1,
                     highlightbackground=BORDER, font=_safe_font(8))

    def _button(self, parent, text, command, width=None):
        kw = dict(text=text, command=command, bg=PANEL_3, fg=TEXT_MUTED, activebackground=BORDER_LIGHT,
                  activeforeground=TEXT, relief="flat", bd=0, font=_safe_font(10, "bold"), padx=5, pady=2)
        if width:
            kw["width"] = width
        return Button(parent, **kw)

    def to_config(self):
        source = self.source.get()
        kind = SOURCE_KIND[source]
        left = {
            "type": kind,
            "field": self.field.get().strip() or "close",
            "period": max(int(float(self.period.get() or 14)), 1),
            "lookback": max(int(float(self.lookback.get() or 20)), 1),
            "direction": self.direction.get(),
            "session_start": self.session_start.get().strip() or "08:30",
            "session_end": self.session_end.get().strip() or "15:00",
        }
        if self.right.get() == "Value":
            try:
                right = {"type": "value", "value": float(self.right_value.get() or 0)}
            except ValueError as exc:
                raise StrategyError("Every numeric condition value must be a number.") from exc
        else:
            right_source = self.right.get()
            right = {
                "type": SOURCE_KIND[right_source],
                "field": "close",
                "period": max(int(float(self.right_period.get() or 14)), 1),
                "lookback": max(int(float(self.lookback.get() or 20)), 1),
                "direction": self.direction.get(),
                "session_start": self.session_start.get().strip() or "08:30",
                "session_end": self.session_end.get().strip() or "15:00",
            }
        return {"left": left, "operator": self.operator.get(), "right": right}

    def destroy(self):
        self.frame.destroy()


class MainWindow:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("T58 Trading — Prop Algo Backtester")
        self.root.geometry("1250x900")
        self.root.minsize(1050, 760)
        self.root.configure(bg=BG)
        self.csv_path: str | None = None
        self.strategy_py_path: str | None = None
        self.strategy_mode = StringVar(value="manual")
        self._configure_styles()

        shell = Frame(root, bg=BG)
        shell.pack(fill="both", expand=True)
        self._build_header(shell)
        self.nb = ttk.Notebook(shell, style="T58.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.tab_data = Frame(self.nb, bg=BG)
        self.tab_strategy = Frame(self.nb, bg=BG)
        self.tab_prop = Frame(self.nb, bg=BG)
        self.tab_risk = Frame(self.nb, bg=BG)
        self.tab_run = Frame(self.nb, bg=BG)
        self.nb.add(self.tab_data, text="  01  DATA  ")
        self.nb.add(self.tab_strategy, text="  02  STRATEGY  ")
        self.nb.add(self.tab_prop, text="  03  PROP RULES  ")
        self.nb.add(self.tab_risk, text="  04  RISK  ")
        self.nb.add(self.tab_run, text="  05  RUN & REPORT  ")
        self._build_data_tab()
        self._build_strategy_tab()
        self._build_prop_tab()
        self._build_risk_tab()
        self._build_run_tab()

    # ------------------------------------------------------------------ shell
    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("T58.TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("T58.TNotebook.Tab", background=PANEL, foreground=TEXT_MUTED, padding=[18, 10],
                        borderwidth=0, font=_safe_font(9, "bold"))
        style.map("T58.TNotebook.Tab", background=[("selected", PANEL_2), ("active", PANEL_3)],
                  foreground=[("selected", METAL_BRIGHT), ("active", TEXT)])
        style.configure("T58.TCombobox", fieldbackground=PANEL_3, background=PANEL_3,
                        foreground=TEXT, bordercolor=BORDER, arrowcolor=METAL, padding=4)
        style.map("T58.TCombobox", fieldbackground=[("readonly", PANEL_3)], foreground=[("readonly", TEXT)])
        style.configure("T58.TCheckbutton", background=PANEL, foreground=TEXT_MUTED, font=_safe_font(9))
        style.map("T58.TCheckbutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
        style.configure("T58.Vertical.TScrollbar", background=PANEL_2, troughcolor=BG, bordercolor=BG, arrowcolor=TEXT_DIM)
        style.configure("T58.Horizontal.TProgressbar", background=METAL, troughcolor=PANEL_3,
                        bordercolor=PANEL_3, lightcolor=METAL, darkcolor=METAL)

    def _build_header(self, parent):
        header = Frame(parent, bg=BG, height=92)
        header.pack(fill="x", padx=18, pady=(16, 8)); header.pack_propagate(False)
        mark = Frame(header, bg=BG); mark.pack(side="left", fill="y")
        Label(mark, text="T58", bg=BG, fg=METAL_BRIGHT, font=_safe_font(32, "bold")).pack(anchor="w")
        Label(mark, text="PROP ALGO BACKTESTER", bg=BG, fg=TEXT_MUTED, font=_safe_font(8, "bold")).pack(anchor="w")
        right = Frame(header, bg=BG); right.pack(side="right", fill="y")
        Label(right, text="VISUAL STRATEGY ENGINE", bg=PANEL_2, fg=METAL, font=_safe_font(8, "bold"), padx=12, pady=5).pack(anchor="e", pady=(17, 0))
        Label(right, text="DATA  •  STRATEGY  •  RISK  •  SIMULATION", bg=BG, fg=TEXT_DIM, font=_safe_font(7)).pack(anchor="e", pady=(7, 0))
        Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=18, pady=(0, 12))

    def _page_header(self, parent, eyebrow, title, description=""):
        box = Frame(parent, bg=BG); box.pack(fill="x", padx=24, pady=(20, 16))
        Label(box, text=eyebrow.upper(), bg=BG, fg=METAL, font=_safe_font(8, "bold")).pack(anchor="w")
        Label(box, text=title, bg=BG, fg=METAL_BRIGHT, font=_safe_font(20, "bold")).pack(anchor="w", pady=(3, 3))
        if description:
            Label(box, text=description, bg=BG, fg=TEXT_MUTED, font=_safe_font(9), wraplength=1100, justify="left").pack(anchor="w")
        Frame(box, bg=BORDER, height=1).pack(fill="x", pady=(13, 0))

    def _button(self, parent, text, command, primary=False, width=None):
        kw = dict(text=text, command=command, font=_safe_font(9, "bold"), relief="flat", bd=0,
                  cursor="hand2", padx=14, pady=7)
        if primary:
            kw.update(bg=METAL_BRIGHT, fg=BG, activebackground=METAL, activeforeground=BG)
        else:
            kw.update(bg=PANEL_3, fg=TEXT, activebackground=BORDER_LIGHT, activeforeground=METAL_BRIGHT)
        if width: kw["width"] = width
        return Button(parent, **kw)

    def _section(self, parent, title, subtitle=""):
        box = Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="x", padx=24, pady=7)
        Label(box, text=title.upper(), bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(anchor="w", padx=18, pady=(13, 2))
        if subtitle:
            Label(box, text=subtitle, bg=PANEL, fg=TEXT_DIM, font=_safe_font(8)).pack(anchor="w", padx=18, pady=(0, 8))
        return box

    # --------------------------------------------------------------- data tab
    def _build_data_tab(self):
        f = self.tab_data
        self._page_header(f, "01 / Market Data", "Market Data", "Select historical OHLCV data. Imported CSVs are stored locally and rediscovered automatically.")
        section = self._section(f, "Available datasets", "DATA/RAW • Automatically discovered at startup")
        list_frame = Frame(section, bg=PANEL); list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 12))
        self.dataset_listbox = Listbox(list_frame, height=9, selectmode=SINGLE, exportselection=False, bg=PANEL_3, fg=TEXT,
                                       selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT, activestyle="none",
                                       relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9))
        self.dataset_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.dataset_listbox.yview, style="T58.Vertical.TScrollbar")
        scrollbar.pack(side="right", fill="y"); self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selected)
        btn_row = Frame(section, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._button(btn_row, "IMPORT CSV(S)", self._browse_csv, primary=True).pack(side="left")
        self._button(btn_row, "REFRESH LIST", self._refresh_dataset_list).pack(side="left", padx=8)
        self.data_status = Label(f, text="●  No dataset selected.", bg=BG, fg=TEXT_MUTED, font=_safe_font(9)); self.data_status.pack(anchor="w", padx=26, pady=(8, 2))
        Label(f, text="Tip: place CSV files directly in data/raw/ and press REFRESH LIST.", bg=BG, fg=TEXT_DIM, font=_safe_font(8)).pack(anchor="w", padx=26)
        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END); self._stored_datasets = list_stored_datasets()
        for ds in self._stored_datasets: self.dataset_listbox.insert(END, f"  {ds.name}")
        if self._stored_datasets and not self.csv_path:
            self.dataset_listbox.selection_set(0); self._select_dataset(self._stored_datasets[0].path)

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        if sel: self._select_dataset(self._stored_datasets[sel[0]].path)

    def _select_dataset(self, path: Path):
        result = import_csv(path)
        if not result.is_valid:
            messagebox.showerror("Import failed", "\n".join(result.errors)); self.data_status.config(text=f"●  {path.name}: import failed.", fg=RED); return
        self.csv_path = str(path)
        n = len(result.dataframe); warn = f"  •  {len(result.warnings)} warning(s)" if result.warnings else ""
        self.data_status.config(text=f"●  ACTIVE  {path.name}  •  {n:,} bars{warn}", fg=GREEN)
        if hasattr(self, "instrument_var") and not self.instrument_var.get(): self.instrument_var.set(path.stem.rstrip("_"))

    def _browse_csv(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not paths: return
        imported, failed = [], []
        for p in paths:
            result = import_csv(p)
            if not result.is_valid: failed.append((os.path.basename(p), "; ".join(result.errors))); continue
            imported.append(store_csv_path(p))
        self._refresh_dataset_list()
        if imported:
            self._select_dataset(imported[-1])
            for i, ds in enumerate(self._stored_datasets):
                if ds.path == imported[-1]: self.dataset_listbox.selection_clear(0, END); self.dataset_listbox.selection_set(i); break
        if failed:
            detail = "\n".join(f"- {name}: {err}" for name, err in failed)
            messagebox.showwarning("Some files failed to import", f"{len(imported)} imported.\n\n{len(failed)} failed:\n{detail}")
        elif imported: messagebox.showinfo("Import complete", f"Imported and stored {len(imported)} file(s) in data/raw/.")

    # ----------------------------------------------------------- strategy tab
    def _build_strategy_tab(self):
        f = self.tab_strategy
        self._page_header(f, "02 / Strategy", "Manual Strategy Builder", "Build a complete rule-based strategy visually. Conditions are translated into a structured configuration consumed by the Manual Strategy engine.")

        # Strategy source controls remain available.
        source = self._section(f, "Strategy source", "MANUAL opens the visual builder; external modes preserve the original file-import workflow.")
        modes = Frame(source, bg=PANEL); modes.pack(anchor="w", padx=18, pady=(2, 8))
        for val, text in [("manual", "MANUAL BUILDER"), ("python", "PYTHON"), ("pinescript", "PINESCRIPT"), ("mql5", "MQL5")]:
            self._button(modes, text, lambda v=val: self._set_strategy_mode(v), primary=(val == "manual")).pack(side="left", padx=(0, 7))
        self.strategy_mode_label = Label(source, text="SELECTED • MANUAL BUILDER", bg=PANEL, fg=GREEN, font=_safe_font(9, "bold")); self.strategy_mode_label.pack(anchor="w", padx=18, pady=(4, 9))
        self._button(source, "BROWSE STRATEGY FILE", self._browse_strategy_file).pack(anchor="w", padx=18, pady=(0, 5))
        self.strategy_file_status = Label(source, text="Only needed for Python / PineScript / MQL5 modes.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8)); self.strategy_file_status.pack(anchor="w", padx=18, pady=(0, 14))

        # Scrollable builder canvas.
        outer = Frame(f, bg=BG); outer.pack(fill="both", expand=True, padx=24, pady=7)
        canvas = Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=scrollbar.set); canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        builder = Frame(canvas, bg=BG); window_id = canvas.create_window((0, 0), window=builder, anchor="nw")
        builder.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self.builder_frame = builder

        info = self._section(builder, "Strategy information", "Identity and documentation for the strategy.")
        self.s_name = LabeledEntry(info, "Strategy Name", DEFAULT_MANUAL_STRATEGY["name"])
        self.s_description = LabeledEntry(info, "Description", DEFAULT_MANUAL_STRATEGY["description"])
        self.s_author = LabeledEntry(info, "Author", "")
        self.s_version = LabeledEntry(info, "Version", "1.0")

        market = self._section(builder, "Market", "Instrument, timeframe, session and direction restrictions.")
        self.instrument_var = StringVar(value="")
        self.timeframe_var = StringVar(value="5m")
        self.session_var = StringVar(value="All")
        self.direction_var = StringVar(value="Both")
        self._market_row(market, "Instrument", self.instrument_var, ["", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD", "USA500IDXUSD", "USATECHIDXUSD"])
        self._market_row(market, "Timeframe", self.timeframe_var, ["1m", "5m", "15m", "30m", "1h", "4h", "1D"])
        self._market_row(market, "Trading Session", self.session_var, ["All", "Asia", "London", "New York", "Custom"])
        self._market_row(market, "Long / Short / Both", self.direction_var, ["Long", "Short", "Both"])
        self.custom_session_start = LabeledEntry(market, "Custom session start (HH:MM)", "08:30")
        self.custom_session_end = LabeledEntry(market, "Custom session end (HH:MM)", "15:00")

        self.entry_long_rows = []
        self.entry_short_rows = []
        self.exit_long_rows = []
        self.exit_short_rows = []
        self.entry_long_section = self._condition_section(builder, "LONG ENTRY", "IF all/any conditions below are true → ENTER LONG", self.entry_long_rows)
        self.entry_short_section = self._condition_section(builder, "SHORT ENTRY", "IF all/any conditions below are true → ENTER SHORT", self.entry_short_rows)
        self.exit_long_section = self._condition_section(builder, "LONG EXIT", "Indicator/structure exit conditions for an open long.", self.exit_long_rows)
        self.exit_short_section = self._condition_section(builder, "SHORT EXIT", "Indicator/structure exit conditions for an open short.", self.exit_short_rows)

        exits = self._section(builder, "Exit & trade management", "Static stops/targets are executed by the existing backtest engine. Other controls are represented in the manual strategy configuration and supported where the current engine exposes signal-level exits.")
        self.stop_type = StringVar(value="Fixed Pips")
        self.stop_value = StringVar(value="20")
        self.target_type = StringVar(value="Fixed Pips")
        self.target_value = StringVar(value="40")
        self.trailing_enabled = BooleanVar(value=False)
        self.trailing_value = StringVar(value="1.0")
        self.break_even_enabled = BooleanVar(value=False)
        self.break_even_r = StringVar(value="1.0")
        self.max_bars = StringVar(value="0")
        self.time_exit_enabled = BooleanVar(value=False)
        self.time_exit = StringVar(value="16:00")
        self.opposite_exit = BooleanVar(value=True)
        self._management_row(exits, "Stop Loss Type", self.stop_type, ["Fixed Pips", "ATR"])
        self._management_entry(exits, "Stop Loss Value", self.stop_value)
        self._management_row(exits, "Take Profit Type", self.target_type, ["Fixed Pips", "ATR"])
        self._management_entry(exits, "Take Profit Value", self.target_value)
        self._check_row(exits, "Trailing Stop", self.trailing_enabled, self.trailing_value, "Distance / ATR")
        self._check_row(exits, "Move Stop to Break Even", self.break_even_enabled, self.break_even_r, "After +R")
        self._management_entry(exits, "Maximum Bars in Trade (0 = off)", self.max_bars)
        self._check_row(exits, "Time-Based Exit", self.time_exit_enabled, self.time_exit, "Exit time HH:MM")
        chk = ttk.Checkbutton(exits, text="Opposite Signal Exit", variable=self.opposite_exit, style="T58.TCheckbutton"); chk.pack(anchor="w", padx=18, pady=6)

        actions = Frame(builder, bg=BG); actions.pack(fill="x", padx=24, pady=14)
        self._button(actions, "ADD LONG CONDITION", lambda: self._add_condition(self.entry_long_rows, self.entry_long_section), primary=True).pack(side="left")
        self._button(actions, "SAVE STRATEGY JSON", self._save_strategy_json).pack(side="left", padx=7)
        self._button(actions, "LOAD STRATEGY JSON", self._load_strategy_json).pack(side="left", padx=7)
        self._button(actions, "PREVIEW CONFIG", self._preview_strategy).pack(side="left", padx=7)
        self.strategy_summary = Label(builder, text="", bg=BG, fg=TEXT_DIM, font=_safe_font(8), justify="left", wraplength=1100); self.strategy_summary.pack(anchor="w", padx=24, pady=(0, 20))

        # Defaults requested by the product spec.
        self._add_condition(self.entry_long_rows, self.entry_long_section, {"source": "Price", "field": "close", "operator": "Greater Than", "right_source": "EMA", "right_period": 50})
        self._add_condition(self.entry_long_rows, self.entry_long_section, {"source": "RSI", "period": 14, "operator": "Greater Than", "right_source": "Value", "right_value": 55})
        self._add_condition(self.entry_long_rows, self.entry_long_section, {"source": "Volume", "operator": "Greater Than", "right_source": "Average Volume", "right_period": 20})
        self._add_condition(self.entry_short_rows, self.entry_short_section, {"source": "Price", "field": "close", "operator": "Less Than", "right_source": "EMA", "right_period": 50})
        self._add_condition(self.entry_short_rows, self.entry_short_section, {"source": "RSI", "period": 14, "operator": "Less Than", "right_source": "Value", "right_value": 45})
        self._add_condition(self.entry_short_rows, self.entry_short_section, {"source": "Volume", "operator": "Greater Than", "right_source": "Average Volume", "right_period": 20})
        self._add_condition(self.exit_long_rows, self.exit_long_section, {"source": "Price", "field": "close", "operator": "Less Than", "right_source": "EMA", "right_period": 50})
        self._add_condition(self.exit_short_rows, self.exit_short_section, {"source": "Price", "field": "close", "operator": "Greater Than", "right_source": "EMA", "right_period": 50})

    def _market_row(self, parent, label, var, values):
        row = Frame(parent, bg=PANEL); row.pack(fill="x", padx=18, pady=3)
        Label(row, text=label, width=31, anchor="w", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
        ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=24, style="T58.TCombobox").pack(side="left", padx=4, ipady=3)

    def _condition_section(self, parent, title, subtitle, rows):
        box = self._section(parent, title, subtitle)
        Label(box, text="LEFT SOURCE / FIELD / PERIOD    OPERATOR    RIGHT SOURCE / VALUE / PERIOD", bg=PANEL, fg=TEXT_DIM, font=(MONO, 7)).pack(anchor="w", padx=20, pady=(0, 2))
        connector = Frame(box, bg=PANEL); connector.pack(fill="x", padx=18, pady=2)
        var = StringVar(value="AND")
        setattr(box, "connector_var", var)
        Label(connector, text="Join conditions with", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8)).pack(side="left")
        ttk.Combobox(connector, textvariable=var, values=["AND", "OR"], state="readonly", width=8, style="T58.TCombobox").pack(side="left", padx=6)
        btn = self._button(connector, "+ ADD CONDITION", lambda: self._add_condition(rows, box), primary=False)
        btn.pack(side="left", padx=6)
        return box

    def _add_condition(self, rows, section, defaults=None):
        row = ConditionRow(section, lambda r: self._remove_condition(rows, r), defaults)
        rows.append(row)

    def _remove_condition(self, rows, row):
        if len(rows) <= 1:
            messagebox.showinfo("Condition required", "Keep at least one condition in this group."); return
        rows.remove(row); row.destroy()

    def _management_row(self, parent, label, var, values):
        row = Frame(parent, bg=PANEL); row.pack(fill="x", padx=18, pady=3)
        Label(row, text=label, width=31, anchor="w", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
        ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=20, style="T58.TCombobox").pack(side="left", padx=4)

    def _management_entry(self, parent, label, var):
        row = Frame(parent, bg=PANEL); row.pack(fill="x", padx=18, pady=3)
        Label(row, text=label, width=31, anchor="w", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
        Entry(row, textvariable=var, width=20, bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat", font=_safe_font(9)).pack(side="left", padx=4, ipady=4)

    def _check_row(self, parent, label, boolean_var, value_var, value_label):
        row = Frame(parent, bg=PANEL); row.pack(fill="x", padx=18, pady=3)
        ttk.Checkbutton(row, text=label, variable=boolean_var, style="T58.TCheckbutton").pack(side="left", padx=(0, 12))
        Label(row, text=value_label, bg=PANEL, fg=TEXT_DIM, font=_safe_font(8)).pack(side="left")
        Entry(row, textvariable=value_var, width=12, bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat", font=_safe_font(9)).pack(side="left", padx=5, ipady=4)

    def _set_strategy_mode(self, mode: str):
        self.strategy_mode.set(mode)
        display = {"manual": "MANUAL BUILDER", "python": "PYTHON STRATEGY", "pinescript": "PINESCRIPT STRATEGY", "mql5": "MQL5 STRATEGY"}.get(mode, mode.upper())
        self.strategy_mode_label.config(text=f"SELECTED • {display}", fg=GREEN if mode == "manual" else METAL)

    def _browse_strategy_file(self):
        ext = {"python": "*.py", "pinescript": "*.pine", "mql5": "*.mq5"}.get(self.strategy_mode.get(), "*.*")
        path = filedialog.askopenfilename(filetypes=[("Strategy file", ext)])
        if path:
            self.strategy_py_path = path; self.strategy_file_status.config(text=f"Selected: {os.path.basename(path)}", fg=GREEN)

    def _collect_conditions(self, rows):
        return [row.to_config() for row in rows]

    def _visual_config(self):
        direction = self.direction_var.get()
        stop_type = self.stop_type.get()
        target_type = self.target_type.get()
        try:
            stop_value = float(self.stop_value.get() or 0)
            target_value = float(self.target_value.get() or 0)
            trailing_value = float(self.trailing_value.get() or 0)
            break_even_r = float(self.break_even_r.get() or 0)
            max_bars = int(float(self.max_bars.get() or 0))
        except ValueError as exc:
            raise StrategyError("Stop, target, trailing, break-even and max-bars values must be numeric.") from exc

        long_entries = self._collect_conditions(self.entry_long_rows)
        short_entries = self._collect_conditions(self.entry_short_rows)
        long_exits = self._collect_conditions(self.exit_long_rows)
        short_exits = self._collect_conditions(self.exit_short_rows)
        cfg = {
            "name": self.s_name.get_str().strip() or "Untitled Strategy",
            "description": self.s_description.get_str().strip(),
            "author": self.s_author.get_str().strip(),
            "version": self.s_version.get_str().strip() or "1.0",
            "market": {
                "instrument": self.instrument_var.get().strip(),
                "timeframe": self.timeframe_var.get(),
                "session": self.session_var.get(),
                "direction": direction,
                "custom_session_start": self.custom_session_start.get_str().strip() or "08:30",
                "custom_session_end": self.custom_session_end.get_str().strip() or "15:00",
            },
            "entry_conditions": {
                "long": long_entries,
                "long_connectors": [getattr(self.entry_long_section, "connector_var").get()] * max(len(long_entries) - 1, 0),
                "short": short_entries,
                "short_connectors": [getattr(self.entry_short_section, "connector_var").get()] * max(len(short_entries) - 1, 0),
            },
            "exit_conditions": {
                "long": long_exits,
                "long_connectors": [getattr(self.exit_long_section, "connector_var").get()] * max(len(long_exits) - 1, 0),
                "short": short_exits,
                "short_connectors": [getattr(self.exit_short_section, "connector_var").get()] * max(len(short_exits) - 1, 0),
            },
            "risk_management": {
                "stop_type": "fixed" if stop_type == "Fixed Pips" else "atr",
                "stop_value": stop_value,
                "target_type": "fixed" if target_type == "Fixed Pips" else "atr",
                "target_value": target_value,
                "trailing_stop": {"enabled": self.trailing_enabled.get(), "value": trailing_value},
                "break_even": {"enabled": self.break_even_enabled.get(), "after_r": break_even_r},
                "time_based_exit": {"enabled": self.time_exit_enabled.get(), "time": self.time_exit.get().strip()},
                "max_bars_in_trade": max_bars or None,
                "indicator_exit": bool(long_exits or short_exits),
                "opposite_signal_exit": self.opposite_exit.get(),
            },
            # Legacy execution fields. The current execution engine supports
            # fixed pip SL/TP directly; ATR/runners remain in the config for
            # the next execution-engine expansion rather than being silently
            # converted to a fake fixed distance.
            "stop_loss_pips": stop_value if stop_type == "Fixed Pips" and stop_value > 0 else None,
            "take_profit_pips": target_value if target_type == "Fixed Pips" and target_value > 0 else None,
        }
        return cfg

    def _build_strategy(self):
        mode = self.strategy_mode.get()
        if mode == "manual":
            return ManualStrategy(self._visual_config())
        if not self.strategy_py_path:
            raise StrategyError(f"No file selected for '{mode}' strategy mode.")
        if mode == "python": return PythonStrategy(self.strategy_py_path)
        if mode == "pinescript": return PineScriptStrategy(self.strategy_py_path)
        if mode == "mql5": return MQL5Strategy(self.strategy_py_path)
        raise StrategyError(f"Unknown strategy mode: {mode}")

    def _preview_strategy(self):
        try:
            cfg = self._visual_config()
            win = __import__("tkinter").Toplevel(self.root); win.title("T58 Strategy Preview"); win.geometry("900x700"); win.configure(bg=BG)
            text = Text(win, bg="#0B0D10", fg=TEXT, insertbackground=TEXT, font=(MONO, 9), relief="flat")
            text.pack(fill="both", expand=True, padx=16, pady=16); text.insert("1.0", json.dumps(cfg, indent=2)); text.config(state="disabled")
        except Exception as exc:
            messagebox.showerror("Invalid strategy", str(exc))

    def _save_strategy_json(self):
        try: cfg = self._visual_config()
        except Exception as exc: messagebox.showerror("Invalid strategy", str(exc)); return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Strategy JSON", "*.json")], initialfile=f"{cfg['name'].replace(' ', '_')}.json")
        if path:
            Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8"); messagebox.showinfo("Strategy saved", f"Saved strategy to:\n{path}")

    def _load_strategy_json(self):
        path = filedialog.askopenfilename(filetypes=[("Strategy JSON", "*.json")])
        if not path: return
        try: cfg = json.loads(Path(path).read_text(encoding="utf-8")); self._load_visual_config(cfg)
        except Exception as exc: messagebox.showerror("Load failed", str(exc))

    def _load_visual_config(self, cfg):
        # Metadata/market are easy to restore. Conditions are rebuilt as rows.
        self.s_name.var.set(cfg.get("name", "Untitled Strategy")); self.s_description.var.set(cfg.get("description", "")); self.s_author.var.set(cfg.get("author", "")); self.s_version.var.set(cfg.get("version", "1.0"))
        market = cfg.get("market", {}); self.instrument_var.set(market.get("instrument", "")); self.timeframe_var.set(market.get("timeframe", "5m")); self.session_var.set(market.get("session", "All")); self.direction_var.set(market.get("direction", "Both"))
        entries = cfg.get("entry_conditions", {}); exits = cfg.get("exit_conditions", {})
        for rows, data, section in [(self.entry_long_rows, entries.get("long", []), self.entry_long_section), (self.entry_short_rows, entries.get("short", []), self.entry_short_section), (self.exit_long_rows, exits.get("long", []), self.exit_long_section), (self.exit_short_rows, exits.get("short", []), self.exit_short_section)]:
            for r in rows: r.destroy()
            rows.clear()
            for cond in data: self._add_condition_from_config(rows, section, cond)
            if not rows: self._add_condition(rows, section)
        self._set_connector(self.entry_long_section, entries.get("long_connectors", ["AND"])[0] if entries.get("long_connectors") else "AND")
        self._set_connector(self.entry_short_section, entries.get("short_connectors", ["AND"])[0] if entries.get("short_connectors") else "AND")
        self._set_connector(self.exit_long_section, exits.get("long_connectors", ["AND"])[0] if exits.get("long_connectors") else "AND")
        self._set_connector(self.exit_short_section, exits.get("short_connectors", ["AND"])[0] if exits.get("short_connectors") else "AND")
        rm = cfg.get("risk_management", {}); self.stop_value.set(str(rm.get("stop_value", cfg.get("stop_loss_pips") or 20))); self.target_value.set(str(rm.get("target_value", cfg.get("take_profit_pips") or 40)))
        self.stop_type.set("Fixed Pips" if rm.get("stop_type", "fixed") == "fixed" else "ATR"); self.target_type.set("Fixed Pips" if rm.get("target_type", "fixed") == "fixed" else "ATR")
        self.max_bars.set(str(rm.get("max_bars_in_trade") or 0)); self.opposite_exit.set(bool(rm.get("opposite_signal_exit", True)))

    def _add_condition_from_config(self, rows, section, cond):
        left = cond.get("left", {}); right = cond.get("right", {})
        kind_to_name = {v: k for k, v in SOURCE_KIND.items()}
        defaults = {"source": kind_to_name.get(left.get("type"), "Price"), "field": left.get("field", "close"), "period": left.get("period", 14), "operator": cond.get("operator", "Greater Than"), "right_source": kind_to_name.get(right.get("type"), "Value"), "right_value": right.get("value", 0), "right_period": right.get("period", 14), "direction": left.get("direction", "Both"), "lookback": left.get("lookback", 20), "session_start": left.get("session_start", "08:30"), "session_end": left.get("session_end", "15:00")}
        self._add_condition(rows, section, defaults)

    @staticmethod
    def _set_connector(section, value):
        if hasattr(section, "connector_var"): section.connector_var.set(value)

    # --------------------------------------------------------------- prop tab
    def _build_prop_tab(self):
        f = self.tab_prop; self._page_header(f, "03 / Prop Firm Rules", "Prop-Firm Rules", "Define evaluation, drawdown, consistency, payout and position constraints.")
        section = self._section(f, "Account & evaluation", "Core evaluation parameters.")
        self.p_account_size = LabeledEntry(section, "Account size ($)", 100000); self.p_profit_target = LabeledEntry(section, "Evaluation profit target (%)", 8); self.p_daily_loss = LabeledEntry(section, "Daily loss limit (%)", 5); self.p_max_dd = LabeledEntry(section, "Maximum drawdown (%)", 10); self.p_dd_type = LabeledEntry(section, "Drawdown type (trailing/static)", "trailing"); self.p_consistency = LabeledEntry(section, "Consistency rule (% best day of total profit)", 30); self.p_min_days = LabeledEntry(section, "Minimum trading days", 5)
        section2 = self._section(f, "Payout & position rules", "Optional payout and position constraints.")
        self.p_payout_threshold = LabeledEntry(section2, "Payout threshold (extra % profit)", 0); self.p_payout_cap = LabeledEntry(section2, "Payout cap (% of profit, blank=100)", 100); self.p_payout_freq = LabeledEntry(section2, "Payout frequency (days)", 14); self.p_buffer = LabeledEntry(section2, "Required buffer (%)", 0); self.p_max_pos = LabeledEntry(section2, "Max position size (units, blank=unlimited)", "")

    def _build_prop_rules(self) -> PropRules:
        cap = self.p_payout_cap.get_str().strip(); max_pos = self.p_max_pos.get_str().strip()
        return PropRules(account_size=self.p_account_size.get_float(100000), evaluation_profit_target_pct=self.p_profit_target.get_float(8), daily_loss_limit_pct=self.p_daily_loss.get_float(5), max_drawdown_pct=self.p_max_dd.get_float(10), drawdown_type=self.p_dd_type.get_str().strip() or "trailing", consistency_rule_pct=self.p_consistency.get_float(30), min_trading_days=self.p_min_days.get_int(5), payout_threshold_pct=self.p_payout_threshold.get_float(0), payout_cap_pct=float(cap) if cap else None, payout_frequency_days=self.p_payout_freq.get_int(14), required_buffer_pct=self.p_buffer.get_float(0), max_position_size=float(max_pos) if max_pos else None)

    # --------------------------------------------------------------- risk tab
    def _build_risk_tab(self):
        f = self.tab_risk; self._page_header(f, "04 / Risk & Execution", "Risk & Execution", "Define position risk, trading frequency, transaction costs and execution assumptions.")
        section = self._section(f, "Risk configuration", "These parameters are passed directly into the backtest engine.")
        self.r_initial_balance = LabeledEntry(section, "Initial balance ($)", 100000); self.r_risk_mode = LabeledEntry(section, "Risk mode (percent/fixed)", "percent"); self.r_risk_value = LabeledEntry(section, "Risk per trade (% or $)", 1.0); self.r_max_trades_day = LabeledEntry(section, "Max trades/day", 10); self.r_commission = LabeledEntry(section, "Commission per trade ($)", 0); self.r_slippage = LabeledEntry(section, "Slippage (pips)", 0.5); self.r_spread = LabeledEntry(section, "Spread (pips)", 1.0); self.r_pip_size = LabeledEntry(section, "Pip size (e.g. 0.0001 FX)", 0.0001)

    def _build_risk_config(self) -> RiskConfig:
        return RiskConfig(initial_balance=self.r_initial_balance.get_float(100000), risk_mode=self.r_risk_mode.get_str().strip() or "percent", risk_value=self.r_risk_value.get_float(1.0), max_trades_per_day=self.r_max_trades_day.get_int(10), commission_per_trade=self.r_commission.get_float(0), slippage_pips=self.r_slippage.get_float(0.5), spread_pips=self.r_spread.get_float(1.0), pip_size=self.r_pip_size.get_float(0.0001))

    # --------------------------------------------------------------- run tab
    def _build_run_tab(self):
        f = self.tab_run; self._page_header(f, "05 / Run & Report", "Run Full Pipeline", "Backtest → Prop Simulation → Monte Carlo → Report.")
        section = self._section(f, "Simulation", "Configure Monte Carlo before starting.")
        self.mc_sims = LabeledEntry(section, "Monte Carlo simulations", 10000); self.mc_method = LabeledEntry(section, "Method (bootstrap/shuffle/block_bootstrap)", "bootstrap")
        button_row = Frame(f, bg=BG); button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN FULL PIPELINE", self._run_clicked, primary=True).pack(side="left")
        self.open_report_btn = self._button(button_row, "OPEN HTML REPORT", self._open_report); self.open_report_btn.config(state="disabled"); self.open_report_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(f, mode="indeterminate", style="T58.Horizontal.TProgressbar"); self.progress.pack(fill="x", padx=24, pady=(2, 10))
        output_section = self._section(f, "Pipeline output", "Live execution log.")
        self.output = Text(output_section, height=18, wrap="word", bg="#0B0D10", fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9)); self.output.pack(fill="both", expand=True, padx=18, pady=(3, 16)); self._last_html_path: Path | None = None

    def _log(self, msg: str):
        self.output.insert(END, msg + "\n"); self.output.see(END); self.root.update_idletasks()

    def _run_clicked(self):
        if not self.csv_path: messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1."); return
        self.output.delete("1.0", END); self.progress.start(10); threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _open_report(self):
        if self._last_html_path: webbrowser.open(f"file://{self._last_html_path.resolve()}")

    def _run_pipeline(self):
        try:
            self._log("Importing market data..."); import_result = import_csv(self.csv_path)
            if not import_result.is_valid: self._log("Import errors:\n" + "\n".join(import_result.errors)); return
            df = import_result.dataframe
            for w in import_result.warnings: self._log(f"  [warning] {w}")
            self._log(f"Loaded {len(df)} bars.")
            self._log("Building strategy..."); strategy = self._build_strategy()
            self._log("Configuring risk & prop rules..."); risk = self._build_risk_config(); rules = self._build_prop_rules()
            self._log("Running historical backtest..."); bt_result = run_backtest(df, strategy, risk)
            self._log(f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}  Win rate: {bt_result.statistics.win_rate:.1f}%  Max DD: {bt_result.statistics.max_drawdown_pct:.2f}%")
            self._log("Running prop-firm simulation on historical sequence..."); trade_pnls = [t.pnl for t in bt_result.trades]; trade_dates = [t.entry_time for t in bt_result.trades]; single_run = simulate_account(trade_pnls, trade_dates, rules)
            self._log(f"  Passed evaluation: {single_run.passed_evaluation}  Reached payout: {single_run.reached_first_payout}  Failed: {single_run.failed} ({single_run.failure_reason})")
            n_sims = self.mc_sims.get_int(10000); method = self.mc_method.get_str().strip() or "bootstrap"; self._log(f"Running Monte Carlo simulation ({n_sims:,} runs, method={method})...")
            mc_result = run_monte_carlo(bt_result.trades, rules, MonteCarloConfig(n_simulations=n_sims, method=method))
            self._log(f"  Evaluation pass probability: {mc_result.evaluation_pass_probability:.1f}%"); self._log(f"  First payout probability: {mc_result.first_payout_probability:.1f}%"); self._log(f"  Expected payout: ${mc_result.expected_payout:,.2f}"); self._log(f"  Risk of ruin: {mc_result.risk_of_ruin_pct:.1f}%")
            self._log("Generating report..."); period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
            paths = generate_full_report(output_dir=OUTPUT_DIR, strategy_name=bt_result.strategy_name, strategy_source_type=strategy.source_type, instrument=os.path.basename(self.csv_path), timeframe=self.timeframe_var.get() if hasattr(self, "timeframe_var") else "unknown", backtest_period=period, backtest_result=bt_result, prop_rules=rules, prop_single_run=single_run, monte_carlo_result=mc_result)
            self._last_html_path = paths["html"]; self.open_report_btn.config(state="normal"); self._log("\nDone. Report written to:")
            for k, p in paths.items(): self._log(f"  {k}: {p}")
        except StrategyError as exc: self._log(f"\nStrategy error: {exc}")
        except Exception: self._log("\nUnexpected error:\n" + traceback.format_exc())
        finally: self.progress.stop()


def launch():
    root = Tk(); MainWindow(root); root.mainloop()
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


# ---------------------------------------------------------------------------
# T58 visual system
# ---------------------------------------------------------------------------

BG = "#080A0D"
PANEL = "#101318"
PANEL_2 = "#15191F"
PANEL_3 = "#1B2027"
BORDER = "#292E36"
BORDER_LIGHT = "#3A414B"

TEXT = "#E5E7EB"
TEXT_MUTED = "#8B929D"
TEXT_DIM = "#626A75"
METAL = "#B8BDC5"
METAL_BRIGHT = "#E1E4E8"

GREEN = "#43D17A"
RED = "#F05B63"
BLUE = "#6FA8FF"

FONT = "Segoe UI"
MONO = "Consolas"


def _safe_font(size=10, weight="normal"):
    return (FONT, size, weight)


class LabeledEntry(Frame):
    def __init__(self, parent, label, default=""):
        super().__init__(parent, bg=PANEL)
        self.configure(height=36)

        Label(
            self,
            text=label,
            width=31,
            anchor="w",
            bg=PANEL,
            fg=TEXT_MUTED,
            font=_safe_font(9),
        ).pack(side="left")

        self.var = StringVar(value=str(default))

        self.entry = Entry(
            self,
            textvariable=self.var,
            width=20,
            bg=PANEL_3,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER_LIGHT,
            font=_safe_font(10),
        )
        self.entry.pack(side="left", ipady=5, padx=(4, 0))

        self.pack(fill="x", pady=3, padx=18)

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
        self.root.title("T58 Trading — Prop Algo Backtester")
        self.root.geometry("1000x760")
        self.root.minsize(900, 680)
        self.root.configure(bg=BG)

        self.csv_path: str | None = None
        self.strategy_py_path: str | None = None
        self.strategy_mode = StringVar(value="manual")

        self._configure_styles()

        # Main application shell.
        shell = Frame(root, bg=BG)
        shell.pack(fill="both", expand=True)

        self._build_header(shell)

        self.nb = ttk.Notebook(shell, style="T58.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.tab_data = Frame(self.nb, bg=BG)
        self.tab_strategy = Frame(self.nb, bg=BG)
        self.tab_prop = Frame(self.nb, bg=BG)
        self.tab_risk = Frame(self.nb, bg=BG)
        self.tab_run = Frame(self.nb, bg=BG)

        self.nb.add(self.tab_data, text="  01  DATA  ")
        self.nb.add(self.tab_strategy, text="  02  STRATEGY  ")
        self.nb.add(self.tab_prop, text="  03  PROP RULES  ")
        self.nb.add(self.tab_risk, text="  04  RISK  ")
        self.nb.add(self.tab_run, text="  05  RUN & REPORT  ")

        self._build_data_tab()
        self._build_strategy_tab()
        self._build_prop_tab()
        self._build_risk_tab()
        self._build_run_tab()

    # -----------------------------------------------------------------------
    # Styling / shell
    # -----------------------------------------------------------------------

    def _configure_styles(self):
        style = ttk.Style(self.root)

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "T58.TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            "T58.TNotebook.Tab",
            background=PANEL,
            foreground=TEXT_MUTED,
            padding=[18, 10],
            borderwidth=0,
            font=_safe_font(9, "bold"),
        )
        style.map(
            "T58.TNotebook.Tab",
            background=[
                ("selected", PANEL_2),
                ("active", PANEL_3),
            ],
            foreground=[
                ("selected", METAL_BRIGHT),
                ("active", TEXT),
            ],
        )

        style.configure(
            "T58.Vertical.TScrollbar",
            background=PANEL_2,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=TEXT_DIM,
        )

        style.configure(
            "T58.Horizontal.TProgressbar",
            background=METAL,
            troughcolor=PANEL_3,
            bordercolor=PANEL_3,
            lightcolor=METAL,
            darkcolor=METAL,
        )

    def _build_header(self, parent):
        header = Frame(parent, bg=BG, height=92)
        header.pack(fill="x", padx=18, pady=(16, 8))
        header.pack_propagate(False)

        # Text-based mark keeps the executable independent of a required
        # image path. Existing logo PNGs can still be added to this header.
        mark = Frame(header, bg=BG)
        mark.pack(side="left", fill="y")

        Label(
            mark,
            text="T58",
            bg=BG,
            fg=METAL_BRIGHT,
            font=_safe_font(32, "bold"),
        ).pack(anchor="w")

        Label(
            mark,
            text="PROP ALGO BACKTESTER",
            bg=BG,
            fg=TEXT_MUTED,
            font=_safe_font(8, "bold"),
        ).pack(anchor="w", pady=(0, 2))

        right = Frame(header, bg=BG)
        right.pack(side="right", fill="y")

        Label(
            right,
            text="MVP",
            bg=PANEL_2,
            fg=METAL,
            font=_safe_font(8, "bold"),
            padx=12,
            pady=5,
        ).pack(anchor="e", pady=(17, 0))

        Label(
            right,
            text="HISTORICAL DATA  •  STRATEGY  •  RISK  •  SIMULATION",
            bg=BG,
            fg=TEXT_DIM,
            font=_safe_font(7),
        ).pack(anchor="e", pady=(7, 0))

        Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=18, pady=(0, 12))

    def _page_header(self, parent, eyebrow, title, description=""):
        box = Frame(parent, bg=BG)
        box.pack(fill="x", padx=24, pady=(20, 16))

        Label(
            box,
            text=eyebrow.upper(),
            bg=BG,
            fg=METAL,
            font=_safe_font(8, "bold"),
        ).pack(anchor="w")

        Label(
            box,
            text=title,
            bg=BG,
            fg=METAL_BRIGHT,
            font=_safe_font(20, "bold"),
        ).pack(anchor="w", pady=(3, 3))

        if description:
            Label(
                box,
                text=description,
                bg=BG,
                fg=TEXT_MUTED,
                font=_safe_font(9),
                wraplength=900,
                justify="left",
            ).pack(anchor="w")

        Frame(box, bg=BORDER, height=1).pack(fill="x", pady=(13, 0))

    def _button(self, parent, text, command, primary=False, width=None):
        kwargs = {
            "text": text,
            "command": command,
            "font": _safe_font(9, "bold"),
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "padx": 14,
            "pady": 7,
        }

        if primary:
            kwargs.update(
                bg=METAL_BRIGHT,
                fg=BG,
                activebackground=METAL,
                activeforeground=BG,
            )
        else:
            kwargs.update(
                bg=PANEL_3,
                fg=TEXT,
                activebackground=BORDER_LIGHT,
                activeforeground=METAL_BRIGHT,
            )

        if width:
            kwargs["width"] = width

        return Button(parent, **kwargs)

    def _section(self, parent, title, subtitle=""):
        box = Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="x", padx=24, pady=7)

        Label(
            box,
            text=title.upper(),
            bg=PANEL,
            fg=METAL,
            font=_safe_font(9, "bold"),
        ).pack(anchor="w", padx=18, pady=(13, 2))

        if subtitle:
            Label(
                box,
                text=subtitle,
                bg=PANEL,
                fg=TEXT_DIM,
                font=_safe_font(8),
            ).pack(anchor="w", padx=18, pady=(0, 8))

        return box

    # -----------------------------------------------------------------------
    # Tab 1 — Market Data
    # -----------------------------------------------------------------------

    def _build_data_tab(self):
        f = self.tab_data

        self._page_header(
            f,
            "01 / Market Data",
            "Market Data",
            "Select historical OHLC data for the backtest. Built-in datasets are "
            "loaded automatically; imported CSVs are stored locally for future sessions.",
        )

        section = self._section(
            f,
            "Available datasets",
            "DATA/RAW • Automatically discovered when the application starts",
        )

        list_frame = Frame(section, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 12))

        self.dataset_listbox = Listbox(
            list_frame,
            height=9,
            selectmode=SINGLE,
            exportselection=False,
            bg=PANEL_3,
            fg=TEXT,
            selectbackground=BORDER_LIGHT,
            selectforeground=METAL_BRIGHT,
            activestyle="none",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.dataset_listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.dataset_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selected)

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(0, 14))

        self._button(
            btn_row, "IMPORT CSV(S)", self._browse_csv, primary=True
        ).pack(side="left")

        self._button(
            btn_row, "REFRESH LIST", self._refresh_dataset_list
        ).pack(side="left", padx=8)

        self.data_status = Label(
            f,
            text="●  No dataset selected.",
            bg=BG,
            fg=TEXT_MUTED,
            font=_safe_font(9),
        )
        self.data_status.pack(anchor="w", padx=26, pady=(8, 2))

        Label(
            f,
            text="Tip: you can also place CSV files directly in data/raw/ and press REFRESH LIST.",
            bg=BG,
            fg=TEXT_DIM,
            font=_safe_font(8),
        ).pack(anchor="w", padx=26)

        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END)
        self._stored_datasets = list_stored_datasets()

        for ds in self._stored_datasets:
            self.dataset_listbox.insert(END, f"  {ds.name}")

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
            self.data_status.config(
                text=f"●  {path.name}: import failed.",
                fg=RED,
            )
            return

        self.csv_path = str(path)
        n = len(result.dataframe)
        warn = f"  •  {len(result.warnings)} warning(s)" if result.warnings else ""

        self.data_status.config(
            text=f"●  ACTIVE  {path.name}  •  {n:,} bars{warn}",
            fg=GREEN,
        )

    def _browse_csv(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("CSV files", "*.csv")]
        )
        if not paths:
            return

        imported, failed = [], []

        for p in paths:
            result = import_csv(p)

            if not result.is_valid:
                failed.append(
                    (os.path.basename(p), "; ".join(result.errors))
                )
                continue

            stored_path = store_csv_path(p)
            imported.append(stored_path)

        self._refresh_dataset_list()

        if imported:
            self._select_dataset(imported[-1])

            for i, ds in enumerate(self._stored_datasets):
                if ds.path == imported[-1]:
                    self.dataset_listbox.selection_clear(0, END)
                    self.dataset_listbox.selection_set(i)
                    break

        if failed:
            detail = "\n".join(
                f"- {name}: {err}" for name, err in failed
            )
            messagebox.showwarning(
                "Some files failed to import",
                f"{len(imported)} file(s) imported successfully.\n\n"
                f"{len(failed)} file(s) failed:\n{detail}",
            )
        elif imported:
            messagebox.showinfo(
                "Import complete",
                f"Imported and stored {len(imported)} file(s) in data/raw/.",
            )

    # -----------------------------------------------------------------------
    # Tab 2 — Strategy
    # -----------------------------------------------------------------------

    def _build_strategy_tab(self):
        f = self.tab_strategy

        self._page_header(
            f,
            "02 / Strategy",
            "Strategy Configuration",
            "Choose a strategy source or build a simple SMA crossover directly in the application.",
        )

        section = self._section(
            f,
            "Strategy source",
            "Select one of the supported strategy formats.",
        )

        modes = Frame(section, bg=PANEL)
        modes.pack(anchor="w", padx=18, pady=(2, 8))

        for val, text in [
            ("manual", "MANUAL"),
            ("python", "PYTHON"),
            ("pinescript", "PINESCRIPT"),
            ("mql5", "MQL5"),
        ]:
            self._button(
                modes,
                text,
                lambda v=val: self._set_strategy_mode(v),
            ).pack(side="left", padx=(0, 7))

        self.strategy_mode_label = Label(
            section,
            text="SELECTED  •  MANUAL — SMA 20/50 CROSS",
            bg=PANEL,
            fg=GREEN,
            font=_safe_font(9, "bold"),
        )
        self.strategy_mode_label.pack(anchor="w", padx=18, pady=(4, 9))

        self._button(
            section,
            "BROWSE STRATEGY FILE",
            self._browse_strategy_file,
        ).pack(anchor="w", padx=18, pady=(0, 5))

        self.strategy_file_status = Label(
            section,
            text="Only needed for Python / PineScript / MQL5 modes.",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
        )
        self.strategy_file_status.pack(anchor="w", padx=18, pady=(0, 14))

        manual = self._section(
            f,
            "Manual builder",
            "Parameters used when strategy mode is MANUAL.",
        )

        self.sma_fast = LabeledEntry(manual, "SMA fast period", 20)
        self.sma_slow = LabeledEntry(manual, "SMA slow period", 50)
        self.sl_pips = LabeledEntry(manual, "Stop loss (pips)", 20)
        self.tp_pips = LabeledEntry(manual, "Take profit (pips)", 40)

    def _set_strategy_mode(self, mode: str):
        self.strategy_mode.set(mode)

        display = {
            "manual": "MANUAL — SMA 20/50 CROSS",
            "python": "PYTHON STRATEGY",
            "pinescript": "PINESCRIPT STRATEGY",
            "mql5": "MQL5 STRATEGY",
        }.get(mode, mode.upper())

        self.strategy_mode_label.config(
            text=f"SELECTED  •  {display}"
        )

    def _browse_strategy_file(self):
        ext = {
            "python": "*.py",
            "pinescript": "*.pine",
            "mql5": "*.mq5",
        }.get(self.strategy_mode.get(), "*.*")

        path = filedialog.askopenfilename(
            filetypes=[("Strategy file", ext)]
        )

        if path:
            self.strategy_py_path = path
            self.strategy_file_status.config(
                text=f"Selected: {os.path.basename(path)}",
                fg=GREEN,
            )

    def _build_strategy(self):
        mode = self.strategy_mode.get()

        if mode == "manual":
            cfg = dict(DEFAULT_MANUAL_STRATEGY)
            cfg["indicators"] = [
                {
                    "type": "sma",
                    "period": self.sma_fast.get_int(20),
                    "column": "close",
                    "as": "sma_fast",
                },
                {
                    "type": "sma",
                    "period": self.sma_slow.get_int(50),
                    "column": "close",
                    "as": "sma_slow",
                },
            ]
            cfg["stop_loss_pips"] = self.sl_pips.get_float(20)
            cfg["take_profit_pips"] = self.tp_pips.get_float(40)
            return ManualStrategy(cfg)

        if not self.strategy_py_path:
            raise StrategyError(
                f"No file selected for '{mode}' strategy mode."
            )

        if mode == "python":
            return PythonStrategy(self.strategy_py_path)
        if mode == "pinescript":
            return PineScriptStrategy(self.strategy_py_path)
        if mode == "mql5":
            return MQL5Strategy(self.strategy_py_path)

        raise StrategyError(f"Unknown strategy mode: {mode}")

    # -----------------------------------------------------------------------
    # Tab 3 — Prop rules
    # -----------------------------------------------------------------------

    def _build_prop_tab(self):
        f = self.tab_prop

        self._page_header(
            f,
            "03 / Prop Firm Rules",
            "Prop-Firm Rules",
            "Define the evaluation, drawdown, consistency, payout, and position constraints.",
        )

        section = self._section(
            f,
            "Account & evaluation",
            "Core evaluation parameters.",
        )

        self.p_account_size = LabeledEntry(section, "Account size ($)", 100000)
        self.p_profit_target = LabeledEntry(
            section, "Evaluation profit target (%)", 8
        )
        self.p_daily_loss = LabeledEntry(
            section, "Daily loss limit (%)", 5
        )
        self.p_max_dd = LabeledEntry(
            section, "Maximum drawdown (%)", 10
        )
        self.p_dd_type = LabeledEntry(
            section, "Drawdown type (trailing/static)", "trailing"
        )
        self.p_consistency = LabeledEntry(
            section, "Consistency rule (% best day of total profit)", 30
        )
        self.p_min_days = LabeledEntry(
            section, "Minimum trading days", 5
        )

        section2 = self._section(
            f,
            "Payout & position rules",
            "Optional payout and position constraints.",
        )

        self.p_payout_threshold = LabeledEntry(
            section2, "Payout threshold (extra % profit)", 0
        )
        self.p_payout_cap = LabeledEntry(
            section2, "Payout cap (% of profit, blank=100)", 100
        )
        self.p_payout_freq = LabeledEntry(
            section2, "Payout frequency (days)", 14
        )
        self.p_buffer = LabeledEntry(
            section2, "Required buffer (%)", 0
        )
        self.p_max_pos = LabeledEntry(
            section2, "Max position size (units, blank=unlimited)", ""
        )

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

    # -----------------------------------------------------------------------
    # Tab 4 — Risk
    # -----------------------------------------------------------------------

    def _build_risk_tab(self):
        f = self.tab_risk

        self._page_header(
            f,
            "04 / Risk & Execution",
            "Risk & Execution",
            "Define position risk, trading frequency, transaction costs, and execution assumptions.",
        )

        section = self._section(
            f,
            "Risk configuration",
            "These parameters are passed directly into the backtest engine.",
        )

        self.r_initial_balance = LabeledEntry(
            section, "Initial balance ($)", 100000
        )
        self.r_risk_mode = LabeledEntry(
            section, "Risk mode (percent/fixed)", "percent"
        )
        self.r_risk_value = LabeledEntry(
            section, "Risk per trade (% or $)", 1.0
        )
        self.r_max_trades_day = LabeledEntry(
            section, "Max trades/day", 10
        )
        self.r_commission = LabeledEntry(
            section, "Commission per trade ($)", 0
        )
        self.r_slippage = LabeledEntry(
            section, "Slippage (pips)", 0.5
        )
        self.r_spread = LabeledEntry(
            section, "Spread (pips)", 1.0
        )
        self.r_pip_size = LabeledEntry(
            section, "Pip size (e.g. 0.0001 FX)", 0.0001
        )

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

    # -----------------------------------------------------------------------
    # Tab 5 — Run & report
    # -----------------------------------------------------------------------

    def _build_run_tab(self):
        f = self.tab_run

        self._page_header(
            f,
            "05 / Run & Report",
            "Run Full Pipeline",
            "Backtest → Prop Simulation → Monte Carlo → Report.",
        )

        section = self._section(
            f,
            "Simulation",
            "Configure the Monte Carlo run before starting.",
        )

        self.mc_sims = LabeledEntry(
            section, "Monte Carlo simulations", 10000
        )
        self.mc_method = LabeledEntry(
            section,
            "Method (bootstrap/shuffle/block_bootstrap)",
            "bootstrap",
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(
            button_row,
            "RUN FULL PIPELINE",
            self._run_clicked,
            primary=True,
        ).pack(side="left")

        self.open_report_btn = self._button(
            button_row,
            "OPEN HTML REPORT",
            self._open_report,
        )
        self.open_report_btn.config(state="disabled")
        self.open_report_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(
            f,
            mode="indeterminate",
            style="T58.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(
            f,
            "Pipeline output",
            "Live execution log.",
        )

        self.output = Text(
            output_section,
            height=18,
            wrap="word",
            bg="#0B0D10",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_html_path: Path | None = None

    def _log(self, msg: str):
        self.output.insert(END, msg + "\n")
        self.output.see(END)
        self.root.update_idletasks()

    def _run_clicked(self):
        if not self.csv_path:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV in Step 1.",
            )
            return

        self.output.delete("1.0", END)
        self.progress.start(10)
        threading.Thread(
            target=self._run_pipeline,
            daemon=True,
        ).start()

    def _open_report(self):
        if self._last_html_path:
            webbrowser.open(
                f"file://{self._last_html_path.resolve()}"
            )

    def _run_pipeline(self):
        try:
            self._log("Importing market data...")

            import_result = import_csv(self.csv_path)

            if not import_result.is_valid:
                self._log(
                    "Import errors:\n"
                    + "\n".join(import_result.errors)
                )
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

            self._log(
                f"  Trades: {len(bt_result.trades)}  "
                f"Net profit: ${bt_result.statistics.net_profit:,.2f}  "
                f"Win rate: {bt_result.statistics.win_rate:.1f}%  "
                f"Max DD: {bt_result.statistics.max_drawdown_pct:.2f}%"
            )

            self._log(
                "Running prop-firm simulation on historical sequence..."
            )

            trade_pnls = [t.pnl for t in bt_result.trades]
            trade_dates = [t.entry_time for t in bt_result.trades]

            single_run = simulate_account(
                trade_pnls,
                trade_dates,
                rules,
            )

            self._log(
                f"  Passed evaluation: {single_run.passed_evaluation}  "
                f"Reached payout: {single_run.reached_first_payout}  "
                f"Failed: {single_run.failed} "
                f"({single_run.failure_reason})"
            )

            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"

            self._log(
                f"Running Monte Carlo simulation "
                f"({n_sims:,} runs, method={method})..."
            )

            mc_cfg = MonteCarloConfig(
                n_simulations=n_sims,
                method=method,
            )

            mc_result = run_monte_carlo(
                bt_result.trades,
                rules,
                mc_cfg,
            )

            self._log(
                f"  Evaluation pass probability: "
                f"{mc_result.evaluation_pass_probability:.1f}%"
            )
            self._log(
                f"  First payout probability: "
                f"{mc_result.first_payout_probability:.1f}%"
            )
            self._log(
                f"  Expected payout: "
                f"${mc_result.expected_payout:,.2f}"
            )
            self._log(
                f"  Risk of ruin: "
                f"{mc_result.risk_of_ruin_pct:.1f}%"
            )

            self._log("Generating report...")

            period = (
                str(df["timestamp"].iloc[0]),
                str(df["timestamp"].iloc[-1]),
            )

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

        except Exception:
            self._log(
                "\nUnexpected error:\n"
                + traceback.format_exc()
            )

        finally:
            self.progress.stop()


def launch():
    root = Tk()
    MainWindow(root)
    root.mainloop()
