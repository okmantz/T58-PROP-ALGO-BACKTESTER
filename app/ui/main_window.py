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
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, Text, END,
    filedialog, messagebox, ttk, Listbox, SINGLE, EXTENDED, BooleanVar, Canvas,
    Checkbutton, PhotoImage,
)

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.storage import list_stored_datasets, store_csv_path
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.ui.condition_builder import ConditionList

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

# NOTE: the condition-row vocabulary (sources/operators/kind mapping) used
# to live here, but now lives in app.ui.condition_builder alongside the
# widget that uses it, so there's a single source of truth.


def _asset_path(filename: str) -> Path:
    """Resolves a bundled UI asset both in dev mode and inside a
    PyInstaller-frozen .exe (where files added via --add-data land under
    sys._MEIPASS instead of next to this source file)."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "app" / "ui" / "assets" / filename
    return Path(__file__).resolve().parent / "assets" / filename


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


class LabeledCombo(Frame):
    def __init__(self, parent, label, values, default=""):
        super().__init__(parent, bg=PANEL)

        Label(
            self, text=label, width=31, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left")

        self.var = StringVar(value=str(default))
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=values, state="readonly",
            width=18, font=_safe_font(9), style="T58.TCombobox",
        )
        self.combo.pack(side="left", padx=(4, 0))
        self.pack(fill="x", pady=3, padx=18)

    def get_str(self):
        return self.var.get()


class LabeledCheckbox(Frame):
    def __init__(self, parent, label, default=False):
        super().__init__(parent, bg=PANEL)

        self.var = BooleanVar(value=default)
        cb = Checkbutton(
            self, variable=self.var, bg=PANEL, activebackground=PANEL,
            highlightthickness=0, bd=0, selectcolor=PANEL_3,
        )
        cb.pack(side="left")

        Label(
            self, text=label, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left", padx=(2, 0))
        self.pack(fill="x", pady=3, padx=18)

    def get(self) -> bool:
        return bool(self.var.get())


class MainWindow:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("T58 Trading — Prop Algo Backtester")
        try:
            icon_path = _asset_path("t58_mark_medium.png")
            if icon_path.exists():
                self._icon_image = PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._icon_image)
        except Exception:
            pass
        self.root.geometry("1000x760")
        self.root.minsize(900, 680)
        self.root.configure(bg=BG)

        self.csv_path: str | None = None
        self.csv_paths: list[str] = []
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
            "T58.TCombobox",
            fieldbackground=PANEL_3,
            background=PANEL_3,
            foreground=TEXT,
            arrowcolor=TEXT_MUTED,
            bordercolor=BORDER,
            lightcolor=PANEL_3,
            darkcolor=PANEL_3,
            selectbackground=PANEL_3,
            selectforeground=TEXT,
            padding=4,
        )
        style.map(
            "T58.TCombobox",
            fieldbackground=[("readonly", PANEL_3)],
            foreground=[("readonly", TEXT)],
            background=[("readonly", PANEL_3)],
        )
        self.root.option_add("*TCombobox*Listbox.background", PANEL_3)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BORDER_LIGHT)

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

        logo_shown = False
        try:
            logo_path = _asset_path("t58_mark_medium.png")
            if logo_path.exists():
                self._logo_image = PhotoImage(file=str(logo_path))
                Label(mark, image=self._logo_image, bg=BG).pack(anchor="w", pady=(6, 2))
                logo_shown = True
        except Exception:
            logo_shown = False

        if not logo_shown:
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

    def _scrollable(self, parent) -> Frame:
        """Wraps `parent` in a mouse-wheel-scrollable canvas and returns an
        inner Frame to build content into. Used for tabs long enough to
        overflow the window (the Manual Strategy Builder, in particular)."""
        outer = Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        inner = Frame(canvas, bg=BG)

        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _wheel(event):
            delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
            canvas.yview_scroll(int(delta), "units")

        canvas.bind("<Enter>", lambda _e: (
            canvas.bind_all("<MouseWheel>", _wheel),
            canvas.bind_all("<Button-4>", _wheel),
            canvas.bind_all("<Button-5>", _wheel),
        ))
        canvas.bind("<Leave>", lambda _e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>"),
        ))

        return inner

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
            "DATA/RAW • Ctrl/Cmd-click or Shift-click to select more than one timeframe "
            "for multi-timeframe analysis (e.g. 60m for bias, 15m for zone, 5m for entry). "
            "The finest timeframe selected becomes the base/entry timeframe; the others are "
            "merged in as 'tfNN_open/high/low/close/volume' context columns.",
        )

        list_frame = Frame(section, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 12))

        self.dataset_listbox = Listbox(
            list_frame,
            height=9,
            selectmode=EXTENDED,
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

        if self._stored_datasets and not self.csv_paths:
            self.dataset_listbox.selection_set(0)
            self._select_datasets([self._stored_datasets[0].path])

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        if not sel:
            return
        paths = [self._stored_datasets[i].path for i in sel]
        self._select_datasets(paths)

    def _select_datasets(self, paths: list[Path]):
        """Load and validate one or more selected CSVs. Multiple files are
        treated as multiple timeframes for a multi-timeframe backtest."""
        results = []
        for path in paths:
            result = import_csv(path)
            if not result.is_valid:
                messagebox.showerror(
                    "Import failed",
                    f"{path.name}:\n" + "\n".join(result.errors),
                )
                self.data_status.config(text=f"●  {path.name}: import failed.", fg=RED)
                return
            results.append((path, result))

        self.csv_paths = [str(p) for p, _ in results]
        self.csv_path = self.csv_paths[0]

        total_warn = sum(len(r.warnings) for _, r in results)
        warn = f"  •  {total_warn} warning(s)" if total_warn else ""

        if len(results) == 1:
            path, result = results[0]
            n = len(result.dataframe)
            self.data_status.config(
                text=f"●  ACTIVE  {path.name}  •  {n:,} bars{warn}",
                fg=GREEN,
            )
        else:
            _, labels = merge_multi_timeframe([r.dataframe for _, r in results])
            names = ", ".join(p.name for p, _ in results)
            self.data_status.config(
                text=f"●  ACTIVE (multi-timeframe)  {names}  •  {' + '.join(labels)}{warn}",
                fg=GREEN,
            )

    # Backward-compatible alias used elsewhere in this file.
    def _select_dataset(self, path: Path):
        self._select_datasets([path])

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
        f = self._scrollable(self.tab_strategy)

        self._page_header(
            f,
            "02 / Strategy",
            "Strategy Configuration",
            "Build a complete strategy visually — no code required — or bring your own "
            "Python / PineScript / MQL5 file.",
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
            text="SELECTED  •  MANUAL STRATEGY BUILDER",
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

        # ------------------------------------------------------------
        # 24.1  Strategy information
        # ------------------------------------------------------------
        info = self._section(
            f,
            "Strategy information",
            "Descriptive info + the market this strategy is meant to trade. None of this "
            "is required to run a backtest — it's just kept with the strategy for your own records.",
        )
        self.s_name = LabeledEntry(info, "Strategy name", "My Strategy")
        self.s_description = LabeledEntry(info, "Description", "")
        self.s_author = LabeledEntry(info, "Author", "")
        self.s_version = LabeledEntry(info, "Version", "1.0")
        self.s_instrument = LabeledEntry(info, "Instrument", "")
        self.s_timeframe = LabeledEntry(info, "Timeframe", "5m")
        self.s_session_start = LabeledEntry(info, "Session start (HH:MM, 24h)", "08:30")
        self.s_session_end = LabeledEntry(info, "Session end (HH:MM, 24h)", "15:00")
        Label(
            info, text="Session Start/End is also used automatically by any Session High/Low "
                       "or Opening Range condition below.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 8))
        self.s_direction = LabeledCombo(info, "Trade direction", ["Both", "Long", "Short"], "Both")

        # ------------------------------------------------------------
        # 24.2  Entry conditions
        # ------------------------------------------------------------
        entry_section = self._section(
            f,
            "Entry conditions",
            "Build one or more rules using AND / OR. A trade only enters once every "
            "condition in the chain evaluates true. Example: Close > EMA(50) AND RSI(14) > 55.",
        )

        Label(entry_section, text="LONG ENTRY", bg=PANEL, fg=GREEN, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        long_entry_container = Frame(entry_section, bg=PANEL)
        long_entry_container.pack(fill="x", padx=18, pady=(0, 4))
        self.long_entry_conditions = ConditionList(long_entry_container, get_session=self._current_session)
        self._button(entry_section, "+ Add Condition", self.long_entry_conditions.add_row).pack(
            anchor="w", padx=18, pady=(0, 14)
        )

        Label(entry_section, text="SHORT ENTRY", bg=PANEL, fg=RED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        short_entry_container = Frame(entry_section, bg=PANEL)
        short_entry_container.pack(fill="x", padx=18, pady=(0, 4))
        self.short_entry_conditions = ConditionList(short_entry_container, get_session=self._current_session)
        self._button(entry_section, "+ Add Condition", self.short_entry_conditions.add_row).pack(
            anchor="w", padx=18, pady=(0, 14)
        )

        # ------------------------------------------------------------
        # 24.3  Exit conditions / risk management
        # ------------------------------------------------------------
        exit_section = self._section(
            f,
            "Exit conditions",
            "Every field here is optional — leave anything you don't want at its default "
            "(None / unchecked / blank) and it's simply ignored.",
        )

        Label(exit_section, text="STOP LOSS", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        self.stop_type = LabeledCombo(exit_section, "Stop loss type", ["None", "Fixed (pips)", "ATR Multiple"], "Fixed (pips)")
        self.stop_value = LabeledEntry(exit_section, "Stop loss value (pips, or ATR multiple e.g. 1.0)", 20)
        self.stop_atr_period = LabeledEntry(exit_section, "Stop loss ATR period", 14)

        Label(exit_section, text="TAKE PROFIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.target_type = LabeledCombo(exit_section, "Take profit type", ["None", "Fixed (pips)", "ATR Multiple"], "Fixed (pips)")
        self.target_value = LabeledEntry(exit_section, "Take profit value (pips, or ATR multiple e.g. 2.0)", 40)
        self.target_atr_period = LabeledEntry(exit_section, "Take profit ATR period", 14)

        Label(exit_section, text="TRAILING STOP  (ATR-based)", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.trailing_enabled = LabeledCheckbox(exit_section, "Enable trailing stop", False)
        self.trailing_value = LabeledEntry(exit_section, "Trailing distance (ATR multiple, e.g. 1.5)", 1.5)
        self.trailing_atr_period = LabeledEntry(exit_section, "Trailing stop ATR period", 14)

        Label(exit_section, text="BREAK EVEN", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.breakeven_enabled = LabeledCheckbox(exit_section, "Move stop to break-even once in profit", False)
        self.breakeven_trigger = LabeledEntry(exit_section, "Trigger, in multiples of initial risk (e.g. 1.0 = +1R)", 1.0)

        Label(exit_section, text="TIME-BASED EXIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.time_exit_enabled = LabeledCheckbox(exit_section, "Flatten any open trade at a fixed time", False)
        self.time_exit_time = LabeledEntry(exit_section, "Exit time (HH:MM, 24h)", "15:55")

        Label(exit_section, text="OTHER EXIT RULES", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.max_bars = LabeledEntry(exit_section, "Maximum bars in trade (blank = unlimited)", "")
        self.opposite_signal_exit = LabeledCheckbox(
            exit_section, "Opposite Signal Exit — an opposite entry signal closes/reverses an open trade", True,
        )

        Label(exit_section, text="INDICATOR EXIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        Label(
            exit_section, text="Optional extra exit rule(s), built the same way as entry conditions above.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        ).pack(anchor="w", padx=18, pady=(0, 4))

        Label(exit_section, text="Long exit", bg=PANEL, fg=GREEN, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        long_exit_container = Frame(exit_section, bg=PANEL)
        long_exit_container.pack(fill="x", padx=18, pady=(0, 4))
        self.long_exit_conditions = ConditionList(long_exit_container, get_session=self._current_session)
        self._button(exit_section, "+ Add Condition", self.long_exit_conditions.add_row).pack(
            anchor="w", padx=18, pady=(0, 10)
        )

        Label(exit_section, text="Short exit", bg=PANEL, fg=RED, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        short_exit_container = Frame(exit_section, bg=PANEL)
        short_exit_container.pack(fill="x", padx=18, pady=(0, 4))
        self.short_exit_conditions = ConditionList(short_exit_container, get_session=self._current_session)
        self._button(exit_section, "+ Add Condition", self.short_exit_conditions.add_row).pack(
            anchor="w", padx=18, pady=(0, 16)
        )

    def _current_session(self) -> tuple[str, str]:
        start = self.s_session_start.get_str().strip() or "08:30"
        end = self.s_session_end.get_str().strip() or "15:00"
        return start, end

    def _set_strategy_mode(self, mode: str):
        self.strategy_mode.set(mode)

        display = {
            "manual": "MANUAL STRATEGY BUILDER",
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

    def _stop_target_block(self, type_widget, value_widget, atr_period_widget) -> tuple[str, float | None, int]:
        label = type_widget.get_str()
        kind = {"None": "none", "Fixed (pips)": "fixed", "ATR Multiple": "atr"}.get(label, "none")
        value = value_widget.get_float(0) if kind != "none" else None
        period = atr_period_widget.get_int(14)
        return kind, value, period

    def _build_strategy(self):
        mode = self.strategy_mode.get()

        if mode == "manual":
            long_entry, long_entry_conn = self.long_entry_conditions.to_condition_list()
            short_entry, short_entry_conn = self.short_entry_conditions.to_condition_list()
            long_exit, long_exit_conn = self.long_exit_conditions.to_condition_list()
            short_exit, short_exit_conn = self.short_exit_conditions.to_condition_list()

            if not long_entry and not short_entry:
                raise StrategyError(
                    "Add at least one Long Entry or Short Entry condition in Step 2 before running."
                )

            stop_kind, stop_value, stop_period = self._stop_target_block(
                self.stop_type, self.stop_value, self.stop_atr_period
            )
            target_kind, target_value, target_period = self._stop_target_block(
                self.target_type, self.target_value, self.target_atr_period
            )

            max_bars_raw = self.max_bars.get_str().strip()

            cfg = {
                "name": self.s_name.get_str().strip() or "Manual Strategy",
                "description": self.s_description.get_str(),
                "author": self.s_author.get_str(),
                "version": self.s_version.get_str(),
                "market": {
                    "instrument": self.s_instrument.get_str(),
                    "timeframe": self.s_timeframe.get_str(),
                    "session_start": self.s_session_start.get_str().strip() or "08:30",
                    "session_end": self.s_session_end.get_str().strip() or "15:00",
                    "direction": self.s_direction.get_str(),
                },
                "entry_conditions": {
                    "long": long_entry, "long_connectors": long_entry_conn,
                    "short": short_entry, "short_connectors": short_entry_conn,
                },
                "exit_conditions": {
                    "long": long_exit, "long_connectors": long_exit_conn,
                    "short": short_exit, "short_connectors": short_exit_conn,
                },
                "risk_management": {
                    "stop_type": stop_kind,
                    "stop_value": stop_value,
                    "stop_atr_period": stop_period,
                    "target_type": target_kind,
                    "target_value": target_value,
                    "target_atr_period": target_period,
                    "trailing_stop": {
                        "enabled": self.trailing_enabled.get(),
                        "value": self.trailing_value.get_float(1.5),
                        "atr_period": self.trailing_atr_period.get_int(14),
                    },
                    "break_even": {
                        "enabled": self.breakeven_enabled.get(),
                        "trigger_r": self.breakeven_trigger.get_float(1.0),
                    },
                    "time_based_exit": {
                        "enabled": self.time_exit_enabled.get(),
                        "time": self.time_exit_time.get_str().strip(),
                    },
                    "max_bars_in_trade": int(max_bars_raw) if max_bars_raw else None,
                    "opposite_signal_exit": self.opposite_signal_exit.get(),
                },
            }
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
        if not self.csv_paths:
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

            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log(
                        f"Import errors ({os.path.basename(p)}):\n"
                        + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))
                for w in result.warnings:
                    self._log(f"  [warning] {os.path.basename(p)}: {w}")

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
                self._log(f"Loaded {len(df)} bars.")
            else:
                df, labels = merge_multi_timeframe(
                    [r.dataframe for _, r in per_file_results]
                )
                self._log(
                    f"Loaded {len(per_file_results)} timeframes and merged them "
                    f"on the finest timeframe: {' + '.join(labels)}  "
                    f"({len(df)} base bars)."
                )

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
                instrument=(
                    os.path.basename(self.csv_paths[0])
                    if len(self.csv_paths) == 1
                    else " + ".join(os.path.basename(p) for p in self.csv_paths)
                ),
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
