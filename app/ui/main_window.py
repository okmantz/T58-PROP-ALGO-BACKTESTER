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

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.storage import list_stored_datasets, store_csv_path
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig, run_iterative_refinement
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.reports.refinement_report import generate_refinement_report
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.search_report import generate_search_report
from app.search.strategy_space import StrategySpaceError, generate_search_space, list_families
from app.strategy.base import StrategyError
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.ui.condition_builder import ConditionList

OUTPUT_DIR = Path.cwd() / "reports"


def _strategy_display_name(strategy) -> str:
    if strategy.source_type == "manual":
        return strategy.config.get("name", "Manual Strategy")
    if strategy.source_type == "python":
        return Path(strategy.file_path).stem
    if strategy.source_type == "pinescript":
        return "PineScript Strategy"
    if strategy.source_type == "mql5":
        return "MQL5 Strategy"
    return "Strategy"

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

BG = "#08090C"
PANEL = "#131720"          # lifted slightly off BG so cards read as elevated surfaces
PANEL_2 = "#171B25"
PANEL_3 = "#1E232E"
PANEL_HOVER = "#242A37"    # hover state for interactive surfaces (buttons, rows)
BORDER = "#272C38"
BORDER_LIGHT = "#3D4453"
TEXT = "#E9EBEF"
TEXT_MUTED = "#8D94A3"
TEXT_DIM = "#5C6472"
METAL = "#B8BDC5"
METAL_BRIGHT = "#E7E9ED"
GREEN = "#3ED685"
RED = "#F0596A"
BLUE = "#6FA8FF"
AMBER = "#D9A441"
# Signature brand accent — used deliberately for primary actions, the active
# tab indicator, focus states and a handful of high-intent highlights. Kept
# out of the status vocabulary (green/red/blue/amber already mean
# success/error/info/warning) so it always reads as "act here."
ACCENT = "#7C6FFF"
ACCENT_HOVER = "#9089FF"
ACCENT_DIM = "#332E5C"     # low-opacity-style accent for subtle fills/left-bars
ACCENT_INK = "#0C0A16"     # near-black used as text on top of the bright accent
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
        self.tab_refine = Frame(self.nb, bg=BG)
        self.tab_search = Frame(self.nb, bg=BG)

        self.nb.add(self.tab_data, text="  01 ▤  DATA  ")
        self.nb.add(self.tab_strategy, text="  02 ⚙  STRATEGY  ")
        self.nb.add(self.tab_prop, text="  03 ⚖  PROP RULES  ")
        self.nb.add(self.tab_risk, text="  04 ◈  RISK  ")
        self.nb.add(self.tab_run, text="  05 ▶  RUN & REPORT  ")
        self.nb.add(self.tab_refine, text="  06 ↻  REFINEMENT  ")
        self.nb.add(self.tab_search, text="  07 ▦  SEARCH LAB  ")

        # Thin accent underline that tracks whichever tab is active, giving
        # the tab strip a focal point beyond the stock ttk selected-color
        # swap. Positioned with place() against the notebook's own
        # coordinate space so it lines up exactly with the live tab bbox.
        self.tab_indicator = Frame(self.nb, bg=ACCENT, height=3, bd=0, highlightthickness=0)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._update_tab_indicator())
        self.root.bind("<Configure>", lambda _e: self._update_tab_indicator())

        self._build_data_tab()
        self._build_strategy_tab()
        self._build_prop_tab()
        self._build_risk_tab()
        self._build_run_tab()
        self._build_refine_tab()
        self._build_search_tab()

        for delay in (50, 150, 400):
            self.root.after(delay, self._update_tab_indicator)

    def _update_tab_indicator(self):
        # bbox() can legitimately return an empty/zero box for one frame
        # right after the window is first realized (before the tab strip
        # has been laid out) — hide the bar rather than leave it stuck at a
        # stale position, and let the next <Configure>/tab-change re-place it.
        try:
            bbox = self.nb.bbox(self.nb.select())
        except Exception:
            bbox = None

        if not bbox or bbox[2] == 0:
            self.tab_indicator.place_forget()
            return

        bx, by, bw, bh = bbox
        self.tab_indicator.place(in_=self.nb, x=bx, y=by + bh - 3, width=bw, height=3)
        self.tab_indicator.lift()

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
            padding=[20, 12],
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
                ("selected", ACCENT_HOVER),
                ("active", TEXT),
            ],
        )

        style.configure(
            "T58.Vertical.TScrollbar",
            background=BORDER_LIGHT,
            troughcolor=PANEL,
            bordercolor=PANEL,
            arrowcolor=TEXT_DIM,
            gripcount=0,
            width=14,
            relief="flat",
        )
        style.map(
            "T58.Vertical.TScrollbar",
            background=[("active", ACCENT), ("pressed", ACCENT_HOVER)],
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
            background=ACCENT,
            troughcolor=PANEL_3,
            bordercolor=PANEL_3,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
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

        sub_row = Frame(mark, bg=BG)
        sub_row.pack(anchor="w", pady=(0, 2))
        Frame(sub_row, bg=ACCENT, width=10, height=2).pack(side="left", pady=(3, 0))
        Label(
            sub_row,
            text="PROP ALGO BACKTESTER",
            bg=BG,
            fg=TEXT_MUTED,
            font=_safe_font(8, "bold"),
        ).pack(side="left", padx=(6, 0))

        right = Frame(header, bg=BG)
        right.pack(side="right", fill="y")

        Label(
            right,
            text="MVP",
            bg=PANEL_2,
            fg=ACCENT_HOVER,
            font=_safe_font(8, "bold"),
            padx=12,
            pady=5,
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
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

        eyebrow_row = Frame(box, bg=BG)
        eyebrow_row.pack(anchor="w")

        Frame(eyebrow_row, bg=ACCENT, width=14, height=2).pack(side="left", pady=(4, 0))
        Label(
            eyebrow_row,
            text=eyebrow.upper(),
            bg=BG,
            fg=ACCENT_HOVER,
            font=_safe_font(8, "bold"),
        ).pack(side="left", padx=(7, 0))

        Label(
            box,
            text=title,
            bg=BG,
            fg=METAL_BRIGHT,
            font=_safe_font(21, "bold"),
        ).pack(anchor="w", pady=(5, 3))

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

        Frame(box, bg=BORDER, height=1).pack(fill="x", pady=(14, 0))

    def _button(self, parent, text, command, primary=False, width=None):
        kwargs = {
            "text": text,
            "command": command,
            "font": _safe_font(9, "bold"),
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "padx": 16,
            "pady": 8,
        }

        if primary:
            kwargs.update(
                bg=ACCENT,
                fg="#FFFFFF",
                activebackground=ACCENT_HOVER,
                activeforeground="#FFFFFF",
                highlightthickness=0,
            )
            hover_bg, idle_bg = ACCENT_HOVER, ACCENT
        else:
            kwargs.update(
                bg=PANEL_3,
                fg=TEXT,
                activebackground=PANEL_HOVER,
                activeforeground=METAL_BRIGHT,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=BORDER_LIGHT,
            )
            hover_bg, idle_bg = PANEL_HOVER, PANEL_3

        if width:
            kwargs["width"] = width

        btn = Button(parent, **kwargs)

        # Tk's native Button only recolors on click (activebackground), not on
        # hover, so real cursor-follows-affordance feedback is added by hand.
        def _on_enter(_e, b=btn, c=hover_bg):
            if str(b["state"]) != "disabled":
                b.configure(bg=c)

        def _on_leave(_e, b=btn, c=idle_bg):
            if str(b["state"]) != "disabled":
                b.configure(bg=c)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)

        return btn

    def _scrollable(self, parent) -> Frame:
        """Wraps `parent` in a mouse-wheel-scrollable canvas and returns an
        inner Frame to build content into. Used for tabs long enough to
        overflow the window (the Manual Strategy Builder, the Iterative
        Refinement builder).

        The wheel binding is done at the root level and dispatched by
        cursor screen-position rather than the old Enter/Leave-on-canvas
        approach: Tkinter fires Leave the instant the pointer moves onto a
        child widget sitting inside the canvas (an Entry, Combobox, or
        Button), which silently killed scrolling over almost all of the
        actual form content and only worked in the bare margins. Binding
        once at the root and checking which registered canvas's bounding
        box contains the pointer fixes that regardless of which widget is
        directly under the cursor.
        """
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

        if not hasattr(self, "_scroll_canvases"):
            self._scroll_canvases = []
        self._scroll_canvases.append(canvas)

        def _dispatch_wheel(event, delta):
            x, y = event.x_root, event.y_root
            for c in self._scroll_canvases:
                try:
                    if not c.winfo_ismapped():
                        continue
                    x1, y1 = c.winfo_rootx(), c.winfo_rooty()
                    x2, y2 = x1 + c.winfo_width(), y1 + c.winfo_height()
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        c.yview_scroll(int(delta), "units")
                        return
                except Exception:
                    continue

        if not getattr(self, "_wheel_bound", False):
            self._wheel_bound = True
            self.root.bind_all(
                "<MouseWheel>",
                lambda e: _dispatch_wheel(e, -1 * (e.delta // 120)),
            )
            self.root.bind_all("<Button-4>", lambda e: _dispatch_wheel(e, -1))
            self.root.bind_all("<Button-5>", lambda e: _dispatch_wheel(e, 1))

        return inner

    def _section(self, parent, title, subtitle="", emphasize=False):
        """A card-style container. `emphasize=True` marks the one primary
        action card on a tab (e.g. the run/import card) with a left accent
        bar, so each screen has a single clear focal point instead of every
        panel competing at the same visual weight."""
        wrap = Frame(parent, bg=BG)
        wrap.pack(fill="x", padx=24, pady=7)

        if emphasize:
            Frame(wrap, bg=ACCENT, width=3).pack(side="left", fill="y")

        box = Frame(wrap, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(side="left", fill="both", expand=True)

        title_row = Frame(box, bg=PANEL)
        title_row.pack(fill="x", padx=18, pady=(14, 2))

        Label(
            title_row,
            text=title.upper(),
            bg=PANEL,
            fg=ACCENT_HOVER if emphasize else METAL,
            font=_safe_font(9, "bold"),
        ).pack(side="left")

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
            emphasize=True,
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
            emphasize=True,
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
            "Core evaluation parameters. Drawdown check mode: 'intrabar' "
            "monitors floating equity in real time (a floor breach can "
            "force-close a position mid-trade); 'eod' only checks once per "
            "day using that day's final balance, so an intraday dip that "
            "recovers by the close doesn't count. Match this to what your "
            "specific firm documents -- getting it backwards makes your "
            "pass-probability estimate too pessimistic or too optimistic.",
            emphasize=True,
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
        self.p_dd_check_mode = LabeledEntry(
            section,
            "Drawdown check mode (intrabar/eod)",
            "intrabar",
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
            drawdown_check_mode=self.p_dd_check_mode.get_str().strip() or "intrabar",
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
            emphasize=True,
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
    # Iterative Refinement — shared execution helper (used by both the
    # standalone button on Tab 6 and the optional auto-run from Tab 5)
    # -----------------------------------------------------------------------

    def _build_refine_config(self) -> RefinementConfig:
        metric_label = self.refine_metric.get_str()
        metric_key = self._refine_metric_label_to_key.get(metric_label, "composite_prop_score")
        return RefinementConfig(
            fitness_metric=metric_key,
            population_size=self.refine_population.get_int(10),
            generations=self.refine_generations.get_int(5),
            elite_count=self.refine_elite.get_int(2),
            mutation_rate=self.refine_mutation_rate.get_float(0.35),
            mutation_strength=self.refine_mutation_strength.get_float(0.25),
            random_immigrants_frac=self.refine_immigrants.get_float(0.15),
            search_monte_carlo_sims=self.refine_search_sims.get_int(500),
            random_seed=self.refine_seed.get_int(42),
        )

    def _execute_refinement(self, df, strategy, risk, rules, mc_cfg, log_fn) -> dict:
        """
        Runs Iterative Refinement against an already-built strategy/df/risk/
        rules/mc_cfg (the exact same objects the normal pipeline would use)
        and writes a second, separate report. Works for Manual, Python,
        PineScript, and MQL5 strategies alike. Raises RefinementError if the
        strategy has no tunable parameters for its source type.
        """
        refine_cfg = self._build_refine_config()
        result = run_iterative_refinement(
            df, strategy, risk, rules, mc_cfg, refine_cfg, progress_cb=log_fn,
        )

        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        instrument = (
            os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
            else " + ".join(os.path.basename(p) for p in self.csv_paths)
        )
        paths = generate_refinement_report(
            output_dir=OUTPUT_DIR,
            result=result,
            strategy_name=_strategy_display_name(strategy),
            instrument=instrument,
            timeframe="unknown",
            backtest_period=period,
            price_df=df,
        )

        self._last_refinement_result = result
        self._last_refinement_html_path = paths["html"]
        self._last_refinement_best_strategy_path = paths.get("best_strategy_file")
        self.open_refine_report_btn.config(state="normal")
        self.apply_best_config_btn.config(state="normal")
        return paths

    # -----------------------------------------------------------------------
    # Tab 6 — Iterative Refinement (optional)
    # -----------------------------------------------------------------------

    def _build_refine_tab(self):
        f = self._scrollable(self.tab_refine)

        self._page_header(
            f,
            "06 / Iterative Refinement",
            "Iterative Refinement (Optional)",
            "Genetic-algorithm-style parameter search: re-runs this strategy many times "
            "with mutated parameters on the SAME historical data, keeps the "
            "best-performing configurations each round, and converges toward the "
            "best-scoring configuration it can find. Produces its own, separate report "
            "-- the normal Run & Report tab and report.html are completely unaffected "
            "unless you enable this below.",
        )

        section = self._section(
            f, "Enable for this run",
            "Off by default, and reset per session -- nothing changes about the normal "
            "pipeline unless this is checked. When ON, clicking RUN FULL PIPELINE in "
            "Step 5 will also run Iterative Refinement afterward and produce a second "
            "report. You can also run it on its own with the button below at any time, "
            "regardless of this setting.",
        )
        self.refine_enabled = LabeledCheckbox(
            section, "Enable Iterative Refinement when running the full pipeline (Step 5)", False,
        )
        Label(
            section,
            text="Works with Manual Strategy Builder, Python, PineScript, and MQL5 strategies. "
                 "For Python it searches every top-level SCREAMING_SNAKE_CASE numeric constant "
                 "(e.g. EMA_FAST, STOP_LOSS_PIPS); for PineScript, every input.int()/input.float() "
                 "value; for MQL5, every iMA()/iRSI() period -- plus the T58_SL_PIPS/T58_TP_PIPS "
                 "directives for all three. A strategy with no such parameters will say so clearly "
                 "rather than run a meaningless search.",
            bg=PANEL, fg=AMBER, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        settings = self._section(
            f, "Search settings",
            "Every setting below has a reasonable default -- you don't need to touch any "
            "of them to run a first search.",
        )

        self._refine_metric_labels = list(FITNESS_METRICS.values())
        self._refine_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.refine_metric = LabeledCombo(
            settings, "Fitness metric (what \u201cbest\u201d means)", self._refine_metric_labels,
            FITNESS_METRICS["composite_prop_score"],
        )
        self.refine_population = LabeledEntry(settings, "Population size (configs per generation)", 10)
        self.refine_generations = LabeledEntry(settings, "Generations (rounds)", 5)
        self.refine_elite = LabeledEntry(settings, "Elite count (top configs carried forward unchanged)", 2)
        self.refine_mutation_rate = LabeledEntry(settings, "Mutation rate (0-1, chance per parameter per child)", 0.35)
        self.refine_mutation_strength = LabeledEntry(settings, "Mutation strength (0-1, fraction of each parameter's search range)", 0.25)
        self.refine_immigrants = LabeledEntry(settings, "Random immigrants fraction (0-1, per generation)", 0.15)
        self.refine_search_sims = LabeledEntry(settings, "Monte Carlo simulations per candidate during search", 500)
        self.refine_seed = LabeledEntry(settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(
            button_row, "RUN ITERATIVE REFINEMENT", self._refine_run_clicked, primary=True,
        ).pack(side="left")

        self.open_refine_report_btn = self._button(
            button_row, "OPEN REFINEMENT REPORT", self._open_refine_report,
        )
        self.open_refine_report_btn.config(state="disabled")
        self.open_refine_report_btn.pack(side="left", padx=8)

        self.apply_best_config_btn = self._button(
            button_row, "APPLY BEST CONFIG TO STRATEGY TAB", self._apply_best_configuration,
        )
        self.apply_best_config_btn.config(state="disabled")
        self.apply_best_config_btn.pack(side="left", padx=8)

        self.refine_progress = ttk.Progressbar(
            f, mode="indeterminate", style="T58.Horizontal.TProgressbar",
        )
        self.refine_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Refinement output", "Live search log.")
        self.refine_output = Text(
            output_section, height=18, wrap="word", bg="#0B0D10", fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.refine_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_refinement_result = None
        self._last_refinement_html_path = None
        self._last_refinement_best_strategy_path = None

    def _log_refine(self, msg: str):
        self.refine_output.insert(END, msg + "\n")
        self.refine_output.see(END)
        self.root.update_idletasks()

    def _open_refine_report(self):
        if self._last_refinement_html_path:
            webbrowser.open(f"file://{self._last_refinement_html_path.resolve()}")

    def _refine_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV in Step 1.",
            )
            return
        self.refine_output.delete("1.0", END)
        self.refine_progress.start(10)
        threading.Thread(target=self._refine_run_pipeline, daemon=True).start()

    def _refine_run_pipeline(self):
        try:
            self._log_refine("Importing market data...")
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_refine(
                        f"Import errors ({os.path.basename(p)}):\n"
                        + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_refine(f"Loaded {len(df)} bars.")

            self._log_refine("Building strategy...")
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()

            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            mc_cfg = MonteCarloConfig(n_simulations=n_sims, method=method)

            self._log_refine("Starting Iterative Refinement search...")
            paths = self._execute_refinement(df, strategy, risk, rules, mc_cfg, self._log_refine)

            self._log_refine("\nDone. Iterative Refinement report written to:")
            for k, p in paths.items():
                self._log_refine(f"  {k}: {p}")

        except StrategyError as exc:
            self._log_refine(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_refine(f"\nIterative Refinement error: {exc}")
        except Exception:
            self._log_refine("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.refine_progress.stop()

    def _apply_best_configuration(self):
        if not getattr(self, "_last_refinement_result", None):
            messagebox.showinfo(
                "No refinement result",
                "Run Iterative Refinement first, then apply its best configuration.",
            )
            return

        result = self._last_refinement_result
        source_type = result.source_type

        if source_type != "manual":
            self._apply_best_code_strategy(result, source_type)
            return

        if self.strategy_mode.get() != "manual":
            messagebox.showwarning(
                "Mode mismatch",
                "This refinement result was optimized for a Manual Strategy Builder "
                "strategy. Switch the Strategy tab to Manual mode to apply it, or "
                "re-run Iterative Refinement against whatever strategy is currently selected.",
            )
            return

        cfg = result.best.config
        entries = cfg.get("entry_conditions", {}) or {}
        exits = cfg.get("exit_conditions", {}) or {}

        self.long_entry_conditions.set_from_conditions(entries.get("long", []), entries.get("long_connectors"))
        self.short_entry_conditions.set_from_conditions(entries.get("short", []), entries.get("short_connectors"))
        self.long_exit_conditions.set_from_conditions(exits.get("long", []), exits.get("long_connectors"))
        self.short_exit_conditions.set_from_conditions(exits.get("short", []), exits.get("short_connectors"))

        rm = cfg.get("risk_management", {}) or {}
        if rm.get("stop_value") is not None:
            self.stop_value.var.set(str(rm["stop_value"]))
        if rm.get("stop_atr_period") is not None:
            self.stop_atr_period.var.set(str(rm["stop_atr_period"]))
        if rm.get("target_value") is not None:
            self.target_value.var.set(str(rm["target_value"]))
        if rm.get("target_atr_period") is not None:
            self.target_atr_period.var.set(str(rm["target_atr_period"]))
        trailing = rm.get("trailing_stop", {}) or {}
        if trailing.get("value") is not None:
            self.trailing_value.var.set(str(trailing["value"]))
        if trailing.get("atr_period") is not None:
            self.trailing_atr_period.var.set(str(trailing["atr_period"]))
        be = rm.get("break_even", {}) or {}
        if be.get("trigger_r") is not None:
            self.breakeven_trigger.var.set(str(be["trigger_r"]))
        if rm.get("max_bars_in_trade") is not None:
            self.max_bars.var.set(str(rm["max_bars_in_trade"]))

        # Legacy top-level fields, in case this config came from a
        # non-visual-builder ManualStrategy (e.g. loaded from an older
        # config or the CLI default strategy).
        if cfg.get("stop_loss_pips") is not None and not entries:
            self.stop_value.var.set(str(cfg["stop_loss_pips"]))
        if cfg.get("take_profit_pips") is not None and not entries:
            self.target_value.var.set(str(cfg["take_profit_pips"]))

        messagebox.showinfo(
            "Applied",
            "The optimized parameters have been loaded into the Strategy tab (Step 2). "
            "Switch tabs to review them, then re-run the normal pipeline (Step 5) to "
            "confirm the result with a fresh, non-search report.",
        )

    def _apply_best_code_strategy(self, result, source_type: str):
        """
        For Python/PineScript/MQL5 strategies, "applying" the winner means
        pointing the Strategy tab's file selector at the already-written
        patched source file (its logic is identical to the original -- only
        the numeric parameter values changed), so the next run uses it.
        """
        path = getattr(self, "_last_refinement_best_strategy_path", None)
        if not path:
            messagebox.showinfo(
                "File not available",
                "The optimized strategy file wasn't written for this run -- try "
                "running Iterative Refinement again.",
            )
            return

        expected_mode = source_type  # "python" | "pinescript" | "mql5"
        if self.strategy_mode.get() != expected_mode:
            self._set_strategy_mode(expected_mode)

        self.strategy_py_path = str(path)
        self.strategy_file_status.config(text=f"Selected: {os.path.basename(str(path))}", fg=GREEN)

        messagebox.showinfo(
            "Applied",
            f"The optimized {FITNESS_METRICS.get(result.fitness_metric, result.fitness_metric)}-tuned "
            f"strategy file has been selected on the Strategy tab (Step 2):\n\n{path}\n\n"
            "Re-run the normal pipeline (Step 5) to confirm the result with a fresh, "
            "non-search report.",
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
            emphasize=True,
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

            if strategy.source_type == "python":
                self._log("Checking for lookahead bias...")
                try:
                    lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
                    self._log(f"  {lookahead_result.summary()}")
                except Exception:
                    # This check is a best-effort audit, not part of the
                    # core pipeline -- never let it take down a run that
                    # would otherwise succeed.
                    self._log("  Lookahead check failed to run (skipped).")

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

            if not bt_result.trades:
                # A strategy that produces zero signals over the given data
                # has nothing for the prop simulation or Monte Carlo to
                # resample -- both are fundamentally undefined here, not
                # just numerically awkward. Previously this fell through to
                # run_monte_carlo(), which correctly raises ValueError, but
                # that surfaced to the user as an unhandled traceback rather
                # than a clear, actionable message. Stop here instead.
                self._log(
                    "\nNo trades were generated by this strategy over the "
                    "given data -- there is nothing to run a prop-firm "
                    "simulation or Monte Carlo simulation on, and no report "
                    "was produced. This usually means the strategy's entry "
                    "conditions never fired (too strict for this data/"
                    "date range) rather than an app problem. Check the "
                    "strategy's signal logic, or try a longer/different "
                    "data range."
                )
                return

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

            self._log("Running out-of-sample holdout check...")
            try:
                holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
            except Exception:
                self._log("  Holdout check skipped (not enough data to split).")
                holdout_comparison = None

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
                holdout_comparison=holdout_comparison,
                risk_config=risk,
                price_df=df,
            )

            self._last_html_path = paths["html"]
            self.open_report_btn.config(state="normal")

            self._log("\nDone. Report written to:")

            for k, p in paths.items():
                self._log(f"  {k}: {p}")

            if getattr(self, "refine_enabled", None) and self.refine_enabled.get():
                self._log("\n--- Iterative Refinement (optional feature enabled on Step 6) ---")
                try:
                    refine_paths = self._execute_refinement(df, strategy, risk, rules, mc_cfg, self._log)
                    self._log("\nIterative Refinement report written to:")
                    for k, p in refine_paths.items():
                        self._log(f"  {k}: {p}")
                except RefinementError as exc:
                    self._log(f"\nIterative Refinement skipped: {exc}")
                except Exception:
                    self._log("\nIterative Refinement failed:\n" + traceback.format_exc())

        except StrategyError as exc:
            self._log(f"\nStrategy error: {exc}")

        except Exception:
            self._log(
                "\nUnexpected error:\n"
                + traceback.format_exc()
            )

        finally:
            self.progress.stop()

    # -----------------------------------------------------------------------
    # Tab 7 — Search Lab (Stages 1-5: cheap filter -> GA refinement ->
    # validation gate -> leaderboard -> champion promotion)
    # -----------------------------------------------------------------------

    _SEARCH_MODE_LABELS = {
        "Family (search a grid of many strategy variations)": "family",
        "Single (re-validate the current Strategy tab config through the full funnel)": "single",
    }

    def _build_search_tab(self):
        f = self._scrollable(self.tab_search)

        self._page_header(
            f,
            "07 / Search Lab",
            "Search Lab (Stages 1-5)",
            "Generates and tests many strategy variations at once instead of one at a "
            "time: a fast filter narrows thousands of candidates down to a shortlist, "
            "the same Iterative Refinement engine from Step 6 tunes each shortlisted "
            "candidate, and a strict validation gate (walk-forward, lookahead check, "
            "parameter-neighborhood robustness, a deflated Sharpe ratio that corrects "
            "for how many candidates were tried) decides what actually survives. "
            "Completely separate from the normal Run & Report pipeline and from Step 6 "
            "-- nothing here changes unless you click Run below.",
        )

        mode_section = self._section(
            f, "What to search",
            "Family mode generates a combinatorial grid of a named strategy hypothesis. "
            "Single mode instead takes whatever is currently configured on the Strategy "
            "tab (Step 2) and puts that ONE strategy through the exact same 5-stage "
            "funnel, as an independent stress test -- useful for re-validating a "
            "hand-built strategy rather than searching a family.",
        )
        self.search_mode = LabeledCombo(
            mode_section, "Mode", list(self._SEARCH_MODE_LABELS.keys()),
            "Family (search a grid of many strategy variations)",
        )
        self.search_mode.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_search_mode_changed())

        family_labels = ["All families (search every hypothesis together)"] + [
            f"{label} [{name}]" for name, label in list_families().items()
        ]
        self._search_family_label_to_key = {"All families (search every hypothesis together)": "all"}
        for name, label in list_families().items():
            self._search_family_label_to_key[f"{label} [{name}]"] = name
        self.search_family = LabeledCombo(
            mode_section, "Strategy family (Family mode only)", family_labels, family_labels[0],
        )

        space_section = self._section(
            f, "Search space size",
            "How many generated candidates Stage 1 actually evaluates. If the full grid "
            "for a family is larger than this, a random (reproducible) sample is taken "
            "instead of just the first N -- see the random seed below.",
        )
        self.search_max_candidates = LabeledEntry(space_section, "Max candidates (Family mode)", 500)
        self.search_workers = LabeledEntry(space_section, "Parallel workers (blank = all CPU cores)", "")
        self.search_seed = LabeledEntry(space_section, "Random seed", 42)

        stage1_section = self._section(
            f, "Stage 1 -- cheap filter",
            "One fast backtest per candidate, no Monte Carlo. Kills the vast majority "
            "of candidates in minutes, not hours. Loosen these if a run reports zero "
            "Stage 1 survivors.",
        )
        self.search_min_trades = LabeledEntry(stage1_section, "Minimum trades to survive", 20)
        self.search_min_pf = LabeledEntry(stage1_section, "Minimum profit factor to survive", 1.05)
        self.search_stage1_top_n = LabeledEntry(stage1_section, "Survivors that advance to Stage 2 (GA)", 40)

        stage2_section = self._section(
            f, "Stage 2 -- GA refinement",
            "The same genetic-algorithm engine as Step 6, applied to every Stage 1 "
            "survivor instead of to one hand-picked strategy.",
        )
        self.search_ga_population = LabeledEntry(stage2_section, "GA population size", 10)
        self.search_ga_generations = LabeledEntry(stage2_section, "GA generations", 4)
        self.search_stage2_top_n = LabeledEntry(stage2_section, "Survivors that advance to Stage 3 (validation)", 10)

        stage3_section = self._section(
            f, "Stage 3 -- validation gate",
            "Full-fidelity Monte Carlo, multi-fold walk-forward, the lookahead-bias "
            "detector, and parameter-neighborhood robustness. A candidate must clear "
            "every gate to become the champion -- this is deliberately strict.",
        )
        self.search_full_mc_sims = LabeledEntry(stage3_section, "Monte Carlo simulations (full fidelity)", 3000)
        self.search_walk_forward_folds = LabeledEntry(stage3_section, "Walk-forward folds (0 disables)", 4)
        self.search_robustness_neighbors = LabeledEntry(
            stage3_section, "Parameter-neighborhood samples (0 disables)", 6,
        )
        self._search_metric_labels = list(FITNESS_METRICS.values())
        self.search_metric = LabeledCombo(
            stage3_section, "Fitness metric (what \u201cbest\u201d means)", self._search_metric_labels,
            FITNESS_METRICS["composite_prop_score"],
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(button_row, "RUN SEARCH LAB", self._search_run_clicked, primary=True).pack(side="left")

        self.open_search_report_btn = self._button(
            button_row, "OPEN LEADERBOARD", self._open_search_report,
        )
        self.open_search_report_btn.config(state="disabled")
        self.open_search_report_btn.pack(side="left", padx=8)

        self.promote_champion_btn = self._button(
            button_row, "PROMOTE CHAMPION TO FULL REPORT", self._promote_search_champion_clicked,
        )
        self.promote_champion_btn.config(state="disabled")
        self.promote_champion_btn.pack(side="left", padx=8)

        self.open_champion_report_btn = self._button(
            button_row, "OPEN CHAMPION REPORT", self._open_champion_report,
        )
        self.open_champion_report_btn.config(state="disabled")
        self.open_champion_report_btn.pack(side="left", padx=8)

        self.search_progress = ttk.Progressbar(
            f, mode="indeterminate", style="T58.Horizontal.TProgressbar",
        )
        self.search_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Search Lab output", "Live funnel log.")
        self.search_output = Text(
            output_section, height=18, wrap="word", bg="#0B0D10", fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.search_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_search_summary = None
        self._last_search_space = None
        self._last_search_html_path = None
        self._last_search_db_path = None
        self._last_search_df = None
        self._last_search_risk = None
        self._last_search_rules = None
        self._last_champion_html_path = None

    def _on_search_mode_changed(self):
        mode = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str())
        # Family-only settings are visually left enabled either way (Tk
        # combobox rebuilding is more churn than it's worth for a disabled
        # look) -- they're simply ignored by _search_run_pipeline in Single
        # mode.

    def _log_search(self, msg: str):
        self.search_output.insert(END, msg + "\n")
        self.search_output.see(END)
        self.root.update_idletasks()

    def _open_search_report(self):
        if self._last_search_html_path:
            webbrowser.open(f"file://{self._last_search_html_path.resolve()}")

    def _open_champion_report(self):
        if self._last_champion_html_path:
            webbrowser.open(f"file://{self._last_champion_html_path.resolve()}")

    def _build_search_stage_config(self) -> SearchStageConfig:
        metric_label = self.search_metric.get_str()
        metric_key = self._refine_metric_label_to_key.get(metric_label, "composite_prop_score")
        workers_raw = self.search_workers.get_str().strip()
        workers = int(workers_raw) if workers_raw else None
        return SearchStageConfig(
            min_trades=self.search_min_trades.get_int(20),
            min_profit_factor=self.search_min_pf.get_float(1.05),
            stage1_top_n=self.search_stage1_top_n.get_int(40),
            ga_population=self.search_ga_population.get_int(10),
            ga_generations=self.search_ga_generations.get_int(4),
            stage2_top_n=self.search_stage2_top_n.get_int(10),
            full_mc_sims=self.search_full_mc_sims.get_int(3000),
            walk_forward_folds=self.search_walk_forward_folds.get_int(4),
            robustness_neighbors=self.search_robustness_neighbors.get_int(6),
            fitness_metric=metric_key,
            workers=workers,
            random_seed=self.search_seed.get_int(42),
        )

    def _search_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV in Step 1.",
            )
            return
        self.search_output.delete("1.0", END)
        self.open_search_report_btn.config(state="disabled")
        self.promote_champion_btn.config(state="disabled")
        self.open_champion_report_btn.config(state="disabled")
        self.search_progress.start(10)
        threading.Thread(target=self._search_run_pipeline, daemon=True).start()

    def _search_run_pipeline(self):
        try:
            self._log_search("Importing market data...")
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_search(
                        f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_search(f"Loaded {len(df)} bars.")

            risk = self._build_risk_config()
            rules = self._build_prop_rules()

            mode = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str(), "family")
            if mode == "single":
                if self.strategy_mode.get() != "manual":
                    self._log_search(
                        "\nSingle mode requires a Manual Strategy Builder configuration on the "
                        "Strategy tab (Step 2) -- Search Lab only runs Manual Strategy configs "
                        "through its funnel. Switch Step 2 to Manual mode, or use Family mode "
                        "instead."
                    )
                    return
                strategy = self._build_strategy()
                space = generate_search_space(mode="single", single_config=strategy.config)
            else:
                family_label = self.search_family.get_str()
                family_key = self._search_family_label_to_key.get(family_label, "all")
                space = generate_search_space(
                    mode="family", family=family_key,
                    max_candidates=self.search_max_candidates.get_int(500),
                    seed=self.search_seed.get_int(42),
                )

            # (run_search() itself logs the "Search space ready..." line via
            # progress_cb once it starts -- not duplicated here.)
            stage_cfg = self._build_search_stage_config()
            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )
            db_path = str(OUTPUT_DIR / "search" / "search.db")

            summary = run_search(
                df, risk, rules, space, stage_cfg, db_path=db_path,
                instrument=instrument, timeframe="unknown", progress_cb=self._log_search,
            )

            report_paths = generate_search_report(
                output_dir=str(OUTPUT_DIR / "search"), summary=summary, space=space,
                instrument=instrument, timeframe="unknown",
            )

            self._last_search_summary = summary
            self._last_search_space = space
            self._last_search_html_path = report_paths["html"]
            self._last_search_db_path = db_path
            self._last_search_df = df
            self._last_search_risk = risk
            self._last_search_rules = rules

            self.open_search_report_btn.config(state="normal")
            if summary.champion_candidate_id:
                self.promote_champion_btn.config(state="normal")

            self._log_search("\nDone. Search leaderboard written to:")
            for k, p in report_paths.items():
                self._log_search(f"  {k}: {p}")

        except StrategySpaceError as exc:
            self._log_search(f"\nSearch space error: {exc}")
        except StrategyError as exc:
            self._log_search(f"\nStrategy error: {exc}")
        except Exception:
            self._log_search("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.search_progress.stop()

    def _promote_search_champion_clicked(self):
        summary = self._last_search_summary
        if not summary or not summary.champion_candidate_id:
            messagebox.showinfo(
                "No champion",
                "Run Search Lab first -- promotion is only available when a run has "
                "at least one candidate that passed every Stage 3 gate.",
            )
            return
        self.promote_champion_btn.config(state="disabled")
        self.search_progress.start(10)
        threading.Thread(target=self._promote_search_champion_pipeline, daemon=True).start()

    def _promote_search_champion_pipeline(self):
        try:
            summary = self._last_search_summary
            self._log_search(
                f"\nPromoting champion candidate {summary.champion_candidate_id} to a full report..."
            )
            promo = promote_champion(
                self._last_search_db_path, summary.run_id, summary.champion_candidate_id,
                self._last_search_df, self._last_search_risk, self._last_search_rules,
                output_dir=str(OUTPUT_DIR / "search" / "champion"),
            )
            self._last_champion_html_path = promo["report_paths"]["html"]
            self.open_champion_report_btn.config(state="normal")
            self._log_search("Champion report written to:")
            for k, p in promo["report_paths"].items():
                self._log_search(f"  {k}: {p}")
        except Exception:
            self._log_search("\nChampion promotion failed:\n" + traceback.format_exc())
        finally:
            self.promote_champion_btn.config(state="normal")
            self.search_progress.stop()


def launch():
    root = Tk()
    MainWindow(root)
    root.mainloop()

