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
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, Text, END,
    filedialog, messagebox, simpledialog, ttk, Listbox, SINGLE, EXTENDED, BooleanVar, Canvas,
    Checkbutton, PhotoImage, Toplevel,
)

import app.evolution.checkpoint as evo_checkpoint
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskError, AdaptiveRiskRule
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig, suggest_pip_size
import app.ai.ollama_settings as ollama_settings_module
from app.ai.ollama_settings import OllamaSettings
import app.ai.strategy_generator as strategy_generator_module
import app.ai.research_library as research_library_module
import app.ai.experiment_memory as experiment_memory_module
from app.ai.research_agent import ResearchAgentContext, ResearchAgent
from app.data import alpaca_credentials
from app.data.alpaca_source import (
    ASSET_CLASSES, ADJUSTMENT_CHOICES, FEED_CHOICES, TIMEFRAME_LABELS,
    AlpacaFetchError, AlpacaImportError, fetch_bars, save_bars_as_csv, test_connection,
)
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.pairs import PairDataError, merge_pair_series
from app.data.storage import EMPTY_DATASET_BYTES, list_datasets_by_instrument, list_stored_datasets, store_csv_path
from app.ensemble.ensemble import EnsembleError, EnsembleVoteConfig, run_ensemble_blend, run_ensemble_vote
import app.forward_test.mt5_settings as mt5_settings_module
from app.forward_test.mt5_settings import MT5Settings
from app.forward_test import mt5_connector as mt5_connector_module
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
from app.live_deploy import prop_firms as live_deploy_prop_firms
from app.live_deploy import live_settings as live_deploy_settings
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import RefinementError, apply_genome, extract_genome
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig, run_iterative_refinement
from app.optimize.multi_objective import DEFAULT_OBJECTIVES, MultiObjectiveConfig, OBJECTIVE_DIRECTIONS, run_multi_objective_refinement
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.orchestration.full_pipeline import (
    FullPipelineBatchItem, FullPipelineConfig, run_full_pipeline, run_full_pipeline_batch,
)
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioError, run_portfolio_backtest
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.reports import run_history
from app.reports.refinement_report import generate_refinement_report
from app.reports.validation_reports import (
    generate_cpcv_report, generate_multi_objective_report, generate_pbo_report,
    generate_portfolio_report, generate_sensitivity_report, generate_walk_forward_report,
    generate_walkforward_ga_report,
)
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.search_report import generate_search_report
from app.search.strategy_space import StrategySpaceError, generate_search_space, list_families
from app.strategy.base import StrategyError
from app.strategy.library import (
    STRATEGY_STATUSES, STRATEGY_TYPES, StrategyAlreadyExists, delete_many,
    delete_saved_strategy, export_library_zip, get_strategy_library_dir,
    list_all_markets, list_all_tags, list_misplaced_files, list_saved_strategies,
    load_strategy_text, record_backtest_result, record_lookahead_result, record_search_result,
    rename_saved_strategy, save_strategy_bytes, save_strategy_metadata,
    save_strategy_path, save_strategy_text, set_strategy_status, set_strategy_tags, status_label,
)

# Display <-> storage-slug maps for the strategy status lifecycle (draft,
# tested_failed, tested_passed, validated, ready_for_demo, ready_for_live)
# -- built once from STRATEGY_STATUSES so every dropdown/listbox/badge
# shows the same human-readable label ("TESTED / FAILED") while the file
# on disk keeps the plain slug ("tested_failed").
STATUS_KEY_TO_LABEL: dict[str, str] = {s: status_label(s) for s in STRATEGY_STATUSES}
STATUS_LABEL_TO_KEY: dict[str, str] = {v: k for k, v in STATUS_KEY_TO_LABEL.items()}
STATUS_LABELS_ORDERED: list[str] = [STATUS_KEY_TO_LABEL[s] for s in STRATEGY_STATUSES]
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.ui.condition_builder import ConditionList
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv
from app.validation.sensitivity import compute_1d_sensitivity, compute_2d_heatmap, list_tunable_parameters
from app.validation.walk_forward_opt import run_walk_forward_optimization
from app.web import live_market
import random

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

# ---------------------------------------------------------------------------
# Theme system -- every color used across the whole app is one of the names
# below, looked up as a plain module global. THEMES holds the full palette
# for each mode; apply_theme() (near the bottom of this block) overwrites
# these globals in place and triggers a full UI rebuild, which is how a
# single Dark/Light toggle reaches every tab, chart, and widget at once
# without threading a theme object through thousands of call sites.
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "BG": "#08090C",
        "PANEL": "#131720",          # lifted slightly off BG so cards read as elevated surfaces
        "PANEL_2": "#171B25",
        "PANEL_3": "#1E232E",
        "PANEL_HOVER": "#242A37",    # hover state for interactive surfaces (buttons, rows)
        "BORDER": "#272C38",
        "BORDER_LIGHT": "#3D4453",
        "TEXT": "#E9EBEF",
        "TEXT_MUTED": "#8D94A3",
        "TEXT_DIM": "#5C6472",
        "METAL": "#B8BDC5",
        "METAL_BRIGHT": "#E7E9ED",
        "LOG_BG": "#0B0D10",         # background for the monospace live-log/output Text widgets
        "GREEN": "#3ED685",
        "RED": "#F0596A",
        "BLUE": "#6FA8FF",
        "AMBER": "#D9A441",
        "ACCENT": "#7C6FFF",
        "ACCENT_HOVER": "#9089FF",
        "ACCENT_DIM": "#332E5C",     # low-opacity-style accent for subtle fills/left-bars
        "ACCENT_INK": "#0C0A16",     # near-black used as text on top of the bright accent
        # Neon accent set -- used for the glowing card borders / ring progress /
        # per-metric coloring on the Dashboard tab, matching the neon-dark
        # reference mockups. Kept separate from the semantic GREEN/RED/AMBER above
        # (which mean pass/fail/warning everywhere else in the app) -- these are
        # purely decorative variety across KPI tiles, the way the mockups give
        # every stat card a different hue rather than making hue mean something.
        "NEON_CYAN": "#00F0FF",
        "NEON_MAGENTA": "#FF2BD6",
        "NEON_LIME": "#B6FF3C",
        "NEON_VIOLET": "#8A5CFF",
        "NEON_AMBER": "#FFB547",
    },
    "light": {
        "BG": "#F4F5F7",
        "PANEL": "#FFFFFF",
        "PANEL_2": "#F1F2F5",
        "PANEL_3": "#E7E9EE",
        "PANEL_HOVER": "#DCE0E8",
        "BORDER": "#D9DCE3",
        "BORDER_LIGHT": "#C3C8D1",
        "TEXT": "#14161B",
        "TEXT_MUTED": "#565D6B",
        "TEXT_DIM": "#8B909C",
        "METAL": "#6B7280",
        "METAL_BRIGHT": "#2F333B",
        "LOG_BG": "#FBFBFC",
        "GREEN": "#0E9B5E",
        "RED": "#D23B52",
        "BLUE": "#2C64D6",
        "AMBER": "#A66A16",
        "ACCENT": "#5B4CE0",
        "ACCENT_HOVER": "#4739C9",
        "ACCENT_DIM": "#E4E1FA",
        "ACCENT_INK": "#FFFFFF",
        # Same decorative role as the dark theme's neon set, deliberately
        # darkened/desaturated from true neon so they stay legible as text
        # and card borders against a near-white background instead of
        # glaring -- same hue identity per tile, tuned for contrast rather
        # than raw brightness.
        "NEON_CYAN": "#0089A3",
        "NEON_MAGENTA": "#B01C93",
        "NEON_LIME": "#5D8A12",
        "NEON_VIOLET": "#6438C9",
        "NEON_AMBER": "#A66A16",
    },
}

_THEME_SETTINGS_FILENAME = "ui_theme.json"


def _load_theme_name() -> str:
    """Never raises -- defaults to 'dark' (this app's original look) if
    nothing saved yet, the settings file is corrupt, or the persistence
    layer isn't reachable for any reason."""
    try:
        import json as _json

        from app.data.storage import get_app_base_dir

        path = get_app_base_dir() / "data" / "config" / _THEME_SETTINGS_FILENAME
        if path.exists():
            name = _json.loads(path.read_text(encoding="utf-8")).get("theme")
            if name in THEMES:
                return name
    except Exception:
        pass
    return "dark"


def _save_theme_name(name: str) -> None:
    try:
        import json as _json

        from app.data.storage import get_app_base_dir

        config_dir = get_app_base_dir() / "data" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / _THEME_SETTINGS_FILENAME).write_text(_json.dumps({"theme": name}), encoding="utf-8")
    except Exception:
        pass  # persistence is a convenience -- never worth crashing the toggle over


def apply_theme(name: str) -> None:
    """Overwrites every color constant below (as module globals) with the
    named theme's palette. Callers must rebuild any already-built widgets
    themselves afterward (see MainWindow._toggle_theme) -- this function
    only updates the source of truth those widgets read their colors
    from; it does not (and structurally cannot, for widgets whose colors
    were baked in at construction time) reach back and recolor anything
    already on screen."""
    if name not in THEMES:
        return
    globals().update(THEMES[name])
    global CURRENT_THEME
    CURRENT_THEME = name
    _save_theme_name(name)


CURRENT_THEME = _load_theme_name()
apply_theme(CURRENT_THEME)

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
    def __init__(self, parent, label, default="", secret=False, width=20):
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

        entry_kwargs = dict(
            textvariable=self.var,
            width=width,
            bg=PANEL_3,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER_LIGHT,
            font=_safe_font(10),
        )
        # secret=True masks the field like a password input -- used for API
        # keys/secrets so they aren't visible over someone's shoulder or in
        # a screen share.
        if secret:
            entry_kwargs["show"] = "\u2022"

        self.entry = Entry(self, **entry_kwargs)
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
    def __init__(self, parent, label, values, default="", width=None):
        super().__init__(parent, bg=PANEL)

        Label(
            self, text=label, width=31, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left")

        self.var = StringVar(value=str(default))
        # width=18 was a fine default back when every combo's options were
        # short labels ("bootstrap", "shuffle", ...); Search Lab's Mode
        # dropdown has long descriptive option text (e.g. "Bulk backtest
        # -- upload multiple Python / PineScript / MQL5 files and run
        # each one") that a fixed width=18 clips hard in both the closed
        # box and ttk's popdown listbox, which sizes off the widget's own
        # width. Auto-sizing to the longest value (capped so it can't blow
        # out the layout) fixes that everywhere a combo is used, not just
        # this one spot -- pass an explicit width to opt back into the
        # old fixed-width behavior for a combo that wants it.
        if width is None:
            longest = max((len(str(v)) for v in values), default=18)
            width = min(60, max(18, longest + 2))
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=values, state="readonly",
            width=width, font=_safe_font(9), style="T58.TCombobox",
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


def _blend_hex(color_hex: str, toward_hex: str, t: float) -> str:
    """Blends color_hex toward toward_hex by fraction t (0=color, 1=toward).
    Standalone version of MainWindow._blend, usable by widgets that aren't
    the main window itself."""
    c = color_hex.lstrip("#")
    b = toward_hex.lstrip("#")
    cr, cg, cb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    br, bg_, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    r = round(cr + (br - cr) * t)
    g = round(cg + (bg_ - cg) * t)
    bl = round(cb + (bb - cb) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


class NeuralProgress(Canvas):
    """A glowing neon loading bar -- a rounded, softly-lit track with an
    animated cyan fill, a "LOADING…." label, and a live percentage
    readout, in the same visual family as this app's reference neon-dark
    mockups. Most call sites in this app are indeterminate (a Search Lab
    generation, a Monte Carlo run, a full pipeline stage -- none report a
    real 0-100% completion), so while running this shows a smoothly
    rising ESTIMATED percentage that eases toward (but never reaches)
    ~96% -- honest about being an estimate, while still giving a much
    more readable sense of "this is progressing" than a bar that just
    bounces back and forth.

    Drop-in replacement for `ttk.Progressbar(mode="indeterminate")`:
    supports the same `.start(interval)` / `.stop()` calls (interval is
    accepted for signature compatibility but this always animates at its
    own fixed frame rate) and the same `.pack(...)` usage -- every
    existing call site swaps in unchanged. Call `.set_progress(pct)`
    instead of/in addition to `.start()` for the rare call site that DOES
    know a real percentage (e.g. "fold 3 of 8") -- it freezes the
    estimate and shows the real number until `.start()` or `.stop()` is
    called again.
    """

    FRAME_MS = 33                     # ~30fps -- smooth without burning CPU on a purely decorative animation
    ESTIMATE_CEILING_PCT = 96.0        # never implies completion on its own -- only stop() does that
    ESTIMATE_TAU_SECONDS = 14.0        # how quickly the estimate eases toward the ceiling
    SWEEP_SECONDS = 1.8                # time for one highlight streak to cross the filled portion
    CORNER_RADIUS = 7

    def __init__(self, parent, height: int = 30, **kwargs):
        super().__init__(parent, height=height, bg=PANEL_2, highlightthickness=0, **kwargs)
        self._height = height
        self._running = False
        self._after_id = None
        self._t0 = 0.0
        self._sweep_phase = 0.0
        self._manual_pct: float | None = None   # set via set_progress(); None means "use the eased estimate"
        self._label = "LOADING...."
        self.bind("<Configure>", lambda _e: self._draw())
        self._draw()  # resting state before the first start()

    def start(self, interval=None):
        if self._running:
            return
        self._running = True
        self._manual_pct = None
        self._t0 = time.monotonic()
        self._sweep_phase = 0.0
        self._tick()

    def set_progress(self, pct: float, label: str | None = None):
        """For the rare call site that knows a real percentage (e.g. fold
        N of M). Freezes the animated estimate at this exact value until
        the next start()/stop(). Starts the sweep animation if not
        already running."""
        self._manual_pct = max(0.0, min(100.0, pct))
        if label is not None:
            self._label = label
        if not self._running:
            self._running = True
            self._sweep_phase = 0.0
            self._tick()
        else:
            self._draw()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._manual_pct = None
        self._label = "LOADING...."
        self._draw()

    def _current_pct(self) -> float:
        if self._manual_pct is not None:
            return self._manual_pct
        elapsed = max(0.0, time.monotonic() - self._t0)
        return self.ESTIMATE_CEILING_PCT * (1.0 - math.exp(-elapsed / self.ESTIMATE_TAU_SECONDS))

    def _tick(self):
        if not self._running:
            return
        self._sweep_phase = (self._sweep_phase + self.FRAME_MS / 1000.0 / self.SWEEP_SECONDS) % 1.0
        self._draw()
        self._after_id = self.after(self.FRAME_MS, self._tick)

    def _rounded_rect_points(self, x0, y0, x1, y1, r):
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        return [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]

    def _draw(self):
        self.delete("all")
        width = max(self.winfo_width(), 80)
        height = self._height
        pad = 3
        x0, y0, x1, y1 = pad, pad, width - pad, height - pad
        r = self.CORNER_RADIUS
        active = self._running

        # Outer glow halo behind the track -- same nested-shrinking-alpha
        # trick GlowCard/RingProgress use elsewhere, muted to almost
        # nothing when idle so a bank of a dozen of these on one tab
        # doesn't read as a wall of neon.
        if active:
            for i, alpha in ((3, 0.05), (2, 0.10), (1, 0.18)):
                self.create_polygon(
                    self._rounded_rect_points(x0 - i, y0 - i, x1 + i, y1 + i, r + i),
                    smooth=True, fill="", outline=_blend_hex(PANEL_2, NEON_CYAN, alpha), width=1.4,
                )

        track_color = _blend_hex(PANEL_2, NEON_CYAN, 0.06)
        border_color = _blend_hex(PANEL_2, NEON_CYAN, 0.85 if active else 0.35)
        self.create_polygon(self._rounded_rect_points(x0, y0, x1, y1, r), smooth=True,
                             fill=track_color, outline=border_color, width=1.4)

        pct = self._current_pct() if active else 0.0
        inner_x0, inner_y0, inner_x1, inner_y1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2
        fill_w = max(0.0, (inner_x1 - inner_x0) * (pct / 100.0))
        if fill_w > 1:
            fx1 = inner_x0 + fill_w
            fill_r = min(r - 1, fill_w / 2, (inner_y1 - inner_y0) / 2)
            # Soft glow bleeding just past the fill's own edge, then the
            # solid glowing fill itself.
            for i, alpha in ((4, 0.10), (2, 0.22)):
                self.create_polygon(
                    self._rounded_rect_points(inner_x0 - i, inner_y0 - i, fx1 + i, inner_y1 + i, fill_r + i),
                    smooth=True, fill=_blend_hex(track_color, NEON_CYAN, alpha), outline="",
                )
            self.create_polygon(
                self._rounded_rect_points(inner_x0, inner_y0, fx1, inner_y1, fill_r),
                smooth=True, fill=_blend_hex(PANEL_2, NEON_CYAN, 0.75), outline="",
            )
            # A brighter highlight streak sliding smoothly through the
            # filled region -- the "make it look smooth" motion cue,
            # independent of the estimate's own (slow, easing) growth.
            streak_w = max(18.0, fill_w * 0.22)
            streak_center = inner_x0 + self._sweep_phase * (fill_w + streak_w) - streak_w / 2
            sx0 = max(inner_x0, streak_center - streak_w / 2)
            sx1 = min(fx1, streak_center + streak_w / 2)
            if sx1 > sx0:
                self.create_polygon(
                    self._rounded_rect_points(sx0, inner_y0, sx1, inner_y1, fill_r),
                    smooth=True, fill=_blend_hex(PANEL_2, METAL_BRIGHT, 0.55), outline="",
                )

        label_color = TEXT if active else TEXT_DIM
        self.create_text(x0 + 10, height / 2, text=self._label, fill=label_color,
                          font=_safe_font(8, "bold"), anchor="w")
        pct_text = f"{pct:.0f}%" if active else ""
        if pct_text:
            self.create_text(x1 - 10, height / 2, text=pct_text, fill=NEON_CYAN,
                              font=_safe_font(8, "bold"), anchor="e")



class GlowCard(Canvas):
    """A rounded panel with a soft neon glow border -- the "gradient
    border card" look from the neon dashboard reference mockups. Tkinter
    has no native box-shadow or rounded Frame, so this draws the panel
    itself: a smoothed rounded-rect polygon for the body, a crisp accent-
    colored edge, and an outward-fading halo built from the same nested/
    dimming-shapes trick NeuralProgress uses for its pulse glow.

    Usage: build widgets into `.body` (a plain Frame) exactly like any
    other container -- `GlowCard(parent, accent=NEON_CYAN).body` is a drop
    -in replacement for `Frame(parent, bg=PANEL_2)` wherever a panel
    currently gets a flat `highlightbackground=BORDER` outline.
    """

    CORNER_RADIUS = 14

    def __init__(self, parent, accent=None, glow=True, **kwargs):
        # accent's default is resolved HERE (call time), not bound into
        # the function signature at module-load time, so it always
        # reflects whichever theme is currently active -- a literal
        # `accent=ACCENT` default would freeze at the color ACCENT held
        # when this module was first imported and never follow a later
        # theme toggle.
        super().__init__(parent, bg=BG, highlightthickness=0, **kwargs)
        self.accent = accent if accent is not None else ACCENT
        self.glow = glow
        self.body = Frame(self, bg=PANEL_2)
        self._window_id = None
        self.bind("<Configure>", lambda e: self._redraw(e.width, e.height))

    def set_accent(self, accent: str):
        self.accent = accent
        self._redraw(self.winfo_width(), self.winfo_height())

    def _rounded_rect(self, x0, y0, x1, y1, r, **kwargs):
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        points = [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _redraw(self, width, height):
        if width < 8 or height < 8:
            return
        self.delete("all")
        r = self.CORNER_RADIUS
        pad = 4  # room for the glow halo to breathe past the card's own edge

        if self.glow:
            for i, alpha in ((3, 0.05), (2, 0.10), (1, 0.20)):
                self._rounded_rect(
                    pad - i * 2, pad - i * 2, width - pad + i * 2, height - pad + i * 2,
                    r + i * 2, outline=_blend_hex(BG, self.accent, alpha), fill="", width=2,
                )

        self._rounded_rect(pad, pad, width - pad, height - pad, r, fill=PANEL_2, outline=self.accent, width=1)

        inner_w, inner_h = max(width - 2 * pad - 2, 1), max(height - 2 * pad - 2, 1)
        if self._window_id is None:
            self._window_id = self.create_window(
                pad + 1, pad + 1, anchor="nw", window=self.body, width=inner_w, height=inner_h,
            )
        else:
            self.coords(self._window_id, pad + 1, pad + 1)
            self.itemconfigure(self._window_id, width=inner_w, height=inner_h)


class RingProgress(Canvas):
    """A glowing circular percentage ring -- the donut readout used for
    "progress toward a target" across the neon dashboard mockups, in place
    of a flat horizontal progress bar. Call `.set(pct)` to update.
    """

    def __init__(self, parent, size=88, thickness=9, accent=None, track=None, **kwargs):
        # Same call-time-default reasoning as GlowCard above -- resolved
        # here rather than bound into the signature, so this keeps
        # following the active theme rather than freezing at import time.
        super().__init__(parent, width=size, height=size, bg=PANEL_2, highlightthickness=0, **kwargs)
        self.size = size
        self.thickness = thickness
        self.accent = accent if accent is not None else ACCENT
        self.track = track if track is not None else BORDER
        self._pct = 0.0
        self._draw()

    def set(self, pct: float, accent: str | None = None):
        self._pct = max(0.0, min(100.0, pct))
        if accent is not None:
            self.accent = accent
        self._draw()

    def _draw(self):
        self.delete("all")
        s, t = self.size, self.thickness
        pad = t / 2 + 5
        for radius_pad, alpha in ((6, 0.05), (3, 0.14)):
            self.create_oval(
                pad - radius_pad, pad - radius_pad, s - pad + radius_pad, s - pad + radius_pad,
                outline=_blend_hex(PANEL_2, self.accent, alpha), width=t + radius_pad,
            )
        self.create_oval(pad, pad, s - pad, s - pad, outline=self.track, width=t)
        extent = -359.9 * (self._pct / 100.0)  # 359.9, not 360 -- a full-circle arc draws nothing in Tk
        if abs(extent) > 0.5:
            self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=extent, style="arc",
                             outline=self.accent, width=t)
        self.create_text(s / 2, s / 2, text=f"{self._pct:.0f}%", fill=self.accent,
                          font=_safe_font(max(int(s * 0.16), 9), "bold"))


_ADAPTIVE_RISK_TRIGGERS = [
    "consecutive_losses", "daily_loss_pct", "daily_profit_pct", "progress_to_target_pct",
]
_ADAPTIVE_RISK_TRIGGER_HELP = {
    "consecutive_losses": "threshold = number of losing trades in a row (today)",
    "daily_loss_pct": "threshold = % of initial balance realized-lost so far today",
    "daily_profit_pct": "threshold = % of initial balance realized-gained so far today",
    "progress_to_target_pct": "threshold = % of the way from 0 to the profit target, all-time",
}


class _AdaptiveRuleDialog(Toplevel):
    """Small modal for adding one AdaptiveRiskRule -- a trigger type, a
    threshold, and a size multiplier. Used from the Risk & Execution tab's
    Adaptive Risk section; see app.backtest.adaptive_risk for the trigger
    semantics."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add adaptive risk rule")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)
        self.result: AdaptiveRiskRule | None = None

        self.trigger_var = StringVar(value=_ADAPTIVE_RISK_TRIGGERS[0])
        self.threshold_var = StringVar(value="2")
        self.multiplier_var = StringVar(value="0.5")

        pad = dict(padx=14, pady=(10, 2))
        Label(self, text="Trigger", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        trigger_combo = ttk.Combobox(
            self, textvariable=self.trigger_var, values=_ADAPTIVE_RISK_TRIGGERS,
            state="readonly", width=28, font=_safe_font(9), style="T58.TCombobox",
        )
        trigger_combo.pack(fill="x", padx=14)
        trigger_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_help())

        self.help_label = Label(
            self, text=_ADAPTIVE_RISK_TRIGGER_HELP[self.trigger_var.get()],
            bg=PANEL, fg=AMBER, font=_safe_font(8), wraplength=280, justify="left", anchor="w",
        )
        self.help_label.pack(fill="x", padx=14, pady=(4, 6))

        Label(self, text="Threshold", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        Entry(
            self, textvariable=self.threshold_var, bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, font=_safe_font(10),
        ).pack(fill="x", padx=14, ipady=4)

        Label(self, text="Risk multiplier (e.g. 0.5 = half size)", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        Entry(
            self, textvariable=self.multiplier_var, bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, font=_safe_font(10),
        ).pack(fill="x", padx=14, ipady=4)

        btn_row = Frame(self, bg=PANEL)
        btn_row.pack(fill="x", padx=14, pady=14)
        Button(btn_row, text="Cancel", command=self._cancel).pack(side="right", padx=(8, 0))
        Button(btn_row, text="Add", command=self._confirm).pack(side="right")

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _update_help(self):
        self.help_label.config(text=_ADAPTIVE_RISK_TRIGGER_HELP[self.trigger_var.get()])

    def _confirm(self):
        try:
            threshold = float(self.threshold_var.get())
            multiplier = float(self.multiplier_var.get())
            self.result = AdaptiveRiskRule(
                trigger=self.trigger_var.get(), threshold=threshold, risk_multiplier=multiplier,
            )
        except (ValueError, AdaptiveRiskError) as exc:
            messagebox.showerror("Invalid rule", str(exc), parent=self)
            return
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    @classmethod
    def ask(cls, parent) -> AdaptiveRiskRule | None:
        dialog = cls(parent)
        parent.wait_window(dialog)
        return dialog.result


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

        # State that must survive a theme toggle's full widget rebuild
        # (see _toggle_theme) lives directly on self, set once here --
        # _build_ui() only ever CREATES widgets, it never resets this.
        self.csv_path: str | None = None
        self.csv_paths: list[str] = []
        self.strategy_py_path: str | None = None
        self._active_library_strategy: tuple[str, str] | None = None
        # Staged queue for TEST SELECTED (BATCH) -- ADD SELECTED TO BATCH QUEUE
        # copies highlighted library rows in here instead of running
        # immediately, so Prop Rules (03) and Risk (04) can be set up
        # *after* picking strategies but *before* anything actually runs.
        # Holds StoredStrategy objects (each already carries its own
        # strategy_type, so mixing Python/PineScript/MQL5 in one queue is fine).
        self._batch_queue: list = []
        self.strategy_mode = StringVar(value="manual")

        self._build_ui()

    def _toggle_theme(self):
        """Dark/Light switch in the header. Repaints the ENTIRE app: color
        constants are plain module globals referenced by name throughout
        this file (not threaded through as a theme object), so the only
        reliable way to make thousands of already-built widgets and
        hand-drawn Canvas charts pick up a new palette is to update those
        globals via apply_theme() and then throw away and rebuild every
        widget from scratch -- reconfiguring each one in place would mean
        separately re-deriving the right color for every Label/Canvas/
        chart/glow effect in the app, which is exactly what building them
        fresh already does correctly. Whichever tab was open stays open
        across the rebuild."""
        new_theme = "light" if CURRENT_THEME == "dark" else "dark"
        current_page = getattr(self, "active_page", "dashboard")
        apply_theme(new_theme)
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()
        self._show_page(current_page)

    def _build_ui(self):
        self.root.configure(bg=BG)
        self._configure_styles()
        self._pump_splash("Configuring theme...")

        # Main application shell.
        shell = Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        self._build_header(shell)
        self._pump_splash("Building navigation...")

        # ---------------------------------------------------------------
        # Sidebar navigation + page switcher (replaces the old top-tab
        # ttk.Notebook). Each page is a plain Frame; only one is gridded
        # into the content area at a time via _show_page(). This gives us
        # full control over the nav's look (icons, active glow, grouping)
        # that ttk.Notebook can't offer, especially for vertical tabs.
        # ---------------------------------------------------------------
        body = Frame(shell, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        # The sidebar itself scrolls: with 18 tabs + section dividers, the
        # full nav list is taller than the sidebar's available height on
        # this app's default/minimum window size, and a fixed (non-
        # scrolling) sidebar simply clips whatever doesn't fit off the
        # bottom -- every tab must stay reachable no matter the window
        # size, so a mouse-wheel-scrollable canvas backs the nav list
        # instead of relying on padding alone to make it fit.
        self.sidebar = Frame(body, bg=PANEL, width=196)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self._sidebar_canvas = Canvas(self.sidebar, bg=PANEL, highlightthickness=0)
        self._sidebar_canvas.pack(side="left", fill="both", expand=True)
        self._sidebar_inner = Frame(self._sidebar_canvas, bg=PANEL)
        self._sidebar_window_id = self._sidebar_canvas.create_window(
            (0, 0), window=self._sidebar_inner, anchor="nw",
        )
        self._sidebar_inner.bind(
            "<Configure>", lambda _e: self._sidebar_canvas.configure(scrollregion=self._sidebar_canvas.bbox("all")),
        )
        self._sidebar_canvas.bind(
            "<Configure>", lambda e: self._sidebar_canvas.itemconfig(self._sidebar_window_id, width=e.width),
        )

        def _sidebar_wheel(event):
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            self._sidebar_canvas.yview_scroll(delta, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._sidebar_canvas.bind(seq, _sidebar_wheel)
            self._sidebar_inner.bind(seq, _sidebar_wheel)

        self.content = Frame(body, bg=BG)
        self.content.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.tab_dashboard = Frame(self.content, bg=BG)
        self.tab_manual = Frame(self.content, bg=BG)
        self.tab_data = Frame(self.content, bg=BG)
        self.tab_strategy = Frame(self.content, bg=BG)
        self.tab_prop = Frame(self.content, bg=BG)
        self.tab_risk = Frame(self.content, bg=BG)
        self.tab_run = Frame(self.content, bg=BG)
        self.tab_refine = Frame(self.content, bg=BG)
        self.tab_search = Frame(self.content, bg=BG)
        self.tab_wfo = Frame(self.content, bg=BG)
        self.tab_cpcv = Frame(self.content, bg=BG)
        self.tab_sensitivity = Frame(self.content, bg=BG)
        self.tab_portfolio = Frame(self.content, bg=BG)
        self.tab_multiobj = Frame(self.content, bg=BG)
        self.tab_wfga = Frame(self.content, bg=BG)
        self.tab_ensemble = Frame(self.content, bg=BG)
        self.tab_fullpipeline = Frame(self.content, bg=BG)
        self.tab_forwardtest = Frame(self.content, bg=BG)
        self.tab_deploylive = Frame(self.content, bg=BG)
        self.tab_livemarket = Frame(self.content, bg=BG)
        self.tab_genstrat = Frame(self.content, bg=BG)
        self.tab_evolution = Frame(self.content, bg=BG)
        self.tab_researchagent = Frame(self.content, bg=BG)

        for f in (
            self.tab_dashboard, self.tab_manual, self.tab_data, self.tab_strategy, self.tab_prop,
            self.tab_risk, self.tab_run, self.tab_refine, self.tab_search,
            self.tab_wfo, self.tab_cpcv, self.tab_sensitivity, self.tab_portfolio,
            self.tab_multiobj, self.tab_wfga, self.tab_ensemble, self.tab_fullpipeline,
            self.tab_forwardtest, self.tab_deploylive, self.tab_livemarket, self.tab_genstrat,
            self.tab_evolution,
        ):
            f.place(in_=self.content, x=0, y=0, relwidth=1, relheight=1)

        # Every entry is (key, icon, label, frame, color). `icon` is kept
        # in the tuple shape for backward compatibility but no longer
        # rendered (see _build_sidebar_nav) -- a leading glyph AND a
        # leading number on every row read as cluttered, so this sidebar
        # picks one signal (the workflow number, where one exists) rather
        # than both. A divider entry has key=None; its `label` slot holds
        # the section header text shown above that group (小-caps style,
        # letter-spaced) instead of a bare line -- every group is named
        # rather than just visually separated, which is what actually
        # makes a long list like this read as organized instead of messy.
        self._nav_items = [
            (None, None, "OVERVIEW", None, None),
            ("dashboard", "", "Dashboard", self.tab_dashboard, NEON_VIOLET),
            ("manual", "", "User Manual", self.tab_manual, METAL_BRIGHT),

            (None, None, "WORKFLOW", None, None),
            ("data", "", "01  Data", self.tab_data, NEON_CYAN),
            ("strategy", "", "02  Strategy", self.tab_strategy, NEON_CYAN),
            ("prop", "", "03  Prop Rules", self.tab_prop, NEON_CYAN),
            ("risk", "", "04  Risk", self.tab_risk, NEON_CYAN),
            ("run", "", "05  Run & Report", self.tab_run, NEON_CYAN),
            ("refine", "", "06  Refinement", self.tab_refine, NEON_CYAN),
            ("search", "", "07  Search Lab", self.tab_search, NEON_CYAN),

            (None, None, "VALIDATION LAB", None, None),
            ("wfo", "", "08  Walk-Forward Opt", self.tab_wfo, BLUE),
            ("cpcv", "", "09  CPCV / PBO", self.tab_cpcv, BLUE),
            ("sensitivity", "", "10  Sensitivity", self.tab_sensitivity, BLUE),
            ("portfolio", "", "11  Portfolio", self.tab_portfolio, BLUE),
            ("multiobj", "", "12  Multi-Objective", self.tab_multiobj, BLUE),
            ("wfga", "", "13  Walk-Forward GA", self.tab_wfga, BLUE),

            (None, None, "FINDING AN EDGE", None, None),
            ("ensemble", "", "14  Ensemble", self.tab_ensemble, NEON_MAGENTA),
            ("genstrat", "", "Generate Strategies (AI)", self.tab_genstrat, NEON_MAGENTA),
            ("evolution", "", "Evolution Lab", self.tab_evolution, NEON_MAGENTA),

            (None, None, "ALL-IN-ONE", None, None),
            ("fullpipeline", "", "15  Full Pipeline", self.tab_fullpipeline, NEON_AMBER),

            (None, None, "LIVE TRADING", None, None),
            ("forwardtest", "", "Live Demo Test", self.tab_forwardtest, NEON_LIME),
            ("deploylive", "", "Deploy Live", self.tab_deploylive, RED),
            ("livemarket", "", "Live Market", self.tab_livemarket, NEON_CYAN),

            (None, None, "AI RESEARCH", None, None),
            ("researchagent", "", "18  Research Agent", self.tab_researchagent, NEON_MAGENTA),
        ]
        self._tab_frame_by_key = {k: frame for k, _icon, _label, frame, _color in self._nav_items if k}
        self._nav_buttons: dict[str, Label] = {}
        self._build_sidebar_nav()
        self.active_page = "dashboard"

        for label, builder in (
            ("Dashboard", self._build_dashboard_tab),
            ("Manual builder", self._build_manual_tab),
            ("Data", self._build_data_tab),
            ("Strategy", self._build_strategy_tab),
            ("Prop rules", self._build_prop_tab),
            ("Risk", self._build_risk_tab),
            ("Run", self._build_run_tab),
            ("Refinement", self._build_refine_tab),
            ("Search Lab", self._build_search_tab),
            ("Walk-forward", self._build_wfo_tab),
            ("CPCV", self._build_cpcv_tab),
            ("Sensitivity", self._build_sensitivity_tab),
            ("Portfolio", self._build_portfolio_tab),
            ("Multi-objective", self._build_multiobj_tab),
            ("Walk-forward GA", self._build_wfga_tab),
            ("Ensemble", self._build_ensemble_tab),
            ("Strategy generator", self._build_generate_strategies_tab),
            ("Evolution Lab", self._build_evolution_lab_tab),
            ("Full Pipeline", self._build_full_pipeline_tab),
            ("Forward Test", self._build_forward_test_tab),
            ("Deploy Live", self._build_deploy_live_tab),
            ("Live Market", self._build_live_market_tab),
            ("Research Agent", self._build_research_agent_tab),
        ):
            self._pump_splash(f"Loading {label}...")
            builder()

        self._show_page("dashboard")

    def _pump_splash(self, status: str) -> None:
        """Best-effort: updates the boot splash's status text and pumps
        the Tk event loop once, so the splash's glow animation actually
        animates through _build_ui()'s otherwise fully synchronous,
        several-second widget construction instead of freezing on its
        first frame and only reappearing once everything is already
        built. Silently does nothing if there's no splash (e.g. this
        MainWindow wasn't created via launch(), such as in tests)."""
        splash = getattr(self.root, "_t58_splash", None)
        if splash is None:
            return
        try:
            splash.set_status(status)
            self.root.update()
        except Exception:
            pass

    def _build_sidebar_nav(self):
        def _wheel(event):
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            self._sidebar_canvas.yview_scroll(delta, "units")

        first_section = True
        for key, _icon, label, frame, color in self._nav_items:
            if key is None:
                # A named section header (small-caps, letter-spaced, muted)
                # rather than a bare line -- every group is labeled, which
                # is what makes a long list like this read as organized
                # sections instead of one long undifferentiated list.
                Label(
                    self._sidebar_inner, text=" ".join(label.upper()), bg=PANEL, fg=TEXT_DIM,
                    font=_safe_font(7, "bold"), anchor="w",
                ).pack(fill="x", padx=16, pady=(14 if not first_section else 4, 4))
                first_section = False
                continue
            row = Frame(self._sidebar_inner, bg=PANEL, cursor="hand2")
            row.pack(fill="x", padx=8, pady=1)
            # height=1 is deliberate: a Tkinter Canvas with no explicit
            # height defaults to a large platform size (the actual bug
            # that made every sidebar row balloon in height and pushed
            # tabs off the bottom of the screen). pack(fill="y") below
            # still correctly stretches this to match the row's real
            # height, which is set by the label -- height=1 just stops
            # the canvas's OWN natural size from inflating that row in
            # the first place.
            accent = Canvas(row, bg=PANEL, width=6, height=1, highlightthickness=0)
            accent.pack(side="left", fill="y")
            accent.bind("<Configure>", lambda _e, k=key: self._draw_nav_accent(k))
            lbl = Label(
                row, text=f"   {label}", bg=PANEL, fg=TEXT_MUTED,
                font=_safe_font(9), anchor="w", padx=6, pady=6,
            )
            lbl.pack(side="left", fill="x", expand=True)
            for widget in (row, accent, lbl):
                widget.bind("<Button-1>", lambda _e, k=key: self._show_page(k))
                widget.bind("<Enter>", lambda _e, k=key: self._on_nav_hover(k, True))
                for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(seq, _wheel)
                widget.bind("<Leave>", lambda _e, k=key: self._on_nav_hover(k, False))
            self._nav_buttons[key] = (row, accent, lbl, color)

    def _draw_nav_accent(self, key: str, state: str | None = None):
        """Paints one sidebar row's accent strip -- a soft, per-tab-colored
        glow bar rather than one flat purple line for every tab. `state`
        is "idle" / "hover" / "active"; omitted (e.g. on a <Configure>
        resize event) means "whatever this row's current state already
        is," re-derived from self.active_page.
        """
        if key not in self._nav_buttons:
            return
        row, accent, _lbl, color = self._nav_buttons[key]
        if state is None:
            state = "active" if key == getattr(self, "active_page", "dashboard") else "idle"
        accent.delete("all")
        w = max(accent.winfo_width(), 8)
        h = max(accent.winfo_height(), 24)
        bg = str(row.cget("bg"))
        accent.configure(bg=bg)
        cx = w / 2

        if state == "active":
            # A soft outward halo (matching the GlowCard/NeuralProgress
            # glow technique elsewhere in this app) behind a solid,
            # full-brightness core bar -- reads as "this tab is lit up in
            # its own color," not just "there's a purple line here."
            for half_width, alpha in ((3.6, 0.12), (2.6, 0.22), (1.8, 0.4)):
                accent.create_rectangle(
                    cx - half_width, 2, cx + half_width, h - 2,
                    fill=_blend_hex(bg, color, alpha), outline="",
                )
            accent.create_rectangle(cx - 1.4, 3, cx + 1.4, h - 3, fill=color, outline="")
        elif state == "hover":
            accent.create_rectangle(cx - 2.2, 3, cx + 2.2, h - 3, fill=_blend_hex(bg, color, 0.55), outline="")
        else:
            # Idle rows still carry a faint, permanent tint of their own
            # color instead of going fully gray -- enough to give each
            # section of the sidebar its own quiet identity at a glance,
            # without competing with whichever tab is actually active.
            accent.create_rectangle(cx - 1.0, 4, cx + 1.0, h - 4, fill=_blend_hex(bg, color, 0.32), outline="")

    def _on_nav_hover(self, key, entering):
        if key == self.active_page:
            return
        row, accent, lbl, color = self._nav_buttons[key]
        bg = PANEL_3 if entering else PANEL
        row.configure(bg=bg)
        lbl.configure(bg=bg, fg=_blend_hex(TEXT_MUTED, color, 0.7) if entering else TEXT_MUTED)
        self._draw_nav_accent(key, "hover" if entering else "idle")

    def _show_page(self, key: str):
        self.active_page = key
        for k, (row, accent, lbl, color) in self._nav_buttons.items():
            active = k == key
            bg = PANEL_2 if active else PANEL
            row.configure(bg=bg)
            lbl.configure(bg=bg, fg=color if active else TEXT_MUTED)
            self._draw_nav_accent(k, "active" if active else "idle")
        for k, _icon, _label, frame, _color in self._nav_items:
            if k == key:
                frame.lift()
        if key == "dashboard":
            self._refresh_dashboard()

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

        top_row = Frame(right, bg=BG)
        top_row.pack(anchor="e", pady=(13, 0))

        theme_label = "LIGHT" if CURRENT_THEME == "dark" else "DARK"
        theme_btn = Label(
            top_row,
            text=f"\u25D1  {theme_label} MODE",
            bg=PANEL_2,
            fg=TEXT_MUTED,
            font=_safe_font(8, "bold"),
            padx=10,
            pady=5,
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
            cursor="hand2",
        )
        theme_btn.pack(side="left", padx=(0, 8))
        theme_btn.bind("<Button-1>", lambda _e: self._toggle_theme())
        theme_btn.bind("<Enter>", lambda _e: theme_btn.configure(bg=PANEL_HOVER, fg=TEXT))
        theme_btn.bind("<Leave>", lambda _e: theme_btn.configure(bg=PANEL_2, fg=TEXT_MUTED))

        Label(
            top_row,
            text="MVP",
            bg=PANEL_2,
            fg=ACCENT_HOVER,
            font=_safe_font(8, "bold"),
            padx=12,
            pady=5,
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
        ).pack(side="left")

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
        inner Frame to build content into. Used for every tab in the app.

        Every tab frame (self.tab_dashboard, self.tab_data, ...) is
        `.place()`-d on top of every other one at the exact same (x=0, y=0,
        relwidth=1, relheight=1) rectangle within self.content, and
        `_show_page` switches between them with `.lift()` -- so a hidden
        tab's canvas is still `winfo_ismapped()` and reports the exact same
        on-screen bounding box as whichever tab actually happens to be on
        top. A wheel handler that dispatches by "which registered canvas's
        bounding box contains the cursor" therefore always resolves to
        whichever canvas was registered FIRST, regardless of which page is
        actually visible -- it can never correctly reach any tab but that
        one. Keying the dispatch off `self.active_page` (which `_show_page`
        always keeps current) instead of screen geometry is correct for
        this app's actual place()+lift() tab-switching, and also means the
        cursor doesn't need to be over any particular widget for the wheel
        to work -- anywhere over the content area scrolls the active tab.
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

        if not hasattr(self, "_scroll_canvas_by_tab"):
            self._scroll_canvas_by_tab = {}
        self._scroll_canvas_by_tab[parent] = canvas

        def _dispatch_wheel(_event, delta):
            frame = self._tab_frame_by_key.get(self.active_page)
            target = self._scroll_canvas_by_tab.get(frame) if frame is not None else None
            if target is not None:
                try:
                    target.yview_scroll(int(delta), "units")
                except Exception:
                    pass

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
                wraplength=820,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 8))

        return box

    # -----------------------------------------------------------------------
    # Dashboard — live stats across every strategy run through the app
    # -----------------------------------------------------------------------

    @staticmethod
    def _blend(color_hex: str, toward_hex: str, t: float) -> str:
        """Blend color_hex toward toward_hex by fraction t (0=color, 1=toward).
        Tkinter's Canvas has no real alpha compositing, so this is how the
        glow halos fake a soft falloff against the known dark background."""
        c = color_hex.lstrip("#")
        b = toward_hex.lstrip("#")
        cr, cg, cb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        br, bg_, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
        r = round(cr + (br - cr) * t)
        g = round(cg + (bg_ - cg) * t)
        bl = round(cb + (bb - cb) * t)
        return f"#{r:02x}{g:02x}{bl:02x}"

    def _glow_line(self, canvas: Canvas, points, color, width=2):
        if len(points) < 4:
            return
        for halo_width, t in ((7, 0.75), (5, 0.55), (3.2, 0.3)):
            canvas.create_line(*points, fill=self._blend(color, PANEL_2, t), width=halo_width, smooth=True)
        canvas.create_line(*points, fill=color, width=width, smooth=True)

    def _glow_dot(self, canvas: Canvas, x, y, r, color, ring_color=None):
        for dr, t in ((r * 2.2, 0.8), (r * 1.6, 0.55)):
            blended = self._blend(color, PANEL_2, t)
            canvas.create_oval(x - dr, y - dr, x + dr, y + dr, fill=blended, outline="")
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=ring_color or color, width=1.5)

    def _stat_card(self, parent, label, value, color=None, accent=None):
        """A single glowing neon KPI tile -- `accent` controls the card's
        border/halo color (defaults to matching `color`, the value text
        color), so callers that only cared about text color before still
        work unchanged. `color`'s default is resolved here (call time),
        not bound into the signature, so it follows theme toggles.

        `value` can be a short number (a Sharpe ratio, a count) or a long
        strategy name (the Leader card) -- rather than one fixed font
        size and a card that clips anything past its edge, long text
        drops to a smaller size, wraps within the card, and the card
        itself grows a bit taller so the FULL value is always visible
        instead of being cut off mid-word.
        """
        color = color if color is not None else ACCENT_HOVER
        text = str(value)
        if len(text) <= 10:
            value_font_size, card_height = 20, 92
        elif len(text) <= 22:
            value_font_size, card_height = 15, 108
        else:
            value_font_size, card_height = 12, 124
        card = GlowCard(parent, accent=accent or color, height=card_height)
        Label(card.body, text=label.upper(), bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(12, 3)
        )
        Label(
            card.body, text=text, bg=PANEL_2, fg=(accent or color), font=_safe_font(value_font_size, "bold"),
            wraplength=190, justify="left", anchor="w",
        ).pack(anchor="w", fill="x", padx=14, pady=(0, 10))
        return card

    def _ring_stat_card(self, parent, label, pct, accent):
        """A KPI tile that shows its value as a glowing ring instead of
        flat text -- the "progress toward a target" donut readout from the
        reference mockups."""
        card = GlowCard(parent, accent=accent, height=92)
        row = Frame(card.body, bg=PANEL_2)
        row.pack(fill="both", expand=True, padx=10, pady=8)
        ring = RingProgress(row, size=68, thickness=7, accent=accent)
        ring.pack(side="left")
        ring.set(pct)
        Label(
            row, text=label.upper(), bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8, "bold"),
            wraplength=90, justify="left",
        ).pack(side="left", padx=(10, 0), anchor="w")
        return card

    def _draw_neural_background(self, canvas, width, height):
        """A soft, glowing node-link pattern -- purely decorative, in the
        spirit of a knowledge-graph view (Obsidian's graph view, etc).
        Deliberately muted (every color blended heavily toward BG) so it
        reads as ambient texture behind/around the Dashboard's real
        content rather than competing with it. Uses a fixed seed so the
        pattern is stable across resizes and relaunches rather than
        jittering into a new random layout every time.
        """
        canvas.delete("neural_bg")
        if width < 20 or height < 20:
            return
        rng = random.Random(20260830)
        palette = [NEON_CYAN, NEON_VIOLET, NEON_MAGENTA, NEON_LIME, BLUE]

        nodes: list[tuple[float, float, str, bool]] = []
        edges: list[tuple[float, float, float, float, str]] = []
        n_clusters = max(5, int((width * height) / 40000))
        for _ in range(n_clusters):
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            color = rng.choice(palette)
            nodes.append((cx, cy, color, True))
            for _ in range(rng.randint(3, 9)):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(26, 100)
                lx, ly = cx + dist * math.cos(angle), cy + dist * math.sin(angle)
                nodes.append((lx, ly, color, False))
                edges.append((cx, cy, lx, ly, color))
                if rng.random() < 0.3:
                    angle2 = rng.uniform(0, 2 * math.pi)
                    dist2 = rng.uniform(14, 40)
                    lx2, ly2 = lx + dist2 * math.cos(angle2), ly + dist2 * math.sin(angle2)
                    nodes.append((lx2, ly2, color, False))
                    edges.append((lx, ly, lx2, ly2, color))
        for _ in range(n_clusters * 5):
            nodes.append((rng.uniform(0, width), rng.uniform(0, height), rng.choice(palette), False))

        for x1, y1, x2, y2, color in edges:
            canvas.create_line(x1, y1, x2, y2, fill=_blend_hex(BG, color, 0.11), width=1, tags="neural_bg")
        for x, y, color, is_hub in nodes:
            if is_hub:
                for dr, t in ((5.5, 0.85), (3.6, 0.6)):
                    canvas.create_oval(
                        x - dr, y - dr, x + dr, y + dr, fill=_blend_hex(BG, color, 1 - t), outline="",
                        tags="neural_bg",
                    )
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=_blend_hex(BG, color, 0.75), outline="", tags="neural_bg")
            else:
                r = rng.uniform(1.1, 2.2)
                canvas.create_oval(
                    x - r, y - r, x + r, y + r, fill=_blend_hex(BG, color, 0.42), outline="", tags="neural_bg",
                )
        canvas.tag_lower("neural_bg")

    def _build_dashboard_scrollable(self, parent) -> Frame:
        """Same scroll-and-wheel contract as _scrollable() (registers into
        the shared self._scroll_canvas_by_tab dispatch table, so the
        existing global mouse-wheel binding just works here too) but the
        backing canvas also carries the soft glowing neural background
        above, with the real content inset by MARGIN on every side so
        that background is genuinely visible framing the page -- Tkinter
        widgets are fully opaque, so showing the pattern truly BEHIND
        every card individually would need a much larger rewrite of this
        tab's whole layout into separately-positioned canvas windows.
        Kept as its own method (rather than changing the shared
        _scrollable()) so this stays purely additive for the one tab
        that asked for it.
        """
        MARGIN = 48
        outer = Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        inner = Frame(canvas, bg=BG)

        window_id = canvas.create_window((MARGIN, MARGIN), window=inner, anchor="nw")

        def _sync_scrollregion(_e=None):
            content_h = inner.winfo_reqheight()
            canvas_w = max(canvas.winfo_width(), 200)
            canvas.configure(scrollregion=(0, 0, canvas_w, content_h + MARGIN * 2))
            self._draw_neural_background(canvas, canvas_w, max(canvas.winfo_height(), content_h + MARGIN * 2))

        def _on_canvas_configure(e):
            canvas.itemconfig(window_id, width=max(1, e.width - MARGIN * 2))
            _sync_scrollregion()

        inner.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if not hasattr(self, "_scroll_canvas_by_tab"):
            self._scroll_canvas_by_tab = {}
        self._scroll_canvas_by_tab[parent] = canvas
        return inner

    def _build_dashboard_tab(self):
        f = self.tab_dashboard
        self._page_header(
            f, "OVERVIEW", "Dashboard",
            "Live stats across every strategy that has been run through the app -- "
            "desktop, mobile web, and Search Lab all feed this automatically.",
        )

        # A dedicated variant of _scrollable() that also paints a soft,
        # glowing neural-graph backdrop around the dashboard's content
        # (see _build_dashboard_scrollable's docstring for why it's a
        # framing margin rather than truly behind every card).
        scroll_frame = self._build_dashboard_scrollable(f)

        self._dash_stats_row = Frame(scroll_frame, bg=BG)
        self._dash_stats_row.pack(fill="x", padx=24, pady=(4, 14))

        hero_wrap = Frame(scroll_frame, bg=PANEL, highlightthickness=1, highlightbackground=NEON_CYAN)
        hero_wrap.pack(fill="x", padx=24, pady=(0, 14))
        Label(hero_wrap, text="● PORTFOLIO EQUITY — BEST STRATEGY", bg=PANEL, fg=NEON_CYAN, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_hero_canvas = Canvas(hero_wrap, bg=PANEL, height=200, highlightthickness=0)
        self._dash_hero_canvas.pack(fill="x", padx=14, pady=(0, 14))

        library_wrap = Frame(scroll_frame, bg=PANEL, highlightthickness=1, highlightbackground=NEON_VIOLET)
        library_wrap.pack(fill="x", padx=24, pady=(0, 14))
        Label(library_wrap, text="● MARKET DATA LIBRARY — data/raw, BY INSTRUMENT", bg=PANEL, fg=NEON_VIOLET, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_library_frame = Frame(library_wrap, bg=PANEL)
        self._dash_library_frame.pack(fill="x", padx=14, pady=(0, 14))

        universe_wrap = Frame(scroll_frame, bg=PANEL, highlightthickness=1, highlightbackground=NEON_CYAN)
        universe_wrap.pack(fill="x", padx=24, pady=(0, 14))
        Label(universe_wrap, text="● STRATEGY UNIVERSE", bg=PANEL, fg=NEON_CYAN, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_universe_canvas = Canvas(universe_wrap, bg=PANEL, height=220, highlightthickness=0)
        self._dash_universe_canvas.pack(fill="x", padx=14, pady=(0, 14))

        charts_row = Frame(scroll_frame, bg=BG)
        charts_row.pack(fill="x", padx=24, pady=(0, 14))

        equity_wrap = Frame(charts_row, bg=PANEL, highlightthickness=1, highlightbackground=NEON_CYAN)
        equity_wrap.pack(side="left", fill="both", expand=True, padx=(0, 7))
        Label(equity_wrap, text="● EQUITY CURVES — TOP STRATEGIES", bg=PANEL, fg=NEON_CYAN, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_equity_canvas = Canvas(equity_wrap, bg=PANEL, height=180, highlightthickness=0)
        self._dash_equity_canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        heatmap_wrap = Frame(charts_row, bg=PANEL, highlightthickness=1, highlightbackground=NEON_MAGENTA)
        heatmap_wrap.pack(side="left", fill="both", expand=True, padx=(7, 0))
        Label(heatmap_wrap, text="● WEEKDAY x HOUR PNL (ALL RUNS)", bg=PANEL, fg=NEON_MAGENTA, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_heatmap_canvas = Canvas(heatmap_wrap, bg=PANEL, height=180, highlightthickness=0)
        self._dash_heatmap_canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        table_wrap = Frame(scroll_frame, bg=PANEL, highlightthickness=1, highlightbackground=NEON_VIOLET)
        table_wrap.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        Label(table_wrap, text="● STRATEGY SCORECARD", bg=PANEL, fg=NEON_VIOLET, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=14, pady=(10, 6)
        )

        style = ttk.Style(self.root)
        style.configure(
            "T58.Treeview", background=PANEL_2, fieldbackground=PANEL_2, foreground=TEXT,
            rowheight=24, borderwidth=0, font=_safe_font(9),
        )
        style.configure("T58.Treeview.Heading", background=PANEL_3, foreground=TEXT_MUTED, font=_safe_font(8, "bold"))
        style.map("T58.Treeview", background=[("selected", PANEL_3)])

        columns = ("strategy", "instrument", "trades", "net", "win", "sharpe", "dd", "runs", "result")
        self._dash_tree = ttk.Treeview(table_wrap, columns=columns, show="headings", style="T58.Treeview", height=8)
        headings = {
            "strategy": "Strategy", "instrument": "Instrument", "trades": "Trades", "net": "Net P/L",
            "win": "Win %", "sharpe": "Sharpe", "dd": "Max DD", "runs": "Runs", "result": "Result",
        }
        for col, text in headings.items():
            self._dash_tree.heading(col, text=text)
            self._dash_tree.column(col, width=100, anchor="w")
        self._dash_tree.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self._refresh_dashboard()

    def _refresh_dashboard(self):
        """Reloads run_history and repaints every dashboard widget. Called
        when the tab is opened, and right after any run/search completes."""
        try:
            data = run_history.dashboard_data()
        except Exception:
            data = {
                "total_strategies": 0, "total_runs": 0, "pass_rate": 0.0, "best": None,
                "strategies": [], "graph": {"nodes": [], "edges": [], "instruments": []},
                "heatmap": [[0.0] * 24 for _ in range(7)], "equity_series": [],
            }

        for child in self._dash_stats_row.winfo_children():
            child.destroy()
        best = data.get("best")
        pass_rate = data["pass_rate"]

        card1 = self._stat_card(self._dash_stats_row, "Strategies tested", str(data["total_strategies"]), accent=NEON_VIOLET)
        card1.pack(side="left", fill="both", expand=True, padx=6)

        card2 = self._ring_stat_card(
            self._dash_stats_row, "Eval pass rate", pass_rate,
            accent=NEON_LIME if pass_rate >= 50 else NEON_MAGENTA,
        )
        card2.pack(side="left", fill="both", expand=True, padx=6)

        card3 = self._stat_card(
            self._dash_stats_row, "Best Sharpe",
            f"{best['sharpe_ratio']:.2f}" if best else "--", accent=NEON_CYAN,
        )
        card3.pack(side="left", fill="both", expand=True, padx=6)

        card4 = self._stat_card(
            self._dash_stats_row, "Leader",
            best["strategy_name"] if best else "--", accent=NEON_AMBER,
        )
        card4.pack(side="left", fill="both", expand=True, padx=6)

        self._paint_data_library()

        self.root.after(30, lambda: self._paint_universe(data["graph"]))
        self.root.after(30, lambda: self._paint_hero_equity(data["equity_series"]))
        self.root.after(30, lambda: self._paint_equity(data["equity_series"]))
        self.root.after(30, lambda: self._paint_heatmap(data["heatmap"]))

        for row_id in self._dash_tree.get_children():
            self._dash_tree.delete(row_id)
        for s in data["strategies"]:
            self._dash_tree.insert("", "end", values=(
                s["strategy_name"], s["instrument"], s["trades"],
                f"${s['net_profit']:,.0f}", f"{s['win_rate']:.1f}%",
                f"{s['sharpe_ratio']:.2f}", f"{s['max_drawdown_pct']:.1f}%",
                s["run_count"], "PASS" if s["single_run_passed"] else "FAIL",
            ))

    def _paint_data_library(self):
        for child in self._dash_library_frame.winfo_children():
            child.destroy()
        try:
            groups = list_datasets_by_instrument()
        except Exception:
            groups = []

        if not groups:
            Label(
                self._dash_library_frame,
                text="No CSVs found under data/raw/ yet — upload one from the Data tab, or drop "
                     "files into instrument subfolders there.",
                bg=PANEL, fg=TEXT_DIM, font=_safe_font(9), wraplength=760, justify="left",
            ).pack(anchor="w", pady=4)
            return

        n_cols = 3
        grid = Frame(self._dash_library_frame, bg=PANEL)
        grid.pack(fill="x")
        for col in range(n_cols):
            grid.grid_columnconfigure(col, weight=1, uniform="lib")

        for i, g in enumerate(groups):
            cell = Frame(grid, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=i // n_cols, column=i % n_cols, sticky="ew", padx=4, pady=4)

            top = Frame(cell, bg=PANEL_2)
            top.pack(fill="x", padx=10, pady=(8, 2))
            Label(top, text=g["instrument"], bg=PANEL_2, fg=TEXT, font=_safe_font(9, "bold")).pack(side="left")
            Label(
                top, text=f"{g['file_count']} file{'s' if g['file_count'] != 1 else ''}",
                bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(7),
            ).pack(side="right")

            detail_text = f"{g['total_rows']:,} rows total"
            Label(cell, text=detail_text, bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8)).pack(
                anchor="w", padx=10, pady=(0, 2)
            )
            if g["empty_count"]:
                Label(
                    cell, text=f"⚠ {g['empty_count']} empty file{'s' if g['empty_count'] != 1 else ''} (0 rows)",
                    bg=PANEL_2, fg=AMBER, font=_safe_font(7, "bold"),
                ).pack(anchor="w", padx=10, pady=(0, 8))
            else:
                Frame(cell, bg=PANEL_2, height=6).pack()

    def _paint_universe(self, graph):
        c = self._dash_universe_canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        h = max(c.winfo_height(), 200)
        nodes = graph.get("nodes", [])
        if not nodes:
            c.create_text(w / 2, h / 2, text="No strategies run yet.", fill=TEXT_DIM, font=_safe_font(9))
            return

        instruments = graph.get("instruments", [])
        n_clusters = max(len(instruments), 1)
        palette = [GREEN, ACCENT, RED, AMBER, BLUE]
        import math
        centers = []
        for i in range(n_clusters):
            angle = (i / n_clusters) * 2 * math.pi
            r = min(w, h) * 0.28 if n_clusters > 1 else 0
            centers.append((w / 2 + r * math.cos(angle), h / 2 + r * math.sin(angle)))

        pos = []
        for i, n in enumerate(nodes):
            cx, cy = centers[n["cluster"]] if n["cluster"] < len(centers) else (w / 2, h / 2)
            angle = (i / max(len(nodes), 1)) * 2 * math.pi
            pos.append([cx + 24 * math.cos(angle), cy + 24 * math.sin(angle)])

        edges = graph.get("edges", [])
        for _ in range(50):
            forces = [[0.0, 0.0] for _ in nodes]
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i == j:
                        continue
                    dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                    d2 = max(dx * dx + dy * dy, 20)
                    forces[i][0] += dx / d2 * 300
                    forces[i][1] += dy / d2 * 300
                cx, cy = centers[nodes[i]["cluster"]] if nodes[i]["cluster"] < len(centers) else (w / 2, h / 2)
                forces[i][0] += (cx - pos[i][0]) * 0.02
                forces[i][1] += (cy - pos[i][1]) * 0.02
            for e in edges:
                a, b = e["source"], e["target"]
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                pull = e["weight"] * 0.02
                forces[a][0] += dx * pull
                forces[a][1] += dy * pull
                forces[b][0] -= dx * pull
                forces[b][1] -= dy * pull
            for i in range(len(nodes)):
                pos[i][0] = min(w - 16, max(16, pos[i][0] + forces[i][0] * 0.3))
                pos[i][1] = min(h - 16, max(16, pos[i][1] + forces[i][1] * 0.3))

        for e in edges:
            a, b = pos[e["source"]], pos[e["target"]]
            width = 0.5 + e["weight"] * 2
            c.create_line(a[0], a[1], b[0], b[1], fill=self._blend(ACCENT, PANEL, 0.35), width=width)

        for i, n in enumerate(nodes):
            color = palette[n["cluster"] % len(palette)]
            r = 5 + min(max(n["sharpe"], 0), 3) * 2
            ring = GREEN if n["passed"] else RED
            self._glow_dot(c, pos[i][0], pos[i][1], r, color, ring_color=ring)

        legend_x = 10
        for i, name in enumerate(instruments):
            color = palette[i % len(palette)]
            c.create_oval(legend_x, h - 16, legend_x + 8, h - 8, fill=color, outline="")
            c.create_text(legend_x + 14, h - 12, text=name, fill=TEXT_MUTED, font=_safe_font(7), anchor="w")
            legend_x += 14 + len(name) * 6 + 14

    def _catmull_rom_points(self, xs: list[float], ys: list[float], samples_per_seg: int = 10) -> list[float]:
        """Densifies a polyline into a smooth Catmull-Rom spline through
        the same original points -- genuinely curved between data points
        rather than relying on Tkinter's own `smooth=True` (a quadratic
        bezier approximation that still visibly kinks with few, unevenly-
        spaced points, which equity curves from real trade data usually
        are). Falls back to the raw points unchanged for fewer than 3
        points, where a spline isn't meaningful anyway."""
        n = len(xs)
        if n < 3:
            out = []
            for x, y in zip(xs, ys):
                out.extend([x, y])
            return out

        def pt(i):
            i = max(0, min(n - 1, i))
            return xs[i], ys[i]

        out = []
        for i in range(n - 1):
            p0, p1, p2, p3 = pt(i - 1), pt(i), pt(i + 1), pt(i + 2)
            for s in range(samples_per_seg):
                t = s / samples_per_seg
                t2, t3 = t * t, t * t * t
                x = 0.5 * (
                    (2 * p1[0]) + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1]) + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                out.extend([x, y])
        out.extend([xs[-1], ys[-1]])
        return out

    def _paint_hero_equity(self, series):
        """The dashboard's featured chart: the single best strategy's
        equity curve, drawn full-width and taller than the small
        multi-strategy comparison chart below, with a genuinely smooth
        Catmull-Rom curve, a soft glow, and a gradient fill under the
        line -- the "make it glow and look smooth" upgrade on top of the
        existing (still useful, more compact) multi-strategy comparison."""
        c = self._dash_hero_canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        h = max(c.winfo_height(), 160)
        pad_x, pad_top, pad_bottom = 20, 16, 28
        if not series:
            c.create_text(w / 2, h / 2, text="No completed runs yet -- run a backtest to populate this chart.",
                           fill=TEXT_DIM, font=_safe_font(9))
            return

        # Feature whichever series is passing (or, failing that, the
        # first available) -- this chart tells one clear story rather
        # than overlaying everything the smaller comparison chart below
        # already shows.
        best_series = next((s for s in series if s.get("passed")), series[0])
        values = best_series["values"]
        if len(values) < 2:
            c.create_text(w / 2, h / 2, text="Not enough closed trades yet for an equity curve.",
                           fill=TEXT_DIM, font=_safe_font(9))
            return

        lo, hi = min(values), max(values)
        rng = (hi - lo) or 1
        n = len(values)
        xs = [pad_x + (i / (n - 1)) * (w - 2 * pad_x) for i in range(n)]
        ys = [h - pad_bottom - ((v - lo) / rng) * (h - pad_top - pad_bottom) for v in values]

        # Gridlines first, underneath everything.
        for frac in (0.0, 0.5, 1.0):
            gy = pad_top + frac * (h - pad_top - pad_bottom)
            c.create_line(pad_x, gy, w - pad_x, gy, fill=BORDER, dash=(2, 3))
        c.create_text(pad_x, pad_top - 8, text=f"${hi:,.0f}", fill=TEXT_DIM, font=_safe_font(7), anchor="sw")
        c.create_text(pad_x, h - pad_bottom + 16, text=f"${lo:,.0f}", fill=TEXT_DIM, font=_safe_font(7), anchor="nw")

        smooth_points = self._catmull_rom_points(xs, ys, samples_per_seg=12)
        color = GREEN if best_series.get("passed") else RED

        # Gradient area fill under the curve -- several stacked bands
        # fading from a visible tint near the line down to nearly nothing
        # at the baseline, faking a vertical gradient the same way every
        # other glow effect in this app fakes soft alpha falloff (Tkinter
        # has no real alpha compositing).
        baseline_y = h - pad_bottom
        n_bands = 10
        for band in range(n_bands, 0, -1):
            frac = band / n_bands
            band_alpha = 0.22 * frac
            poly = []
            for i in range(0, len(smooth_points), 2):
                poly.extend([smooth_points[i], smooth_points[i + 1]])
            top_of_band_scale = frac
            scaled = []
            for i in range(0, len(poly), 2):
                x, y = poly[i], poly[i + 1]
                y_scaled = baseline_y - (baseline_y - y) * top_of_band_scale
                scaled.extend([x, y_scaled])
            polygon_points = scaled + [smooth_points[-2], baseline_y, smooth_points[0], baseline_y]
            c.create_polygon(polygon_points, fill=_blend_hex(PANEL, color, band_alpha), outline="")

        self._glow_line(c, smooth_points, color, width=2.2)

        # A glowing dot on the final (most recent) point -- draws the eye
        # to "where this strategy stands right now."
        self._glow_dot(c, xs[-1], ys[-1], 4.5, color, ring_color=METAL_BRIGHT)

        label = f"{best_series['name']} — {'PASSING' if best_series.get('passed') else 'FAILING'}"
        c.create_oval(w - pad_x - 90, pad_top - 4, w - pad_x - 84, pad_top + 2, fill=color, outline="")
        c.create_text(w - pad_x - 78, pad_top - 1, text=label, fill=TEXT_MUTED, font=_safe_font(7, "bold"), anchor="w")

    def _paint_equity(self, series):
        c = self._dash_equity_canvas
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 140)
        pad = 16
        if not series:
            c.create_text(w / 2, h / 2, text="No completed runs yet.", fill=TEXT_DIM, font=_safe_font(9))
            return

        lo = min(v for s in series for v in s["values"]) if series else 0
        hi = max(v for s in series for v in s["values"]) if series else 1
        rng = (hi - lo) or 1
        max_len = max(len(s["values"]) for s in series)

        # Each strategy gets its own distinct hue from this app's
        # decorative neon palette, so multiple lines on the same chart
        # stay visually distinguishable. Coloring every line strictly by
        # pass/fail (GREEN/RED) meant every failing strategy rendered as
        # the exact same solid red -- indistinguishable from one another
        # whenever most (or, for a fresh batch, all) of the ranked
        # strategies happened to be failing, which is the common case.
        # Pass/fail is still shown -- just via the legend's status text/
        # color instead of collapsing every line to one of two colors.
        palette = [NEON_CYAN, NEON_AMBER, NEON_VIOLET, NEON_LIME, NEON_MAGENTA, BLUE]

        c.create_line(pad, h - pad, w - pad, h - pad, fill=BORDER)
        for idx, s in enumerate(series):
            color = palette[idx % len(palette)]
            n = len(s["values"])
            if n < 2:
                continue
            points = []
            for i, v in enumerate(s["values"]):
                x = pad + (i / max(max_len - 1, 1)) * (w - 2 * pad)
                y = h - pad - ((v - lo) / rng) * (h - 2 * pad)
                points.extend([x, y])
            self._glow_line(c, points, color, width=1.6)

        legend_y = 6
        for idx, s in enumerate(series):
            color = palette[idx % len(palette)]
            status_color = GREEN if s["passed"] else RED
            status_text = "PASS" if s["passed"] else "FAIL"
            c.create_oval(6, legend_y, 12, legend_y + 6, fill=color, outline="")
            c.create_text(18, legend_y + 3, text=s["name"], fill=TEXT_MUTED, font=_safe_font(7), anchor="w")
            c.create_text(w - 8, legend_y + 3, text=status_text, fill=status_color, font=_safe_font(7, "bold"), anchor="e")
            legend_y += 13

    def _paint_heatmap(self, grid):
        c = self._dash_heatmap_canvas
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 140)
        left_pad = 26
        cell_w = (w - left_pad - 4) / 24
        cell_h = (h - 4) / 7
        max_abs = max((abs(v) for row in grid for v in row), default=0.0001) or 0.0001
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for d, row in enumerate(grid):
            c.create_text(4, 2 + d * cell_h + cell_h / 2, text=days[d], fill=TEXT_DIM, font=_safe_font(6), anchor="w")
            for hr, v in enumerate(row):
                intensity = min(abs(v) / max_abs, 1)
                base = GREEN if v >= 0 else RED
                color = self._blend(base, PANEL, 1 - (0.15 + intensity * 0.8))
                x0 = left_pad + hr * cell_w
                y0 = 2 + d * cell_h
                c.create_rectangle(x0, y0, x0 + cell_w - 1, y0 + cell_h - 1, fill=color, outline="")

    # -----------------------------------------------------------------------
    # User Manual
    # -----------------------------------------------------------------------

    def _build_manual_tab(self):
        f = self.tab_manual
        self._page_header(
            f, "GUIDE", "User Manual",
            "A dummy-proof, step-by-step walkthrough of the whole app -- start to finish, "
            "from importing your first candle of data to reading a Full Pipeline verdict.",
        )

        wrap = Frame(f, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        wrap.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        text = Text(
            wrap, wrap="word", bg=PANEL, fg=TEXT, relief="flat", bd=0,
            highlightthickness=0, font=_safe_font(9), padx=22, pady=18, cursor="arrow",
            spacing1=1, spacing3=1,
        )
        scrollbar = ttk.Scrollbar(wrap, orient="vertical", command=text.yview, style="T58.Vertical.TScrollbar")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            text.bind(seq, lambda e: self._generic_text_wheel(text, e))

        text.tag_configure("h1", font=_safe_font(17, "bold"), foreground=ACCENT_HOVER, spacing1=4, spacing3=10)
        text.tag_configure("h2", font=_safe_font(12, "bold"), foreground=NEON_CYAN, spacing1=20, spacing3=8)
        text.tag_configure("h3", font=_safe_font(10, "bold"), foreground=METAL_BRIGHT, spacing1=12, spacing3=4)
        text.tag_configure("body", font=_safe_font(9), foreground=TEXT_MUTED, spacing3=5, lmargin1=2, lmargin2=2)
        text.tag_configure("bullet", font=_safe_font(9), foreground=TEXT_MUTED, lmargin1=22, lmargin2=38, spacing3=4)
        text.tag_configure("numstep", font=_safe_font(9, "bold"), foreground=TEXT, lmargin1=4, lmargin2=26, spacing1=8, spacing3=2)
        text.tag_configure("substep", font=_safe_font(9), foreground=TEXT_MUTED, lmargin1=30, lmargin2=46, spacing3=3)
        text.tag_configure("mono", font=(MONO, 9), foreground=NEON_LIME)
        text.tag_configure("warn", font=_safe_font(9, "bold"), foreground=AMBER, lmargin1=4, lmargin2=20, spacing1=8, spacing3=8)
        text.tag_configure("tip", font=_safe_font(9), foreground=GREEN, lmargin1=4, lmargin2=20, spacing1=6, spacing3=8)
        text.tag_configure("divider", font=_safe_font(4), foreground=BORDER, spacing3=14)

        def h1(t):
            text.insert(END, t + "\n", "h1")

        def h2(t):
            text.insert(END, t + "\n", "h2")

        def h3(t):
            text.insert(END, t + "\n", "h3")

        def body(t):
            text.insert(END, t + "\n", "body")

        def bullet(t):
            text.insert(END, "•  " + t + "\n", "bullet")

        def numstep(n, t):
            text.insert(END, f"{n}.  " + t + "\n", "numstep")

        def substep(t):
            text.insert(END, "–  " + t + "\n", "substep")

        def warn(t):
            text.insert(END, "⚠  " + t + "\n", "warn")

        def tip(t):
            text.insert(END, "✓  " + t + "\n", "tip")

        def rule():
            text.insert(END, "―" * 90 + "\n", "divider")

        # -------------------------------------------------------------
        # Quick Start (short version) -- up top, for anyone who just
        # wants the fastest path and already knows their way around a
        # backtester.
        # -------------------------------------------------------------
        h1("QUICK START (SHORT VERSION)")
        body(
            "For anyone in a hurry: this is the whole app in five clicks. If any of these "
            "words don't mean anything yet, skip down to the FULL WALKTHROUGH below -- it "
            "explains every one of these in plain terms."
        )
        numstep(1, "01 DATA — import a CSV, or fetch data with a free Alpaca key, then select it.")
        numstep(2, "02 STRATEGY — pick Manual / Python / PineScript / MQL5 and set it up.")
        numstep(3, "03 PROP RULES — enter your prop firm's account size, targets, and drawdown rules.")
        numstep(4, "04 RISK — set your risk per trade and starting balance (defaults are fine to start).")
        numstep(5, "15 FULL PIPELINE — click RUN FULL PIPELINE and read the READY / MARGINAL / NOT READY verdict.")
        tip(
            "That's it. Steps 06-14 (Refinement, Search Lab, Validation Lab, Ensemble) are all "
            "optional extra tools for later — you do not need them for a first pass."
        )
        rule()

        # -------------------------------------------------------------
        # Full walkthrough
        # -------------------------------------------------------------
        h1("FULL WALKTHROUGH (STEP BY STEP)")
        body(
            "Follow these in order the first time. Every tab is numbered in the sidebar on "
            "the left in the same order as this guide."
        )

        h2("STEP 1 — Import your market data (01 DATA)")
        body("The backtester needs historical price candles (open/high/low/close/volume) before anything else can run.")
        numstep(1, "Click 01 DATA in the sidebar.")
        numstep(2, "If you already have a CSV file of price data, click IMPORT CSV(S) and select it.")
        substep("A CSV needs columns for timestamp, open, high, low, and close (volume is optional). Headers are "
                "auto-detected — 'time'/'date', 'o'/'open', etc. all work.")
        substep(
            "You can select more than one timeframe at once (Ctrl/Cmd-click or Shift-click) for multi-timeframe "
            "strategies — e.g. a 60-minute file for bias plus a 5-minute file for entries. Ignore this at first."
        )
        numstep(3, "Don't have a CSV? Use the built-in Alpaca fetch instead — see the box below.")
        numstep(4, "Click a dataset in the list to select it. That's the data every later tab will use.")
        tip("Tip: you can also drop CSV files straight into the app's data/raw/ folder and click REFRESH LIST.")

        h3("Getting a free Alpaca API key (for fetching data without a CSV)")
        body(
            "Alpaca is a brokerage that gives out free market data through an API — you do not need to fund "
            "an account or place any real trades to use this."
        )
        numstep(1, "Go to alpaca.markets and sign up for a free account (choose Paper Trading, not a live account).")
        numstep(2, "In the Alpaca dashboard, generate an API Key ID and a Secret Key (usually under 'API Keys' "
                    "or 'Paper Trading' settings).")
        numstep(3, "Back in this app, on the 01 DATA tab, paste both into the 'Fetch data from Alpaca' section.")
        numstep(4, "Click TEST CONNECTION to confirm the keys work.")
        numstep(5, "Choose an asset class, symbol, and timeframe, then click FETCH & SAVE. It's saved locally as "
                    "a CSV, so you only need to fetch it once.")
        tip("Your keys are saved locally on this computer so you don't have to re-enter them every time.")
        rule()

        h2("STEP 2 — Build or upload a strategy (02 STRATEGY)")
        body("This is the trading logic that will be tested against your data. Four options:")
        bullet("MANUAL — a visual, no-code builder: pick indicators and entry/exit conditions from dropdowns.")
        bullet("PYTHON — upload your own .py strategy file, or load one from the built-in Strategy Library.")
        bullet("PINESCRIPT — upload a TradingView-style .pine strategy (a supported subset of syntax).")
        bullet("MQL5 — upload a MetaTrader .mq5 Expert Advisor (a supported subset of syntax).")
        numstep(1, "Click 02 STRATEGY and pick one of the four buttons at the top.")
        numstep(2, "MANUAL: fill in the visual builder — indicators, then entry/exit rules, then a stop-loss/"
                    "take-profit. PYTHON/PINESCRIPT/MQL5: upload a file, or load a saved one from the Strategy Library.")
        tip(
            "New to this? Start with MANUAL and a simple moving-average crossover — it's the fastest way to "
            "see the whole app work end to end before bringing in your own code."
        )
        rule()

        h2("STEP 3 — Enter your prop firm's rules (03 PROP RULES)")
        body(
            "These numbers come straight from your prop firm's rulebook — check their FAQ/PDF, since getting "
            "this wrong makes every result downstream meaningless."
        )
        bullet("Account size, evaluation profit target %, daily loss limit %, maximum drawdown %.")
        bullet("Drawdown type (trailing vs static) and drawdown check mode (intrabar vs end-of-day) — match "
               "these exactly to what your firm documents.")
        bullet("Optional: payout threshold/cap/frequency, minimum trading days, position size limits.")
        rule()

        h2("STEP 4 — Set risk & execution (04 RISK)")
        body("Position sizing, trading costs, and execution assumptions used by the backtest engine.")
        bullet("Initial balance, risk mode (percent of equity vs fixed), and risk per trade.")
        bullet("Spread/slippage/commission — leave at realistic defaults unless your broker publishes different numbers.")
        tip("Not sure what to put here? The defaults are sane starting points — you can always come back and tune them.")
        rule()

        h2("STEP 5 — Run a single backtest (05 RUN & REPORT)")
        body(
            "This runs your strategy exactly as configured, once, and produces one HTML report: "
            "Backtest → Prop Simulation → Monte Carlo → Report."
        )
        numstep(1, "Click 05 RUN & REPORT, set the number of Monte Carlo simulations (10,000 is a good default).")
        numstep(2, "Click RUN FULL PIPELINE on this tab, then OPEN HTML REPORT once it finishes.")
        warn(
            "This tab's button is also labeled RUN FULL PIPELINE, but it is NOT the same as the 15 FULL "
            "PIPELINE tab below — this one just runs your strategy once, as-is. The 15 FULL PIPELINE tab "
            "(Step 9 in this guide) automatically searches for a better configuration and validates it out-"
            "of-sample before giving you a verdict. For a first pass, most people skip straight to Step 9."
        )
        rule()

        h2("STEPS 6-8, 10-14 — optional deeper validation tools")
        body(
            "Everything between here and the Full Pipeline tab is optional and can be safely skipped on a "
            "first pass through the app:"
        )
        bullet("06 REFINEMENT / 13 WALK-FORWARD GA — automatically search for better parameter values.")
        bullet("07 SEARCH LAB — generate and test many strategy variants at once.")
        bullet("08 WALK-FORWARD OPT / 09 CPCV-PBO / 10 SENSITIVITY / 12 MULTI-OBJECTIVE — deeper robustness "
               "checks quants use to catch overfitting.")
        bullet("11 PORTFOLIO — backtest several strategies/instruments together as one portfolio.")
        bullet("14 ENSEMBLE — combine several strategies' signals into one.")
        tip(
            "These are all genuinely useful once you have a strategy worth digging into further — just not "
            "required to get your first result. The 15 FULL PIPELINE tab below already runs a solid, "
            "automated version of parameter search and out-of-sample validation on its own."
        )
        rule()

        h2("STEP 9 — Run the Full Pipeline (15 FULL PIPELINE) — the recommended one-button path")
        body(
            "This is the fastest way to get a trustworthy answer: it backtests your strategy as given, "
            "automatically searches for a configuration that generalizes (scored only on data it wasn't "
            "tuned on), re-validates the winner with a fresh Monte Carlo run, checks it holds up across "
            "several different historical stretches, and gives you one plain verdict."
        )
        numstep(1, "Make sure Steps 1-4 above are filled in (data selected, strategy built, prop rules and risk set).")
        numstep(2, "Click 15 FULL PIPELINE, leave the default settings for a first run, and click RUN FULL PIPELINE.")
        numstep(3, "Watch the live progress log — it can take anywhere from under a minute to several minutes "
                    "depending on your data size and settings.")
        numstep(4, "Read the VERDICT box at the top once it finishes:")
        substep("READY — passed every check: Monte Carlo pass probability, drawdown, out-of-sample stability, "
                "and the signal-quality (ICIR) gate.")
        substep("MARGINAL — passed some but not all checks — worth a closer look before trusting it.")
        substep("NOT READY — failed enough checks that this configuration isn't trustworthy as-is; check the "
                "listed reasons for exactly why.")
        substep("The Verdict box lists the specific reason for every pass/fail — scroll down within the box "
                "(or the page) if the full list runs long.")
        tip(
            "For Python/PineScript/MQL5 strategies, the winning version is automatically saved into the "
            "Strategy Library, tagged 'validated', ready to use again later or take straight into 16 LIVE "
            "DEMO TEST."
        )
        rule()

        h2("STEP 10 — Live Demo Test on a real broker feed (LIVE TRADING → Live Demo Test)")
        body(
            "Once a strategy looks good in the Full Pipeline, this deploys it to a free MetaTrader 5 (MT5) "
            "demo account so you can watch it trade forward against real, live broker prices — still no real "
            "money, no live/funded order path exists on this tab."
        )
        h3("One-time MT5 setup")
        numstep(1, "Download and install the MT5 terminal (64-bit) — any MT5-supporting broker's website offers "
                    "a free download, or your prop firm's own site if they provide one.")
        numstep(2, "Open the terminal and create a free demo account from within it (File → Open an Account → "
                    "choose a demo account) — this gives you a login number, server name, and password.")
        numstep(3, "In this app's Live Demo Test tab, enter that login, server, and password under "
                    "'MT5 Demo Account'.")
        numstep(4, "Click SAVE & TEST CONNECTION.")
        substep(
            "If it fails saying the terminal wasn't found, click AUTO-DETECT first (it searches the common "
            "Windows install locations), or BROWSE... to point directly at terminal64.exe if MT5 is installed "
            "somewhere unusual."
        )
        substep("This tab only works on Windows with the MT5 terminal actually installed and running.")
        h3("Running a session")
        numstep(1, "Pick a saved strategy from the Strategy Library dropdown.")
        numstep(2, "Set the symbol (must match the exact name in your MT5 broker's Market Watch, e.g. 'XAUUSD') "
                    "and your risk settings.")
        numstep(3, "Click START LIVE DEMO TEST. Watch trades appear in the trade journal and live log as they happen.")
        warn(
            "The red KILL SWITCH — FLATTEN & STOP button immediately closes every open position and stops the "
            "session. Use it any time something looks wrong."
        )
        rule()

        h2("STEP 11 — Deploy Live (LIVE TRADING → Deploy Live) — real money, read this first")
        warn(
            "This is NOT the same as Live Demo Test. An account connected here trades with real capital in a "
            "live or funded prop-firm account. Confirm your prop firm actually permits automated/EA trading "
            "on that account before connecting it — many firms restrict or ban it outright, and violating "
            "that can get a funded account terminated regardless of performance."
        )
        numstep(1, "Pick your prop firm from the dropdown (or 'Other' if it isn't listed) — this fills in which "
                    "platform(s) it uses.")
        substep(
            "Only MT4/MT5 accounts are connectable today. Futures-only firms (Apex, Topstep, MyFundedFutures) "
            "use Tradovate/Rithmic/NinjaTrader instead — a different integration this app doesn't have yet."
        )
        numstep(2, "Enter your account's nickname, login, server, and password, check the confirmation box, "
                    "and click SAVE ACCOUNT.")
        numstep(3, "Click TEST CONNECTION to confirm it reaches your real account and reports its balance.")
        numstep(4, "Pick a validated strategy and click START LIVE TRADING.")
        substep(
            "The same red KILL SWITCH from Live Demo Test works identically here — use it any time something "
            "looks wrong."
        )
        rule()

        h2("STEP 12 — Live Market (LIVE TRADING → Live Market)")
        body(
            "A live-updating candlestick chart for any symbol — live via MT5 if connected, delayed via Alpaca "
            "if you've saved keys, or a steady replay of a local CSV if neither is available. Pick a source, "
            "symbol, and timeframe, then click OPEN LIVE CHART."
        )
        rule()

        # -------------------------------------------------------------
        # Short version recap
        # -------------------------------------------------------------
        h1("SHORT VERSION — RECAP")
        body("Once you've done the full walkthrough once, this is all you need to remember for next time:")
        numstep(1, "01 DATA → select or fetch your data.")
        numstep(2, "02 STRATEGY → pick/build your strategy.")
        numstep(3, "03 PROP RULES + 04 RISK → confirm these still match your firm/settings.")
        numstep(4, "15 FULL PIPELINE → RUN FULL PIPELINE → read the verdict.")
        numstep(5, "If READY and you want to see it trade live → LIVE TRADING → Live Demo Test (paper) or "
                    "Deploy Live (real capital, once you're confident).")
        tip("That's the whole loop. Everything else in the sidebar is there for when you want to dig deeper.")

        text.config(state="disabled")

    def _generic_text_wheel(self, text_widget, event):
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        text_widget.yview_scroll(delta * 3, "units")
        return "break"

    # -----------------------------------------------------------------------
    # Tab 1 — Market Data
    # -----------------------------------------------------------------------

    def _build_data_tab(self):
        f = self._scrollable(self.tab_data)

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

        self._build_alpaca_section(f)

        self._refresh_dataset_list()

    # -----------------------------------------------------------------------
    # Tab 1 — Market Data — Alpaca API fetch
    # -----------------------------------------------------------------------

    def _build_alpaca_section(self, parent):
        """An alternative to picking local files above: fetch bars directly
        from Alpaca using saved (or freshly entered) API keys. Fetched data
        is written into data/raw/<SYMBOL>/ as a normal CSV, so it shows up
        in the 'Available datasets' list above and can be picked -- alone
        or Ctrl/Cmd-clicked together with other files -- exactly like any
        manually imported CSV. This mirrors how the falsification-kit
        scripts (fetch_5m.py / fetch_cache.py) pull bars via alpaca-py,
        except keys are entered once here and saved for reuse instead of
        being read from ALPACA_API_KEY/ALPACA_SECRET_KEY environment
        variables every run."""
        section = self._section(
            parent,
            "Fetch data from Alpaca",
            "Alternative to local files above — pulls bars via the Alpaca API "
            "(stocks + crypto only; no forex/futures/CFD feed) and saves them into "
            "data/raw/ so they join the dataset list above.",
        )

        saved = alpaca_credentials.load_credentials()
        prefill_key = saved.api_key if saved else ""
        prefill_secret = saved.secret_key if saved else ""

        self.alp_api_key = LabeledEntry(section, "Alpaca API key", prefill_key, secret=True, width=32)
        self.alp_secret_key = LabeledEntry(section, "Alpaca secret key", prefill_secret, secret=True, width=32)
        self.alp_save_keys = LabeledCheckbox(
            section, "Save these keys on this computer for next time", default=bool(saved)
        )

        self.alp_asset_class = LabeledCombo(section, "Asset class", ASSET_CLASSES, default=ASSET_CLASSES[0])
        self.alp_asset_class.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_alpaca_asset_class_changed())
        self.alp_symbols = LabeledEntry(
            section, "Symbol(s), comma-separated", "AAPL", width=32
        )
        self.alp_timeframe = LabeledCombo(
            section, "Timeframe", TIMEFRAME_LABELS, default="1Day"
        )
        self.alp_start = LabeledEntry(section, "Start date (YYYY-MM-DD)", "2024-01-01")
        self.alp_end = LabeledEntry(section, "End date (YYYY-MM-DD)", "2026-01-01")
        self.alp_feed = LabeledCombo(section, "Feed (stocks only)", FEED_CHOICES, default="iex")
        self.alp_adjustment = LabeledCombo(
            section, "Adjustment (stocks only)", ADJUSTMENT_CHOICES, default="raw"
        )

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(4, 4))

        self.alp_test_btn = self._button(btn_row, "TEST CONNECTION", self._test_alpaca_connection)
        self.alp_test_btn.pack(side="left")

        self.alp_fetch_btn = self._button(btn_row, "FETCH & SAVE", self._fetch_alpaca_clicked, primary=True)
        self.alp_fetch_btn.pack(side="left", padx=8)

        self.alp_forget_btn = self._button(btn_row, "FORGET SAVED KEYS", self._forget_alpaca_keys)
        self.alp_forget_btn.pack(side="left")

        self.alpaca_status = Label(
            section,
            text="●  A free Alpaca paper-trading account provides API keys for data access.",
            bg=PANEL,
            fg=TEXT_MUTED,
            font=_safe_font(9),
            wraplength=760,
            justify="left",
        )
        self.alpaca_status.pack(anchor="w", padx=18, pady=(2, 14))

    def _on_alpaca_asset_class_changed(self):
        # Feed/adjustment only apply to stock bars on Alpaca; crypto has
        # neither concept, so grey them out rather than let them silently
        # do nothing when Crypto is selected.
        is_stock = self.alp_asset_class.get_str() == "Stock"
        state = "readonly" if is_stock else "disabled"
        self.alp_feed.combo.config(state=state)
        self.alp_adjustment.combo.config(state=state)

    def _set_alpaca_status(self, text, color=None):
        self.alpaca_status.config(text=f"●  {text}", fg=color or TEXT_MUTED)
        self.root.update_idletasks()

    def _set_alpaca_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.alp_test_btn.config(state=state)
        self.alp_fetch_btn.config(state=state)
        self.alp_forget_btn.config(state=state)

    def _maybe_save_alpaca_keys(self, api_key: str, secret_key: str):
        if self.alp_save_keys.get():
            alpaca_credentials.save_credentials(api_key, secret_key)

    def _forget_alpaca_keys(self):
        alpaca_credentials.clear_credentials()
        self.alp_api_key.var.set("")
        self.alp_secret_key.var.set("")
        self.alp_save_keys.var.set(False)
        self._set_alpaca_status("Saved keys removed from this computer.", GREEN)

    def _test_alpaca_connection(self):
        api_key = self.alp_api_key.get_str().strip()
        secret_key = self.alp_secret_key.get_str().strip()
        if not api_key or not secret_key:
            messagebox.showwarning("Missing keys", "Enter both an API key and a secret key first.")
            return

        self._set_alpaca_buttons_enabled(False)
        self._set_alpaca_status("Testing connection...", AMBER)

        def worker():
            try:
                message = test_connection(api_key, secret_key)
                self._maybe_save_alpaca_keys(api_key, secret_key)
                self._set_alpaca_status(message, GREEN)
            except (AlpacaImportError, AlpacaFetchError) as exc:
                self._set_alpaca_status(str(exc), RED)
            except Exception as exc:  # pragma: no cover - defensive
                self._set_alpaca_status(f"Unexpected error: {exc}", RED)
            finally:
                self._set_alpaca_buttons_enabled(True)

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_alpaca_clicked(self):
        api_key = self.alp_api_key.get_str().strip()
        secret_key = self.alp_secret_key.get_str().strip()
        symbols = [s.strip() for s in self.alp_symbols.get_str().split(",") if s.strip()]
        asset_class = self.alp_asset_class.get_str()
        timeframe_label = self.alp_timeframe.get_str()
        start = self.alp_start.get_str().strip()
        end = self.alp_end.get_str().strip()
        feed = self.alp_feed.get_str()
        adjustment = self.alp_adjustment.get_str()

        if not api_key or not secret_key:
            messagebox.showwarning("Missing keys", "Enter both an API key and a secret key first.")
            return
        if not symbols:
            messagebox.showwarning("Missing symbol", "Enter at least one symbol (comma-separated for more than one).")
            return

        self._set_alpaca_buttons_enabled(False)
        threading.Thread(
            target=self._fetch_alpaca_pipeline,
            args=(api_key, secret_key, symbols, asset_class, timeframe_label, start, end, feed, adjustment),
            daemon=True,
        ).start()

    def _fetch_alpaca_pipeline(self, api_key, secret_key, symbols, asset_class, timeframe_label, start, end, feed, adjustment):
        try:
            self._maybe_save_alpaca_keys(api_key, secret_key)
            saved_paths = []
            for i, symbol in enumerate(symbols, start=1):
                self._set_alpaca_status(f"Fetching {symbol} ({i}/{len(symbols)})...", AMBER)
                df = fetch_bars(
                    api_key, secret_key, symbol, asset_class, timeframe_label, start, end,
                    feed=feed, adjustment=adjustment,
                )
                dest = save_bars_as_csv(df, symbol, timeframe_label)
                saved_paths.append(dest)
                self._set_alpaca_status(f"Saved {symbol}: {len(df):,} bars -> {dest.name}", GREEN)

            self._refresh_dataset_list()
            if len(saved_paths) == 1:
                self._select_datasets(saved_paths, silent=True)
                self._set_alpaca_status(
                    f"Done. {saved_paths[0].name} is now the active dataset.", GREEN
                )
            else:
                names = ", ".join(p.name for p in saved_paths)
                self._set_alpaca_status(
                    f"Done. Saved {len(saved_paths)} file(s): {names}. "
                    "Ctrl/Cmd-click them in the list above to combine as multi-timeframe, "
                    "or pick just one.",
                    GREEN,
                )
        except (AlpacaImportError, AlpacaFetchError) as exc:
            self._set_alpaca_status(str(exc), RED)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_alpaca_status(f"Unexpected error: {exc}", RED)
        finally:
            self._set_alpaca_buttons_enabled(True)

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END)
        self._stored_datasets = list_stored_datasets()
        by_name = {ds.name: i for i, ds in enumerate(self._stored_datasets)}

        # Grouped by instrument folder (e.g. data/raw/EURUSD/...) rather than
        # a flat mtime-sorted list, so opening the Data tab actually shows
        # "the different instrument folders" instead of one long file list.
        groups = list_datasets_by_instrument()
        self._dataset_row_map: dict[int, int] = {}      # listbox row -> _stored_datasets index
        self._dataset_index_to_row: dict[int, int] = {}  # _stored_datasets index -> listbox row
        row = 0
        first_nonempty_row = None

        for g in groups:
            self.dataset_listbox.insert(END, f"\u2500\u2500 {g['instrument']} \u2500\u2500")
            self.dataset_listbox.itemconfig(row, fg=TEXT_MUTED, selectforeground=TEXT_MUTED, selectbackground=PANEL)
            row += 1
            for file_info in g["files"]:
                idx = by_name.get(file_info["full_name"])
                if idx is None:
                    continue
                label = f"    {file_info['name']}" + ("   (empty)" if file_info["empty"] else "")
                self.dataset_listbox.insert(END, label)
                if file_info["empty"]:
                    self.dataset_listbox.itemconfig(row, fg=TEXT_DIM)
                elif first_nonempty_row is None:
                    first_nonempty_row = row
                self._dataset_row_map[row] = idx
                self._dataset_index_to_row[idx] = row
                row += 1

        if self._stored_datasets and not self.csv_paths:
            # Auto-select the first dataset with real data in it, not just
            # the most-recently-modified file — with data organized into
            # per-instrument folders it's common for some timeframes to be
            # empty placeholder exports (0 rows), and mtimes on a fresh
            # checkout/extraction can tie or favor one of those. Landing on
            # an empty file here used to silently leave "no data" active
            # (or pop a blocking error dialog) the moment the app opened.
            target_row = first_nonempty_row if first_nonempty_row is not None else next(iter(self._dataset_row_map), None)
            if target_row is not None:
                chosen = self._stored_datasets[self._dataset_row_map[target_row]]
                self.dataset_listbox.selection_set(target_row)
                self.dataset_listbox.see(target_row)
                self._select_datasets([chosen.path], silent=True)

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        paths = [self._stored_datasets[self._dataset_row_map[i]].path for i in sel if i in self._dataset_row_map]
        if not paths:
            return
        self._select_datasets(paths)

    def _select_datasets(self, paths: list[Path], silent: bool = False):
        """Load and validate one or more selected CSVs. Multiple files are
        treated as multiple timeframes for a multi-timeframe backtest.

        silent=True (used only for the automatic startup pick above) never
        pops a blocking messagebox — an empty/invalid file just leaves the
        status line red with an explanation, so the app can never freeze
        on launch waiting for a dialog no one is there to dismiss."""
        results = []
        for path in paths:
            result = import_csv(path)
            if not result.is_valid:
                detail = "; ".join(result.errors) or "no rows found"
                if silent:
                    self.data_status.config(
                        text=f"●  {path.name}: {detail} — pick another dataset from the list.",
                        fg=AMBER,
                    )
                else:
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
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")]
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
                    row = self._dataset_index_to_row.get(i)
                    if row is not None:
                        self.dataset_listbox.selection_clear(0, END)
                        self.dataset_listbox.selection_set(row)
                        self.dataset_listbox.see(row)
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
        # Strategy library — save/load strategies inside the app itself
        # ------------------------------------------------------------
        library_section = self._section(
            f,
            "Strategy library",
            "Strategies live here inside the app instead of only on your computer. "
            "Importing a file above automatically saves a copy into the library; "
            "you can also load, rename, or delete a previously saved one below. "
            "Ctrl/Cmd-click or Shift-click to select more than one for bulk delete/export. "
            "Note: a packaged .exe's library lives next to the .exe, not in your "
            "git repo — use EXPORT LIBRARY AS ZIP below and unzip it into the "
            "repo's strategies/ folder to keep them in sync.",
        )

        self.strategy_search = LabeledEntry(library_section, "Search (name / market / tags)", "")
        self.strategy_search.entry.bind("<KeyRelease>", lambda _e: self._refresh_strategy_library())

        filter_row = Frame(library_section, bg=PANEL)
        filter_row.pack(fill="x", padx=0, pady=(0, 2))

        self.strategy_filter_market = LabeledCombo(filter_row, "Browse market", ["All markets"], default="All markets")
        self.strategy_filter_market.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        self.strategy_filter_tag = LabeledCombo(filter_row, "Browse tag", ["All tags"], default="All tags")
        self.strategy_filter_tag.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        self.strategy_filter_status = LabeledCombo(
            filter_row, "Browse status", ["All statuses", *STATUS_LABELS_ORDERED], default="All statuses"
        )
        self.strategy_filter_status.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        lib_list_frame = Frame(library_section, bg=PANEL)
        lib_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.strategy_library_listbox = Listbox(
            lib_list_frame,
            height=6,
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
        self.strategy_library_listbox.pack(side="left", fill="both", expand=True)

        lib_scrollbar = ttk.Scrollbar(
            lib_list_frame,
            orient="vertical",
            command=self.strategy_library_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        lib_scrollbar.pack(side="right", fill="y")
        self.strategy_library_listbox.config(yscrollcommand=lib_scrollbar.set)
        self.strategy_library_listbox.bind(
            "<Double-Button-1>", lambda _e: self._load_selected_library_strategy()
        )
        self.strategy_library_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._on_library_selection_changed()
        )

        lib_btn_row = Frame(library_section, bg=PANEL)
        lib_btn_row.pack(anchor="w", padx=18, pady=(0, 6))

        self._button(
            lib_btn_row, "LOAD SELECTED", self._load_selected_library_strategy, primary=True
        ).pack(side="left")
        self._button(
            lib_btn_row, "RENAME SELECTED", self._rename_selected_library_strategy
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row, "DELETE SELECTED", self._delete_selected_library_strategy
        ).pack(side="left")
        self._button(
            lib_btn_row, "REFRESH LIBRARY", self._refresh_strategy_library
        ).pack(side="left", padx=8)

        lib_btn_row_3 = Frame(library_section, bg=PANEL)
        lib_btn_row_3.pack(anchor="w", padx=18, pady=(0, 6))

        self._button(
            lib_btn_row_3, "ADD SELECTED TO BATCH QUEUE", self._add_selected_to_batch_queue, primary=True
        ).pack(side="left")
        self._button(
            lib_btn_row_3, "VIEW CODE / CONFIG", self._view_selected_strategy_code
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row_3, "OPTIMIZE SELECTED", self._optimize_selected_library_strategies
        ).pack(side="left", padx=8)

        Label(
            library_section,
            text="ADD SELECTED TO BATCH QUEUE stages every currently highlighted strategy "
            "(Ctrl/Cmd or Shift-click for more than one) into the Batch test queue below "
            "-- it does NOT start a test yet. Queue up strategies from Python, PineScript, "
            "and MQL5 here (any mix is fine), then go set up 03 Prop Rules and 04 Risk the "
            "way you want them, and come back here and click RUN BATCH TEST when you're "
            "ready. VIEW CODE / CONFIG shows the saved source for a selected "
            "Python/PineScript/MQL5 strategy, or the built config for whatever's currently "
            "set up in Manual mode. OPTIMIZE SELECTED runs the same walk-forward-aware GA "
            "Full Pipeline uses (Step 2) against just the highlighted strategy(ies) -- "
            "quicker than a full Full Pipeline run, and saves each winning configuration "
            "into the library as a new '<name>_optimized' file (status: draft) without "
            "touching the original.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        # ------------------------------------------------------------
        # Batch test queue -- staged strategies waiting on RUN BATCH TEST
        # ------------------------------------------------------------
        queue_section = self._section(
            f, "Batch test queue",
            "Strategies staged here run one after another through the same backtest -> "
            "prop-sim -> Monte Carlo -> report pipeline as Run & Report, using whatever "
            "Prop Rules (03) and Risk (04) are set at the moment you click RUN BATCH TEST "
            "-- one saved report per strategy, all showing up on the Dashboard afterward. "
            "Highlight a row and click REMOVE SELECTED FROM QUEUE if something got added "
            "by mistake.",
        )
        queue_list_frame = Frame(queue_section, bg=PANEL)
        queue_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.batch_queue_listbox = Listbox(
            queue_list_frame,
            height=6,
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
        self.batch_queue_listbox.pack(side="left", fill="both", expand=True)
        queue_scrollbar = ttk.Scrollbar(
            queue_list_frame, orient="vertical", command=self.batch_queue_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        queue_scrollbar.pack(side="right", fill="y")
        self.batch_queue_listbox.config(yscrollcommand=queue_scrollbar.set)

        queue_btn_row = Frame(queue_section, bg=PANEL)
        queue_btn_row.pack(anchor="w", padx=18, pady=(0, 4))
        self._button(
            queue_btn_row, "LOAD SELECTED FROM QUEUE", self._load_selected_from_batch_queue, primary=True
        ).pack(side="left")
        self._button(
            queue_btn_row, "REMOVE SELECTED FROM QUEUE", self._remove_selected_from_batch_queue
        ).pack(side="left", padx=8)
        self._button(
            queue_btn_row, "CLEAR QUEUE", self._clear_batch_queue
        ).pack(side="left", padx=8)
        self._button(
            queue_btn_row, "RUN BATCH TEST", self._run_batch_queue_clicked, primary=True
        ).pack(side="left")
        self._button(
            queue_btn_row, "RUN FULL PIPELINE (BATCH)", self._run_full_pipeline_queue_clicked, primary=True
        ).pack(side="left", padx=8)

        Label(
            queue_section, text="RUN BATCH TEST runs every queued strategy through the plain "
            "backtest -> prop-sim -> Monte Carlo pipeline (fast). RUN FULL PIPELINE (BATCH) runs "
            "every queued strategy through the FULL 15 Full Pipeline instead (baseline -> "
            "walk-forward-aware GA search -> re-validated Monte Carlo -> out-of-sample check -> "
            "holdout check -> ICIR gate -> verdict) -- slower per strategy since it includes the "
            "GA search, but this is how you batch-test hundreds of selected strategies through "
            "the full validation ladder without opening and running Full Pipeline on each one by "
            "hand. Either way: one saved report per strategy, using whatever settings are "
            "currently configured on 03 Prop Rules / 04 Risk (and, for the Full Pipeline batch, "
            "15 Full Pipeline's own GA/AI Assist settings) at the moment you click.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(4, 0))
        Label(
            queue_section, text="LOAD SELECTED FROM QUEUE puts one queued strategy into the "
            "STRATEGY SOURCE slot above -- for the single-strategy tabs (06 Refinement, 08 "
            "Walk-Forward Opt, 10 Sensitivity, 12 Multi-Objective) that only ever run against ONE "
            "loaded strategy, so you can step through this same queue one at a time instead of "
            "re-browsing/re-selecting for each one.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(4, 0))

        self.batch_queue_status = Label(
            queue_section, text="Queue is empty -- select strategies above and click "
            "ADD SELECTED TO BATCH QUEUE.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.batch_queue_status.pack(anchor="w", padx=18, pady=(0, 10))

        lib_btn_row_2 = Frame(library_section, bg=PANEL)
        lib_btn_row_2.pack(anchor="w", padx=18, pady=(0, 10))

        self._button(
            lib_btn_row_2, "OPEN LIBRARY FOLDER", self._open_strategy_library_folder
        ).pack(side="left")
        self._button(
            lib_btn_row_2, "EXPORT SELECTED AS ZIP", self._export_selected_library_strategies
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row_2, "EXPORT LIBRARY AS ZIP", self._export_strategy_library
        ).pack(side="left")

        Label(
            library_section,
            text="Tip: OPEN LIBRARY FOLDER is only for copying/dropping files in Explorer "
            "(or Finder). To actually use a saved strategy, select it in the list above and "
            "click LOAD SELECTED (or double-click it) -- double-clicking a .py/.pine/.mq5 "
            "file itself in Explorer won't work, since Windows has no default app registered "
            "for those extensions and will say so.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        # ---- Metadata for the selected saved strategy -------------
        meta_frame = Frame(library_section, bg=PANEL)
        meta_frame.pack(fill="x", padx=0, pady=(0, 4))

        self.strategy_meta_description = LabeledEntry(meta_frame, "Description", "")
        self.strategy_meta_market = LabeledEntry(meta_frame, "Market / timeframe", "")
        self.strategy_meta_tags = LabeledEntry(meta_frame, "Tags (comma-separated)", "")
        self.strategy_meta_status = LabeledCombo(
            meta_frame, "Status", STATUS_LABELS_ORDERED, default=STATUS_KEY_TO_LABEL["draft"]
        )

        self._button(
            meta_frame, "SAVE INFO TO SELECTED", self._save_selected_library_metadata
        ).pack(anchor="w", padx=18, pady=(2, 4))

        self.strategy_meta_last_run = Label(
            library_section,
            text="",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
            justify="left",
        )
        self.strategy_meta_last_run.pack(anchor="w", padx=18, pady=(0, 8))

        self.strategy_library_status = Label(
            library_section,
            text="",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
        )
        self.strategy_library_status.pack(anchor="w", padx=18, pady=(0, 14))

        self._strategy_library_items: list = []
        self._refresh_strategy_library()

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
        if hasattr(self, "strategy_library_listbox"):
            self._refresh_strategy_library()

    def _browse_strategy_file(self):
        mode = self.strategy_mode.get()
        ext = {
            "python": "*.py",
            "pinescript": "*.pine",
            "mql5": "*.mq5",
        }.get(mode, "*.*")

        path = filedialog.askopenfilename(
            filetypes=[("Strategy file", ext)]
        )

        if not path:
            return

        if mode in STRATEGY_TYPES:
            stored_path = self._save_to_library_with_overwrite_prompt(
                lambda overwrite: save_strategy_path(path, mode, overwrite=overwrite),
                fallback_path=Path(path),
            )
            if stored_path is None:
                return  # user cancelled the overwrite/rename prompt
            self.strategy_py_path = str(stored_path)
            self._active_library_strategy = (mode, stored_path.name)
            self.strategy_file_status.config(
                text=f"Selected: {stored_path.name}  (saved to library)",
                fg=GREEN,
            )
            self._refresh_strategy_library()
        else:
            self.strategy_py_path = path
            self._active_library_strategy = None
            self.strategy_file_status.config(
                text=f"Selected: {os.path.basename(path)}",
                fg=GREEN,
            )

    def _save_to_library_with_overwrite_prompt(self, save_fn, fallback_path: Path):
        """Try `save_fn(overwrite=False)`. On a name collision, ask the user
        whether to overwrite the existing saved strategy or save this one as
        a new, separately-named copy; on OSError, fall back to using the
        file in place (still usable for this run, just not library-backed).
        Returns the stored Path, or None if the user cancelled."""
        try:
            return save_fn(overwrite=False)
        except StrategyAlreadyExists as exc:
            choice = messagebox.askyesnocancel(
                "Strategy already saved",
                f"'{exc.filename}' is already in the strategy library.\n\n"
                "Yes = overwrite the saved copy with this one\n"
                "No = save this as a new, separately-named copy\n"
                "Cancel = don't save to the library",
            )
            if choice is None:
                return None
            if choice:
                return save_fn(overwrite=True)
            new_name = simpledialog.askstring(
                "Save as new copy",
                "New filename for this copy:",
                initialvalue=exc.filename,
            )
            if not new_name:
                return None
            try:
                return self._save_strategy_copy_as(fallback_path, exc.strategy_type, new_name)
            except StrategyAlreadyExists:
                messagebox.showerror(
                    "Name taken", f"'{new_name}' is also already saved. Try again with a different name."
                )
                return None
        except OSError as exc:
            messagebox.showwarning(
                "Saved locally only",
                f"Selected the file, but couldn't copy it into the strategy "
                f"library ({exc}). It will still work for this run.",
            )
            return fallback_path

    def _save_strategy_copy_as(self, source_path: Path, strategy_type: str, new_name: str) -> Path:
        """Copy source_path's bytes into the library under a caller-chosen
        filename (save_strategy_path always keeps the source's own
        filename, so a rename-on-save needs this instead)."""
        content = Path(source_path).read_bytes()
        return save_strategy_bytes(content, new_name, strategy_type, overwrite=False)

    def _refresh_strategy_library(self):
        mode = self.strategy_mode.get()
        self.strategy_library_listbox.delete(0, END)
        self._strategy_library_items = []

        if mode not in STRATEGY_TYPES:
            self.strategy_library_status.config(
                text="Manual strategies aren't files — nothing to show here.",
                fg=TEXT_DIM,
            )
            return

        # Keep the market/tag filter dropdowns' options current -- new
        # values show up here as soon as they're saved on any strategy.
        markets = list_all_markets(mode)
        self.strategy_filter_market.combo["values"] = ["All markets", *markets]
        if self.strategy_filter_market.get_str() not in ("All markets", *markets):
            self.strategy_filter_market.var.set("All markets")
        tags = list_all_tags(mode)
        self.strategy_filter_tag.combo["values"] = ["All tags", *tags]
        if self.strategy_filter_tag.get_str() not in ("All tags", *tags):
            self.strategy_filter_tag.var.set("All tags")

        query = self.strategy_search.get_str().strip() if hasattr(self, "strategy_search") else ""
        market_filter = self.strategy_filter_market.get_str()
        tag_filter = self.strategy_filter_tag.get_str()
        status_filter = self.strategy_filter_status.get_str()

        self._strategy_library_items = list_saved_strategies(
            mode,
            query=query,
            market=None if market_filter in ("", "All markets") else market_filter,
            tag=None if tag_filter in ("", "All tags") else tag_filter,
            status=None if status_filter in ("", "All statuses") else STATUS_LABEL_TO_KEY.get(status_filter, status_filter),
        )
        for item in self._strategy_library_items:
            kb = item.size_bytes / 1024
            desc = item.metadata.get("description", "")
            suffix = f"  —  {desc}" if desc else ""
            lookahead = item.metadata.get("lookahead")
            if lookahead is None:
                badge = ""
            elif lookahead.get("clean"):
                badge = "  ✓clean"
            else:
                badge = "  ⚠LOOKAHEAD"
            idx = self.strategy_library_listbox.size()
            self.strategy_library_listbox.insert(
                END, f"  [{item.status_display}]  {item.name}  ({kb:.1f} KB){badge}{suffix}"
            )
            self.strategy_library_listbox.itemconfig(idx, fg=self._status_color(item.status))

        d = get_strategy_library_dir(mode)
        total = len(list_saved_strategies(mode))
        filtered = query or market_filter not in ("", "All markets") or \
            tag_filter not in ("", "All tags") or status_filter not in ("", "All statuses")
        misplaced = list_misplaced_files(mode)
        misplaced_note = ""
        if misplaced:
            shown_names = ", ".join(misplaced[:3]) + (f", +{len(misplaced) - 3} more" if len(misplaced) > 3 else "")
            misplaced_note = (
                f"\n\u26a0 {len(misplaced)} file(s) in this folder don't match the "
                f"{mode} extension and won't show up here: {shown_names}. If one of "
                f"these is a strategy you dropped in, it's probably in the wrong "
                f"language's subfolder -- move it into strategies/"
                f"{{python|pinescript|mql5}}/ as appropriate, then REFRESH LIBRARY."
            )
        if self._strategy_library_items:
            shown = (
                f"{len(self._strategy_library_items)} of {total} saved {mode} strategy(ies)"
                if filtered else f"{total} saved {mode} strategy(ies)"
            )
            self.strategy_library_status.config(text=f"{shown}  •  {d}{misplaced_note}", fg=TEXT_DIM)
        elif total and filtered:
            self.strategy_library_status.config(
                text=f"No saved {mode} strategies match the current search/filters.{misplaced_note}", fg=TEXT_DIM,
            )
        else:
            self.strategy_library_status.config(
                text=f"No saved {mode} strategies yet. Import one above, or drop a file "
                f"directly in {d} and press REFRESH LIBRARY.{misplaced_note}",
                fg=TEXT_DIM,
            )
        self._clear_strategy_metadata_panel()

    def _status_color(self, status: str) -> str:
        """Rough color cue for a strategy's lifecycle status, so it reads
        at a glance in the library listbox instead of only via the text
        label -- resolved at call time (not module load) since GREEN/RED/
        AMBER/etc. are theme globals that apply_theme() can overwrite."""
        return {
            "draft": TEXT_DIM,
            "tested_failed": RED,
            "tested_passed": AMBER,
            "validated": BLUE,
            "ready_for_demo": GREEN,
            "ready_for_live": GREEN,
        }.get(status, TEXT_DIM)

    def _selected_library_item(self):
        """First selected item, for actions that only make sense on one at
        a time (load, rename, edit info)."""
        items = self._selected_library_items()
        return items[0] if items else None

    def _selected_library_items(self):
        """All currently highlighted items, for bulk actions (delete,
        export). The listbox allows multi-select (Ctrl/Cmd/Shift-click)."""
        mode = self.strategy_mode.get()
        if mode not in STRATEGY_TYPES:
            return []
        sel = self.strategy_library_listbox.curselection()
        return [self._strategy_library_items[i] for i in sel]

    def _clear_strategy_metadata_panel(self):
        if hasattr(self, "strategy_meta_description"):
            self.strategy_meta_description.var.set("")
            self.strategy_meta_market.var.set("")
            self.strategy_meta_tags.var.set("")
            self.strategy_meta_status.var.set(STATUS_KEY_TO_LABEL["draft"])
            self.strategy_meta_last_run.config(text="")

    def _on_library_selection_changed(self):
        items = self._selected_library_items()
        if len(items) != 1:
            self._clear_strategy_metadata_panel()
            if len(items) > 1:
                self.strategy_meta_last_run.config(
                    text=f"{len(items)} strategies selected — bulk delete/export only "
                    "(load/rename/info apply to a single selection)."
                )
            return
        item = items[0]
        self.strategy_meta_description.var.set(item.metadata.get("description", ""))
        self.strategy_meta_market.var.set(item.metadata.get("market", ""))
        self.strategy_meta_tags.var.set(", ".join(item.tags))
        self.strategy_meta_status.var.set(item.status_display)

        lines = []
        last_run = item.metadata.get("last_run")
        if last_run:
            lines.append("Last run — " + "  •  ".join(f"{k}: {v}" for k, v in last_run.items()))
        lookahead = item.metadata.get("lookahead")
        if lookahead:
            lines.append(f"Lookahead — {'clean' if lookahead.get('clean') else 'FAILED'}: "
                         f"{lookahead.get('summary', '')}")
        last_search = item.metadata.get("last_search")
        if last_search:
            lines.append("Last search — " + "  •  ".join(f"{k}: {v}" for k, v in last_search.items()))
        self.strategy_meta_last_run.config(
            text="\n".join(lines) if lines else "No backtest, lookahead check, or search recorded yet."
        )

    def _save_selected_library_metadata(self):
        mode = self.strategy_mode.get()
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a single saved strategy from the list first.")
            return
        save_strategy_metadata(mode, item.name, {
            "description": self.strategy_meta_description.get_str().strip(),
            "market": self.strategy_meta_market.get_str().strip(),
        })
        tags_raw = self.strategy_meta_tags.get_str().strip()
        set_strategy_tags(mode, item.name, [t.strip() for t in tags_raw.split(",")] if tags_raw else [])
        status_label_selected = self.strategy_meta_status.get_str()
        set_strategy_status(mode, item.name, STATUS_LABEL_TO_KEY.get(status_label_selected, status_label_selected))
        self._refresh_strategy_library()

    def _load_library_item_into_active_slot(self, item):
        """Loads a StoredStrategy (from the main library list OR the Batch
        test queue) into the single 'active strategy' slot at the top of
        Step 02 Strategy -- the slot Full Pipeline (15), Refinement (06),
        Walk-Forward Opt (08), Sensitivity (10), and Multi-Objective (12)
        all read via _build_strategy(). Switches the STRATEGY SOURCE mode
        too, so loading a PineScript item while Python is the active tab
        still works -- that's what lets you step through a mixed-language
        Batch test queue and run each one through those single-strategy
        tabs without re-browsing for the file."""
        if self.strategy_mode.get() != item.strategy_type:
            self._set_strategy_mode(item.strategy_type)
        self.strategy_py_path = str(item.path)
        self._active_library_strategy = (item.strategy_type, item.name)
        self.strategy_file_status.config(
            text=f"Loaded from library: {item.name}  [{item.status_display}]",
            fg=GREEN,
        )

    def _load_selected_library_strategy(self):
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return
        self._load_library_item_into_active_slot(item)

    def _load_selected_from_batch_queue(self):
        """LOAD SELECTED FROM QUEUE -- picks the first highlighted row in
        the Batch test queue and loads it into the top active-strategy
        slot, so you can build the queue once and then step through it,
        one strategy at a time, for Full Pipeline / Refinement /
        Walk-Forward Opt / Sensitivity / Multi-Objective -- the
        single-strategy tabs that RUN BATCH TEST doesn't cover."""
        sel = list(self.batch_queue_listbox.curselection())
        if not sel:
            messagebox.showinfo("No selection", "Select a row in the Batch test queue below first.")
            return
        item = self._batch_queue[sel[0]]
        self._load_library_item_into_active_slot(item)

    def _rename_selected_library_strategy(self):
        mode = self.strategy_mode.get()
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a single saved strategy from the list first.")
            return
        new_name = simpledialog.askstring(
            "Rename saved strategy", "New filename:", initialvalue=item.name
        )
        if not new_name or new_name == item.name:
            return
        try:
            new_path = rename_saved_strategy(mode, item.name, new_name, overwrite=False)
        except StrategyAlreadyExists:
            if not messagebox.askyesno(
                "Name already taken",
                f"'{new_name}' is already a saved {mode} strategy. Overwrite it?",
            ):
                return
            new_path = rename_saved_strategy(mode, item.name, new_name, overwrite=True)
        if self.strategy_py_path == str(item.path):
            self.strategy_py_path = str(new_path)
            self._active_library_strategy = (mode, new_path.name)
        self._refresh_strategy_library()

    def _delete_selected_library_strategy(self):
        mode = self.strategy_mode.get()
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo("No selection", "Select one or more saved strategies from the list first.")
            return
        names = ", ".join(i.name for i in items)
        prompt = (
            f"Permanently delete '{items[0].name}' from the strategy library? This cannot be undone."
            if len(items) == 1 else
            f"Permanently delete {len(items)} strategies from the library? This cannot be undone.\n\n{names}"
        )
        if not messagebox.askyesno("Delete saved strategy(ies)", prompt):
            return
        deleted, failed = delete_many((mode, i.name) for i in items)
        if self._active_library_strategy and self._active_library_strategy[0] == mode and \
                any(self._active_library_strategy[1] == i.name for i in items):
            self.strategy_py_path = None
            self._active_library_strategy = None
            self.strategy_file_status.config(
                text="Only needed for Python / PineScript / MQL5 modes.",
                fg=TEXT_DIM,
            )
        if failed:
            messagebox.showwarning("Some deletions failed", "\n".join(failed))
        self._refresh_strategy_library()

    def _show_text_viewer(self, title: str, text: str):
        """Read-only popup showing raw text (strategy source, or a
        manual strategy's built config as JSON) with copy-to-clipboard --
        used by VIEW CODE / CONFIG."""
        win = Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("860x640")

        Label(
            win, text=title, bg=BG, fg=TEXT, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        txt = Text(
            text_frame, wrap="none", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            font=(MONO, 10), relief="flat", bd=0,
        )
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        hs = ttk.Scrollbar(text_frame, orient="horizontal", command=txt.xview)
        txt.config(yscrollcommand=vs.set, xscrollcommand=hs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        txt.insert("1.0", text)
        txt.config(state="disabled")

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 12))

        def _copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

        self._button(btn_row, "COPY TO CLIPBOARD", _copy, primary=True).pack(side="left")
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left", padx=8)

    def _view_selected_strategy_code(self):
        mode = self.strategy_mode.get()
        if mode == "manual":
            try:
                strategy = self._build_strategy()
            except Exception as exc:
                messagebox.showerror("Could not build strategy", str(exc))
                return
            text = json.dumps(strategy.config, indent=2)
            self._show_text_viewer(f"Manual strategy config -- {strategy.config.get('name', 'Manual Strategy')}", text)
            return
        if mode not in STRATEGY_TYPES:
            messagebox.showinfo("No strategy type selected", "Choose Python, PineScript, MQL5, or Manual above first.")
            return
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return
        try:
            text = load_strategy_text(mode, item.name)
        except Exception as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        self._show_text_viewer(f"{item.name}  ({mode})", text)

    def _open_progress_window(self, title: str):
        """A small Toplevel with a scrolling log -- used for TEST SELECTED
        (BATCH) so a multi-strategy run has somewhere to show progress
        without borrowing the Search Lab tab's own console."""
        win = Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("760x520")
        Label(win, text=title, bg=BG, fg=TEXT, font=_safe_font(11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        txt = Text(text_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 9), relief="flat", bd=0)
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        txt.config(yscrollcommand=vs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left")

        def append(msg: str) -> None:
            def _do():
                txt.insert(END, msg + "\n")
                txt.see(END)
            try:
                self.root.after(0, _do)
            except Exception:
                pass

        return win, append

    def _add_selected_to_batch_queue(self):
        """Stages the currently highlighted library row(s) into the Batch
        test queue below, instead of running anything immediately. This is
        deliberately a separate step from RUN BATCH TEST so Prop Rules (03)
        and Risk (04) can be configured *after* picking strategies but
        *before* the run actually starts."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved Python/PineScript/MQL5 strategies from the list "
                "above first (Ctrl/Cmd-click or Shift-click for more than one).",
            )
            return
        existing = {(s.strategy_type, s.name) for s in self._batch_queue}
        added = 0
        skipped_dupe = 0
        for item in items:
            key = (item.strategy_type, item.name)
            if key in existing:
                skipped_dupe += 1
                continue
            self._batch_queue.append(item)
            existing.add(key)
            self.batch_queue_listbox.insert(END, f"  [{item.strategy_type}] [{item.status_display}] {item.name}")
            added += 1
        self._refresh_batch_queue_status(added=added, skipped_dupe=skipped_dupe)

    def _refresh_batch_queue_status(self, added: int = 0, skipped_dupe: int = 0):
        n = len(self._batch_queue)
        if n == 0:
            self.batch_queue_status.config(
                text="Queue is empty -- select strategies above and click "
                "ADD SELECTED TO BATCH QUEUE.",
                fg=TEXT_DIM,
            )
            return
        bits = [f"{n} strategy(ies) queued"]
        if added:
            bits.append(f"+{added} just added")
        if skipped_dupe:
            bits.append(f"{skipped_dupe} already in queue, skipped")
        self.batch_queue_status.config(text="  •  ".join(bits) + " -- click RUN BATCH TEST when ready.", fg=GREEN)

    def _remove_selected_from_batch_queue(self):
        sel = list(self.batch_queue_listbox.curselection())
        if not sel:
            messagebox.showinfo("No selection", "Select one or more rows in the queue below first.")
            return
        for i in reversed(sel):
            self.batch_queue_listbox.delete(i)
            del self._batch_queue[i]
        self._refresh_batch_queue_status()

    def _clear_batch_queue(self):
        self.batch_queue_listbox.delete(0, END)
        self._batch_queue = []
        self._refresh_batch_queue_status()

    def _run_batch_queue_clicked(self):
        items = list(self._batch_queue)
        if not items:
            messagebox.showinfo(
                "Queue is empty",
                "Nothing is queued yet. Select one or more saved strategies above and click "
                "ADD SELECTED TO BATCH QUEUE first.",
            )
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1 before testing strategies.")
            return
        win, append = self._open_progress_window(f"Testing {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_batch_test_pipeline, args=(items, append), daemon=True,
        ).start()

    def _run_library_batch_test_pipeline(self, items, log):
        """Runs every queued Strategy Library item through
        app.orchestration.batch_test.run_batch_test -- the same pipeline
        Bulk Backtest already uses, just sourced from the Batch test queue
        instead of a fresh file upload, and recording each result back onto
        that strategy's own library metadata. Prop Rules and Risk are read
        fresh right here, so whatever is set on 03/04 at the moment RUN
        BATCH TEST is clicked is what every queued strategy gets tested
        against."""
        try:
            log(f"Loading {len(self.csv_paths)} market data file(s)...")
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    log(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                    return
                per_file_results.append((p, result))
            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            log(f"Loaded {len(df)} bars.\n")

            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )

            batch_items = []
            for item in items:
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped {item.name} -- could not load: {exc}")
                    continue
                batch_items.append(
                    BatchTestItem(label=item.name, strategy=strategy, library_ref=(item.strategy_type, item.name))
                )

            if not batch_items:
                log("\nNothing to test -- every queued strategy failed to load.")
                return

            modes_present = {item.strategy_type for item in items}
            prefix_mode = modes_present.pop() if len(modes_present) == 1 else "mixed"

            summary = run_batch_test(
                df, batch_items, risk, rules, OUTPUT_DIR,
                instrument=instrument, mc_sims=n_sims, mc_method=method,
                basename_prefix=f"library_{prefix_mode}", progress_cb=log,
            )
            if summary.succeeded:
                ranked = sorted(summary.succeeded, key=lambda o: o.eval_pass_probability, reverse=True)
                log("\nRanked by eval pass probability:")
                for o in ranked:
                    log(f"  {o.eval_pass_probability:5.1f}%  ${o.net_profit:>12,.2f}   {o.label}")
            # This whole method runs on a background thread (see
            # _run_batch_queue_clicked) -- Tkinter widgets can only safely
            # be touched from the main thread, so both refreshes are handed
            # to the mainloop via root.after instead of being called here
            # directly. Calling them straight from a worker thread is what
            # produces the occasional glitchy/frozen UI after a batch run.
            def _refresh_after_run():
                try:
                    self._refresh_dashboard()
                except Exception:
                    pass
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _optimize_selected_library_strategies(self):
        """OPTIMIZE SELECTED -- runs app.orchestration.quick_optimize against
        every currently-highlighted Strategy Library row. Unlike the Batch
        test queue, this acts directly on whatever's selected in the list
        right now (no separate queue/stage step), since it's meant as a
        quick "is this worth a real Full Pipeline run" check."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved strategies from the list above first "
                "(Ctrl/Cmd-click or Shift-click for more than one).",
            )
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data file in Step 1 before optimizing.")
            return
        win, append = self._open_progress_window(f"Optimizing {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_quick_optimize, args=(items, append), daemon=True,
        ).start()

    def _run_library_quick_optimize(self, items, log):
        from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

        try:
            log(f"Loading {len(self.csv_paths)} market data file(s)...")
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    log(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                    return
                per_file_results.append((p, result))
            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            log(f"Loaded {len(df)} bars.\n")

            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            cfg = QuickOptimizeConfig()
            results = []
            for i, item in enumerate(items, start=1):
                log(f"===== [{i}/{len(items)}] Optimizing: {item.name} =====")
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped -- could not load: {exc}\n")
                    continue
                try:
                    res = run_quick_optimize(df, strategy, risk, rules, cfg, progress_cb=lambda m: log(f"  {m}"))
                    results.append((item.name, res))
                except Exception as exc:
                    log(f"  Optimize failed: {exc}\n")
                    continue
                log("")

            if results:
                log("Summary (eval-pass probability, before -> after):")
                for name, res in sorted(results, key=lambda t: t[1].optimized_eval_pass_probability, reverse=True):
                    marker = "IMPROVED" if res.improved else "no improvement"
                    log(
                        f"  {name}: {res.baseline_eval_pass_probability:.1f}% -> "
                        f"{res.optimized_eval_pass_probability:.1f}%  ({marker})"
                    )

            def _refresh_after_run():
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _run_full_pipeline_queue_clicked(self):
        items = list(self._batch_queue)
        if not items:
            messagebox.showinfo(
                "Queue is empty",
                "Nothing is queued yet. Select one or more saved strategies above and click "
                "ADD SELECTED TO BATCH QUEUE first.",
            )
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1 before testing strategies.")
            return
        proceed = messagebox.askokcancel(
            "Run Full Pipeline on multiple strategies?",
            f"This runs the FULL Full Pipeline (baseline, walk-forward-aware GA search, "
            f"re-validated Monte Carlo, out-of-sample check, holdout check, ICIR gate) for all "
            f"{len(items)} queued strategy(ies), one after another. This is much slower per "
            f"strategy than RUN BATCH TEST -- for a large queue (dozens to hundreds) this can "
            f"run for a long time in the background. Continue?",
        )
        if not proceed:
            return
        win, append = self._open_progress_window(f"Full Pipeline: {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_full_pipeline_batch, args=(items, append), daemon=True,
        ).start()

    def _run_library_full_pipeline_batch(self, items, log):
        """Runs every queued Strategy Library item through
        app.orchestration.full_pipeline.run_full_pipeline_batch -- the same
        FULL 7-step pipeline the 15 Full Pipeline tab's single-strategy
        RUN FULL PIPELINE button uses, just looped over every strategy
        staged in the Batch test queue instead of one loaded strategy.
        GA population/generations/etc. and AI Assist settings are read
        fresh from the 15 Full Pipeline tab's own widgets, so whatever is
        configured there (and on 03/04) at the moment this is clicked is
        what every queued strategy runs with."""
        try:
            log(f"Loading {len(self.csv_paths)} market data file(s)...")
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    log(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                    return
                per_file_results.append((p, result))
            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            log(f"Loaded {len(df)} bars.\n")

            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )

            # Same settings the 15 Full Pipeline tab's own RUN FULL PIPELINE
            # button reads -- see _fullpipeline_run_pipeline. Falls back to
            # FullPipelineConfig()'s defaults if that tab was never opened
            # this session (its widgets are only created once the tab is
            # built, which happens unconditionally at startup, so this
            # should always be available in practice).
            try:
                metric_key = self._fp_metric_label_to_key.get(self.fp_metric.get_str(), "composite_prop_score")
                cfg = FullPipelineConfig(
                    n_folds=self.fp_folds.get_int(4),
                    window_mode=self.fp_window_mode.get_str(),
                    ga_population=self.fp_population.get_int(12),
                    ga_generations=self.fp_generations.get_int(6),
                    ga_search_mc_sims=self.fp_search_mc_sims.get_int(200),
                    fitness_metric=metric_key,
                    final_mc_sims=self.fp_final_mc_sims.get_int(10000),
                    holdout_frac=self.fp_holdout_frac.get_float(0.2),
                    oos_check_folds=self.fp_folds.get_int(4),
                    random_seed=self.fp_seed.get_int(42),
                    save_to_library=self.fp_save_to_library.var.get(),
                    library_status=self._fp_status_label_to_key.get(self.fp_library_status.get_str()),
                )
                ollama_settings = self._build_ollama_settings()
            except Exception:
                cfg = FullPipelineConfig()
                ollama_settings = None
            log(
                f"Using Full Pipeline settings from the 15 Full Pipeline tab: "
                f"population={cfg.ga_population}, generations={cfg.ga_generations}, "
                f"folds={cfg.n_folds}.\n"
            )

            batch_items = []
            for item in items:
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped {item.name} -- could not load: {exc}")
                    continue
                batch_items.append(
                    FullPipelineBatchItem(label=item.name, strategy=strategy, library_ref=(item.strategy_type, item.name))
                )

            if not batch_items:
                log("\nNothing to test -- every queued strategy failed to load.")
                return

            # Run several strategies' pipelines concurrently instead of one
            # after another -- Step 2's GA search (by far the dominant cost
            # per strategy) already parallelizes ACROSS a genome population;
            # this additionally parallelizes ACROSS strategies, which is
            # what actually shrinks a large multi-strategy batch's total
            # wall-clock time. Auto-picked from the core count rather than
            # a fixed number so it scales with whatever machine this runs
            # on without needing its own settings widget; each strategy's
            # own GA worker count is capped in turn by run_full_pipeline_batch
            # so the two forms of parallelism don't oversubscribe the CPU.
            max_parallel = max(1, min(len(batch_items), (os.cpu_count() or 4) // 2))
            summary = run_full_pipeline_batch(
                df, batch_items, risk, rules, OUTPUT_DIR / "full_pipeline",
                cfg=cfg, instrument=instrument, ollama_settings=ollama_settings, progress_cb=log,
                max_parallel_strategies=max_parallel,
            )
            if summary.succeeded:
                ranked = sorted(summary.succeeded, key=lambda o: o.eval_pass_probability, reverse=True)
                log("\nRanked by eval pass probability:")
                for o in ranked:
                    log(f"  {o.eval_pass_probability:5.1f}%  ${o.net_profit:>12,.2f}   [{o.verdict}]   {o.label}")
            if summary.failed:
                log("\nFailed / skipped:")
                for o in summary.failed:
                    log(f"  {o.label}: {o.reason}")

            # Background thread -- see _run_library_batch_test_pipeline for
            # why these refreshes are routed through root.after instead of
            # called directly.
            def _refresh_after_run():
                try:
                    self._refresh_dashboard()
                except Exception:
                    pass
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _open_strategy_library_folder(self):
        # This folder-open button is a frequent source of confusion: people
        # naturally double-click a .py/.pine/.mq5 file once Explorer/Finder
        # is open, which fails with an OS-level "no program configured to
        # open this file" error (Windows has no default app registered for
        # those extensions). That error comes from Windows itself, not this
        # app, and it can't be silently fixed here -- so we head it off with
        # an explicit heads-up *before* opening the folder, rather than
        # relying on the small print label under the button, which is easy
        # to miss or scroll past. Cancelling here skips opening the folder
        # entirely, so a person who reads the warning never sees the error.
        proceed = messagebox.askokcancel(
            "Before you open this folder",
            "This opens the raw strategy files on disk -- useful for backing "
            "them up or copying them to another device.\n\n"
            "Don't double-click a strategy file from there: Windows has no "
            "program registered for .py / .pine / .mq5 files and will show "
            "\"no program configured to open this file.\" That's expected "
            "and isn't something this app can fix.\n\n"
            "To actually use a saved strategy, come back to this screen, "
            "select it in the list above, and click LOAD SELECTED (or "
            "double-click it there instead).\n\n"
            "Click OK to open the folder anyway, or Cancel to go back.",
        )
        if not proceed:
            return
        d = get_strategy_library_dir(self.strategy_mode.get()) \
            if self.strategy_mode.get() in STRATEGY_TYPES else get_strategy_library_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # noqa: S606 -- opening a known local folder, not user input
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(d)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(d)])
        except OSError as exc:
            messagebox.showinfo("Strategy library folder", f"{d}\n\n(Couldn't open it automatically: {exc})")

    def _export_strategy_library(self):
        dest = filedialog.asksaveasfilename(
            title="Export strategy library",
            defaultextension=".zip",
            initialfile="t58_strategy_library_backup.zip",
            filetypes=[("Zip archive", "*.zip")],
        )
        if not dest:
            return
        try:
            export_library_zip(dest)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo(
            "Export complete",
            f"Exported the full strategy library to:\n{dest}\n\n"
            "Unzip it into your repo's strategies/ folder to keep a packaged "
            ".exe's saved strategies in sync with GitHub.",
        )

    def _export_selected_library_strategies(self):
        mode = self.strategy_mode.get()
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo("No selection", "Select one or more saved strategies from the list first.")
            return
        dest = filedialog.asksaveasfilename(
            title="Export selected strategies",
            defaultextension=".zip",
            initialfile="t58_strategy_selection.zip",
            filetypes=[("Zip archive", "*.zip")],
        )
        if not dest:
            return
        try:
            export_library_zip(dest, selection=[(mode, i.name) for i in items])
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete", f"Exported {len(items)} strategy(ies) to:\n{dest}")

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
        f = self._scrollable(self.tab_prop)

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
        f = self._scrollable(self.tab_risk)

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
        pip_detect_row = Frame(section, bg=PANEL)
        pip_detect_row.pack(anchor="w", padx=18, pady=(0, 8))
        self._button(
            pip_detect_row, "DETECT PIP SIZE FROM DATA", self._detect_pip_size_from_data
        ).pack(side="left")
        self.pip_detect_status = Label(
            pip_detect_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.pip_detect_status.pack(side="left", padx=(10, 0))

        adaptive_section = self._section(
            f, "Adaptive risk (optional)",
            "Declarative, engine-level money-management rules -- de-risk after a losing "
            "streak, cut size once a daily loss threshold is hit, or coast once a "
            "percentage of the way to the profit target. No strategy source can see its "
            "own trade outcomes at signal-generation time; these rules apply at the "
            "engine level instead, the same way the Daily Loss Limit circuit breaker "
            "(Step 3, Prop Rules) already does. Off by default -- nothing changes about "
            "position sizing unless enabled and at least one rule is added below. Active "
            "rules stack multiplicatively (e.g. two active 0.5x rules together size at "
            "0.25x, not 0.5x).",
        )
        self.adaptive_risk_enabled = LabeledCheckbox(adaptive_section, "Enable adaptive risk for Run & Report (Step 5)", False)
        self.adaptive_profit_target_pct = LabeledEntry(
            adaptive_section, "Profit target (% of balance, for 'progress to target' rules)", 8.0,
        )

        rules_frame = Frame(adaptive_section, bg=PANEL)
        rules_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))
        self.adaptive_rule_listbox = Listbox(
            rules_frame, height=5, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.adaptive_rule_listbox.pack(side="left", fill="both", expand=True)
        rule_scroll = ttk.Scrollbar(rules_frame, orient="vertical", command=self.adaptive_rule_listbox.yview, style="T58.Vertical.TScrollbar")
        rule_scroll.pack(side="right", fill="y")
        self.adaptive_rule_listbox.config(yscrollcommand=rule_scroll.set)

        rule_btn_row = Frame(adaptive_section, bg=PANEL)
        rule_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(rule_btn_row, "ADD RULE", self._adaptive_add_rule, primary=True).pack(side="left")
        self._button(rule_btn_row, "REMOVE SELECTED RULE", self._adaptive_remove_rule).pack(side="left", padx=8)

        self._adaptive_rules: list[AdaptiveRiskRule] = []

    def _adaptive_add_rule(self):
        rule = _AdaptiveRuleDialog.ask(self.root)
        if rule is None:
            return
        self._adaptive_rules.append(rule)
        self.adaptive_rule_listbox.insert(END, f"{rule.trigger} >= {rule.threshold}  ->  x{rule.risk_multiplier}")

    def _adaptive_remove_rule(self):
        sel = self.adaptive_rule_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.adaptive_rule_listbox.delete(idx)
        del self._adaptive_rules[idx]

    def _build_adaptive_risk_config(self) -> AdaptiveRiskConfig | None:
        """Returns None (not just a disabled config) when adaptive risk isn't
        enabled, so callers can pass it straight through to run_backtest()'s
        `adaptive_risk=` parameter unconditionally -- None there means
        "behave exactly as if this parameter never existed," same as a
        disabled config would, but avoids constructing one at all on the
        common path where nobody has touched this section."""
        if not self.adaptive_risk_enabled.get() or not self._adaptive_rules:
            return None
        target_pct = self.adaptive_profit_target_pct.get_float(8.0)
        balance = self.r_initial_balance.get_float(100000)
        return AdaptiveRiskConfig(
            enabled=True,
            rules=list(self._adaptive_rules),
            profit_target_amount=balance * target_pct / 100.0,
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

    def _detect_pip_size_from_data(self):
        """Suggests a pip_size from whatever's currently selected in Step 1
        (Market Data), rather than leaving pip_size at its FX default
        (0.0001) for non-FX instruments -- the single most common cause of
        a strategy's fixed-pips stop translating into a nonsensical
        position size. Only ever suggests a starting value; the person
        still confirms it by seeing it land in the field."""
        if not self.csv_paths:
            self.pip_detect_status.config(
                text="Select a market data CSV in Step 1 first.", fg=AMBER,
            )
            return
        try:
            result = import_csv(self.csv_paths[0])
            if not result.is_valid:
                self.pip_detect_status.config(text="Couldn't read that CSV.", fg=RED)
                return
            suggested = suggest_pip_size(result.dataframe)
            self.r_pip_size.var.set(str(suggested))
            self.pip_detect_status.config(
                text=f"Suggested {suggested} from {os.path.basename(self.csv_paths[0])} "
                     f"-- confirm this matches the instrument before running a backtest.",
                fg=GREEN,
            )
        except Exception as exc:
            self.pip_detect_status.config(text=f"Couldn't detect: {exc}", fg=RED)

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
            cost_stress_enabled=self.refine_cost_stress_enabled.get(),
            cost_stress_multiplier=self.refine_cost_stress_multiplier.get_float(2.0),
            cost_stress_penalty_weight=self.refine_cost_stress_weight.get_float(0.35),
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

        cost_stress_section = self._section(
            f, "Cost-stress penalty (on by default)",
            "ALSO re-backtests every candidate at spread/slippage/commission multiplied "
            "by the factor below, and blends that stressed-cost result into the fitness "
            "the GA actually selects on -- biasing the search toward strategies whose "
            "edge survives worse execution, not just strategies that look best under the "
            "default cost assumptions. Reported statistics always stay nominal (un-"
            "stressed); only the scalar the GA breeds toward is adjusted.",
        )
        self.refine_cost_stress_enabled = LabeledCheckbox(cost_stress_section, "Enable cost-stress penalty", True)
        self.refine_cost_stress_multiplier = LabeledEntry(cost_stress_section, "Cost multiplier for the stressed re-run", 2.0)
        self.refine_cost_stress_weight = LabeledEntry(cost_stress_section, "Penalty weight (0=ignore, 1=full)", 0.35)

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

        self.refine_progress = NeuralProgress(f)
        self.refine_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Refinement output", "Live search log.")
        self.refine_output = Text(
            output_section, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
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
        f = self._scrollable(self.tab_run)

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

        self.progress = NeuralProgress(f)
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
            bg=LOG_BG,
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
            adaptive_risk = self._build_adaptive_risk_config()
            if adaptive_risk is not None:
                self._log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} rule(s)")

            self._log("Running historical backtest...")
            bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)

            if strategy.source_type == "python":
                self._log("Checking for lookahead bias...")
                try:
                    lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
                    self._log(f"  {lookahead_result.summary()}")
                    active_lib_strategy = getattr(self, "_active_library_strategy", None)
                    if active_lib_strategy:
                        try:
                            record_lookahead_result(*active_lib_strategy, {
                                "clean": not lookahead_result.bug_detected,
                                "summary": lookahead_result.summary(),
                            })
                        except (FileNotFoundError, ValueError):
                            pass  # strategy renamed/deleted mid-run -- not worth failing the run over
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
            try:
                self._refresh_dashboard()
            except Exception:
                pass

            active_lib_strategy = getattr(self, "_active_library_strategy", None)
            if active_lib_strategy:
                lib_mode, lib_filename = active_lib_strategy
                try:
                    record_backtest_result(lib_mode, lib_filename, {
                        "trades": len(bt_result.trades),
                        "net_profit": round(bt_result.statistics.net_profit, 2),
                        "win_rate": round(bt_result.statistics.win_rate, 1),
                        "max_dd": round(bt_result.statistics.max_drawdown_pct, 2),
                        "passed_evaluation": single_run.passed_evaluation,
                        "report_html": str(paths["html"]),
                    })
                except (FileNotFoundError, ValueError):
                    pass  # strategy was renamed/deleted mid-run -- not worth failing the run over

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
        "Family -- named hypothesis grid (Manual strategies only)": "family_named",
        "Family -- grid around my current Strategy tab config (any strategy type)": "family_grid",
        "Single -- re-validate my current Strategy tab config exactly as configured": "single",
        "Bulk backtest -- upload multiple Python / PineScript / MQL5 files and run each one": "bulk_upload",
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
            "Works with Manual, Python, PineScript, and MQL5 strategies alike. "
            "Completely separate from the normal Run & Report pipeline and from Step 6 "
            "-- nothing here changes unless you click Run below.",
        )

        mode_section = self._section(
            f, "What to search",
            "\u2022 Named hypothesis grid expands one of this app's built-in trading hypotheses "
            "(Manual strategies only).\n"
            "\u2022 Grid around my current Strategy tab config discovers whatever's currently "
            "configured on Step 2's own tunable numeric parameters (indicator periods, stop/target "
            "values, or -- for Python/PineScript/MQL5 -- SCREAMING_SNAKE_CASE constants, "
            "input.int()/input.float() values, or iMA()/iRSI() periods) and grid-searches around "
            "them. Works for any of the 4 strategy types.\n"
            "\u2022 Single re-validates that one exact strategy through the same 5-stage funnel, "
            "with no parameter search at all -- an independent stress test.",
        )
        self.search_mode = LabeledCombo(
            mode_section, "Mode", list(self._SEARCH_MODE_LABELS.keys()),
            "Family -- named hypothesis grid (Manual strategies only)",
        )
        self.search_mode.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_search_mode_changed())

        family_labels = ["All families (search every hypothesis together)"] + [
            f"{label} [{name}]" for name, label in list_families().items()
        ]
        self._search_family_label_to_key = {"All families (search every hypothesis together)": "all"}
        for name, label in list_families().items():
            self._search_family_label_to_key[f"{label} [{name}]"] = name
        self.search_family = LabeledCombo(
            mode_section, "Named hypothesis family (Manual-only mode above)", family_labels, family_labels[0],
        )
        self.search_grid_points = LabeledEntry(
            mode_section, "Grid points per parameter (grid-around-config mode above)", 3,
        )

        bulk_section = self._section(
            f, "Strategies to upload (Bulk backtest mode above)",
            "Runs every file added here through the exact same pipeline as Run & Report -- "
            "full historical backtest, prop-firm simulation, Monte Carlo, and a saved HTML "
            "report each -- reusing the Prop Rules (Step 3) and Risk & Execution (Step 4) "
            "settings and the dataset(s) currently loaded on the Data tab, so every strategy "
            "is judged on the same terms. Mixing Python (.py), PineScript (.pine/.txt), and "
            "MQL5 (.mq5) files in the same batch is fine -- each is detected by extension. "
            "Every result is recorded automatically and shows up on the Dashboard afterward.",
        )
        bulk_list_frame = Frame(bulk_section, bg=PANEL)
        bulk_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.bulk_strategy_listbox = Listbox(
            bulk_list_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
        )
        self.bulk_strategy_listbox.pack(side="left", fill="both", expand=True)
        bulk_scroll = ttk.Scrollbar(
            bulk_list_frame, orient="vertical", command=self.bulk_strategy_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        bulk_scroll.pack(side="right", fill="y")
        self.bulk_strategy_listbox.config(yscrollcommand=bulk_scroll.set)

        bulk_btn_row = Frame(bulk_section, bg=BG)
        bulk_btn_row.pack(fill="x", padx=18, pady=(0, 12))
        self._button(bulk_btn_row, "ADD STRATEGY FILES...", self._bulk_add_files).pack(side="left")
        self._button(bulk_btn_row, "REMOVE SELECTED", self._bulk_remove_selected).pack(side="left", padx=8)
        self._button(bulk_btn_row, "CLEAR ALL", self._bulk_clear_files).pack(side="left")

        self._bulk_strategy_paths: list[Path] = []

        space_section = self._section(
            f, "Search space size",
            "How many generated candidates Stage 1 actually evaluates. If the full grid "
            "for a family is larger than this, a random (reproducible) sample is taken "
            "instead of just the first N -- see the random seed below.",
        )
        self.search_max_candidates = LabeledEntry(space_section, "Max candidates (Family mode)", 500)
        self.search_workers = LabeledEntry(space_section, "Parallel workers (blank = all CPU cores)", "")
        self.search_seed = LabeledEntry(space_section, "Random seed", 42)

        pair_section = self._section(
            f, "Pair instrument (Statistical Pairs / Relative Value family only)",
            "Merges a second instrument's close price in as a 'pair_close' column so the "
            "'stat_pairs' family can be searched. Leave blank to search every OTHER family "
            "as usual (a family=All search simply skips stat_pairs without this).",
        )
        pair_row = Frame(pair_section, bg=PANEL)
        pair_row.pack(fill="x", padx=18, pady=(2, 10))
        self.search_pair_csv_label = Label(
            pair_row, text="No pair instrument selected.", bg=PANEL, fg=TEXT_MUTED,
            font=_safe_font(9), anchor="w",
        )
        self.search_pair_csv_label.pack(side="left", fill="x", expand=True)
        self._button(pair_row, "CHOOSE PAIR CSV...", self._search_choose_pair_csv).pack(side="right")
        self._button(pair_row, "CLEAR", self._search_clear_pair_csv).pack(side="right", padx=(0, 8))
        self._search_pair_csv_path: str | None = None

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
        self.search_cost_stress_enabled = LabeledCheckbox(stage2_section, "Enable cost-stress penalty in the GA fitness", True)
        self.search_cost_stress_multiplier = LabeledEntry(stage2_section, "Cost multiplier for the stressed re-run", 2.0)
        self.search_cost_stress_weight = LabeledEntry(stage2_section, "Penalty weight (0=ignore, 1=full)", 0.35)

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

        self.search_progress = NeuralProgress(f)
        self.search_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Search Lab output", "Live funnel log.")
        self.search_output = Text(
            output_section, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
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

    def _bulk_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add strategy files (Python / PineScript / MQL5)",
            filetypes=[
                ("Supported strategy files", "*.py *.pine *.pinescript *.mq5 *.mqh *.txt"),
                ("Python strategies", "*.py"),
                ("PineScript strategies", "*.pine *.pinescript *.txt"),
                ("MQL5 strategies", "*.mq5 *.mqh"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        existing = {str(p) for p in self._bulk_strategy_paths}
        for p in paths:
            if p not in existing:
                self._bulk_strategy_paths.append(Path(p))
                self.bulk_strategy_listbox.insert(END, f"  {Path(p).name}")

    def _bulk_remove_selected(self):
        sel = list(self.bulk_strategy_listbox.curselection())
        for i in reversed(sel):
            self.bulk_strategy_listbox.delete(i)
            del self._bulk_strategy_paths[i]

    def _bulk_clear_files(self):
        self.bulk_strategy_listbox.delete(0, END)
        self._bulk_strategy_paths = []

    @staticmethod
    def _load_bulk_strategy(path: Path):
        """Detect a strategy's source type by extension and construct it --
        mirrors what the Strategy tab does per-type, just driven by a file
        list instead of the tab's radio selection."""
        suffix = path.suffix.lower()
        if suffix == ".py":
            return PythonStrategy(path)
        if suffix in (".pine", ".pinescript"):
            return PineScriptStrategy(path)
        if suffix in (".mq5", ".mqh"):
            return MQL5Strategy(path)
        if suffix == ".txt":
            # Ambiguous extension -- sniff for PineScript's declaration
            # syntax before falling back to treating it as one.
            try:
                head = path.read_text(encoding="utf-8", errors="ignore")[:400]
            except OSError:
                head = ""
            if "//@version" in head or "strategy(" in head or "indicator(" in head:
                return PineScriptStrategy(path)
            return PineScriptStrategy(path)
        raise StrategyError(f"Unsupported strategy file type: {path.name}")

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

    def _search_choose_pair_csv(self):
        path = filedialog.askopenfilename(
            title="Select the second instrument's market data CSV (for the stat_pairs family)",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not path:
            return
        self._search_pair_csv_path = path
        self.search_pair_csv_label.config(text=os.path.basename(path), fg=TEXT)

    def _search_clear_pair_csv(self):
        self._search_pair_csv_path = None
        self.search_pair_csv_label.config(text="No pair instrument selected.", fg=TEXT_MUTED)

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
            cost_stress_enabled=self.search_cost_stress_enabled.get(),
            cost_stress_multiplier=self.search_cost_stress_multiplier.get_float(2.0),
            cost_stress_penalty_weight=self.search_cost_stress_weight.get_float(0.35),
        )

    def _search_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV in Step 1.",
            )
            return

        mode_key = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str(), "family_named")
        if mode_key == "bulk_upload":
            if not self._bulk_strategy_paths:
                messagebox.showwarning(
                    "No strategy files added",
                    "Add at least one Python (.py), PineScript (.pine), or MQL5 (.mq5) "
                    "file above before running a bulk backtest.",
                )
                return
            self.search_output.delete("1.0", END)
            self.open_search_report_btn.config(state="disabled")
            self.promote_champion_btn.config(state="disabled")
            self.open_champion_report_btn.config(state="disabled")
            self.search_progress.start(10)
            threading.Thread(target=self._run_bulk_backtest_pipeline, daemon=True).start()
            return

        self.search_output.delete("1.0", END)
        self.open_search_report_btn.config(state="disabled")
        self.promote_champion_btn.config(state="disabled")
        self.open_champion_report_btn.config(state="disabled")
        self.search_progress.start(10)
        threading.Thread(target=self._search_run_pipeline, daemon=True).start()

    def _run_bulk_backtest_pipeline(self):
        """Runs every uploaded strategy file through the exact same
        backtest -> prop-firm sim -> Monte Carlo -> report pipeline as
        Run & Report, one after another, reusing the currently configured
        Prop Rules / Risk / Monte Carlo settings so every strategy is
        judged on the same terms. Each report funnels through
        generate_full_report(), so every result is automatically recorded
        into run_history and shows up on the Dashboard afterward -- no
        separate wiring needed here."""
        try:
            self._log_search(f"Loading {len(self.csv_paths)} market data file(s)...")
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
            self._log_search(f"Loaded {len(df)} bars.\n")

            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )
            period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))

            total = len(self._bulk_strategy_paths)
            results_summary = []

            for i, path in enumerate(self._bulk_strategy_paths, start=1):
                self._log_search(f"[{i}/{total}] {path.name}")
                try:
                    strategy = self._load_bulk_strategy(path)
                except (StrategyError, Exception) as exc:
                    self._log_search(f"  Skipped -- could not load strategy: {exc}\n")
                    continue

                try:
                    bt_result = run_backtest(df, strategy, risk)
                except Exception as exc:
                    self._log_search(f"  Skipped -- backtest error: {exc}\n")
                    continue

                if not bt_result.trades:
                    self._log_search("  Skipped -- 0 trades generated on this data.\n")
                    continue

                self._log_search(
                    f"  Trades: {len(bt_result.trades)}  "
                    f"Net profit: ${bt_result.statistics.net_profit:,.2f}  "
                    f"Win rate: {bt_result.statistics.win_rate:.1f}%"
                )

                trade_pnls = [t.pnl for t in bt_result.trades]
                trade_dates = [t.entry_time for t in bt_result.trades]
                single_run = simulate_account(trade_pnls, trade_dates, rules)

                mc_cfg = MonteCarloConfig(n_simulations=n_sims, method=method)
                mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)

                try:
                    holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
                except Exception:
                    holdout_comparison = None

                paths = generate_full_report(
                    output_dir=OUTPUT_DIR,
                    basename=f"bulk_{i:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)}",
                    strategy_name=path.stem,
                    strategy_source_type=strategy.source_type,
                    instrument=instrument,
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

                self._log_search(
                    f"  Eval pass probability: {mc_result.evaluation_pass_probability:.1f}%  "
                    f"Report: {paths['html'].name}\n"
                )
                results_summary.append({
                    "name": path.stem,
                    "net_profit": bt_result.statistics.net_profit,
                    "eval_pass": mc_result.evaluation_pass_probability,
                    "html": paths["html"],
                })

            self._log_search(f"\nDone. {len(results_summary)}/{total} strategies produced a report.")
            if results_summary:
                ranked = sorted(results_summary, key=lambda r: r["eval_pass"], reverse=True)
                self._log_search("\nRanked by eval pass probability:")
                for r in ranked:
                    self._log_search(
                        f"  {r['eval_pass']:5.1f}%  ${r['net_profit']:>12,.2f}   {r['name']}"
                    )
                self._last_search_html_path = ranked[0]["html"]
                self.open_search_report_btn.config(state="normal")
                self._log_search(
                    "\nOpen Leaderboard above opens the top-ranked strategy's report. "
                    "Full results for all strategies are on the Dashboard tab."
                )
            try:
                self._refresh_dashboard()
            except Exception:
                pass

        except Exception:
            self._log_search("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.search_progress.stop()

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

            has_pair_data = False
            if self._search_pair_csv_path:
                pair_result = import_csv(str(store_csv_path(self._search_pair_csv_path)))
                if not pair_result.is_valid:
                    self._log_search(
                        "Pair CSV import errors:\n" + "\n".join(pair_result.errors)
                    )
                    return
                df = merge_pair_series(df, pair_result.dataframe)
                has_pair_data = True
                self._log_search(
                    f"Merged pair instrument from {os.path.basename(self._search_pair_csv_path)} "
                    "(enables the 'stat_pairs' family)."
                )

            risk = self._build_risk_config()
            rules = self._build_prop_rules()

            mode_key = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str(), "family_named")
            if mode_key == "single":
                strategy = self._build_strategy()  # any of the 4 types, whatever's configured
                space = generate_search_space(mode="single", strategy=strategy)
            elif mode_key == "family_grid":
                strategy = self._build_strategy()
                space = generate_search_space(
                    mode="family", strategy=strategy,
                    grid_points_per_gene=self.search_grid_points.get_int(3),
                    max_candidates=self.search_max_candidates.get_int(500),
                    seed=self.search_seed.get_int(42),
                )
            else:  # family_named
                family_label = self.search_family.get_str()
                family_key = self._search_family_label_to_key.get(family_label, "all")
                space = generate_search_space(
                    mode="family", family=family_key,
                    max_candidates=self.search_max_candidates.get_int(500),
                    seed=self.search_seed.get_int(42),
                    has_pair_data=has_pair_data,
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

            active_lib_strategy = getattr(self, "_active_library_strategy", None)
            if mode_key in ("single", "family_grid") and active_lib_strategy and summary.leaderboard:
                try:
                    record_search_result(*active_lib_strategy, {
                        "candidates_tested": summary.total_candidates,
                        "best_fitness": round(summary.leaderboard[0].get("fitness", 0), 4),
                        "fitness_metric": stage_cfg.fitness_metric,
                        "report_html": str(report_paths["html"]),
                    })
                except (FileNotFoundError, ValueError):
                    pass  # base strategy renamed/deleted mid-search -- not worth failing the run over

            self._last_search_summary = summary
            self._last_search_space = space
            try:
                self._refresh_dashboard()
            except Exception:
                pass
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
        except PairDataError as exc:
            self._log_search(f"\nPair CSV error: {exc}")
        except StrategyError as exc:
            self._log_search(f"\nStrategy error: {exc}")
        except BrokenProcessPool:
            self._log_search(
                "\nSearch Lab crashed: a worker process was terminated abruptly "
                "(BrokenProcessPool).\n\n"
                "If you're running the built .exe: this is almost always caused "
                "by a .exe built before this app called "
                "multiprocessing.freeze_support() at startup -- without it, every "
                "worker process a packaged .exe spawns re-launches the whole app "
                "instead of running as a plain worker, and immediately dies. "
                "Rebuild/redownload the .exe (run_app.py now calls "
                "freeze_support() first thing) and try again.\n\n"
                "If you're running from source (python run_app.py) and still see "
                "this, it usually means a worker genuinely ran out of memory or "
                "crashed hard -- try lowering 'Parallel workers' on this tab, or "
                "reducing the candidate pool / population size."
            )
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


    # -----------------------------------------------------------------------
    # Validation Lab — shared helpers used by all six tabs below
    # -----------------------------------------------------------------------

    def _load_df_for_page(self, log_fn) -> "pd.DataFrame | None":
        """Loads (and, if multiple files are selected on Step 1, merges) the
        currently-selected market data the exact same way every other tab
        does. Returns None (after logging why) if nothing usable is
        selected -- callers should stop rather than proceed with no data."""
        if not self.csv_paths:
            log_fn("Please select a market data CSV in Step 1 (Market Data) first.")
            return None
        per_file_results = []
        for p in self.csv_paths:
            result = import_csv(p)
            if not result.is_valid:
                log_fn(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                return None
            per_file_results.append((p, result))
        if len(per_file_results) == 1:
            df = per_file_results[0][1].dataframe
        else:
            df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
        log_fn(f"Loaded {len(df)} bars.")
        return df

    def _validation_mc_config(self, n_simulations: int | None = None) -> MonteCarloConfig:
        n_sims = n_simulations if n_simulations is not None else self.mc_sims.get_int(10000)
        method = self.mc_method.get_str().strip() or "bootstrap"
        return MonteCarloConfig(n_simulations=n_sims, method=method)

    # -----------------------------------------------------------------------
    # Tab 8 — Walk-Forward Optimization
    # -----------------------------------------------------------------------

    def _build_wfo_tab(self):
        f = self._scrollable(self.tab_wfo)

        self._page_header(
            f,
            "08 / Validation Lab",
            "Walk-Forward Optimization",
            "Re-optimizes this strategy fresh on each rolling or anchored fold's "
            "training window using a small GA search, applies the winning "
            "configuration UNCHANGED to that fold's held-out test window, and "
            "chains every fold's out-of-sample trades into one continuous equity "
            "curve. This is the number to trust over a single in-sample backtest "
            "-- it never lets a fold's optimizer see the data it will be judged on.",
        )

        settings = self._section(
            f, "Fold settings",
            "'Rolling' slides a fixed-size train window forward each fold (better if "
            "you suspect a regime-dependent edge). 'Anchored' always starts training "
            "at bar 0 and grows it (better if you believe the edge is stable "
            "over time, so more data only helps).",
            emphasize=True,
        )
        self.wfo_window_mode = LabeledCombo(settings, "Window mode", ["rolling", "anchored"], "rolling")
        self.wfo_folds = LabeledEntry(settings, "Number of folds", 5)
        self.wfo_train_frac = LabeledEntry(settings, "Train fraction per fold (rolling mode)", 0.6)

        ga_settings = self._section(
            f, "Per-fold GA search settings",
            "A fresh, smaller GA search runs once per fold -- keep these modest, "
            "since the total work is population x generations x folds.",
        )
        self._wfo_metric_labels = list(FITNESS_METRICS.values())
        self._wfo_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.wfo_metric = LabeledCombo(
            ga_settings, "Fitness metric", self._wfo_metric_labels, FITNESS_METRICS["composite_prop_score"],
        )
        self.wfo_population = LabeledEntry(ga_settings, "Population size per fold", 8)
        self.wfo_generations = LabeledEntry(ga_settings, "Generations per fold", 3)
        self.wfo_seed = LabeledEntry(ga_settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN WALK-FORWARD OPTIMIZATION", self._wfo_run_clicked, primary=True).pack(side="left")
        self.open_wfo_report_btn = self._button(button_row, "OPEN REPORT", self._open_wfo_report)
        self.open_wfo_report_btn.config(state="disabled")
        self.open_wfo_report_btn.pack(side="left", padx=8)

        self.wfo_progress = NeuralProgress(f)
        self.wfo_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Walk-forward output", "Live progress log.")
        self.wfo_output = Text(
            output_section, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.wfo_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_wfo_html_path = None

    def _log_wfo(self, msg: str):
        self.wfo_output.insert(END, msg + "\n")
        self.wfo_output.see(END)
        self.root.update_idletasks()

    def _open_wfo_report(self):
        if self._last_wfo_html_path:
            webbrowser.open(f"file://{self._last_wfo_html_path.resolve()}")

    def _wfo_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.wfo_output.delete("1.0", END)
        self.wfo_progress.start(10)
        threading.Thread(target=self._wfo_run_pipeline, daemon=True).start()

    def _wfo_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_wfo)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            mc_cfg = self._validation_mc_config()

            metric_key = self._wfo_metric_label_to_key.get(self.wfo_metric.get_str(), "composite_prop_score")
            refine_cfg = RefinementConfig(
                population_size=self.wfo_population.get_int(8),
                generations=self.wfo_generations.get_int(3),
                fitness_metric=metric_key,
                random_seed=self.wfo_seed.get_int(42),
            )

            self._log_wfo("Starting walk-forward optimization...")
            result = run_walk_forward_optimization(
                df, strategy, risk, rules, mc_cfg,
                n_folds=self.wfo_folds.get_int(5),
                window_mode=self.wfo_window_mode.get_str(),
                train_frac=self.wfo_train_frac.get_float(0.6),
                refine_cfg=refine_cfg,
                random_seed=self.wfo_seed.get_int(42),
                progress_cb=self._log_wfo,
            )
            paths = generate_walk_forward_report(OUTPUT_DIR / "walk_forward", result)
            self._last_wfo_html_path = paths["html"]
            self.open_wfo_report_btn.config(state="normal")
            self._log_wfo("\nDone. Walk-forward optimization report written to:")
            for k, p in paths.items():
                self._log_wfo(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_wfo(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_wfo(f"\nWalk-forward optimization error: {exc}")
        except Exception:
            self._log_wfo("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.wfo_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 9 — CPCV / PBO
    # -----------------------------------------------------------------------

    def _build_cpcv_tab(self):
        f = self._scrollable(self.tab_cpcv)

        self._page_header(
            f,
            "09 / Validation Lab",
            "Combinatorial Purged Cross-Validation & PBO",
            "CPCV stress-tests this strategy across many different combinatorial "
            "train/test partitions of the same data, instead of just one holdout "
            "split. Probability of Backtest Overfitting (PBO) goes further: it "
            "checks a small POOL of candidate configurations (this strategy plus "
            "a few automatically perturbed variants) and reports the probability "
            "that whichever one looks best in-sample is really just noise.",
        )

        cpcv_settings = self._section(
            f, "CPCV settings (single strategy)",
            "Splits the data into N groups and evaluates every combination of "
            "k groups as the test set, with the rest as train.",
            emphasize=True,
        )
        self.cpcv_n_groups = LabeledEntry(cpcv_settings, "Number of groups (N)", 6)
        self.cpcv_n_test_groups = LabeledEntry(cpcv_settings, "Test groups per path (k)", 2)
        self.cpcv_metric = LabeledEntry(cpcv_settings, "Metric (e.g. profit_factor, sharpe_ratio, net_profit)", "profit_factor")
        self.cpcv_max_paths = LabeledEntry(cpcv_settings, "Max combinatorial paths to evaluate", 30)

        cpcv_btn_row = Frame(f, bg=BG)
        cpcv_btn_row.pack(fill="x", padx=24, pady=(4, 4))
        self._button(cpcv_btn_row, "RUN CPCV", self._cpcv_run_clicked, primary=True).pack(side="left")
        self.open_cpcv_report_btn = self._button(cpcv_btn_row, "OPEN CPCV REPORT", self._open_cpcv_report)
        self.open_cpcv_report_btn.config(state="disabled")
        self.open_cpcv_report_btn.pack(side="left", padx=8)

        Frame(f, bg=BORDER, height=1).pack(fill="x", padx=24, pady=14)

        pbo_settings = self._section(
            f, "PBO settings (candidate pool)",
            "The pool is this strategy's own configuration plus N-1 variants "
            "with its numeric parameters randomly perturbed -- a quick, "
            "self-contained way to check whether the search process itself is "
            "trustworthy. For a genuine multi-strategy PBO, use the CLI "
            "(--pbo) or call app.validation.cpcv.compute_pbo() directly with a "
            "Search Lab leaderboard slice.",
        )
        self.pbo_n_groups = LabeledEntry(pbo_settings, "Number of groups (N)", 6)
        self.pbo_n_test_groups = LabeledEntry(pbo_settings, "Test groups per path (k)", 2)
        self.pbo_metric = LabeledEntry(pbo_settings, "Metric to rank candidates by", "sharpe_ratio")
        self.pbo_max_paths = LabeledEntry(pbo_settings, "Max combinatorial paths to evaluate", 30)
        self.pbo_n_candidates = LabeledEntry(pbo_settings, "Number of candidates (baseline + perturbed)", 5)
        self.pbo_seed = LabeledEntry(pbo_settings, "Random seed (candidate perturbation)", 42)

        pbo_btn_row = Frame(f, bg=BG)
        pbo_btn_row.pack(fill="x", padx=24, pady=(4, 10))
        self._button(pbo_btn_row, "RUN PBO", self._pbo_run_clicked, primary=True).pack(side="left")
        self.open_pbo_report_btn = self._button(pbo_btn_row, "OPEN PBO REPORT", self._open_pbo_report)
        self.open_pbo_report_btn.config(state="disabled")
        self.open_pbo_report_btn.pack(side="left", padx=8)

        self.cpcv_progress = NeuralProgress(f)
        self.cpcv_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "CPCV / PBO output", "Live progress log.")
        self.cpcv_output = Text(
            output_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.cpcv_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_cpcv_html_path = None
        self._last_pbo_html_path = None

    def _log_cpcv(self, msg: str):
        self.cpcv_output.insert(END, msg + "\n")
        self.cpcv_output.see(END)
        self.root.update_idletasks()

    def _open_cpcv_report(self):
        if self._last_cpcv_html_path:
            webbrowser.open(f"file://{self._last_cpcv_html_path.resolve()}")

    def _open_pbo_report(self):
        if self._last_pbo_html_path:
            webbrowser.open(f"file://{self._last_pbo_html_path.resolve()}")

    def _cpcv_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.cpcv_output.delete("1.0", END)
        self.cpcv_progress.start(10)
        threading.Thread(target=self._cpcv_run_pipeline, daemon=True).start()

    def _cpcv_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_cpcv)
            if df is None:
                return
            risk = self._build_risk_config()
            # A fresh Strategy instance per call, since some strategy sources
            # cache state keyed to the data they last saw.
            strategy_builder = self._build_strategy

            self._log_cpcv("Running CPCV...")
            result = run_cpcv(
                df, strategy_builder, risk,
                n_groups=self.cpcv_n_groups.get_int(6),
                n_test_groups=self.cpcv_n_test_groups.get_int(2),
                metric=self.cpcv_metric.get_str().strip() or "profit_factor",
                max_paths=self.cpcv_max_paths.get_int(30),
            )
            paths = generate_cpcv_report(OUTPUT_DIR / "cpcv", result)
            self._last_cpcv_html_path = paths["html"]
            self.open_cpcv_report_btn.config(state="normal")
            self._log_cpcv(f"  Mean OOS {result.metric}: {result.mean_oos_metric:.3f}  (robust: {result.is_robust})")
            self._log_cpcv("\nDone. CPCV report written to:")
            for k, p in paths.items():
                self._log_cpcv(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_cpcv(f"\nStrategy error: {exc}")
        except CPCVError as exc:
            self._log_cpcv(f"\nCPCV error: {exc}")
        except Exception:
            self._log_cpcv("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.cpcv_progress.stop()

    def _pbo_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.cpcv_output.delete("1.0", END)
        self.cpcv_progress.start(10)
        threading.Thread(target=self._pbo_run_pipeline, daemon=True).start()

    def _pbo_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_cpcv)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()

            n_candidates = self.pbo_n_candidates.get_int(5)
            specs = [{"source_type": strategy.source_type,
                      "config": dict(strategy.config)} if strategy.source_type == "manual"
                     else {"source_type": strategy.source_type, "code_text": Path(strategy.file_path).read_text(),
                           "code_extension": Path(strategy.file_path).suffix}]

            if strategy.source_type == "manual":
                genes = extract_genome(strategy.config)
                rng = random.Random(self.pbo_seed.get_int(42))
                for _ in range(max(n_candidates - 1, 0)):
                    if not genes:
                        break
                    genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
                    specs.append({"source_type": "manual", "config": apply_genome(strategy.config, genes, genome)})
                if not genes:
                    self._log_cpcv(
                        "This strategy has no tunable numeric parameters -- PBO will run "
                        "with a single candidate, which is a degenerate (but still valid) case."
                    )
            else:
                self._log_cpcv(
                    f"'{strategy.source_type}' candidate perturbation isn't wired up in the "
                    "desktop UI yet -- running PBO with just this one strategy as the pool "
                    "(degenerate case). Use the CLI or compute_pbo() directly for a real "
                    "multi-candidate code-strategy pool."
                )

            self._log_cpcv(f"Running PBO across {len(specs)} candidate(s)...")
            result = compute_pbo(
                df, specs, risk,
                n_groups=self.pbo_n_groups.get_int(6),
                n_test_groups=self.pbo_n_test_groups.get_int(2),
                metric=self.pbo_metric.get_str().strip() or "sharpe_ratio",
                max_paths=self.pbo_max_paths.get_int(30),
            )
            paths = generate_pbo_report(OUTPUT_DIR / "pbo", result)
            self._last_pbo_html_path = paths["html"]
            self.open_pbo_report_btn.config(state="normal")
            self._log_cpcv(f"  PBO: {result.pbo * 100:.1f}%")
            self._log_cpcv("\nDone. PBO report written to:")
            for k, p in paths.items():
                self._log_cpcv(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_cpcv(f"\nStrategy error: {exc}")
        except CPCVError as exc:
            self._log_cpcv(f"\nPBO error: {exc}")
        except Exception:
            self._log_cpcv("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.cpcv_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 10 — Parameter Sensitivity
    # -----------------------------------------------------------------------

    def _build_sensitivity_tab(self):
        f = self._scrollable(self.tab_sensitivity)

        self._page_header(
            f,
            "10 / Validation Lab",
            "Parameter Sensitivity",
            "Sweeps every tunable numeric parameter across +/- a percentage of its "
            "current value, holding everything else fixed, and flags a 'cliff' "
            "wherever the metric drops sharply between adjacent steps -- the sign "
            "of a knife-edge parameter rather than a real, stable plateau. "
            "Optionally also produces a 2D heatmap for a chosen pair of parameters, "
            "since two parameters can interact even when each looks fine alone.",
        )

        settings = self._section(
            f, "Sweep settings",
            "Applies to every tunable numeric parameter this strategy has.",
            emphasize=True,
        )
        self.sens_metric = LabeledEntry(settings, "Metric (e.g. profit_factor, net_profit, sharpe_ratio)", "profit_factor")
        self.sens_pct_range = LabeledEntry(settings, "Sweep range (+/- fraction of current value)", 0.5)
        self.sens_steps = LabeledEntry(settings, "Steps per 1D sweep", 9)

        heatmap_settings = self._section(
            f, "Optional 2D heatmap",
            "Click LIST PARAMETERS to discover this strategy's tunable parameter "
            "labels, then pick two for a heatmap. Leave blank to skip the heatmap "
            "and only run the 1D sweeps above.",
        )
        list_row = Frame(heatmap_settings, bg=PANEL)
        list_row.pack(fill="x", padx=18, pady=(0, 6))
        self._button(list_row, "LIST PARAMETERS", self._sens_list_parameters).pack(side="left")
        self.sens_param_status = Label(
            list_row, text="No parameters listed yet.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.sens_param_status.pack(side="left", padx=10)

        self.sens_heatmap_a = LabeledCombo(heatmap_settings, "Heatmap parameter A", [], "")
        self.sens_heatmap_b = LabeledCombo(heatmap_settings, "Heatmap parameter B", [], "")

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN SENSITIVITY", self._sens_run_clicked, primary=True).pack(side="left")
        self.open_sens_report_btn = self._button(button_row, "OPEN REPORT", self._open_sens_report)
        self.open_sens_report_btn.config(state="disabled")
        self.open_sens_report_btn.pack(side="left", padx=8)

        self.sens_progress = NeuralProgress(f)
        self.sens_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Sensitivity output", "Live progress log.")
        self.sens_output = Text(
            output_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.sens_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_sens_html_path = None

    def _log_sens(self, msg: str):
        self.sens_output.insert(END, msg + "\n")
        self.sens_output.see(END)
        self.root.update_idletasks()

    def _open_sens_report(self):
        if self._last_sens_html_path:
            webbrowser.open(f"file://{self._last_sens_html_path.resolve()}")

    def _sens_list_parameters(self):
        try:
            strategy = self._build_strategy()
            labels = list_tunable_parameters(strategy)
            self.sens_heatmap_a.combo.config(values=labels)
            self.sens_heatmap_b.combo.config(values=labels)
            if labels:
                self.sens_param_status.config(
                    text=f"{len(labels)} parameter(s) found.", fg=GREEN,
                )
            else:
                self.sens_param_status.config(
                    text="This strategy has no tunable numeric parameters.", fg=AMBER,
                )
        except StrategyError as exc:
            messagebox.showerror("Strategy error", str(exc))
        except Exception:
            messagebox.showerror("Error", traceback.format_exc())

    def _sens_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.sens_output.delete("1.0", END)
        self.sens_progress.start(10)
        threading.Thread(target=self._sens_run_pipeline, daemon=True).start()

    def _sens_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_sens)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            mc_cfg = self._validation_mc_config(n_simulations=min(self.mc_sims.get_int(10000), 1000))
            metric = self.sens_metric.get_str().strip() or "profit_factor"

            self._log_sens("Running 1D sensitivity sweeps...")
            sweeps = compute_1d_sensitivity(
                df, strategy, risk, rules, mc_cfg, metric=metric,
                pct_range=self.sens_pct_range.get_float(0.5), n_steps=self.sens_steps.get_int(9),
            )
            for r in sweeps:
                flag = " <-- CLIFF" if r.cliff_detected else ""
                self._log_sens(f"  {r.gene_label}: max adjacent-step drop {r.max_pct_drop_between_adjacent_steps:.0f}%{flag}")

            heatmap = None
            a_label, b_label = self.sens_heatmap_a.get_str().strip(), self.sens_heatmap_b.get_str().strip()
            if a_label and b_label:
                self._log_sens(f"Running 2D heatmap for {a_label} x {b_label}...")
                heatmap = compute_2d_heatmap(df, strategy, risk, rules, mc_cfg, a_label, b_label, metric=metric)

            paths = generate_sensitivity_report(OUTPUT_DIR / "sensitivity", sweeps, heatmap)
            self._last_sens_html_path = paths["html"]
            self.open_sens_report_btn.config(state="normal")
            self._log_sens("\nDone. Sensitivity report written to:")
            for k, p in paths.items():
                self._log_sens(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_sens(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_sens(f"\nSensitivity error: {exc}")
        except Exception:
            self._log_sens("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.sens_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 11 — Multi-Asset Portfolio
    # -----------------------------------------------------------------------

    def _build_portfolio_tab(self):
        f = self._scrollable(self.tab_portfolio)

        self._page_header(
            f,
            "11 / Validation Lab",
            "Multi-Asset Portfolio",
            "Applies the strategy currently configured on Step 2 to every instrument "
            "listed below, computes the correlation matrix of their daily returns, "
            "re-weights each instrument's risk (correlated instruments get sized "
            "down), and merges every instrument's trades into one shared account "
            "equity curve -- the way trading a portfolio out of one prop account "
            "actually works.",
        )

        legs_section = self._section(
            f, "Instrument legs (at least 2 required)",
            "Each leg uses the SAME strategy configuration from Step 2 and the SAME "
            "base risk settings from Step 4, applied to that instrument's own data.",
            emphasize=True,
        )
        legs_frame = Frame(legs_section, bg=PANEL)
        legs_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.portfolio_leg_listbox = Listbox(
            legs_frame, height=6, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.portfolio_leg_listbox.pack(side="left", fill="both", expand=True)
        leg_scroll = ttk.Scrollbar(legs_frame, orient="vertical", command=self.portfolio_leg_listbox.yview, style="T58.Vertical.TScrollbar")
        leg_scroll.pack(side="right", fill="y")
        self.portfolio_leg_listbox.config(yscrollcommand=leg_scroll.set)

        leg_btn_row = Frame(legs_section, bg=PANEL)
        leg_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(leg_btn_row, "ADD LEG (CSV)", self._portfolio_add_leg, primary=True).pack(side="left")
        self._button(leg_btn_row, "REMOVE SELECTED LEG", self._portfolio_remove_leg).pack(side="left", padx=8)

        self._portfolio_legs: list[dict] = []

        settings = self._section(f, "Portfolio settings", "")
        self.portfolio_balance = LabeledEntry(settings, "Shared initial account balance ($)", 100000)
        self.portfolio_corr_strength = LabeledEntry(
            settings, "Correlation penalty strength (0=ignore, 1=full re-weighting)", 0.6,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN PORTFOLIO BACKTEST", self._portfolio_run_clicked, primary=True).pack(side="left")
        self.open_portfolio_report_btn = self._button(button_row, "OPEN REPORT", self._open_portfolio_report)
        self.open_portfolio_report_btn.config(state="disabled")
        self.open_portfolio_report_btn.pack(side="left", padx=8)

        self.portfolio_progress = NeuralProgress(f)
        self.portfolio_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Portfolio output", "Live progress log.")
        self.portfolio_output = Text(
            output_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.portfolio_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_portfolio_html_path = None

    def _log_portfolio(self, msg: str):
        self.portfolio_output.insert(END, msg + "\n")
        self.portfolio_output.see(END)
        self.root.update_idletasks()

    def _open_portfolio_report(self):
        if self._last_portfolio_html_path:
            webbrowser.open(f"file://{self._last_portfolio_html_path.resolve()}")

    def _portfolio_add_leg(self):
        path = filedialog.askopenfilename(
            title="Select market data CSV for this instrument",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not path:
            return
        weight = simpledialog.askfloat(
            "Instrument weight",
            f"Nominal (pre-correlation) risk weight for {os.path.basename(path)}:",
            initialvalue=1.0, minvalue=0.01, maxvalue=10.0,
        )
        if weight is None:
            return
        self._portfolio_legs.append({"path": path, "weight": weight})
        self.portfolio_leg_listbox.insert(END, f"{os.path.basename(path)}  (weight={weight:g})")

    def _portfolio_remove_leg(self):
        sel = self.portfolio_leg_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.portfolio_leg_listbox.delete(idx)
        del self._portfolio_legs[idx]

    def _portfolio_run_clicked(self):
        if len(self._portfolio_legs) < 2:
            messagebox.showwarning(
                "Not enough legs",
                "Portfolio backtesting requires at least 2 instrument legs -- use ADD LEG (CSV) above.",
            )
            return
        self.portfolio_output.delete("1.0", END)
        self.portfolio_progress.start(10)
        threading.Thread(target=self._portfolio_run_pipeline, daemon=True).start()

    def _portfolio_run_pipeline(self):
        try:
            legs = []
            for leg_spec in self._portfolio_legs:
                stored_path = str(store_csv_path(leg_spec["path"]))
                result = import_csv(stored_path)
                if not result.is_valid:
                    self._log_portfolio(f"Import errors ({os.path.basename(stored_path)}):\n" + "\n".join(result.errors))
                    return
                self._log_portfolio(f"Loaded {len(result.dataframe)} bars from {os.path.basename(stored_path)}")
                legs.append(InstrumentLeg(
                    name=Path(stored_path).stem, df=result.dataframe,
                    strategy=self._build_strategy(), risk=self._build_risk_config(),
                    weight=leg_spec["weight"],
                ))

            self._log_portfolio(f"Running portfolio backtest across {len(legs)} instrument(s)...")
            config = PortfolioConfig(
                initial_balance=self.portfolio_balance.get_float(100000),
                correlation_penalty_strength=self.portfolio_corr_strength.get_float(0.6),
            )
            result = run_portfolio_backtest(legs, config)
            paths = generate_portfolio_report(OUTPUT_DIR / "portfolio", result)
            self._last_portfolio_html_path = paths["html"]
            self.open_portfolio_report_btn.config(state="normal")
            self._log_portfolio(f"  Combined net profit: ${result.combined_statistics.net_profit:,.2f}")
            for w in result.warnings:
                self._log_portfolio(f"  WARNING: {w}")
            self._log_portfolio("\nDone. Portfolio report written to:")
            for k, p in paths.items():
                self._log_portfolio(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_portfolio(f"\nStrategy error: {exc}")
        except PortfolioError as exc:
            self._log_portfolio(f"\nPortfolio error: {exc}")
        except Exception:
            self._log_portfolio("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.portfolio_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 12 — Multi-Objective Optimization
    # -----------------------------------------------------------------------

    def _build_multiobj_tab(self):
        f = self._scrollable(self.tab_multiobj)

        self._page_header(
            f,
            "12 / Validation Lab",
            "Multi-Objective Optimization",
            "Runs a real NSGA-II search across several objectives at once (e.g. "
            "Sharpe, max drawdown, eval-pass probability) instead of collapsing "
            "them into one weighted score the way Iterative Refinement's GA does. "
            "Produces a Pareto front -- a set of candidates where none is "
            "strictly worse than any other on the front. Picking a final winner "
            "from that list is left as your call.",
        )

        obj_section = self._section(
            f, "Objectives (pick at least 2)",
            "Direction (higher/lower is better) is handled automatically for each.",
            emphasize=True,
        )
        self._multiobj_vars: dict[str, BooleanVar] = {}
        obj_grid = Frame(obj_section, bg=PANEL)
        obj_grid.pack(fill="x", padx=18, pady=(0, 10))
        for i, (obj_name, direction) in enumerate(OBJECTIVE_DIRECTIONS.items()):
            var = BooleanVar(value=obj_name in DEFAULT_OBJECTIVES)
            self._multiobj_vars[obj_name] = var
            cb = Checkbutton(
                obj_grid, text=f"{obj_name}  ({direction})", variable=var,
                bg=PANEL, fg=TEXT_MUTED, activebackground=PANEL, selectcolor=PANEL_3,
                highlightthickness=0, bd=0, font=_safe_font(9), anchor="w",
            )
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=6, pady=3)

        settings = self._section(f, "Search settings", "")
        self.mo_population = LabeledEntry(settings, "Population size", 20)
        self.mo_generations = LabeledEntry(settings, "Generations", 8)
        self.mo_seed = LabeledEntry(settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN MULTI-OBJECTIVE SEARCH", self._multiobj_run_clicked, primary=True).pack(side="left")
        self.open_multiobj_report_btn = self._button(button_row, "OPEN REPORT", self._open_multiobj_report)
        self.open_multiobj_report_btn.config(state="disabled")
        self.open_multiobj_report_btn.pack(side="left", padx=8)

        self.multiobj_progress = NeuralProgress(f)
        self.multiobj_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Multi-objective output", "Live progress log.")
        self.multiobj_output = Text(
            output_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.multiobj_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_multiobj_html_path = None

    def _log_multiobj(self, msg: str):
        self.multiobj_output.insert(END, msg + "\n")
        self.multiobj_output.see(END)
        self.root.update_idletasks()

    def _open_multiobj_report(self):
        if self._last_multiobj_html_path:
            webbrowser.open(f"file://{self._last_multiobj_html_path.resolve()}")

    def _multiobj_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        selected = [name for name, var in self._multiobj_vars.items() if var.get()]
        if len(selected) < 2:
            messagebox.showwarning("Not enough objectives", "Select at least 2 objectives.")
            return
        self.multiobj_output.delete("1.0", END)
        self.multiobj_progress.start(10)
        threading.Thread(target=self._multiobj_run_pipeline, args=(selected,), daemon=True).start()

    def _multiobj_run_pipeline(self, selected_objectives: list[str]):
        try:
            df = self._load_df_for_page(self._log_multiobj)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            mc_cfg = self._validation_mc_config()

            mo_cfg = MultiObjectiveConfig(
                objectives=selected_objectives,
                population_size=self.mo_population.get_int(20),
                generations=self.mo_generations.get_int(8),
                random_seed=self.mo_seed.get_int(42),
            )
            self._log_multiobj(f"Running multi-objective search for: {selected_objectives}...")
            result = run_multi_objective_refinement(df, strategy, risk, rules, mc_cfg, mo_cfg, progress_cb=self._log_multiobj)
            paths = generate_multi_objective_report(OUTPUT_DIR / "multi_objective", result)
            self._last_multiobj_html_path = paths["html"]
            self.open_multiobj_report_btn.config(state="normal")
            self._log_multiobj(f"  Final Pareto front size: {len(result.pareto_front)}")
            self._log_multiobj("\nDone. Multi-objective report written to:")
            for k, p in paths.items():
                self._log_multiobj(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_multiobj(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_multiobj(f"\nMulti-objective error: {exc}")
        except Exception:
            self._log_multiobj("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.multiobj_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 13 — Walk-Forward-Aware GA
    # -----------------------------------------------------------------------

    def _build_wfga_tab(self):
        f = self._scrollable(self.tab_wfga)

        self._page_header(
            f,
            "13 / Validation Lab",
            "Walk-Forward-Aware GA",
            "Same crossover/mutation/tournament-selection GA as Iterative Refinement "
            "(Step 6), but every candidate's fitness is scored ONLY on chained "
            "out-of-sample fold data -- never on the training windows or the full "
            "dataset. A genome that only fits one historical stretch simply scores "
            "lower here and gets selected against, generation over generation.",
        )

        settings = self._section(
            f, "Fold + search settings",
            "'Rolling' or 'anchored' fold windows, same meaning as Walk-Forward Optimization (Step 8).",
            emphasize=True,
        )
        self.wfga_window_mode = LabeledCombo(settings, "Window mode", ["rolling", "anchored"], "rolling")
        self.wfga_folds = LabeledEntry(settings, "Number of folds", 4)
        self._wfga_metric_labels = list(FITNESS_METRICS.values())
        self._wfga_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.wfga_metric = LabeledCombo(
            settings, "Fitness metric", self._wfga_metric_labels, FITNESS_METRICS["composite_prop_score"],
        )
        self.wfga_population = LabeledEntry(settings, "Population size", 12)
        self.wfga_generations = LabeledEntry(settings, "Generations", 6)
        self.wfga_seed = LabeledEntry(settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN WALK-FORWARD-AWARE GA", self._wfga_run_clicked, primary=True).pack(side="left")
        self.open_wfga_report_btn = self._button(button_row, "OPEN REPORT", self._open_wfga_report)
        self.open_wfga_report_btn.config(state="disabled")
        self.open_wfga_report_btn.pack(side="left", padx=8)

        self.wfga_progress = NeuralProgress(f)
        self.wfga_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Walk-forward-aware GA output", "Live progress log.")
        self.wfga_output = Text(
            output_section, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.wfga_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_wfga_html_path = None

    def _log_wfga(self, msg: str):
        self.wfga_output.insert(END, msg + "\n")
        self.wfga_output.see(END)
        self.root.update_idletasks()

    def _open_wfga_report(self):
        if self._last_wfga_html_path:
            webbrowser.open(f"file://{self._last_wfga_html_path.resolve()}")

    def _wfga_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.wfga_output.delete("1.0", END)
        self.wfga_progress.start(10)
        threading.Thread(target=self._wfga_run_pipeline, daemon=True).start()

    def _wfga_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_wfga)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            mc_cfg = self._validation_mc_config()

            metric_key = self._wfga_metric_label_to_key.get(self.wfga_metric.get_str(), "composite_prop_score")
            refine_cfg = RefinementConfig(
                population_size=self.wfga_population.get_int(12),
                generations=self.wfga_generations.get_int(6),
                fitness_metric=metric_key,
                random_seed=self.wfga_seed.get_int(42),
            )

            self._log_wfga("Starting walk-forward-aware GA...")
            result = run_walkforward_aware_refinement(
                df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg,
                n_folds=self.wfga_folds.get_int(4), window_mode=self.wfga_window_mode.get_str(),
                progress_cb=self._log_wfga,
            )
            paths = generate_walkforward_ga_report(OUTPUT_DIR / "walkforward_ga", result)
            self._last_wfga_html_path = paths["html"]
            self.open_wfga_report_btn.config(state="normal")
            self._log_wfga(f"  Best chained-OOS fitness: {result.best.fitness:.3f}  (overfitting gap: {result.overfitting_gap})")
            for w in result.warnings:
                self._log_wfga(f"  WARNING: {w}")
            self._log_wfga("\nDone. Walk-forward-aware GA report written to:")
            for k, p in paths.items():
                self._log_wfga(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_wfga(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_wfga(f"\nWalk-forward-aware GA error: {exc}")
        except Exception:
            self._log_wfga("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.wfga_progress.stop()


    # -----------------------------------------------------------------------
    # Tab 14 — Multi-Strategy Ensemble (blend or vote)
    # -----------------------------------------------------------------------

    _ENSEMBLE_MODE_LABELS = {
        "Blend (each strategy trades independently, correlation-adjusted risk)": "blend",
        "Vote (one combined signal, entered only once enough strategies agree)": "vote",
    }

    def _build_ensemble_tab(self):
        f = self._scrollable(self.tab_ensemble)

        self._page_header(
            f,
            "14 / Finding an Edge",
            "Multi-Strategy Ensemble",
            "The mirror case of Step 11's Multi-Asset Portfolio: several DIFFERENT "
            "strategies combined on the SAME instrument, instead of one strategy across "
            "several instruments. Two modes -- Blend keeps every strategy trading "
            "independently at a correlation-adjusted share of the account's risk budget "
            "(reuses the Portfolio feature's own math); Vote combines every strategy's "
            "signal into one entry, taken only once enough of them agree on direction. "
            "Uses the market data currently selected on Step 1.",
        )

        legs_section = self._section(
            f, "Strategy legs (at least 2 required)",
            "Python (.py), PineScript (.pine), or MQL5 (.mq5) files -- mixing types in "
            "the same ensemble is fine, each is detected by extension. Each leg trades "
            "the SAME instrument, using the SAME base risk settings from Step 4.",
            emphasize=True,
        )
        legs_frame = Frame(legs_section, bg=PANEL)
        legs_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.ensemble_leg_listbox = Listbox(
            legs_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ensemble_leg_listbox.pack(side="left", fill="both", expand=True)
        leg_scroll = ttk.Scrollbar(legs_frame, orient="vertical", command=self.ensemble_leg_listbox.yview, style="T58.Vertical.TScrollbar")
        leg_scroll.pack(side="right", fill="y")
        self.ensemble_leg_listbox.config(yscrollcommand=leg_scroll.set)

        leg_btn_row = Frame(legs_section, bg=PANEL)
        leg_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(leg_btn_row, "ADD STRATEGY FILES...", self._ensemble_add_legs, primary=True).pack(side="left")
        self._button(leg_btn_row, "REMOVE SELECTED", self._ensemble_remove_legs).pack(side="left", padx=8)
        self._button(leg_btn_row, "CLEAR ALL", self._ensemble_clear_legs).pack(side="left")

        self._ensemble_leg_paths: list[Path] = []

        settings = self._section(f, "Ensemble settings", "")
        self.ensemble_mode = LabeledCombo(
            settings, "Combination mode", list(self._ENSEMBLE_MODE_LABELS.keys()),
            "Blend (each strategy trades independently, correlation-adjusted risk)",
        )
        self.ensemble_min_agreement = LabeledEntry(settings, "Min agreement (Vote mode only)", 2)
        self.ensemble_balance = LabeledEntry(settings, "Shared initial account balance ($)", 100000)
        self.ensemble_corr_strength = LabeledEntry(
            settings, "Correlation penalty strength (Blend mode, 0=ignore, 1=full re-weighting)", 0.6,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN ENSEMBLE BACKTEST", self._ensemble_run_clicked, primary=True).pack(side="left")
        self.open_ensemble_report_btn = self._button(button_row, "OPEN REPORT", self._open_ensemble_report)
        self.open_ensemble_report_btn.config(state="disabled")
        self.open_ensemble_report_btn.pack(side="left", padx=8)

        self.ensemble_progress = NeuralProgress(f)
        self.ensemble_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Ensemble output", "Live progress log.")
        self.ensemble_output = Text(
            output_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ensemble_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_ensemble_html_path = None

    def _log_ensemble(self, msg: str):
        self.ensemble_output.insert(END, msg + "\n")
        self.ensemble_output.see(END)
        self.root.update_idletasks()

    def _open_ensemble_report(self):
        if self._last_ensemble_html_path:
            webbrowser.open(f"file://{self._last_ensemble_html_path.resolve()}")

    def _ensemble_add_legs(self):
        paths = filedialog.askopenfilenames(
            title="Add strategy files for this ensemble (Python / PineScript / MQL5)",
            filetypes=[
                ("Supported strategy files", "*.py *.pine *.pinescript *.mq5 *.mqh *.txt"),
                ("Python strategies", "*.py"),
                ("PineScript strategies", "*.pine *.pinescript *.txt"),
                ("MQL5 strategies", "*.mq5 *.mqh"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        existing = {str(p) for p in self._ensemble_leg_paths}
        for p in paths:
            if p not in existing:
                self._ensemble_leg_paths.append(Path(p))
                self.ensemble_leg_listbox.insert(END, f"  {Path(p).name}")

    def _ensemble_remove_legs(self):
        sel = list(self.ensemble_leg_listbox.curselection())
        for i in reversed(sel):
            self.ensemble_leg_listbox.delete(i)
            del self._ensemble_leg_paths[i]

    def _ensemble_clear_legs(self):
        self.ensemble_leg_listbox.delete(0, END)
        self._ensemble_leg_paths = []

    def _ensemble_run_clicked(self):
        if len(self._ensemble_leg_paths) < 2:
            messagebox.showwarning(
                "Not enough legs",
                "An ensemble requires at least 2 strategy legs -- use ADD STRATEGY FILES above.",
            )
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.ensemble_output.delete("1.0", END)
        self.open_ensemble_report_btn.config(state="disabled")
        self.ensemble_progress.start(10)
        threading.Thread(target=self._ensemble_run_pipeline, daemon=True).start()

    def _ensemble_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_ensemble)
            if df is None:
                return

            strategies, names = [], []
            for path in self._ensemble_leg_paths:
                try:
                    strategies.append(self._load_bulk_strategy(path))
                except (StrategyError, Exception) as exc:
                    self._log_ensemble(f"Skipped '{path.name}' -- could not load strategy: {exc}")
                    continue
                names.append(path.stem)
            if len(strategies) < 2:
                self._log_ensemble("\nFewer than 2 strategy legs loaded successfully -- nothing to combine.")
                return
            self._log_ensemble(f"Loaded {len(strategies)} strategy leg(s): {', '.join(names)}")

            mode_label = self.ensemble_mode.get_str()
            mode = self._ENSEMBLE_MODE_LABELS.get(mode_label, "blend")
            balance = self.ensemble_balance.get_float(100000)
            risk = RiskConfig(initial_balance=balance)
            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )

            if mode == "blend":
                self._log_ensemble(f"Running blend ensemble across {len(strategies)} strategies...")
                config = PortfolioConfig(
                    initial_balance=balance,
                    correlation_penalty_strength=self.ensemble_corr_strength.get_float(0.6),
                )
                result = run_ensemble_blend(df, strategies, risk, names=names, config=config)
                paths = generate_portfolio_report(OUTPUT_DIR / "ensemble", result)
                self._last_ensemble_html_path = paths["html"]
                self.open_ensemble_report_btn.config(state="normal")
                self._log_ensemble(f"  Combined net profit: ${result.combined_statistics.net_profit:,.2f}")
                for w in result.warnings:
                    self._log_ensemble(f"  WARNING: {w}")
                self._log_ensemble("\nDone. Ensemble (blend) report written to:")
                for k, p in paths.items():
                    self._log_ensemble(f"  {k}: {p}")

            elif mode == "vote":
                min_agreement = self.ensemble_min_agreement.get_int(2)
                self._log_ensemble(
                    f"Running vote ensemble across {len(strategies)} strategies "
                    f"(min_agreement={min_agreement})..."
                )
                bt_result = run_ensemble_vote(
                    df, strategies, risk, names=names,
                    vote_config=EnsembleVoteConfig(min_agreement=min_agreement),
                )
                self._log_ensemble(
                    f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}"
                )
                if not bt_result.trades:
                    self._log_ensemble("\nNo trades were generated by this vote ensemble -- nothing further to report.")
                    return
                rules = PropRules(account_size=balance)
                period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
                trade_pnls = [t.pnl for t in bt_result.trades]
                trade_dates = [t.entry_time for t in bt_result.trades]
                single_run = simulate_account(trade_pnls, trade_dates, rules)
                mc_cfg = self._validation_mc_config()
                mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)
                paths = generate_full_report(
                    output_dir=OUTPUT_DIR / "ensemble", strategy_name=bt_result.strategy_name,
                    strategy_source_type="ensemble_vote", instrument=instrument, timeframe="unknown",
                    backtest_period=period, backtest_result=bt_result, prop_rules=rules,
                    prop_single_run=single_run, monte_carlo_result=mc_result, holdout_comparison=None,
                    risk_config=risk, price_df=df,
                )
                self._last_ensemble_html_path = paths["html"]
                self.open_ensemble_report_btn.config(state="normal")
                self._log_ensemble(f"  Evaluation pass probability: {mc_result.evaluation_pass_probability:.1f}%")
                self._log_ensemble("\nDone. Ensemble (vote) report written to:")
                for k, p in paths.items():
                    self._log_ensemble(f"  {k}: {p}")
            else:
                self._log_ensemble(f"\nUnknown ensemble mode '{mode_label}'.")
        except EnsembleError as exc:
            self._log_ensemble(f"\nEnsemble error: {exc}")
        except StrategyError as exc:
            self._log_ensemble(f"\nStrategy error: {exc}")
        except Exception:
            self._log_ensemble("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.ensemble_progress.stop()

    # -----------------------------------------------------------------------
    # Generate Strategies (AI) -- draft new strategy code from a plain-
    # language idea via a local Ollama model, grounded in research/ papers
    # and your own best existing strategies. See
    # app.ai.strategy_generator for the full safety rationale: every
    # result lands in the library tagged DRAFT, never higher, and nothing
    # here ever runs generated code automatically.
    # -----------------------------------------------------------------------

    def _build_generate_strategies_tab(self):
        f = self._scrollable(self.tab_genstrat)
        self._page_header(
            f,
            "Finding an Edge / AI",
            "Generate Strategies",
            "Drafts a NEW strategy's source code from a plain-language idea, using a local "
            "Ollama model -- grounded in whatever papers you've dropped into the research/ "
            "folder and in your own best-performing saved strategies of the same language, "
            "so it matches this app's actual code style and constraints instead of guessing. "
            "Every strategy this produces is saved tagged DRAFT and has to earn its way "
            "through the normal backtest -> validation ladder like anything else -- nothing "
            "is ever run automatically.",
        )

        warn = Frame(f, bg="#3A2E0E", highlightthickness=1, highlightbackground=AMBER)
        warn.pack(fill="x", padx=24, pady=(0, 16))
        Label(
            warn, text="Read before using: a local model can write code that LOOKS reasonable "
            "but has a lookahead bug, an unsupported PineScript/MQL5 construct, or exit logic "
            "that silently never fires. Treat every generated strategy as a rough first draft, "
            "not a finished one -- run it through 05 Run & Report and ideally 15 Full Pipeline "
            "before trusting any of its numbers, exactly like a stranger's strategy you found "
            "online.",
            bg="#3A2E0E", fg=AMBER, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=16, pady=12)

        idea_section = self._section(
            f, "Strategy idea",
            "Describe entry/exit logic, indicators, market/timeframe, and anything else you "
            "want reflected -- the more specific, the better the draft. Research excerpts and "
            "your own prior strategies (if any exist yet for the chosen language) are added "
            "automatically as grounding.",
            emphasize=True,
        )
        self.genstrat_language = LabeledCombo(idea_section, "Language", list(STRATEGY_TYPES), "python")
        self.genstrat_idea = Text(
            idea_section, height=6, wrap="word", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        self.genstrat_idea.pack(fill="x", padx=18, pady=(4, 12))

        self._build_ai_assist_section(f, prefix="genstrat_ai")

        resource_section = self._section(
            f, "Resource usage (local Ollama)",
            "Keeps this one-off code-generation call lean on a local machine -- a smaller "
            "context window and output cap mean less RAM/VRAM used and a faster response, "
            "without materially limiting a strategy file (a few hundred lines fits "
            "comfortably within these defaults). The request also streams its response "
            "instead of waiting for one giant reply, so a slow model that's still actively "
            "producing tokens won't get cut off just for taking a while -- only a model that "
            "goes fully quiet mid-response, or runs past the hard time cap below, does. Raise "
            "these only if generation keeps visibly getting cut off mid-file.",
        )
        self.genstrat_num_ctx = LabeledEntry(resource_section, "Context window (tokens)", strategy_generator_module.DEFAULT_NUM_CTX)
        self.genstrat_num_predict = LabeledEntry(resource_section, "Max output length (tokens)", strategy_generator_module.DEFAULT_NUM_PREDICT)
        self.genstrat_stall_timeout = LabeledEntry(
            resource_section, "Stall timeout -- seconds of total silence before giving up", strategy_generator_module.DEFAULT_TIMEOUT_SECONDS,
        )
        self.genstrat_max_total = LabeledEntry(
            resource_section, "Hard time cap (seconds, even if still generating)", strategy_generator_module.DEFAULT_MAX_TOTAL_SECONDS,
        )

        btn_row = Frame(f, bg=BG)
        btn_row.pack(fill="x", padx=24, pady=(2, 10))
        self.genstrat_run_btn = self._button(btn_row, "GENERATE DRAFT", self._genstrat_generate_clicked, primary=True)
        self.genstrat_run_btn.pack(side="left")
        self.genstrat_progress = NeuralProgress(f)
        self.genstrat_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Generated code", "Nothing has been saved yet -- review it below first.")
        self.genstrat_filename = LabeledEntry(output_section, "Filename (no extension)", "")
        self.genstrat_output = Text(
            output_section, height=22, wrap="none", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        self.genstrat_output.pack(fill="both", expand=True, padx=18, pady=(3, 10))

        out_btn_row = Frame(output_section, bg=PANEL)
        out_btn_row.pack(anchor="w", padx=18, pady=(0, 6))
        self.genstrat_save_btn = self._button(
            out_btn_row, "SAVE TO LIBRARY AS DRAFT", self._genstrat_save_clicked, primary=True
        )
        self.genstrat_save_btn.config(state="disabled")
        self.genstrat_save_btn.pack(side="left")
        Label(
            output_section, text="Saved strategies land in the Strategy Library (02 Strategy) tagged "
            "DRAFT -- open that tab, LOAD SELECTED (or ADD TO BATCH QUEUE), and run it through "
            "05 Run & Report / 15 Full Pipeline like anything else before trusting it.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        self.genstrat_status = Label(
            f, text="", bg=BG, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left",
        )
        self.genstrat_status.pack(anchor="w", padx=26, pady=(0, 20))

        self._genstrat_last_result = None

    def _genstrat_generate_clicked(self):
        idea = self.genstrat_idea.get("1.0", END).strip()
        if not idea:
            messagebox.showinfo("Describe the idea first", "Type a strategy idea in the box above first.")
            return
        language = self.genstrat_language.get_str()
        settings = self._build_ollama_settings(prefix="genstrat_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.genstrat_output.delete("1.0", END)
        self.genstrat_filename.var.set("")
        self.genstrat_save_btn.config(state="disabled")
        self.genstrat_status.config(text="Generating... this can take a while on a local model.", fg=AMBER)
        self.genstrat_run_btn.config(state="disabled")
        self.genstrat_progress.start(10)
        num_ctx = self.genstrat_num_ctx.get_int(strategy_generator_module.DEFAULT_NUM_CTX)
        num_predict = self.genstrat_num_predict.get_int(strategy_generator_module.DEFAULT_NUM_PREDICT)
        stall_timeout = self.genstrat_stall_timeout.get_int(strategy_generator_module.DEFAULT_TIMEOUT_SECONDS)
        max_total = self.genstrat_max_total.get_int(strategy_generator_module.DEFAULT_MAX_TOTAL_SECONDS)
        threading.Thread(
            target=self._genstrat_run,
            args=(settings, language, idea, num_ctx, num_predict, stall_timeout, max_total),
            daemon=True,
        ).start()

    def _genstrat_progress(self, tokens: int, elapsed: float) -> None:
        def _update():
            self.genstrat_status.config(
                text=f"Generating... {tokens} token(s) received so far, {elapsed:.0f}s elapsed.", fg=AMBER,
            )
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _genstrat_run(self, settings, language, idea, num_ctx, num_predict, stall_timeout, max_total):
        from app.ai.strategy_generator import generate_strategy

        try:
            result = generate_strategy(
                settings, language, idea,
                timeout=stall_timeout, max_total_seconds=max_total,
                num_ctx=num_ctx, num_predict=num_predict,
                progress_cb=self._genstrat_progress,
            )
        except Exception as exc:
            result = None
            error = f"Unexpected error: {exc}"
        else:
            error = result.error

        def _finish():
            self.genstrat_progress.stop()
            self.genstrat_run_btn.config(state="normal")
            if result is None or result.code is None:
                self.genstrat_status.config(text=error or "Generation failed.", fg=RED)
                return
            self._genstrat_last_result = result
            self.genstrat_output.insert("1.0", result.code)
            self.genstrat_filename.var.set(result.filename_hint or "generated_strategy")
            self.genstrat_save_btn.config(state="normal")
            self.genstrat_status.config(
                text="Draft generated -- review the code below (it is NOT saved or tested yet), "
                "then SAVE TO LIBRARY AS DRAFT when you're ready.",
                fg=GREEN,
            )

        try:
            self.root.after(0, _finish)
        except Exception:
            pass

    def _genstrat_save_clicked(self):
        if self._genstrat_last_result is None or not self._genstrat_last_result.code:
            return
        language = self.genstrat_language.get_str()
        filename_stem = self.genstrat_filename.get_str().strip() or self._genstrat_last_result.filename_hint or "generated_strategy"
        filename_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", filename_stem).strip("_") or "generated_strategy"
        ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[language]
        filename = f"{filename_stem}{ext}"
        # Whatever's currently in the text box wins -- lets you hand-edit
        # the draft before saving without re-generating.
        code_text = self.genstrat_output.get("1.0", END).rstrip("\n")
        try:
            saved_path = self._save_to_library_with_overwrite_prompt(
                lambda overwrite: save_strategy_text(code_text, filename, language, overwrite=overwrite),
                fallback_path=Path(filename),
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        if saved_path is None:
            return
        # Always DRAFT, regardless of anything else -- see
        # app.ai.strategy_generator's module docstring for why an
        # AI-generated strategy is never allowed to start higher than this.
        try:
            set_strategy_status(language, saved_path.name, "draft")
            save_strategy_metadata(language, saved_path.name, {
                "description": f"AI-drafted from idea: {self.genstrat_idea.get('1.0', END).strip()[:200]}",
            })
        except Exception:
            pass
        self.genstrat_status.config(
            text=f"Saved as {saved_path.name}  [DRAFT]. Open 02 Strategy to load and test it.", fg=GREEN,
        )
        self._refresh_strategy_library()

    # -----------------------------------------------------------------------
    # Tab 15 — Full Pipeline (run everything, hand back the champion)
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Evolution Lab -- natural-selection-based strategy discovery.
    # See app.evolution.engine for the actual generate -> pre-filter ->
    # robustness/OOS/Monte Carlo/prop-sim -> CPCV/PBO -> stress -> cluster
    # -> keep top N -> mutate -> repeat loop this tab drives.
    # ------------------------------------------------------------------
    def _build_evolution_lab_tab(self):
        f = self.tab_evolution
        Label(
            f, text="EVOLUTION LAB", bg=BG, fg=TEXT, font=_safe_font(16, "bold"),
        ).pack(anchor="w", padx=24, pady=(20, 4))
        Label(
            f,
            text=(
                "Natural-selection-based strategy discovery: each generation generates a fresh "
                "population of strategies across every family (07 Search Lab's own generator), "
                "runs them through pre-filter -> robustness -> walk-forward -> Monte Carlo -> prop "
                "simulation -> real CPCV / PBO -> a stress test at higher execution costs -> a "
                "correlation-based cluster dedupe, keeps the top N by PROP FITNESS (not raw profit -- "
                "see app/evolution/prop_fitness.py), mutates them, and repeats. Click START and it runs "
                "in the background -- safe to leave running for hours while you work in other tabs. "
                "Every candidate evaluated (not just the winners) is logged to the knowledge graph, so "
                "later generations' journal entries can say what's historically worked before, not just "
                "what this generation found."
            ),
            bg=BG, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 10))

        cfg_section = self._section(
            f, "Run configuration",
            "Uses whatever market data is loaded in 01 Data and whatever's set on 03 Prop Rules / "
            "04 Risk at the moment you click START -- changing those tabs after starting has no "
            "effect on an already-running Evolution Lab run.",
            emphasize=True,
        )
        self.evo_population = LabeledEntry(cfg_section, "Population size per generation", "60")
        self.evo_elite_keep = LabeledEntry(cfg_section, "Elite keep (top N)", "10")
        self.evo_max_generations = LabeledEntry(cfg_section, "Max generations (blank = run until stopped)", "")
        self.evo_min_trades = LabeledEntry(cfg_section, "Min trades (pre-filter)", "20")
        self.evo_mc_sims = LabeledEntry(cfg_section, "Monte Carlo sims per candidate", "1000")
        self.evo_cpcv_top_n = LabeledEntry(cfg_section, "CPCV / PBO pool size (most expensive stage)", "8")
        self.evo_stress_mult = LabeledEntry(cfg_section, "Stress test cost multiplier", "2.0")

        families_frame = Frame(cfg_section, bg=PANEL)
        families_frame.pack(fill="x", padx=18, pady=(4, 4))
        Label(
            families_frame, text="Families to include (none selected = every family)",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(anchor="w")
        self.evo_families_listbox = Listbox(
            families_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.evo_families_listbox.pack(fill="x", pady=(2, 0))
        try:
            for fam in sorted(list_families().keys()):
                self.evo_families_listbox.insert(END, fam)
        except Exception:
            pass

        btn_row = Frame(cfg_section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(10, 6))
        self._button(btn_row, "START EVOLUTION LAB", self._start_evolution_lab, primary=True).pack(side="left")
        self._button(btn_row, "STOP", self._stop_evolution_lab).pack(side="left", padx=8)
        self._button(btn_row, "RESET (discard saved progress)", self._reset_evolution_lab).pack(side="left", padx=8)

        self.evo_status_label = Label(
            cfg_section, text="Not running.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(9),
        )
        self.evo_status_label.pack(anchor="w", padx=18, pady=(0, 2))
        Label(
            cfg_section,
            text=(
                "Progress (generation, leaderboard, journal) is saved to disk after every generation. "
                "STOP then START AGAIN resumes from exactly where it left off, as long as the same market "
                "data is still loaded -- click RESET first if you want a genuinely fresh run instead."
            ),
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        lb_section = self._section(
            f, "Leaderboard (all-time best seen)",
            "Refreshed after every generation. Sorted by PROP FITNESS, which already accounts for "
            "pass probability, payout probability, robustness, OOS consistency, drawdown, and the "
            "penalties described above -- not just net profit. Double-click (or select + VIEW "
            "DETAILS) to see the full stat breakdown and generated code/config for any leader, and "
            "PROMOTE it straight into the Strategy Library to run it through 15 Full Pipeline.",
        )
        lb_frame = Frame(lb_section, bg=PANEL)
        lb_frame.pack(fill="x", padx=18, pady=(2, 6))
        lb_frame.columnconfigure(0, weight=1)
        self.evo_leaderboard_listbox = Listbox(
            lb_frame, height=16, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        evo_lb_scrollbar = ttk.Scrollbar(
            lb_frame, orient="vertical", command=self.evo_leaderboard_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        self.evo_leaderboard_listbox.config(yscrollcommand=evo_lb_scrollbar.set)
        self.evo_leaderboard_listbox.grid(row=0, column=0, sticky="nsew")
        evo_lb_scrollbar.grid(row=0, column=1, sticky="ns")
        self.evo_leaderboard_listbox.bind("<Double-Button-1>", lambda e: self._view_evolution_leader_detail())

        lb_btn_row = Frame(lb_section, bg=PANEL)
        lb_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(lb_btn_row, "VIEW DETAILS", self._view_evolution_leader_detail, primary=True).pack(side="left")
        self._button(lb_btn_row, "PROMOTE TO STRATEGY LIBRARY", self._promote_selected_evolution_leader).pack(side="left", padx=8)
        self._button(lb_btn_row, "REFRESH FROM DISK", self._load_evolution_leaderboard_from_disk).pack(side="left", padx=8)

        self._evo_leaderboard_cache: list[dict] = []  # index-aligned with evo_leaderboard_listbox rows

        tested_section = self._section(
            f, "Tested strategies (most recent first)",
            "Every candidate the PRE-FILTER stage has actually backtested this run -- pass or fail, "
            "with the reason it was rejected if it was, and how far it got if it passed. Persists "
            "across STOP/START and survives closing the app; click REFRESH to pull the latest.",
        )
        tested_btn_row = Frame(tested_section, bg=PANEL)
        tested_btn_row.pack(anchor="w", padx=18, pady=(2, 4))
        self._button(tested_btn_row, "REFRESH", self._refresh_evolution_tested).pack(side="left")
        self.evo_tested_listbox = Listbox(
            tested_section, height=10, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.evo_tested_listbox.pack(fill="x", padx=18, pady=(2, 12))

        log_section = self._section(
            f, "Live log + hypothesis journal",
            "Each generation ends with a numbered HYPOTHESIS entry (test/result/OOS/CPCV/stress/"
            "winner/confidence) plus a knowledge-graph similarity readout for that generation's winner.",
        )
        log_frame = Frame(log_section, bg=PANEL)
        log_frame.pack(fill="both", expand=True, padx=18, pady=(2, 14))
        self.evo_log_text = Text(
            log_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 9), relief="flat", bd=0, height=18,
        )
        evo_log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.evo_log_text.yview, style="T58.Vertical.TScrollbar",
        )
        self.evo_log_text.config(yscrollcommand=evo_log_scrollbar.set)
        self.evo_log_text.pack(side="left", fill="both", expand=True)
        evo_log_scrollbar.pack(side="right", fill="y")

        self._evolution_runner = None
        try:
            self._load_evolution_leaderboard_from_disk()
            self._refresh_evolution_tested()
        except Exception:
            pass

    def _evo_log(self, msg: str) -> None:
        def _do():
            try:
                self.evo_log_text.insert(END, msg + "\n")
                self.evo_log_text.see(END)
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def _start_evolution_lab(self):
        if self._evolution_runner is not None and self._evolution_runner.is_running:
            messagebox.showinfo(
                "Already running",
                "Evolution Lab is already running. Click STOP first if you want to change settings "
                "and start a fresh run.",
            )
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data file in 01 Data before starting the Evolution Lab.")
            return

        from app.evolution.engine import EvolutionConfig, EvolutionRunner

        try:
            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    messagebox.showerror("Import error", f"{os.path.basename(p)}:\n" + "\n".join(result.errors))
                    return
                per_file_results.append(result)
            if len(per_file_results) == 1:
                df = per_file_results[0].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for r in per_file_results])
        except Exception as exc:
            messagebox.showerror("Could not load data", str(exc))
            return

        risk = self._build_risk_config()
        rules = self._build_prop_rules()
        selected_families = [self.evo_families_listbox.get(i) for i in self.evo_families_listbox.curselection()] or None
        max_gen_raw = self.evo_max_generations.get_str().strip()
        max_generations = int(max_gen_raw) if max_gen_raw.isdigit() else None

        cfg = EvolutionConfig(
            population_size=self.evo_population.get_int(60),
            elite_keep=self.evo_elite_keep.get_int(10),
            families=selected_families,
            min_trades=self.evo_min_trades.get_int(20),
            mc_sims=self.evo_mc_sims.get_int(1000),
            cpcv_top_n=self.evo_cpcv_top_n.get_int(8),
            stress_cost_multiplier=self.evo_stress_mult.get_float(2.0),
            max_generations=max_generations,
        )
        self.evo_log_text.delete("1.0", END)
        self._evolution_runner = EvolutionRunner(df, risk, rules, cfg, progress_cb=self._evo_log)
        self._evolution_runner.start()
        self._evo_log(
            f"Evolution Lab started -- population {cfg.population_size}, elite keep {cfg.elite_keep}, "
            f"{'unlimited generations' if cfg.max_generations is None else f'max {cfg.max_generations} generations'}."
        )
        self._poll_evolution_status()
        self._refresh_evolution_tested()

    def _stop_evolution_lab(self):
        if self._evolution_runner is None or not self._evolution_runner.is_running:
            messagebox.showinfo("Not running", "Evolution Lab isn't currently running.")
            return
        self._evolution_runner.stop()
        self._evo_log("Stop requested -- finishing the current generation, then stopping. Progress up to "
                       "that point is saved -- START again to resume.")

    def _reset_evolution_lab(self):
        if self._evolution_runner is not None and self._evolution_runner.is_running:
            messagebox.showwarning(
                "Still running", "Click STOP first, then RESET once it's finished the current generation."
            )
            return
        if not messagebox.askyesno(
            "Reset Evolution Lab",
            "This discards the saved generation/leaderboard/journal checkpoint and the tested-strategies "
            "log, so the next START begins a completely fresh run. This cannot be undone. Continue?",
        ):
            return
        if self._evolution_runner is None:
            evo_checkpoint.clear_checkpoint()
            evo_checkpoint.clear_tested_log()
        else:
            self._evolution_runner.reset()
        self.evo_log_text.delete("1.0", END)
        self.evo_leaderboard_listbox.delete(0, END)
        self.evo_tested_listbox.delete(0, END)
        self._evo_leaderboard_cache = []
        self.evo_status_label.config(text="Not running. (Saved progress cleared.)", fg=TEXT_DIM)
        self._evo_log("Evolution Lab progress reset -- the next START begins a fresh run.")

    def _refresh_evolution_tested(self):
        try:
            self.evo_tested_listbox.delete(0, END)
            if self._evolution_runner is not None:
                rows = self._evolution_runner.tested_candidates(limit=300)
            else:
                rows = evo_checkpoint.read_tested_rows(limit=300)
            for row in reversed(rows):
                gen = row.get("generation", "?")
                fam = str(row.get("family", "?"))[:16]
                stage = row.get("stage", "?")
                if row.get("passed"):
                    if stage == "full_eval":
                        score = row.get("fitness_score")
                        detail = f"PASSED prefilter, fitness {score:.2f}" if isinstance(score, (int, float)) else "PASSED prefilter"
                    else:
                        detail = "PASSED prefilter"
                else:
                    reasons = ", ".join(row.get("reasons") or []) or (row.get("error") or "unknown")
                    detail = f"rejected: {reasons}"
                trades = row.get("n_trades")
                trades_str = f"{trades} trades" if trades is not None else ""
                self.evo_tested_listbox.insert(
                    END, f"  gen {gen:>3}  {fam:16s}  {trades_str:12s}  {detail}"
                )
        except Exception:
            pass

    def _refresh_evo_leaderboard_listbox(self, records: list[dict]) -> None:
        """`records` is a list of checkpoint-shaped dicts (candidate_id,
        spec, meta, stats, mc_summary, fitness, ...) -- the same shape
        whether they came from a live runner (via to_checkpoint_dict())
        or straight off disk. Keeps `self._evo_leaderboard_cache`
        index-aligned with the listbox rows so VIEW DETAILS / PROMOTE
        can look up the right one from the current selection."""
        self.evo_leaderboard_listbox.delete(0, END)
        self._evo_leaderboard_cache = records
        for r in records:
            fitness = r.get("fitness") or {}
            score = fitness.get("final_score", float("nan"))
            fam = str((r.get("meta") or {}).get("family", "?"))[:18]
            stats = r.get("stats") or {}
            wr = stats.get("win_rate")
            wr_str = f"{wr:5.1f}% WR" if isinstance(wr, (int, float)) else "   n/a WR"
            trades = stats.get("total_trades")
            trades_str = f"{trades:4d} trades" if isinstance(trades, int) else "   ? trades"
            self.evo_leaderboard_listbox.insert(
                END,
                f"  {score:8.2f}   {fam:18s}  {trades_str}  {wr_str}  {r.get('candidate_id', '?')}",
            )

    def _load_evolution_leaderboard_from_disk(self) -> None:
        """Populates the leaderboard from the saved checkpoint even when
        no EvolutionRunner is currently alive in this process (e.g. the
        app was closed and reopened after an overnight run) -- Evolution
        Lab saves progress to disk after every generation specifically so
        this works."""
        if self._evolution_runner is not None:
            try:
                self._refresh_evo_leaderboard_listbox([r.to_checkpoint_dict() for r in self._evolution_runner.leaderboard])
                return
            except Exception:
                pass
        try:
            checkpoint = evo_checkpoint.load_checkpoint()
        except Exception:
            checkpoint = None
        if checkpoint is None:
            return
        try:
            self._refresh_evo_leaderboard_listbox(list(checkpoint.leaderboard))
        except Exception:
            pass

    def _selected_evolution_leader(self) -> dict | None:
        sel = self.evo_leaderboard_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select a strategy from the Leaderboard first.")
            return None
        idx = sel[0]
        if idx >= len(self._evo_leaderboard_cache):
            return None
        return self._evo_leaderboard_cache[idx]

    def _view_evolution_leader_detail(self) -> None:
        record = self._selected_evolution_leader()
        if record is None:
            return
        stats = record.get("stats") or {}
        mc = record.get("mc_summary") or {}
        fitness = record.get("fitness") or {}
        robustness = record.get("robustness") or {}
        walk_forward = record.get("walk_forward") or {}

        def pct(v):
            return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"

        def num(v, fmt="{:.2f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

        lines = [
            f"Candidate: {record.get('candidate_id', '?')}",
            f"Family: {(record.get('meta') or {}).get('family', '?')}",
            "",
            "-- Prop fitness --",
            f"  Final score:          {num(fitness.get('final_score'))}",
            f"  Pass probability:     {pct(fitness.get('pass_probability'))}",
            f"  Payout probability:   {pct(fitness.get('payout_probability'))}",
            f"  Robustness:           {num(fitness.get('robustness'))}",
            f"  OOS consistency:      {num(fitness.get('oos_consistency'))}",
            "",
            "-- Backtest stats --",
            f"  Total trades:         {stats.get('total_trades', 'n/a')}",
            f"  Win rate:             {(num(stats.get('win_rate'), '{:.1f}') + '%') if isinstance(stats.get('win_rate'), (int, float)) else 'n/a'}",
            f"  Net profit:           ${num(stats.get('net_profit'))}",
            f"  Profit factor:        {num(stats.get('profit_factor'))}",
            f"  Sharpe ratio:         {num(stats.get('sharpe_ratio'))}",
            f"  Sortino ratio:        {num(stats.get('sortino_ratio'))}",
            f"  Max drawdown:         {num(stats.get('max_drawdown_pct'), '{:.1f}')}%",
            f"  Expectancy:           {num(stats.get('expectancy'))}",
            "",
            "-- Monte Carlo / eval simulation --",
            f"  Evaluation pass probability:  {pct(mc.get('evaluation_pass_probability'))}",
            f"  First payout probability:     {pct(mc.get('first_payout_probability'))}",
            "",
            "-- Robustness / walk-forward --",
            f"  Parameter-neighborhood stable:  {robustness.get('is_stable', 'n/a')}  "
            f"(stability ratio {num(robustness.get('stability_ratio'))})",
            f"  Walk-forward stable:            {walk_forward.get('is_stable', 'n/a')}  "
            f"(WF efficiency {num(walk_forward.get('walk_forward_efficiency'))})",
        ]
        if record.get("pbo") is not None:
            lines.append(f"  PBO (probability of backtest overfitting): {num(record.get('pbo'))}")
        if record.get("cpcv_degradation") is not None:
            lines.append(f"  CPCV degradation: {num(record.get('cpcv_degradation'))}")

        win = Toplevel(self.root)
        win.title(f"Leaderboard detail -- {record.get('candidate_id', '?')}")
        win.configure(bg=BG)
        win.geometry("760x680")
        Label(
            win, text="LEADERBOARD DETAIL", bg=BG, fg=TEXT, font=_safe_font(13, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 6))

        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        txt = Text(
            text_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 10), relief="flat", bd=0,
        )
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        txt.config(yscrollcommand=vs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))
        config = (record.get("spec") or {}).get("config")

        def _view_code():
            if not config:
                messagebox.showinfo("No config", "This candidate has no saved manual-builder config to show.")
                return
            self._show_text_viewer(
                f"Config -- {record.get('candidate_id', '?')}", json.dumps(config, indent=2),
            )

        def _promote():
            self._promote_evolution_leader_record(record)

        self._button(btn_row, "VIEW CODE / CONFIG", _view_code, primary=True).pack(side="left")
        self._button(btn_row, "PROMOTE TO STRATEGY LIBRARY", _promote).pack(side="left", padx=8)
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left", padx=8)

    def _promote_selected_evolution_leader(self) -> None:
        record = self._selected_evolution_leader()
        if record is None:
            return
        self._promote_evolution_leader_record(record)

    def _promote_evolution_leader_record(self, record: dict) -> None:
        """Saves a specific leaderboard entry into the Strategy Library
        (manual-builder JSON, same format Evolution Lab's own
        save_to_library option writes) tagged 'promoted', independent of
        whether auto-save-to-library is enabled for the run -- this is
        the on-demand version so a leader found overnight can be pushed
        into the library and run through 15 Full Pipeline without
        needing to re-run the whole generation with that option on."""
        config = (record.get("spec") or {}).get("config")
        if not config:
            messagebox.showwarning(
                "Nothing to promote",
                "This candidate has no manual-builder config attached, so there's nothing to save "
                "to the Strategy Library.",
            )
            return
        family = (record.get("meta") or {}).get("family", "strategy")
        cid = record.get("candidate_id", "unknown")
        filename = f"evolab_promoted_{family}_{cid[-8:]}.json"
        text = json.dumps(config, indent=2)
        try:
            try:
                save_strategy_text(text, filename, "manual", overwrite=False)
            except StrategyAlreadyExists:
                if not messagebox.askyesno(
                    "Already promoted", f"'{filename}' is already in the Strategy Library. Overwrite it?",
                ):
                    return
                save_strategy_text(text, filename, "manual", overwrite=True)
            set_strategy_status("manual", filename, "promoted")
        except Exception as exc:
            messagebox.showerror("Could not promote", str(exc))
            return
        messagebox.showinfo(
            "Promoted",
            f"Saved to Strategy Library as '{filename}' (manual, status: promoted). "
            "Find it in 06 Strategy Library / 15 Full Pipeline's batch queue to run it through "
            "the full validation pipeline.",
        )
        try:
            self._refresh_strategy_library()
        except Exception:
            pass

    def _poll_evolution_status(self):
        runner = self._evolution_runner
        if runner is None:
            return
        status = runner.status()
        resumed_note = " (resumed)" if status.get("resumed") else ""
        self.evo_status_label.config(
            text=(
                f"{'RUNNING' if status['running'] else 'STOPPED'} -- generation {status['generation']}, "
                f"leaderboard size {status['leaderboard_size']}{resumed_note}"
            ),
            fg=GREEN if status["running"] else TEXT_DIM,
        )
        try:
            self._refresh_evo_leaderboard_listbox([r.to_checkpoint_dict() for r in runner.leaderboard])
        except Exception:
            pass
        if status["running"]:
            try:
                self.root.after(2000, self._poll_evolution_status)
                self.root.after(4000, self._refresh_evolution_tested)
            except Exception:
                pass

    def _build_full_pipeline_tab(self):
        f = self._scrollable(self.tab_fullpipeline)

        self._page_header(
            f,
            "15 / All-In-One",
            "Full Pipeline",
            "One button, the whole workflow: backtests the strategy as given, runs "
            "app.optimize.walkforward_ga to search for a configuration that generalizes "
            "(scored ONLY on out-of-sample fold data, never in-sample -- so it isn't just "
            "curve-fit harder), re-validates the winner with a fresh full-fidelity Monte "
            "Carlo, checks it holds up across several distinct historical stretches with "
            "no further re-tuning, and produces one final report with a plain READY / "
            "MARGINAL / NOT READY verdict. For Python/PineScript/MQL5 strategies, the "
            "winning source is also saved straight into the Strategy Library, auto-tagged "
            "TESTED / PASSED, VALIDATED, or TESTED / FAILED based on that verdict (or "
            "whatever fixed status you pick below), ready to use. Uses the strategy, data, "
            "prop rules, and risk settings already configured in Steps 01-04.",
        )

        settings = self._section(
            f, "Pipeline settings",
            "Sensible defaults for a single run -- raise population/generations for a more "
            "thorough (slower) search once you know a strategy is worth the time.",
            emphasize=True,
        )
        self.fp_window_mode = LabeledCombo(settings, "GA fold window mode", ["rolling", "anchored"], "rolling")
        self.fp_folds = LabeledEntry(settings, "Number of folds (GA + OOS check)", 4)
        self._fp_metric_labels = list(FITNESS_METRICS.values())
        self._fp_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.fp_metric = LabeledCombo(
            settings, "Fitness metric", self._fp_metric_labels, FITNESS_METRICS["composite_prop_score"],
        )
        self.fp_population = LabeledEntry(settings, "GA population size", 12)
        self.fp_generations = LabeledEntry(settings, "GA generations", 6)
        self.fp_search_mc_sims = LabeledEntry(settings, "Monte Carlo sims during search", 200)
        self.fp_final_mc_sims = LabeledEntry(settings, "Monte Carlo sims for final report", 10000)
        self.fp_holdout_frac = LabeledEntry(settings, "Final holdout fraction", 0.2)
        self.fp_seed = LabeledEntry(settings, "Random seed", 42)

        library_section = self._section(
            f, "Strategy Library",
            "Code strategies only (Manual Strategy Builder configs aren't files, so there's "
            "nothing to save) -- the winning source is written under a new filename, never "
            "overwriting the strategy you started from.",
        )
        self.fp_save_to_library = LabeledCheckbox(
            library_section, "Save the winning strategy to the Strategy Library when finished", True,
        )
        _AUTO_STATUS_LABEL = "Auto (based on READY / MARGINAL / NOT READY verdict)"
        self._fp_status_label_to_key = {_AUTO_STATUS_LABEL: None}
        self._fp_status_label_to_key.update({status_label(s): s for s in STRATEGY_STATUSES})
        self.fp_library_status = LabeledCombo(
            library_section, "Status to tag it with",
            [_AUTO_STATUS_LABEL] + [status_label(s) for s in STRATEGY_STATUSES],
            _AUTO_STATUS_LABEL,
        )

        self._build_ai_assist_section(f)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN FULL PIPELINE", self._fullpipeline_run_clicked, primary=True).pack(side="left")
        self.open_fullpipeline_report_btn = self._button(button_row, "OPEN REPORT", self._open_fullpipeline_report)
        self.open_fullpipeline_report_btn.config(state="disabled")
        self.open_fullpipeline_report_btn.pack(side="left", padx=8)

        self.fullpipeline_progress = NeuralProgress(f)
        self.fullpipeline_progress.pack(fill="x", padx=24, pady=(2, 10))

        verdict_section = self._section(f, "Verdict", "Filled in once a run completes.")
        self.fullpipeline_verdict_label = Label(
            verdict_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=820, anchor="w",
        )
        self.fullpipeline_verdict_label.pack(anchor="w", fill="x", padx=18, pady=(2, 10))

        output_section = self._section(f, "Full Pipeline output", "Live progress log.")
        self.fullpipeline_output = Text(
            output_section, height=20, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.fullpipeline_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._last_fullpipeline_html_path = None

    def _build_ai_assist_section(self, parent, prefix: str = "ai"):
        """Optional local-Ollama AI assistant (see app.ai.ollama_client):
        off by default and everywhere. When enabled, Step 2's search asks
        a local Ollama model for candidate parameter values once per
        generation, seeded into the population alongside the normal
        random/bred candidates -- the model only ever proposes numbers
        for the strategy's already-discovered tunable parameters, never
        code, and every suggestion still goes through the exact same
        backtest/prop-sim/Monte Carlo evaluation as any other candidate.
        Any failure to reach Ollama (not running, wrong host, model not
        pulled) degrades silently to the search running exactly as if
        this were disabled.

        `prefix` namespaces the widget attributes (e.g. "genstrat_ai" ->
        self.genstrat_ai_enabled) so more than one tab can each have its
        own independent copy of this section without colliding -- the
        Generate Strategies tab uses this for actual code generation
        (see app.ai.strategy_generator), a materially different, riskier
        use of the same underlying Ollama connection than the numeric-only
        parameter suggestions this section was originally written for."""
        section = self._section(
            parent, "AI Assist (optional, local Ollama)",
            "Off by default. When enabled, a local Ollama model suggests parameter values "
            "to try during Step 2's search, once per generation, alongside the normal "
            "random search -- every suggestion still has to pass the exact same "
            "backtest/prop-simulation/Monte Carlo checks as anything else. Requires "
            "Ollama installed and running locally (https://ollama.com) -- nothing is sent "
            "anywhere except to the host you configure below.",
        )
        saved = ollama_settings_module.load_settings()
        setattr(self, f"{prefix}_enabled", LabeledCheckbox(section, "Enable AI Assist for this run", saved.enabled))
        setattr(self, f"{prefix}_host", LabeledEntry(section, "Ollama host", saved.host))
        setattr(self, f"{prefix}_model", LabeledEntry(
            section, "Model (must already be pulled, e.g. `ollama pull llama3.1`)", saved.model,
        ))
        setattr(self, f"{prefix}_api_key", LabeledEntry(
            section, "API key (optional -- only for a remote/proxied Ollama)", saved.api_key, secret=True,
        ))

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(2, 8))
        test_btn = self._button(btn_row, "TEST CONNECTION", lambda: self._test_ollama_connection(prefix), primary=True)
        test_btn.pack(side="left")
        setattr(self, f"{prefix}_test_btn", test_btn)
        status = Label(section, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left")
        status.pack(anchor="w", padx=18, pady=(0, 12))
        setattr(self, f"{prefix}_status", status)

    def _build_ollama_settings(self, prefix: str = "ai") -> "OllamaSettings":
        settings = OllamaSettings(
            enabled=getattr(self, f"{prefix}_enabled").get(),
            host=getattr(self, f"{prefix}_host").get_str().strip() or ollama_settings_module.DEFAULT_HOST,
            model=getattr(self, f"{prefix}_model").get_str().strip() or ollama_settings_module.DEFAULT_MODEL,
            api_key=getattr(self, f"{prefix}_api_key").get_str().strip(),
        )
        ollama_settings_module.save_settings(settings)
        return settings

    def _test_ollama_connection(self, prefix: str = "ai"):
        from app.ai.ollama_client import OllamaClient

        settings = self._build_ollama_settings(prefix)
        test_btn = getattr(self, f"{prefix}_test_btn")
        status = getattr(self, f"{prefix}_status")
        test_btn.config(state="disabled")
        status.config(text="Testing connection...", fg=AMBER)

        def run():
            try:
                ok, message = OllamaClient(settings).test_connection()
            except Exception as exc:
                ok, message = False, f"Unexpected error: {exc}"

            def _finish():
                status.config(text=message, fg=GREEN if ok else RED)
                test_btn.config(state="normal")

            # Runs on a background thread -- Tkinter widgets are only safe
            # to touch from the main thread, so hand the update back to it.
            try:
                self.root.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _log_fullpipeline(self, msg: str):
        self.fullpipeline_output.insert(END, msg + "\n")
        self.fullpipeline_output.see(END)
        self.root.update_idletasks()

    def _open_fullpipeline_report(self):
        if self._last_fullpipeline_html_path:
            webbrowser.open(f"file://{self._last_fullpipeline_html_path.resolve()}")

    def _fullpipeline_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        self.fullpipeline_output.delete("1.0", END)
        self.fullpipeline_verdict_label.config(text="Running...", fg=TEXT_DIM)
        self.fullpipeline_progress.start(10)
        threading.Thread(target=self._fullpipeline_run_pipeline, daemon=True).start()

    def _fullpipeline_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_fullpipeline)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            rules = self._build_prop_rules()

            metric_key = self._fp_metric_label_to_key.get(self.fp_metric.get_str(), "composite_prop_score")
            cfg = FullPipelineConfig(
                n_folds=self.fp_folds.get_int(4),
                window_mode=self.fp_window_mode.get_str(),
                ga_population=self.fp_population.get_int(12),
                ga_generations=self.fp_generations.get_int(6),
                ga_search_mc_sims=self.fp_search_mc_sims.get_int(200),
                fitness_metric=metric_key,
                final_mc_sims=self.fp_final_mc_sims.get_int(10000),
                holdout_frac=self.fp_holdout_frac.get_float(0.2),
                oos_check_folds=self.fp_folds.get_int(4),
                random_seed=self.fp_seed.get_int(42),
                save_to_library=self.fp_save_to_library.var.get(),
                library_status=self._fp_status_label_to_key.get(self.fp_library_status.get_str()),
            )

            self._log_fullpipeline(f"Starting Full Pipeline for '{_strategy_display_name(strategy)}'...\n")
            instrument = (
                os.path.basename(self.csv_paths[0])
                if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )
            ollama_settings = self._build_ollama_settings()
            result = run_full_pipeline(
                df, strategy, risk, rules, OUTPUT_DIR / "full_pipeline", cfg,
                progress_cb=self._log_fullpipeline, instrument=instrument,
                ollama_settings=ollama_settings,
            )

            self._last_fullpipeline_html_path = result.report_paths["html"]
            self.open_fullpipeline_report_btn.config(state="normal")

            verdict_color = {"READY": GREEN, "MARGINAL": AMBER, "NOT READY": RED}.get(result.verdict, TEXT_DIM)
            self.fullpipeline_verdict_label.config(
                text=f"{result.verdict}\n" + "\n".join(f"  \u2022 {r}" for r in result.verdict_reasons),
                fg=verdict_color,
            )

            self._log_fullpipeline(
                f"\nBaseline -> Final:  "
                f"trades {len(result.baseline_bt.trades)} -> {len(result.final_bt.trades)}  |  "
                f"net ${result.baseline_bt.statistics.net_profit:,.2f} -> ${result.final_bt.statistics.net_profit:,.2f}  |  "
                f"eval pass {result.baseline_mc.evaluation_pass_probability:.1f}% -> {result.final_mc.evaluation_pass_probability:.1f}%  |  "
                f"payout {result.baseline_mc.first_payout_probability:.1f}% -> {result.final_mc.first_payout_probability:.1f}%"
            )
            if result.refinement_skip_reason:
                self._log_fullpipeline(f"\nOptimization was skipped: {result.refinement_skip_reason}")
            if result.saved_library_note:
                self._log_fullpipeline(f"\n{result.saved_library_note}")
            for w in result.warnings:
                self._log_fullpipeline(f"WARNING: {w}")

            try:
                self._refresh_dashboard()
            except Exception:
                pass

            self._log_fullpipeline(f"\nVerdict: {result.verdict}")
            self._log_fullpipeline("\nDone. Full Pipeline report written to:")
            for k, p in result.report_paths.items():
                self._log_fullpipeline(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_fullpipeline(f"\nStrategy error: {exc}")
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
        except RefinementError as exc:
            self._log_fullpipeline(f"\nFull Pipeline error: {exc}")
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
        except Exception:
            self._log_fullpipeline("\nUnexpected error:\n" + traceback.format_exc())
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
        finally:
            self.fullpipeline_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 18 — AI Research Agent (T58 AI Research Engine)
    #
    # The "research analyst" upgrade to AI Assist: instead of a single
    # request/response call (numeric parameter suggestions, or a one-shot
    # generated strategy file), this hands a local Ollama model a fixed
    # toolbox of READ-ONLY analysis actions -- backtest, prop-simulation,
    # Monte Carlo, walk-forward, regime analysis, parameter sensitivity,
    # cost stress, plus the research/ paper library and T58's own memory
    # of every past experiment -- and lets it reason across several steps
    # before answering. See app.ai.research_agent's module docstring for
    # the full safety rationale: every tool call runs this app's OWN
    # already-validated engine, so the model can propose a next step or a
    # diagnosis, but it can never invent a number or write/change code.
    # Uses the strategy, data, prop rules, and risk settings already
    # configured in Steps 01-04, exactly like 15 Full Pipeline does.
    # -----------------------------------------------------------------------

    def _build_research_agent_tab(self):
        f = self._scrollable(self.tab_researchagent)
        self._page_header(
            f,
            "18 / AI Research",
            "AI Research Agent",
            "Ask a local Ollama model to investigate the strategy configured in Steps 01-04. "
            "It can call run_backtest, run_prop_simulation, run_monte_carlo, run_walk_forward, "
            "run_regime_analysis, run_parameter_sensitivity, run_cost_stress, compare_strategies, "
            "search_research (your research/ paper library), and search_experiments (T58's own "
            "memory of every past strategy test) -- each one runs this app's real, already-"
            "validated engine, never a guess. The model can recommend a next step (e.g. a "
            "parameter worth testing), but cannot apply it itself -- take any recommendation "
            "into 09 Refinement / Quick Optimize / 15 Full Pipeline to actually test it.",
        )

        question_section = self._section(
            f, "Research question",
            "What do you want the agent to investigate? Be specific -- \"Is this strategy "
            "robust enough for a 50K prop account?\" gets a more useful investigation than "
            "\"is this good\".",
            emphasize=True,
        )
        self.ra_question = Text(
            question_section, height=4, wrap="word", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ra_question.insert(
            "1.0",
            "Is this strategy robust, or does its edge depend on a fragile parameter or a "
            "specific market regime? What's the most promising next thing to test?",
        )
        self.ra_question.pack(fill="x", padx=18, pady=(4, 12))
        self.ra_max_steps = LabeledEntry(question_section, "Max tool-calling steps", 6)

        self._build_ai_assist_section(f, prefix="ra_ai")

        memory_section = self._section(
            f, "Research Library + T58 Research Memory",
            "The research/ paper library (RAG) and the durable record of every strategy this "
            "app has tested. Embedding the library is optional (needs an Ollama embedding "
            "model, e.g. `ollama pull nomic-embed-text`) -- without it, search_research still "
            "works via plain keyword matching.",
        )
        btn_row = Frame(memory_section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(2, 6))
        self._button(btn_row, "EMBED RESEARCH LIBRARY", self._ra_embed_library_clicked, primary=True).pack(side="left")
        self._button(btn_row, "REFRESH MEMORY SUMMARY", self._ra_refresh_memory_clicked).pack(side="left", padx=8)
        self.ra_memory_status = Label(
            memory_section, text="Click REFRESH MEMORY SUMMARY to see how many experiments T58 has recorded.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=900, justify="left",
        )
        self.ra_memory_status.pack(anchor="w", padx=18, pady=(0, 12))

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self.ra_run_btn = self._button(button_row, "RUN RESEARCH AGENT", self._ra_run_clicked, primary=True)
        self.ra_run_btn.pack(side="left")

        self.ra_progress = NeuralProgress(f)
        self.ra_progress.pack(fill="x", padx=24, pady=(2, 10))

        answer_section = self._section(f, "Final Answer", "Filled in once the agent finishes.")
        self.ra_answer_label = Label(
            answer_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.ra_answer_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Agent transcript", "Thought / Action / Observation, one step at a time.")
        self.ra_output = Text(
            output_section, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ra_output.pack(fill="both", expand=True, padx=18, pady=(3, 16))

    def _log_research_agent(self, msg: str):
        self.ra_output.insert(END, msg + "\n")
        self.ra_output.see(END)
        self.root.update_idletasks()

    def _ra_embed_library_clicked(self):
        settings = self._build_ollama_settings(prefix="ra_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.ra_memory_status.config(text="Embedding research library...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                stats = research_library_module.embed_index(settings)
                if stats.error:
                    text = f"Embedded {stats.chunks_embedded} new chunk(s); stopped early: {stats.error}"
                    color = AMBER
                else:
                    text = (
                        f"Research library ready: {stats.chunks_embedded} newly embedded, "
                        f"{stats.chunks_already_current} already current, {stats.total_chunks} total chunks."
                    )
                    color = GREEN
            except Exception as exc:
                text, color = f"Embedding failed: {exc}", RED

            def _finish():
                self.ra_memory_status.config(text=text, fg=color)

            try:
                self.root.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _ra_refresh_memory_clicked(self):
        try:
            counts = experiment_memory_module.get_summary_counts()
        except Exception as exc:
            self.ra_memory_status.config(text=f"Could not read Research Memory: {exc}", fg=RED)
            return
        if counts["total"] == 0:
            text = "No experiments recorded yet -- run 15 Full Pipeline, Quick Optimize, or a Batch Test first."
        else:
            breakdown = ", ".join(f"{v}: {n}" for v, n in counts["by_verdict"].items())
            text = f"T58 Research Memory: {counts['total']} experiments recorded. By verdict -- {breakdown}."
        self.ra_memory_status.config(text=text, fg=TEXT_DIM)

    def _ra_run_clicked(self):
        question = self.ra_question.get("1.0", END).strip()
        if not question:
            messagebox.showinfo("Ask a question first", "Type a research question in the box above.")
            return
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 1.")
            return
        settings = self._build_ollama_settings(prefix="ra_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.ra_output.delete("1.0", END)
        self.ra_answer_label.config(text="Running...", fg=TEXT_DIM)
        self.ra_run_btn.config(state="disabled")
        self.ra_progress.start(10)
        threading.Thread(target=self._ra_run_agent, args=(question, settings), daemon=True).start()

    def _ra_run_agent(self, question: str, settings: "OllamaSettings"):
        try:
            df = self._load_df_for_page(self._log_research_agent)
            if df is None:
                return
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            max_steps = self.ra_max_steps.get_int(6)
            instrument = (
                os.path.basename(self.csv_paths[0])
                if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )

            # Zero-arg builder consistent with every other tab's "always
            # build fresh" convention (see app.search.robustness /
            # app.validation.regime_testing) -- rereads current Step 02
            # strategy config on every call, exactly like Full Pipeline's
            # own strategy_builder does.
            strategy_snapshot = self._build_strategy()
            ctx = ResearchAgentContext(
                df=df, strategy_builder=self._build_strategy,
                strategy_name=_strategy_display_name(strategy_snapshot),
                source_type=strategy_snapshot.source_type,
                risk=risk, prop_rules=rules, instrument=instrument,
            )

            self._log_research_agent(f"Investigating '{ctx.strategy_name}' on {instrument}...\n")
            agent = ResearchAgent(settings, max_steps=max_steps)
            result = agent.run(question, ctx, progress_cb=self._log_research_agent)

            for step in result.steps:
                self._log_research_agent(f"\n--- Step {step.step_index} ---")
                if step.thought:
                    self._log_research_agent(f"Thought: {step.thought}")
                if step.action:
                    self._log_research_agent(f"Action: {step.action}({json.dumps(step.action_input or {})})")
                if step.observation is not None:
                    self._log_research_agent(f"Observation: {json.dumps(step.observation, indent=2)[:2000]}")
                if step.note:
                    self._log_research_agent(f"Note: {step.note}")

            if result.final_answer:
                self.ra_answer_label.config(text=result.final_answer, fg=GREEN)
                self._log_research_agent(f"\nFinal Answer: {result.final_answer}")
            elif result.error:
                self.ra_answer_label.config(text=f"Stopped: {result.error}", fg=RED)
                self._log_research_agent(f"\nStopped: {result.error}")
            else:
                self.ra_answer_label.config(text=f"Stopped without a final answer: {result.stopped_reason}", fg=AMBER)
                self._log_research_agent(f"\nStopped without a final answer: {result.stopped_reason}")

            # Best-effort: record this investigation in T58 Research Memory too,
            # using whatever the agent's own run_backtest/run_monte_carlo tool
            # calls already computed (never a fresh computation just for this).
            try:
                bt = ctx.cache_get("__baseline_bt__")
                if bt is not None and bt.trades:
                    experiment_memory_module.record_experiment(
                        origin="research_agent", strategy_name=ctx.strategy_name,
                        source_type=ctx.source_type, instrument=instrument,
                        verdict="INVESTIGATED", trades=len(bt.trades),
                        net_profit=bt.statistics.net_profit, win_rate=bt.statistics.win_rate,
                        profit_factor=bt.statistics.profit_factor,
                        max_drawdown_pct=bt.statistics.max_drawdown_pct,
                        lesson=(result.final_answer or "")[:500], settings=settings,
                    )
            except Exception:
                pass
        except StrategyError as exc:
            self._log_research_agent(f"\nStrategy error: {exc}")
            self.ra_answer_label.config(text="Failed -- see log.", fg=RED)
        except Exception:
            self._log_research_agent("\nUnexpected error:\n" + traceback.format_exc())
            self.ra_answer_label.config(text="Failed -- see log.", fg=RED)
        finally:
            self.ra_progress.stop()
            self.ra_run_btn.config(state="normal")

    # -----------------------------------------------------------------------
    # Tab 16 — Live Demo Test (MT5 Demo)
    # -----------------------------------------------------------------------

    def _build_forward_test_tab(self):
        f = self._scrollable(self.tab_forwardtest)
        self._ft_session: ForwardTestSession | None = None
        self._ft_journal = None

        self._page_header(
            f,
            "16 / Going Live",
            "Live Demo Test (MT5 Demo)",
            "Deploy any Strategy Library strategy to a free MetaTrader 5 demo account and "
            "watch it trade forward, bar by bar, against real broker prices instead of a "
            "CSV. Uses the exact same signal engine and position-sizing math as the "
            "backtester -- no separate re-implementation to drift out of sync. This is a "
            "demo account only; there is no live/funded order path here.",
        )

        if not mt5_connector_module.is_available():
            notice = self._section(f, "MetaTrader 5 not detected", "", emphasize=True)
            Label(
                notice, text=mt5_connector_module.unavailable_reason(), bg=PANEL, fg=AMBER,
                font=_safe_font(9), wraplength=820, justify="left",
            ).pack(anchor="w", padx=18, pady=(2, 6))
            Label(
                notice,
                text="To use this tab: run the app on Windows with an MT5 terminal installed "
                     "and logged into a demo account (any MT5 broker's site offers a free demo "
                     "account signup), and make sure `pip install MetaTrader5` succeeded. The "
                     "rest of this tab still works for entering settings -- Start Live Demo Test "
                     "will just fail with a clear message until MT5 is reachable.",
                bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 12))

        strat_section = self._section(
            f, "Strategy",
            "Pulled from your Strategy Library (Step 02) -- Python, PineScript, and MQL5 "
            "strategies all work here since they share the same signal interface.",
            emphasize=True,
        )
        picker_row = Frame(strat_section, bg=PANEL)
        picker_row.pack(fill="x", padx=18, pady=(2, 4))
        Label(
            picker_row, text="Strategy", width=31, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left")
        self.ft_strategy_var = StringVar(value="")
        self.ft_strategy_combo = ttk.Combobox(
            picker_row, textvariable=self.ft_strategy_var, values=[], state="readonly",
            width=52, font=_safe_font(9), style="T58.TCombobox",
        )
        self.ft_strategy_combo.pack(side="left", padx=(4, 8))
        self._button(picker_row, "REFRESH", self._ft_refresh_strategies).pack(side="left")
        self._ft_strategy_items: list = []
        self._ft_refresh_strategies()

        market_section = self._section(
            f, "Market",
            "Symbol must match this exact string on your MT5 account/broker (e.g. 'XAUUSD', "
            "'EURUSD', 'US30' -- some brokers append a suffix like 'XAUUSD.m', check your "
            "MT5 Market Watch if the exact symbol isn't found).",
        )
        self.ft_symbol = LabeledEntry(market_section, "Symbol", mt5_settings_module.DEFAULT_SYMBOL)
        self.ft_timeframe = LabeledCombo(
            market_section, "Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day"],
            "15 minutes",
        )

        account_section = self._section(
            f, "MT5 Demo Account",
            "Saved locally (password via your OS keyring where available). Get a free demo "
            "login from any MT5-supporting broker's website, or your prop firm's own demo/"
            "trial account if they offer one.",
        )
        saved_mt5 = mt5_settings_module.load_settings()
        self.ft_login = LabeledEntry(account_section, "Login (account number)", saved_mt5.login)
        self.ft_server = LabeledEntry(account_section, "Server", saved_mt5.server)
        self.ft_password = LabeledEntry(account_section, "Password", saved_mt5.password, secret=True)
        self.ft_terminal_path = LabeledEntry(
            account_section, "Terminal path (optional, only if auto-detect fails)", saved_mt5.terminal_path,
        )
        terminal_btn_row = Frame(account_section, bg=PANEL)
        terminal_btn_row.pack(anchor="w", padx=18, pady=(0, 8))
        Label(
            terminal_btn_row, text="", width=31, bg=PANEL,
        ).pack(side="left")  # spacer matching LabeledEntry's label column so buttons line up under the field
        self._button(terminal_btn_row, "AUTO-DETECT", self._ft_auto_detect_terminal).pack(side="left")
        self._button(terminal_btn_row, "BROWSE...", self._ft_browse_terminal).pack(side="left", padx=(6, 0))
        conn_row = Frame(account_section, bg=PANEL)
        conn_row.pack(anchor="w", padx=18, pady=(2, 10))
        self.ft_test_conn_btn = self._button(conn_row, "SAVE & TEST CONNECTION", self._ft_test_connection, primary=True)
        self.ft_test_conn_btn.pack(side="left")
        self.ft_conn_status = Label(conn_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=760, justify="left")
        self.ft_conn_status.pack(side="left", padx=(12, 0))

        risk_section = self._section(
            f, "Risk (same fields as Step 04, applied to this live demo account)",
            "Position sizing calls the exact same RiskConfig.position_size(...) the "
            "backtester uses -- if a strategy defines its own dynamic stop, that's honored "
            "first; otherwise a fixed-pip or 1%-of-price fallback applies, same precedence "
            "as a backtest run.",
        )
        self.ft_risk_pct = LabeledEntry(risk_section, "Risk per trade (% of equity)", 1.0)
        self.ft_daily_loss_limit = LabeledEntry(risk_section, "Daily loss limit (% of balance, halts new entries)", 5.0)
        self.ft_max_trades_per_day = LabeledEntry(risk_section, "Max trades per day", 10)
        self.ft_pip_size = LabeledEntry(risk_section, "Pip size (leave blank to auto-detect from MT5 symbol info)", "")
        self.ft_baseline_win_rate = LabeledEntry(
            risk_section, "Backtest win rate %, for drift comparison (optional)", "",
        )

        control_section = self._section(f, "Session control", "")
        btn_row = Frame(control_section, bg=PANEL)
        btn_row.pack(fill="x", padx=18, pady=(2, 4))
        self.ft_start_btn = self._button(btn_row, "START LIVE DEMO TEST", self._ft_start_clicked, primary=True)
        self.ft_start_btn.pack(side="left")
        self.ft_stop_btn = self._button(btn_row, "STOP", self._ft_stop_clicked)
        self.ft_stop_btn.pack(side="left", padx=8)
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn = Button(
            btn_row, text="KILL SWITCH — FLATTEN & STOP", command=self._ft_kill_clicked,
            bg=RED, fg="#1a0a0d", activebackground="#c94656", activeforeground="#1a0a0d",
            relief="flat", bd=0, font=_safe_font(9, "bold"), padx=14, pady=8, cursor="hand2",
        )
        self.ft_kill_btn.pack(side="left", padx=8)
        self.ft_kill_btn.config(state="disabled")

        self.ft_progress = NeuralProgress(control_section)
        self.ft_progress.pack(fill="x", padx=18, pady=(6, 10))

        status_row = Frame(control_section, bg=PANEL)
        status_row.pack(fill="x", padx=18, pady=(0, 12))
        self.ft_status_label = Label(
            status_row, text="Not running.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(10, "bold"), justify="left", wraplength=820,
        )
        self.ft_status_label.pack(anchor="w")

        journal_section = self._section(f, "Trade journal (this session)", "")
        columns = ("time", "direction", "volume", "entry", "exit", "pnl", "status")
        self.ft_journal_tree = ttk.Treeview(journal_section, columns=columns, show="headings", height=8)
        for col, label, w in (
            ("time", "Entry Time", 140), ("direction", "Dir", 50), ("volume", "Size", 70),
            ("entry", "Entry", 90), ("exit", "Exit", 90), ("pnl", "P&L", 80), ("status", "Status", 70),
        ):
            self.ft_journal_tree.heading(col, text=label)
            self.ft_journal_tree.column(col, width=w, anchor="center")
        self.ft_journal_tree.pack(fill="x", padx=18, pady=(2, 12))

        log_section = self._section(f, "Live log", "")
        self.ft_log = Text(
            log_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ft_log.pack(fill="both", expand=True, padx=18, pady=(3, 16))

    def _ft_refresh_strategies(self):
        items = list_saved_strategies(None)
        self._ft_strategy_items = items
        labels = [f"[{it.strategy_type}] {it.name}" for it in items]
        self.ft_strategy_combo["values"] = labels
        if labels and not self.ft_strategy_var.get():
            self.ft_strategy_var.set(labels[0])

    def _ft_selected_strategy_item(self):
        label = self.ft_strategy_var.get()
        for it in self._ft_strategy_items:
            if f"[{it.strategy_type}] {it.name}" == label:
                return it
        return None

    def _ft_build_strategy_instance(self, item):
        if item.strategy_type == "python":
            return PythonStrategy(item.path)
        text = item.path.read_text(encoding="utf-8")
        if item.strategy_type == "pinescript":
            return PineScriptStrategy(text)
        if item.strategy_type == "mql5":
            return MQL5Strategy(text)
        raise StrategyError(f"Unsupported strategy type for live demo test: {item.strategy_type}")

    def _ft_timeframe_minutes(self) -> int:
        mapping = {
            "1 minute": 1, "5 minutes": 5, "15 minutes": 15, "30 minutes": 30,
            "1 hour": 60, "4 hours": 240, "1 day": 1440,
        }
        return mapping.get(self.ft_timeframe.get_str(), 15)

    def _ft_log_line(self, level: str, message: str):
        color = {"error": RED, "warn": AMBER, "info": TEXT}.get(level, TEXT)
        tag = f"lvl_{level}"
        self.ft_log.tag_config(tag, foreground=color)
        self.ft_log.insert(END, f"[{level.upper()}] {message}\n", tag)
        self.ft_log.see(END)
        try:
            self.root.after(0, self.root.update_idletasks)
        except Exception:
            pass

    def _ft_browse_terminal(self):
        path = filedialog.askopenfilename(
            title="Locate your MT5 terminal64.exe",
            filetypes=[("MT5 terminal", "terminal64.exe"), ("All files", "*.*")],
        )
        if path:
            self.ft_terminal_path.var.set(path)
            self.ft_conn_status.config(
                text=f"Terminal path set to {path}. Click Save & Test Connection to verify it.", fg=TEXT_MUTED,
            )

    def _ft_auto_detect_terminal(self):
        if not mt5_connector_module.is_available():
            self.ft_conn_status.config(text=mt5_connector_module.unavailable_reason(), fg=AMBER)
            return
        self.ft_conn_status.config(text="Searching common install locations for terminal64.exe...", fg=AMBER)
        self.root.update_idletasks()
        candidates = mt5_connector_module.find_terminal_candidates()
        if not candidates:
            self.ft_conn_status.config(
                text="Couldn't find terminal64.exe in any common install location (Program Files, "
                     "AppData\\MetaQuotes). If MT5 is installed somewhere else, use Browse... to "
                     "point at it directly -- or install it fresh from your broker's site if it "
                     "isn't installed at all.",
                fg=AMBER,
            )
            return
        self.ft_terminal_path.var.set(candidates[0])
        extra = f" ({len(candidates) - 1} other install(s) also found -- this is the most recently used.)" if len(candidates) > 1 else ""
        self.ft_conn_status.config(
            text=f"Found: {candidates[0]}{extra} Click Save & Test Connection to verify it.", fg=GREEN,
        )

    def _ft_save_mt5_settings(self) -> MT5Settings:
        settings = MT5Settings(
            login=self.ft_login.get_str().strip(),
            server=self.ft_server.get_str().strip(),
            password=self.ft_password.get_str(),
            symbol=self.ft_symbol.get_str().strip() or mt5_settings_module.DEFAULT_SYMBOL,
            timeframe_minutes=self._ft_timeframe_minutes(),
            terminal_path=self.ft_terminal_path.get_str().strip(),
        )
        mt5_settings_module.save_settings(settings)
        return settings

    def _ft_test_connection(self):
        settings = self._ft_save_mt5_settings()
        if not mt5_connector_module.is_available():
            self.ft_conn_status.config(text=mt5_connector_module.unavailable_reason(), fg=AMBER)
            return
        self.ft_test_conn_btn.config(state="disabled")
        self.ft_conn_status.config(text="Connecting...", fg=AMBER)

        def run():
            from app.forward_test.mt5_connector import MT5Connector
            connector = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
            result = connector.connect()
            if result.ok:
                msg = (f"Connected: account {result.account_login} @ {result.account_server} — "
                       f"balance {result.balance:,.2f} {result.currency}, equity {result.equity:,.2f}.")
                # Auto-detection may have found a working terminal path this
                # session even though the field was left blank -- persist
                # it now so future connections (and the Forward Test
                # session itself) don't have to rediscover it every time.
                if result.resolved_terminal_path and not settings.terminal_path:
                    settings.terminal_path = result.resolved_terminal_path
                    mt5_settings_module.save_settings(settings)
                    self.root.after(0, lambda: self.ft_terminal_path.var.set(result.resolved_terminal_path))
                    msg += f" (auto-detected and saved terminal path: {result.resolved_terminal_path})"
                connector.disconnect()
            else:
                msg = result.message
            self.root.after(0, lambda: self.ft_conn_status.config(text=msg, fg=GREEN if result.ok else RED))
            self.root.after(0, lambda: self.ft_test_conn_btn.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _ft_start_clicked(self):
        item = self._ft_selected_strategy_item()
        if item is None:
            messagebox.showwarning("No strategy selected", "Choose a strategy from the Strategy Library first.")
            return
        if not mt5_connector_module.is_available():
            messagebox.showwarning("MT5 not available", mt5_connector_module.unavailable_reason())
            return

        settings = self._ft_save_mt5_settings()
        if not settings.is_usable:
            messagebox.showwarning("Missing credentials", "Enter your MT5 demo login, server, and password.")
            return

        try:
            strategy = self._ft_build_strategy_instance(item)
        except Exception as exc:
            messagebox.showerror("Strategy error", f"Could not load strategy: {exc}")
            return

        from app.forward_test.mt5_connector import MT5Connector

        connector = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
        pip_size_str = self.ft_pip_size.get_str().strip()
        pip_size = None
        if pip_size_str:
            try:
                pip_size = float(pip_size_str)
            except ValueError:
                pip_size = None

        def resolve_pip_and_start():
            nonlocal pip_size
            probe = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
            conn = probe.connect()
            if not conn.ok:
                self.root.after(0, lambda: self._ft_log_line("error", conn.message))
                return
            if pip_size is None:
                try:
                    pip_size = probe.symbol_point(settings.symbol)
                except Exception as exc:
                    self.root.after(0, lambda: self._ft_log_line(
                        "warn", f"Could not auto-detect pip size ({exc}); falling back to 0.0001."))
                    pip_size = 0.0001
            probe.disconnect()

            risk = RiskConfig(
                initial_balance=conn.balance or 10_000.0,
                risk_mode="percent",
                risk_value=self.ft_risk_pct.get_float(1.0),
                max_trades_per_day=self.ft_max_trades_per_day.get_int(10),
                pip_size=pip_size,
                daily_loss_limit_pct=self.ft_daily_loss_limit.get_float(5.0) or None,
            )
            baseline_str = self.ft_baseline_win_rate.get_str().strip()
            baseline_win_rate = float(baseline_str) if baseline_str else None

            cfg = ForwardTestConfig(
                symbol=settings.symbol, timeframe_minutes=settings.timeframe_minutes,
                risk=risk, baseline_win_rate=baseline_win_rate,
            )
            self._ft_journal = ForwardTestJournal()
            session = ForwardTestSession(
                strategy=strategy, strategy_type=item.strategy_type, strategy_filename=item.name,
                connector=connector, journal=self._ft_journal, config=cfg,
                on_log=lambda level, msg: self.root.after(0, lambda: self._ft_log_line(level, msg)),
                on_status=lambda status: self.root.after(0, lambda: self._ft_update_status(status)),
            )
            ok, msg = session.start()
            if not ok:
                self.root.after(0, lambda: self._ft_log_line("error", msg))
                return
            self._ft_session = session
            self.root.after(0, self._ft_on_started)

        threading.Thread(target=resolve_pip_and_start, daemon=True).start()
        self.ft_start_btn.config(state="disabled")
        self.ft_status_label.config(text="Connecting...", fg=AMBER)

    def _ft_on_started(self):
        self.ft_stop_btn.config(state="normal")
        self.ft_kill_btn.config(state="normal")
        self.ft_progress.start()
        self.ft_status_label.config(text="Running.", fg=GREEN)

    def _ft_update_status(self, status):
        parts = [f"Running: {status.running}", f"Last signal: {status.last_signal}"]
        if status.balance is not None:
            parts.append(f"Balance ${status.balance:,.2f}")
        if status.equity is not None:
            parts.append(f"Equity ${status.equity:,.2f}")
        if status.n_trades_closed:
            parts.append(f"Closed trades: {status.n_trades_closed}")
        if status.win_rate is not None:
            parts.append(f"Win rate {status.win_rate:.1f}%")
        if status.halted_reason:
            parts.append(f"HALTED: {status.halted_reason}")
        color = RED if status.halted_reason else (GREEN if status.running else TEXT_DIM)
        if status.drift_flag:
            parts.append(f"⚠ {status.drift_flag}")
            color = AMBER
        self.ft_status_label.config(text="  |  ".join(parts), fg=color)
        self._ft_refresh_journal_view()

    def _ft_refresh_journal_view(self):
        if self._ft_journal is None or self._ft_session is None or self._ft_session._session_id is None:
            return
        for row in self.ft_journal_tree.get_children():
            self.ft_journal_tree.delete(row)
        trades = self._ft_journal.all_trades(self._ft_session._session_id)
        for t in trades[:100]:
            import datetime
            entry_time = datetime.datetime.fromtimestamp(t.entry_time).strftime("%m-%d %H:%M")
            direction = "LONG" if t.direction == 1 else "SHORT"
            exit_price = f"{t.exit_price:.5f}" if t.exit_price else ""
            pnl = f"{t.pnl:,.2f}" if t.pnl is not None else ""
            self.ft_journal_tree.insert("", END, values=(
                entry_time, direction, f"{t.volume:.2f}", f"{t.entry_price:.5f}", exit_price, pnl, t.status,
            ))

    def _ft_stop_clicked(self):
        if self._ft_session is None:
            return
        self.ft_status_label.config(text="Stopping...", fg=AMBER)
        threading.Thread(target=self._ft_session.stop, daemon=True).start()
        self.ft_start_btn.config(state="normal")
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn.config(state="disabled")
        self.ft_progress.stop()

    def _ft_kill_clicked(self):
        if self._ft_session is None:
            return
        if not messagebox.askyesno(
            "Kill switch",
            "This closes every open position on this symbol immediately and stops the "
            "session. Continue?",
        ):
            return
        self.ft_status_label.config(text="Flattening and stopping...", fg=RED)
        threading.Thread(target=self._ft_session.flatten_all_and_stop, daemon=True).start()
        self.ft_start_btn.config(state="normal")
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn.config(state="disabled")
        self.ft_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 17 — Live Market
    # -----------------------------------------------------------------------

    def _build_live_market_tab(self):
        f = self._scrollable(self.tab_livemarket)
        self._page_header(
            f,
            "17 / Live Market",
            "Live Market",
            "Two ways to see a live-style chart for a symbol. OPEN TRADINGVIEW CHART (recommended) "
            "just opens the real tradingview.com chart for the symbol below in your browser -- no "
            "local server, no MT5/Alpaca connection needed, works the same on Windows/Mac/Linux. "
            "OPEN LIVE CHART is this app's own built-in chart (candlesticks, volume, EMA 20/50, "
            "VWAP, trade markers from your Live Demo Test sessions) fed from MT5/Alpaca/a local "
            "CSV replay -- it starts a small local server on your own machine and opens the chart "
            "in its own window (or your default browser if a native chart window isn't available).",
        )

        try:
            saved_mt5 = mt5_settings_module.load_settings()
        except Exception:
            saved_mt5 = None

        tv_section = self._section(
            f, "TradingView chart (recommended)",
            "Opens the real TradingView website for this symbol in your default browser. "
            "Simplest and most reliable option -- no local server, no broker connection required, "
            "and it always shows genuinely live data straight from TradingView.",
            emphasize=True,
        )
        self.lm_tv_symbol = LabeledEntry(
            tv_section, "Symbol", saved_mt5.symbol if (saved_mt5 is not None and saved_mt5.is_usable) else "XAUUSD",
        )
        tv_btn_row = Frame(tv_section, bg=PANEL)
        tv_btn_row.pack(anchor="w", padx=18, pady=(0, 4))
        self._button(tv_btn_row, "OPEN TRADINGVIEW CHART", self._lm_open_tradingview, primary=True).pack(side="left")
        Label(
            tv_section, text="Common FX/metals/index/crypto tickers (XAUUSD, EURUSD, US30, "
            "BTCUSD, ...) are mapped to a sensible TradingView exchange automatically -- for "
            "anything else, type whatever symbol/ticker TradingView itself would recognize.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        section = self._section(
            f, "Built-in chart data source (advanced)",
            "Live (MT5) is used automatically when your Live Demo Test MT5 account is "
            "configured; Alpaca is used if you've saved API keys on the Data tab; with neither, "
            "the chart replays an already-imported local CSV bar by bar so there's always "
            "something real on screen. Requires this app's own local server; if OPEN LIVE CHART "
            "ever errors or the native window doesn't appear (most common on Linux without the "
            "optional GTK/QT webview packages installed), use OPEN TRADINGVIEW CHART above instead.",
        )

        # Pick a sane default source instead of always defaulting to "mt5"
        # -- this mirrors the exact fallback the Flask page itself uses
        # (see live_market_page() in app/web/server.py), which used to
        # only ever kick in when NO source was passed in the URL at all.
        # Since OPEN LIVE CHART always sent an explicit source= param, an
        # un-configured MT5 (the common case -- MT5 needs Windows + a
        # running terminal) meant every click silently requested MT5 data,
        # which fails with an empty bar list and a permanently blank chart
        # (status badge stuck on UNAVAILABLE) -- with no error, indistin-
        # guishable from the chart "just not opening". (saved_mt5 was
        # already loaded above, for the TradingView symbol field's default.)
        mt5_usable = bool(
            saved_mt5 is not None and saved_mt5.is_usable and mt5_connector_module.is_available()
        )
        try:
            has_alpaca = bool(alpaca_credentials.load_credentials())
        except Exception:
            has_alpaca = False
        default_source = "mt5" if mt5_usable else ("alpaca" if has_alpaca else "replay")

        self.lm_source = LabeledCombo(section, "Source", ["mt5", "alpaca", "replay"], default_source)
        self.lm_source.combo.bind("<<ComboboxSelected>>", lambda _e: self._lm_sync_source_fields())
        self.lm_symbol = LabeledEntry(section, "Symbol (MT5 / Alpaca)", "XAUUSD")
        if saved_mt5 is not None and saved_mt5.is_usable:
            self.lm_symbol.var.set(saved_mt5.symbol)
        self.lm_timeframe = LabeledCombo(
            section, "Timeframe (minutes)", [str(m) for m in (1, 5, 15, 30, 60, 240, 1440)], "15",
        )
        self.lm_dataset = LabeledCombo(section, "Dataset (Replay only)", [""], "")

        status_lines = []
        if mt5_connector_module.is_available():
            status_lines.append("MT5 package: available.")
        else:
            status_lines.append("MT5 package: not available on this platform (Windows + MT5 terminal required).")
        status_lines.append(f"Alpaca keys saved: {'yes' if has_alpaca else 'no (set them on the 01 DATA tab)'}.")
        status_lines.append(f"Source defaulted to '{default_source}' based on what's actually configured above -- "
                             "switch it manually any time.")
        Label(
            section, text="  •  ".join(status_lines), bg=PANEL, fg=TEXT_MUTED,
            font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._button(btn_row, "REFRESH DATASETS", self._lm_refresh_datasets).pack(side="left")
        self.lm_open_btn = self._button(btn_row, "OPEN LIVE CHART", self._lm_open_chart, primary=True)
        self.lm_open_btn.pack(side="left", padx=8)

        self.lm_status = Label(
            f, text="Not started yet -- click OPEN LIVE CHART.", bg=BG, fg=TEXT_MUTED,
            font=_safe_font(9), wraplength=900, justify="left",
        )
        self.lm_status.pack(anchor="w", padx=26, pady=(4, 20))

        self._lm_refresh_datasets()
        self._lm_sync_source_fields()

    def _lm_sync_source_fields(self):
        is_replay = self.lm_source.get_str() == "replay"
        state = "disabled" if is_replay else "normal"
        self.lm_symbol.entry.config(state=state)

    def _lm_refresh_datasets(self):
        try:
            names = live_market.list_replay_datasets()
        except Exception:
            names = []
        self.lm_dataset.combo.config(values=names or [""])
        if names:
            self.lm_dataset.var.set(names[0])

    def _lm_ensure_server(self) -> int:
        if getattr(self, "_lm_port", None):
            return self._lm_port
        import socket

        from werkzeug.serving import make_server

        from app.web.server import app as flask_app

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = make_server("127.0.0.1", port, flask_app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._lm_port = port
        self._lm_server = server
        return port

    def _lm_open_in_native_window(self, url: str) -> bool:
        """Best-effort: opens the chart in a chromeless native window (via
        the optional `pywebview` package) so it feels like part of the app
        rather than a separate browser tab. Tkinter itself has no way to
        run real JavaScript/Canvas content -- what Lightweight Charts
        needs -- so this is the closest thing to "embedded" achievable
        without a much heavier dependency (bundling a full Chromium
        build). Returns False on any failure (not installed, no supported
        native webview backend on this machine, etc.) so the caller can
        fall back to the system's default browser -- still fully
        functional either way.

        On Linux, pywebview needs an actual GTK or QT webview backend
        (e.g. `pip install pywebview[gtk]` plus the system
        `python3-gi`/`gir1.2-webkit2-*` packages) -- when that's missing,
        `webview.start()` can raise from inside the background thread
        started below rather than at import time, which previously left
        the person staring at "Opened in a live chart window" with no
        window and no explanation. That failure is now caught and folded
        into a browser fallback instead."""
        try:
            import webview
        except ImportError:
            return False
        try:
            existing = getattr(self, "_lm_webview_window", None)
            if existing is not None:
                existing.load_url(url)
                return True

            def run():
                try:
                    window = webview.create_window(
                        "Live Market — T58", url, width=1300, height=880, min_size=(900, 600),
                    )
                    self._lm_webview_window = window
                    webview.start()
                except Exception as exc:
                    # The native webview backend genuinely isn't usable on
                    # this machine (common on Linux without the GTK/QT
                    # webview system packages installed) -- fall back to
                    # the default browser instead of leaving the status
                    # line claiming a window opened that never appeared.
                    def _fallback():
                        opened = webbrowser.open(url)
                        self.lm_status.config(
                            text=(f"Native chart window isn't available on this machine ({exc}) -- "
                                  f"opened in your browser instead." if opened else
                                  f"Native chart window isn't available on this machine ({exc}), and "
                                  f"no browser could be auto-launched either -- open this URL "
                                  f"manually: {url}"),
                            fg=AMBER if opened else RED,
                        )
                    try:
                        self.root.after(0, _fallback)
                    except Exception:
                        pass
                finally:
                    self._lm_webview_window = None  # the window was closed (or never opened)

            threading.Thread(target=run, daemon=True).start()
            return True
        except Exception:
            return False

    # Rough symbol -> TradingView-ticker mapping for the most common cases
    # this app already deals with (FX majors/gold/silver traded through a
    # prop-firm-style broker, plus indices) -- TradingView needs an
    # EXCHANGE:TICKER pair for its best match, and guessing a reasonable
    # one here means OPEN TRADINGVIEW CHART works immediately for the
    # common case without the person needing to know TradingView's symbol
    # syntax. Anything not recognized is passed through as-is (TradingView
    # itself still does a best-effort search on a bare ticker), and the
    # symbol field can always just be hand-edited to whatever TradingView
    # expects.
    _TRADINGVIEW_FX_METALS = {
        "XAUUSD": "OANDA:XAUUSD", "XAGUSD": "OANDA:XAGUSD",
        "EURUSD": "OANDA:EURUSD", "GBPUSD": "OANDA:GBPUSD", "USDJPY": "OANDA:USDJPY",
        "USDCHF": "OANDA:USDCHF", "USDCAD": "OANDA:USDCAD", "AUDUSD": "OANDA:AUDUSD",
        "NZDUSD": "OANDA:NZDUSD", "EURJPY": "OANDA:EURJPY", "GBPJPY": "OANDA:GBPJPY",
        "EURGBP": "OANDA:EURGBP",
        "US30": "OANDA:US30USD", "NAS100": "OANDA:NAS100USD", "SPX500": "OANDA:SPX500USD",
        "USOIL": "OANDA:WTICOUSD", "BTCUSD": "COINBASE:BTCUSD", "ETHUSD": "COINBASE:ETHUSD",
    }

    @classmethod
    def _tradingview_symbol_for(cls, symbol: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]", "", (symbol or "")).upper()
        return cls._TRADINGVIEW_FX_METALS.get(s, s or "OANDA:XAUUSD")

    def _lm_open_tradingview(self):
        """Opens the real tradingview.com chart for the current symbol in
        the default browser -- no local Flask server, no MT5/Alpaca
        connection, no pywebview dependency, so this always works
        (including on Linux, and even with none of the optional live-data
        sources configured) as long as there's internet access and a
        browser installed. webbrowser.open() uses xdg-open/$BROWSER under
        the hood on Linux, exactly like _open_strategy_library_folder's
        folder-opening already relies on for that platform."""
        raw_symbol = self.lm_tv_symbol.get_str().strip() or "XAUUSD"
        tv_symbol = self._tradingview_symbol_for(raw_symbol)
        url = f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(tv_symbol)}"
        opened = webbrowser.open(url)
        if opened:
            self.lm_status.config(text=f"Opened TradingView chart for {tv_symbol} in your browser.", fg=GREEN)
        else:
            self.lm_status.config(
                text=f"Couldn't auto-launch a browser -- open this URL manually: {url}", fg=AMBER,
            )

    def _lm_open_chart(self):
        self.lm_status.config(text="Starting local server...", fg=AMBER)
        self.root.update_idletasks()
        try:
            port = self._lm_ensure_server()
        except Exception as exc:
            self.lm_status.config(text=f"Couldn't start the local server: {exc}", fg=RED)
            return

        source = self.lm_source.get_str()
        symbol = self.lm_dataset.get_str() if source == "replay" else self.lm_symbol.get_str().strip()
        if not symbol:
            self.lm_status.config(text="Enter a symbol (or pick a dataset for Replay) first.", fg=AMBER)
            return
        params = urllib.parse.urlencode({
            "source": source, "symbol": symbol, "timeframe": self.lm_timeframe.get_str(),
            "theme": CURRENT_THEME,
        })
        url = f"http://127.0.0.1:{port}/live-market?{params}"
        if self._lm_open_in_native_window(url):
            self.lm_status.config(text="Opened in a live chart window.", fg=GREEN)
        else:
            opened = webbrowser.open(url)
            if opened:
                self.lm_status.config(text=f"Opened in your browser: {url}", fg=GREEN)
            else:
                # webbrowser.open() returning False means it couldn't find/launch
                # anything -- previously this branch showed "Opened in your
                # browser" regardless, which looked exactly like "nothing
                # happened" with no indication of why. Surface the URL instead
                # so it can be opened by hand.
                self.lm_status.config(
                    text=f"Couldn't auto-launch a browser -- open this URL manually: {url}", fg=AMBER,
                )

    # -----------------------------------------------------------------------
    # Tab 18 — Deploy Live
    # -----------------------------------------------------------------------

    def _build_deploy_live_tab(self):
        f = self._scrollable(self.tab_deploylive)
        self._page_header(
            f,
            "18 / Going Live For Real",
            "Deploy Live",
            "Connect a validated strategy directly to a real, funded prop-firm account for "
            "automated live trading -- no manual clicking required once it's running.",
        )

        warn_wrap = Frame(f, bg="#3A0E14", highlightthickness=1, highlightbackground=RED)
        warn_wrap.pack(fill="x", padx=24, pady=(0, 16))
        Label(
            warn_wrap, text="⚠  THIS TRADES REAL MONEY -- READ BEFORE CONNECTING ANY ACCOUNT",
            bg="#3A0E14", fg="#FF8FA0", font=_safe_font(10, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 4))
        Label(
            warn_wrap,
            text="This is not the same as Live Demo Test. An account connected here places real "
                 "orders against real capital in a live or funded prop-firm account. Before using this:\n"
                 "  1) Confirm your prop firm's rules actually PERMIT automated/EA trading on this "
                 "account -- many firms restrict or ban certain automation, and violating that can get "
                 "a funded account terminated regardless of performance.\n"
                 "  2) Validate the strategy thoroughly first (15 FULL PIPELINE) and run it on 16 LIVE "
                 "DEMO TEST for a meaningful stretch before ever pointing this at real capital.\n"
                 "  3) Never share account credentials anywhere else -- this app stores the password "
                 "locally via your OS's secure credential store (the same mechanism as the Live Demo "
                 "Test tab), the same way, for this reason.\n"
                 "  4) Start with the smallest account/position size your firm allows.",
            bg="#3A0E14", fg="#F0C7CC", font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 14))

        # ---- How this actually connects to a prop firm ---------------------
        info_section = self._section(
            f, "How this connects to a prop firm -- read this first",
            "",
            emphasize=True,
        )
        Label(
            info_section,
            text="There is no single universal \"prop firm API\" this app can plug into -- each firm "
                 "provides one of a small number of trading platforms for its funded accounts, and this "
                 "app connects the same way a human trader using that platform would:\n\n"
                 "  •  MT4 / MT5 (by far the most common) -- the exact same connection this app already "
                 "uses for Live Demo Test, just pointed at your live/funded login instead of a demo "
                 "one. This is the only path fully wired up below today.\n"
                 "  •  cTrader (a growing number of firms) -- cTrader has its own Open API (OAuth-based, "
                 "different from MT5 entirely). Not implemented yet -- see the note below.\n"
                 "  •  DXtrade / Match-Trader / proprietary platforms -- each would need its own "
                 "integration; none are implemented yet.\n\n"
                 "Practically: if your prop firm gives you an MT5 login for your funded account (most "
                 "do), you can connect it below today. If they use cTrader or something else, that "
                 "account isn't supported yet -- let me know which platform your firm uses and that's "
                 "the next one to wire up.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(2, 14))

        # ---- Saved live accounts --------------------------------------------
        accounts_section = self._section(
            f, "Live prop-firm accounts",
            "Each saved account remembers its prop firm, platform, and login -- the password is "
            "stored the same secure way as the Live Demo Test tab (OS keyring), never in plain text.",
            emphasize=True,
        )

        list_frame = Frame(accounts_section, bg=PANEL)
        list_frame.pack(fill="both", padx=18, pady=(2, 8))
        self.dl_accounts_listbox = Listbox(
            list_frame, height=6, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.dl_accounts_listbox.pack(side="left", fill="both", expand=True)
        dl_scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.dl_accounts_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        dl_scrollbar.pack(side="right", fill="y")
        self.dl_accounts_listbox.config(yscrollcommand=dl_scrollbar.set)
        self.dl_accounts_listbox.bind("<<ListboxSelect>>", self._dl_on_account_selected)

        list_btn_row = Frame(accounts_section, bg=PANEL)
        list_btn_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._button(list_btn_row, "DELETE SELECTED", self._dl_delete_account).pack(side="left")

        # ---- Add / edit an account -------------------------------------------
        add_section = self._section(
            f, "Add / edit an account",
            "Pick your prop firm (or Other) and platform, then enter the login details from your "
            "firm's account-issued email or dashboard.",
            emphasize=True,
        )

        firm_names = [firm.name for firm in live_deploy_prop_firms.PROP_FIRMS]
        self.dl_firm = LabeledCombo(add_section, "Prop firm", firm_names, firm_names[0] if firm_names else "Other")
        self.dl_firm.combo.bind("<<ComboboxSelected>>", lambda _e: self._dl_on_firm_changed())
        self.dl_platform = LabeledCombo(
            add_section, "Platform",
            ["MT5", "MT4", "cTrader (not yet supported)", "Tradovate (not yet supported)",
             "Rithmic (not yet supported)", "NinjaTrader (not yet supported)", "Other (not yet supported)"],
            "MT5",
        )
        self.dl_nickname = LabeledEntry(add_section, "Account nickname (yours, e.g. 'FTMO 100k #1')", "")
        self.dl_login = LabeledEntry(add_section, "Login / account number", "")
        self.dl_server = LabeledEntry(add_section, "Server (from your firm's account email)", "")
        self.dl_password = LabeledEntry(add_section, "Password", "", secret=True)
        self.dl_terminal_path = LabeledEntry(add_section, "Terminal path (optional, only if auto-detect fails)", "")

        firm_note_row = Frame(add_section, bg=PANEL)
        firm_note_row.pack(anchor="w", padx=18, pady=(0, 8))
        self.dl_firm_note = Label(
            firm_note_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        )
        self.dl_firm_note.pack(anchor="w")

        confirm_row = Frame(add_section, bg=PANEL)
        confirm_row.pack(anchor="w", padx=18, pady=(2, 4))
        self.dl_confirm_var = BooleanVar(value=False)
        Checkbutton(
            confirm_row, variable=self.dl_confirm_var, bg=PANEL, activebackground=PANEL,
            selectcolor=PANEL_3, highlightthickness=0,
            text="I understand this account will trade with real capital, and I've confirmed my prop "
                 "firm permits automated trading on it.",
            fg=TEXT_MUTED, font=_safe_font(8), wraplength=780, justify="left", anchor="w",
        ).pack(anchor="w")

        add_btn_row = Frame(add_section, bg=PANEL)
        add_btn_row.pack(anchor="w", padx=18, pady=(8, 14))
        self._button(add_btn_row, "SAVE ACCOUNT", self._dl_save_account, primary=True).pack(side="left")
        self._button(add_btn_row, "AUTO-DETECT TERMINAL", self._dl_auto_detect_terminal).pack(side="left", padx=8)
        self._button(add_btn_row, "BROWSE...", self._dl_browse_terminal).pack(side="left")

        # ---- Market + risk (same shape as Live Demo Test's own fields) -----
        market_section = self._section(
            f, "Market",
            "Which symbol and timeframe the strategy trades on this account.",
        )
        self.dl_symbol = LabeledEntry(market_section, "Symbol", "XAUUSD")
        self.dl_timeframe = LabeledCombo(
            market_section, "Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day"],
            "15 minutes",
        )

        risk_section = self._section(
            f, "Risk (same fields as Step 04, applied to this live account)",
            "Position sizing calls the exact same RiskConfig.position_size(...) the backtester "
            "and Live Demo Test use -- if a strategy defines its own dynamic stop, that's "
            "honored first; otherwise a fixed-pip or 1%-of-price fallback applies, same "
            "precedence as everywhere else in this app.",
        )
        self.dl_risk_pct = LabeledEntry(risk_section, "Risk per trade (% of equity)", 1.0)
        self.dl_daily_loss_limit = LabeledEntry(risk_section, "Daily loss limit (% of balance, halts new entries)", 5.0)
        self.dl_max_trades_per_day = LabeledEntry(risk_section, "Max trades per day", 10)
        self.dl_pip_size = LabeledEntry(risk_section, "Pip size (leave blank to auto-detect from MT5 symbol info)", "")
        self.dl_baseline_win_rate = LabeledEntry(
            risk_section, "Backtest win rate %, for drift comparison (optional)", "",
        )

        # ---- Connect + deploy -------------------------------------------------
        deploy_section = self._section(
            f, "Connect and deploy",
            "Same underlying engine as Live Demo Test -- pick a saved account above, pick a "
            "validated strategy from the Strategy Library, and start. The kill switch works "
            "identically.",
            emphasize=True,
        )
        self.dl_strategy_combo = LabeledCombo(deploy_section, "Strategy (from Strategy Library)", [""], "")
        deploy_btn_row = Frame(deploy_section, bg=PANEL)
        deploy_btn_row.pack(anchor="w", padx=18, pady=(6, 6))
        self._button(deploy_btn_row, "TEST CONNECTION", self._dl_test_connection).pack(side="left")
        self.dl_start_btn = self._button(deploy_btn_row, "START LIVE TRADING", self._dl_start_clicked, primary=True)
        self.dl_start_btn.pack(side="left", padx=8)
        self.dl_start_btn.config(state="disabled")
        self.dl_stop_btn = self._button(deploy_btn_row, "STOP", self._dl_stop_clicked)
        self.dl_stop_btn.pack(side="left", padx=8)
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn = Button(
            deploy_btn_row, text="KILL SWITCH — FLATTEN & STOP", command=self._dl_kill_clicked,
            bg=RED, fg="#1a0a0d", activebackground="#c94656", activeforeground="#1a0a0d",
            relief="flat", bd=0, font=_safe_font(9, "bold"), padx=14, pady=8, cursor="hand2",
        )
        self.dl_kill_btn.pack(side="left", padx=8)
        self.dl_kill_btn.config(state="disabled")

        self.dl_progress = NeuralProgress(deploy_section)
        self.dl_progress.pack(fill="x", padx=18, pady=(6, 10))

        self.dl_status = Label(
            deploy_section, text="Select a saved account above, then Test Connection.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(10, "bold"), wraplength=900, justify="left",
        )
        self.dl_status.pack(anchor="w", padx=18, pady=(2, 16))

        journal_section = self._section(f, "Trade journal (this session)", "")
        columns = ("time", "direction", "volume", "entry", "exit", "pnl", "status")
        self.dl_journal_tree = ttk.Treeview(journal_section, columns=columns, show="headings", height=8)
        for col, label, w in (
            ("time", "Entry Time", 140), ("direction", "Dir", 50), ("volume", "Size", 70),
            ("entry", "Entry", 90), ("exit", "Exit", 90), ("pnl", "P&L", 80), ("status", "Status", 70),
        ):
            self.dl_journal_tree.heading(col, text=label)
            self.dl_journal_tree.column(col, width=w, anchor="center")
        self.dl_journal_tree.pack(fill="x", padx=18, pady=(2, 12))

        log_section = self._section(f, "Live log", "")
        self.dl_log = Text(
            log_section, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.dl_log.pack(fill="both", expand=True, padx=18, pady=(3, 16))

        self._dl_accounts: list = []
        self._dl_editing_id: str | None = None
        self._dl_session = None
        self._dl_journal = None
        self._dl_selected_account = None
        self._dl_strategy_items: list = []
        self._dl_refresh_accounts()
        self._dl_on_firm_changed()
        self._dl_refresh_strategy_list()

    def _dl_timeframe_minutes(self) -> int:
        mapping = {
            "1 minute": 1, "5 minutes": 5, "15 minutes": 15, "30 minutes": 30,
            "1 hour": 60, "4 hours": 240, "1 day": 1440,
        }
        return mapping.get(self.dl_timeframe.get_str(), 15)

    def _dl_log_line(self, level: str, message: str):
        color = {"error": RED, "warn": AMBER, "info": TEXT}.get(level, TEXT)
        tag = f"lvl_{level}"
        self.dl_log.tag_config(tag, foreground=color)
        self.dl_log.insert(END, f"[{level.upper()}] {message}\n", tag)
        self.dl_log.see(END)
        try:
            self.root.after(0, self.root.update_idletasks)
        except Exception:
            pass

    def _dl_selected_strategy_item(self):
        label = self.dl_strategy_combo.get_str()
        for it in self._dl_strategy_items:
            if f"[{it.strategy_type}] {it.name}" == label:
                return it
        return None

    def _dl_refresh_journal_view(self):
        for row in self.dl_journal_tree.get_children():
            self.dl_journal_tree.delete(row)
        if self._dl_journal is None or self._dl_session is None or self._dl_session._session_id is None:
            return
        trades = self._dl_journal.all_trades(self._dl_session._session_id)
        for t in trades[:100]:
            import datetime

            entry_time = datetime.datetime.fromtimestamp(t.entry_time).strftime("%m-%d %H:%M")
            direction = "LONG" if t.direction == 1 else "SHORT"
            exit_price = f"{t.exit_price:.5f}" if t.exit_price else ""
            pnl = f"{t.pnl:,.2f}" if t.pnl is not None else ""
            self.dl_journal_tree.insert("", END, values=(
                entry_time, direction, f"{t.volume:.2f}", f"{t.entry_price:.5f}", exit_price, pnl, t.status,
            ))

    def _dl_update_status(self, status):
        parts = [f"Running: {status.running}", f"Last signal: {status.last_signal}"]
        if status.balance is not None:
            parts.append(f"Balance ${status.balance:,.2f}")
        if status.equity is not None:
            parts.append(f"Equity ${status.equity:,.2f}")
        if status.n_trades_closed:
            parts.append(f"Closed trades: {status.n_trades_closed}")
        if status.win_rate is not None:
            parts.append(f"Win rate {status.win_rate:.1f}%")
        if status.halted_reason:
            parts.append(f"HALTED: {status.halted_reason}")
        color = RED if status.halted_reason else (GREEN if status.running else TEXT_DIM)
        if status.drift_flag:
            parts.append(f"⚠ {status.drift_flag}")
            color = AMBER
        self.dl_status.config(text="  |  ".join(parts), fg=color)
        self._dl_refresh_journal_view()

    def _dl_on_started(self):
        self.dl_start_btn.config(state="disabled")
        self.dl_stop_btn.config(state="normal")
        self.dl_kill_btn.config(state="normal")
        self.dl_progress.start()
        self.dl_status.config(text="Running -- placing real orders on this account.", fg=GREEN)

    def _dl_on_firm_changed(self):
        name = self.dl_firm.get_str()
        firm = live_deploy_prop_firms.find(name)
        if firm is None:
            self.dl_firm_note.config(text="")
            return
        connectable_note = (
            "" if firm.connectable_today else
            " ⚠ None of this firm's platforms are supported by this app yet -- you can save the "
            "account details below, but connecting/trading won't work until that platform is added."
        )
        self.dl_firm_note.config(
            text=f"Platform(s): {', '.join(firm.platforms)}. {firm.notes}{connectable_note} Always "
                 f"confirm the exact server name from your own account-issued email -- server names "
                 f"change and vary by region/account type even within one firm.",
            fg=TEXT_DIM if firm.connectable_today else AMBER,
        )
        preferred = next((p for p in firm.platforms if p in ("MT4", "MT5")), firm.platforms[0] if firm.platforms else "")
        matching = next(
            (v for v in self.dl_platform.combo.cget("values") if v == preferred or v.startswith(preferred + " ")),
            None,
        )
        if matching:
            self.dl_platform.var.set(matching)

    def _dl_refresh_accounts(self):
        self._dl_accounts = live_deploy_settings.load_accounts()
        self.dl_accounts_listbox.delete(0, END)
        for acct in self._dl_accounts:
            self.dl_accounts_listbox.insert(
                END, f"{acct.nickname}  —  {acct.firm_name} ({acct.platform})  —  login {acct.login}",
            )

    def _dl_on_account_selected(self, _event=None):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            return
        acct = self._dl_accounts[sel[0]]
        self._dl_editing_id = acct.id
        self._dl_selected_account = acct
        self.dl_firm.var.set(acct.firm_name)
        self.dl_platform.var.set(acct.platform)
        self.dl_nickname.var.set(acct.nickname)
        self.dl_login.var.set(acct.login)
        self.dl_server.var.set(acct.server)
        self.dl_terminal_path.var.set(acct.terminal_path)
        self.dl_firm_note.config(text="Editing this saved account. Password is never re-displayed -- leave it "
                                       "blank to keep the existing one, or enter a new one to replace it.")
        connectable = "(not yet supported)" not in acct.platform
        self.dl_start_btn.config(state="normal" if connectable and self._dl_session is None else "disabled")
        self.dl_status.config(
            text=f"Selected '{acct.nickname}'. Click TEST CONNECTION, then START LIVE TRADING when ready."
            if connectable else
            f"'{acct.nickname}' uses {acct.platform}, which isn't connectable yet -- see the note above.",
            fg=TEXT_DIM if connectable else AMBER,
        )

    def _dl_delete_account(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete account", "Select an account in the list first.")
            return
        acct = self._dl_accounts[sel[0]]
        if not messagebox.askyesno("Delete account", f"Remove the saved connection for '{acct.nickname}'? This does not affect the account at your prop firm itself."):
            return
        live_deploy_settings.delete_account(acct.id)
        self._dl_refresh_accounts()

    def _dl_browse_terminal(self):
        path = filedialog.askopenfilename(
            title="Locate your MT5/MT4 terminal executable",
            filetypes=[("Terminal executable", "terminal64.exe terminal.exe"), ("All files", "*.*")],
        )
        if path:
            self.dl_terminal_path.var.set(path)

    def _dl_auto_detect_terminal(self):
        candidates = mt5_connector_module.find_terminal_candidates()
        if not candidates:
            messagebox.showinfo("Auto-detect", "Couldn't find a terminal in any common install location. Use Browse... instead.")
            return
        self.dl_terminal_path.var.set(candidates[0])

    def _dl_save_account(self):
        platform = self.dl_platform.get_str()
        if "(not yet supported)" in platform:
            messagebox.showwarning(
                "Not supported yet",
                f"{platform.split(' (')[0]} accounts aren't wired up yet -- this app can only place "
                "live orders through MT4/MT5 today. Saving these account details now so they're "
                "ready the moment that integration exists, but START LIVE TRADING will stay disabled "
                "for this account until then.",
            )
        if not self.dl_confirm_var.get():
            messagebox.showwarning(
                "Confirmation required",
                "Check the confirmation box above before saving a live account -- this is a real-money "
                "account, not a demo.",
            )
            return
        if not self.dl_nickname.get_str().strip() or not self.dl_login.get_str().strip() or not self.dl_server.get_str().strip():
            messagebox.showwarning("Missing fields", "Nickname, login, and server are all required.")
            return
        live_deploy_settings.save_account(live_deploy_settings.LiveAccount(
            id=getattr(self, "_dl_editing_id", None),
            nickname=self.dl_nickname.get_str().strip(),
            firm_name=self.dl_firm.get_str(),
            platform=platform,
            login=self.dl_login.get_str().strip(),
            server=self.dl_server.get_str().strip(),
            password=self.dl_password.get_str(),
            terminal_path=self.dl_terminal_path.get_str().strip(),
        ))
        self._dl_editing_id = None
        self.dl_password.var.set("")
        self.dl_confirm_var.set(False)
        self._dl_refresh_accounts()
        messagebox.showinfo("Saved", "Account saved. Select it in the list above, then Test Connection.")

    def _dl_refresh_strategy_list(self):
        try:
            items = list_saved_strategies(None)
        except Exception:
            items = []
        self._dl_strategy_items = items
        labels = [f"[{it.strategy_type}] {it.name}" for it in items]
        self.dl_strategy_combo.combo.config(values=labels or [""])
        if labels and not self.dl_strategy_combo.get_str():
            self.dl_strategy_combo.var.set(labels[0])

    def _dl_test_connection(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showinfo("Test connection", "Select a saved account first.")
            return
        acct = self._dl_accounts[sel[0]]
        if "(not yet supported)" in acct.platform:
            self.dl_status.config(text=f"{acct.platform} isn't supported yet -- see the note above.", fg=AMBER)
            return
        if not mt5_connector_module.is_available():
            self.dl_status.config(text=mt5_connector_module.unavailable_reason(), fg=AMBER)
            return
        self.dl_status.config(text="Connecting...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            connector = mt5_connector_module.MT5Connector(acct.login, acct.password, acct.server, acct.terminal_path)
            result = connector.connect()
            if result.ok:
                msg = (f"Connected: account {result.account_login} @ {result.account_server} — "
                       f"balance {result.balance:,.2f} {result.currency}, equity {result.equity:,.2f}. "
                       f"Ready -- pick a strategy below and click START LIVE TRADING when you're sure.")
                connector.disconnect()
                color = GREEN
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal" if self._dl_session is None else "disabled"))
            else:
                msg, color = result.message, RED
            self.root.after(0, lambda: self.dl_status.config(text=msg, fg=color))

        threading.Thread(target=run, daemon=True).start()

    def _dl_start_clicked(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showwarning("No account selected", "Select a saved live account first.")
            return
        acct = self._dl_accounts[sel[0]]
        if "(not yet supported)" in acct.platform:
            messagebox.showwarning("Not supported yet", f"{acct.platform} isn't wired up yet -- see the note above.")
            return
        if not mt5_connector_module.is_available():
            messagebox.showwarning("MT5 not available", mt5_connector_module.unavailable_reason())
            return

        item = self._dl_selected_strategy_item()
        if item is None:
            messagebox.showwarning("No strategy selected", "Choose a strategy from the Strategy Library first.")
            return

        symbol = self.dl_symbol.get_str().strip()
        if not symbol:
            messagebox.showwarning("Missing symbol", "Enter the symbol to trade (must match your broker's Market Watch name).")
            return

        if not messagebox.askyesno(
            "Confirm live trading",
            f"This will place REAL orders with REAL money on account '{acct.nickname}' "
            f"({acct.firm_name}, login {acct.login}) trading strategy '{item.name}' on {symbol}.\n\n"
            "Have you validated this strategy (15 FULL PIPELINE) and run it on Live Demo Test first, "
            "and confirmed your prop firm permits automated trading on this account?\n\n"
            "This cannot be undone once trades are placed. Continue?",
            icon="warning",
        ):
            return

        try:
            strategy = self._ft_build_strategy_instance(item)
        except Exception as exc:
            messagebox.showerror("Strategy error", f"Could not load strategy: {exc}")
            return

        from app.forward_test.mt5_connector import MT5Connector

        connector = MT5Connector(acct.login, acct.password, acct.server, acct.terminal_path)
        pip_size_str = self.dl_pip_size.get_str().strip()
        pip_size = None
        if pip_size_str:
            try:
                pip_size = float(pip_size_str)
            except ValueError:
                pip_size = None

        def resolve_pip_and_start():
            nonlocal pip_size
            probe = MT5Connector(acct.login, acct.password, acct.server, acct.terminal_path)
            conn = probe.connect()
            if not conn.ok:
                self.root.after(0, lambda: self._dl_log_line("error", conn.message))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            if pip_size is None:
                try:
                    pip_size = probe.symbol_point(symbol)
                except Exception as exc:
                    self.root.after(0, lambda: self._dl_log_line(
                        "warn", f"Could not auto-detect pip size ({exc}); falling back to 0.0001."))
                    pip_size = 0.0001
            probe.disconnect()

            risk = RiskConfig(
                initial_balance=conn.balance or 10_000.0,
                risk_mode="percent",
                risk_value=self.dl_risk_pct.get_float(1.0),
                max_trades_per_day=self.dl_max_trades_per_day.get_int(10),
                pip_size=pip_size,
                daily_loss_limit_pct=self.dl_daily_loss_limit.get_float(5.0) or None,
            )
            baseline_str = self.dl_baseline_win_rate.get_str().strip()
            baseline_win_rate = float(baseline_str) if baseline_str else None

            cfg = ForwardTestConfig(
                symbol=symbol, timeframe_minutes=self._dl_timeframe_minutes(),
                risk=risk, baseline_win_rate=baseline_win_rate,
            )
            self._dl_journal = ForwardTestJournal()
            session = ForwardTestSession(
                strategy=strategy, strategy_type=item.strategy_type, strategy_filename=item.name,
                connector=connector, journal=self._dl_journal, config=cfg,
                on_log=lambda level, msg: self.root.after(0, lambda: self._dl_log_line(level, msg)),
                on_status=lambda status: self.root.after(0, lambda: self._dl_update_status(status)),
            )
            ok, msg = session.start()
            if not ok:
                self.root.after(0, lambda: self._dl_log_line("error", msg))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            self._dl_session = session
            self.root.after(0, self._dl_on_started)

        threading.Thread(target=resolve_pip_and_start, daemon=True).start()
        self.dl_start_btn.config(state="disabled")
        self.dl_status.config(text="Connecting...", fg=AMBER)
        self._dl_log_line("info", f"Starting live trading on '{acct.nickname}' ({symbol}, {item.name})...")

    def _dl_stop_clicked(self):
        if self._dl_session is None:
            return
        self.dl_status.config(text="Stopping...", fg=AMBER)
        threading.Thread(target=self._dl_session.stop, daemon=True).start()
        self.dl_start_btn.config(state="normal")
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn.config(state="disabled")
        self.dl_progress.stop()
        self._dl_session = None

    def _dl_kill_clicked(self):
        if self._dl_session is None:
            return
        if not messagebox.askyesno(
            "Kill switch",
            "This closes every open REAL position on this account immediately and stops the "
            "session. Continue?",
        ):
            return
        self.dl_status.config(text="Flattening and stopping...", fg=RED)
        session = self._dl_session
        threading.Thread(target=session.flatten_all_and_stop, daemon=True).start()
        self.dl_start_btn.config(state="normal")
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn.config(state="disabled")
        self.dl_progress.stop()
        self._dl_session = None


class _SplashScreen(Toplevel):
    """Dark, borderless boot splash shown while MainWindow builds its ~17
    tabs' worth of widgets. Without this, the very first thing the user
    sees is the real window rendering itself piece by piece (sidebar, then
    each tab's frame, then charts) which reads as glitchy -- the splash
    covers exactly that window with something intentional instead, and
    disappears the instant the real UI is fully built and ready to show."""

    WIDTH, HEIGHT = 520, 320

    def __init__(self, parent: Tk):
        super().__init__(parent)
        self.overrideredirect(True)
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        bg = "#05060A"  # deliberately darker/more "ominous" than the app's own BG
        self.configure(bg=bg)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - self.WIDTH) // 2, (sh - self.HEIGHT) // 2
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

        self._canvas = Canvas(self, width=self.WIDTH, height=self.HEIGHT, bg=bg, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._glow_color = NEON_CYAN
        self._bg = bg
        self._tick = 0
        self._status = "Booting T58 Prop Algo Backtester..."
        self._draw()
        self._animate()

    def set_status(self, text: str) -> None:
        self._status = text
        self._draw()

    def _draw(self) -> None:
        c = self._canvas
        c.delete("all")
        cx, cy = self.WIDTH / 2, self.HEIGHT / 2 - 20

        # Pulsing neon-blue glow halo behind the wordmark -- several
        # progressively larger, more transparent rings, the same
        # "layered outline" trick GlowCard uses elsewhere in this app,
        # with the pulse driven by a slow sine wave instead of a fixed
        # radius so it reads as alive rather than static.
        import math
        pulse = 0.5 + 0.5 * math.sin(self._tick / 14.0)
        for i, base_alpha in ((5, 0.04), (4, 0.07), (3, 0.11), (2, 0.16), (1, 0.24)):
            alpha = base_alpha * (0.6 + 0.4 * pulse)
            r = 70 + i * 16
            c.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=_blend_hex(self._bg, self._glow_color, alpha), width=3,
            )

        c.create_text(
            cx, cy, text="T58", fill=_blend_hex(self._bg, self._glow_color, 0.85 + 0.15 * pulse),
            font=_safe_font(46, "bold"),
        )
        c.create_text(
            cx, cy + 44, text="PROP ALGO BACKTESTER", fill=self._glow_color,
            font=_safe_font(11, "bold"),
        )
        c.create_text(
            cx, self.HEIGHT - 34, text=self._status, fill="#5C6472",
            font=_safe_font(9),
        )
        # Small horizontal "loading" tick marching left-to-right, since a
        # real determinate percentage isn't available for widget
        # construction -- just enough motion to read as "working," not
        # "frozen."
        bar_w = 220
        bx0, by = cx - bar_w / 2, self.HEIGHT - 54
        c.create_line(bx0, by, bx0 + bar_w, by, fill="#1E232E", width=3)
        pos = (self._tick * 6) % (bar_w + 60) - 30
        c.create_line(
            max(bx0, bx0 + pos - 30), by, min(bx0 + bar_w, bx0 + pos + 30), by,
            fill=self._glow_color, width=3,
        )

    def _animate(self) -> None:
        self._tick += 1
        try:
            self._draw()
            self.after(60, self._animate)
        except Exception:
            pass  # splash already destroyed


def launch():
    root = Tk()
    root.withdraw()  # hidden while the real window builds, splash covers that gap
    splash = None
    try:
        splash = _SplashScreen(root)
        root._t58_splash = splash
        root.update()
    except Exception:
        splash = None  # never let a cosmetic splash failure block the real app

    window = MainWindow(root)

    if splash is not None:
        try:
            del root._t58_splash
            splash.destroy()
        except Exception:
            pass
    root.deiconify()
    try:
        root.lift()
        root.focus_force()
    except Exception:
        pass
    root.mainloop()


