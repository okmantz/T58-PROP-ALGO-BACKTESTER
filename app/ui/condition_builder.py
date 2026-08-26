"""
Visual condition-row builder for the Manual Strategy Builder.

Each ConditionRow lets a user build one rule of the form:

    <left source>  <operator>  <right source>

e.g.  Close  >  EMA(50)          or          RSI(14)  Greater Than  55

without writing any code. A ConditionList manages a vertical stack of rows
plus the AND/OR connector between consecutive rows, and can serialize
itself into the plain dict structure that app.strategy.manual.ManualStrategy
expects (see `to_condition_list()`).

Fields that don't apply to the selected source (e.g. a "Period" for Price,
or a Field for RSI) are hidden automatically, so the user is never asked
for an input a condition doesn't need. Conditions whose source is inherently
true/false (Swing High, a directional Liquidity Sweep, etc.) hide the
operator and right-hand side entirely, since there's nothing to compare.
"""
from __future__ import annotations

from tkinter import Frame, Label, Entry, StringVar, ttk

# ---------------------------------------------------------------------------
# Palette (kept local to avoid a circular import with main_window.py)
# ---------------------------------------------------------------------------
PANEL_2 = "#15191F"
PANEL_3 = "#1B2027"
BORDER = "#292E36"
TEXT = "#E5E7EB"
TEXT_DIM = "#626A75"
GREEN = "#43D17A"
RED = "#F05B63"
BLUE = "#6FA8FF"
FONT = "Segoe UI"

# ---------------------------------------------------------------------------
# Source / operator vocabulary — mirrors app.strategy.manual's supported
# operand "type" values exactly, so anything selectable here is guaranteed
# to be understood by the backtest engine.
# ---------------------------------------------------------------------------
CONDITION_SOURCES = [
    "Price", "EMA", "SMA", "WMA", "VWAP", "RSI", "MACD", "MACD Signal", "MACD Histogram",
    "ATR", "Bollinger Upper", "Bollinger Mid", "Bollinger Lower", "Highest High", "Lowest Low",
    "Volume", "Average Volume", "Candle Direction", "Candle Range", "Percentage Change",
    "Swing High", "Swing Low", "Liquidity Sweep", "Break of Structure", "Change of Character",
    "Fair Value Gap", "Order Block", "Session High", "Session Low", "Previous Day High",
    "Previous Day Low", "Previous Day Close", "Opening Range High", "Opening Range Low",
    "ATR Regime", "Volatility Regime",
]
RIGHT_SOURCES = ["Value"] + CONDITION_SOURCES

SOURCE_KIND = {
    "Price": "price", "EMA": "ema", "SMA": "sma", "WMA": "wma", "VWAP": "vwap", "RSI": "rsi",
    "MACD": "macd", "MACD Signal": "macd_signal", "MACD Histogram": "macd_histogram", "ATR": "atr",
    "Bollinger Upper": "bollinger_upper", "Bollinger Mid": "bollinger_mid", "Bollinger Lower": "bollinger_lower",
    "Highest High": "highest_high", "Lowest Low": "lowest_low", "Volume": "volume",
    "Average Volume": "average_volume", "Candle Direction": "candle_direction", "Candle Range": "candle_range",
    "Percentage Change": "percentage_change", "Swing High": "swing_high", "Swing Low": "swing_low",
    "Liquidity Sweep": "liquidity_sweep", "Break of Structure": "break_of_structure",
    "Change of Character": "change_of_character", "Fair Value Gap": "fair_value_gap",
    "Order Block": "order_block", "Session High": "session_high", "Session Low": "session_low",
    "Previous Day High": "previous_day_high", "Previous Day Low": "previous_day_low",
    "Previous Day Close": "previous_day_close", "Opening Range High": "opening_range_high",
    "Opening Range Low": "opening_range_low", "ATR Regime": "atr_regime", "Volatility Regime": "volatility_regime",
}

OPERATORS = [
    "Greater Than", "Greater Than or Equal", "Less Than", "Less Than or Equal",
    "Equal To", "Not Equal", "Cross Above", "Cross Below",
]
OPERATOR_CODE = {
    "Greater Than": ">", "Greater Than or Equal": ">=", "Less Than": "<", "Less Than or Equal": "<=",
    "Equal To": "==", "Not Equal": "!=", "Cross Above": "cross above", "Cross Below": "cross below",
}
CODE_TO_OPERATOR = {v: k for k, v in OPERATOR_CODE.items()}

# Reverse of SOURCE_KIND ("ema" -> "EMA", etc.) used to repopulate a row's
# dropdowns from a saved/optimized condition dict (see ConditionRow.load_from
# and ConditionList.set_from_conditions) -- e.g. what Iterative Refinement's
# "Apply Best Configuration" button uses to push optimized parameters back
# into this builder.
KIND_TO_SOURCE = {v: k for k, v in SOURCE_KIND.items()}
KIND_TO_SOURCE["price"] = "Price"

PERIOD_KINDS = {
    "ema", "sma", "wma", "rsi", "atr", "macd", "macd_signal", "macd_histogram",
    "bollinger_mid", "bollinger_upper", "bollinger_lower", "highest_high", "lowest_low",
    "average_volume", "percentage_change", "swing_high", "swing_low", "liquidity_sweep",
    "break_of_structure", "change_of_character", "fair_value_gap", "order_block",
    "atr_regime", "volatility_regime",
}
DEFAULT_PERIOD = {
    "rsi": 14, "atr": 14, "atr_regime": 14, "volatility_regime": 14, "average_volume": 20,
    "highest_high": 20, "lowest_low": 20, "percentage_change": 1, "swing_high": 5, "swing_low": 5,
    "liquidity_sweep": 20, "break_of_structure": 20, "change_of_character": 10, "fair_value_gap": 2,
    "order_block": 20, "bollinger_mid": 20, "bollinger_upper": 20, "bollinger_lower": 20,
}

# Kinds where the source column (open/high/low/close) can be chosen.
FIELD_KINDS = {"price", "ema", "sma", "wma", "rsi", "macd", "macd_signal", "macd_histogram",
                "bollinger_mid", "bollinger_upper", "bollinger_lower", "percentage_change"}

# Kinds with a Both/Bullish/Bearish direction filter.
DIRECTION_KINDS = {
    "candle_direction", "liquidity_sweep", "break_of_structure", "change_of_character",
    "fair_value_gap", "order_block",
}

# Kinds that are always a plain yes/no condition (no operator/right side needed).
ALWAYS_BOOLEAN_KINDS = {"swing_high", "swing_low"}

# Kinds whose level depends on the strategy's global session window.
SESSION_KINDS = {"session_high", "session_low", "opening_range_high", "opening_range_low"}


def _combo(parent, values, default, width=14):
    var = StringVar(value=default)
    box = ttk.Combobox(
        parent, textvariable=var, values=values, state="readonly",
        width=width, font=(FONT, 9), style="T58.TCombobox",
    )
    return var, box


class ConditionRow(Frame):
    """One visual rule: <left> <operator> <right>."""

    def __init__(self, parent, on_remove, show_connector: bool, connector_default="AND"):
        super().__init__(parent, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
        self._on_remove = on_remove

        self.connector_var = StringVar(value=connector_default)
        if show_connector:
            conn_row = Frame(self, bg=PANEL_2)
            conn_row.pack(fill="x", padx=10, pady=(8, 0))
            _, conn_box = _combo(conn_row, ["AND", "OR"], connector_default, width=5)
            conn_box.configure(textvariable=self.connector_var)
            conn_box.pack(side="left")
            Label(conn_row, text="the previous condition", bg=PANEL_2, fg=TEXT_DIM,
                  font=(FONT, 8)).pack(side="left", padx=(6, 0))

        # --- left side ---
        left_row = Frame(self, bg=PANEL_2)
        left_row.pack(fill="x", padx=10, pady=(8, 3))
        Label(left_row, text="IF", bg=PANEL_2, fg=BLUE, font=(FONT, 9, "bold")).pack(side="left", padx=(0, 6))

        self.left_kind_var, left_box = _combo(left_row, CONDITION_SOURCES, "Price", width=16)
        left_box.pack(side="left", padx=(0, 6))
        left_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        self.left_extra = Frame(left_row, bg=PANEL_2)
        self.left_extra.pack(side="left")
        self.left_period_var = StringVar(value="14")
        self.left_period_entry = Entry(self.left_extra, textvariable=self.left_period_var, width=5,
                                        bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat")
        self.left_field_var, self.left_field_box = _combo(self.left_extra, ["Open", "High", "Low", "Close"], "Close", width=6)
        self.left_direction_var, self.left_direction_box = _combo(self.left_extra, ["Both", "Bullish", "Bearish"], "Both", width=8)
        self.left_direction_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        # --- operator + right side ---
        self.op_row = Frame(self, bg=PANEL_2)
        self.op_row.pack(fill="x", padx=10, pady=(0, 8))

        self.operator_var, self.operator_box = _combo(self.op_row, OPERATORS, "Greater Than", width=16)

        self.right_kind_var, self.right_kind_box = _combo(self.op_row, RIGHT_SOURCES, "Value", width=16)
        self.right_kind_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        self.right_extra = Frame(self.op_row, bg=PANEL_2)
        self.right_value_var = StringVar(value="0")
        self.right_value_entry = Entry(self.right_extra, textvariable=self.right_value_var, width=8,
                                        bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat")
        self.right_period_var = StringVar(value="14")
        self.right_period_entry = Entry(self.right_extra, textvariable=self.right_period_var, width=5,
                                         bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat")
        self.right_field_var, self.right_field_box = _combo(self.right_extra, ["Open", "High", "Low", "Close"], "Close", width=6)
        self.right_direction_var, self.right_direction_box = _combo(self.right_extra, ["Both", "Bullish", "Bearish"], "Both", width=8)

        self.plain_label = Label(self.op_row, text="\u2713 true / false condition \u2014 no value needed",
                                  bg=PANEL_2, fg=TEXT_DIM, font=(FONT, 8, "italic"))

        self.remove_btn = Label(self.op_row, text="\u2715 Remove", bg=PANEL_2, fg=RED,
                                 font=(FONT, 8, "bold"), cursor="hand2")
        self.remove_btn.pack(side="right", padx=(6, 0))
        self.remove_btn.bind("<Button-1>", lambda _e: self._on_remove(self))

        self._refresh()

    # -- helpers ---------------------------------------------------------
    def _kind_of(self, label: str) -> str:
        return SOURCE_KIND.get(label, "value" if label == "Value" else "price")

    def _is_boolean_left(self) -> bool:
        kind = self._kind_of(self.left_kind_var.get())
        if kind in ALWAYS_BOOLEAN_KINDS:
            return True
        if kind in DIRECTION_KINDS and self.left_direction_var.get() != "Both":
            return True
        return False

    @staticmethod
    def _layout_side(period_entry, field_box, direction_box, kind: str):
        for w in (period_entry, field_box, direction_box):
            w.pack_forget()
        if kind in PERIOD_KINDS:
            period_entry.pack(side="left", padx=(0, 6))
        if kind in FIELD_KINDS:
            field_box.pack(side="left", padx=(0, 6))
        if kind in DIRECTION_KINDS:
            direction_box.pack(side="left", padx=(0, 6))

    def _refresh(self):
        left_kind = self._kind_of(self.left_kind_var.get())
        if left_kind in DEFAULT_PERIOD and self.left_period_var.get() in ("14", ""):
            self.left_period_var.set(str(DEFAULT_PERIOD[left_kind]))
        self._layout_side(self.left_period_entry, self.left_field_box, self.left_direction_box, left_kind)

        boolean_left = self._is_boolean_left()

        for w in (self.operator_box, self.right_kind_box, self.right_extra, self.plain_label,
                  self.right_value_entry, self.right_period_entry, self.right_field_box, self.right_direction_box):
            w.pack_forget()

        if boolean_left:
            self.plain_label.pack(side="left", padx=(24, 0))
        else:
            self.operator_box.pack(side="left", padx=(24, 6))
            self.right_kind_box.pack(side="left", padx=(0, 6))
            self.right_extra.pack(side="left")

            right_label = self.right_kind_var.get()
            if right_label == "Value":
                self.right_value_entry.pack(side="left", padx=(0, 6))
            else:
                right_kind = self._kind_of(right_label)
                if right_kind in DEFAULT_PERIOD and self.right_period_var.get() in ("14", ""):
                    self.right_period_var.set(str(DEFAULT_PERIOD[right_kind]))
                self._layout_side(self.right_period_entry, self.right_field_box, self.right_direction_box, right_kind)

        self.remove_btn.pack(side="right", padx=(6, 0))

    def to_condition(self, session_start: str, session_end: str) -> dict:
        left = self._operand(
            self.left_kind_var.get(), self.left_period_var, self.left_field_var,
            self.left_direction_var, None, session_start, session_end,
        )
        if self._is_boolean_left():
            return {"left": left, "operator": "==", "right": {"type": "value", "value": 1}}

        right_label = self.right_kind_var.get()
        right = self._operand(
            right_label, self.right_period_var, self.right_field_var,
            self.right_direction_var, self.right_value_var, session_start, session_end,
        )
        op_code = OPERATOR_CODE.get(self.operator_var.get(), ">")
        return {"left": left, "operator": op_code, "right": right}

    def _operand(self, label, period_var, field_var, direction_var, value_var, session_start, session_end):
        if label == "Value":
            try:
                val = float((value_var.get() if value_var else "0") or 0)
            except ValueError:
                val = 0.0
            return {"type": "value", "value": val}

        kind = self._kind_of(label)
        if kind == "price":
            return {"type": "price", "field": field_var.get().lower()}

        operand: dict = {"type": kind}
        if kind in PERIOD_KINDS:
            try:
                p = int(float(period_var.get() or DEFAULT_PERIOD.get(kind, 14)))
            except ValueError:
                p = DEFAULT_PERIOD.get(kind, 14)
            operand["period"] = max(p, 1)
            operand["lookback"] = max(p, 1)
        if kind in FIELD_KINDS:
            operand["field"] = field_var.get().lower()
        if kind in DIRECTION_KINDS:
            operand["direction"] = direction_var.get().lower()
        if kind in SESSION_KINDS:
            operand["session_start"] = session_start
            operand["session_end"] = session_end
        return operand

    # -- reverse of the above: repopulate this row's widgets from a saved --
    # -- or optimized condition dict (see ConditionList.set_from_conditions) --
    def _load_operand_into(self, operand, kind_var: StringVar, period_var: StringVar,
                            field_var: StringVar, direction_var: StringVar) -> None:
        if not isinstance(operand, dict):
            return
        kind = str(operand.get("type", "price")).lower().strip()
        kind_var.set(KIND_TO_SOURCE.get(kind, "Price"))
        if "period" in operand and operand["period"] is not None:
            period_var.set(str(operand["period"]))
        if "field" in operand and operand["field"]:
            field_var.set(str(operand["field"]).capitalize())
        if "direction" in operand and operand["direction"]:
            direction_var.set(str(operand["direction"]).capitalize())

    def load_from(self, condition: dict, connector: str | None = None) -> None:
        """Repopulate this row's widgets from a condition dict of the same
        shape to_condition() produces. Used to push an Iterative Refinement
        result (or any saved config) back into the visual builder."""
        if connector is not None:
            self.connector_var.set(connector)

        left = condition.get("left", {}) if isinstance(condition, dict) else {}
        self._load_operand_into(left, self.left_kind_var, self.left_period_var,
                                 self.left_field_var, self.left_direction_var)

        right = condition.get("right", {}) if isinstance(condition, dict) else {}
        is_boolean_shorthand = (
            condition.get("operator") == "==" and isinstance(right, dict)
            and right.get("type") == "value" and right.get("value") == 1
        )
        if not is_boolean_shorthand:
            op_code = str(condition.get("operator", ">"))
            self.operator_var.set(CODE_TO_OPERATOR.get(op_code, "Greater Than"))
            if isinstance(right, dict) and right.get("type") in ("value", "constant", "number"):
                self.right_kind_var.set("Value")
                self.right_value_var.set(str(right.get("value", 0)))
            else:
                self._load_operand_into(right, self.right_kind_var, self.right_period_var,
                                         self.right_field_var, self.right_direction_var)

        self._refresh()


class ConditionList:
    """Manages a vertical stack of ConditionRow widgets inside `container`."""

    def __init__(self, container: Frame, get_session=lambda: ("08:30", "15:00")):
        self.container = container
        self.rows: list[ConditionRow] = []
        self._get_session = get_session
        self.empty_label = Label(
            container, text="No conditions yet \u2014 click \u201c+ Add Condition\u201d below.",
            bg=container["bg"], fg=TEXT_DIM, font=(FONT, 8, "italic"),
        )
        self._refresh_empty_label()

    def add_row(self):
        show_connector = len(self.rows) > 0
        row = ConditionRow(self.container, self._remove_row, show_connector)
        row.pack(fill="x", pady=(0, 6))
        self.rows.append(row)
        self._refresh_empty_label()

    def _remove_row(self, row: ConditionRow):
        row.destroy()
        self.rows.remove(row)
        self._refresh_empty_label()

    def _refresh_empty_label(self):
        if self.rows:
            self.empty_label.pack_forget()
        else:
            self.empty_label.pack(anchor="w", pady=(0, 6))

    def to_condition_list(self) -> tuple[list[dict], list[str]]:
        session_start, session_end = self._get_session()
        conditions = [r.to_condition(session_start, session_end) for r in self.rows]
        connectors = [r.connector_var.get() for r in self.rows[1:]]
        return conditions, connectors

    def set_from_conditions(self, conditions: list[dict], connectors: list[str] | None = None) -> None:
        """
        Clears this list and rebuilds it from a list of condition dicts (the
        same shape to_condition_list() produces) plus their AND/OR
        connectors. Used by the "Apply Best Configuration" action after an
        Iterative Refinement run, and generally to load any saved config.
        """
        connectors = connectors or []
        while self.rows:
            self._remove_row(self.rows[0])
        for i, condition in enumerate(conditions or []):
            self.add_row()
            row = self.rows[-1]
            connector = connectors[i - 1] if i > 0 and (i - 1) < len(connectors) else "AND"
            row.load_from(condition, connector if i > 0 else None)
