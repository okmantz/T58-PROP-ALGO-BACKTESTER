"""
Declarative, config-driven adaptive risk rules.

No strategy source this app supports (Manual, Python, PineScript, MQL5)
can see its own realized trade outcomes at signal-generation time --
generate_signals() / ManualStrategy.generate() run once, statelessly, over
the whole price series before any P&L exists. Money-management behavior
that depends on the account's own trading history -- "cut size after 2
losses in a row", "coast once 80% of the way to the profit target",
"de-risk once today's realized P&L is down X%" -- is the same category of
gap that RiskConfig.daily_loss_limit_pct was added to close for a hard
circuit breaker: it cannot be expressed inside a strategy's own signal
logic, no matter which source type, so it has to be a first-class ENGINE
feature instead. This module makes that possible for graduated,
size-scaling money management (not just a binary stop-trading breaker),
so any strategy can be backtested with or without a given adaptive-risk
overlay applied on top, without changing a single line of the strategy
itself.

Rules are declarative and stack MULTIPLICATIVELY: every rule whose
condition is currently true contributes a position-size multiplier for
the NEXT entry, and the multiplier actually applied is the PRODUCT of
every active rule's multiplier. This keeps each rule simple and
independent -- no rule needs to know about any other rule -- while still
letting several de-risking conditions compound, the way a real trader
stacking "I'm down today" with "I've lost twice in a row" would actually
behave more cautiously than either alone.

Multipliers only ever affect the SIZE of new entries. They never touch
stop-loss/take-profit placement, and they are evaluated only at the
moment a new position is opened (this engine trades one position at a
time -- see app.backtest.execution's module docstring -- so there is
never a "resize an open position" case to handle).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_VALID_TRIGGERS = {"consecutive_losses", "daily_loss_pct", "daily_profit_pct", "progress_to_target_pct", "drawdown_pct"}


class AdaptiveRiskError(Exception):
    """Raised for an invalid adaptive-risk rule or config."""


@dataclass
class AdaptiveRiskRule:
    trigger: str
    # "consecutive_losses"     -- threshold = number of losing trades in a row (resets to 0 on any winning trade)
    # "daily_loss_pct"         -- threshold = % of initial_balance realized-lost so far TODAY
    # "daily_profit_pct"       -- threshold = % of initial_balance realized-gained so far TODAY
    # "progress_to_target_pct" -- threshold = % of the way from 0 to profit_target_amount the account has reached
    #                             so far, ALL-TIME (not reset daily) -- requires
    #                             AdaptiveRiskConfig.profit_target_amount to be set; a rule using this trigger is
    #                             simply never active (multiplier 1.0) if that amount isn't supplied.
    threshold: float
    # "drawdown_pct"           -- threshold = % of initial_balance the account is currently down from its
    #                             own realized-equity PEAK (all-time, not reset daily) -- this is the trigger
    #                             that targets a prop firm's overall max-drawdown floor specifically, as
    #                             opposed to daily_loss_pct which only looks at today.
    risk_multiplier: float   # position-size multiplier for new entries while this rule is active (e.g. 0.5 = half size)
    label: str = ""          # optional human-readable name, surfaced in reports/UI only

    def __post_init__(self):
        if self.trigger not in _VALID_TRIGGERS:
            raise AdaptiveRiskError(
                f"Unknown adaptive risk trigger '{self.trigger}'. Supported: {sorted(_VALID_TRIGGERS)}"
            )
        self.threshold = float(self.threshold)
        self.risk_multiplier = max(float(self.risk_multiplier), 0.0)
        if not self.label:
            self.label = f"{self.trigger} >= {self.threshold} -> x{self.risk_multiplier}"


@dataclass
class AdaptiveRiskConfig:
    enabled: bool = False
    rules: list = field(default_factory=list)   # list[AdaptiveRiskRule]
    profit_target_amount: float | None = None   # required only by "progress_to_target_pct" rules; in account
    # currency (e.g. initial_balance * evaluation_profit_target_pct / 100 -- the caller computes this from
    # whatever PropRules/target the backtest is being run against, since this module has no PropRules dependency).

    def __post_init__(self):
        if self.rules and not all(isinstance(r, AdaptiveRiskRule) for r in self.rules):
            self.rules = [r if isinstance(r, AdaptiveRiskRule) else AdaptiveRiskRule(**r) for r in self.rules]


@dataclass
class AdaptiveRiskState:
    """Mutable per-backtest bookkeeping the execution loop updates as
    trades close. Deliberately never exposed to a Strategy -- only
    app.backtest.execution.run_execution reads/writes this, which is the
    entire point: the strategy's own signal logic still can't see it."""
    initial_balance: float = 0.0
    consecutive_losses: int = 0
    day_realized_pnl: float = 0.0
    cumulative_realized_pnl: float = 0.0
    peak_realized_balance: float = 0.0   # highest realized-equity level reached so far; set to initial_balance on first use

    def __post_init__(self):
        if self.peak_realized_balance <= 0:
            self.peak_realized_balance = self.initial_balance

    def record_trade_close(self, pnl: float, is_new_day: bool) -> None:
        if is_new_day:
            self.day_realized_pnl = 0.0
        self.day_realized_pnl += pnl
        self.cumulative_realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
        # a scratch trade (pnl == 0) leaves the streak counter unchanged
        current_balance = self.initial_balance + self.cumulative_realized_pnl
        self.peak_realized_balance = max(self.peak_realized_balance, current_balance)

    def current_drawdown_pct(self) -> float:
        """% of initial_balance the account is currently down from its own
        realized-equity peak -- a realized-P&L proxy for the same
        "how close is this account to its overall drawdown floor" question
        PropRules.max_drawdown_pct answers, evaluated at every ENTRY
        decision using only outcomes already realized as of that bar (no
        lookahead, same convention as every other adaptive-risk trigger)."""
        if not self.initial_balance:
            return 0.0
        peak = self.peak_realized_balance or self.initial_balance
        current_balance = self.initial_balance + self.cumulative_realized_pnl
        return max(0.0, (peak - current_balance) / self.initial_balance * 100.0)

    def _rule_active(self, rule: AdaptiveRiskRule, profit_target_amount: float | None) -> bool:
        if rule.trigger == "consecutive_losses":
            return self.consecutive_losses >= rule.threshold
        if rule.trigger == "daily_loss_pct":
            loss_pct = (-self.day_realized_pnl / self.initial_balance * 100.0) if self.initial_balance else 0.0
            return loss_pct >= rule.threshold
        if rule.trigger == "daily_profit_pct":
            profit_pct = (self.day_realized_pnl / self.initial_balance * 100.0) if self.initial_balance else 0.0
            return profit_pct >= rule.threshold
        if rule.trigger == "progress_to_target_pct":
            if not profit_target_amount:
                return False
            progress_pct = self.cumulative_realized_pnl / profit_target_amount * 100.0
            return progress_pct >= rule.threshold
        if rule.trigger == "drawdown_pct":
            return self.current_drawdown_pct() >= rule.threshold
        return False

    def active_multiplier(self, config: AdaptiveRiskConfig) -> float:
        """The product of every currently-triggered rule's multiplier --
        1.0 (no de-risking) if this config is disabled or no rule fires."""
        if not config.enabled or not config.rules:
            return 1.0
        mult = 1.0
        for rule in config.rules:
            if self._rule_active(rule, config.profit_target_amount):
                mult *= rule.risk_multiplier
        return mult

    def active_rule_labels(self, config: AdaptiveRiskConfig) -> list[str]:
        """Which rules are currently firing -- used to annotate a trade
        with WHY its size was scaled, for report transparency."""
        if not config.enabled:
            return []
        return [rule.label for rule in config.rules if self._rule_active(rule, config.profit_target_amount)]

def build_limit_aware_preset(
    prop_rules,
    daily_profit_lock_pct: float | None = 80.0,
    account_size: float | None = None,
) -> AdaptiveRiskConfig:
    """One-click "risk throttling that targets THIS metric" preset,
    built directly from a strategy's own PropRules instead of asking a
    person to hand-derive percentages off the firm's daily-loss/max-
    drawdown limits themselves.

    This is deliberately graduated (several rules stacking multiplicatively,
    same mechanism as any hand-built AdaptiveRiskConfig -- see this
    module's docstring) rather than one hard cutoff at the limit itself:
    cutting size in HALF once an account is already halfway to a limit,
    then to a QUARTER close to it, buys more room to recover from a
    losing streak before the hard circuit breaker
    (RiskConfig.daily_loss_limit_pct / max_account_drawdown_pct) would
    otherwise end the day or the whole account outright. None of this
    changes a strategy's edge -- it changes how that edge's risk is
    survived, which is exactly what "reach the target before hitting a
    limit" as an objective calls for independent of the edge itself.

    daily_profit_lock_pct: once today's realized profit reaches this %
    of the firm's daily loss limit's ABSOLUTE dollar size (a proportional,
    firm-size-relative stand-in for "a good day, stop pushing it" -- many
    firms explicitly allow banking a day's gains rather than risking them
    back), new size is zeroed for the rest of that trading day. None
    disables this specific rule (the drawdown/daily-loss throttling rules
    still apply).

    account_size: overrides prop_rules.account_size for the
    profit_target_amount used by any progress_to_target_pct rule a
    caller adds on top of this preset -- not used by this preset itself,
    exposed only so callers can share one account size across both.
    """
    rules = [
        # Daily-loss floor: prop_rules.daily_loss_limit_pct is the hard
        # circuit breaker (RiskConfig.daily_loss_limit_pct stops ALL new
        # entries once crossed) -- these throttle size DOWN on the way
        # there, so a losing streak is less likely to reach that floor
        # at full size in the first place.
        AdaptiveRiskRule(
            trigger="daily_loss_pct", threshold=prop_rules.daily_loss_limit_pct * 0.5,
            risk_multiplier=0.5, label="Halved: 50% of the way to today's daily-loss limit",
        ),
        AdaptiveRiskRule(
            trigger="daily_loss_pct", threshold=prop_rules.daily_loss_limit_pct * 0.75,
            risk_multiplier=0.5, label="Quartered: 75% of the way to today's daily-loss limit",
        ),
        # Overall drawdown floor: same graduated idea, against
        # prop_rules.max_drawdown_pct instead of the daily limit.
        AdaptiveRiskRule(
            trigger="drawdown_pct", threshold=prop_rules.max_drawdown_pct * 0.5,
            risk_multiplier=0.5, label="Halved: 50% of the way to the overall max-drawdown limit",
        ),
        AdaptiveRiskRule(
            trigger="drawdown_pct", threshold=prop_rules.max_drawdown_pct * 0.75,
            risk_multiplier=0.5, label="Quartered: 75% of the way to the overall max-drawdown limit",
        ),
    ]
    if daily_profit_lock_pct is not None:
        lock_threshold_pct_of_balance = prop_rules.daily_loss_limit_pct * (daily_profit_lock_pct / 100.0)
        rules.append(AdaptiveRiskRule(
            trigger="daily_profit_pct", threshold=lock_threshold_pct_of_balance,
            risk_multiplier=0.0,
            label=f"Locked for the day: banked profit >= {daily_profit_lock_pct:g}% of the daily-loss limit's size",
        ))
    return AdaptiveRiskConfig(enabled=True, rules=rules)
