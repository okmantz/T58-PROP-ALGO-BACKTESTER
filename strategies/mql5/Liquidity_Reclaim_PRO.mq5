// NOTE (2026-09-03, rewritten for the T58 parser): the previous version of this file was a
// full 1,252-line MQL5 Expert Advisor (CTrade object usage, a `t == lastBar` new-bar guard,
// custom structs/state, session windows, sweep/FVG price-action detection, a daily-risk state
// machine). None of that is inside the T58 MQL5 adapter's supported subset (see
// app/strategy/mql5.py's module docstring) -- like the PineScript adapter, this is a
// line-based parser for a small, deliberate set of constructs, not a full MQL5 implementation.
// Trying to load the old file raised:
//   StrategyError: Failed to evaluate expression 't == lastBar' (if-condition): name 't' is not defined
// This rewrite keeps the same underlying idea -- trade a reclaim back through value in the
// direction of trend, after price has stretched away from it -- using only what the parser
// actually understands: iMA/iRSI, boolean comparisons, and +-*/ arithmetic over previously
// defined variables. The MQL5 parser has no crossover/crossunder primitive at all (unlike the
// PineScript adapter), so the "just reclaimed" moment is approximated with a narrow band on
// price's %-distance from the fast MA instead of a one-bar crossover event -- this is the same
// per-bar state-condition architecture momentum_regime.mq5 already uses successfully in this
// repo. What could NOT be preserved because the parser has no equivalent at all: session-time
// windows, liquidity-sweep/FVG detection, and the daily trade/loss-limit state machine (use
// RiskConfig.daily_loss_limit_pct / max_trades_per_day in the app's own risk settings for
// that, same as every other strategy in this repo). If you want the FULL sweep/FVG/session-
// state design preserved exactly, that needs to be a Python strategy (app/strategy/python.py
// runs real Python, no regex subset limit) -- Winning_Liquidity_Reclaim.py is that version.
// T58 PROP STRATEGY — Liquidity Reclaim (Simplified)
// Trades a reclaim back through the fast MA, with trend and RSI-recovery filters.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_ATR_MULT=1.25
// T58_TP_ATR_MULT=2.0
// T58_ATR_PERIOD=14

double fastMA = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
double slowMA = iMA(_Symbol, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

// %-distance of price above/below the fast MA -- a narrow positive band just
// after crossing up (or negative band just after crossing down) stands in
// for the one-bar "reclaim" event this parser can't express directly.
double priceVsFastPct = (close - fastMA) / fastMA;

if (fastMA > slowMA && priceVsFastPct > 0.0 && priceVsFastPct < 0.004 && rsiVal >= 28 && rsiVal <= 48) {
    trade.Buy(0.10, _Symbol);
}

if (fastMA < slowMA && priceVsFastPct < 0.0 && priceVsFastPct > -0.004 && rsiVal >= 52 && rsiVal <= 72) {
    trade.Sell(0.10, _Symbol);
}

// Exit once price has fully lost the fast MA against the position (same
// trend-loss exit logic momentum_regime.mq5 uses, for the same reason:
// this parser evaluates conditions as per-bar state, not one-off events,
// so the exit must be a clearly-opposite condition or it closes trades the
// bar after they open).
if (close < fastMA && fastMA < slowMA) {
    trade.PositionClose(_Symbol);
}

if (close > fastMA && fastMA > slowMA) {
    trade.PositionClose(_Symbol);
}
