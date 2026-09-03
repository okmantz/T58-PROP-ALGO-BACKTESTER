// NOTE (2026-09-03 switched to ATR-mult stops): the fixed-pip SL/TP this file used to carry
// went through a manual "~25x for gold" rescale and STILL blew the account on a real GC1! 1-min
// run: the guess assumed pip_size ~0.01, but that run was actually configured with pip_size=1.0,
// so the 550/1250 "pip" SL/TP resolved to a $550/$1250 stop/target on an instrument trading in
// the low thousands -- a 20%+ per-trade risk that triggered the account-survivability floor
// almost immediately (see the Full Pipeline batch log, "Account BLOWN" + "fixed-pips stop ...
// implausible fraction of price" warnings). A fixed pip/point count is fundamentally fragile:
// it's only ever correct for the one pip_size/instrument it was tuned at. Switched to
// T58_SL_ATR_MULT/T58_TP_ATR_MULT below instead -- these compute the stop/target as a multiple
// of the instrument's OWN actual ATR at backtest time, in raw price units, so they're correct on
// gold, an FX pair, an index, or crypto without ever touching pip_size at all. See
// app/strategy/mql5.py's module docstring for how these are parsed.
// T58 PROP STRATEGY 03 — Momentum Regime
// Designed to participate only when trend and momentum agree.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_ATR_MULT=1.5
// T58_TP_ATR_MULT=3.0
// T58_ATR_PERIOD=14

double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_EMA, PRICE_CLOSE);
double midMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
double trendMA = iMA(_Symbol, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

if (fastMA > midMA && midMA > trendMA && close > fastMA && rsiVal >= 55 && rsiVal <= 72) {
    trade.Buy(0.10, _Symbol);
}

if (fastMA < midMA && midMA < trendMA && close < fastMA && rsiVal >= 28 && rsiVal <= 45) {
    trade.Sell(0.10, _Symbol);
}

// FIX (2026-08-31): the previous exit rules were
//   if (close < midMA || rsiVal < 50) trade.PositionClose(_Symbol);
//   if (close > midMA || rsiVal > 50) trade.PositionClose(_Symbol);
// which are each true on very nearly every single tick (rsiVal is on
// whichever side of 50 essentially always), so this parser's per-tick
// STATE evaluation (not an event) closed any position practically the
// bar after it opened, regardless of the trend/momentum regime that
// justified the entry. That alone is enough to explain a near-zero
// trade count with the position never given room to work. Replaced
// with an actual trend-loss exit: close only once price has crossed
// back through the fast MA against the position, which is the same
// invalidation logic the entry filters already use to define "still in
// regime."
if (close < fastMA && fastMA < midMA) {
    trade.PositionClose(_Symbol);
}

if (close > fastMA && fastMA > midMA) {
    trade.PositionClose(_Symbol);
}
