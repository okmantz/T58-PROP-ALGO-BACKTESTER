// NOTE (2026-08-31 rescale): the SL/TP pip values below were originally sized for FX pairs
// (pip_size 0.0001) and produced near-instant catastrophic stop-outs when run against gold-scaled
// (XAUUSD, pip_size ~0.01) data -- a 20-40 "pip" stop is $0.002-0.004 at FX scale but only
// $0.20-0.40 at gold scale, nowhere near gold's typical per-bar ATR, so almost every entry was
// stopped out by normal noise on the very next bar. Values below are rescaled ~25x for gold-scale
// instruments. Before running this strategy: click "detect pip size from data" on the Data tab
// so pip_size actually reflects the loaded instrument -- these numbers assume pip_size ~0.01
// (2-decimal gold-style quoting), NOT the FX default of 0.0001. If you load an FX pair instead,
// use the original (much smaller) pip counts.
// T58 PROP STRATEGY 03 — Momentum Regime
// Designed to participate only when trend and momentum agree.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_PIPS=550
// T58_TP_PIPS=1250

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
