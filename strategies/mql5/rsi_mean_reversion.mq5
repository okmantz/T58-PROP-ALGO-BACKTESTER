// NOTE (2026-08-31 rescale): the SL/TP pip values below were originally sized for FX pairs
// (pip_size 0.0001) and produced near-instant catastrophic stop-outs when run against gold-scaled
// (XAUUSD, pip_size ~0.01) data -- a 20-40 "pip" stop is $0.002-0.004 at FX scale but only
// $0.20-0.40 at gold scale, nowhere near gold's typical per-bar ATR, so almost every entry was
// stopped out by normal noise on the very next bar. Values below are rescaled ~25x for gold-scale
// instruments. Before running this strategy: click "detect pip size from data" on the Data tab
// so pip_size actually reflects the loaded instrument -- these numbers assume pip_size ~0.01
// (2-decimal gold-style quoting), NOT the FX default of 0.0001. If you load an FX pair instead,
// use the original (much smaller) pip counts.
// T58 PROP STRATEGY 02 — RSI Mean Reversion
// Designed for high win-rate behavior and smaller, repeatable moves.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_PIPS=450
// T58_TP_PIPS=750

double meanMA = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

if (close < meanMA && rsiVal <= 30) {
    trade.Buy(0.10, _Symbol);
}

if (close > meanMA && rsiVal >= 70) {
    trade.Sell(0.10, _Symbol);
}

if (close >= meanMA || rsiVal >= 50) {
    trade.PositionClose(_Symbol);
}

if (close <= meanMA || rsiVal <= 50) {
    trade.PositionClose(_Symbol);
}
