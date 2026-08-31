// T58 PROP STRATEGY 02 — RSI Mean Reversion
// Designed for high win-rate behavior and smaller, repeatable moves.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_PIPS=5.4
// T58_TP_PIPS=72.415268

double meanMA = iMA(_Symbol, PERIOD_CURRENT, 25, 0, MODE_SMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 4, PRICE_CLOSE);

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
