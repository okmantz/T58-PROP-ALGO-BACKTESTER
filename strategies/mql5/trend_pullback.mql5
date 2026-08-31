// T58 PROP STRATEGY 01 — Trend Pullback
// Designed for high evaluation-pass probability with controlled frequency.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_PIPS=20
// T58_TP_PIPS=40

double fastMA = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
double slowMA = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

if (fastMA > slowMA && close > fastMA && rsiVal >= 52 && rsiVal <= 68) {
    trade.Buy(0.10, _Symbol);
}

if (fastMA < slowMA && close < fastMA && rsiVal >= 32 && rsiVal <= 48) {
    trade.Sell(0.10, _Symbol);
}

if (close < fastMA || rsiVal < 48) {
    trade.PositionClose(_Symbol);
}

if (close > fastMA || rsiVal > 52) {
    trade.PositionClose(_Symbol);
}
