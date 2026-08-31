// T58 PROP STRATEGY 03 — Momentum Regime
// Designed to participate only when trend and momentum agree.
// Uses only constructs supported by the T58 MQL5 parser.
//
// T58_SL_PIPS=6.6
// T58_TP_PIPS=54.067375

double fastMA = iMA(_Symbol, PERIOD_CURRENT, 3, 0, MODE_EMA, PRICE_CLOSE);
double midMA = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
double trendMA = iMA(_Symbol, PERIOD_CURRENT, 90, 0, MODE_EMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 30, PRICE_CLOSE);

if (fastMA > midMA && midMA > trendMA && close > fastMA && rsiVal >= 55 && rsiVal <= 72) {
    trade.Buy(0.10, _Symbol);
}

if (fastMA < midMA && midMA < trendMA && close < fastMA && rsiVal >= 28 && rsiVal <= 45) {
    trade.Sell(0.10, _Symbol);
}

if (close < midMA || rsiVal < 50) {
    trade.PositionClose(_Symbol);
}

if (close > midMA || rsiVal > 50) {
    trade.PositionClose(_Symbol);
}
