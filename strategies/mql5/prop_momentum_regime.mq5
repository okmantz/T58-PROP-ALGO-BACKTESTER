//+------------------------------------------------------------------+
//|                                     prop_momentum_regime.mq5      |
//|                                  Prop Firm Compliant EA Logic     |
//+------------------------------------------------------------------+
#property copyright "Prop Strategy"
#property version   "1.00"

// T58_SL_PIPS=30
// T58_TP_PIPS=60

void OnTick()
{
    // Indicator calls conforming to parser specifications
    double fastEMA = iMA(_Symbol, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
    double slowEMA = iMA(_Symbol, PERIOD_CURRENT, 89, 0, MODE_EMA, PRICE_CLOSE);
    double rsiVal  = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

    // Regime definitions
    bool bullTrend = (fastEMA > slowEMA);
    bool bearTrend = (fastEMA < slowEMA);

    // Entry triggers: Entering on directional momentum inside structural trend
    bool longCondition  = bullTrend && (rsiVal > 52.0 && rsiVal < 65.0);
    bool shortCondition = bearTrend && (rsiVal < 48.0 && rsiVal > 35.0);

    // Dynamic exit / structure invalidation
    bool exitLong  = (rsiVal > 78.0) || (fastEMA < slowEMA);
    bool exitShort = (rsiVal < 22.0) || (fastEMA > slowEMA);

    // Order dispatch
    if (longCondition)
    {
        trade.Buy(0.1, _Symbol);
    }
    else if (shortCondition)
    {
        trade.Sell(0.1, _Symbol);
    }

    if (exitLong)
    {
        trade.PositionClose(_Symbol);
    }
    else if (exitShort)
    {
        trade.PositionClose(_Symbol);
    }
}
