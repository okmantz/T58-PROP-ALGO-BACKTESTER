//+------------------------------------------------------------------+
//|                                Triple EMA Alignment Momentum.mq5  |
//+------------------------------------------------------------------+
// DESIGN NOTE (read this before running):
// This is a from-scratch design, built specifically to work within T58's
// actual MQL5 parser subset (see app/strategy/mql5.py) -- only direct-
// value iMA()/iRSI() calls and boolean comparisons over already-defined
// variables are supported. No CopyBuffer handles, no custom indicators,
// no arrays, no crossover() helper (MQL5's iMA shift argument is parsed
// but not actually used -- there is no "previous bar's MA" available
// here). Every one of the 7 uploaded MQL5 files reviewed earlier failed
// to load for exactly this reason; this one is written to actually run
// tonight.
//
// MECHANISM: deliberately different from the PineScript strategy shipped
// alongside this file. Where that one fades a pullback WITHIN a trend
// (counter-trend entries, RSI extremes), this one is pure momentum
// CONTINUATION: it only trades when three EMAs are FULLY aligned (fast >
// mid > slow, or the mirror) AND RSI confirms the same-direction
// momentum is still building (above/below 50, but not yet past 75/25 --
// avoids chasing an already-exhausted move). It buys/sells STRENGTH, not
// weakness, and requires all three trend timeframes to agree before it
// will act at all -- a more conservative filter than a simple two-MA
// cross. There is no separate exit rule: a position is held until the
// fixed stop/target fires, or the opposite alignment fires and reverses
// it (T58's standard long/flat/short signal model does this
// automatically for every strategy source).
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// T58_SL_PIPS=30
// T58_TP_PIPS=60

int OnInit()
{
   trade.SetExpertMagicNumber(583920);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   double emaFast = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_EMA, PRICE_CLOSE);
   double emaMid  = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
   double emaSlow = iMA(_Symbol, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
   double rsiVal  = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

   bool bullAligned = emaFast > emaMid && emaMid > emaSlow;
   bool bearAligned = emaFast < emaMid && emaMid < emaSlow;
   bool momentumUp = rsiVal > 50 && rsiVal < 75;
   bool momentumDown = rsiVal < 50 && rsiVal > 25;

   if (bullAligned && momentumUp)
   {
      trade.Buy(0.1);
   }
   if (bearAligned && momentumDown)
   {
      trade.Sell(0.1);
   }
}
