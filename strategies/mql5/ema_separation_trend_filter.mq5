//+------------------------------------------------------------------+
//|                       EMA Separation Trend Filter.mq5             |
//+------------------------------------------------------------------+
// NEW STRATEGY (built from scratch, 2026-08-30) -- trend-following
// continuation, but with a materially different filter from the deleted
// triple_ema_alignment_momentum.mq5: instead of asking "are 3 EMAs in
// ordinal order" (a condition that a razor-thin, noisy separation can
// satisfy just as easily as a real trend), this measures the PERCENTAGE
// separation between two EMAs and only trades once that separation
// clears a real threshold -- filtering out the marginal/choppy "aligned
// but barely" cases that plausibly explain some of the old file's -$50k
// result. Percentage-based, so the threshold means the same thing on
// AAPL, XAUUSD, or EURUSD without any rescaling.
//
// Same parser subset restriction as before applies: only direct-value
// iMA()/iRSI() calls, plain C-style comparisons/arithmetic over already-
// defined doubles, and trade.Buy()/trade.Sell(). No arrays, no
// CopyBuffer, no previous-bar access -- so, like the file it replaces,
// this evaluates a per-tick STATE, not a crossover EVENT (see
// app/strategy/mql5.py).
//
// MECHANISM: trend direction and strength come from the normalized gap
// between a fast and slow EMA; RSI confirms momentum is actually active
// but not yet exhausted (avoids buying/selling into an already-stretched
// move). No separate exit rule -- the position closes on the opposite-
// direction signal, or on the fixed stop/target below (T58's standard
// long/flat/short signal model).
//
// NOT YET VALIDATED: this has not been run through T58's Full Pipeline.
// No GA search, no walk-forward, no Monte Carlo, no significance gate --
// treat every constant below as a starting point for the GA to refine,
// not a finished, tested setting.
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// T58_SL_PIPS=40
// T58_TP_PIPS=80

int OnInit()
{
   trade.SetExpertMagicNumber(583921);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   double emaFast = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
   double emaSlow = iMA(_Symbol, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
   double rsiVal  = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

   double trendStrengthPct = (emaFast - emaSlow) / emaSlow;

   bool bullTrend = trendStrengthPct > 0.003;    // fast EMA >0.3% above slow EMA
   bool bearTrend = trendStrengthPct < -0.003;   // fast EMA >0.3% below slow EMA
   bool momentumUp = rsiVal > 55 && rsiVal < 75;
   bool momentumDown = rsiVal < 45 && rsiVal > 25;

   if (bullTrend && momentumUp)
   {
      trade.Buy(0.1);
   }
   if (bearTrend && momentumDown)
   {
      trade.Sell(0.1);
   }
}
