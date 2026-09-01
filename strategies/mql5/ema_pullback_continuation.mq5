// T58 PROP STRATEGY — EMA Pullback Continuation (new, 2026-08-31)
//
// Deliberately a different entry SHAPE from the two existing MQL5
// strategies in this folder, not just different constants:
//   - momentum_regime.mq5 trades WITH already-running momentum (RSI
//     55-72 / 28-45 -- momentum already extended).
//   - ema_separation_trend_filter.mq5 trades once EMA separation clears
//     a threshold, with no requirement on WHERE price currently is
//     relative to that trend.
//   - This one buys/sells the RETRACEMENT into an established trend:
//     price must be pulled back close to a fast EMA (not chasing an
//     extended move) with RSI back in a neutral 40-60 zone (the
//     signature of a genuine pullback, not a reversal or an already-
//     exhausted move).
//
// IMPORTANT design point found while validating this file (also fixed
// in momentum_regime.mq5 in this same pass): the trend-regime filter
// deliberately does NOT use the same fast EMA that "pullback proximity"
// and RSI-neutral are measured against. An earlier draft defined the
// uptrend/downtrend regime off the fast/slow EMA pair -- but the fast
// EMA reacts to the exact same short-term pullback that pushes RSI back
// to neutral and price back near the fast EMA, so by the time all three
// conditions were true, the fast EMA had already been pulled toward the
// slow EMA and the "trend" condition measured off that same fast EMA
// had usually already weakened or flipped -- the three conditions
// fought each other and essentially never coincided. Fixed by defining
// the trend regime off a SLOWER, more stable mid/slow EMA pair instead,
// decoupled from the fast EMA used for pullback proximity -- the same
// insight applied to strategies/pinescript/ema_ribbon_rsi_reversal.pine
// in this same delivery.
//
// Same T58 MQL5 parser subset restriction as every other strategy here:
// only direct-value iMA()/iRSI() calls, C-style comparisons/arithmetic
// over already-defined doubles, and trade.Buy()/Sell()/PositionClose().
// No arrays, no CopyBuffer, no previous-bar access -- this evaluates a
// per-tick STATE, not a crossover EVENT (see app/strategy/mql5.py).
//
// SL/TP sizing: gold-scale (pip_size ~0.01), NOT the FX default of
// 0.0001 -- click "detect pip size from data" on the Data tab before
// running this. See the rescale note added to the other mql5/pinescript
// files in this folder: the original FX-style 15-40 "pip" stops in
// those files are $0.20-0.40 at gold scale, nowhere near gold's
// per-bar ATR, so they stopped out almost instantly (1-2 trades,
// catastrophic loss) when actually run against real gold data. The
// 700/1400 values below are a reasonable starting point at gold scale,
// not a validated setting -- widen or narrow via Search Lab / Iterative
// Refinement once this has a real baseline.
//
// NOT YET VALIDATED: run through Full Pipeline before trusting any
// constant below -- these are GA starting points, not finished settings.
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// T58_SL_PIPS=700
// T58_TP_PIPS=1400

int OnInit()
{
   trade.SetExpertMagicNumber(583944);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   double emaFast = iMA(_Symbol, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   double emaMid  = iMA(_Symbol, PERIOD_CURRENT, 55, 0, MODE_EMA, PRICE_CLOSE);
   double emaSlow = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);
   double rsiVal  = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);

   // Trend regime: slower mid/slow pair only -- stays stable through the
   // same short-term pullback that pulls RSI back to neutral and price
   // back near the fast EMA, so this and the entry trigger below measure
   // genuinely different things instead of fighting each other.
   double trendPct = (emaMid - emaSlow) / emaSlow;
   bool uptrend = trendPct > 0.0020;
   bool downtrend = trendPct < -0.0020;

   // Entry trigger: price back near the fast EMA (a real pullback, not
   // an extended chase) with RSI back in a neutral zone.
   double pullbackPct = (close - emaFast) / emaFast;
   bool pulledBackToFast = pullbackPct > -0.0020 && pullbackPct < 0.0020;
   bool rsiNeutral = rsiVal > 38 && rsiVal < 62;

   if (uptrend && pulledBackToFast && rsiNeutral && close > open)
   {
      trade.Buy(0.10, _Symbol);
   }

   if (downtrend && pulledBackToFast && rsiNeutral && close < open)
   {
      trade.Sell(0.10, _Symbol);
   }

   // Exit once price loses the fast EMA against the position -- the
   // pullback that was bought/sold has failed to reclaim, which is a
   // cleaner invalidation than waiting for the slower trend filter
   // itself to flip.
   if (close < emaFast && emaFast < emaMid)
   {
      trade.PositionClose(_Symbol);
   }

   if (close > emaFast && emaFast > emaMid)
   {
      trade.PositionClose(_Symbol);
   }
}
