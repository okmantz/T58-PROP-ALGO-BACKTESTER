// T58_SL_PIPS=20
// T58_TP_PIPS=34

#include <Trade/Trade.mqh>

CTrade trade;

void OnTick()
{
   double fastMA = iMA(
      _Symbol,
      PERIOD_CURRENT,
      21,
      0,
      MODE_EMA,
      PRICE_CLOSE
   );

   double slowMA = iMA(
      _Symbol,
      PERIOD_CURRENT,
      55,
      0,
      MODE_EMA,
      PRICE_CLOSE
   );

   double rsiVal = iRSI(
      _Symbol,
      PERIOD_CURRENT,
      14,
      PRICE_CLOSE
   );

   // Long-side invalidation.
   if (fastMA < slowMA || rsiVal < 48)
   {
      trade.PositionClose(_Symbol);
   }

   // Short-side invalidation.
   if (fastMA > slowMA || rsiVal > 52)
   {
      trade.PositionClose(_Symbol);
   }

   // Long momentum window.
   if (fastMA > slowMA && rsiVal > 55 && rsiVal < 68)
   {
      trade.Buy(0.10, _Symbol);
   }

   // Short momentum window.
   if (fastMA < slowMA && rsiVal < 45 && rsiVal > 32)
   {
      trade.Sell(0.10, _Symbol);
   }
}
