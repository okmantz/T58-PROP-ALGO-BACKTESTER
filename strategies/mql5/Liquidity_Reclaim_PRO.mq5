//+------------------------------------------------------------------+
//| Regime-Gated Liquidity Reclaim PRO                              |
//| Prop-Firm-Oriented Expert Advisor                               |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>

CTrade trade;

//==================================================================
// INPUTS
//==================================================================

input int ATR_PERIOD = 20;

input int FAST_EMA = 50;
input int SLOW_EMA = 200;

input int LIQUIDITY_LOOKBACK = 36;

input int FVG_EXPIRY_BARS = 8;
input int SWEEP_RECENCY_BARS = 12;

input double MIN_DISPLACEMENT_ATR = 0.80;
input double MIN_CLOSE_LOCATION = 0.68;

input double MIN_ATR_RATIO = 0.80;
input double MAX_ATR_RATIO = 1.75;

input double STOP_BUFFER_ATR = 0.12;
input double MAX_STOP_ATR = 1.50;

input double TARGET_R = 1.40;
input double BREAKEVEN_R = 0.75;

input int MAX_HOLD_BARS = 16;

input int MAX_TRADES_PER_DAY = 3;
input int MAX_LOSSES_PER_DAY = 2;

input double MAX_DAILY_LOSS_R = 1.0;
input double DAILY_PROFIT_LOCK_R = 1.50;

input int MAX_CONSECUTIVE_LOSSES = 2;

input int COOLDOWN_BARS = 6;
input int LOSS_COOLDOWN_BARS = 12;

input double RISK_PERCENT = 0.25;

//==================================================================
// INDICATOR HANDLES
//==================================================================

int atrHandle;
int fastHandle;
int slowHandle;

//==================================================================
// STATE
//==================================================================

datetime lastBar = 0;

int tradesToday = 0;
int lossesToday = 0;
int consecutiveLosses = 0;

double dailyR = 0.0;

int cooldown = 0;

int barsHeld = 0;

double entryPrice = 0.0;
double stopPrice = 0.0;
double targetPrice = 0.0;
double riskDistance = 0.0;

datetime currentDay = 0;

//==================================================================
// INITIALIZATION
//==================================================================

int OnInit()
{
   atrHandle = iATR(
      _Symbol,
      PERIOD_CURRENT,
      ATR_PERIOD
   );

   fastHandle = iMA(
      _Symbol,
      PERIOD_CURRENT,
      FAST_EMA,
      0,
      MODE_EMA,
      PRICE_CLOSE
   );

   slowHandle = iMA(
      _Symbol,
      PERIOD_CURRENT,
      SLOW_EMA,
      0,
      MODE_EMA,
      PRICE_CLOSE
   );

   if(
      atrHandle == INVALID_HANDLE ||
      fastHandle == INVALID_HANDLE ||
      slowHandle == INVALID_HANDLE
   )
   {
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber(58001);

   return INIT_SUCCEEDED;
}

//==================================================================
// NEW BAR
//==================================================================

bool IsNewBar()
{
   datetime t = iTime(
      _Symbol,
      PERIOD_CURRENT,
      0
   );

   if(t == lastBar)
      return false;

   lastBar = t;

   return true;
}

//==================================================================
// NEW DAY
//==================================================================

void ResetDailyState()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   datetime dayStart =
      StructToTime(dt)
      -
      dt.hour * 3600
      -
      dt.min * 60
      -
      dt.sec;

   if(dayStart != currentDay)
   {
      currentDay = dayStart;

      tradesToday = 0;
      lossesToday = 0;
      consecutiveLosses = 0;

      dailyR = 0.0;
   }
}

//==================================================================
// SESSION
//==================================================================

bool InTradingSession()
{
   MqlDateTime dt;

   TimeToStruct(
      TimeCurrent(),
      dt
   );

   int minutes =
      dt.hour * 60 +
      dt.min;

   bool session1 =
      minutes >= 930 &&
      minutes <= 1130;

   bool session2 =
      minutes >= 1330 &&
      minutes <= 1530;

   return session1 || session2;
}

//==================================================================
// ATR
//==================================================================

double GetATR(int shift)
{
   double buffer[];

   ArraySetAsSeries(
      buffer,
      true
   );

   if(
      CopyBuffer(
         atrHandle,
         0,
         shift,
         1,
         buffer
      ) <= 0
   )
      return 0.0;

   return buffer[0];
}

//==================================================================
// EMA
//==================================================================

double GetEMA(
   int handle,
   int shift
)
{
   double buffer[];

   ArraySetAsSeries(
      buffer,
      true
   );

   if(
      CopyBuffer(
         handle,
         0,
         shift,
         1,
         buffer
      ) <= 0
   )
      return 0.0;

   return buffer[0];
}

//==================================================================
// LIQUIDITY HIGH
//==================================================================

double PriorHigh()
{
   double highest = -DBL_MAX;

   for(
      int i = 2;
      i <= LIQUIDITY_LOOKBACK + 1;
      i++
   )
   {
      double h =
         iHigh(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      if(h > highest)
         highest = h;
   }

   return highest;
}

//==================================================================
// LIQUIDITY LOW
//==================================================================

double PriorLow()
{
   double lowest = DBL_MAX;

   for(
      int i = 2;
      i <= LIQUIDITY_LOOKBACK + 1;
      i++
   )
   {
      double l =
         iLow(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      if(l < lowest)
         lowest = l;
   }

   return lowest;
}

//==================================================================
// FIND RECENT LOW SWEEP
//==================================================================

int RecentLowSweep()
{
   for(
      int i = 1;
      i <= SWEEP_RECENCY_BARS;
      i++
   )
   {
      double low =
         iLow(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      double close =
         iClose(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      double level = 0.0;

      double lowest = DBL_MAX;

      for(
         int j = i + 1;
         j <= i + LIQUIDITY_LOOKBACK;
         j++
      )
      {
         double x =
            iLow(
               _Symbol,
               PERIOD_CURRENT,
               j
            );

         if(x < lowest)
            lowest = x;
      }

      level = lowest;

      if(
         low < level &&
         close > level
      )
      {
         return i;
      }
   }

   return -1;
}

//==================================================================
// FIND RECENT HIGH SWEEP
//==================================================================

int RecentHighSweep()
{
   for(
      int i = 1;
      i <= SWEEP_RECENCY_BARS;
      i++
   )
   {
      double high =
         iHigh(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      double close =
         iClose(
            _Symbol,
            PERIOD_CURRENT,
            i
         );

      double highest = -DBL_MAX;

      for(
         int j = i + 1;
         j <= i + LIQUIDITY_LOOKBACK;
         j++
         )
      {
         double x =
            iHigh(
               _Symbol,
               PERIOD_CURRENT,
               j
            );

         if(x > highest)
            highest = x;
      }

      if(
         high > highest &&
         close < highest
      )
      {
         return i;
      }
   }

   return -1;
}

//==================================================================
// POSITION SIZE
//==================================================================

double CalculateLots(
   double stopDistance
)
{
   double balance =
      AccountInfoDouble(
         ACCOUNT_BALANCE
      );

   double riskMoney =
      balance *
      RISK_PERCENT /
      100.0;

   double tickValue =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_TRADE_TICK_VALUE
      );

   double tickSize =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_TRADE_TICK_SIZE
      );

   if(
      tickValue <= 0 ||
      tickSize <= 0
   )
      return 0.0;

   double moneyPerLot =
      stopDistance /
      tickSize *
      tickValue;

   if(moneyPerLot <= 0)
      return 0.0;

   double lots =
      riskMoney /
      moneyPerLot;

   double minLot =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_VOLUME_MIN
      );

   double maxLot =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_VOLUME_MAX
      );

   double step =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_VOLUME_STEP
      );

   lots =
      MathFloor(
         lots / step
      ) * step;

   lots =
      MathMax(
         minLot,
         MathMin(
            maxLot,
            lots
         )
      );

   return lots;
}

//==================================================================
// MANAGE POSITION
//==================================================================

void ManagePosition()
{
   if(!PositionSelect(_Symbol))
      return;

   long type =
      PositionGetInteger(
         POSITION_TYPE
      );

   double price =
      type == POSITION_TYPE_BUY
      ? SymbolInfoDouble(
           _Symbol,
           SYMBOL_BID
        )
      : SymbolInfoDouble(
           _Symbol,
           SYMBOL_ASK
        );

   double atr =
      GetATR(1);

   if(atr <= 0)
      return;

   //===============================================================
   // BREAKEVEN
   //===============================================================

   if(type == POSITION_TYPE_BUY)
   {
      if(
         price >=
         entryPrice +
         BREAKEVEN_R *
         riskDistance
      )
      {
         double newStop =
            entryPrice;

         if(newStop > stopPrice)
         {
            stopPrice =
               newStop;

            trade.PositionModify(
               _Symbol,
               stopPrice,
               targetPrice
            );
         }
      }

      if(
         price <= stopPrice
      )
      {
         ClosePosition(-1.0);
         return;
      }

      if(
         price >= targetPrice
      )
      {
         ClosePosition(TARGET_R);
         return;
      }

      double fast =
         GetEMA(
            fastHandle,
            1
         );

      double close =
         iClose(
            _Symbol,
            PERIOD_CURRENT,
            1
         );

      if(
         close <
         fast -
         0.25 * atr
      )
      {
         ClosePosition(0.0);
         return;
      }
   }
   else
   {
      if(
         price <=
         entryPrice -
         BREAKEVEN_R *
         riskDistance
      )
      {
         double newStop =
            entryPrice;

         if(
            newStop < stopPrice
         )
         {
            stopPrice =
               newStop;

            trade.PositionModify(
               _Symbol,
               stopPrice,
               targetPrice
            );
         }
      }

      if(
         price >= stopPrice
      )
      {
         ClosePosition(-1.0);
         return;
      }

      if(
         price <= targetPrice
      )
      {
         ClosePosition(TARGET_R);
         return;
      }

      double fast =
         GetEMA(
            fastHandle,
            1
         );

      double close =
         iClose(
            _Symbol,
            PERIOD_CURRENT,
            1
         );

      if(
         close >
         fast +
         0.25 * atr
      )
      {
         ClosePosition(0.0);
         return;
      }
   }

   barsHeld++;

   if(
      barsHeld >= MAX_HOLD_BARS ||
      !InTradingSession()
   )
   {
      ClosePosition(0.0);
   }
}

//==================================================================
// CLOSE POSITION
//==================================================================

void ClosePosition(
   double resultR
)
{
   if(
      trade.PositionClose(
         _Symbol
      )
   )
   {
      dailyR += resultR;

      if(resultR < 0)
      {
         lossesToday++;
         consecutiveLosses++;

         cooldown =
            LOSS_COOLDOWN_BARS;
      }
      else
      {
         consecutiveLosses = 0;

         cooldown =
            COOLDOWN_BARS;
      }

      barsHeld = 0;
   }
}

//==================================================================
// LONG ENTRY
//==================================================================

void CheckLong()
{
   double atr =
      GetATR(1);

   if(atr <= 0)
      return;

   double fast =
      GetEMA(
         fastHandle,
         1
      );

   double slow =
      GetEMA(
         slowHandle,
         1
      );

   double fastPrev =
      GetEMA(
         fastHandle,
         6
      );

   double slowPrev =
      GetEMA(
         slowHandle,
         11
      );

   bool trend =
      fast > slow &&
      fast > fastPrev &&
      slow >= slowPrev;

   if(!trend)
      return;

   int sweep =
      RecentLowSweep();

   if(sweep < 0)
      return;

   double low =
      iLow(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double high2 =
      iHigh(
         _Symbol,
         PERIOD_CURRENT,
         3
      );

   double close =
      iClose(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double open =
      iOpen(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double range =
      high2 - low;

   if(range <= 0)
      return;

   //===============================================================
   // BULL FVG
   //===============================================================

   bool fvg =
      low >
      high2 &&
      close >
      open;

   if(!fvg)
      return;

   double body =
      MathAbs(
         close -
         open
      );

   if(
      body <
      MIN_DISPLACEMENT_ATR *
      atr
   )
      return;

   double location =
      (
         close -
         low
      ) /
      range;

   if(
      location <
      MIN_CLOSE_LOCATION
   )
      return;

   //===============================================================
   // RECLAIM
   //===============================================================

   double bullTop =
      high2;

   if(low > bullTop)
      return;

   if(close <= bullTop)
      return;

   //===============================================================
   // STOP
   //===============================================================

   double sweepLow =
      iLow(
         _Symbol,
         PERIOD_CURRENT,
         sweep
      );

   double structuralStop =
      MathMin(
         low,
         sweepLow
      )
      -
      STOP_BUFFER_ATR *
      atr;

   double risk =
      MathAbs(
         close -
         structuralStop
      );

   if(
      risk >
      MAX_STOP_ATR *
      atr
   )
      return;

   risk =
      MathMax(
         risk,
         0.35 * atr
      );

   entryPrice =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_ASK
      );

   riskDistance =
      risk;

   stopPrice =
      entryPrice -
      risk;

   targetPrice =
      entryPrice +
      TARGET_R *
      risk;

   double lots =
      CalculateLots(
         risk
      );

   if(lots <= 0)
      return;

   if(
      trade.Buy(
         lots,
         _Symbol,
         entryPrice,
         stopPrice,
         targetPrice,
         "Liquidity Reclaim PRO"
      )
   )
   {
      tradesToday++;
      barsHeld = 0;
   }
}

//==================================================================
// SHORT ENTRY
//==================================================================

void CheckShort()
{
   double atr =
      GetATR(1);

   if(atr <= 0)
      return;

   double fast =
      GetEMA(
         fastHandle,
         1
      );

   double slow =
      GetEMA(
         slowHandle,
         1
      );

   double fastPrev =
      GetEMA(
         fastHandle,
         6
      );

   double slowPrev =
      GetEMA(
         slowHandle,
         11
      );

   bool trend =
      fast < slow &&
      fast < fastPrev &&
      slow <= slowPrev;

   if(!trend)
      return;

   int sweep =
      RecentHighSweep();

   if(sweep < 0)
      return;

   double high =
      iHigh(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double low2 =
      iLow(
         _Symbol,
         PERIOD_CURRENT,
         3
      );

   double close =
      iClose(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double open =
      iOpen(
         _Symbol,
         PERIOD_CURRENT,
         1
      );

   double range =
      high -
      low2;

   if(range <= 0)
      return;

   bool fvg =
      high <
      low2 &&
      close <
      open;

   if(!fvg)
      return;

   double body =
      MathAbs(
         close -
         open
      );

   if(
      body <
      MIN_DISPLACEMENT_ATR *
      atr
   )
      return;

   double location =
      (
         close -
         low2
      ) /
      range;

   if(
      location >
      1.0 -
      MIN_CLOSE_LOCATION
   )
      return;

   double bearTop =
      low2;

   if(high < bearTop)
      return;

   if(close >= bearTop)
      return;

   double sweepHigh =
      iHigh(
         _Symbol,
         PERIOD_CURRENT,
         sweep
      );

   double structuralStop =
      MathMax(
         high,
         sweepHigh
      )
      +
      STOP_BUFFER_ATR *
      atr;

   double risk =
      MathAbs(
         close -
         structuralStop
      );

   if(
      risk >
      MAX_STOP_ATR *
      atr
   )
      return;

   risk =
      MathMax(
         risk,
         0.35 * atr
      );

   entryPrice =
      SymbolInfoDouble(
         _Symbol,
         SYMBOL_BID
      );

   riskDistance =
      risk;

   stopPrice =
      entryPrice +
      risk;

   targetPrice =
      entryPrice -
      TARGET_R *
      risk;

   double lots =
      CalculateLots(
         risk
      );

   if(lots <= 0)
      return;

   if(
      trade.Sell(
         lots,
         _Symbol,
         entryPrice,
         stopPrice,
         targetPrice,
         "Liquidity Reclaim PRO"
      )
   )
   {
      tradesToday++;
      barsHeld = 0;
   }
}

//==================================================================
// MAIN
//==================================================================

void OnTick()
{
   ResetDailyState();

   if(PositionSelect(_Symbol))
   {
      ManagePosition();
      return;
   }

   if(!IsNewBar())
      return;

   if(cooldown > 0)
   {
      cooldown--;
      return;
   }

   if(!InTradingSession())
      return;

   //===============================================================
   // PROP RISK GOVERNOR
   //===============================================================

   if(
      tradesToday >=
      MAX_TRADES_PER_DAY
   )
      return;

   if(
      lossesToday >=
      MAX_LOSSES_PER_DAY
   )
      return;

   if(
      dailyR <=
      -MAX_DAILY_LOSS_R
   )
      return;

   if(
      dailyR >=
      DAILY_PROFIT_LOCK_R
   )
      return;

   if(
      consecutiveLosses >=
      MAX_CONSECUTIVE_LOSSES
   )
      return;

   //===============================================================
   // VOLATILITY
   //===============================================================

   double atr =
      GetATR(1);

   double atrOld =
      GetATR(80);

   if(
      atr <= 0 ||
      atrOld <= 0
   )
      return;

   double ratio =
      atr /
      atrOld;

   if(
      ratio < MIN_ATR_RATIO ||
      ratio > MAX_ATR_RATIO
   )
      return;

   //===============================================================
   // ENTRIES
   //===============================================================

   CheckLong();

   if(!PositionSelect(_Symbol))
      CheckShort();
}

//+------------------------------------------------------------------+
