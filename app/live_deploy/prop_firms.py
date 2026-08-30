"""
A curated, informational reference list of prop firms and which trading
platform(s) each is known to use -- so the Deploy Live tab can show a
dropdown with a sensible starting point instead of a blank text field.

This is NOT a live directory and nothing here is fetched from the firms
themselves -- server names, exact platform lineups, and which firms are
even still operating all change over time (one well-known firm, for
example, shut down with little warning in February 2026). Always confirm
the exact server name from your own account-issued email or firm
dashboard before connecting; the `notes` field flags anything more
important than that to know up front.

The single most important fact this module exists to surface: MT4/MT5 and
futures-focused platforms (Tradovate, Rithmic, NinjaTrader, TopStep's
ProjectX/TopStepX) are entirely different integrations. This app's live
connector (reusing app.forward_test.mt5_connector) only speaks MT4/MT5
today. A firm listed with only futures platforms below is NOT
connectable yet through Deploy Live, regardless of how good the firm is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropFirm:
    name: str
    platforms: list[str]          # what this app can act on now: only entries containing "MT4"/"MT5"
    asset_focus: str              # "Forex/CFD", "Futures", etc. -- informational
    connectable_today: bool       # True only if at least one platform is MT4/MT5
    notes: str = ""


PROP_FIRMS: list[PropFirm] = [
    PropFirm(
        "FTMO", ["MT4", "MT5", "cTrader", "DXtrade"], "Forex/CFD", True,
        "One of the longest-running firms (since 2015). MT4/MT5 accounts work with this app's "
        "connector today; cTrader/DXtrade accounts do not.",
    ),
    PropFirm(
        "FundedNext", ["MT5", "cTrader", "Match-Trader"], "Forex/CFD", True,
        "MT5 accounts work with this app's connector; cTrader/Match-Trader accounts do not.",
    ),
    PropFirm(
        "The5ers", ["MT5"], "Forex/CFD", True,
        "Historically MT5-only, which this app's connector supports.",
    ),
    PropFirm(
        "E8 Markets", ["MT5", "cTrader"], "Forex/CFD", True,
        "MT5 accounts work with this app's connector; cTrader accounts do not.",
    ),
    PropFirm(
        "Blue Guardian", ["MT4", "MT5"], "Forex/CFD", True,
        "MT4/MT5 accounts work with this app's connector.",
    ),
    PropFirm(
        "Apex Trader Funding", ["Tradovate", "Rithmic", "NinjaTrader", "TradingView"], "Futures", False,
        "NOT connectable yet -- Apex is futures-only via Tradovate/Rithmic/NinjaTrader, a "
        "completely different integration than MT4/MT5. Explicitly permits automated/EA trading "
        "on its current account lineup, for when this integration exists.",
    ),
    PropFirm(
        "Topstep", ["TopStepX / ProjectX", "NinjaTrader", "Tradovate", "Rithmic"], "Futures", False,
        "NOT connectable yet -- futures-only, routed through Topstep's own TopStepX/ProjectX stack "
        "(or NinjaTrader/Tradovate/Rithmic depending on account type), not MT4/MT5.",
    ),
    PropFirm(
        "MyFundedFutures", ["Tradovate", "Rithmic", "NinjaTrader"], "Futures", False,
        "NOT connectable yet -- futures-only. Reversed an earlier ban on automated trading in "
        "mid-2025; check their current rules before assuming any given automation is still allowed.",
    ),
    PropFirm(
        "Other / not listed", ["MT4", "MT5", "cTrader", "Other"], "Unknown", True,
        "Pick this and select MT4/MT5 as the platform if your firm isn't listed here but uses "
        "MetaTrader for funded accounts -- the connection works the same way regardless of firm "
        "name. If your firm uses something else, it isn't connectable yet.",
    ),
]


def find(name: str) -> PropFirm | None:
    return next((f for f in PROP_FIRMS if f.name == name), None)
