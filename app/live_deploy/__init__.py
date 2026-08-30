"""
Deploy Live -- connecting a validated strategy to a REAL, funded prop-firm
account for automated live trading.

This is deliberately a separate package from app.forward_test, even though
it reuses the same MT5 connection code: forward_test exists specifically
for demo accounts (see that package's own docstring), and every safety
default there assumes "nothing real is at stake." Live deployment code
must never quietly inherit those assumptions, so it gets its own settings
storage (app.live_deploy.live_settings, supporting multiple named live
accounts rather than the one demo account forward_test manages) and its
own curated prop-firm reference data (app.live_deploy.prop_firms).

As of this module's introduction, actual live order placement is NOT
wired up -- the UI (see MainWindow._build_deploy_live_tab) supports
saving/testing live account connections end-to-end, but the "start live
trading" action is an intentional stub. Turning that on for real is a
separate, deliberate step once the account-management plumbing here has
been used and reviewed.
"""
