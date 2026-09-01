# Web/exe feature parity roadmap

Full parity means every desktop tab has a working web equivalent. This is
being done in rounds -- each round ports 2-4 tabs completely (routes,
templates, background-job wiring where the tab is slow, tests) rather than
stubbing all of them thinly at once.

## Done

- [x] **Round 1 infra**: Cloud Run + Firebase Hosting deploy pipeline
      (`Dockerfile`, `firebase.json`, `.firebaserc`, GitHub Actions workflow).
      See `FIREBASE_DEPLOY.md`.
- [x] **Round 1 feature**: Step 06 Iterative Refinement (`/refine`,
      `/refine/start`, `/refine/job/<id>`) -- same background-job/poll
      pattern as the existing Search Lab.
- [x] **Round 2 feature**: Step 15 Full Pipeline (`/full-pipeline`,
      `/full-pipeline/start`, `/full-pipeline/job/<id>`) -- baseline -> GA
      search -> final Monte Carlo -> OOS/holdout -> verdict, plus an
      optional AI Assist (Ollama) section matching the desktop tab.
      End-to-end tested for real (not just route smoke tests): posted a
      synthetic dataset through the actual route, polled the job to
      completion, verified a genuine verdict/report came back and the
      report file serves correctly.
- [x] **Round 3 features**: Step 08 Walk-Forward Optimization
      (`/walk-forward-opt`), Step 09 Multi-Objective/NSGA-II
      (`/multi-objective`), Step 10 Walk-Forward-Aware GA
      (`/walk-forward-ga`) -- all three follow the same background-job/poll
      pattern. Each end-to-end tested for real against synthetic data
      (posted through the actual routes, polled jobs to completion,
      verified genuine results and that report files serve). Confirmed
      Multi-Objective's real validation (2+ objectives required) surfaces
      correctly as a 400 error on the web form, not a silent failure.
- [x] **Round 4 features**: Step 11 Multi-Asset Portfolio (`/portfolio`)
      and Step 12 Multi-Strategy Ensemble (`/ensemble`, blend + vote
      modes). Both are fast enough (no GA search) to run synchronously
      like `/run` rather than needing the background-job pattern. Web UI
      simplification vs. desktop: fixed 4 leg slots instead of an
      unlimited add/remove list -- covers realistic use, noted in the
      roadmap in case more are ever needed. End-to-end tested for real:
      Portfolio with 2 distinct-instrument legs, Ensemble in both blend
      and vote mode with 2 real Python strategy legs, all through the
      actual routes with real reports confirmed serving.
- [x] **Round 5 features (final round -- full parity reached)**:
  - Step 13 **CPCV** (`/cpcv`) -- single-strategy combinatorial purged
    cross-validation. Genuine multi-candidate PBO was NOT wired up (it
    needs a pool of already-tried candidates -- e.g. a Search Lab
    leaderboard or a Refinement run's final generation -- as input, not a
    single strategy config; a natural follow-on once one of those outputs
    can be piped in as a candidate pool).
  - Step 14 **Parameter Sensitivity** (`/sensitivity`) -- 1D sweeps only.
    2D heatmap NOT wired up (needs a two-step UI: discover tunable
    parameter names, then let the person pick 2 by name).
  - **Quick Optimize** (`/quick-optimize`) -- one-click GA tune, saves to
    library tagged "draft".
  - **Evolution Lab** (`/evolution`) -- genuinely different shape from
    every other tab: a single global `EvolutionRunner` instance (matches
    the app's single-user/LAN trust model) with start/stop controls, a
    live leaderboard, and a journal, polled the same way as the job-based
    tabs. Its own on-disk checkpoint makes STOP-then-START resume
    correctly, including across a server restart.
  - **18 Research Agent** (`/research-agent`) -- ReAct tool-calling agent,
    read-only engine tools only. Tested both failure paths for real: an
    empty question correctly 400s, and no-Ollama-reachable surfaces a
    clear connection error through the job status instead of hanging.
  - **Forward Test (MT5)** (`/forward-test`) -- NOT ported (see note
    below); this page explains why instead of silently missing.

  Every job-based feature above (CPCV, Sensitivity, Quick Optimize,
  Evolution Lab) was run end-to-end for real against synthetic data
  through the actual routes, not just page-load checks. Full test suite
  (587 tests) passes with zero regressions after all 5 rounds.

## Full web/exe parity reached

All 18 desktop tabs now have a web equivalent, with two documented,
deliberate exceptions:
- **Forward Test (MT5)** -- platform constraint (Windows-only MT5
  terminal), not a missing feature. See `/forward-test` in the app.
- **PBO** and the **2D sensitivity heatmap** -- both need a slightly
  different UI shape (a candidate pool; a two-step parameter picker) than
  anything else on this site uses, so they were scoped out rather than
  half-built. Small, well-understood follow-ups whenever they're wanted.

## Already had web parity (built in earlier rounds)

- Step 01 Market Data, Step 02 Strategy (Manual/Python/PineScript/MQL5),
  Step 03 Prop Rules, Step 04 Risk & Execution, Step 05 Run & Report
  (`/`, `/run`)
- Strategy Library (`/strategies/*`) incl. batch test, view code, rename,
  tags/status, bulk export/delete
- Step 07 Search Lab (`/search`, `/search/start`, `/search/job/<id>`)
- Live Market data view (`/live-market`)

## Not yet ported

- **PBO** (`app.validation.cpcv.compute_pbo`) -- needs a multi-candidate
  pool as input (e.g. wire up "send this Search Lab leaderboard to PBO" or
  "send this Refinement run's final generation to PBO" as an action on
  those pages) rather than a fresh single-strategy form.
- **2D sensitivity heatmap** (`app.validation.sensitivity.compute_2d_heatmap`)
  -- needs a two-step UI: call `list_tunable_parameters()` first so the
  person can pick 2 parameter names from what the strategy actually has,
  then run the heatmap.
- **Forward Test (MT5) / Live Deploy** -- talks to a locally-running
  MetaTrader5 terminal, which only exists on Windows. **Cannot** get true
  web parity by nature (not a bug, a platform constraint). `/forward-test`
  explains this in the app rather than leaving a dead or missing link. The
  honest options if this ever needs solving: leave it desktop-only
  permanently, or build a separate always-on Windows relay service the
  web app talks to over the network (a real project, not a quick port).

## Known simplification vs. desktop

- **Portfolio / Ensemble leg counts**: the web forms offer 4 fixed leg
  slots (2 required, 2 optional) instead of desktop's unlimited
  add/remove list. Covers realistic use; if a real need for 5+ legs shows
  up, revisit as a small follow-up (dynamic JS rows) rather than a full
  re-port.

## Notes for next round

Say which of the above to do next (or "just go in order"), and whether
Cloud Run's ephemeral-storage tradeoff (see `FIREBASE_DEPLOY.md`) needs
solving before we go further, or can wait.
