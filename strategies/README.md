# Strategy Library

This folder is the app's built-in strategy library. Strategies saved here
(from the desktop app's Strategy tab or the mobile web app) are available
on every future run without re-uploading them from your computer or phone.

```
strategies/
├── python/       *.py strategy files (+ *.py.meta.json sidecars)
├── pinescript/   *.pine strategy files (+ *.pine.meta.json sidecars)
└── mql5/         *.mq5 strategy files (+ *.mq5.meta.json sidecars)
```

## Features

- **Save / load / delete** a strategy without leaving the app.
- **Overwrite vs. duplicate**: saving under a name that's already taken
  asks whether to overwrite the existing file or save as a new, separately
  named copy — it will never silently create a confusing " (2)" duplicate.
- **Rename** a saved strategy in place (its metadata moves with it).
- **Metadata**: each strategy can carry a description, market/timeframe,
  and (recorded automatically after a run) its last backtest's stats —
  trades, net profit, win rate, max drawdown, report link.
- **Search**: filter the library list by filename, description, or market.
- **Export**: download the entire library as one zip, for backup or for
  manually syncing a packaged .exe's saved strategies into this repo.

## A note on the packaged .exe vs. this git repo

When you run the app straight out of this repo (`python run_app.py` /
`python -m app.web.server`), saved strategies land directly in this
`strategies/` folder and show up in `git status` right away.

When you run a **packaged .exe** build, the library instead lives in a
persistent app-data folder *next to that .exe* — not inside this repo.
That's intentional (it's the same rule the app already follows for your
market-data CSVs), but it means a strategy you save from the built .exe
won't appear on GitHub until you bring it over yourself:

1. In the app, click **EXPORT LIBRARY AS ZIP** (desktop) or **"Download
   entire strategy library as a zip"** (web).
2. Unzip it into this repo's `strategies/` folder.
3. Commit as usual.

You can also drop a file directly into the matching subfolder yourself —
the app picks it up the next time you press "REFRESH LIBRARY" (desktop) or
reload the page (web).

Uploading a file from your computer or phone still works exactly as
before and is unaffected by any of this — in fact it's how files get
saved here in the first place.
