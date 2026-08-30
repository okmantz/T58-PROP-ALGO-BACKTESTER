"""Strategy Generator -- the optional local-Ollama feature that drafts a NEW
strategy's source code from a plain-language idea, grounded in your own
research/ papers and your own already-tested strategies.

This is a meaningfully bigger step than app.ai.ollama_client's numeric-only
parameter suggestions: it produces a whole new file, not just numbers for an
existing one. Kept in its own module (rather than added to OllamaClient) so
that module's documented scope -- "never asked to write or edit strategy
code" -- stays true, and so this riskier, opt-in feature is easy to find and
reason about on its own.

Because a local model can hallucinate subtly-broken trading logic (a
lookahead bug, an unsupported PineScript/MQL5 construct, exit logic that
silently never fires) that reads as fine but isn't, every strategy this
module produces is treated as unproven by construction:
  - It is only ever written to disk tagged "draft" (see
    app.strategy.library.DEFAULT_STATUS) -- the caller (the UI) is
    responsible for actually saving it; this module only returns text.
    Nothing here ever sets a higher status.
  - It goes through this app's OWN strict parsers (PythonStrategy /
    PineScriptStrategy / MQL5Strategy) the very first time it's tested, so
    a script using an unsupported construct fails loudly immediately
    rather than quietly misbehaving.
  - It is never executed automatically -- generating code and running it
    are two separate, deliberate steps the person takes.

Reuses app.ai.ollama_settings for host/model/api-key (same "off by default,
local by default" posture as the rest of the AI features) and
app.ai.research_library for the same free, deterministic keyword-overlap
retrieval already used for parameter tuning -- no new dependency, no vector
DB, no extra network calls beyond the one Ollama request that actually
generates the code.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from app.ai.ollama_settings import OllamaSettings

# This is a STALL timeout (max seconds of silence between streamed tokens),
# not a total-runtime timeout -- see generate_strategy(). A local model can
# legitimately take several minutes to draft a whole file; what actually
# indicates trouble is the model going quiet mid-response.
DEFAULT_TIMEOUT_SECONDS = 180
# Hard ceiling on total wall-clock time even while tokens keep arriving, so a
# model that's technically still "responding" (one token every few seconds,
# forever) can't hang the UI indefinitely.
DEFAULT_MAX_TOTAL_SECONDS = 900
DEFAULT_N_RESEARCH_EXCERPTS = 3
DEFAULT_N_PRIOR_EXAMPLES = 2

# Deliberately modest defaults -- "use minimal resources but get the job
# done." A strategy file this app can actually load is at most a few hundred
# lines, so neither a huge context window nor a huge output cap buys
# anything here; both just cost RAM/VRAM and time on whatever machine Ollama
# is running on. Raised these only if generation keeps visibly getting cut
# off mid-file for a particular idea/model.
DEFAULT_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 1800

# Model unloads from memory this long after the request finishes, instead of
# lingering resident indefinitely (Ollama's own default keep_alive is 5
# minutes, which is fine to stay explicit about here rather than depend on
# whatever the local install's default happens to be).
KEEP_ALIVE = "5m"

ProgressCallback = Callable[[int, float], None]  # (tokens_so_far, elapsed_seconds)

LANGUAGE_EXTENSIONS = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}


# ---------------------------------------------------------------------------
# Per-language contracts -- what this app's OWN parser for that language
# actually accepts. Sent to the model as hard constraints, not suggestions:
# code outside these subsets fails to load the first time it's tested.
# ---------------------------------------------------------------------------

_PYTHON_CONTRACT = """\
Output a single self-contained Python file. Hard requirements:
- Must define: def generate_signals(df: pd.DataFrame) -> pd.Series
  Returns a pandas Series, same length/index as df, containing only -1
  (short), 0 (flat), or 1 (long) per bar.
- df has columns: timestamp, open, high, low, close, volume (lowercase).
  Only use these -- no other columns exist.
- May define module-level constants STRATEGY_NAME (str), STOP_LOSS_PIPS
  and TAKE_PROFIT_PIPS (float, fixed whole-backtest stop/target) -- or, for
  a per-trade computed stop/target (e.g. an ATR multiple), attach it via
  signals.attrs on the returned Series instead of a fixed pip count:
    signals.attrs["stop_loss_distance"]     -> Series/array, |entry - stop| in raw price units, only set on entry bars
    signals.attrs["take_profit_distance"]   -> Series/array, |entry - target| in raw price units, only set on entry bars
    signals.attrs["trailing_stop_distance"] -> Series/array, raw-price trailing distance
    signals.attrs["breakeven_trigger_r"]    -> scalar float, e.g. 1.0 means "+1R"
  If you compute a stop/target inside the function and never attach it
  here, the backtest engine has no way to know about it and will silently
  fall back to its own generic stop/target instead -- this is the ONLY
  path a computed stop/target reaches execution.
- generate_signals(df) is called ONCE, statelessly, over the entire
  dataset before any trade has been opened or closed. Never write logic
  that assumes knowledge of the strategy's own past trade outcomes (e.g.
  "stop trading after N losses today") -- there is no way to compute that
  correctly inside this function, and it will silently be a no-op.
- LOOKAHEAD: every value used to decide bar i's signal must be computable
  using only data at or before bar i. A common, tempting bug: resampling
  to a higher timeframe and filtering with `htf[htf.index < timestamp]` --
  this LEAKS the still-forming current higher-timeframe bar (a resampled
  bar is labeled by its start time), because it was built using bars later
  than `timestamp` that haven't happened yet in the lower timeframe. Use
  only fully-closed prior bars/values (e.g. `.shift(1)`, a rolling window
  ending at the previous bar) for anything resembling a higher-timeframe
  or lookback filter.
- Only pandas and numpy may be imported. No file I/O, no network calls, no
  other third-party packages.
"""

_PINESCRIPT_CONTRACT = """\
Output PineScript v5. This app's parser only understands a RESTRICTED
SUBSET -- anything outside it fails to load. You may ONLY use:
- Price references: open, high, low, close, hl2, hlc3, ohlc4
- x = input.int(20, ...) / input.float(1.5, ...) -- becomes a constant
  using the given default value; no other input.* types
- x = ta.sma(src, len), ta.ema(src, len), ta.wma(src, len), ta.rsi(src, len)
  -- no other ta.* functions
- x = ta.crossover(a, b), ta.crossunder(a, b)
- Boolean rule variables built from comparisons/and/or/not over the above,
  e.g. longCondition = ta.crossover(fast, slow) and rsiVal < 70
- Entries, inline or inside an if block:
    strategy.entry("Long", strategy.long, when=longCondition)
    if longCondition
        strategy.entry("Long", strategy.long)
- Exits: strategy.close("Long", when=exitLongCondition)
- Stop-loss/take-profit as special directive comments (not strategy.exit
  price offsets):
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
Do NOT use: custom functions, arrays/matrices, security()/multi-timeframe
requests, repainting constructs, plotting, alerts, or any ta.* function
not listed above -- all of these fail to parse.
"""

_MQL5_CONTRACT = """\
Output MQL5 Expert Advisor source. This app's parser only understands a
RESTRICTED SUBSET -- anything outside it fails to load. You may ONLY use:
- Direct-value indicator calls (the simplified/legacy calling style):
    double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
    double slowMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
    double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
  (only MODE_SMA/MODE_EMA/MODE_LWMA; only iMA and iRSI as indicators)
- Boolean conditions with C-style operators: > < >= <= == != && || !
- if (condition) { ... } or single-statement if (condition) statement;
- Entries inside a condition's guard: trade.Buy(...) / trade.Sell(...)
  (or OrderSend(..., ORDER_TYPE_BUY/ORDER_TYPE_SELL, ...))
- Exits inside a condition's guard: trade.PositionClose(...) / OrderClose(...)
- Stop-loss/take-profit as special directive comments:
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
Do NOT use: CopyBuffer()-based indicator handles, custom indicators,
arrays/structs, multi-symbol/multi-timeframe logic, trailing stops, or any
indicator beyond iMA/iRSI -- all of these fail to parse.
"""

_CONTRACTS = {"python": _PYTHON_CONTRACT, "pinescript": _PINESCRIPT_CONTRACT, "mql5": _MQL5_CONTRACT}
_FENCE_LANGS = {"python": ("python", "py"), "pinescript": ("pinescript", "pine"), "mql5": ("mql5", "cpp", "c")}


@dataclass
class GenerationResult:
    code: str | None = None
    filename_hint: str = ""
    rationale: str = ""
    error: str | None = None  # human-readable reason, set only when code is None because something went wrong


def _build_prompt(
    language: str,
    idea: str,
    research_excerpts: list[dict] | None = None,
    prior_examples: list[dict] | None = None,
) -> str:
    """Pure function: idea + retrieved context -> prompt text. Kept
    separate from any network code so this is testable without Ollama
    installed.

    research_excerpts: pre-retrieved (systematic keyword search, not AI --
    see app.ai.research_library.find_relevant_excerpts) excerpts from your
    research/ folder.

    prior_examples: pre-selected (systematic -- newest/best-status per
    language, not AI) short excerpts of your own already-saved strategies
    of this same language, so the model can match this codebase's actual
    style and constraints instead of guessing generic textbook Pine/MQL5.
    """
    contract = _CONTRACTS[language]

    research_section = ""
    if research_excerpts:
        body = "\n\n".join(
            f"  [{i + 1}] (from {e['source']}): {e['text']}" for i, e in enumerate(research_excerpts)
        )
        research_section = f"""
Relevant excerpts from your research library (use as grounding where they \
genuinely apply -- ignore anything that doesn't fit this idea):
{body}
"""

    examples_section = ""
    if prior_examples:
        body = "\n\n".join(
            f"  --- {e['name']} (status: {e['status']}) ---\n{e['excerpt']}" for e in prior_examples
        )
        examples_section = f"""
Examples of your own existing {language} strategies in this exact codebase \
(match this style/structure -- these are known to load and run correctly):
{body}
"""

    return f"""You are drafting a NEW algorithmic trading strategy for a private backtesting \
app. You are being asked to write complete, runnable source code this one time -- \
this is different from ordinary parameter tuning.

Strategy idea / instructions from the trader:
{idea.strip()}
{research_section}{examples_section}
STRICT OUTPUT CONTRACT for this app's {language} loader -- code that doesn't follow \
this will fail to load:
{contract}
Respond with ONLY the code, inside a single fenced code block, no other prose before \
or after it. Also include, as a comment near the top of the file, a one-to-three \
line explanation of the strategy's logic (entry, exit, stop/target).
"""


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n([\s\S]*?)```")


def _extract_code(raw_text: str) -> str | None:
    """Pure function: pulls the first fenced code block out of a model's
    response. Falls back to the whole response (stripped) if no fence is
    found but the text looks like actual code rather than prose -- small
    local models sometimes forget the fence. Returns None only when
    there's nothing usable at all."""
    if not raw_text or not raw_text.strip():
        return None
    match = _FENCE_RE.search(raw_text)
    if match:
        code = match.group(1).strip()
        return code or None
    # No fence -- accept the raw response only if it doesn't read like prose
    # (a real strategy file has multiple lines and no sentence-ending
    # "." followed by a capital letter mid-paragraph is a weak signal, so
    # keep this conservative: require it to at least look structurally
    # like code for one of the three languages).
    text = raw_text.strip()
    code_markers = ("def ", "import ", "//@version", "strategy(", "void ", "double ", "input.")
    if any(marker in text for marker in code_markers):
        return text
    return None


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug[:max_len].rstrip("_")) or "generated_strategy"


def gather_prior_examples(language: str, max_examples: int = DEFAULT_N_PRIOR_EXAMPLES, max_chars: int = 1200) -> list[dict]:
    """Pulls short excerpts of your own best-status saved strategies of
    this language, newest first within each status tier, best tier first
    -- purely deterministic (no AI), so the model sees real, known-working
    code from this exact codebase instead of guessing generic syntax.
    Never raises; returns [] if the library is empty or unreadable."""
    try:
        from app.strategy.library import list_saved_strategies
    except Exception:
        return []
    try:
        items = list_saved_strategies(language)
    except Exception:
        return []
    if not items:
        return []
    _tier = {"ready_for_live": 0, "ready_for_demo": 1, "validated": 2, "tested_passed": 3, "draft": 4, "tested_failed": 5}
    items = sorted(items, key=lambda it: (_tier.get(it.status, 9), -it.modified))
    out = []
    for item in items[:max_examples]:
        try:
            text = item.path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        excerpt = text if len(text) <= max_chars else text[:max_chars] + "\n... (truncated)"
        out.append({"name": item.name, "status": item.status, "excerpt": excerpt})
    return out


def generate_strategy(
    settings: OllamaSettings,
    language: str,
    idea: str,
    n_research_excerpts: int = DEFAULT_N_RESEARCH_EXCERPTS,
    n_prior_examples: int = DEFAULT_N_PRIOR_EXAMPLES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_total_seconds: int = DEFAULT_MAX_TOTAL_SECONDS,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    progress_cb: ProgressCallback | None = None,
) -> GenerationResult:
    """Asks the configured local Ollama model to draft a new strategy file
    for the given language, grounded in your research/ folder (systematic
    keyword retrieval, not AI) and your own best existing strategies of
    that language (also systematic, not AI). Returns a non-fatal
    GenerationResult (with `.error` explaining why) on any failure --
    never raises. The caller is responsible for saving the result (always
    as a "draft" -- see app.strategy.library.DEFAULT_STATUS) and for ever
    actually testing/running it; nothing here does either.

    Streams the response instead of waiting for one giant reply: `timeout`
    is a STALL timeout (max seconds of silence between tokens) rather than
    a wall-clock deadline on the whole generation, since a small local
    model can legitimately take minutes to write a few hundred lines --
    what actually signals trouble is the model going quiet mid-response,
    not simply taking a while. `max_total_seconds` is a separate hard
    ceiling so a model that's technically still trickling out tokens
    forever still eventually gets cut off. `num_ctx`/`num_predict` are
    passed to Ollama to keep this one-off call's memory/compute footprint
    modest (see the module-level DEFAULT_NUM_CTX/DEFAULT_NUM_PREDICT
    comment) -- raise them from the UI only if a particular idea/model
    keeps visibly getting cut off mid-file."""
    import requests

    if language not in _CONTRACTS:
        return GenerationResult(error=f"Unknown language '{language}'. Expected one of {list(_CONTRACTS)}.")
    if not idea or not idea.strip():
        return GenerationResult(error="Describe the strategy idea first -- entry/exit logic, indicators, market, etc.")
    if not settings.is_usable:
        return GenerationResult(error="Ollama isn't enabled/configured -- set it up first (see AI Assist settings).")

    try:
        from app.ai.research_library import find_relevant_excerpts
        research_excerpts = find_relevant_excerpts(idea, max_excerpts=n_research_excerpts)
    except Exception:
        research_excerpts = []
    prior_examples = gather_prior_examples(language, max_examples=n_prior_examples)

    prompt = _build_prompt(language, idea, research_excerpts=research_excerpts, prior_examples=prior_examples)
    host = (settings.host or "").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    payload = {
        "model": settings.model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            # Keep resource usage minimal: a small context window and a
            # capped output length are both plenty for a strategy file
            # this app's own parsers can actually load, and both directly
            # reduce RAM/VRAM and per-token compute versus a model's
            # (often much larger) defaults.
            "num_ctx": max(512, int(num_ctx)),
            "num_predict": max(64, int(num_predict)),
            "temperature": 0.2,
        },
    }

    raw_text_parts: list[str] = []
    t0 = time.monotonic()
    last_progress = t0
    try:
        # timeout=(connect, read): the read half applies PER SOCKET READ
        # while streaming, not to the request as a whole -- so as long as
        # another token/chunk keeps arriving within `timeout` seconds, the
        # request keeps going even past `timeout` seconds of total elapsed
        # time. This is what actually fixes "didn't respond in time" for a
        # slow local model that's still working, as opposed to one that's
        # genuinely stuck.
        resp = requests.post(
            f"{host}/api/generate", headers=headers, json=payload,
            stream=True, timeout=(10, timeout),
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            elapsed = time.monotonic() - t0
            if elapsed > max_total_seconds:
                resp.close()
                return GenerationResult(
                    error=f"Ollama at {host} has been generating for over "
                          f"{max_total_seconds // 60} minutes without finishing -- stopped it. "
                          f"Try a smaller/more specific idea or a smaller, faster model.",
                    rationale="".join(raw_text_parts)[:1000],
                )
            try:
                chunk = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            piece = chunk.get("response", "")
            if piece:
                raw_text_parts.append(piece)
                if progress_cb and (time.monotonic() - last_progress) >= 1.5:
                    last_progress = time.monotonic()
                    try:
                        progress_cb(len(raw_text_parts), elapsed)
                    except Exception:
                        pass
            if chunk.get("error"):
                return GenerationResult(error=f"Ollama error: {chunk['error']}")
            if chunk.get("done"):
                break
        raw_text = "".join(raw_text_parts)
    except requests.exceptions.ConnectionError:
        return GenerationResult(error=f"Couldn't reach Ollama at {host} (is it running?).")
    except requests.exceptions.Timeout:
        return GenerationResult(
            error=f"Ollama at {host} went quiet for over {timeout}s mid-generation (no new output at "
                  f"all in that window) -- the model may be stuck, or the machine is out of resources. "
                  f"Try a smaller idea, a smaller/faster model, or lowering the context/output-length "
                  f"settings above.",
            rationale="".join(raw_text_parts)[:1000],
        )
    except Exception as exc:
        return GenerationResult(error=f"Ollama request failed: {exc}")

    code = _extract_code(raw_text)
    if not code:
        return GenerationResult(
            error="Ollama responded, but nothing that looked like code could be extracted from it.",
            rationale=raw_text[:1000],
        )
    return GenerationResult(code=code, filename_hint=_slugify(idea), rationale=raw_text[:2000])
