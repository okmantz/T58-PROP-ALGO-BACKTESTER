"""Client for the optional local-Ollama AI trading assistant.

Scope, deliberately narrow: the model is only ever asked to suggest
NUMERIC VALUES for a strategy's already-discovered tunable parameters (see
app.optimize.code_parameter_space / app.optimize.parameter_space's gene
lists) -- never asked to write or edit strategy code, and never given the
ability to. Every suggestion it returns is just a genome (a list of
numbers, one per existing gene) that gets evaluated through the exact same
backtest -> prop-simulation -> Monte Carlo pipeline as every other
candidate the GA tries. The AI can propose; it can never bypass validation.

This keeps three concerns cleanly separated so the parsing/prompt logic
can be unit-tested without a live Ollama server:
  - _build_prompt(...)      pure function, strategy/gene/stats -> prompt text
  - _parse_suggestions(...) pure function, raw model text -> list[dict]
  - OllamaClient            thin IO wrapper around requests calls

Fails safe everywhere: unreachable host, timeout, bad JSON, a model that
hallucinates gene names, wrong list length -- all of these result in an
empty suggestion list (and a human-readable reason), never an exception
that could take down a Full Pipeline run. Full Pipeline treats "no
suggestions" exactly like "AI assist is off": it just proceeds with its
normal random/mutated population.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.ai.ollama_settings import OllamaSettings

DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_N_SUGGESTIONS = 3


@dataclass
class AISuggestionResult:
    genomes: list[list[float]] = field(default_factory=list)
    rationale: str = ""
    error: str | None = None  # human-readable reason, set only when genomes is empty because something went wrong


def _build_prompt(
    strategy_name: str,
    source_type: str,
    genes: list,
    baseline_stats: dict,
    prop_rules_summary: dict,
    n_suggestions: int,
) -> str:
    """Pure function: describes the strategy's tunable parameters and how
    it's currently performing, and asks for a strict-JSON response. Kept
    separate from any network code so this can be tested without Ollama
    installed."""
    gene_lines = "\n".join(
        f'  - "{g.label}": currently {g.base_value} (allowed range {g.lo} to {g.hi}, '
        f'{"integer" if g.is_int else "decimal"})'
        for g in genes
    )
    stats_lines = "\n".join(f"  - {k}: {v}" for k, v in baseline_stats.items())
    rules_lines = "\n".join(f"  - {k}: {v}" for k, v in prop_rules_summary.items())

    return f"""You are helping tune the numeric parameters of an existing algorithmic \
trading strategy called "{strategy_name}" ({source_type}). You are NOT being asked to \
write or modify any code -- only to propose new numeric values for the parameters \
listed below, within their allowed ranges.

Tunable parameters (in order):
{gene_lines}

Current (baseline) backtest performance:
{stats_lines}

Prop-firm rules this strategy must pass:
{rules_lines}

Propose {n_suggestions} different candidate parameter sets you think are worth trying \
to improve profitability and the odds of passing evaluation and reaching payout. Each \
candidate must give a value for every parameter listed above, in the same order, \
within its allowed range.

Respond with ONLY a JSON array, no other text, no markdown code fences, in exactly \
this shape:
[
  [value_for_param_1, value_for_param_2, ...],
  [value_for_param_1, value_for_param_2, ...]
]
"""


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_suggestions(raw_text: str, genes: list) -> list[list[float]]:
    """Pure function: extracts a list of numeric genomes from a model's
    raw text response. Tolerant of the common failure modes of small local
    models -- markdown code fences, leading/trailing prose, a single
    genome instead of a list of them -- but never guesses past outright
    malformed or wrong-shaped JSON; those candidates are just dropped
    rather than risking a nonsensical genome reaching the backtester."""
    if not raw_text or not raw_text.strip():
        return []

    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []
    # Tolerate a model that returns a single flat genome instead of a list of them.
    if parsed and all(isinstance(x, (int, float)) for x in parsed):
        parsed = [parsed]

    n_genes = len(genes)
    genomes: list[list[float]] = []
    for candidate in parsed:
        if not isinstance(candidate, list) or len(candidate) != n_genes:
            continue
        if not all(isinstance(v, (int, float)) for v in candidate):
            continue
        clamped = []
        for value, gene in zip(candidate, genes):
            v = max(gene.lo, min(gene.hi, float(value)))
            clamped.append(round(v) if gene.is_int else v)
        genomes.append(clamped)
    return genomes


class OllamaClient:
    def __init__(self, settings: OllamaSettings, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.settings = settings
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def test_connection(self) -> tuple[bool, str]:
        """Pings the configured host and confirms the configured model is
        actually pulled. Never raises -- returns (ok, message) for direct
        display in the UI."""
        import requests

        host = (self.settings.host or "").rstrip("/")
        if not host:
            return False, "No Ollama host configured."
        try:
            resp = requests.get(f"{host}/api/tags", headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return False, f"Couldn't reach Ollama at {host}. Is it running? (Try: ollama serve)"
        except requests.exceptions.Timeout:
            return False, f"Timed out reaching Ollama at {host}."
        except Exception as exc:
            return False, f"Couldn't reach Ollama at {host}: {exc}"

        try:
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            names = []
        model = self.settings.model
        if names and not any(n == model or n.startswith(f"{model}:") for n in names):
            available = ", ".join(names[:8]) or "(none)"
            return False, (
                f"Connected to Ollama, but model '{model}' isn't pulled. "
                f"Available: {available}. Run: ollama pull {model}"
            )
        return True, f"Connected to Ollama at {host} -- model '{model}' is ready."

    def suggest_parameter_adjustments(
        self,
        strategy_name: str,
        source_type: str,
        genes: list,
        baseline_stats: dict,
        prop_rules_summary: dict,
        n_suggestions: int = DEFAULT_N_SUGGESTIONS,
    ) -> AISuggestionResult:
        """Asks the model for `n_suggestions` candidate genomes. Returns an
        empty, non-fatal result (with `.error` explaining why) on any
        failure -- callers should treat that exactly like AI assist being
        disabled and continue with their normal search."""
        import requests

        if not genes:
            return AISuggestionResult(error="Strategy has no tunable parameters for the AI to suggest values for.")

        prompt = _build_prompt(strategy_name, source_type, genes, baseline_stats, prop_rules_summary, n_suggestions)
        host = (self.settings.host or "").rstrip("/")
        try:
            resp = requests.post(
                f"{host}/api/generate",
                headers=self._headers(),
                json={"model": self.settings.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return AISuggestionResult(error=f"Couldn't reach Ollama at {host} (is it running?).")
        except requests.exceptions.Timeout:
            return AISuggestionResult(error=f"Ollama at {host} didn't respond in time.")
        except Exception as exc:
            return AISuggestionResult(error=f"Ollama request failed: {exc}")

        genomes = _parse_suggestions(raw_text, genes)
        if not genomes:
            return AISuggestionResult(
                error="Ollama responded, but its suggestions couldn't be parsed into valid parameter sets -- "
                      "proceeding without them.",
                rationale=raw_text[:500],
            )
        return AISuggestionResult(genomes=genomes, rationale=raw_text[:2000])
