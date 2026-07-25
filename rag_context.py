"""
RAG grounding for sports LLM analysis. Pure prompt/data assembly — no scoring,
no odds logic, no decision-making. Only turns already-fetched real facts into
a strict context block so the LLM has nothing to invent from.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a zero-hallucination sports research assistant.

RULES (strict, no exceptions):
1. Use ONLY facts given in the CONTEXT block below. Do not use outside knowledge,
   training data, or assumptions about teams/players/stats not explicitly listed.
2. If a fact needed to answer is not in CONTEXT, say "not in provided data" —
   never guess, estimate, or infer a number that isn't given.
3. Never invent injuries, records, standings, weather, or lineup news unless
   it appears verbatim in CONTEXT.
4. Odds/prices you state must match CONTEXT exactly — no rounding narratives,
   no implied-probability math unless CONTEXT already contains it.
5. Keep output to one factual sentence. Flag uncertainty explicitly if present.
6. The engine's own verdict fields are authoritative. Never contradict, soften,
   or re-derive them:
   - f5_play / f5_tier is the first-five-innings verdict.
   - ml_play / ml_tier is the full-game moneyline verdict.
   - A value of PASS means NO bet on that market. If f5_play is PASS you must
     not describe an F5 play as recommended, leaning, or preferred; same for
     ml_play. If both are PASS, state that the engine passes on both markets.
   - Supporting notes explain the reasoning behind a verdict. They never
     override it — a bullish note alongside PASS is still PASS.
7. Do not restate a stat with a different meaning than CONTEXT gives it.
   "27 H" is 27 hits, not 27 home runs. Copy units and labels exactly.
"""


UNKNOWN = "not in provided data"

CONTAINER_KEYS = ("games", "slate", "board", "cards", "plays")

FIELD_ALIASES = {
    "home": ["home_team", "home", "homeTeam", "home_name"],
    "away": ["away_team", "away", "awayTeam", "away_name"],
    "matchup": ["matchup", "game", "teams"],
    "commence": ["commence_time", "start_time", "game_time", "time_et", "date", "commence"],
    "odds": ["odds", "price_line", "line", "ml", "moneyline"],
    "total": ["total", "ou", "over_under"],
    "sport": ["sport", "league"],
    "home_pitcher": ["home_pitcher", "home_sp", "home_starter", "homeProbablePitcher"],
    "away_pitcher": ["away_pitcher", "away_sp", "away_starter", "awayProbablePitcher"],
}

# Engine-produced fields passed through verbatim so the LLM grounds on the
# actual board output instead of only the raw market line.
PASSTHROUGH_FIELDS = (
    "f5_play", "f5_tier", "ml_play", "ml_tier",
    "pen_edge", "pen_mismatch", "pen_tag", "notes",
)


def _first(entry: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if entry.get(key) not in (None, "", [], {}):
            return entry[key]
    return None


def _pitcher_name(value: Any) -> str:
    """Pitcher may be a plain name string or a MLB Stats API person dict."""
    if isinstance(value, dict):
        return value.get("fullName") or value.get("name") or UNKNOWN
    return str(value)


def extract_games(data: Any) -> list[dict[str, Any]]:
    """Pull the list of game/play entries out of a board or report JSON payload."""
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    if isinstance(data, dict):
        for key in CONTAINER_KEYS:
            if isinstance(data.get(key), list):
                return [g for g in data[key] if isinstance(g, dict)]
    return []


def build_game_context(entry: dict[str, Any]) -> str:
    """Assemble a strict CONTEXT block from whatever real fields an entry has.

    Handles split home/away schemas (Odds-API style) and a single combined
    "matchup" string (slate-tracker board style). Only fields actually present
    are emitted — nothing is inferred or filled in.
    """
    matchup = _first(entry, FIELD_ALIASES["matchup"])
    if not matchup:
        home = _first(entry, FIELD_ALIASES["home"])
        away = _first(entry, FIELD_ALIASES["away"])
        matchup = f"{away} @ {home}" if home and away else UNKNOWN

    lines = [
        "CONTEXT:",
        f"- sport: {_first(entry, FIELD_ALIASES['sport']) or UNKNOWN}",
        f"- matchup: {matchup}",
        f"- start_time: {_first(entry, FIELD_ALIASES['commence']) or UNKNOWN}",
        f"- line: {_first(entry, FIELD_ALIASES['odds']) or UNKNOWN}",
    ]

    total = _first(entry, FIELD_ALIASES["total"])
    if total:
        lines.append(f"- total: {total}")

    away_p = _first(entry, FIELD_ALIASES["away_pitcher"])
    if away_p:
        lines.append(f"- away probable pitcher: {_pitcher_name(away_p)}")
    home_p = _first(entry, FIELD_ALIASES["home_pitcher"])
    if home_p:
        lines.append(f"- home probable pitcher: {_pitcher_name(home_p)}")

    for field in PASSTHROUGH_FIELDS:
        value = entry.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            # Render as nested bullets rather than a Python list repr, so the
            # model reads discrete facts instead of one quoted blob.
            lines.append(f"- {field}:")
            lines.extend(f"    - {item}" for item in value)
        else:
            lines.append(f"- {field}: {value}")

    return "\n".join(lines)


def build_prompt(context_block: str, question: str) -> str:
    """Wrap a context block + question into the final user-turn prompt."""
    return f"{context_block}\n\nQUESTION: {question}"
