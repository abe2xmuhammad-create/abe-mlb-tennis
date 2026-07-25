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
"""


def build_game_context(
    sport: str,
    home: str,
    away: str,
    commence: str,
    price_line: str,
    home_pitcher: dict[str, Any] | None = None,
    away_pitcher: dict[str, Any] | None = None,
) -> str:
    """Assemble a strict CONTEXT block from already-fetched real fields only."""
    lines = [
        "CONTEXT:",
        f"- sport: {sport}",
        f"- matchup: {away} @ {home}",
        f"- commence_time: {commence}",
        f"- odds: {price_line if price_line else 'not in provided data'}",
    ]
    if home_pitcher:
        name = home_pitcher.get("fullName", "not in provided data")
        lines.append(f"- home probable pitcher: {name}")
    if away_pitcher:
        name = away_pitcher.get("fullName", "not in provided data")
        lines.append(f"- away probable pitcher: {name}")
    return "\n".join(lines)


def build_prompt(context_block: str, question: str) -> str:
    """Wrap a context block + question into the final user-turn prompt."""
    return f"{context_block}\n\nQUESTION: {question}"
