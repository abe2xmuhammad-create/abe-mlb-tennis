"""
Standalone RAG analysis layer. Reads an existing board/report JSON file
(produced by whatever pipeline this repo already has) and layers grounded
LLM commentary on top. Does NOT touch, call, or modify any existing
fetch/scoring/gate logic — pure additive, read-only against the JSON file.

Usage:
    python3 rag_analyze.py --input data/some_board.json --llm ollama
    python3 rag_analyze.py --input data/some_board.json --llm gemini --llm-model gemini-2.0-flash

Input JSON shape: a list of game dicts, or a dict containing a list under
a common key ("games", "slate", "board", "cards"). Field names are matched
best-effort against common aliases so it works across different repos'
schemas without hardcoding one repo's internals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_provider import get_llm
from rag_context import SYSTEM_PROMPT, build_prompt

FIELD_ALIASES = {
    "home": ["home_team", "home", "homeTeam", "home_name"],
    "away": ["away_team", "away", "awayTeam", "away_name"],
    "commence": ["commence_time", "start_time", "game_time", "date", "commence"],
    "odds": ["odds", "price_line", "ml", "moneyline", "line"],
    "sport": ["sport", "league"],
}


def _first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _extract_games(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    if isinstance(data, dict):
        for key in ("games", "slate", "board", "cards"):
            if isinstance(data.get(key), list):
                return [g for g in data[key] if isinstance(g, dict)]
    return []


def build_context_line(game: dict[str, Any]) -> str:
    home = _first(game, FIELD_ALIASES["home"]) or "not in provided data"
    away = _first(game, FIELD_ALIASES["away"]) or "not in provided data"
    commence = _first(game, FIELD_ALIASES["commence"]) or "not in provided data"
    odds = _first(game, FIELD_ALIASES["odds"]) or "not in provided data"
    sport = _first(game, FIELD_ALIASES["sport"]) or "not in provided data"
    return (
        "CONTEXT:\n"
        f"- sport: {sport}\n"
        f"- matchup: {away} @ {home}\n"
        f"- commence_time: {commence}\n"
        f"- odds: {odds}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to board/report JSON file")
    parser.add_argument("--llm", default=None, help="ollama|gemini|deepseek|openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--output", default=None, help="Where to write analysis (default: <input>.analysis.md)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text())
    games = _extract_games(data)
    if not games:
        sys.exit("No game-like entries found in input JSON (expected a list of dicts, or a dict with 'games'/'slate'/'board'/'cards' key).")

    try:
        llm = get_llm(args.llm, args.llm_model)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"LLM init failed: {exc}")

    print(f"RAG analysis via: {llm.name} — {len(games)} games")
    lines = [f"# RAG Analysis ({llm.name})", ""]
    for game in games:
        context = build_context_line(game)
        prompt = build_prompt(context, "Give one factual, betting-relevant note using only the CONTEXT above.")
        try:
            note = llm.generate(prompt, system=SYSTEM_PROMPT).strip()
        except Exception as exc:
            note = f"[ERROR] {exc}"
        lines.append(context.replace("CONTEXT:\n", ""))
        lines.append(f"- LLM note: {note}")
        lines.append("")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".analysis.md")
    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
