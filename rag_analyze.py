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

from llm_provider import get_llm
from rag_context import SYSTEM_PROMPT, build_game_context, build_prompt, extract_games


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
    games = extract_games(data)
    if not games:
        sys.exit("No game-like entries found in input JSON (expected a list of dicts, or a dict with 'games'/'slate'/'board'/'cards' key).")

    try:
        llm = get_llm(args.llm, args.llm_model)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"LLM init failed: {exc}")

    print(f"RAG analysis via: {llm.name} — {len(games)} games")
    lines = [f"# RAG Analysis ({llm.name})", ""]
    for game in games:
        context = build_game_context(game)
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
