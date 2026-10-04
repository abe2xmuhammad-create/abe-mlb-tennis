"""
Standalone RAG analysis layer. Reads an existing board/report JSON file
(produced by whatever pipeline this repo already has) and layers grounded
LLM commentary on top. Does NOT touch, call, or modify any existing
fetch/scoring/gate logic — pure additive, read-only against the JSON file.

Usage:
    python3 rag_analyze.py --input data/f5_ml_plays_2026-07-25.json --llm ollama

    # stronger reasoning, no extra API key (Ollama-cloud model):
    python3 rag_analyze.py --input board.json --llm ollama \
        --llm-model deepseek-v4-flash:cloud

    # hosted providers:
    python3 rag_analyze.py --input board.json --llm anthropic
    python3 rag_analyze.py --input board.json --llm gemini --llm-model gemini-3.5-flash

Exit status is meaningful:
    0  every game produced a note
    1  bad usage / unreadable input / no games
    2  the run completed but some games failed (report written, incomplete)
    3  --strict-notes was requested and a note dropped a sample-window qualifier

Usage note: if the requested provider is unreachable, the whole slate fails
rather than silently degrading. That is deliberate — four of the findings in
AUDIT.md were only invisible because failure used to look like success.

Input JSON shape: a list of entry dicts, or a dict containing a list under
"games", "slate", "board", "cards", or "plays". Field names are matched
best-effort against common aliases, covering both Odds-API-shaped records
(home_team/away_team/commence_time) and slate-tracker board output
(matchup/time_et/line). Engine verdict fields (f5_play, f5_tier, ml_play,
ml_tier, pen_*, notes) are passed into CONTEXT verbatim when present.

Parallelism: each game is an independent, stateless request, so games are
analyzed concurrently (--workers, default 4, or RAG_WORKERS). Output order
always follows the input order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from llm_provider import LLMError, LLMProvider, get_llm
from rag_context import (
    SYSTEM_PROMPT,
    build_game_context,
    build_prompt,
    extract_games,
    find_window_violations,
)

# Exception types that mean "the provider had a problem", as opposed to a bug in
# this program. Anything else propagates with a traceback, where it belongs —
# the old blind `except Exception` filed TypeErrors as provider errors.
PROVIDER_EXCEPTIONS = (LLMError, requests.RequestException, KeyError, IndexError, TypeError)

# Environment variables whose values must never reach a written report.
SECRET_ENV_VARS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_KEY",
)

NOTE_QUESTION = "Give one factual, betting-relevant note using only the CONTEXT above."


def scrub(text: str) -> str:
    """Blank out any configured credential appearing in `text`.

    Provider errors used to be written into the report verbatim, and the Gemini
    key travelled in the request URL — so a single HTTP error could publish the
    key into a file that gets committed or pasted into a chat.
    """
    for var in SECRET_ENV_VARS:
        secret = os.getenv(var, "")
        if len(secret) >= 8:
            text = text.replace(secret, "***REDACTED***")
    return text


@dataclass
class GameResult:
    context: str
    note: str
    error: str | None = None
    violations: list[str] | None = None


def analyze_one(llm: LLMProvider, game: dict[str, Any]) -> GameResult:
    """Analyze a single game. Never raises for provider problems."""
    context = build_game_context(game)
    prompt = build_prompt(context, NOTE_QUESTION)
    try:
        note = llm.generate(prompt, system=SYSTEM_PROMPT).strip()
    except PROVIDER_EXCEPTIONS as exc:
        return GameResult(context, "", error=f"{type(exc).__name__}: {scrub(str(exc))}")

    if not note:
        # Providers raise on empty output, but a custom one might not. An empty
        # note used to be written as "- LLM note:" and pass for output.
        return GameResult(context, "", error="provider returned an empty note")

    return GameResult(context, note, violations=find_window_violations(note, context))


def render_report(llm_name: str, results: list[GameResult], failures: int) -> str:
    lines = [f"# RAG Analysis ({llm_name})", ""]
    if failures:
        lines.append(
            f"> **Incomplete:** {failures} of {len(results)} games produced no note. "
            "The contexts below are real; the missing notes are errors, not silence."
        )
        lines.append("")
    for result in results:
        # Drop the leading "CONTEXT:" label; the report structure supplies it.
        body = result.context.split("\n", 1)[1] if "\n" in result.context else result.context
        lines.append(body)
        if result.error:
            lines.append(f"- **LLM note: FAILED** — {result.error}")
        else:
            lines.append(f"- LLM note: {result.note}")
            for figure in result.violations or []:
                lines.append(
                    f"  - **WINDOW CHECK:** note states `{figure}` but CONTEXT does not "
                    "supply it in that form — possible dropped sample-window qualifier (rule 8)"
                )
        lines.append("")
    return "\n".join(lines)


def write_report(output_path: Path, text: str) -> None:
    """Write the report, creating parent directories and always using UTF-8.

    Without encoding="utf-8" a run under a C locale (cron, systemd, most CI
    images) raised UnicodeDecodeError on an accented player name — or died at
    this final write after every LLM call had already been paid for.
    """
    parent = output_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Overwriting existing report: {output_path}")
    output_path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to board/report JSON file")
    parser.add_argument(
        "--llm",
        default=None,
        help="ollama|gemini|deepseek|anthropic|openai (default: LLM_PROVIDER env, else ollama)",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--output", default=None, help="Where to write analysis (default: <input>.analysis.md)")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("RAG_WORKERS", "4")),
        help="Concurrent game analyses (default 4, env RAG_WORKERS)",
    )
    parser.add_argument(
        "--strict-notes",
        action="store_true",
        help="Exit non-zero if any note drops a sample-window qualifier from a rate stat",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Tolerate this many failed games before exiting 2 (default 0)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    if input_path.is_dir():
        sys.exit(f"Input path is a directory, expected a JSON file: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"Input file is not valid JSON: {input_path} ({exc})")
    except OSError as exc:
        sys.exit(f"Could not read {input_path}: {exc}")

    games = extract_games(data)
    if not games:
        sys.exit(
            "No game-like entries found in input JSON (expected a list of dicts, "
            "or a dict with 'games'/'slate'/'board'/'cards' key)."
        )

    try:
        llm = get_llm(args.llm, args.llm_model)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"LLM init failed: {exc}")

    workers = max(1, args.workers)
    print(f"RAG analysis via: {llm.name} ({llm.model}) — {len(games)} games, {workers} worker(s)")

    if workers == 1:
        results = [analyze_one(llm, game) for game in games]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda g: analyze_one(llm, g), games))

    failures = sum(1 for r in results if r.error)
    violations = [v for r in results for v in (r.violations or [])]

    output_path = Path(args.output) if args.output else input_path.with_suffix(".analysis.md")
    try:
        write_report(output_path, render_report(llm.name, results, failures))
    except OSError as exc:
        sys.exit(f"Could not write {output_path}: {exc}")
    print(f"Wrote {output_path}")

    if failures:
        print(
            f"{failures}/{len(games)} analyses failed — {output_path} is incomplete",
            file=sys.stderr,
        )
        if failures > args.max_failures:
            sys.exit(2)
    if violations and args.strict_notes:
        print(
            f"{len(violations)} sample-window qualifier violation(s): {violations}",
            file=sys.stderr,
        )
        sys.exit(3)
    if violations:
        print(
            f"warning: {len(violations)} sample-window qualifier violation(s) flagged "
            f"in {output_path}: {violations} (use --strict-notes to fail the run)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
