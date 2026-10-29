"""Command-line interface for the CityWheels Help Desk Agent."""

from __future__ import annotations

import argparse
import sys

from help_desk_agent.agent import HelpDeskAgent
from help_desk_agent.llm.gemini import GeminiLLMClient
from help_desk_agent.tools.registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="help-desk-agent",
        description="CityWheels AI Help Desk Agent (Gemini + Tool Calling)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Customer question to ask the agent.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn chat session with memory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print tool calls and arguments executed during the turn.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Gemini model name (default from GEMINI_MODEL).",
    )
    parser.add_argument(
        "--search-mode",
        choices=["auto", "bm25", "hybrid", "semantic"],
        default="auto",
        help="Retrieval mode for search_policies (default: auto).",
    )
    return parser


def _print_response(
    agent: HelpDeskAgent,
    question: str,
    verbose: bool = False,
) -> None:
    response = agent.ask(question)
    if verbose and response.tool_calls:
        print("\n[Tool Calls]")
        for idx, tc in enumerate(response.tool_calls, start=1):
            print(f"  {idx}. {tc.name}({tc.args})")
        print()
    print(response.answer)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the help-desk-agent CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.question and not args.interactive:
        parser.print_help()
        return 0

    try:
        llm = GeminiLLMClient(model=args.model)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    registry = ToolRegistry(search_mode=args.search_mode)
    agent = HelpDeskAgent(llm=llm, tool_registry=registry)

    if args.question:
        try:
            _print_response(agent, args.question, verbose=args.verbose)
        except (RuntimeError, ValueError, OSError, ConnectionError) as exc:
            print(f"Error communicating with LLM: {exc}", file=sys.stderr)
            return 1
        if not args.interactive:
            return 0

    print(
        "CityWheels Help Desk Interactive Mode "
        "(type 'exit' or 'quit' to stop, 'reset' to clear memory)"
    )
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset_memory()
            print("[Conversation memory cleared]")
            continue

        try:
            print("\nAgent: ", end="", flush=True)
            _print_response(agent, user_input, verbose=args.verbose)
        except (RuntimeError, ValueError, OSError, ConnectionError) as exc:
            print(f"\nError: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
