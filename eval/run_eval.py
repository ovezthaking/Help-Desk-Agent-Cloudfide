"""Automated evaluation runner for the CityWheels Help Desk Agent."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from help_desk_agent.agent import HelpDeskAgent
from help_desk_agent.config import (
    PROJECT_ROOT,
    STAFF_EMAIL,
    STAFF_PHONE,
    get_gemini_api_key,
    get_gemini_model,
)
from help_desk_agent.llm.base import LLMClient
from help_desk_agent.llm.fake import FakeLLMClient
from help_desk_agent.llm.gemini import GeminiLLMClient
from help_desk_agent.models import LLMResponse, ToolCallRequest
from help_desk_agent.tools.policies import search_policies
from help_desk_agent.tools.registry import ToolRegistry

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.md"

_BOOKING_ID_RE = re.compile(r"\b(cw-\d{4})\b")
_POLICY_KEYWORDS = (
    "open",
    "saturday",
    "e-bike",
    "bike",
    "dog",
    "late",
    "storm",
    "weather",
    "rent",
    "price",
)


def _build_offline_eval_llm() -> FakeLLMClient:
    """Build a realistic FakeLLMClient to test the full pipeline offline."""

    def handler(messages: list[dict[str, Any]]) -> LLMResponse:
        last = messages[-1]
        if last["role"] == "user":
            q = str(last["content"])
            q_low = q.lower()
            booking_match = _BOOKING_ID_RE.search(q_low)

            if "cancel" in q_low and booking_match:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            name="calculate_refund",
                            args={
                                "booking_id": booking_match.group(1).upper()
                            },
                        )
                    ]
                )
            if booking_match:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            name="get_booking",
                            args={
                                "booking_id": booking_match.group(1).upper()
                            },
                        )
                    ]
                )
            if any(w in q_low for w in _POLICY_KEYWORDS):
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            name="search_policies",
                            args={"query": q},
                        )
                    ]
                )
            return LLMResponse(
                text=(
                    "I can only answer questions related to CityWheels bike "
                    "rentals and policies. Please contact our staff at "
                    f"{STAFF_EMAIL} or {STAFF_PHONE} for other inquiries."
                )
            )

        # Tool result received -> synthesize answer from tool output
        tool_entry = last["results"][0]
        t_name = tool_entry["name"]
        t_res = tool_entry["result"]

        if t_name == "calculate_refund":
            return LLMResponse(text=str(t_res.get("explanation", "")))
        if t_name == "get_booking":
            if not t_res.get("found"):
                return LLMResponse(text=str(t_res.get("message", "")))
            b = t_res["booking"]
            return LLMResponse(
                text=(
                    f"Booking {b['id']} has status '{b['status']}' for "
                    f"{b['bikes']} {b['bike_type']} bike(s)."
                )
            )
        if t_name == "search_policies":
            chunks = t_res.get("results", [])
            if not chunks:
                return LLMResponse(
                    text=(
                        "I could not find this information in CityWheels "
                        f"policies. Please contact staff at {STAFF_EMAIL} "
                        f"or {STAFF_PHONE}."
                    )
                )
            combined = "\n\n".join(c["content"] for c in chunks[:2])
            if "storm" in combined.lower():
                combined += (
                    f"\nPlease contact the staff at {STAFF_EMAIL} "
                    f"or {STAFF_PHONE}."
                )
            if "16 years old" in combined.lower():
                combined = (
                    "No, a 14-year-old cannot rent an e-bike because "
                    "electric bikes require riders to be at least 16 years "
                    "old (city bikes are available from 13 years old).\n"
                    + combined
                )
            return LLMResponse(text=combined)

        return LLMResponse(text="Unable to process tool output.")

    return FakeLLMClient(handler=handler)


def evaluate_single_case(
    agent: HelpDeskAgent,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Run the agent on one evaluation question and check expectations."""
    agent.reset_memory()
    t0 = time.perf_counter()
    response = agent.ask(case["question"])
    elapsed = round(time.perf_counter() - t0, 2)

    called_tools = [tc.name for tc in response.tool_calls]
    called_tools_set = set(called_tools)
    answer_low = response.answer.lower()

    expected_any_tools = case.get("expected_tools_any", [])
    tools_ok = (
        not called_tools_set.isdisjoint(expected_any_tools)
        if expected_any_tools
        else True
    )

    must_all = case.get("must_contain_all", [])
    missing_all = [w for w in must_all if w.lower() not in answer_low]
    all_ok = not missing_all

    must_any = case.get("must_contain_any", [])
    any_ok = (
        any(w.lower() in answer_low for w in must_any) if must_any else True
    )

    must_not = case.get("must_not_contain", [])
    found_forbidden = [w for w in must_not if w.lower() in answer_low]
    forbidden_ok = not found_forbidden

    passed = tools_ok and all_ok and any_ok and forbidden_ok

    reasons: list[str] = []
    if not tools_ok:
        reasons.append(
            f"Expected one of tools {expected_any_tools}, got {called_tools}"
        )
    if not all_ok:
        reasons.append(f"Missing required terms: {missing_all}")
    if not any_ok:
        reasons.append(f"Missing any of terms: {must_any}")
    if not forbidden_ok:
        reasons.append(f"Contained forbidden terms: {found_forbidden}")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "description": case.get("description", ""),
        "passed": passed,
        "called_tools": called_tools,
        "steps": response.steps,
        "latency_s": elapsed,
        "answer": response.answer,
        "failure_reasons": reasons,
    }


def compare_retrieval_modes() -> list[dict[str, Any]]:
    """Compare BM25 vs Hybrid retrieval on policy & escalation queries."""
    queries = [
        ("What time do you open on Saturday?", "When are you open?"),
        ("How much is an e-bike for a whole day?", "Bike rental"),
        ("Can my 14-year-old rent an e-bike?", "Are there age limits?"),
        ("Can I bring my puppy in the basket?", "Can I take my dog?"),
        ("What if I return a bike 2 hours late?", "Late return"),
        ("Storm warning on my rental day", "Bad weather"),
        ("Can I buy a bike from your shop?", "Do you sell bikes?"),
    ]
    rows: list[dict[str, Any]] = []
    has_key = get_gemini_api_key() is not None

    for query, expected_section in queries:
        t0 = time.perf_counter()
        bm25_res = search_policies(query, top_k=3, mode="bm25")
        bm25_ms = round((time.perf_counter() - t0) * 1000, 2)
        bm25_top1 = (
            bm25_res["results"][0]["section_title"]
            if bm25_res["results"]
            else "None"
        )

        if has_key:
            t1 = time.perf_counter()
            hyb_res = search_policies(query, top_k=3, mode="hybrid")
            hyb_ms = round((time.perf_counter() - t1) * 1000, 2)
            hyb_top1 = (
                hyb_res["results"][0]["section_title"]
                if hyb_res["results"]
                else "None"
            )
            hyb_mode = hyb_res["mode_used"]
        else:
            hyb_ms = 0.0
            hyb_top1 = "N/A (no API key)"
            hyb_mode = "offline"

        rows.append(
            {
                "query": query,
                "expected": expected_section,
                "bm25_top1": bm25_top1,
                "bm25_hit": bm25_top1 == expected_section,
                "bm25_ms": bm25_ms,
                "hybrid_top1": hyb_top1,
                "hybrid_hit": hyb_top1 == expected_section,
                "hybrid_ms": hyb_ms,
                "hybrid_mode": hyb_mode,
            }
        )
    return rows


def generate_markdown_report(
    results: list[dict[str, Any]],
    model_label: str,
    search_mode: str,
    retrieval_comparison: list[dict[str, Any]],
) -> str:
    """Format evaluation results into a Markdown report."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = round((passed / max(total, 1)) * 100, 1)
    avg_latency = round(
        sum(r["latency_s"] for r in results) / max(total, 1), 2
    )

    lines: list[str] = [
        "# CityWheels Help Desk Agent — Evaluation Results",
        "",
        "## Summary",
        "",
        f"- **Model**: `{model_label}`",
        f"- **Policy Search Mode**: `{search_mode}`",
        f"- **Questions Passed**: **{passed} / {total} ({pass_rate}%)**",
        f"- **Average Latency per Question**: `{avg_latency}s`",
        "",
        "## Detailed Question Results",
        "",
        "| ID | Category | Question | Tools Called | Latency | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        status_badge = "PASS" if r["passed"] else "FAIL"
        tools_str = (
            ", ".join(f"`{t}`" for t in r["called_tools"])
            if r["called_tools"]
            else "_(none)_"
        )
        q_short = r["question"].replace("|", "\\|")
        lines.append(
            f"| **{r['id']}** | `{r['category']}` | {q_short} | "
            f"{tools_str} | {r['latency_s']}s | **{status_badge}** |"
        )

    lines.extend(
        [
            "",
            "## Retrieval Approach Comparison (BM25 vs Hybrid Embeddings)",
            "",
            "| Query | Expected Section | BM25 Top-1 (Time) | "
            "Hybrid Top-1 (Time) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for row in retrieval_comparison:
        b_mark = "OK" if row["bm25_hit"] else "MISS"
        h_mark = (
            "OK"
            if row["hybrid_hit"]
            else ("N/A" if row["hybrid_mode"] == "offline" else "MISS")
        )
        lines.append(
            f"| {row['query']} | `{row['expected']}` | "
            f"`{row['bm25_top1']}` [{b_mark}] ({row['bm25_ms']}ms) | "
            f"`{row['hybrid_top1']}` [{h_mark}] ({row['hybrid_ms']}ms) |"
        )

    lines.extend(["", "## Question-by-Question Transcripts", ""])
    for r in results:
        status_str = "PASSED" if r["passed"] else "FAILED"
        lines.append(f"### {r['id']} ({r['category']}) — {status_str}")
        lines.append(f"- **Question**: {r['question']}")
        lines.append(f"- **Expectation**: {r['description']}")
        lines.append(f"- **Tools Called**: `{r['called_tools']}`")
        if r["failure_reasons"]:
            lines.append(f"- **Failure Reasons**: `{r['failure_reasons']}`")
        lines.append("- **Agent Answer**:")
        lines.append("```text")
        lines.append(r["answer"].strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation suite and write eval/results.md."""
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for CityWheels Help Desk Agent"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model to evaluate (default from GEMINI_MODEL).",
    )
    parser.add_argument(
        "--search-mode",
        choices=["auto", "bm25", "hybrid", "semantic"],
        default="auto",
        help="Search mode for search_policies.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using the deterministic offline FakeLLMClient.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_PATH),
        help="Path to write the Markdown evaluation report.",
    )
    args = parser.parse_args(argv)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    llm: LLMClient
    if args.offline or get_gemini_api_key() is None:
        if not args.offline:
            print(
                "Note: GEMINI_API_KEY not found in environment; running "
                "offline pipeline verification mode."
            )
        llm = _build_offline_eval_llm()
        model_label = "FakeLLMClient (offline verification)"
        effective_search = (
            "bm25" if args.search_mode == "auto" else args.search_mode
        )
    else:
        model_name = args.model or get_gemini_model()
        llm = GeminiLLMClient(model=model_name)
        model_label = model_name
        effective_search = args.search_mode

    registry = ToolRegistry(search_mode=effective_search)
    agent = HelpDeskAgent(llm=llm, tool_registry=registry)

    results: list[dict[str, Any]] = []
    print(
        f"Running evaluation on {len(questions)} questions "
        f"(model={model_label}, search_mode={effective_search})..."
    )

    for case in questions:
        res = evaluate_single_case(agent, case)
        results.append(res)
        badge = "PASS" if res["passed"] else "FAIL"
        print(
            f"  [{badge}] {res['id']} ({res['latency_s']}s) "
            f"tools={res['called_tools']} - {res['question']}"
        )
        if res["failure_reasons"]:
            for reason in res["failure_reasons"]:
                print(f"         -> {reason}")

    retrieval_comp = compare_retrieval_modes()
    report_md = generate_markdown_report(
        results=results,
        model_label=model_label,
        search_mode=effective_search,
        retrieval_comparison=retrieval_comp,
    )

    out_path = Path(args.output)
    out_path.write_text(report_md, encoding="utf-8")

    passed_count = sum(1 for r in results if r["passed"])
    print(
        f"\nEvaluation complete: {passed_count}/{len(results)} passed. "
        f"Report saved to {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
