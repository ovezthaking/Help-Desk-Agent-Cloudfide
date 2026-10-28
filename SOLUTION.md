# SOLUTION — CityWheels Help Desk AI Agent

## 1. Installation and Running Guide

You can run the agent, tests, and evaluation either locally with `uv` (Python 3.12+) or inside containers using **Docker Compose**.

### Configuration (`.env`)

Copy `.env.example` to `.env` and set your Google Gemini API key:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY="your-real-api-key"
```

---

### Option A: Local Execution with `uv`

1. **Install dependencies**:
   ```bash
    uv sync
   ```
2. **Ask a single question from the command line**:
   ```bash
    uv run help-desk-agent "What time do you open on Saturday?"
    uv run help-desk-agent -v "I want to cancel booking CW-1002. How much will I get back?"
   ```

    *(Use* `-v` */* `--verbose` *to inspect the exact tool calls and arguments executed by the agent).*
3. **Run interactive multi-turn mode (with conversation memory)**:
   ```bash
    uv run help-desk-agent --interactive
   ```

    Example follow-up conversation:
   - `You:` *How much will I get back if I cancel booking CW-1002?*
   - `You:` *And what if I cancel tomorrow at 10:00 instead?*
4. **Run all unit and agent-loop tests (single command, 100% offline, no API key needed)**:
   ```bash
    uv run pytest
   ```
5. **Run the linter**:
   ```bash
    uv run ruff check .
   ```
6. **Run the automated evaluation suite**:
   ```bash
    uv run python -m eval.run_eval
   ```

    *(Writes the full report to* [`eval/results.md`](eval/results.md)*. You can also run* `uv run python -m eval.run_eval --offline` *to test the evaluation harness without an API key).*

---

### Option B: Execution with Docker Compose

I provide a multi-service [`docker-compose.yml`](docker-compose.yml) built on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`:

1. **Ask a question or start interactive chat**:
   ```bash
    docker compose run --rm agent "How much is an e-bike for a whole day?"
    docker compose run --rm agent --interactive
   ```
2. **Run linter and all tests (offline)**:
   ```bash
    docker compose run --rm test
   ```
3. **Run the evaluation suite (updates** `./eval/results.md` **on the host)**:
   ```bash
    docker compose run --rm eval
   ```

---

## 2. Model and Provider Choice

- **Provider**: **Google Gemini API** accessed directly via the official `google-genai` Python SDK (`v2.33+`).
- **Generation Model**: `gemini-3.5-flash` (with automatic fallback to `gemini-3-flash-preview` and `gemini-3.1-flash-lite` on transient `503`/`429` spikes) at `temperature=0.0`.
- **Embedding Model**: `gemini-embedding-001` (with fallback to `gemini-embedding-2`).
- **Why this choice?**
  1. **Generous Free Tier**: Google AI Studio provides free API access for both generation and embeddings, allowing any reviewer to test the project without cost.
  2. **Strong Native Tool Calling**: Gemini Flash models support multi-step parallel and sequential function calling with high schema adherence and low latency (~1.7s per turn).
  3. **Direct SDK over Heavy Frameworks**: Instead of hiding the agent loop inside LangChain or LangGraph, calling `google-genai` directly behind a thin `LLMClient` protocol (`src/help_desk_agent/llm/base.py`) keeps the control flow 100% transparent, easy to debug, and trivial to mock with `FakeLLMClient` in unit tests.

---

## 3. How the Agent Works

### Architecture Overview

```text
User Question (CLI)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ HelpDeskAgent.ask()  (src/help_desk_agent/agent.py)      │
│  • Appends question to conversation memory               │
│  • Loops up to max_steps=6                               │
│  • Calls LLMClient.generate(messages, system_prompt, ...)│
└──────────────┬───────────────────────────────────────────┘
               │
       ┌───────┴────────────────────┐
       │ Has tool_calls?            │ No tool_calls?
       ▼ Yes                        ▼
┌─────────────────────────────┐  ┌─────────────────────────┐
│ ToolRegistry.execute()      │  │ Return AgentResponse    │
│  • search_policies          │  │  • Final grounded text  │
│  • get_booking              │  │  • Recorded tool_calls  │
│  • calculate_refund         │  └─────────────────────────┘
│  (appends result to history │
│   and repeats loop)         │
└─────────────────────────────┘
```

### The Three Tools (`src/help_desk_agent/tools/`)

1. `search_policies(query: str)` ([`policies.py`](src/help_desk_agent/tools/policies.py))
   - Splits `data/policies/prices.md`, `cancellations.md`, and `faq.md` by Markdown `##`  headings into 20 self-contained semantic chunks (`PolicyChunk`), preserving document and section titles.
   - Supports three retrieval modes:
     - `bm25`: Lexical BM25 with title-token boosting (`2x`) and lightweight domain synonym expansion (e.g. `"puppy"` -> `"dog"`, `"saturday"` -> `"open"`). Works 100% offline.
     - `semantic`: Cosine similarity over vectors from `gemini-embedding-001` cached in memory / `data/.embeddings_cache.json`.
     - `hybrid` **/** `auto`: Combines normalized BM25 and semantic cosine similarity when an API key is available, and falls back to `bm25` automatically in offline environments.
2. `get_booking(booking_id: str)` ([`bookings.py`](src/help_desk_agent/tools/bookings.py))
   - Normalizes the booking ID (case-insensitive, strips whitespace) and looks it up in `data/bookings.json`.
   - Returns `{"found": True, "booking": {...}}` or a safe `{"found": False, "booking_id": ..., "message": ...}` without ever raising an unhandled exception.
3. `calculate_refund(booking_id: str, cancel_at: str | None = None)` ([`refunds.py`](src/help_desk_agent/tools/refunds.py))
   - Computes refund amounts **100% deterministically in Python code**—never delegated to the LLM:
     - Uses the fixed current date `2026-06-10T09:00:00+02:00` by default, or an optional ISO timestamp `cancel_at` (which enables accurate follow-up questions like *"what if I cancel tomorrow at 10:00 instead?"*).
     - Status `in_progress`, `completed`, or `cancelled` -> `can_cancel = False`, `0.00 EUR` refund.
     - Status `confirmed`:
       - '>= 48h before start -> **100%** of `rental_price_eur` + **100%** of `deposit_eur`.
       - 24h <= time < 48h before start -> **50%** of `rental_price_eur` + **100%** of `deposit_eur`.
       - < 24h before start -> **0%** of `rental_price_eur` + **100%** of `deposit_eur`.
     - `booking_fee_eur` is always non-refundable (`0.00 EUR`).
     - Handles payment method rules: `card` refunds go to the same card within 5 working days; `voucher` rental refunds are issued as a **new voucher of the same value** (never as cash), while the deposit returns the same way it was paid.

### System Prompt Instructions ([`prompts.py`](src/help_desk_agent/prompts.py))

The system prompt enforces four strict guardrails:

1. **Fixed Reference Clock**: Injects `2026-06-10 09:00, Europe/Warsaw time (UTC+02:00)`.
2. **Mandatory Tool Usage**: Instructs the model to always call `search_policies` for policy/price/FAQ questions, `get_booking` for booking lookups, and `calculate_refund` for any cancellation/refund question.
3. **Zero Mental Math**: Prohibits the LLM from calculating refunds itself; it must report the exact figures returned by `calculate_refund`.
4. **Strict Grounding & Refusal**: Prohibits answering off-topic questions (e.g. capital of Australia, cooking recipes) and directs customers to CityWheels staff (`help@citywheels.example`, `+48 000 000 000`) whenever an answer is not in the documents or requires staff discretion (storm warnings, stolen bikes).

---

## 4. Evaluation Results & Troubleshooting

The evaluation dataset in [`eval/questions.json`](eval/questions.json) contains **14 test questions** across 4 categories:

- **5 policy/FAQ questions** (`Q01`–`Q05`), including a multi-hop question (`Q05`: returning a city bike 2 hours late, which requires combining the `1.5x` late fee rule with the `4 EUR/h` city bike rate).
- **5 refund calculation questions** (`Q06`–`Q10`), covering 24–48h (`CW-1002`), <24h deposit-only (`CW-1003`), voucher refund (`CW-1004`), `in_progress` non-cancellable (`CW-1008`), and the exact 48h boundary (`CW-1007`).
- **1 unknown booking lookup** (`Q11`: `CW-9999`).
- **3 refusal / staff escalation questions** (`Q12`–`Q14`: capital of Australia, baking a cake, and storm warning refund policy).

### Final Evaluation Score: **14 / 14 Passed (100.0%)**

Full results and transcripts are committed in [`eval/results.md`](eval/results.md).

### What Failed During Development and How I Fixed It

1. **Failure #1 — Retired Model** `404 NOT_FOUND` **(**`gemini-2.5-flash` **&** `text-embedding-004`**)**:
   - *What happened*: Our first evaluation run failed immediately with `404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new users`.
   - *Why*: Google retired `gemini-2.5-flash` and `text-embedding-004` for newer API keys in favor of the Gemini 3.x family and `gemini-embedding-001`.
   - *Fix*: Queried `client.models.list()` to inspect active models on the account and updated the defaults.
2. **Failure #2 — Backend** `503 UNAVAILABLE` **/** `504 DEADLINE_EXCEEDED` **on** `gemini-3.8-flash`:
   - *What happened*: During the second evaluation run using `gemini-3.8-flash`, questions `Q01`–`Q06` passed, but each took 130–480s because the `google-genai` internal retry loop was hitting `503 UNAVAILABLE (high demand)` on `gemini-3.8-flash`, and `Q07` eventually aborted with a 503 error.
   - *Why*: `gemini-3.8-flash` was experiencing server-side congestion on the free tier.
   - *Fix*: Benchmarked all available Flash models (`gemini-3.5-flash` responded in `1.71s`, `gemini-3-flash-preview` in `1.91s`), set an explicit HTTP timeout (`30s`), switched the primary model to `gemini-3.5-flash`, and added an automatic fallback chain in `GeminiLLMClient.generate()` so transient `503`/`504`/`429` errors transparently fail over to the next available Flash model. After this fix, all 14/14 questions passed with an average latency of `10.35s`.

---

## 5. Comparison of Two Approaches (Extras)

I compared **two retrieval approaches** inside `search_policies` across 7 representative customer queries (full table in [`eval/results.md`](eval/results.md)):

| Metric | Approach A: Lexical BM25 + Synonyms | Approach B: Hybrid (`BM25` + `gemini-embedding-001`) |
| :--- | :--- | :--- |
| **Top-1 Section Accuracy (7 queries)** | **7 / 7 (100%)** | **7 / 7 (100%)** |
| **Retrieval Latency per Query** | **2.7 ms – 5.1 ms** | **435 ms – 843 ms** (first call) / **<5 ms** (cached) |
| **Offline / Zero-Key Capability** | Yes (100% offline) | Requires API key on cache miss |
| **Paraphrase Resilience** | Relies on curated `SYNONYM_MAP` | High (768+ dim dense representations) |

**What I found**:

- For a small, well-structured policy corpus (~20 Markdown sections), **section-level chunking + BM25 with title boosting** is remarkably effective and 100x faster (~4ms vs ~500ms uncached) than calling a remote embedding endpoint.
- However, **Hybrid Search** provides a valuable safety net when customers use unexpected phrasing not covered by a static synonym dictionary, while local caching (`data/.embeddings_cache.json`) eliminates the latency penalty on subsequent queries.

---

## 6. How AI Coding Assistants Were Used

An AI coding assistant was used during development to scaffold boilerplate and an implementation plan (Pydantic models, CLI argument parsing, initial unit test cases, and Markdown report formatting). Then I reviewed the code generated by the assistant.

### Concrete Example Where AI Output Was Wrong and Needed Changes

1. **Outdated Gemini Model Identifiers (**`gemini-2.5-flash` **and** `text-embedding-004`**)**:
   - *What the AI generated*: Based on its training data, the assistant configured `GEMINI_MODEL="gemini-2.5-flash"` and `GEMINI_EMBEDDING_MODEL="text-embedding-004"` in `config.py` and `.env.example`.
   - *How I noticed*: When executing `uv run python -m eval.run_eval` against the live Gemini API, the SDK raised `ClientError: 404 NOT_FOUND` stating `models/gemini-2.5-flash is no longer available to new users`. Furthermore, when switching to `gemini-3.8-flash`, I observed 200+ second latencies caused by `503 UNAVAILABLE` retries.
   - *How I fixed it*: I wrote a diagnostic script calling `client.models.list()` and benchmarking live response times across all returned model IDs. I discovered that `gemini-3.5-flash` (for generation) and `gemini-embedding-001` (for embeddings) were the active, low-latency models on the endpoint, and I implemented an automatic model fallback list with a 30s HTTP timeout in `GeminiLLMClient`.
2. **Ruff Line-Length Violations (**`E501`**)**:
   - *What the AI generated*: In `tests/test_policies.py`, the generated docstrings exceeded the strict `line-length = 79` PEP 8 limit configured in `pyproject.toml` (80 and 81 characters).
   - *How I noticed*: Running `uv run ruff check .` flagged two `E501 Line too long` errors before running `pytest`.
   - *How I fixed it*: Shortened the docstrings and verified with `uv run ruff check .`.
3. **Unnecessary Nested Loops and Suboptimal Time Complexity in** `policies.py` **&** `bookings.py`:
   - *What the AI generated*: In `src/help_desk_agent/tools/policies.py`, the initial implementation of `compute_bm25_scores`, `load_policy_chunks`, and `search_policies` relied on deeply nested `for` loops and `if/elif/else` chains—re-tokenizing every document on each query, manually counting term and document frequencies via `dict.get(tok, 0) + 1` across the entire corpus vocabulary, recomputing `idf = math.log(...)` inside the inner document loop, and sorting the full list of chunks (O(N log N)) instead of selecting only the top-k. Similarly, `bookings.py` re-read `bookings.json` and performed an O(N) linear list scan on every lookup, and `refunds.py` serialized the `Booking` model to a dict only to immediately re-validate it back into a `Booking` instance.
   - *How I noticed*: During code review of the tool implementations, I spotted the nested loops, repeated regex compilations, linear list scans, and avoidable per-document dictionary rebuilds.
   - *How I fixed it*: I refactored `policies.py` to use `collections.Counter` and `@lru_cache` (`_chunk_token_stats`, `_normalize_token`), precomputed weighted `idf` once per unique query token before the document loop, replaced inner-loop membership checks with O(min(|Q|, |D|)) set intersections (`tf_map.keys() & query_term_set`), split Markdown sections with a precompiled multiline regex (`_SECTION_SPLIT_RE`), used C-backed `math.hypot` and `operator.mul` in `cosine_similarity`, and switched top-k selection to `heapq.nlargest` (O(N log k)). In `bookings.py` and `refunds.py`, I added an `mtime`-invalidated `@lru_cache` dictionary index (`{id: Booking}`) for O(1) lookups (`find_booking_model`) and `frozenset` status checks.

---

## 7. Limitations of Current Solution

1. **Read-Only Bookings**: The agent can look up bookings (`get_booking`) and calculate refunds (`calculate_refund`), but it does not mutate `data/bookings.json` to actually perform the cancellation.
2. **In-Memory Conversation State**: Conversation memory in `--interactive` mode is kept in process memory (`self.messages`) and is not persisted across CLI restarts.
3. **Substring/Keyword Assertions in Auto-Eval**: `eval/run_eval.py` checks tool call traces and required/forbidden terms deterministically. While fast and reproducible, it does not use an LLM-as-a-Judge to grade subtle tone or formatting nuances.
4. **No Authentication**: Any user who knows a booking ID (`CW-1002`) can query its details; in production, booking lookups would require verifying customer identity.

---

## 8. What I Would Do Next With More Time

1. **Add Write Actions with Human-in-the-Loop Confirmation**: Implement a `cancel_booking(booking_id)` tool that requires explicit user confirmation (`y/N`) before updating booking status to `cancelled` in a real database (PostgreSQL/SQLite).
2. **Add an LLM-as-a-Judge Grader to** `eval/run_eval.py`: Complement the deterministic keyword/tool checks with a rubric-based judge that scores faithfulness, completeness, and tone on a 1–5 scale.
3. **Price Estimation Tool (**`calculate_rental_quote`**)**: Just as `calculate_refund` prevents LLM arithmetic errors on cancellations, a deterministic `calculate_rental_quote(bike_type, hours_or_days, add_ons)` tool would guarantee 100% accurate quotes for complex multi-day rentals with the 7-day 15% discount and >6h daily cap.
4. **Persistent Session Store**: Store conversation history in SQLite keyed by `session_id` so customers can resume conversations later.
