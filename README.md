# CityWheels — Help Desk AI Agent

Command-line AI Help Desk Agent for the **CityWheels** bike rental shop, built with Python 3.12, `uv`, Google Gemini (`google-genai`), deterministic tool calling, and an automated evaluation suite.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# 2. Install dependencies
uv sync

# 3. Ask a question
uv run help-desk-agent "What time do you open on Saturday?"
uv run help-desk-agent -v "I want to cancel booking CW-1002. How much will I get back?"

# 4. Interactive chat mode (with conversation memory)
uv run help-desk-agent --interactive

# 5. Run linter and unit/loop tests (100% offline, no API key required)
uv run ruff check .
uv run pytest

# 6. Run automated evaluation suite (14 questions -> eval/results.md)
uv run python -m eval.run_eval
```

## Documentation

- [`SOLUTION.md`](SOLUTION.md) — Full architecture explanation, model choice rationale, evaluation results, BM25 vs Hybrid RAG comparison, AI assistant review, and limitations.
- [`eval/results.md`](eval/results.md) — Latest evaluation run report (**14/14 passed**).
