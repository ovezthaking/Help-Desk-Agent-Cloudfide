"""Unit tests for the search_policies tool (chunking, BM25, and hybrid)."""

from __future__ import annotations

from help_desk_agent.tools.policies import load_policy_chunks, search_policies


def test_load_policy_chunks() -> None:
    """Verify markdown documents split into section chunks with metadata."""
    chunks = load_policy_chunks()
    assert len(chunks) >= 15
    sources = {c.source_file for c in chunks}
    assert sources == {"prices.md", "cancellations.md", "faq.md"}


def test_search_policies_opening_hours() -> None:
    """Verify searching for Saturday opening hours returns FAQ opening info."""
    res = search_policies("What time do you open on Saturday?", mode="bm25")
    assert res["mode_used"] == "bm25"
    assert len(res["results"]) > 0
    top_text = "\n".join(r["content"] for r in res["results"])
    assert "08:00" in top_text
    assert "20:00" in top_text


def test_search_policies_ebike_price() -> None:
    """Verify searching for e-bike day price returns prices.md bike rental."""
    res = search_policies(
        "How much is an e-bike for a whole day?",
        mode="bm25",
    )
    assert len(res["results"]) > 0
    assert res["results"][0]["source_file"] == "prices.md"
    assert "40 EUR" in res["results"][0]["content"]


def test_search_policies_age_limits() -> None:
    """Verify searching for 14-year-old e-bike rental retrieves age limits."""
    res = search_policies(
        "Can my 14-year-old rent an e-bike?",
        mode="bm25",
    )
    top_sections = [r["section_title"] for r in res["results"]]
    assert "Are there age limits?" in top_sections


def test_search_policies_synonyms_puppy_dog() -> None:
    """Verify synonym expansion finds dog rule when asking about a puppy."""
    res = search_policies("Can I bring my puppy in the basket?", mode="bm25")
    assert len(res["results"]) > 0
    top_text = "\n".join(r["content"] for r in res["results"])
    assert "Dogs are not allowed" in top_text


def test_search_policies_hybrid_with_mock_embedder() -> None:
    """Verify hybrid search works deterministically with injected embed_fn."""

    def fake_embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            low = t.lower()
            v0 = 1.0 if ("storm" in low or "weather" in low) else 0.1
            v1 = 1.0 if ("price" in low or "eur" in low) else 0.1
            vectors.append([v0, v1])
        return vectors

    res = search_policies(
        "bad weather storm warning refund",
        mode="hybrid",
        embed_fn=fake_embed,
    )
    assert res["mode_used"] == "hybrid"
    assert len(res["results"]) > 0
    top_titles = [r["section_title"] for r in res["results"]]
    assert "Bad weather" in top_titles


def test_search_policies_empty_query() -> None:
    """Verify empty query returns empty results without crashing."""
    res = search_policies("   ", mode="bm25")
    assert res["results"] == []
    assert "email" in res["staff_contact"]
