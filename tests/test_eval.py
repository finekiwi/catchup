"""Unit tests for eval/baseline.py, eval/before_after.py, and eval/golden_set.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import eval.baseline as baseline_module
import eval.before_after as ba_module
from eval.baseline import BaselineChunk, chunk_text, extract_text_pypdf, index_baseline, query_baseline
from eval.before_after import (
    ComparisonReport,
    ComparisonResult,
    TierStats,
    _keyword_score,
    _score_answer,
    run_comparison,
    save_report,
)
from rag.qa_chain import QAResult, SourceBlock

# ---------------------------------------------------------------------------
# Constants / shared helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR = [0.1] * 1536  # text-embedding-3-small output dimension

GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "golden_set.json"


def _mock_collection(count: int = 0, query_result: dict | None = None) -> MagicMock:
    """Build a minimal ChromaDB collection mock.

    Args:
        count: Value returned by collection.count().
        query_result: Optional dict returned by collection.query(); defaults to empty result.

    Returns:
        MagicMock configured to behave like a ChromaDB collection.
    """
    col = MagicMock()
    col.count.return_value = count
    col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    col.query.return_value = query_result or {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    return col


def _fake_qa_result(answer: str = "Test answer.", source: str = "doc.pdf") -> QAResult:
    """Build a minimal QAResult for mocking pipeline outputs.

    Args:
        answer: The answer text to embed in the result.
        source: Source filename for the single SourceBlock.

    Returns:
        A QAResult with one SourceBlock and dummy token counts.
    """
    return QAResult(
        question="What is X?",
        answer=answer,
        source_blocks=[
            SourceBlock(
                document_id="doc-001",
                source=source,
                block_order=0,
                block_type="text",
                content_preview="Some content.",
                page=1,
                cell_index=None,
            )
        ],
        model="gpt-4o-mini",
        latency_ms=50.0,
        input_tokens=100,
        output_tokens=40,
    )


def _patch_log(monkeypatch) -> None:
    """Suppress log_api_call DB writes during tests."""
    monkeypatch.setattr("eval.baseline.log_api_call", lambda **kw: None)


# ===========================================================================
# eval/baseline.py — chunk_text
# ===========================================================================


def test_chunk_text_basic():
    """chunk_text splits text into the expected number of non-empty chunks."""
    text = "A" * 2500
    chunks = chunk_text(text, source="file.pdf", chunk_size=1000, overlap=100)
    # step = 900; starts at 0, 900, 1800, 2700 — but 2700 > 2500, so 3 iterations
    assert len(chunks) == 3
    assert all(isinstance(c, BaselineChunk) for c in chunks)


def test_chunk_text_overlap():
    """Consecutive chunks share the correct character overlap."""
    text = "B" * 1200
    chunks = chunk_text(text, source="file.pdf", chunk_size=1000, overlap=200)
    # chunk 0: [0:1000], chunk 1: [800:1800] — 200-char shared region
    assert len(chunks) == 2
    assert chunks[0].content[-200:] == chunks[1].content[:200]


def test_chunk_text_empty():
    """chunk_text returns an empty list when text is blank or whitespace only."""
    assert chunk_text("", source="file.pdf") == []
    assert chunk_text("   \n\t  ", source="file.pdf") == []


def test_chunk_text_short_text():
    """Text shorter than chunk_size produces a single chunk containing all content."""
    text = "Short text."
    chunks = chunk_text(text, source="short.pdf", chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].content == text.strip()


def test_chunk_text_assigns_source():
    """Every BaselineChunk produced carries the correct source filename."""
    text = "X" * 3000
    chunks = chunk_text(text, source="my_doc.pdf", chunk_size=1000, overlap=0)
    assert all(c.source == "my_doc.pdf" for c in chunks)


# ===========================================================================
# eval/baseline.py — extract_text_pypdf
# ===========================================================================


def test_extract_text_pypdf_missing_dep():
    """extract_text_pypdf raises ImportError when pypdf is not importable."""
    original = sys.modules.get("pypdf")
    sys.modules["pypdf"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pypdf not installed"):
            extract_text_pypdf(Path("dummy.pdf"))
    finally:
        if original is None:
            sys.modules.pop("pypdf", None)
        else:
            sys.modules["pypdf"] = original


def test_extract_text_pypdf_file_not_found(tmp_path, monkeypatch):
    """extract_text_pypdf raises FileNotFoundError when the PDF path does not exist.

    pypdf may or may not be installed in CI, so we stub the import to ensure the
    function reaches the file-existence check regardless of the environment.
    """
    # Provide a fake pypdf module so the import guard passes
    fake_pypdf = MagicMock()
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    missing = tmp_path / "no_such_file.pdf"
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_text_pypdf(missing)


# ===========================================================================
# eval/baseline.py — index_baseline
# ===========================================================================


def test_index_baseline_skips_if_already_indexed(monkeypatch, tmp_path):
    """index_baseline is a no-op when the source already has chunks in the collection."""
    # Prepare collection mock that reports existing ids for this source
    col = _mock_collection(count=5)
    col.get.return_value = {"ids": ["doc.pdf:0", "doc.pdf:1"], "documents": [], "metadatas": []}

    monkeypatch.setattr(baseline_module, "_get_baseline_collection", lambda: col)

    # Create a dummy PDF path (file existence not checked because early-exit happens first)
    dummy_pdf = tmp_path / "doc.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 fake")

    index_baseline(dummy_pdf)

    # upsert must never be called when skipping
    col.upsert.assert_not_called()


# ===========================================================================
# eval/baseline.py — query_baseline
# ===========================================================================


def test_query_baseline_empty_collection(monkeypatch):
    """query_baseline returns a fallback QAResult when the collection is empty."""
    col = _mock_collection(count=0)
    monkeypatch.setattr(baseline_module, "_get_baseline_collection", lambda: col)
    _patch_log(monkeypatch)

    result = query_baseline("What is gradient descent?")

    assert isinstance(result, QAResult)
    assert result.source_blocks == []
    assert "index_baseline" in result.answer or len(result.answer) > 0
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_query_baseline_success(monkeypatch):
    """query_baseline returns a populated QAResult with answer and source_blocks on success."""
    query_result = {
        "ids": [["doc.pdf:0"]],
        "documents": [["Gradient descent minimizes a loss function."]],
        "metadatas": [[{"source": "doc.pdf", "chunk_index": 0}]],
        "distances": [[0.12]],
    }
    col = _mock_collection(count=3, query_result=query_result)
    monkeypatch.setattr(baseline_module, "_get_baseline_collection", lambda: col)
    monkeypatch.setattr(baseline_module, "_get_openai_embedding", lambda text: (FAKE_VECTOR, 20))
    monkeypatch.setattr(baseline_module, "_call_openai", lambda model, system, user: ("Baseline answer.", 80, 30))
    _patch_log(monkeypatch)

    result = query_baseline("What is gradient descent?")

    assert isinstance(result, QAResult)
    assert result.answer == "Baseline answer."
    assert len(result.source_blocks) == 1
    assert result.source_blocks[0].source == "doc.pdf"
    assert result.input_tokens == 80
    assert result.output_tokens == 30


# ===========================================================================
# eval/before_after.py — _keyword_score
# ===========================================================================


def test_keyword_score_high():
    """_keyword_score returns >= 0.7 when the actual answer contains most expected keywords."""
    expected = "gradient descent minimizes loss function"
    actual = "gradient descent is used to minimize the loss function over many iterations"
    score = _keyword_score(expected, actual)
    assert score >= 0.7


def test_keyword_score_low():
    """_keyword_score returns 0.0 for answers with completely different vocabulary."""
    expected = "photosynthesis converts sunlight into glucose"
    actual = "neural networks optimize parameters"
    score = _keyword_score(expected, actual)
    assert score == 0.0


def test_keyword_score_empty_expected():
    """_keyword_score returns 0.0 when the expected string has no qualifying keywords."""
    # All words shorter than _MIN_KEYWORD_LENGTH (4)
    score = _keyword_score("a b cc", "a b cc ddd eee")
    assert score == 0.0


# ===========================================================================
# eval/before_after.py — _score_answer
# ===========================================================================


def test_score_answer_high_keyword_skips_llm(monkeypatch):
    """_score_answer returns 1.0 without calling the LLM when keyword score >= 0.7."""
    llm_called = []

    monkeypatch.setattr(ba_module, "_llm_judge_score", lambda *a, **kw: llm_called.append(True) or 0.5)

    expected = "gradient descent minimizes loss function iteratively updating parameters"
    actual = "gradient descent minimizes the loss function by iteratively updating parameters"

    result = _score_answer("Q", expected, actual)

    assert result == 1.0
    assert llm_called == [], "LLM judge must not be called for clearly high keyword scores"


def test_score_answer_low_keyword_skips_llm(monkeypatch):
    """_score_answer returns 0.0 without calling the LLM when keyword score <= 0.2."""
    llm_called = []

    monkeypatch.setattr(ba_module, "_llm_judge_score", lambda *a, **kw: llm_called.append(True) or 0.5)

    expected = "photosynthesis converts sunlight into glucose within chloroplasts"
    actual = "neural networks learn features from data"

    result = _score_answer("Q", expected, actual)

    assert result == 0.0
    assert llm_called == [], "LLM judge must not be called for clearly low keyword scores"


# ===========================================================================
# eval/before_after.py — run_comparison
# ===========================================================================


def test_run_comparison_file_not_found(tmp_path):
    """run_comparison raises FileNotFoundError when golden_set_path does not exist."""
    missing = tmp_path / "no_golden.json"
    with pytest.raises(FileNotFoundError):
        run_comparison(golden_set_path=missing)


def test_run_comparison_empty_items(tmp_path):
    """run_comparison raises ValueError when the golden set contains no items."""
    empty_gs = tmp_path / "empty.json"
    empty_gs.write_text(json.dumps({"items": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="No items found"):
        run_comparison(golden_set_path=empty_gs)


def test_run_comparison_success(monkeypatch, tmp_path):
    """run_comparison returns a valid ComparisonReport when both pipelines and scoring are mocked."""
    # Minimal two-item golden set covering two different tiers
    golden_data = {
        "items": [
            {
                "id": "gs_t1",
                "tier": 1,
                "question": "What is gradient descent?",
                "expected_answer": "Gradient descent minimizes loss.",
                "expected_sources": ["doc.pdf"],
                "document_format": "pdf",
            },
            {
                "id": "gs_t2",
                "tier": 2,
                "question": "How many parameters does BERT have?",
                "expected_answer": "BERT has 110 million parameters.",
                "expected_sources": ["bert.pdf"],
                "document_format": "pdf",
            },
        ]
    }
    gs_path = tmp_path / "golden_set.json"
    gs_path.write_text(json.dumps(golden_data), encoding="utf-8")

    # Mock both pipeline query functions
    monkeypatch.setattr(ba_module, "query_baseline", lambda q, top_k, model: _fake_qa_result("baseline answer text"))
    monkeypatch.setattr(ba_module, "query_catchup", lambda q, top_k, model: _fake_qa_result("catchup answer text"))
    # Mock scoring to avoid LLM calls
    monkeypatch.setattr(ba_module, "_score_answer", lambda q, exp, act, judge_model="gpt-4o": 0.5)

    report = run_comparison(golden_set_path=gs_path, model="gpt-4o-mini")

    assert isinstance(report, ComparisonReport)
    assert report.total == 2
    assert report.model == "gpt-4o-mini"
    assert len(report.results) == 2
    assert 1 in report.by_tier
    assert 2 in report.by_tier
    assert report.overall_before == pytest.approx(0.5)
    assert report.overall_after == pytest.approx(0.5)


# ===========================================================================
# eval/before_after.py — save_report
# ===========================================================================


def test_save_report_creates_file(tmp_path):
    """save_report writes a timestamped JSON file to the specified output directory."""
    report = ComparisonReport(
        total=1,
        by_tier={1: TierStats(total=1, before_score=0.5, after_score=0.8, improvement_pct=60.0)},
        overall_before=0.5,
        overall_after=0.8,
        overall_improvement=60.0,
        results=[
            ComparisonResult(
                question="Q?",
                tier=1,
                case_id="gs_001",
                before_answer="before",
                after_answer="after",
                before_sources=["a.pdf"],
                after_sources=["a.pdf"],
                before_score=0.5,
                after_score=0.8,
                expected_answer="expected",
            )
        ],
        model="gpt-4o-mini",
    )

    out_path = save_report(report, output_dir=tmp_path)

    assert out_path.exists()
    assert out_path.suffix == ".json"
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["total"] == 1
    assert saved["model"] == "gpt-4o-mini"
    assert saved["overall_before"] == pytest.approx(0.5)


def test_comparison_report_tier_aggregation(monkeypatch, tmp_path):
    """by_tier statistics aggregate correctly across multiple questions in the same tier."""
    # Three tier-1 items: scores 1.0, 0.0, 1.0 -> avg = 0.6667
    golden_data = {
        "items": [
            {
                "id": f"gs_{i:03d}",
                "tier": 1,
                "question": f"Question {i}",
                "expected_answer": "some answer",
                "expected_sources": ["doc.pdf"],
                "document_format": "pdf",
            }
            for i in range(3)
        ]
    }
    gs_path = tmp_path / "golden_set.json"
    gs_path.write_text(json.dumps(golden_data), encoding="utf-8")

    scores = iter([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])  # before/after alternating for each item
    monkeypatch.setattr(ba_module, "query_baseline", lambda q, top_k, model: _fake_qa_result("b answer"))
    monkeypatch.setattr(ba_module, "query_catchup", lambda q, top_k, model: _fake_qa_result("c answer"))
    monkeypatch.setattr(ba_module, "_score_answer", lambda q, exp, act, judge_model="gpt-4o": next(scores))

    report = run_comparison(golden_set_path=gs_path)

    tier1 = report.by_tier[1]
    assert tier1.total == 3
    assert tier1.before_score == pytest.approx(round((1.0 + 0.0 + 1.0) / 3, 4))
    assert tier1.after_score == pytest.approx(round((1.0 + 0.0 + 1.0) / 3, 4))


# ===========================================================================
# eval/golden_set.json — structural validation
# ===========================================================================


def test_golden_set_loads():
    """golden_set.json is valid JSON and loads without error."""
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_golden_set_has_15_items():
    """golden_set.json declares total == 15 and contains exactly 15 items."""
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    assert data["total"] == 15
    assert len(data["items"]) == 15


def test_golden_set_all_required_fields():
    """Every item in golden_set.json has the required fields."""
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    required = {"id", "tier", "question", "expected_answer", "expected_sources", "document_format"}
    for item in data["items"]:
        missing = required - item.keys()
        assert missing == set(), f"Item {item.get('id', '?')} is missing fields: {missing}"


def test_golden_set_tier_distribution():
    """golden_set.json has exactly 4 items in tier 1, 4 in tier 2, 4 in tier 3, and 3 in tier 4."""
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    from collections import Counter

    counts = Counter(item["tier"] for item in data["items"])
    assert counts[1] == 4
    assert counts[2] == 4
    assert counts[3] == 4
    assert counts[4] == 3


def test_golden_set_valid_tiers():
    """All tier values in golden_set.json are within the valid set {1, 2, 3, 4}."""
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    valid = {1, 2, 3, 4}
    for item in data["items"]:
        assert item["tier"] in valid, f"Item {item['id']} has invalid tier: {item['tier']}"
