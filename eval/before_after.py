"""Before/After comparison: PyPDF baseline vs CatchUp structured pipeline.

Experimental controls (must be identical):
  - embedding: text-embedding-3-small
  - top_k: 5
  - LLM: gpt-4o-mini
  - temperature: 0.0 (fixed in LLM calls)
  - Only variable: parsing + chunking quality

Reference: OHRBench (2024) — top OCR vs perfect parsing -> 14%+ RAG accuracy gap
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from eval.baseline import query_baseline
from prompts.eval_judge import JUDGE_SYSTEM_PROMPT, judge_user_prompt
from rag.qa_chain import query as query_catchup
from utils.logging import log_api_call

LOGGER = logging.getLogger(__name__)

_DEFAULT_GOLDEN_PATH = Path("eval/golden_set.json")
_DEFAULT_OUTPUT_DIR = Path("data/eval_results")
_DEFAULT_MODEL = "gpt-4o-mini"

# Simple keyword scoring: fraction of expected answer keywords found in actual answer
_MIN_KEYWORD_LENGTH = 4  # ignore short stop-like words


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Per-question comparison between baseline and CatchUp answers.

    Attributes:
        question: The input question.
        tier: Golden-set tier (1-4).
        case_id: Identifier from the golden set (e.g., gs_001).
        before_answer: Answer produced by the PyPDF baseline pipeline.
        after_answer: Answer produced by the CatchUp structured pipeline.
        before_sources: Source filenames cited by the baseline answer.
        after_sources: Source filenames cited by the CatchUp answer.
        before_score: Binary score (0 or 1) for baseline answer correctness.
        after_score: Binary score (0 or 1) for CatchUp answer correctness.
        expected_answer: Ground-truth reference from the golden set.
    """

    question: str
    tier: int
    case_id: str
    before_answer: str
    after_answer: str
    before_sources: list[str]
    after_sources: list[str]
    before_score: float
    after_score: float
    expected_answer: str


@dataclass
class TierStats:
    """Aggregated stats for a single tier.

    Attributes:
        total: Number of questions in this tier.
        before_score: Average score for the baseline pipeline in this tier.
        after_score: Average score for the CatchUp pipeline in this tier.
        improvement_pct: Percentage improvement of after over before.
    """

    total: int
    before_score: float
    after_score: float
    improvement_pct: float


@dataclass
class ComparisonReport:
    """Full Before/After comparison report.

    Attributes:
        total: Total number of questions compared.
        by_tier: Per-tier statistics keyed by tier number.
        overall_before: Overall average score for the baseline.
        overall_after: Overall average score for CatchUp.
        overall_improvement: Percentage improvement of CatchUp over baseline.
        results: Detailed per-question ComparisonResult list.
        model: LLM model used for both pipelines.
        timestamp: ISO-8601 UTC timestamp of report generation.
    """

    total: int
    by_tier: dict[int, TierStats]
    overall_before: float
    overall_after: float
    overall_improvement: float
    results: list[ComparisonResult] = field(default_factory=list)
    model: str = _DEFAULT_MODEL
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _keyword_score(expected: str, actual: str) -> float:
    """Compute a simple keyword overlap score in [0, 1].

    Extracts meaningful words (length >= _MIN_KEYWORD_LENGTH) from the expected
    answer and checks what fraction appear in the actual answer (case-insensitive).

    Args:
        expected: Ground-truth reference answer.
        actual: Generated answer to score.

    Returns:
        Float in [0, 1] representing keyword recall.
    """
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    keywords = [
        w for w in expected_lower.split()
        if len(w) >= _MIN_KEYWORD_LENGTH and w.isalpha()
    ]
    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw in actual_lower)
    return hits / len(keywords)


def _llm_judge_score(
    question: str,
    expected: str,
    actual: str,
    model: str = "gpt-4o",
) -> float:
    """Use an LLM as a binary judge to score answer correctness.

    Returns 1.0 if the actual answer is judged correct/sufficient, 0.0 otherwise.
    Falls back to keyword score on API failure.

    Args:
        question: The original question.
        expected: Ground-truth reference answer.
        actual: Generated answer to evaluate.
        model: OpenAI model to use as the judge (should differ from answer model).

    Returns:
        1.0 (correct) or 0.0 (incorrect).
    """
    from rag.qa_chain import _call_openai  # lazy import to avoid circular issues

    try:
        t0 = time.perf_counter()
        raw, input_tokens, output_tokens = _call_openai(
            model, JUDGE_SYSTEM_PROMPT, judge_user_prompt(question, expected, actual)
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        log_api_call(
            model=model,
            stage="ba_judge",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=True,
        )
        # Use exact match to avoid "INCORRECT" being treated as containing "CORRECT"
        verdict = raw.strip().upper()
        return 1.0 if verdict == "CORRECT" else 0.0
    except Exception as exc:
        LOGGER.warning("LLM judge failed (%s), falling back to keyword score: %s", model, exc)
        log_api_call(
            model=model,
            stage="ba_judge",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        # Fallback: use keyword score thresholded at 0.5
        kw = _keyword_score(expected, actual)
        return 1.0 if kw >= 0.5 else 0.0


def _score_answer(
    question: str,
    expected: str,
    actual: str,
    judge_model: str = "gpt-4o",
) -> float:
    """Combined scoring: keyword check first, then LLM judge for borderline cases.

    If keyword score is clearly high (>= 0.7) or clearly low (<= 0.2), use it directly
    as a binary signal. Otherwise, delegate to the LLM judge.

    Args:
        question: The original question.
        expected: Reference answer.
        actual: Candidate answer to score.
        judge_model: Model to use for LLM judging.

    Returns:
        Binary score: 1.0 or 0.0.
    """
    kw = _keyword_score(expected, actual)
    if kw >= 0.7:
        return 1.0
    if kw <= 0.2:
        return 0.0
    # Borderline: delegate to LLM judge
    return _llm_judge_score(question, expected, actual, model=judge_model)


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_comparison(
    golden_set_path: Path = _DEFAULT_GOLDEN_PATH,
    model: str = _DEFAULT_MODEL,
) -> ComparisonReport:
    """Run Before/After comparison across all golden-set questions.

    Loads the golden set, queries both pipelines for each question, scores answers,
    and aggregates results by tier.

    Args:
        golden_set_path: Path to golden_set.json.
        model: LLM model for both pipelines. Judge uses gpt-4o regardless.

    Returns:
        ComparisonReport with per-question and per-tier statistics.

    Raises:
        FileNotFoundError: If golden_set_path does not exist.
        ValueError: If golden set JSON is malformed.
    """
    if not golden_set_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_set_path}")

    golden_data = json.loads(golden_set_path.read_text(encoding="utf-8"))
    items = golden_data.get("items", [])
    if not items:
        raise ValueError(f"No items found in golden set at {golden_set_path}")

    LOGGER.info(
        "Before/After comparison: %d questions, model=%s", len(items), model
    )

    results: list[ComparisonResult] = []

    for item in items:
        case_id: str = item.get("id", "unknown")
        question: str = item["question"]
        expected: str = item["expected_answer"]
        tier: int = item.get("tier", 0)

        LOGGER.debug("Comparing case %s (tier %d)", case_id, tier)

        # Query baseline pipeline
        try:
            before_result = query_baseline(question, top_k=5, model=model)
            before_answer = before_result.answer
            before_sources = list({sb.source for sb in before_result.source_blocks})
        except Exception as exc:
            LOGGER.error("Baseline query failed for case %s: %s", case_id, exc)
            before_answer = f"[ERROR] {exc}"
            before_sources = []

        # Query CatchUp pipeline
        try:
            after_result = query_catchup(question, top_k=5, model=model)
            after_answer = after_result.answer
            after_sources = list({sb.source for sb in after_result.source_blocks})
        except Exception as exc:
            LOGGER.error("CatchUp query failed for case %s: %s", case_id, exc)
            after_answer = f"[ERROR] {exc}"
            after_sources = []

        # Score both answers
        before_score = _score_answer(question, expected, before_answer)
        after_score = _score_answer(question, expected, after_answer)

        results.append(ComparisonResult(
            question=question,
            tier=tier,
            case_id=case_id,
            before_answer=before_answer,
            after_answer=after_answer,
            before_sources=before_sources,
            after_sources=after_sources,
            before_score=before_score,
            after_score=after_score,
            expected_answer=expected,
        ))

    # Aggregate by tier
    tier_groups: dict[int, list[ComparisonResult]] = {}
    for r in results:
        tier_groups.setdefault(r.tier, []).append(r)

    by_tier: dict[int, TierStats] = {}
    for t, group in sorted(tier_groups.items()):
        n = len(group)
        before_avg = sum(r.before_score for r in group) / n
        after_avg = sum(r.after_score for r in group) / n
        improvement = (
            ((after_avg - before_avg) / before_avg * 100.0)
            if before_avg > 0.0
            else float("inf") if after_avg > 0.0 else 0.0
        )
        by_tier[t] = TierStats(
            total=n,
            before_score=round(before_avg, 4),
            after_score=round(after_avg, 4),
            improvement_pct=round(improvement, 2),
        )

    n_total = len(results)
    overall_before = sum(r.before_score for r in results) / n_total if n_total else 0.0
    overall_after = sum(r.after_score for r in results) / n_total if n_total else 0.0
    overall_improvement = (
        ((overall_after - overall_before) / overall_before * 100.0)
        if overall_before > 0.0
        else float("inf") if overall_after > 0.0 else 0.0
    )

    report = ComparisonReport(
        total=n_total,
        by_tier=by_tier,
        overall_before=round(overall_before, 4),
        overall_after=round(overall_after, 4),
        overall_improvement=round(overall_improvement, 2),
        results=results,
        model=model,
    )

    LOGGER.info(
        "Comparison complete: before=%.4f after=%.4f improvement=%.1f%%",
        overall_before, overall_after, overall_improvement,
    )
    return report


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


def save_report(
    report: ComparisonReport,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist a ComparisonReport to a timestamped JSON file.

    Args:
        report: The ComparisonReport to save.
        output_dir: Directory in which to write the report.

    Returns:
        Path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"before_after_{ts}.json"

    # Convert dataclasses to dict; handle nested TierStats dict keyed by int
    report_dict = asdict(report)
    # by_tier keys become strings in JSON — that's acceptable
    out_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("ComparisonReport saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry: python -m eval.before_after --golden eval/golden_set.json --model gpt-4o-mini.

    Args:
        argv: Optional argument list for testing; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Run Before/After comparison: PyPDF baseline vs CatchUp RAG pipeline."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=_DEFAULT_GOLDEN_PATH,
        help="Path to golden_set.json (default: eval/golden_set.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="LLM model for both pipelines (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to save the comparison report (default: data/eval_results)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    report = run_comparison(golden_set_path=args.golden, model=args.model)
    out_path = save_report(report, output_dir=args.output_dir)

    print("\n=== Before/After Comparison Results ===")
    print(f"Total questions: {report.total}")
    print(f"Model: {report.model}")
    print(f"Overall Before: {report.overall_before:.4f}")
    print(f"Overall After:  {report.overall_after:.4f}")
    print(f"Improvement:    {report.overall_improvement:+.1f}%")
    print("\nPer-tier breakdown:")
    for tier, stats in sorted(report.by_tier.items()):
        print(
            f"  Tier {tier}: before={stats.before_score:.4f}  after={stats.after_score:.4f}"
            f"  improvement={stats.improvement_pct:+.1f}%  (n={stats.total})"
        )
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "ComparisonResult",
    "ComparisonReport",
    "TierStats",
    "run_comparison",
    "save_report",
    "main",
]
