"""Unified Round evaluation runner: before_after + DeepEval in one command.

Usage:
    python -m eval.run_round --round 2
    python -m eval.run_round --round 1 --model gpt-4o-mini --skip-deepeval

Output files (in data/eval_results/):
    before_after_round{N}_{ts}.json   — binary correctness comparison (PyPDF vs CatchUp)
    deepeval_round{N}_{ts}.json       — DeepEval metrics (Faithfulness, ContextPrecision, Citation)

Round 1 reference: data/eval_results/before_after_20260312T062837.json (binary only)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

_GOLDEN_PATH = Path("eval/golden_set.json")
_OUTPUT_DIR = Path("data/eval_results")
_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Before/After runner
# ---------------------------------------------------------------------------


def run_before_after(round_num: int, model: str) -> Path:
    """Run before_after comparison and save to round-tagged file."""
    from eval.before_after import run_comparison, save_report, ComparisonReport

    LOGGER.info("=== Before/After (Round %d) ===", round_num)
    report: ComparisonReport = run_comparison(golden_set_path=_GOLDEN_PATH, model=model)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = _OUTPUT_DIR / f"before_after_round{round_num}_{ts}.json"

    report_dict = asdict(report)
    out_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Before/After report saved to %s", out_path)

    print(f"\n=== Before/After Round {round_num} ===")
    print(f"Overall Before (PyPDF): {report.overall_before:.4f}")
    print(f"Overall After  (CatchUp): {report.overall_after:.4f}")
    print(f"Improvement: {report.overall_improvement:+.1f}%")
    print("Per-tier:")
    for tier, stats in sorted(report.by_tier.items()):
        print(
            f"  Tier {tier}: before={stats.before_score:.4f}  after={stats.after_score:.4f}"
            f"  {stats.improvement_pct:+.1f}%  (n={stats.total})"
        )
    return out_path


# ---------------------------------------------------------------------------
# DeepEval runner
# ---------------------------------------------------------------------------


def _build_eval_cases(model: str) -> list:
    """Build EvalCase list by running CatchUp pipeline on golden set questions."""
    from eval.evaluator import EvalCase
    from rag.qa_chain import query as query_catchup

    golden_data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden_data.get("items", [])

    cases = []
    for item in items:
        question = item["question"]
        expected = item["expected_answer"]
        case_id = item.get("id", "unknown")
        tier = item.get("tier", 0)

        LOGGER.info("  DeepEval: querying %s", case_id)
        try:
            result = query_catchup(question, top_k=5, model=model)
            actual_answer = result.answer
            retrieved_contexts = [
                f"[{sb.source}]\n{sb.content_preview}"
                for sb in result.source_blocks
            ]
            sources = [sb.source for sb in result.source_blocks]
        except Exception as exc:
            LOGGER.warning("CatchUp query failed for %s: %s", case_id, exc)
            actual_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            sources = []

        cases.append(EvalCase(
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            retrieved_contexts=retrieved_contexts,
            sources=sources,
            case_id=case_id,
            tier=tier,
        ))

    return cases


def run_deepeval(round_num: int, model: str) -> Path:
    """Run DeepEval metrics on CatchUp pipeline outputs and save to round-tagged file."""
    from eval.evaluator import run_evaluation, EvalReport
    from dataclasses import asdict as dc_asdict

    LOGGER.info("=== DeepEval (Round %d) ===", round_num)

    cases = _build_eval_cases(model)
    report: EvalReport = run_evaluation(cases, model=model)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = _OUTPUT_DIR / f"deepeval_round{round_num}_{ts}.json"

    out_path.write_text(json.dumps(dc_asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("DeepEval report saved to %s", out_path)

    print(f"\n=== DeepEval Round {round_num} ===")
    print(f"Faithfulness:       {report.faithfulness_score:.4f}")
    print(f"Context Precision:  {report.context_precision_score:.4f}")
    print(f"Citation Accuracy:  {report.citation_score:.4f}")
    print(f"Overall:            {report.overall_score:.4f}")
    print(f"Passed:             {report.passed_cases}/{report.total_cases}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full evaluation round: before_after + DeepEval"
    )
    parser.add_argument("--round", type=int, default=2, help="Round number (default: 2)")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="Answer model")
    parser.add_argument(
        "--skip-deepeval", action="store_true",
        help="Run before_after only (skip DeepEval — saves cost)",
    )
    parser.add_argument(
        "--skip-before-after", action="store_true",
        help="Run DeepEval only (skip before_after comparison)",
    )
    args = parser.parse_args(argv)

    LOGGER.info("Starting Round %d evaluation (model=%s)", args.round, args.model)

    results: dict[str, str] = {}

    if not args.skip_before_after:
        ba_path = run_before_after(args.round, args.model)
        results["before_after"] = str(ba_path)

    if not args.skip_deepeval:
        de_path = run_deepeval(args.round, args.model)
        results["deepeval"] = str(de_path)

    print(f"\nRound {args.round} complete. Output files:")
    for key, path in results.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
