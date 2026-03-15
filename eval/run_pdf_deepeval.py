"""PDF golden-set DeepEval evaluation — 4 metrics, Baseline vs CatchUp comparison.

Mirrors run_ipynb_round1.py but targets the PDF golden set.
Pre-requisite: python -m eval.setup_golden_index  (PDFs already indexed from CU-09)

Usage:
    python -m eval.run_pdf_deepeval
    python -m eval.run_pdf_deepeval --skip-index --output eval/results/pdf_deepeval_round1.json

Output:
    eval/results/pdf_deepeval_round1.json  — Baseline + CatchUp DeepEval report
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

_GOLDEN_PATH = Path("eval/golden_set.json")
_OUTPUT_PATH = Path("eval/results/pdf_deepeval_round1.json")
_DEFAULT_MODEL = "gpt-4o-mini"


def _build_eval_cases(query_fn, label: str, model: str, items: list) -> list:
    """Build EvalCase list by querying a pipeline for each golden-set item."""
    from eval.evaluator import EvalCase

    cases = []
    for item in items:
        question: str = item["question"]
        expected: str = item["expected_answer"]
        case_id: str = item.get("id", "unknown")
        tier: int = item.get("tier", 0)

        LOGGER.info("  [%s] querying %s", label, case_id)
        try:
            result = query_fn(question, top_k=5, model=model)
            actual_answer = result.answer
            retrieved_contexts = [
                f"[{sb.source}] page {sb.page if sb.page is not None else sb.block_order}\n{sb.content_preview}"
                for sb in result.source_blocks
            ]
            sources = [sb.source for sb in result.source_blocks]
        except Exception as exc:
            LOGGER.warning("%s query failed for %s: %s", label, case_id, exc)
            actual_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            sources = []

        cases.append(
            EvalCase(
                question=question,
                expected_answer=expected,
                actual_answer=actual_answer,
                retrieved_contexts=retrieved_contexts,
                sources=sources,
                case_id=case_id,
                tier=tier,
            )
        )

    return cases


def run_deepeval(model: str) -> dict[str, dict]:
    """Run DeepEval on Baseline and CatchUp-Chunked pipelines.

    Args:
        model: LLM model for answer generation.

    Returns:
        Dict keyed by pipeline slug → serialised EvalReport dict.
    """
    from eval.evaluator import run_evaluation
    from rag.qa_chain import query_chunked as query_catchup_chunked
    from eval.baseline import query_baseline

    golden_data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden_data.get("items", [])
    LOGGER.info("Loaded %d golden items from %s", len(items), _GOLDEN_PATH)

    pipelines = [
        ("Baseline", query_baseline),
        ("CatchUp", query_catchup_chunked),
    ]

    reports: dict[str, dict] = {}

    for label, query_fn in pipelines:
        LOGGER.info("=== DeepEval: %s ===", label)
        cases = _build_eval_cases(query_fn, label, model, items)
        report = run_evaluation(cases, model=model)

        slug = label.lower()
        reports[slug] = asdict(report)

        print(f"\n=== {label} ===")
        print(f"  Faithfulness:       {report.faithfulness_score:.4f}")
        print(f"  Context Precision:  {report.context_precision_score:.4f}")
        print(f"  Context Recall:     {report.context_recall_score:.4f}")
        print(f"  Citation Accuracy:  {report.citation_score:.4f}")
        print(f"  Overall:            {report.overall_score:.4f}")
        print(f"  Passed:             {report.passed_cases}/{report.total_cases}")

    return reports


def save_results(reports: dict[str, dict], output_path: Path) -> None:
    """Save all pipeline reports to a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "golden_set": str(_GOLDEN_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipelines": reports,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Results saved to %s", output_path)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_pdf_deepeval."""
    parser = argparse.ArgumentParser(
        description="Run PDF golden-set DeepEval evaluation — Baseline vs CatchUp."
    )
    parser.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL, help="Answer LLM model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output JSON path (default: {_OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)

    if not _GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden set not found: {_GOLDEN_PATH}")

    reports = run_deepeval(args.model)
    save_results(reports, args.output)

    print(f"\nPDF DeepEval complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
