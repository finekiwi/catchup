"""ipynb golden-set evaluation — Round 1.

Mirrors the CU-09 PDF evaluation framework (run_round.py) but targets the ipynb
golden set and the ipynb-aware indexing pipeline.

Pre-requisite:
    python -m eval.setup_golden_index   # indexes all golden/*.ipynb into ChromaDB

Usage:
    python -m eval.run_ipynb_round1
    python -m eval.run_ipynb_round1 --model gpt-4o-mini --skip-deepeval

Output:
    eval/results/ipynb_round1.json  — combined DeepEval report (3 pipelines)
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

_GOLDEN_PATH = Path("eval/golden_set_ipynb.json")
_OUTPUT_PATH = Path("eval/results/ipynb_round1.json")
_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Index step
# ---------------------------------------------------------------------------


def run_index() -> None:
    """Index all golden/*.ipynb into baseline + CatchUp collections."""
    from eval.setup_golden_index import (
        index_all_baseline_ipynb,
        index_all_catchup_ipynb,
        index_all_catchup_chunked_ipynb,
    )

    LOGGER.info("=== Indexing golden notebooks ===")
    index_all_baseline_ipynb()
    index_all_catchup_ipynb()
    index_all_catchup_chunked_ipynb()


# ---------------------------------------------------------------------------
# DeepEval runner
# ---------------------------------------------------------------------------


def _build_eval_cases(query_fn, label: str, model: str, items: list) -> list:
    """Build EvalCase list by querying a pipeline for each golden-set item.

    Args:
        query_fn: Callable(question, top_k, model) -> QAResult.
        label: Human-readable pipeline label for logging.
        model: LLM model identifier.
        items: Golden set items list.

    Returns:
        List of EvalCase instances.
    """
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
                f"[{sb.source}] cell {sb.cell_index if sb.cell_index is not None else sb.block_order}\n{sb.content_preview}"
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
    """Run DeepEval on all three pipelines and return aggregated results.

    Pipelines:
        - Baseline:         nbformat flat text + fixed chunking
        - CatchUp-Raw:      ipynb_parser block-level (variable size)
        - CatchUp-Chunked:  ipynb_parser + HybridChunker max_tokens=500

    Args:
        model: LLM model for answer generation.

    Returns:
        Dict keyed by pipeline slug → serialised EvalReport dict.
    """
    from eval.evaluator import run_evaluation
    from rag.qa_chain import (
        query as query_catchup,
        query_chunked as query_catchup_chunked,
    )
    from eval.baseline import query_baseline

    golden_data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden_data.get("items", [])
    LOGGER.info("Loaded %d golden items from %s", len(items), _GOLDEN_PATH)

    pipelines = [
        ("Baseline", query_baseline),
        ("CatchUp-Raw", query_catchup),
        ("CatchUp-Chunked", query_catchup_chunked),
    ]

    reports: dict[str, dict] = {}

    for label, query_fn in pipelines:
        LOGGER.info("=== DeepEval: %s ===", label)
        cases = _build_eval_cases(query_fn, label, model, items)
        report = run_evaluation(cases, model=model)

        slug = label.lower().replace("-", "_")
        report_dict = asdict(report)
        reports[slug] = report_dict

        print(f"\n=== {label} ===")
        print(f"  Faithfulness:       {report.faithfulness_score:.4f}")
        print(f"  Context Precision:  {report.context_precision_score:.4f}")
        print(f"  Context Recall:     {report.context_recall_score:.4f}")
        print(f"  Citation Accuracy:  {report.citation_score:.4f}")
        print(f"  Overall:            {report.overall_score:.4f}")
        print(f"  Passed:             {report.passed_cases}/{report.total_cases}")

    return reports


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_results(reports: dict[str, dict], output_path: Path) -> None:
    """Save all pipeline reports to a single JSON file.

    Args:
        reports: Dict keyed by pipeline slug → EvalReport dict.
        output_path: Destination path for the JSON output.
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_ipynb_round1.

    Args:
        argv: Optional argument list for testing; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Run ipynb golden-set evaluation (Round 1) — DeepEval 4 metrics."
    )
    parser.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL, help="Answer LLM model"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing step (use if golden notebooks are already indexed)",
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

    if not args.skip_index:
        run_index()

    reports = run_deepeval(args.model)
    save_results(reports, args.output)

    print(f"\nipynb Round 1 complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
