"""Before/After RAG evaluation: vanilla query vs query rewriting (CU-13).

Runs both pipelines on the combined 30-question golden set (PDF 15 + ipynb 15),
then computes subgroup analysis for rewrite-needed cases and Wilcoxon Signed-Rank Test.

Usage:
    python -m eval.run_rewrite_eval
    python -m eval.run_rewrite_eval --model gpt-4o-mini --output eval/results/rewrite_eval.json
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

_GOLDEN_PDF_PATH = Path("eval/golden_set.json")
_GOLDEN_IPYNB_PATH = Path("eval/golden_set_ipynb.json")
_OUTPUT_PATH = Path("eval/results/rewrite_eval.json")
_DEFAULT_MODEL = "gpt-4o-mini"
_REWRITE_MODEL = "gpt-4.1-nano"

# ---------------------------------------------------------------------------
# Manual tagging: cases where query rewriting is expected to improve retrieval.
# Criteria:
#   - Korean technical terms that need English expansion for better embedding match
#   - Abbreviations whose full names appear in the document text
#   - Mixed KO/EN queries against Korean-language documents
# ---------------------------------------------------------------------------
REWRITE_NEEDED_CASES: set[str] = {
    "gs_005",        # "MLP" abbreviation → "MLP (Multilayer Perceptron)" helps match EN+KO docs
    "gs_008",        # EN query against KO git book; "inline commit" → "인라인 커밋 (inline commit, git commit -m)"
    "gs_ipynb_003",  # "RAG" acronym → "RAG (Retrieval-Augmented Generation, 검색 증강 생성)" in KO notebook
    "gs_ipynb_008",  # mixed KO query containing "노드 조회" — KO/EN expansion helpful
    "gs_ipynb_009",  # EN query about KO RAG pipeline; "search_db tool" + "RAG pipeline" expansion helpful
    "gs_ipynb_010",  # pure Korean query "삼성전자는 어떤 기술을 개발하고 있나요?" against KO notebook
}


def _load_golden_items() -> list[dict]:
    """Load and merge PDF + ipynb golden set items."""
    items: list[dict] = []
    for path in (_GOLDEN_PDF_PATH, _GOLDEN_IPYNB_PATH):
        data = json.loads(path.read_text(encoding="utf-8"))
        items.extend(data.get("items", []))
    LOGGER.info("Loaded %d golden items total", len(items))
    return items


def _build_eval_cases(query_fn, label: str, model: str, items: list[dict]) -> tuple[list, list[str | None]]:
    """Build EvalCase list, also capturing rewritten queries for comparison output.

    Returns:
        (cases, rewritten_queries) — rewritten_queries[i] is the rewritten string or None.
    """
    from eval.evaluator import EvalCase

    cases = []
    rewritten_queries: list[str | None] = []

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
            rewritten_queries.append(getattr(result, "rewritten_query", None))
        except Exception as exc:
            LOGGER.warning("%s query failed for %s: %s", label, case_id, exc)
            actual_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            sources = []
            rewritten_queries.append(None)

        cases.append(EvalCase(
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            retrieved_contexts=retrieved_contexts,
            sources=sources,
            case_id=case_id,
            tier=tier,
        ))

    return cases, rewritten_queries


def _run_pipeline(query_fn, label: str, model: str, items: list[dict]) -> tuple[dict, list[float], list[str | None]]:
    """Run evaluation pipeline and return (report_dict, per_case_overall_scores, rewritten_queries)."""
    from eval.evaluator import run_evaluation

    LOGGER.info("=== Pipeline: %s ===", label)
    cases, rewritten_queries = _build_eval_cases(query_fn, label, model, items)
    report = run_evaluation(cases, model=model)

    per_case_scores = [cr.overall_score for cr in report.per_case_results]

    print(f"\n=== {label} ===")
    print(f"  Faithfulness:       {report.faithfulness_score:.4f}")
    print(f"  Context Precision:  {report.context_precision_score:.4f}")
    print(f"  Context Recall:     {report.context_recall_score:.4f}")
    print(f"  Citation Accuracy:  {report.citation_score:.4f}")
    print(f"  Overall:            {report.overall_score:.4f}")
    print(f"  Passed:             {report.passed_cases}/{report.total_cases}")

    return asdict(report), per_case_scores, rewritten_queries


def _binary_comparison(
    items: list[dict],
    vanilla_scores: list[float],
    rewrite_scores: list[float],
    rewrite_queries: list[str | None],
) -> list[dict]:
    """Build per-case binary comparison list."""
    _PASS_THRESHOLD = 0.5
    result = []
    for i, item in enumerate(items):
        case_id = item.get("id", f"case_{i}")
        result.append({
            "case_id": case_id,
            "question": item["question"],
            "rewritten_query": rewrite_queries[i],
            "rewrite_needed": case_id in REWRITE_NEEDED_CASES,
            "vanilla_overall": round(vanilla_scores[i], 4),
            "rewrite_overall": round(rewrite_scores[i], 4),
            "vanilla_correct": vanilla_scores[i] >= _PASS_THRESHOLD,
            "rewrite_correct": rewrite_scores[i] >= _PASS_THRESHOLD,
        })
    return result


def _subgroup_analysis(per_case: list[dict]) -> dict:
    """Compute subgroup metrics for rewrite_needed vs not_needed cases."""
    needed = [c for c in per_case if c["rewrite_needed"]]
    not_needed = [c for c in per_case if not c["rewrite_needed"]]

    def _stats(subset: list[dict]) -> dict:
        if not subset:
            return {"count": 0, "vanilla_overall": 0.0, "rewrite_overall": 0.0, "delta": 0.0}
        v_avg = sum(c["vanilla_overall"] for c in subset) / len(subset)
        r_avg = sum(c["rewrite_overall"] for c in subset) / len(subset)
        return {
            "count": len(subset),
            "vanilla_overall": round(v_avg, 4),
            "rewrite_overall": round(r_avg, 4),
            "delta": round(r_avg - v_avg, 4),
        }

    return {
        "rewrite_needed": _stats(needed),
        "rewrite_not_needed": _stats(not_needed),
    }


def _wilcoxon_test(vanilla_scores: list[float], rewrite_scores: list[float]) -> dict:
    """Run Wilcoxon Signed-Rank Test on paired score lists."""
    try:
        from scipy.stats import wilcoxon

        n = len(vanilla_scores)
        differences = [r - v for v, r in zip(vanilla_scores, rewrite_scores)]
        if all(d == 0 for d in differences):
            return {"test": "wilcoxon", "n": n, "statistic": None, "p_value": None,
                    "significant": False, "note": "all differences are zero"}
        stat, p_value = wilcoxon(rewrite_scores, vanilla_scores)
        return {
            "test": "wilcoxon",
            "n": n,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
        }
    except Exception as exc:
        LOGGER.warning("Wilcoxon test failed: %s", exc)
        return {"test": "wilcoxon", "error": str(exc)}


def run_rewrite_eval(model: str) -> dict:
    """Run vanilla vs rewrite evaluation on the combined 30Q golden set.

    Args:
        model: LLM model for answer generation (same for both pipelines).

    Returns:
        Full result dict with pipelines, binary_comparison, subgroup_analysis,
        and statistical_test sections.
    """
    from functools import partial
    from rag.qa_chain import query_chunked

    items = _load_golden_items()

    # Vanilla pipeline: query_chunked with rewrite=False (default)
    vanilla_report, vanilla_scores, _ = _run_pipeline(
        partial(query_chunked, rewrite=False),
        label="Vanilla",
        model=model,
        items=items,
    )

    # Rewrite pipeline: query_chunked with rewrite=True
    rewrite_report, rewrite_scores, rewrite_queries = _run_pipeline(
        partial(query_chunked, rewrite=True),
        label="Rewrite",
        model=model,
        items=items,
    )

    per_case = _binary_comparison(items, vanilla_scores, rewrite_scores, rewrite_queries)
    subgroup = _subgroup_analysis(per_case)

    # Wilcoxon tests: all 30 + rewrite_needed subset only
    needed_indices = [i for i, c in enumerate(per_case) if c["rewrite_needed"]]
    vanilla_all, rewrite_all = vanilla_scores, rewrite_scores
    vanilla_needed = [vanilla_scores[i] for i in needed_indices]
    rewrite_needed = [rewrite_scores[i] for i in needed_indices]

    stat_all = _wilcoxon_test(vanilla_all, rewrite_all)
    stat_needed = _wilcoxon_test(vanilla_needed, rewrite_needed) if needed_indices else {"note": "no rewrite_needed cases"}

    vanilla_correct = sum(1 for c in per_case if c["vanilla_correct"])
    rewrite_correct = sum(1 for c in per_case if c["rewrite_correct"])

    print(f"\n=== Binary Comparison ===")
    print(f"  Total: {len(per_case)}")
    print(f"  Vanilla correct:  {vanilla_correct}/{len(per_case)}")
    print(f"  Rewrite correct:  {rewrite_correct}/{len(per_case)}")
    print(f"\n=== Subgroup Analysis ===")
    for grp, stats in subgroup.items():
        print(f"  {grp}: n={stats['count']}, vanilla={stats['vanilla_overall']:.4f}, "
              f"rewrite={stats['rewrite_overall']:.4f}, delta={stats['delta']:+.4f}")
    print(f"\n=== Statistical Test (all) ===")
    print(f"  p={stat_all.get('p_value')}, significant={stat_all.get('significant')}")
    print(f"=== Statistical Test (rewrite_needed) ===")
    print(f"  p={stat_needed.get('p_value')}, significant={stat_needed.get('significant')}")

    return {
        "golden_set": f"{_GOLDEN_PDF_PATH} + {_GOLDEN_IPYNB_PATH}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer_model": model,
        "rewrite_model": _REWRITE_MODEL,
        "pipelines": {
            "vanilla": vanilla_report,
            "rewrite": rewrite_report,
        },
        "binary_comparison": {
            "total": len(per_case),
            "vanilla_correct": vanilla_correct,
            "rewrite_correct": rewrite_correct,
            "per_case": per_case,
        },
        "subgroup_analysis": subgroup,
        "statistical_test": {
            "all_cases": stat_all,
            "rewrite_needed_only": stat_needed,
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_rewrite_eval."""
    parser = argparse.ArgumentParser(
        description="Run Before/After RAG evaluation: vanilla vs query rewriting (CU-13)."
    )
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="Answer LLM model")
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_PATH,
        help=f"Output JSON path (default: {_OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)

    for path in (_GOLDEN_PDF_PATH, _GOLDEN_IPYNB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Golden set not found: {path}")

    results = run_rewrite_eval(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Results saved to %s", args.output)
    print(f"\nRewrite eval complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
