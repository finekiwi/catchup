"""CU-09 evaluation package: RAG pipeline quality measurement."""

from __future__ import annotations

try:
    from eval.evaluator import run_evaluation, EvalCase, EvalReport  # noqa: F401

    __all_evaluator__ = ["run_evaluation", "EvalCase", "EvalReport"]
except ImportError:
    __all_evaluator__ = []

from eval.before_after import run_comparison, ComparisonReport  # noqa: F401

__all__ = [
    *__all_evaluator__,
    "run_comparison",
    "ComparisonReport",
]
