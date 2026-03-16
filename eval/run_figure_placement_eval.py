"""Figure placement quality evaluation.

Measures whether each enriched figure is placed in a semantically relevant
note section.  Primary scoring uses **embedding cosine similarity** between
the figure's VLM description and the surrounding section text (cheap,
deterministic, no LLM chat cost).  An optional ``--llm-judge`` flag enables
an LLM-as-judge cross-check.

Placement logic mirrors ui/demo.py `_render_note_with_figures`:
  - figures are mapped to sections by linear page interpolation
  - same-page figures are spread across consecutive sections

Usage:
    # Evaluate by original PDF path (embedding similarity only):
    python -m eval.run_figure_placement_eval --file path/to/doc.pdf

    # Evaluate by document id (uses SQLite + parse cache):
    python -m eval.run_figure_placement_eval --document-id <doc_id>

    # Dry-run: zero API calls, just print the figure-to-section mapping:
    python -m eval.run_figure_placement_eval --file path/to/doc.pdf --dry-run

    # Embedding + LLM cross-check:
    python -m eval.run_figure_placement_eval --file path/to/doc.pdf --llm-judge

    # Custom output path:
    python -m eval.run_figure_placement_eval --file path/to/doc.pdf \\
        --output eval/results/figure_placement.json

Output JSON schema:
    {
      "document_id": str,
      "source": str,
      "embedding_model": str,
      "judge_model": str,
      "timestamp": str,
      "summary": {
        "total_figures": int,
        "placed_figures": int,
        "mean_cosine_sim": float,
        "mean_placement_score": float,
        "pass_rate": float,        # fraction with placement_score >= 0.8
        "optimal_rate": float      # fraction where placed == best-match section
      },
      "per_figure": [
        {
          "figure_order": int,
          "image_path": str | null,
          "page": int | null,
          "section_idx": int,
          "section_heading": str,
          "figure_description": str,
          "cosine_sim": float,
          "best_sim": float,
          "placement_score": float,
          "is_optimal": bool,
          "optimality_gap": float,
          "best_match_section_idx": int,
          "best_match_heading": str,
          "passed": bool,
          "llm_score": float | null,
          "llm_reason": str
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

_DEFAULT_JUDGE_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"
_PASS_THRESHOLD = 0.8
_OUTPUT_PATH = Path("eval/results/figure_placement_eval.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FigurePlacementCase:
    """One figure-to-section pair to evaluate."""

    figure_order: int
    image_path: Optional[str]
    page: Optional[int]
    section_idx: int
    section_heading: str
    section_body: str
    figure_description: str


@dataclass
class FigurePlacementResult:
    """Embedding-based result for a single figure placement."""

    figure_order: int
    image_path: Optional[str]
    page: Optional[int]
    section_idx: int
    section_heading: str
    figure_description: str
    cosine_sim: float
    best_sim: float
    placement_score: float        # cosine_sim / best_sim; 1.0 = optimal
    is_optimal: bool              # placed section == best-match section
    optimality_gap: float         # best_sim - cosine_sim; 0.0 = optimal
    best_match_section_idx: int
    best_match_heading: str
    passed: bool                  # placement_score >= _PASS_THRESHOLD
    llm_score: Optional[float] = None
    llm_reason: str = ""


@dataclass
class FigurePlacementReport:
    """Aggregated placement quality report."""

    document_id: str
    source: str
    embedding_model: str
    judge_model: str
    timestamp: str
    total_figures: int
    placed_figures: int
    mean_cosine_sim: float
    mean_placement_score: float
    pass_rate: float
    optimal_rate: float
    per_figure: list[FigurePlacementResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Section splitting (mirrors llm/note_editor._split_sections)
# ---------------------------------------------------------------------------


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at ## boundaries."""
    lines = markdown.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
        if not in_code_fence and line.startswith("## ") and not line.startswith("### "):
            if current_heading or current_body_lines:
                sections.append((current_heading, "\n".join(current_body_lines).strip()))
            current_heading = line
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_heading or current_body_lines:
        sections.append((current_heading, "\n".join(current_body_lines).strip()))

    return sections


# ---------------------------------------------------------------------------
# Figure placement mapping (mirrors ui/demo.py _render_note_with_figures)
# ---------------------------------------------------------------------------


def _build_placement(
    sections: list[tuple[str, str]],
    fig_blocks: list,  # list[Block] with image_path set
) -> dict[int, list]:
    """Map each figure to a section index using page interpolation.

    Returns dict[section_idx → list[Block]].
    """
    n = len(sections)
    if n == 0:
        return {}

    pages = [b.metadata.page for b in fig_blocks]
    valid_pages = [p for p in pages if p is not None]
    page_min = min(valid_pages) if valid_pages else 1
    page_max = max(valid_pages) if valid_pages else 1
    page_range = max(page_max - page_min, 1)

    section_figs: dict[int, list] = defaultdict(list)
    page_counters: dict = defaultdict(int)

    for b in fig_blocks:
        p = b.metadata.page if b.metadata.page is not None else page_max
        ratio = (p - page_min) / page_range
        base_idx = min(int(ratio * n), n - 1)
        count = page_counters[b.metadata.page]
        page_counters[b.metadata.page] += 1
        idx = min(base_idx + count, n - 1)
        section_figs[idx].append(b)

    return dict(section_figs)


# ---------------------------------------------------------------------------
# Figure description extraction
# ---------------------------------------------------------------------------


def _get_figure_description(block) -> str:  # block: Block
    """Build a human-readable description of a figure block.

    Uses block.content (VLM analysis text) and block.metadata.caption.
    """
    parts = []
    if block.metadata.caption:
        parts.append(f"Caption: {block.metadata.caption}")
    if block.content and block.content != "[figure]":
        parts.append(block.content[:800])
    return "\n".join(parts).strip() or "[no description available]"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts via one OpenAI API call.

    Returns embeddings in the same order as the input list.
    """
    import openai

    client = openai.OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # resp.data items carry an .index field; sort to preserve input order
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# LLM judge (optional cross-check)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an educational content quality reviewer. "
    "Your task is to evaluate how well a figure placed in a study note matches "
    "the section where it appears. "
    "Reply ONLY with valid JSON: "
    '{"score": <float 0.0-1.0>, "reason": "<one sentence>"}'
)

_JUDGE_USER_TMPL = """\
SECTION HEADING: {heading}

SECTION TEXT (first 600 chars):
{body}

FIGURE DESCRIPTION:
{figure_desc}

Score 1.0 if the figure directly illustrates or supports the section content.
Score 0.5 if the figure is loosely related.
Score 0.0 if the figure is irrelevant or decorative noise.
"""


def _judge_one(case: FigurePlacementCase, judge_model: str) -> tuple[float, str]:
    """Call LLM judge to score one figure-section pair.

    Returns (score, reason).
    """
    import openai

    client = openai.OpenAI()
    body_excerpt = case.section_body[:600]
    user_msg = _JUDGE_USER_TMPL.format(
        heading=case.section_heading,
        body=body_excerpt,
        figure_desc=case.figure_description,
    )

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=128,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = str(payload.get("reason", ""))
    except Exception as exc:
        LOGGER.warning(
            "Judge call failed for figure order=%d: %s", case.figure_order, exc
        )
        score = 0.0
        reason = f"judge error: {exc}"

    latency_ms = (time.perf_counter() - t0) * 1000
    LOGGER.debug(
        "Judged figure order=%d section=%d score=%.2f latency=%.0fms",
        case.figure_order,
        case.section_idx,
        score,
        latency_ms,
    )
    return score, reason


# ---------------------------------------------------------------------------
# Embedding scoring
# ---------------------------------------------------------------------------


def _score_by_embedding(
    cases: list[FigurePlacementCase],
    sections: list[tuple[str, str]],
) -> list[FigurePlacementResult]:
    """Score all cases using embedding cosine similarity.

    Embeds all section texts and figure descriptions in a single batch call,
    then for each case computes placed_sim, best_sim, placement_score, is_optimal.
    """
    if not cases:
        return []

    # Build texts for batch embedding
    section_texts = [f"{heading}\n{body}" for heading, body in sections]
    figure_descs = [c.figure_description for c in cases]

    LOGGER.info(
        "Embedding %d sections + %d figure descriptions (1 API call)",
        len(section_texts),
        len(figure_descs),
    )
    all_texts = section_texts + figure_descs
    all_embeddings = _embed_texts_batch(all_texts)

    section_embeddings = all_embeddings[: len(section_texts)]
    figure_embeddings = all_embeddings[len(section_texts) :]

    results: list[FigurePlacementResult] = []
    for case, fig_vec in zip(cases, figure_embeddings):
        # Similarity with every section
        sims = [_cosine_sim(fig_vec, sec_vec) for sec_vec in section_embeddings]

        placed_sim = sims[case.section_idx]
        best_sim = max(sims)
        best_idx = int(sims.index(best_sim))
        placement_score = placed_sim / best_sim if best_sim > 0.0 else 0.0

        results.append(
            FigurePlacementResult(
                figure_order=case.figure_order,
                image_path=case.image_path,
                page=case.page,
                section_idx=case.section_idx,
                section_heading=case.section_heading,
                figure_description=case.figure_description,
                cosine_sim=round(placed_sim, 6),
                best_sim=round(best_sim, 6),
                placement_score=round(placement_score, 6),
                is_optimal=(case.section_idx == best_idx),
                optimality_gap=round(best_sim - placed_sim, 6),
                best_match_section_idx=best_idx,
                best_match_heading=sections[best_idx][0],
                passed=placement_score >= _PASS_THRESHOLD,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_document_by_id(document_id: str):
    """Load a Document from the parsed JSON cache by document id.

    Scans data/parsed/ for a cached document whose .id matches.
    """
    from models.document import Document

    cache_dir = Path("data/parsed")
    if not cache_dir.is_dir():
        return None

    for cache_file in cache_dir.glob("*.json"):
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            doc_raw = raw.get("document", {})
            if doc_raw.get("id") == document_id:
                return Document.model_validate(doc_raw)
        except Exception:
            continue
    return None


def _load_note_markdown(document_id: str) -> Optional[str]:
    """Load the latest note markdown for a document from SQLite."""
    from db.sqlite import list_notes_for_document

    notes = list_notes_for_document(document_id)
    if not notes:
        return None
    # list_notes_for_document returns newest first
    result = notes[0].get("result", {})
    return result.get("note_markdown") or result.get("note", {}).get("note_markdown")


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


def build_cases(doc, note_markdown: str) -> list[FigurePlacementCase]:
    """Build placement cases from a Document and its note markdown.

    Returns a list of FigurePlacementCase — one per placed figure.
    Figures without image_path (not enriched) are excluded.
    """
    from models.document import BlockType

    fig_blocks = [
        b for b in doc.blocks
        if b.type == BlockType.FIGURE and b.image_path
    ]

    if not fig_blocks:
        LOGGER.info("No enriched FIGURE blocks found in document %s", doc.id)
        return []

    sections = _split_sections(note_markdown)
    if not sections:
        LOGGER.warning("Note has no ## sections — cannot compute placement")
        return []

    placement = _build_placement(sections, fig_blocks)

    cases: list[FigurePlacementCase] = []
    for section_idx, blocks in placement.items():
        heading, body = sections[section_idx]
        for b in blocks:
            cases.append(
                FigurePlacementCase(
                    figure_order=b.order,
                    image_path=b.image_path,
                    page=b.metadata.page,
                    section_idx=section_idx,
                    section_heading=heading,
                    section_body=body,
                    figure_description=_get_figure_description(b),
                )
            )

    LOGGER.info(
        "Built %d placement cases from %d figures across %d sections",
        len(cases),
        len(fig_blocks),
        len(sections),
    )
    return cases


def run_placement_eval(
    doc,
    note_markdown: str,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_llm_judge: bool = False,
) -> FigurePlacementReport:
    """Run figure placement quality evaluation.

    Primary scoring uses embedding cosine similarity.  If ``use_llm_judge``
    is True, also calls the LLM judge and stores the result in
    ``llm_score``/``llm_reason`` on each ``FigurePlacementResult``.

    Args:
        doc: Parsed Document with enriched FIGURE blocks.
        note_markdown: Raw note markdown string.
        judge_model: OpenAI model to use for optional LLM judge.
        dry_run: If True, skip all API calls and return zero-filled results.
        use_llm_judge: If True, run LLM-as-judge in addition to embeddings.

    Returns:
        FigurePlacementReport with per-figure scores.
    """
    from models.document import BlockType

    total_figures = sum(
        1 for b in doc.blocks
        if b.type == BlockType.FIGURE and b.image_path
    )

    cases = build_cases(doc, note_markdown)
    sections = _split_sections(note_markdown)

    if dry_run or not cases:
        # Return zero-filled results without any API calls
        results: list[FigurePlacementResult] = [
            FigurePlacementResult(
                figure_order=c.figure_order,
                image_path=c.image_path,
                page=c.page,
                section_idx=c.section_idx,
                section_heading=c.section_heading,
                figure_description=c.figure_description,
                cosine_sim=0.0,
                best_sim=0.0,
                placement_score=0.0,
                is_optimal=False,
                optimality_gap=0.0,
                best_match_section_idx=c.section_idx,
                best_match_heading=c.section_heading,
                passed=False,
                llm_score=None,
                llm_reason="dry-run: no API calls",
            )
            for c in cases
        ]
    else:
        results = _score_by_embedding(cases, sections)

        if use_llm_judge:
            case_map = {c.figure_order: c for c in cases}
            for r in results:
                case = case_map[r.figure_order]
                llm_score, llm_reason = _judge_one(case, judge_model)
                r.llm_score = round(llm_score, 4)
                r.llm_reason = llm_reason

    placed = len(results)
    mean_cosine = sum(r.cosine_sim for r in results) / placed if placed else 0.0
    mean_placement = sum(r.placement_score for r in results) / placed if placed else 0.0
    pass_rate = sum(1 for r in results if r.passed) / placed if placed else 0.0
    optimal_rate = sum(1 for r in results if r.is_optimal) / placed if placed else 0.0

    return FigurePlacementReport(
        document_id=doc.id,
        source=doc.source,
        embedding_model=EMBED_MODEL,
        judge_model=judge_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_figures=total_figures,
        placed_figures=placed,
        mean_cosine_sim=round(mean_cosine, 4),
        mean_placement_score=round(mean_placement, 4),
        pass_rate=round(pass_rate, 4),
        optimal_rate=round(optimal_rate, 4),
        per_figure=results,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(report: FigurePlacementReport) -> None:
    print(f"\n=== Figure Placement Eval: {report.source} ===")
    print(f"  Document ID        : {report.document_id}")
    print(f"  Embedding model    : {report.embedding_model}")
    if any(r.llm_score is not None for r in report.per_figure):
        print(f"  LLM judge model    : {report.judge_model}")
    print(f"  Total figures      : {report.total_figures}")
    print(f"  Placed figures     : {report.placed_figures}")
    print(f"  Mean cosine sim    : {report.mean_cosine_sim:.4f}")
    print(f"  Mean placement scr : {report.mean_placement_score:.4f}")
    print(f"  Pass rate (≥{_PASS_THRESHOLD}): {report.pass_rate:.1%}")
    print(f"  Optimal rate       : {report.optimal_rate:.1%}")
    print()
    for r in report.per_figure:
        status = "✓" if r.passed else "✗"
        opt_mark = "*" if r.is_optimal else " "
        print(
            f"  [{status}]{opt_mark} order={r.figure_order:3d} page={str(r.page):>4s}"
            f"  section[{r.section_idx}] {r.section_heading[:35]:35s}"
            f"  sim={r.cosine_sim:.3f}  score={r.placement_score:.3f}"
            f"  gap={r.optimality_gap:.3f}"
        )
        if r.llm_score is not None:
            print(f"       LLM={r.llm_score:.2f} → {r.llm_reason}")
        elif not r.passed:
            best_h = r.best_match_heading[:40]
            print(f"       best: section[{r.best_match_section_idx}] {best_h}")


def _save_report(report: FigurePlacementReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "document_id": report.document_id,
        "source": report.source,
        "embedding_model": report.embedding_model,
        "judge_model": report.judge_model,
        "timestamp": report.timestamp,
        "summary": {
            "total_figures": report.total_figures,
            "placed_figures": report.placed_figures,
            "mean_cosine_sim": report.mean_cosine_sim,
            "mean_placement_score": report.mean_placement_score,
            "pass_rate": report.pass_rate,
            "optimal_rate": report.optimal_rate,
        },
        "per_figure": [asdict(r) for r in report.per_figure],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_figure_placement_eval."""
    parser = argparse.ArgumentParser(
        description="Evaluate figure placement quality in generated study notes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Original PDF file path")
    group.add_argument("--document-id", type=str, help="Document ID (from parse cache)")
    parser.add_argument(
        "--judge",
        type=str,
        default=_DEFAULT_JUDGE_MODEL,
        help=f"LLM judge model (default: {_DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output JSON path (default: {_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all API calls; just print the figure-to-section mapping",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run optional LLM-as-judge scoring in addition to embedding similarity",
    )
    args = parser.parse_args(argv)

    # Load document
    if args.file:
        from parsers.pdf_parser import parse_pdf
        from utils.cache import load_cached_parse

        file_path = args.file
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        LOGGER.info("Loading document from cache or parsing %s", file_path)
        doc = load_cached_parse(file_path)
        if doc is None:
            LOGGER.info("Cache miss — running parse_pdf (this may take a while)")
            doc = parse_pdf(str(file_path))
    else:
        doc = _load_document_by_id(args.document_id)
        if doc is None:
            raise ValueError(f"Document not found in cache for id={args.document_id}")

    LOGGER.info("Loaded document id=%s source=%s blocks=%d", doc.id, doc.source, len(doc.blocks))

    # Load note
    note_markdown = _load_note_markdown(doc.id)
    if not note_markdown:
        raise ValueError(
            f"No note found in SQLite for document_id={doc.id}. "
            "Generate a note first via the UI or note generator."
        )

    LOGGER.info("Loaded note markdown (%d chars)", len(note_markdown))

    # Run eval
    report = run_placement_eval(
        doc,
        note_markdown,
        judge_model=args.judge,
        dry_run=args.dry_run,
        use_llm_judge=args.llm_judge,
    )

    _print_summary(report)

    if not args.dry_run:
        _save_report(report, args.output)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
