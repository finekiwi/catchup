"""PDF figure enricher: replace [figure] placeholder blocks with VLM analysis."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.document import Block, BlockType, Document
from parsers.image_parser import (
    ImageType,
    _PROMPT_GETTER_BY_IMAGE_TYPE,
    classify_image,
    map_vlm_output_to_block,
    parse_vlm_output,
)
from utils.cache import load_docling_doc, save_docling_doc
from vlm.client import call_vlm

# Image types worth including in the study note.
# OTHER = logos, cover art, decorative illustrations → skip.
_USEFUL_IMAGE_TYPES = {
    ImageType.CODE_SCREENSHOT,
    ImageType.DIAGRAM,
    ImageType.TEXT_CAPTURE,
    ImageType.EQUATION,
}
_POST_FILTER_IMAGE_TYPES = {
    ImageType.DIAGRAM,
    ImageType.TEXT_CAPTURE,
}
_DECORATIVE_ANALYSIS_KEYWORDS = (
    "logo",
    "publisher logo",
    "company logo",
    "oreilly",
    "o'reilly",
    "packt",
    "hanbit",
    "wikibooks",
    "book cover",
    "cover art",
    "cover image",
    "title page",
    "chapter divider",
    "divider art",
    "decorative",
    "decorative illustration",
    "illustration",
    "mascot",
    "mascot character",
    "cartoon",
    "cartoon character",
    "paper craft",
    "paper model",
    "origami",
    "표지",
    "책 표지",
    "로고",
    "한빛미디어",
    "위키북스",
    "출판사 로고",
    "장식",
    "장식 이미지",
    "장식 일러스트",
    "일러스트",
    "삽화",
    "마스코트",
    "캐릭터",
    "만화 캐릭터",
    "챕터 구분",
    "챕터 디바이더",
    "종이 모형",
    "종이 모델",
    "종이 공예",
)
_NON_INSTRUCTIONAL_TEXT_CAPTURE_KEYWORDS = (
    "예제 코드 제공",
    "샘플 코드 제공",
    "sample code provided",
    "example code provided",
    "download code",
    "download sample code",
    "code provided",
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_FIGURES_DIR = Path("data/figures")


@dataclass(frozen=True)
class _PreparedFigure:
    """Figure candidate with a materialized image path ready for VLM work."""

    block: Block
    image_path: str


def enrich_pdf_figures(
    doc: Document,
    vlm_model: str,
    file_path: str,
    *,
    figures_dir: Path | None = None,
    max_figures: int = 30,
    language: str = "ko",
) -> Document:
    """Enrich FIGURE blocks with VLM analysis. Mutates doc in-place and returns it.

    Loads the cached DoclingDocument for file_path, matches PictureItems to
    FIGURE blocks by document order + page cross-check, extracts each image,
    runs VLM classification + analysis, and replaces the placeholder block
    content with the structured VLM description.

    Args:
        doc: Parsed Document with FIGURE blocks containing "[figure]" placeholders.
        vlm_model: VLM model identifier (passed to call_vlm).
        file_path: Original PDF path used to look up the DoclingDocument cache.
        figures_dir: Directory to save extracted figure images.
                     Defaults to data/figures/{doc.id}/.
        max_figures: Maximum number of figures to process. Extra figures are
                     skipped with a warning to limit cost and latency.

    Returns:
        The same Document with enriched FIGURE blocks (mutated in-place).
    """
    dl_doc = load_docling_doc(Path(file_path))
    if dl_doc is None:
        # Cache miss or stale (e.g. built without generate_picture_images=True).
        # Re-run DocumentConverter with picture image generation and rebuild the cache.
        dl_doc = _reconvert_with_images(file_path)
        if dl_doc is None:
            LOGGER.warning(
                "DoclingDocument unavailable for %s — skipping figure enrichment", file_path
            )
            return doc

    # Collect figure blocks and picture items in document order
    figure_blocks: list[Block] = [b for b in doc.blocks if b.type == BlockType.FIGURE]
    picture_items: list[tuple[object, int | None]] = [
        (item, _extract_page(item))
        for item, _level in dl_doc.iterate_items()
        if _is_picture_item(item)
    ]

    if not figure_blocks or not picture_items:
        return doc

    # Check whether the DoclingDocument actually has extractable picture images.
    # Docling's JSON cache serialises image=None for PictureItems, so
    # PictureItem.get_image() returns None for every item when loaded from cache.
    # We test the first item and fall back to live re-conversion if needed.
    #
    # Skip the test when the output directory already exists: parse_pdf pre-saves
    # all figure images during the initial conversion (while the live DoclingDocument
    # is available), so _enrich_one can use those files directly without get_image().
    # This also covers re-uploads where images were saved in a previous session.
    output_dir_early = figures_dir or (_DEFAULT_FIGURES_DIR / doc.id)
    _enriched_before = output_dir_early.is_dir()
    if not _enriched_before:
        _test_item = picture_items[0][0]
        _test_img = None
        try:
            _test_img = _test_item.get_image(dl_doc)
        except Exception:  # noqa: BLE001
            pass
        if _test_img is None:
            LOGGER.info(
                "Cached DoclingDocument has no extractable picture images "
                "(Docling JSON cache limitation) — re-converting %s with live pipeline",
                Path(file_path).name,
            )
            fresh = _reconvert_with_images(file_path)
            if fresh is not None:
                dl_doc = fresh

    # Match FIGURE blocks to PictureItems by page number (greedy)
    matched = _match_figures_by_page(figure_blocks, picture_items)

    if not matched:
        return doc

    output_dir = figures_dir or (_DEFAULT_FIGURES_DIR / doc.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_figures(matched, dl_doc, output_dir)
    if not prepared:
        return doc

    if len(prepared) > max_figures:
        LOGGER.warning(
            "PDF has %d unique figures after dedup; capping at max_figures=%d — skipping remaining %d",
            len(prepared),
            max_figures,
            len(prepared) - max_figures,
        )
        for skipped in prepared[max_figures:]:
            _mark_block_skipped(skipped.block)
        prepared = prepared[:max_figures]

    # Detect Streamlit runtime to avoid ThreadPoolExecutor deadlock
    use_sequential = _is_streamlit_runtime()
    if use_sequential:
        _process_sequential(doc, prepared, vlm_model, language)
    else:
        _process_parallel(doc, prepared, vlm_model, language)

    return doc


def _prepare_figures(
    matched: list[tuple[Block, object]],
    dl_doc: object,
    output_dir: Path,
) -> list[_PreparedFigure]:
    """Materialize figure images and deduplicate them before VLM analysis."""
    prepared: list[_PreparedFigure] = []
    seen_hashes: set[str] = set()

    for block, picture_item in matched:
        image_path = _ensure_image_path(block, picture_item, dl_doc, output_dir)
        if image_path is None:
            _mark_block_skipped(block)
            continue

        try:
            img_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to hash figure image for block order=%d: %s", block.order, exc)
            _mark_block_skipped(block)
            continue

        if img_hash in seen_hashes:
            LOGGER.info(
                "Duplicate figure skipped (hash=%s, block_order=%d)",
                img_hash[:8],
                block.order,
            )
            _mark_block_skipped(block)
            continue

        seen_hashes.add(img_hash)
        prepared.append(_PreparedFigure(block=block, image_path=str(image_path)))

    return prepared


def _process_sequential(
    doc: Document,
    prepared: list[_PreparedFigure],
    vlm_model: str,
    language: str = "ko",
) -> None:
    """Process figures sequentially (used inside Streamlit runtime)."""
    for candidate in prepared:
        _enrich_one(doc, candidate, vlm_model, language)


def _process_parallel(
    doc: Document,
    prepared: list[_PreparedFigure],
    vlm_model: str,
    language: str = "ko",
) -> None:
    """Process figures in parallel using ThreadPoolExecutor."""
    max_workers = min(len(prepared), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_enrich_one, doc, candidate, vlm_model, language): candidate.block
            for candidate in prepared
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                fb = futures[future]
                LOGGER.warning(
                    "Figure enrichment failed for block order=%d: %s", fb.order, exc
                )


def _enrich_one(
    doc: Document,
    candidate: _PreparedFigure,
    vlm_model: str,
    language: str = "ko",
) -> None:
    """Classify and analyze one prepared figure, replacing the original block on success."""
    block = candidate.block
    image_path_str = candidate.image_path

    try:
        image_type = classify_image(image_path_str, vlm_model)
        if image_type not in _USEFUL_IMAGE_TYPES:
            LOGGER.info(
                "Non-educational figure skipped (type=%s, block_order=%d)",
                image_type.value,
                block.order,
            )
            _mark_block_skipped(block)
            return
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "VLM analysis failed for figure (block_order=%d): %s",
            block.order,
            exc,
        )
        _mark_block_skipped(block)
        return

    analysis_prompt = _PROMPT_GETTER_BY_IMAGE_TYPE[image_type](language)
    try:
        vlm_result = call_vlm(vlm_model, image_path_str, analysis_prompt, stage="figure_analysis")
        if not vlm_result.success:
            LOGGER.warning(
                "VLM analysis failed for figure (block_order=%d): %s",
                block.order,
                vlm_result.error,
            )
            _mark_block_skipped(block)
            return

        parsed = parse_vlm_output(vlm_result.content, image_type)
        if parsed.errors:
            LOGGER.warning(
                "VLM analysis failed for figure (block_order=%d): %s",
                block.order,
                parsed.errors,
            )
            _mark_block_skipped(block)
            return
        if _analysis_looks_decorative(image_type, parsed):
            LOGGER.info(
                "Decorative figure skipped after analysis (type=%s, block_order=%d)",
                image_type.value,
                block.order,
            )
            _mark_block_skipped(block)
            return
        enriched = map_vlm_output_to_block(
            image_type=image_type,
            payload=parsed,
            order=block.order,
            image_path=image_path_str,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "VLM analysis failed for figure (block_order=%d): %s",
            block.order,
            exc,
        )
        _mark_block_skipped(block)
        return

    # Preserve original page metadata and existing caption fallback.
    enriched.metadata.page = block.metadata.page
    if enriched.metadata.caption is None:
        enriched.metadata.caption = block.metadata.caption

    _replace_block(doc, block, enriched)


def _ensure_image_path(
    block: Block,
    picture_item: object,
    dl_doc: object,
    output_dir: Path,
) -> Path | None:
    """Return a saved PNG path for the figure, extracting it if needed."""
    image_path = output_dir / f"{block.order}.png"
    if image_path.exists():
        LOGGER.debug("Figure block order=%d: using pre-extracted image from disk", block.order)
        return image_path

    pil_image = None
    try:
        pil_image = picture_item.get_image(dl_doc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("get_image() failed for block order=%d: %s", block.order, exc)

    if pil_image is None:
        LOGGER.warning(
            "No raster image data for block order=%d (vector graphic?) — keeping placeholder",
            block.order,
        )
        return None

    try:
        pil_image.save(image_path, format="PNG")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to save figure image for block order=%d: %s", block.order, exc)
        return None
    return image_path


def _mark_block_skipped(block: Block) -> None:
    """Normalize skipped/failed figures so downstream renderers ignore them."""
    block.content = "[figure]"
    block.image_path = None


def _match_figures_by_page(
    figure_blocks: list[Block],
    picture_items: list[tuple[object, int | None]],
) -> list[tuple[Block, object]]:
    """Match FIGURE blocks to PictureItems by page number (greedy).

    For each FIGURE block, finds the first unmatched PictureItem on the same page.
    Falls back to order-based matching when either side has page=None.
    Unmatched blocks are left as-is (retain [figure] placeholder).
    Extra PictureItems with no corresponding block are silently ignored.
    """
    matched: list[tuple[Block, object]] = []
    used: set[int] = set()

    for fb in figure_blocks:
        fb_page = fb.metadata.page
        for idx, (pi, pi_page) in enumerate(picture_items):
            if idx in used:
                continue
            # Match: same page, or either side has no page info (fallback)
            if fb_page is None or pi_page is None or fb_page == pi_page:
                matched.append((fb, pi))
                used.add(idx)
                break
        # If no match found for fb, it retains its [figure] placeholder (no action needed)

    unmatched = len(figure_blocks) - len(matched)
    if unmatched > 0:
        LOGGER.info(
            "Figure matching: %d/%d blocks matched (%d unmatched, %d PictureItems available)",
            len(matched),
            len(figure_blocks),
            unmatched,
            len(picture_items),
        )
    return matched


def _analysis_looks_decorative(image_type: ImageType, payload: Any) -> bool:
    """Return True when VLM analysis describes a decorative/cover-like image."""
    if image_type not in _POST_FILTER_IMAGE_TYPES:
        return False

    normalized_fragments = [
        normalized
        for fragment in _collect_text_fragments(payload)
        if (normalized := _normalize_analysis_text(fragment))
    ]
    if any(
        keyword in normalized
        for normalized in normalized_fragments
        for keyword in _DECORATIVE_ANALYSIS_KEYWORDS
    ):
        return True

    return image_type == ImageType.TEXT_CAPTURE and _text_capture_looks_non_instructional(
        payload,
        normalized_fragments,
    )


def _collect_text_fragments(value: Any) -> list[str]:
    """Recursively collect text fields from parsed VLM payloads."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        fragments: list[str] = []
        for item in value.values():
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, (list, tuple, set)):
        fragments = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if hasattr(value, "model_dump"):
        return _collect_text_fragments(value.model_dump())
    if hasattr(value, "__dict__"):
        return _collect_text_fragments(vars(value))
    return []


def _normalize_analysis_text(text: str) -> str:
    """Normalize analysis text before keyword matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _text_capture_looks_non_instructional(
    payload: Any,
    normalized_fragments: list[str],
) -> bool:
    """Return True when a TEXT_CAPTURE payload is only a short badge-like label."""
    joined = " ".join(normalized_fragments)
    return any(keyword in joined for keyword in _NON_INSTRUCTIONAL_TEXT_CAPTURE_KEYWORDS)


def _replace_block(doc: Document, block: Block, enriched: Block) -> None:
    """Replace the original block instance in the document."""
    for i, b in enumerate(doc.blocks):
        if b is block:
            doc.blocks[i] = enriched
            break


def _extract_page(item: object) -> int | None:
    """Extract page number from a Docling item's provenance."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(first, "page_no", None)
    if page_no is None:
        return None
    try:
        return int(page_no)
    except (TypeError, ValueError):
        return None


def _is_picture_item(item: object) -> bool:
    """Return True if item is a Docling PictureItem (patchable for tests)."""
    try:
        from docling_core.types.doc import PictureItem

        return isinstance(item, PictureItem)
    except ImportError:
        return False


def _reconvert_with_images(file_path: str) -> object | None:
    """Re-run DocumentConverter with generate_picture_images=True and cache the result.

    Used as a fallback when the docling cache is missing or was built without
    picture image data (stale v1 cache). Returns None on any failure.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        LOGGER.info("Re-converting %s with generate_picture_images=True", file_path)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(file_path, raises_on_error=False)
        dl_doc = result.document
        save_docling_doc(Path(file_path), dl_doc)
        return dl_doc
    except Exception as exc:
        LOGGER.warning("Re-conversion with picture images failed for %s: %s", file_path, exc)
        return None


def _is_streamlit_runtime() -> bool:
    """Return True only when running inside an active Streamlit server."""
    try:
        import streamlit.runtime as st_runtime

        return st_runtime.exists()
    except Exception:  # noqa: BLE001
        return False


__all__ = ["enrich_pdf_figures"]
