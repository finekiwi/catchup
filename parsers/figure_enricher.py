"""PDF figure enricher: replace [figure] placeholder blocks with VLM analysis."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.document import Block, BlockType, Document
from parsers.image_parser import (
    ImageType,
    _PROMPT_BY_IMAGE_TYPE,
    classify_image,
    map_vlm_output_to_block,
    parse_vlm_output,
)

# Image types worth including in the study note.
# OTHER = logos, cover art, decorative illustrations → skip.
_USEFUL_IMAGE_TYPES = {
    ImageType.CODE_SCREENSHOT,
    ImageType.DIAGRAM,
    ImageType.TEXT_CAPTURE,
    ImageType.EQUATION,
}
from utils.cache import load_docling_doc, save_docling_doc
from vlm.client import call_vlm

LOGGER = logging.getLogger(__name__)

_DEFAULT_FIGURES_DIR = Path("data/figures")


def enrich_pdf_figures(
    doc: Document,
    vlm_model: str,
    file_path: str,
    *,
    figures_dir: Path | None = None,
    max_figures: int = 15,
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

    # Match by zip (document order) with page number cross-check
    matched: list[tuple[Block, object]] = []
    for fb, (pi, pi_page) in zip(figure_blocks, picture_items):
        fb_page = fb.metadata.page
        if fb_page is not None and pi_page is not None and fb_page != pi_page:
            LOGGER.warning(
                "Figure-PictureItem page mismatch (block page=%s, item page=%s) — skipping pair",
                fb_page,
                pi_page,
            )
            continue
        matched.append((fb, pi))

    if not matched:
        return doc

    if len(matched) > max_figures:
        LOGGER.warning(
            "PDF has %d figures; capping at max_figures=%d — skipping remaining %d",
            len(matched),
            max_figures,
            len(matched) - max_figures,
        )
        matched = matched[:max_figures]

    output_dir = figures_dir or (_DEFAULT_FIGURES_DIR / doc.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect Streamlit runtime to avoid ThreadPoolExecutor deadlock
    use_sequential = _is_streamlit_runtime()

    seen_hashes: set[str] = set()  # image dedup across all figures in this doc
    if use_sequential:
        _process_sequential(doc, matched, dl_doc, output_dir, vlm_model, seen_hashes)
    else:
        _process_parallel(doc, matched, dl_doc, output_dir, vlm_model, seen_hashes)

    return doc


def _process_sequential(
    doc: Document,
    matched: list[tuple[Block, object]],
    dl_doc: object,
    output_dir: Path,
    vlm_model: str,
    seen_hashes: set[str],
) -> None:
    """Process figures sequentially (used inside Streamlit runtime)."""
    for fb, pi in matched:
        _enrich_one(doc, fb, pi, dl_doc, output_dir, vlm_model, seen_hashes)


def _process_parallel(
    doc: Document,
    matched: list[tuple[Block, object]],
    dl_doc: object,
    output_dir: Path,
    vlm_model: str,
    seen_hashes: set[str],
) -> None:
    """Process figures in parallel using ThreadPoolExecutor."""
    max_workers = min(len(matched), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_enrich_one, doc, fb, pi, dl_doc, output_dir, vlm_model, seen_hashes): fb
            for fb, pi in matched
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
    block: Block,
    picture_item: object,
    dl_doc: object,
    output_dir: Path,
    vlm_model: str,
    seen_hashes: set[str],
) -> None:
    """Extract, classify, analyze one figure and replace its block in doc.blocks."""
    image_path = output_dir / f"{block.order}.png"

    if not image_path.exists():
        # Image not yet on disk — extract from DoclingDocument.
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
            return

        try:
            pil_image.save(image_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to save figure image for block order=%d: %s", block.order, exc)
            return
    else:
        LOGGER.debug("Figure block order=%d: using pre-extracted image from disk", block.order)

    image_path_str = str(image_path)

    # Dedup: skip if identical image content already processed (e.g. repeated cover/logo).
    try:
        img_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
        if img_hash in seen_hashes:
            LOGGER.info("Duplicate figure skipped for block order=%d (hash=%s)", block.order, img_hash)
            image_path.unlink(missing_ok=True)
            return
        seen_hashes.add(img_hash)
    except Exception:
        pass  # File not written (e.g. test mock) — skip dedup

    # VLM classify + analyze
    try:
        image_type = classify_image(image_path_str, vlm_model)

        # Skip decorative/logo images — only meaningful types go into the note.
        if image_type not in _USEFUL_IMAGE_TYPES:
            LOGGER.info(
                "Figure block order=%d classified as %s — skipping (non-educational content)",
                block.order,
                image_type,
            )
            image_path.unlink(missing_ok=True)
            return
        analysis_prompt = _PROMPT_BY_IMAGE_TYPE[image_type]
        vlm_result = call_vlm(vlm_model, image_path_str, analysis_prompt, stage="figure_analysis")

        if not vlm_result.success:
            LOGGER.warning(
                "VLM analysis failed for block order=%d: %s", block.order, vlm_result.error
            )
            return

        parsed = parse_vlm_output(vlm_result.content, image_type)
        if parsed.errors:
            LOGGER.warning(
                "VLM analysis returned errors for block order=%d: %s — skipping image",
                block.order,
                parsed.errors,
            )
            return
        enriched = map_vlm_output_to_block(
            image_type=image_type,
            payload=parsed,
            order=block.order,
            image_path=image_path_str,
        )
        # Preserve original page metadata
        enriched.metadata.page = block.metadata.page
        if enriched.metadata.caption is None:
            enriched.metadata.caption = block.metadata.caption

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "VLM enrichment failed for block order=%d: %s — keeping placeholder",
            block.order,
            exc,
        )
        return

    # Replace the block in doc.blocks (thread-safe: each block is unique by order)
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
