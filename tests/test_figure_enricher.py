"""Unit tests for parsers/figure_enricher.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.document import Block, BlockMetadata, BlockType, Document, DocumentFormat, ImageType
from parsers.figure_enricher import enrich_pdf_figures
from vlm.client import VLMResult


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _figure_block(order: int = 0, page: int | None = 1) -> Block:
    """Return a FIGURE placeholder block."""
    return Block(
        type=BlockType.FIGURE,
        content="[figure]",
        order=order,
        metadata=BlockMetadata(page=page),
    )


def _text_block(order: int = 1) -> Block:
    """Return a TEXT block."""
    return Block(
        type=BlockType.TEXT,
        content="Some text content here.",
        order=order,
        metadata=BlockMetadata(page=1),
    )


def _make_doc(blocks: list[Block] | None = None) -> Document:
    """Build a minimal Document fixture."""
    return Document(
        id="abcd1234abcd1234",
        source="sample.pdf",
        format=DocumentFormat.PDF,
        blocks=blocks or [_figure_block()],
    )


def _picture_item(page: int | None = 1) -> MagicMock:
    """Build a fake PictureItem with get_image()."""
    item = MagicMock()
    item.prov = [SimpleNamespace(page_no=page)] if page is not None else []
    pil_image = MagicMock()
    pil_image.save = MagicMock()
    item.get_image = MagicMock(return_value=pil_image)
    return item


def _vlm_result(content: str, success: bool = True) -> VLMResult:
    return VLMResult(content=content, model="gpt-4o-mini", success=success, error=None if success else "err")


def _diagram_json() -> str:
    return json.dumps({
        "schema_version": "v1.1.0",
        "diagram_type": "flowchart",
        "title": "Pipeline",
        "description": "End-to-end data pipeline diagram.",
        "components": [{"name": "A", "role": "input"}, {"name": "B", "role": "output"}],
        "relationships": [{"from": "A", "to": "B", "label": "feeds"}],
        "flow_summary": "A feeds B",
        "has_truncation": False,
        "confidence": 0.92,
        "errors": [],
    })


def _classify_json(image_type: str = "diagram") -> str:
    return json.dumps({"image_type": image_type, "confidence": 0.9})


def _dl_doc_with_items(picture_items: list[MagicMock]) -> MagicMock:
    """Build a fake DoclingDocument that iterates the given PictureItems."""
    dl_doc = MagicMock()
    dl_doc.iterate_items.return_value = [(item, 0) for item in picture_items]
    return dl_doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _std_patches(tmp_path: Path | None = None) -> list:
    """Return the standard set of patches used in most enricher tests."""
    return [
        patch("parsers.figure_enricher.load_docling_doc"),
        patch("parsers.figure_enricher._is_picture_item", return_value=True),
        patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
        patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
        patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(_diagram_json())),
    ]


class TestEnrichReplacesPlaceholder:
    """Core happy-path: [figure] placeholder replaced with VLM description."""

    def test_enrich_replaces_placeholder_with_vlm_content(self, tmp_path: Path) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result is doc
        assert len(result.blocks) == 1
        enriched = result.blocks[0]
        assert enriched.content != "[figure]"
        assert len(enriched.content) > 20
        assert enriched.image_path is not None

    def test_enrich_skips_non_figure_blocks(self, tmp_path: Path) -> None:
        """TEXT blocks must not be modified."""
        text = _text_block(order=0)
        fig = _figure_block(order=1, page=1)
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([text, fig])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert doc.blocks[0].content == "Some text content here."


class TestEnrichGracefulDegradation:
    """Enrichment should never crash the pipeline on failures."""

    def test_enrich_graceful_no_docling_cache(self) -> None:
        """Missing DoclingDocument cache → doc returned unchanged."""
        doc = _make_doc()
        with patch("parsers.figure_enricher.load_docling_doc", return_value=None):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf")
        assert result.blocks[0].content == "[figure]"

    def test_enrich_graceful_vlm_failure(self, tmp_path: Path) -> None:
        """VLM call failure → block keeps [figure] placeholder."""
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result("", success=False),
            ),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"

    def test_enrich_graceful_no_image_data(self, tmp_path: Path) -> None:
        """get_image() returns None → block keeps [figure] placeholder."""
        pi = _picture_item(page=1)
        pi.get_image.return_value = None
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"


class TestEnrichOrdering:
    """Multiple figures must be processed in document order."""

    def test_enrich_multiple_figures(self, tmp_path: Path) -> None:
        """Two figures enriched; original non-figure blocks untouched."""
        fig0 = _figure_block(order=0, page=1)
        fig1 = _figure_block(order=2, page=2)
        text = _text_block(order=1)
        pi0 = _picture_item(page=1)
        pi1 = _picture_item(page=2)
        dl_doc = _dl_doc_with_items([pi0, pi1])
        doc = _make_doc([fig0, text, fig1])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert doc.blocks[0].content != "[figure]"
        assert doc.blocks[1].content == "Some text content here."
        assert doc.blocks[2].content != "[figure]"

    def test_enrich_caps_at_max_figures(self, tmp_path: Path) -> None:
        """Only first max_figures figures are processed; rest keep placeholder."""
        blocks = [_figure_block(order=i, page=1) for i in range(5)]
        pis = [_picture_item(page=1) for _ in range(5)]
        dl_doc = _dl_doc_with_items(pis)
        doc = _make_doc(blocks)

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path, max_figures=2)

        enriched_count = sum(1 for b in doc.blocks if b.content != "[figure]")
        placeholder_count = sum(1 for b in doc.blocks if b.content == "[figure]")
        assert enriched_count == 2
        assert placeholder_count == 3


class TestEnrichImagePath:
    """Extracted figure image must be saved to expected path."""

    def test_enrich_saves_image_to_expected_path(self, tmp_path: Path) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=7, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        # PIL save was called with path containing block order
        save_args = pi.get_image.return_value.save.call_args
        saved_path = Path(save_args[0][0])
        assert saved_path.name == "7.png"
        assert saved_path.parent == tmp_path


class TestEnrichPageMismatch:
    """Page number cross-check should skip mismatched pairs."""

    def test_page_mismatch_skips_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """PictureItem page != Block page → pair skipped, block keeps placeholder."""
        pi = _picture_item(page=3)  # page 3, but block is page 1
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        import logging
        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            caplog.at_level(logging.WARNING, logger="parsers.figure_enricher"),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert "page mismatch" in caplog.text.lower()


@pytest.mark.integration
class TestEnrichedFiguresPassNoiseFilter:
    """Enriched FIGURE blocks must pass the noise filter (content > 20 chars)."""

    def test_enriched_figures_pass_noise_filter(self, tmp_path: Path) -> None:
        from llm.block_filter import is_noise_block

        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ),
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        enriched_block = doc.blocks[0]
        assert not is_noise_block(enriched_block), (
            f"Enriched figure block should pass noise filter; content={enriched_block.content!r}"
        )
