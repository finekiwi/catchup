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
        image_path=None,
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


def _picture_item(page: int | None = 1, *, image_bytes: bytes | None = None) -> MagicMock:
    """Build a fake PictureItem with get_image()."""
    item = MagicMock()
    item.prov = [SimpleNamespace(page_no=page)] if page is not None else []
    if image_bytes is None:
        image_bytes = f"img-{id(item)}".encode("utf-8")
    pil_image = MagicMock()

    def _save(path: str | Path, format: str = "PNG") -> None:
        del format
        Path(path).write_bytes(image_bytes)

    pil_image.save = MagicMock(side_effect=_save)
    item.get_image = MagicMock(return_value=pil_image)
    return item


def _vlm_result(content: str, success: bool = True) -> VLMResult:
    return VLMResult(content=content, model="gpt-4o-mini", success=success, error=None if success else "err")


def _diagram_json(errors: list[str] | None = None) -> str:
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
        "errors": errors or [],
    })


def _code_json() -> str:
    return json.dumps({
        "schema_version": "v1.1.0",
        "language": "python",
        "code": "print('hello')",
        "code_markdown": "```python\nprint('hello')\n```",
        "description": "Simple code snippet.",
        "has_truncation": False,
        "confidence": 0.9,
        "errors": [],
    })


def _text_json(*, title: str, content: str, has_math: bool = False) -> str:
    return json.dumps({
        "schema_version": "v1.1.0",
        "text_type": "lecture_slide",
        "title": title,
        "content": content,
        "key_points": ["Point A", "Point B"],
        "has_math": has_math,
        "has_truncation": False,
        "confidence": 0.9,
        "errors": [],
    })


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
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

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
        assert result.blocks[0].image_path is None

    def test_enrich_graceful_no_image_data(self, tmp_path: Path) -> None:
        """get_image() returns None → block keeps [figure] placeholder."""
        pi = _picture_item(page=1)
        pi.get_image.return_value = None
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None


class TestEnrichFiltering:
    """Filtering logic should suppress duplicates and non-educational images."""

    def test_dedup_skips_duplicate_figures(self, tmp_path: Path) -> None:
        dup_bytes = b"duplicate-figure"
        pi0 = _picture_item(page=1, image_bytes=dup_bytes)
        pi1 = _picture_item(page=1, image_bytes=dup_bytes)
        dl_doc = _dl_doc_with_items([pi0, pi1])
        doc = _make_doc([
            _figure_block(order=0, page=1),
            _figure_block(order=1, page=1),
        ])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM) as classify_mock,
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(_diagram_json()),
            ) as call_vlm_mock,
        ):
            enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert doc.blocks[0].content != "[figure]"
        assert doc.blocks[0].image_path is not None
        assert doc.blocks[1].content == "[figure]"
        assert doc.blocks[1].image_path is None
        assert classify_mock.call_count == 1
        assert call_vlm_mock.call_count == 1

    def test_other_type_skipped(self, tmp_path: Path) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.OTHER),
            patch("parsers.figure_enricher.call_vlm") as call_vlm_mock,
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None
        call_vlm_mock.assert_not_called()

    @pytest.mark.parametrize(
        ("image_type", "raw_response"),
        [
            (
                ImageType.DIAGRAM,
                json.dumps(
                    {
                        "schema_version": "v1.1.0",
                        "diagram_type": "flowchart",
                        "title": "Mascot character chapter divider",
                        "description": "Decorative illustration used on a title page.",
                        "components": [{"name": "cartoon bee", "role": "mascot"}],
                        "relationships": [],
                        "flow_summary": "Book cover style decorative graphic.",
                        "has_truncation": False,
                        "confidence": 0.88,
                        "errors": [],
                    }
                ),
            ),
            (
                ImageType.TEXT_CAPTURE,
                _text_json(
                    title="장식 일러스트",
                    content="캐릭터 중심의 표지 이미지와 출판사 로고가 포함된 챕터 구분 페이지.",
                ),
            ),
        ],
    )
    def test_decorative_analysis_keywords_clear_image_path(
        self,
        tmp_path: Path,
        image_type: ImageType,
        raw_response: str,
    ) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=image_type),
            patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(raw_response)),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None

    def test_short_text_capture_badge_is_skipped(self, tmp_path: Path) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.TEXT_CAPTURE),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(
                    _text_json(title="예제 코드 제공", content="예제 코드 제공")
                ),
            ),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None

    def test_publisher_name_text_capture_is_skipped(self, tmp_path: Path) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.TEXT_CAPTURE),
            patch(
                "parsers.figure_enricher.call_vlm",
                return_value=_vlm_result(
                    _text_json(title="위키북스", content="위키북스")
                ),
            ),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None

    @pytest.mark.parametrize("failure_mode", ["parsed_errors", "parse_exception"])
    def test_vlm_error_clears_image_path(self, tmp_path: Path, failure_mode: str) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        call_vlm_result = _vlm_result(
            _diagram_json(errors=["Unable to extract or interpret text from image."])
        )
        parse_patch = patch("parsers.figure_enricher.parse_vlm_output")
        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch("parsers.figure_enricher.call_vlm", return_value=call_vlm_result),
            parse_patch as parse_mock,
        ):
            if failure_mode == "parsed_errors":
                parse_mock.side_effect = None
                parse_mock.return_value = SimpleNamespace(errors=["Unable to extract or interpret text from image."])
            else:
                parse_mock.side_effect = ValueError("bad json")
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content == "[figure]"
        assert result.blocks[0].image_path is None

    @pytest.mark.parametrize(
        ("image_type", "raw_response"),
        [
            (ImageType.CODE_SCREENSHOT, _code_json()),
            (ImageType.DIAGRAM, _diagram_json()),
            (
                ImageType.TEXT_CAPTURE,
                _text_json(
                    title="Slide Summary",
                    content="Important lecture notes about data pipelines.",
                ),
            ),
            (
                ImageType.EQUATION,
                _text_json(
                    title="Equation",
                    content="y = mx + b with explanatory text.",
                    has_math=True,
                ),
            ),
        ],
    )
    def test_educational_types_enriched(
        self,
        tmp_path: Path,
        image_type: ImageType,
        raw_response: str,
    ) -> None:
        pi = _picture_item(page=1)
        dl_doc = _dl_doc_with_items([pi])
        doc = _make_doc([_figure_block(order=0, page=1)])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=image_type),
            patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(raw_response)),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content != "[figure]"
        assert result.blocks[0].image_path is not None
        assert result.blocks[0].metadata.page == 1


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
    """Page-based matching logic: correct pairs matched, unmatched blocks keep placeholder."""

    def test_page_no_match_block_keeps_placeholder(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """No PictureItem on same page as block → block keeps [figure] placeholder."""
        pi = _picture_item(page=3)  # page 3, but block is page 1 — no match
        dl_doc = _dl_doc_with_items([pi])
        block = _figure_block(order=0, page=1)
        block.image_path = "/tmp/stale.png"
        doc = _make_doc([block])

        import logging
        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            caplog.at_level(logging.INFO, logger="parsers.figure_enricher"),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        # Block keeps [figure] content; image_path left as-is (no _mark_block_skipped called)
        assert result.blocks[0].content == "[figure]"
        # New logic logs INFO about unmatched figures, not a WARNING about page mismatch
        assert "page mismatch" not in caplog.text.lower()
        assert "figure matching" in caplog.text.lower()

    def test_page_based_match_same_page_two_figures(self, tmp_path: Path) -> None:
        """Two FIGURE blocks and two PictureItems on the same page → both matched and enriched."""
        pi0 = _picture_item(page=2)
        pi1 = _picture_item(page=2)
        dl_doc = _dl_doc_with_items([pi0, pi1])
        doc = _make_doc([
            _figure_block(order=0, page=2),
            _figure_block(order=1, page=2),
        ])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(_diagram_json())),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content != "[figure]"
        assert result.blocks[1].content != "[figure]"

    def test_page_based_match_extra_picture_items_ignored(self, tmp_path: Path) -> None:
        """3 PictureItems but only 2 FIGURE blocks (pages 1, 3) → 2 matched, extra item ignored."""
        pi_p1 = _picture_item(page=1)
        pi_p2 = _picture_item(page=2)  # extra — no corresponding FIGURE block
        pi_p3 = _picture_item(page=3)
        dl_doc = _dl_doc_with_items([pi_p1, pi_p2, pi_p3])
        doc = _make_doc([
            _figure_block(order=0, page=1),
            _figure_block(order=1, page=3),
        ])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(_diagram_json())),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        assert result.blocks[0].content != "[figure]"
        assert result.blocks[1].content != "[figure]"

    def test_page_based_match_cascade_prevented(self, tmp_path: Path) -> None:
        """Old zip logic cascaded: blocks pages=[1,3], items pages=[2,3] → both skipped.
        New page-based logic: block p=1 has no match (skip), block p=3 matches item p=3 (enriched).
        """
        pi_p2 = _picture_item(page=2)  # no block on page 2
        pi_p3 = _picture_item(page=3)
        dl_doc = _dl_doc_with_items([pi_p2, pi_p3])
        doc = _make_doc([
            _figure_block(order=0, page=1),
            _figure_block(order=1, page=3),
        ])

        with (
            patch("parsers.figure_enricher.load_docling_doc", return_value=dl_doc),
            patch("parsers.figure_enricher._is_picture_item", return_value=True),
            patch("parsers.figure_enricher._is_streamlit_runtime", return_value=False),
            patch("parsers.figure_enricher.classify_image", return_value=ImageType.DIAGRAM),
            patch("parsers.figure_enricher.call_vlm", return_value=_vlm_result(_diagram_json())),
        ):
            result = enrich_pdf_figures(doc, "gpt-4o-mini", "sample.pdf", figures_dir=tmp_path)

        # block p=1: no matching item → keeps placeholder
        assert result.blocks[0].content == "[figure]"
        # block p=3: matched with pi_p3 → enriched
        assert result.blocks[1].content != "[figure]"


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
