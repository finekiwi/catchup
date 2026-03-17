"""AppTest-based runtime coverage for the Streamlit demo UI."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
import streamlit.testing.v1.local_script_runner as local_script_runner
from llm.note_editor import NoteEditResult
from llm.note_generator import SUPPORTED_LLM_MODELS
from llm.section_splitter import SectionInfo
from models.document import (
    Block,
    BlockMetadata,
    BlockType,
    Document,
    DocumentFormat,
    DocumentMetadata,
)
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.testing.v1 import AppTest
from vlm.client import SUPPORTED_MODELS

DEMO_APP_PATH = Path(__file__).resolve().parents[1] / "ui" / "demo.py"
DEFAULT_VLM_MODEL = SUPPORTED_MODELS[0]
DEFAULT_LLM_MODEL = SUPPORTED_LLM_MODELS[0]
TEST_SESSION_ID = "test session id"
UNSET = object()
ANALYSIS_CACHE_VERSION = "v2"


@dataclass(frozen=True)
class UploadFixture:
    """Single uploaded file fixture injected into Streamlit's test uploader."""

    file_id: str
    name: str
    mime_type: str
    data: bytes

    @property
    def file_hash(self) -> str:
        """Return the SHA256 used by the app's cache keys."""
        return hashlib.sha256(self.data).hexdigest()

    def to_uploaded_file_rec(self) -> UploadedFileRec:
        """Convert the fixture into Streamlit's in-memory uploaded file record."""
        return UploadedFileRec(
            file_id=self.file_id,
            name=self.name,
            type=self.mime_type,
            data=self.data,
        )


@dataclass
class DemoAppHarness:
    """Stateful wrapper around AppTest plus the patched external dependencies."""

    app: AppTest
    parse_pdf: MagicMock
    parse_ipynb: MagicMock
    parse_image: MagicMock
    generate_note: MagicMock
    edit_section: MagicMock
    save_document: MagicMock
    save_note: MagicMock
    get_document: MagicMock
    get_note: MagicMock
    list_documents: MagicMock
    list_notes_for_document: MagicMock
    delete_document: MagicMock
    delete_document_index: MagicMock
    index_document: MagicMock
    has_document_vectors: MagicMock
    rag_query: MagicMock
    rewrite_query: MagicMock
    enrich_pdf_figures: MagicMock
    link_concepts: MagicMock
    pyperclip_copy: MagicMock
    documents_db: dict[str, Document]
    notes_db: dict[tuple[str, str, str], dict[str, Any]]
    vector_documents: set[str]
    _current_upload: dict[str, UploadFixture | None]

    def run(self, upload: UploadFixture | None | object = UNSET) -> AppTest:
        """Rerun the app, preserving or overriding the current uploaded file."""
        if upload is not UNSET:
            self._current_upload["value"] = upload

        current_upload = self._current_upload["value"]
        if current_upload is None:
            self.app = self.app.run()
            return self.app

        widget_states = self.app._tree.get_widget_states()
        uploader = self.app.get("file_uploader")[0]
        target_state = next(
            (
                widget
                for widget in widget_states.widgets
                if widget.id == uploader.proto.id
            ),
            None,
        )
        if target_state is None:
            target_state = widget_states.widgets.add()
            target_state.id = uploader.proto.id
        else:
            target_state.ClearField("file_uploader_state_value")

        file_info = target_state.file_uploader_state_value.uploaded_file_info.add()
        file_info.file_id = current_upload.file_id
        file_info.name = current_upload.name
        file_info.size = len(current_upload.data)

        self.app = self.app._run(widget_states)
        return self.app

    def click_button(
        self,
        *,
        label: str | None = None,
        key: str | None = None,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Click a button by label or key and rerun the app."""
        if key is not None:
            self.app.button(key=key).click()
            return self.run(upload=upload)
        for button in self.app.button:
            if button.label == label:
                button.click()
                return self.run(upload=upload)
        raise AssertionError(f"Button not found: label={label!r}, key={key!r}")

    def set_radio(
        self,
        key: str,
        value: Any,
        *,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Set a radio widget value and rerun the app."""
        self.app.radio(key=key).set_value(value)
        return self.run(upload=upload)

    def set_selectbox(
        self,
        key: str,
        value: Any,
        *,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Set a selectbox widget value and rerun the app."""
        self.app.selectbox(key=key).set_value(value)
        return self.run(upload=upload)

    def set_toggle(
        self,
        key: str,
        value: bool,
        *,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Set a toggle widget value and rerun the app."""
        self.app.toggle(key=key).set_value(value)
        return self.run(upload=upload)

    def set_checkbox(
        self,
        key: str,
        value: bool,
        *,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Set a checkbox widget value and rerun the app."""
        self.app.checkbox(key=key).set_value(value)
        return self.run(upload=upload)

    def set_text_area(
        self,
        key: str,
        value: str,
        *,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Set a text area value and rerun the app."""
        self.app.text_area(key=key).set_value(value)
        return self.run(upload=upload)

    def set_chat_input(
        self,
        value: str,
        *,
        index: int = 0,
        upload: UploadFixture | None | object = UNSET,
    ) -> AppTest:
        """Submit a chat input value and rerun the app."""
        self.app.chat_input[index].set_value(value)
        return self.run(upload=upload)

    def session_value(self, key: str, default: Any = None) -> Any:
        """Return a session-state value or a fallback when the key is absent."""
        try:
            return self.app.session_state[key]
        except KeyError:
            return default

    def has_session_key(self, key: str) -> bool:
        """Return True when the key exists in Streamlit session state."""
        return key in set(self.app.session_state._state._keys())

    def cache_key(
        self,
        upload: UploadFixture,
        *,
        vlm_model: str = DEFAULT_VLM_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> str:
        """Return the app's session cache key for a rendered note result."""
        return f"result_{ANALYSIS_CACHE_VERSION}_{upload.file_hash}_{vlm_model}_{llm_model}"

    def doc_cache_key(
        self,
        upload: UploadFixture,
        *,
        vlm_model: str = DEFAULT_VLM_MODEL,
    ) -> str:
        """Return the app's session cache key for a parsed document."""
        return f"doc_{ANALYSIS_CACHE_VERSION}_{upload.file_hash}_{vlm_model}"

    def seed_library(
        self,
        document: Document,
        note_row: dict[str, Any],
        *,
        has_vectors: bool = True,
    ) -> None:
        """Populate the fake SQLite/RAG stores with a saved library document."""
        self.documents_db[document.id] = copy.deepcopy(document)
        key = (document.id, note_row["vlm_model"], note_row["llm_model"])
        self.notes_db[key] = copy.deepcopy(note_row)
        if has_vectors:
            self.vector_documents.add(document.id)
        else:
            self.vector_documents.discard(document.id)


class _StopDemoImport(RuntimeError):
    """Sentinel exception used to stop ui/demo.py after helper definitions."""


class _RecordedExpander:
    """Small expander stub used by the source dedup unit test."""

    def __init__(self, recorder: "_StreamlitRecorder", label: str) -> None:
        self._recorder = recorder
        self._data = {"label": label, "captions": [], "texts": []}

    def __enter__(self) -> "_RecordedExpander":
        self._recorder.current_stack.append(self._data)
        self._recorder.expanders.append(self._data)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._recorder.current_stack.pop()
        return False


@dataclass
class _StreamlitRecorder:
    """Captures caption/text writes for direct helper-function unit tests."""

    captions: list[str]
    expanders: list[dict[str, Any]]
    current_stack: list[dict[str, Any]]

    def caption(self, text: str) -> None:
        """Record top-level or expander-local captions."""
        if self.current_stack:
            self.current_stack[-1]["captions"].append(text)
        else:
            self.captions.append(text)

    def expander(self, label: str, expanded: bool = False) -> _RecordedExpander:
        """Return a context manager that records nested writes."""
        del expanded
        return _RecordedExpander(self, label)

    def text(self, value: str) -> None:
        """Record expander text output."""
        self.current_stack[-1]["texts"].append(value)


def _pdf_upload(
    *,
    name: str = "sample.pdf",
    data: bytes = b"%PDF-1.4 fake pdf bytes",
    file_id: str = "upload-pdf",
) -> UploadFixture:
    """Return a deterministic PDF upload fixture."""
    return UploadFixture(
        file_id=file_id, name=name, mime_type="application/pdf", data=data
    )


def _image_upload(
    *,
    name: str = "sample.png",
    data: bytes = b"\x89PNG\r\n\x1a\nfake image bytes",
    file_id: str = "upload-image",
) -> UploadFixture:
    """Return a deterministic image upload fixture."""
    return UploadFixture(file_id=file_id, name=name, mime_type="image/png", data=data)


def _sample_document(
    *,
    doc_id: str = "doc-pdf",
    source: str = "sample.pdf",
    fmt: DocumentFormat = DocumentFormat.PDF,
    blocks: list[Block] | None = None,
    tags: list[str] | None = None,
) -> Document:
    """Build a parsed document fixture that matches the app's expectations."""
    default_blocks = [
        Block(
            type=BlockType.TEXT,
            content="첫 번째 블록 내용입니다.",
            order=0,
            metadata=BlockMetadata(page=1),
        ),
        Block(
            type=BlockType.CODE,
            content="print('hello world')",
            order=1,
            metadata=BlockMetadata(page=2, language="python"),
        ),
    ]
    return Document(
        id=doc_id,
        source=source,
        format=fmt,
        blocks=blocks if blocks is not None else default_blocks,
        metadata=DocumentMetadata(title=Path(source).stem, tags=tags or []),
        created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )


def _sample_document_row(
    *,
    doc_id: str = "doc-library",
    source: str = "library.pdf",
    fmt: DocumentFormat = DocumentFormat.PDF,
    blocks: list[Block] | None = None,
) -> Document:
    """Build a library document row like db.sqlite.get_document() returns."""
    return Document(
        id=doc_id,
        source=source,
        format=fmt,
        blocks=copy.deepcopy(blocks) if blocks is not None else _sample_document().blocks,
        metadata=DocumentMetadata(title=Path(source).stem),
        created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )


def _sample_note_result(
    *,
    title: str = "샘플 노트",
    summary: str = "핵심 내용을 요약한 노트입니다.",
    note_markdown: str | None = None,
) -> dict[str, Any]:
    """Return a generated-note payload used by most UI tests."""
    return {
        "title": title,
        "summary": summary,
        "note_markdown": note_markdown
        or "## 개요\n\n개요 내용\n\n## 핵심 개념\n\n핵심 내용",
        "key_concepts": ["개요", "핵심 개념"],
        "difficulty_level": "beginner",
        "estimated_read_time_min": 3,
        "errors": [],
    }


def _sample_note_row(
    *,
    document_id: str = "doc-library",
    file_hash: str = "library-hash",
    result: dict[str, Any] | None = None,
    vlm_model: str = DEFAULT_VLM_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    is_image: bool = False,
) -> dict[str, Any]:
    """Return a fake SQLite note row."""
    del document_id
    return {
        "result": copy.deepcopy(result or _sample_note_result(title="복원 노트")),
        "file_hash": file_hash,
        "vlm_model": vlm_model,
        "llm_model": llm_model,
        "is_image": is_image,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _sample_edit_result(
    *,
    edited_markdown: str | None = None,
    edited_section_body: str = "개요 수정",
    edited_section: str = "## 개요",
) -> NoteEditResult:
    """Return a successful section-edit response."""
    return NoteEditResult(
        edited_markdown=edited_markdown
        or "## 개요\n\n개요 수정\n\n## 핵심 개념\n\n핵심 내용",
        edited_section_body=edited_section_body,
        edited_section=edited_section,
        model=DEFAULT_LLM_MODEL,
        latency_ms=12.0,
        input_tokens=40,
        output_tokens=18,
        success=True,
    )


def _qa_result(
    answer: str, source_blocks: list[dict[str, Any]] | None = None
) -> SimpleNamespace:
    """Build a lightweight rag.query() response object."""
    return SimpleNamespace(answer=answer, source_blocks=source_blocks or [])


def _stored_document_copy(document: Document) -> Document:
    """Store a SQLite-like copy of a document with full block content preserved."""
    return copy.deepcopy(document)


def _session_messages(harness: DemoAppHarness, doc_id: str) -> list[dict[str, Any]]:
    """Return the Q&A chat history for a document."""
    return harness.session_value(f"chat_messages_{doc_id}", [])


def _assert_no_exception(app: AppTest) -> None:
    """Assert that the latest AppTest run completed without an exception."""
    assert not app.exception, f"Unexpected error: {app.exception}"


def _contains_markdown(app: AppTest, needle: str) -> bool:
    """Return True when any rendered markdown node contains the substring."""
    return any(needle in getattr(markdown, "value", "") for markdown in app.markdown)


def _analyze_upload(harness: DemoAppHarness, upload: UploadFixture) -> AppTest:
    """Drive the app through upload -> analyze for a single file."""
    harness.run()
    harness.run(upload=upload)
    harness.click_button(label="분석 시작")
    _assert_no_exception(harness.app)
    return harness.app


@pytest.fixture
def make_app() -> Iterator[Any]:
    """Create a fully mocked AppTest harness for ui/demo.py."""

    @contextmanager
    def _make_app() -> Iterator[DemoAppHarness]:
        documents_db: dict[str, Document] = {}
        notes_db: dict[tuple[str, str, str], dict[str, Any]] = {}
        vector_documents: set[str] = set()
        upload_holder: dict[str, UploadFixture | None] = {"value": None}
        original_init = local_script_runner.LocalScriptRunner.__init__
        rewrite_query = MagicMock(
            side_effect=lambda question: (f"rewritten::{question}", 0.0, 0, 0)
        )

        def _rag_query(question: str, **kwargs: Any) -> SimpleNamespace:
            retrieval_query = question
            if kwargs.get("rewrite"):
                try:
                    retrieval_query = rewrite_query(question)[0]
                except Exception:
                    retrieval_query = question
            return _qa_result(f"{retrieval_query}에 대한 답변")

        def _save_document(document: Document) -> None:
            documents_db[document.id] = _stored_document_copy(document)

        def _save_note(
            document_id: str,
            file_hash: str,
            result: dict[str, Any],
            vlm_model: str,
            llm_model: str,
            is_image: bool = False,
        ) -> None:
            notes_db[(document_id, vlm_model, llm_model)] = {
                "result": copy.deepcopy(result),
                "file_hash": file_hash,
                "vlm_model": vlm_model,
                "llm_model": llm_model,
                "is_image": is_image,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        def _get_document(document_id: str) -> Document | None:
            return copy.deepcopy(documents_db.get(document_id))

        def _get_note(
            document_id: str, vlm_model: str, llm_model: str
        ) -> dict[str, Any] | None:
            row = notes_db.get((document_id, vlm_model, llm_model))
            return copy.deepcopy(row) if row is not None else None

        def _list_documents(limit: int = 10) -> list[Document]:
            docs = sorted(
                documents_db.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            return [copy.deepcopy(doc) for doc in docs[:limit]]

        def _list_notes_for_document(document_id: str) -> list[dict[str, Any]]:
            rows = [
                copy.deepcopy(row)
                for (doc_id, _vlm, _llm), row in notes_db.items()
                if doc_id == document_id
            ]
            rows.sort(key=lambda row: row["updated_at"], reverse=True)
            return rows

        def _delete_document(document_id: str) -> None:
            documents_db.pop(document_id, None)
            for key in list(notes_db):
                if key[0] == document_id:
                    notes_db.pop(key)

        def _index_document(document: Document) -> None:
            vector_documents.add(document.id)

        def _delete_document_index(document_id: str) -> None:
            vector_documents.discard(document_id)

        def _has_document_vectors(document_id: str) -> bool:
            return document_id in vector_documents

        def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            current_upload = upload_holder["value"]
            if current_upload is not None:
                self._uploaded_file_mgr.add_file(
                    TEST_SESSION_ID,
                    current_upload.to_uploaded_file_rec(),
                )

        with ExitStack() as stack:
            preview_dir = stack.enter_context(tempfile.TemporaryDirectory())
            preview_env = os.environ.get("CATCHUP_IMAGE_PREVIEW_DIR", preview_dir)
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {"CATCHUP_IMAGE_PREVIEW_DIR": preview_env},
                )
            )
            stack.enter_context(
                patch.object(
                    local_script_runner.LocalScriptRunner, "__init__", _patched_init
                )
            )
            parse_pdf = stack.enter_context(
                patch(
                    "parsers.pdf_parser.parse_pdf",
                    new=MagicMock(side_effect=lambda _path: _sample_document()),
                )
            )
            parse_ipynb = stack.enter_context(
                patch(
                    "parsers.ipynb_parser.parse_ipynb",
                    new=MagicMock(
                        side_effect=lambda _path: _sample_document(
                            doc_id="doc-ipynb",
                            source="sample.ipynb",
                            fmt=DocumentFormat.IPYNB,
                        )
                    ),
                )
            )
            parse_image = stack.enter_context(
                patch(
                    "parsers.image_parser.parse_image",
                    new=MagicMock(
                        side_effect=lambda _path, model=DEFAULT_VLM_MODEL, language="ko": (
                            _sample_document(
                                doc_id="doc-image",
                                source="sample.png",
                                fmt=DocumentFormat.IMAGE,
                                blocks=[
                                    Block(
                                        type=BlockType.FIGURE,
                                        content="슬라이드 이미지 설명",
                                        order=0,
                                        metadata=BlockMetadata(confidence=0.95),
                                    )
                                ],
                            )
                        )
                    ),
                )
            )
            generate_note = stack.enter_context(
                patch(
                    "llm.note_generator.generate_note_sectioned",
                    new=MagicMock(
                        side_effect=lambda doc, model=DEFAULT_LLM_MODEL, language="ko": (
                            _sample_note_result()
                        )
                    ),
                )
            )
            edit_section = stack.enter_context(
                patch(
                    "llm.note_editor.edit_section",
                    new=MagicMock(side_effect=lambda **kwargs: _sample_edit_result()),
                )
            )
            save_document = stack.enter_context(
                patch(
                    "db.sqlite.save_document", new=MagicMock(side_effect=_save_document)
                )
            )
            save_note = stack.enter_context(
                patch("db.sqlite.save_note", new=MagicMock(side_effect=_save_note))
            )
            get_document = stack.enter_context(
                patch(
                    "db.sqlite.get_document", new=MagicMock(side_effect=_get_document)
                )
            )
            get_note = stack.enter_context(
                patch("db.sqlite.get_note", new=MagicMock(side_effect=_get_note))
            )
            list_documents = stack.enter_context(
                patch(
                    "db.sqlite.list_documents",
                    new=MagicMock(side_effect=_list_documents),
                )
            )
            list_notes_for_document = stack.enter_context(
                patch(
                    "db.sqlite.list_notes_for_document",
                    new=MagicMock(side_effect=_list_notes_for_document),
                )
            )
            delete_document = stack.enter_context(
                patch(
                    "db.sqlite.delete_document",
                    new=MagicMock(side_effect=_delete_document),
                )
            )
            delete_document_index = stack.enter_context(
                patch(
                    "rag.delete_document_index",
                    new=MagicMock(side_effect=_delete_document_index),
                )
            )
            index_document = stack.enter_context(
                patch("rag.index_document", new=MagicMock(side_effect=_index_document))
            )
            has_document_vectors = stack.enter_context(
                patch(
                    "rag.has_document_vectors",
                    new=MagicMock(side_effect=_has_document_vectors),
                )
            )
            rag_query = stack.enter_context(
                patch(
                    "rag.query",
                    new=MagicMock(side_effect=_rag_query),
                )
            )
            enrich_pdf_figures = stack.enter_context(
                patch(
                    "parsers.figure_enricher.enrich_pdf_figures",
                    new=MagicMock(side_effect=lambda doc, **kwargs: doc),
                )
            )
            pyperclip_copy = stack.enter_context(
                patch("pyperclip.copy", new=MagicMock())
            )
            link_concepts = stack.enter_context(
                patch(
                    "llm.concept_linker.link_concepts",
                    new=MagicMock(return_value=[]),
                )
            )
            stack.enter_context(
                patch(
                    "llm.concept_linker.delete_document_concepts",
                    new=MagicMock(return_value=None),
                )
            )
            stack.enter_context(
                patch(
                    "db.sqlite.get_concept_links_for_document",
                    new=MagicMock(return_value=[]),
                )
            )

            harness = DemoAppHarness(
                app=AppTest.from_file("ui/demo.py"),
                parse_pdf=parse_pdf,
                parse_ipynb=parse_ipynb,
                parse_image=parse_image,
                generate_note=generate_note,
                edit_section=edit_section,
                save_document=save_document,
                save_note=save_note,
                get_document=get_document,
                get_note=get_note,
                list_documents=list_documents,
                list_notes_for_document=list_notes_for_document,
                delete_document=delete_document,
                delete_document_index=delete_document_index,
                index_document=index_document,
                has_document_vectors=has_document_vectors,
                rag_query=rag_query,
                rewrite_query=rewrite_query,
                enrich_pdf_figures=enrich_pdf_figures,
                link_concepts=link_concepts,
                pyperclip_copy=pyperclip_copy,
                documents_db=documents_db,
                notes_db=notes_db,
                vector_documents=vector_documents,
                _current_upload=upload_holder,
            )
            yield harness

    yield _make_app


def test_render_source_block_expanders_deduplicates_by_source_and_location() -> None:
    """Duplicate source/page pairs should render only one expander."""
    recorder = _StreamlitRecorder(captions=[], expanders=[], current_stack=[])
    spec = importlib.util.spec_from_file_location("ui_demo_unit_test", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None, (
        "ui/demo.py should be importable for testing"
    )
    module = importlib.util.module_from_spec(spec)

    with (
        patch.object(st, "set_page_config", side_effect=_StopDemoImport),
        patch.object(st, "caption", side_effect=recorder.caption),
        patch.object(st, "expander", side_effect=recorder.expander),
        patch.object(st, "text", side_effect=recorder.text),
    ):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass

        module._render_source_block_expanders(
            [
                {
                    "source": "guide.pdf",
                    "page": 1,
                    "block_type": "text",
                    "block_order": 0,
                    "content_preview": "첫 번째 미리보기",
                },
                {
                    "source": "guide.pdf",
                    "page": 1,
                    "block_type": "text",
                    "block_order": 1,
                    "content_preview": "중복 미리보기",
                },
                {
                    "source": "guide.pdf",
                    "page": 2,
                    "block_type": "code",
                    "block_order": 2,
                    "content_preview": "다른 위치 미리보기",
                },
            ]
        )

    assert recorder.captions == ["참조 블록"], (
        "Source block caption should render exactly once"
    )
    assert len(recorder.expanders) == 2, (
        "Duplicate source/page pairs should collapse into one expander"
    )
    assert recorder.expanders[0]["captions"] == ["block_order: 0"], (
        "First unique block should keep its metadata"
    )
    assert recorder.expanders[1]["texts"] == ["다른 위치 미리보기"], (
        "Second unique source should still render its preview"
    )


def test_filter_inline_figure_blocks_skips_front_matter_images() -> None:
    """Inline note rendering should ignore image_path blocks before the first body page."""
    spec = importlib.util.spec_from_file_location("ui_demo_unit_test", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None, (
        "ui/demo.py should be importable for testing"
    )
    module = importlib.util.module_from_spec(spec)

    with patch.object(st, "set_page_config", side_effect=_StopDemoImport):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass

    front_text = Block(
        type=BlockType.TEXT,
        content="예제 코드 제공",
        order=0,
        image_path="front.png",
        metadata=BlockMetadata(page=1),
    )
    body_figure = Block(
        type=BlockType.FIGURE,
        content="본문 다이어그램",
        order=1,
        image_path="body.png",
        metadata=BlockMetadata(page=17),
    )
    doc = _sample_document(blocks=[front_text, body_figure])
    grouped_sections = [
        SectionInfo(
            heading="1. 본문",
            level=1,
            start_block_order=1,
            end_block_order=None,
            blocks=[
                Block(
                    type=BlockType.TEXT,
                    content="본문 설명이 충분히 이어집니다.",
                    order=10,
                    metadata=BlockMetadata(page=17),
                )
            ],
            from_toc=True,
        )
    ]

    with (
        patch("llm.section_splitter.extract_sections", return_value=grouped_sections),
        patch("llm.section_splitter.group_blocks_by_section", return_value=grouped_sections),
    ):
        filtered = module._filter_inline_figure_blocks(doc, [front_text, body_figure])

    assert [block.order for block in filtered] == [1], (
        "Only body-page images should remain eligible for inline note rendering"
    )


class TestUploadFlow:
    """Upload and parsing scenarios for the main pipeline."""

    def test_running_without_upload_is_safe(self, make_app: Any) -> None:
        """The landing page should render without a file and without crashing."""
        with make_app() as harness:
            app = harness.run()
            _assert_no_exception(app)

    def test_parse_failure_skips_note_generation(self, make_app: Any) -> None:
        """Parse failures should stop before note generation and persistence."""
        failed_doc = _sample_document(
            blocks=[],
            tags=["parse_failed"],
        )
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.parse_pdf.side_effect = lambda _path: failed_doc
            harness.run()
            harness.run(upload=pdf_upload)
            app = harness.click_button(label="분석 시작")

            _assert_no_exception(app)
            assert harness.generate_note.call_count == 0, (
                "generate_note should not run when parsing fails"
            )
            assert harness.save_document.call_count == 0, (
                "save_document should not run for parse-failed documents"
            )
            assert harness.save_note.call_count == 0, (
                "save_note should not run for parse-failed documents"
            )

    def test_successful_pdf_upload_flow_has_no_exceptions(self, make_app: Any) -> None:
        """PDF upload -> parse -> note generation should finish without runtime errors."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            app = _analyze_upload(harness, pdf_upload)
            cache_key = harness.cache_key(pdf_upload)

            _assert_no_exception(app)
            assert harness.parse_pdf.call_count == 1, (
                "PDF parsing should run exactly once for the first analysis"
            )
            assert harness.generate_note.call_count == 1, (
                "Note generation should run for non-image uploads"
            )
            assert harness.save_document.call_count == 1, (
                "Analyzed documents should be persisted"
            )
            assert harness.save_note.call_count == 1, (
                "Generated notes should be persisted"
            )
            assert harness.has_session_key(cache_key), (
                "Successful analysis should populate the session cache"
            )
            assert len(app.chat_input) == 1, (
                "A successful PDF analysis should expose the Q&A chat input"
            )

    def test_same_file_upload_uses_session_cache(self, make_app: Any) -> None:
        """Re-uploading the same file in one session should not call the APIs again."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.run(upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.parse_pdf.call_count == 1, (
                "Cached uploads should skip parse_pdf on rerun"
            )
            assert harness.generate_note.call_count == 1, (
                "Cached uploads should skip generate_note on rerun"
            )

    def test_sectioned_note_renders_all_sections(self, make_app: Any) -> None:
        """Multi-section note_markdown from generate_note_sectioned should render all sections in the UI."""
        pdf_upload = _pdf_upload()
        sectioned_result = _sample_note_result(
            note_markdown=(
                "## 1. 소개\n\n소개 내용입니다.\n\n"
                "## 2. 본론\n\n본론 내용입니다.\n\n"
                "## 3. 결론\n\n결론 내용입니다."
            ),
        )

        with make_app() as harness:
            harness.generate_note.side_effect = (
                lambda doc, model=DEFAULT_LLM_MODEL, language="ko": sectioned_result
            )
            app = _analyze_upload(harness, pdf_upload)

            _assert_no_exception(app)
            assert _contains_markdown(app, "소개 내용"), (
                "Section 1 content should be rendered"
            )
            assert _contains_markdown(app, "본론 내용"), (
                "Section 2 content should be rendered"
            )
            assert _contains_markdown(app, "결론 내용"), (
                "Section 3 content should be rendered"
            )

    def test_generate_note_sectioned_called_once_on_pdf_upload(
        self, make_app: Any
    ) -> None:
        """generate_note_sectioned must be called exactly once when a PDF is analysed."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            assert harness.generate_note.call_count == 1, (
                "generate_note_sectioned should be called exactly once per analysis"
            )
            _, kwargs = harness.generate_note.call_args
            assert "model" in kwargs or harness.generate_note.call_args.args, (
                "generate_note_sectioned should receive a model argument"
            )

    def test_llm_model_hint_caption_shown_for_known_model(
        self, make_app: Any
    ) -> None:
        """A hint caption should appear below the LLM model selector for known models."""
        with make_app() as harness:
            app = harness.run()
            _assert_no_exception(app)
            # The default model (index 0, gpt-4o-mini) has a known hint
            caption_values = [c.value for c in app.caption]
            assert any("균형" in v or "권장" in v or "저렴" in v for v in caption_values), (
                "A model hint caption should be visible for the selected LLM model"
            )

    def test_note_generation_failure_shows_error(self, make_app: Any) -> None:
        """generate_note_sectioned exceptions should surface an error in the UI."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.generate_note.side_effect = RuntimeError("LLM API quota exceeded")
            harness.run()
            harness.run(upload=pdf_upload)
            app = harness.click_button(label="분석 시작")

            _assert_no_exception(app)
            assert any("노트 생성 실패" in err.value for err in app.error), (
                "Note generation failures should show a user-facing error message"
            )
            assert harness.save_document.call_count == 0, (
                "Failed note generation should not persist documents"
            )


class TestTabState:
    """State-preservation checks across panels and toggles."""

    def test_tab_switch_preserves_qna_state_and_edit_toggle_keeps_panel(
        self,
        make_app: Any,
    ) -> None:
        """Q&A history should survive panel switches, and edit mode should not reset the right panel."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("Q&A 메시지", upload=pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_selectbox(
                "editor_doc-pdf_edit_section_selectbox", 1, upload=pdf_upload
            )
            harness.set_radio("active_right_panel", "💬 Q&A", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert len(_session_messages(harness, "doc-pdf")) == 2, (
                "Q&A history should survive a round-trip to the note editor"
            )

            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_toggle("note_edit_toggle", True, upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.session_value("active_right_panel") == "✏️ 노트 수정", (
                "Toggling left-panel edit mode should not reset the active right panel"
            )

    def test_qna_and_note_editor_models_are_independent(self, make_app: Any) -> None:
        """The Q&A model selectbox and note-editor model selectbox should not overwrite each other."""
        pdf_upload = _pdf_upload()
        qa_model = SUPPORTED_LLM_MODELS[1]
        editor_model = SUPPORTED_LLM_MODELS[5]

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_selectbox("qa_model_select", qa_model, upload=pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_selectbox(
                "editor_doc-pdf_note_editor_model_select",
                editor_model,
                upload=pdf_upload,
            )
            harness.set_radio("active_right_panel", "💬 Q&A", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.app.selectbox(key="qa_model_select").value == qa_model, (
                "Q&A model selection should persist after visiting the note editor"
            )
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            assert (
                harness.app.selectbox(
                    key="editor_doc-pdf_note_editor_model_select"
                ).value
                == editor_model
            ), "Note editor model selection should stay independent from the Q&A model"


class TestRagChat:
    """Runtime checks for the Q&A panel."""

    def test_emotional_query_returns_empathy_response(self, make_app: Any) -> None:
        """Emotional questions should surface a non-empty empathetic answer."""
        pdf_upload = _pdf_upload()

        def _rag_side_effect(
            question: str,
            **kwargs: Any,
        ) -> SimpleNamespace:
            del kwargs
            if question == "너무 어렵다":
                return _qa_result("괜찮아요. 어려운 부분부터 하나씩 같이 정리해볼게요.")
            return _qa_result("기본 응답")

        with make_app() as harness:
            harness.rag_query.side_effect = _rag_side_effect
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("너무 어렵다", upload=pdf_upload)

            _assert_no_exception(harness.app)
            messages = _session_messages(harness, "doc-pdf")
            assert messages[-1]["content"], (
                "Emotional queries should still produce a non-empty assistant message"
            )
            assert "같이 정리" in messages[-1]["content"], (
                "The assistant should return the empathetic fallback response"
            )

    def test_followup_pill_renders_and_query_receives_document_filter(
        self,
        make_app: Any,
    ) -> None:
        """RAG responses should expose follow-up pills and pass document_id into query()."""
        pdf_upload = _pdf_upload()
        source_blocks = [
            {
                "document_id": "doc-pdf",
                "source": "sample.pdf",
                "page": 1,
                "block_order": 0,
                "block_type": "text",
                "content_preview": "참조 블록 미리보기",
            }
        ]
        answer = (
            "답변 본문\n---SUGGESTIONS---\n1. 후속 질문 A\n2. 후속 질문 B\n---END---"
        )

        with make_app() as harness:
            harness.rag_query.side_effect = lambda question, **kwargs: _qa_result(
                answer,
                source_blocks,
            )
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("첫 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rag_query.call_args.kwargs["document_id"] == "doc-pdf", (
                "rag.query should always be filtered to the active document"
            )
            assert any(
                button.label == "후속 질문 A" for button in harness.app.button
            ), "The first follow-up suggestion pill should render after a RAG answer"


class TestQueryRewriting:
    """Coverage for the CU-13 query rewriting toggle and fallback flow."""

    def test_query_rewriting_checkbox_renders_with_default_off(
        self,
        make_app: Any,
    ) -> None:
        """The Q&A panel should render the Query Rewriting checkbox as opt-in."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            checkbox = harness.app.checkbox(key="qa_use_rewrite")
            assert checkbox.label == "Query Rewriting", (
                "The Q&A panel should expose the Query Rewriting checkbox"
            )
            assert checkbox.value is False, "Query Rewriting should default to OFF"

    def test_rewrite_is_not_called_when_checkbox_is_off(self, make_app: Any) -> None:
        """The rewrite path should stay idle while the toggle remains off."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("오프 상태 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 0, (
                "rewrite_query should not run while Query Rewriting is OFF"
            )
            assert harness.rag_query.call_args.kwargs["rewrite"] is False, (
                "rag.query should receive rewrite=False when the toggle is OFF"
            )

    def test_rewrite_is_called_once_when_checkbox_is_on(self, make_app: Any) -> None:
        """Enabling the checkbox should route the next question through rewrite_query exactly once."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_use_rewrite", True, upload=pdf_upload)
            harness.set_chat_input("온 상태 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, (
                "rewrite_query should run exactly once for a rewritten question"
            )
            assert harness.rag_query.call_args.kwargs["rewrite"] is True, (
                "rag.query should receive rewrite=True when the toggle is ON"
            )

    def test_toggling_query_rewriting_on_then_off_stays_stable(
        self,
        make_app: Any,
    ) -> None:
        """Turning rewrite on and back off should keep the Q&A flow stable across reruns."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_use_rewrite", True, upload=pdf_upload)
            harness.set_chat_input("첫 질문", upload=pdf_upload)
            harness.set_checkbox("qa_use_rewrite", False, upload=pdf_upload)
            harness.set_chat_input("두 번째 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, (
                "rewrite_query should stop being called after the toggle is turned back OFF"
            )
            assert harness.rag_query.call_args.kwargs["rewrite"] is False, (
                "The second question should pass rewrite=False after toggling OFF"
            )

    def test_rewrite_failure_falls_back_to_original_query(self, make_app: Any) -> None:
        """Rewrite failures should not crash Q&A and should answer from the original query text."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.rewrite_query.side_effect = RuntimeError("rewrite failed")
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_use_rewrite", True, upload=pdf_upload)
            harness.set_chat_input("원본 쿼리", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, (
                "rewrite_query should still be attempted once when rewriting is enabled"
            )
            assert harness.rag_query.call_args.kwargs["rewrite"] is True, (
                "rag.query should still be called with rewrite=True when rewrite fails"
            )
            assert (
                "원본 쿼리" in _session_messages(harness, "doc-pdf")[-1]["content"]
            ), (
                "Rewrite failures should fall back to the original query instead of crashing"
            )


class TestImagePipeline:
    """Image-specific flow coverage added in CU-10."""

    def test_image_upload_skips_note_generation_and_persists_workspace(
        self,
        make_app: Any,
    ) -> None:
        """Image uploads should bypass generate_note and keep the image workspace on rerun."""
        image_upload = _image_upload()

        with make_app() as harness:
            _analyze_upload(harness, image_upload)
            cache_key = harness.cache_key(image_upload)

            _assert_no_exception(harness.app)
            assert harness.parse_image.call_count == 1, (
                "Image uploads should route through parse_image"
            )
            assert harness.generate_note.call_count == 0, (
                "Image uploads should bypass note generation"
            )
            assert harness.session_value(f"is_image_{cache_key}") is True, (
                "Image analyses should set the is_image session flag"
            )
            assert len(harness.app.radio) == 0, (
                "Image mode should not render the PDF-only right-panel radio"
            )

            harness.run(upload=image_upload)

            _assert_no_exception(harness.app)
            assert harness.parse_image.call_count == 1, (
                "Rerunning the same image should reuse the cached analysis"
            )
            assert harness.session_value(f"is_image_{cache_key}") is True, (
                "The image workspace should survive reruns while cached"
            )

    def test_image_upload_persists_original_preview_file(
        self,
        make_app: Any,
        tmp_path: Path,
    ) -> None:
        """Image uploads should persist original bytes for later library-mode preview restore."""
        image_upload = _image_upload()
        preview_dir = tmp_path / "image-previews"

        with patch.dict(os.environ, {"CATCHUP_IMAGE_PREVIEW_DIR": str(preview_dir)}):
            with make_app() as harness:
                _analyze_upload(harness, image_upload)

        persisted_path = preview_dir / f"{image_upload.file_hash}.png"
        assert persisted_path.exists(), (
            "Analyzed image uploads should persist the original file for future preview restore"
        )
        assert persisted_path.read_bytes() == image_upload.data, (
            "Persisted preview bytes should match the original uploaded image"
        )

    def test_image_api_key_error_surfaces_message(self, make_app: Any) -> None:
        """Image parser auth failures should show a user-facing API key error."""
        image_upload = _image_upload(name="slide.webp", file_id="upload-webp")

        with make_app() as harness:
            harness.parse_image.side_effect = RuntimeError(
                "API key missing for image parse"
            )
            harness.run()
            harness.run(upload=image_upload)
            harness.click_button(label="분석 시작")

            _assert_no_exception(harness.app)
            assert any("API 키" in error.value for error in harness.app.error), (
                "API-key failures should surface the dedicated st.error message"
            )
            assert harness.generate_note.call_count == 0, (
                "generate_note should never run when image parsing fails"
            )

    def test_pdf_upload_still_reaches_qna_after_image_branch_changes(
        self,
        make_app: Any,
    ) -> None:
        """The PDF pipeline should keep the original parse -> note -> Q&A flow."""
        pdf_upload = _pdf_upload(name="regression.pdf", file_id="upload-pdf-regression")

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.generate_note.call_count == 1, (
                "PDF uploads should still generate a note after the image branch changes"
            )
            assert harness.app.radio(key="active_right_panel").value == "💬 Q&A", (
                "Successful PDF analysis should still land on the Q&A panel"
            )
            assert len(harness.app.chat_input) == 1, (
                "PDF analyses should still expose the Q&A chat input"
            )


class TestNoteEditor:
    """Coverage for the note-editor chatbot and direct-edit flows."""

    def test_edit_chat_apply_updates_only_target_section(self, make_app: Any) -> None:
        """Applying a section edit should update only the targeted markdown section."""
        pdf_upload = _pdf_upload()
        edited_markdown = "## 개요\n\n개요 수정\n\n## 핵심 개념\n\n핵심 내용"

        with make_app() as harness:
            harness.edit_section.return_value = _sample_edit_result(
                edited_markdown=edited_markdown,
                edited_section_body="개요 수정",
            )
            _analyze_upload(harness, pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_chat_input("개요 섹션을 수정해줘", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.edit_section.call_count == 1, (
                "Submitting an edit prompt should call edit_section"
            )
            assert harness.edit_section.call_args.kwargs["document_id"] == "doc-pdf", (
                "Section-edit RAG grounding should be scoped to the current document"
            )

            harness.click_button(label="✅ 적용", upload=pdf_upload)
            cache_key = harness.cache_key(pdf_upload)
            updated_result = harness.session_value(cache_key)

            _assert_no_exception(harness.app)
            assert updated_result["note_markdown"] == edited_markdown, (
                "Apply should replace the pending note markdown"
            )
            assert "핵심 내용" in updated_result["note_markdown"], (
                "Applying one section edit must preserve other sections"
            )
            assert not harness.has_session_key(
                "editor_doc-pdf_edit_pending_markdown"
            ), "Pending preview markdown should be cleared after Apply"
            assert harness.save_note.call_count >= 2, (
                "Applying a note edit should persist the dirty note back to SQLite"
            )

    def test_cancel_clears_pending_preview(self, make_app: Any) -> None:
        """Cancel should drop the preview state without mutating the saved markdown."""
        pdf_upload = _pdf_upload()
        original_markdown = _sample_note_result()["note_markdown"]

        with make_app() as harness:
            harness.edit_section.return_value = _sample_edit_result()
            _analyze_upload(harness, pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_chat_input("미리보기만 만들어줘", upload=pdf_upload)
            harness.click_button(label="❌ 취소", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert not harness.has_session_key(
                "editor_doc-pdf_edit_pending_markdown"
            ), "Cancel should clear edit_pending_markdown"
            assert (
                harness.session_value(harness.cache_key(pdf_upload))["note_markdown"]
                == original_markdown
            ), "Cancel must leave the saved note markdown untouched"

    def test_empty_edit_input_is_safe(self, make_app: Any) -> None:
        """An empty note-editor chat input should be ignored without crashing."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.app.chat_input[0].set_value("")
            harness.run(upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.edit_section.call_count == 0, (
                "Empty edit prompts should not call edit_section"
            )

    def test_undo_restores_previous_markdown(self, make_app: Any) -> None:
        """Undo should restore the last applied section body."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.edit_section.return_value = _sample_edit_result()
            _analyze_upload(harness, pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_chat_input("개요를 수정해줘", upload=pdf_upload)
            harness.click_button(label="✅ 적용", upload=pdf_upload)
            harness.set_toggle("note_edit_toggle", True, upload=pdf_upload)
            harness.click_button(key="undo_sec_0", upload=pdf_upload)

            _assert_no_exception(harness.app)
            restored_markdown = harness.session_value(harness.cache_key(pdf_upload))[
                "note_markdown"
            ]
            assert "개요 내용" in restored_markdown, (
                "Undo should restore the previous section body"
            )
            assert "개요 수정" not in restored_markdown, (
                "Undo should remove the applied edited body"
            )


class TestLibraryPersistence:
    """Coverage for saved-document restoration and session persistence."""

    def test_analysis_saves_and_sidebar_lists_document(self, make_app: Any) -> None:
        """Completed analyses should persist and show up in the sidebar library on next rerun."""
        pdf_upload = _pdf_upload(name="library.pdf", file_id="upload-library")
        parsed_doc = _sample_document(doc_id="doc-library", source="library.pdf")

        with make_app() as harness:
            harness.parse_pdf.side_effect = lambda _path: copy.deepcopy(parsed_doc)
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.save_document.call_count == 1, (
                "Finished analyses should call save_document"
            )
            assert harness.save_note.call_count == 1, (
                "Finished analyses should call save_note once"
            )

            # The sidebar re-renders with the new document on the next user interaction.
            # (The old code called st.rerun() here which caused the note to not show immediately;
            # sidebar library updates on the following rerun instead.)
            harness.run(upload=pdf_upload)
            _assert_no_exception(harness.app)
            assert harness.app.button(key="lib_doc-library").label.startswith("📄"), (
                "Saved documents should render in the sidebar library after next rerun"
            )

    def test_library_load_restores_without_reanalysis_and_survives_stale_uploader(
        self,
        make_app: Any,
    ) -> None:
        """Library restore should hydrate note/Q&A state without rerunning the pipeline, even with a stale uploader value."""
        upload = _pdf_upload(name="library.pdf", file_id="upload-library-stale")
        document = _sample_document_row(doc_id="doc-library", source="library.pdf")
        note_row = _sample_note_row(
            document_id=document.id,
            file_hash=upload.file_hash,
            result=_sample_note_result(title="복원 노트"),
        )

        with make_app() as harness:
            harness.seed_library(document, note_row, has_vectors=True)
            harness.run()
            harness.click_button(key="lib_doc-library", upload=None)

            _assert_no_exception(harness.app)
            assert harness.parse_pdf.call_count == 0, (
                "Library restores should not rerun parse_pdf"
            )
            assert harness.generate_note.call_count == 0, (
                "Library restores should not rerun generate_note"
            )
            assert harness.session_value("_library_mode") is True, (
                "Loading from the library should enable library mode"
            )
            assert harness.session_value("_library_doc_id") == document.id, (
                "The active library document ID should be tracked in session state"
            )

            harness.run(upload=upload)

            _assert_no_exception(harness.app)
            assert harness.session_value("_library_mode") is True, (
                "A stale uploader value for the same file must not cancel library mode"
            )
            assert harness.parse_pdf.call_count == 0, (
                "A stale uploader value should not restart the pipeline"
            )

    def test_library_load_renders_inline_figures_when_blocks_are_restored(
        self,
        make_app: Any,
        tmp_path: Path,
    ) -> None:
        """Library-loaded notes should restore figure-bearing blocks for inline rendering."""
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0x8AAAAASUVORK5CYII="
        )
        figure_path = tmp_path / "library-body.png"
        figure_path.write_bytes(png_bytes)

        document = _sample_document_row(
            doc_id="doc-library-fig",
            source="library-fig.pdf",
            blocks=[
                Block(
                    type=BlockType.TEXT,
                    content="본문 설명 블록입니다.",
                    order=0,
                    image_path=str(figure_path),
                    metadata=BlockMetadata(page=17),
                ),
                Block(
                    type=BlockType.TEXT,
                    content="충분히 긴 본문 설명이 이어집니다.",
                    order=1,
                    metadata=BlockMetadata(page=17),
                ),
            ],
        )
        note_row = _sample_note_row(
            document_id=document.id,
            file_hash="library-fig-hash",
            result=_sample_note_result(
                title="이미지 복원 노트",
                note_markdown="## 서론\n\n본문 설명입니다.\n\n## 본문\n\n추가 설명입니다.",
            ),
        )

        with make_app() as harness:
            harness.seed_library(document, note_row, has_vectors=True)
            harness.run()
            harness.click_button(key="lib_doc-library-fig", upload=None)

            _assert_no_exception(harness.app)
            doc_cache_key = harness.session_value("_library_doc_cache_key")
            restored_doc = harness.session_value(doc_cache_key)
            assert restored_doc is not None
            assert len(restored_doc.blocks) == 2, (
                "Library-loaded documents should restore full block content"
            )
            assert restored_doc.blocks[0].image_path == str(figure_path), (
                "Restored blocks should keep image_path so note rendering can inject figures"
            )

    def test_library_image_load_restores_original_preview(
        self,
        make_app: Any,
        tmp_path: Path,
    ) -> None:
        """Library-loaded image docs should render the persisted original preview instead of the fallback warning."""
        image_upload = _image_upload(name="library-image.png", file_id="upload-library-image")
        preview_dir = tmp_path / "image-previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        persisted_path = preview_dir / f"{image_upload.file_hash}.png"
        persisted_path.write_bytes(image_upload.data)

        document = _sample_document_row(
            doc_id="doc-library-image",
            source="library-image.png",
            fmt=DocumentFormat.IMAGE,
            blocks=[
                Block(
                    type=BlockType.FIGURE,
                    content="이미지 설명",
                    order=0,
                    metadata=BlockMetadata(page=1),
                )
            ],
        )
        note_row = _sample_note_row(
            document_id=document.id,
            file_hash=image_upload.file_hash,
            result={},
            is_image=True,
        )

        with patch.dict(os.environ, {"CATCHUP_IMAGE_PREVIEW_DIR": str(preview_dir)}):
            with make_app() as harness:
                harness.seed_library(document, note_row, has_vectors=True)
                harness.run()
                harness.click_button(key="lib_doc-library-image", upload=None)

                _assert_no_exception(harness.app)
                assert not _contains_markdown(harness.app, "이미지 미리보기를 복원하지 못했습니다"), (
                    "Library image restore should not show the missing-preview warning when persisted bytes exist"
                )
                assert _contains_markdown(harness.app, "원본 미리보기"), (
                    "Library image restore should render the original image preview card"
                )

    def test_delete_evicts_cache_then_reupload_reprocesses(self, make_app: Any) -> None:
        """Deleting a library document should clear caches and force re-analysis on re-upload."""
        upload = _pdf_upload(name="delete-me.pdf", file_id="upload-delete")
        parsed_doc = _sample_document(doc_id="doc-delete", source="delete-me.pdf")
        deletion_order: list[str] = []

        with make_app() as harness:
            harness.parse_pdf.side_effect = lambda _path: copy.deepcopy(parsed_doc)

            original_delete_index = harness.delete_document_index.side_effect
            original_delete_document = harness.delete_document.side_effect

            def _record_delete_index(document_id: str) -> None:
                deletion_order.append("delete_document_index")
                original_delete_index(document_id)

            def _record_delete_document(document_id: str) -> None:
                deletion_order.append("delete_document")
                original_delete_document(document_id)

            harness.delete_document_index.side_effect = _record_delete_index
            harness.delete_document.side_effect = _record_delete_document

            _analyze_upload(harness, upload)
            cache_key = harness.cache_key(upload)
            harness.run(upload=None)
            harness.click_button(key="lib_del_doc-delete", upload=None)

            _assert_no_exception(harness.app)
            assert deletion_order == ["delete_document_index", "delete_document"], (
                "Sidebar deletion should remove vector index before deleting SQLite rows"
            )
            assert not harness.has_session_key(cache_key), (
                "Deleting a document should evict its cached note result from session_state"
            )

            harness.run(upload=upload)
            harness.click_button(label="분석 시작", upload=upload)

            _assert_no_exception(harness.app)
            assert harness.parse_pdf.call_count == 2, (
                "Re-uploading a deleted document should rerun parse_pdf instead of hitting the old cache"
            )

    def test_library_chats_are_isolated_between_documents(self, make_app: Any) -> None:
        """Loading document B after chatting on document A should start with an empty Q&A history."""
        doc_a = _sample_document_row(doc_id="doc-a", source="alpha.pdf")
        doc_b = _sample_document_row(doc_id="doc-b", source="beta.pdf")
        note_a = _sample_note_row(
            document_id=doc_a.id,
            file_hash="hash-a",
            result=_sample_note_result(title="알파 노트"),
        )
        note_b = _sample_note_row(
            document_id=doc_b.id,
            file_hash="hash-b",
            result=_sample_note_result(title="베타 노트"),
        )

        with make_app() as harness:
            harness.seed_library(doc_a, note_a, has_vectors=True)
            harness.seed_library(doc_b, note_b, has_vectors=True)
            harness.run()
            harness.click_button(key="lib_doc-a", upload=None)
            harness.set_chat_input("문서 A 질문", upload=None)

            _assert_no_exception(harness.app)
            assert len(_session_messages(harness, "doc-a")) == 2, (
                "Document A should record its Q&A chat history"
            )

            harness.click_button(key="lib_doc-b", upload=None)

            _assert_no_exception(harness.app)
            assert _session_messages(harness, "doc-b") == [], (
                "Document B should start with an empty chat history"
            )
            assert len(_session_messages(harness, "doc-a")) == 2, (
                "Document A chat history should remain isolated under its own session_state key"
            )

    def test_qna_is_disabled_when_vectors_are_missing(self, make_app: Any) -> None:
        """Library documents without vectors should not render an active Q&A chat input."""
        document = _sample_document_row(doc_id="doc-no-vectors", source="novectors.pdf")
        note_row = _sample_note_row(
            document_id=document.id, file_hash="hash-no-vectors"
        )

        with make_app() as harness:
            harness.seed_library(document, note_row, has_vectors=False)
            harness.run()
            harness.click_button(key="lib_doc-no-vectors", upload=None)

            _assert_no_exception(harness.app)
            assert len(harness.app.chat_input) == 0, (
                "Q&A should be disabled when no document vectors are available"
            )
            assert any(
                "벡터" in el.value for el in harness.app.markdown
            ), "The UI should explain why Q&A is unavailable without vectors"

    def test_library_note_edit_persists_dirty_note(self, make_app: Any) -> None:
        """Editing a library-loaded note should save with the restored file hash and models."""
        document = _sample_document_row(
            doc_id="doc-library-edit", source="editable.pdf"
        )
        note_row = _sample_note_row(
            document_id=document.id,
            file_hash="editable-hash",
            result=_sample_note_result(title="편집 전 노트"),
        )
        updated_markdown = "## 개요\n\n직접 편집된 내용\n\n## 핵심 개념\n\n핵심 내용"

        with make_app() as harness:
            harness.seed_library(document, note_row, has_vectors=True)
            harness.run()
            harness.click_button(key="lib_doc-library-edit", upload=None)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=None)
            harness.set_radio(
                "editor_doc-library-edit_note_editor_method",
                "⌨️ 직접 편집",
                upload=None,
            )
            harness.set_text_area(
                "editor_doc-library-edit_direct_edit_textarea",
                updated_markdown,
                upload=None,
            )
            harness.click_button(
                key="editor_doc-library-edit_direct_edit_save", upload=None
            )

            _assert_no_exception(harness.app)
            last_args = harness.save_note.call_args_list[-1].args
            assert last_args[0] == document.id, (
                "Dirty library edits should save back to the loaded document"
            )
            assert last_args[1] == note_row["file_hash"], (
                "Dirty library edits should preserve the restored file_hash"
            )
            assert last_args[3] == note_row["vlm_model"], (
                "Dirty library edits should preserve the restored VLM model"
            )
            assert last_args[4] == note_row["llm_model"], (
                "Dirty library edits should preserve the restored LLM model"
            )
            assert (
                harness.notes_db[
                    (document.id, note_row["vlm_model"], note_row["llm_model"])
                ]["result"]["note_markdown"]
                == updated_markdown
            ), "Dirty library edits should update the persisted note body"


class TestFigureEnrichment:
    """PDF upload flow should trigger figure enrichment and degrade gracefully on failure."""

    def test_pdf_upload_triggers_figure_enrichment(self, make_app: Any) -> None:
        """Figure enrichment must be called once per PDF analysis."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.enrich_pdf_figures.call_count == 1, (
                "enrich_pdf_figures should be called once per PDF upload"
            )

    def test_pdf_enrichment_failure_does_not_crash_pipeline(self, make_app: Any) -> None:
        """If enrichment raises, the pipeline should still complete successfully."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.enrich_pdf_figures.side_effect = RuntimeError("VLM quota exceeded")
            app = _analyze_upload(harness, pdf_upload)

            _assert_no_exception(app)
            assert harness.generate_note.call_count == 1, (
                "Note generation should proceed even if enrichment fails"
            )
            assert harness.save_document.call_count == 1, (
                "Document should be persisted even if enrichment fails"
            )

    def test_note_renders_figures_inline(self, make_app: Any, tmp_path: Path) -> None:
        """Note read view should not crash when enriched FIGURE blocks have image_path set."""
        from PIL import Image as _PILImage
        import io as _io

        buf = _io.BytesIO()
        _PILImage.new("RGB", (4, 4), color=(128, 128, 128)).save(buf, "PNG")
        img_file = tmp_path / "fig.png"
        img_file.write_bytes(buf.getvalue())
        img_path = str(img_file)

        pdf_upload = _pdf_upload()
        fh = pdf_upload.file_hash
        doc_cache_key = f"doc_{ANALYSIS_CACHE_VERSION}_{fh}_{DEFAULT_VLM_MODEL}"

        with make_app() as harness:

            def _enrich_inject_figure(doc: Document, **kwargs: Any) -> Document:
                doc.blocks.append(
                    Block(
                        type=BlockType.FIGURE,
                        content="extracted figure",
                        order=99,
                        image_path=img_path,
                        metadata=BlockMetadata(page=1, confidence=0.9),
                    )
                )
                return doc

            harness.enrich_pdf_figures.side_effect = _enrich_inject_figure
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            cached_doc = harness.session_value(doc_cache_key)
            assert cached_doc is not None, "Document should be cached in session state"
            fig_blocks = [b for b in cached_doc.blocks if b.type == BlockType.FIGURE]
            assert fig_blocks, "Cached doc should contain the injected FIGURE block"
            assert fig_blocks[0].image_path == img_path, (
                "FIGURE block image_path should be set by the enrichment mock"
            )

    def test_qa_source_block_shows_figure_image(self, make_app: Any, tmp_path: Path) -> None:
        """Q&A chat should not crash and should preserve image_path in source blocks."""
        from PIL import Image as _PILImage
        import io as _io

        buf = _io.BytesIO()
        _PILImage.new("RGB", (4, 4), color=(64, 64, 64)).save(buf, "PNG")
        img_file = tmp_path / "src_fig.png"
        img_file.write_bytes(buf.getvalue())
        img_path = str(img_file)

        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.rag_query.side_effect = lambda question, **kwargs: _qa_result(
                "다이어그램은 데이터 흐름을 보여줍니다.",
                source_blocks=[
                    {
                        "document_id": "doc-pdf",
                        "source": "sample.pdf",
                        "block_order": 0,
                        "block_type": "figure",
                        "content_preview": "Fig. 1",
                        "page": 1,
                        "cell_index": None,
                        "image_path": img_path,
                    }
                ],
            )

            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("이 다이어그램이 뭘 보여주나?", upload=pdf_upload)

            _assert_no_exception(harness.app)
            messages = _session_messages(harness, "doc-pdf")
            assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
            assert assistant_msgs, "Assistant message expected after Q&A"
            last_srcs = assistant_msgs[-1].get("source_blocks", [])
            assert any(
                s.get("image_path") == img_path for s in last_srcs
            ), "image_path should be propagated into chat message source_blocks"


# ---------------------------------------------------------------------------
# _render_qa_notice palette unit tests
# ---------------------------------------------------------------------------


def test_render_qa_notice_loading_uses_info_palette() -> None:
    """Loading state should render Steel Blue info palette (#DDE8ED / #5A7B8C)."""
    spec = importlib.util.spec_from_file_location("ui_demo_qa_notice_loading", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    captured: list[str] = []
    with (
        patch.object(st, "set_page_config", side_effect=_StopDemoImport),
        patch.object(st, "markdown", side_effect=lambda h, **kw: captured.append(str(h))),
    ):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass
        module._render_qa_notice("Q&A 인덱싱 중입니다. 잠시 기다려 주세요...", is_loading=True)

    assert captured, "st.markdown should be called by _render_qa_notice"
    html = captured[0]
    assert "#F5EDE4" in html, "Loading notice background should use warm accent palette"
    assert "#C4553A" in html, "Loading notice border should use warm accent palette"
    assert "⏳" in html, "Loading notice should include a spinner icon"


def test_render_qa_notice_disabled_uses_warning_palette() -> None:
    """Disabled state (vectors missing) should render Amber warning palette (#F5EBDB / #C4883A)."""
    spec = importlib.util.spec_from_file_location("ui_demo_qa_notice_disabled", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    captured: list[str] = []
    with (
        patch.object(st, "set_page_config", side_effect=_StopDemoImport),
        patch.object(st, "markdown", side_effect=lambda h, **kw: captured.append(str(h))),
    ):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass
        module._render_qa_notice("Q&A를 사용하려면 문서 벡터가 필요합니다.", is_loading=False)

    assert captured, "st.markdown should be called by _render_qa_notice"
    html = captured[0]
    assert "#F5EBDB" in html, "Disabled notice background should use Amber warning palette"
    assert "#C4883A" in html, "Disabled notice border should use Amber warning palette"
    assert "ℹ️" in html, "Disabled notice should include an info icon"


# ---------------------------------------------------------------------------
# Language parameter propagation
# ---------------------------------------------------------------------------


class TestLanguageParam:
    """Output language selectbox should wire through to all downstream callers."""

    def test_language_selectbox_renders_with_default_ko(self, make_app: Any) -> None:
        """The language selectbox should default to 'ko' on first load."""
        with make_app() as harness:
            harness.run()
            lang_sb = harness.app.selectbox(key="output_language_select")
            assert lang_sb.value == "ko", "Language selectbox should default to Korean"

    def test_pdf_note_generation_called_with_language_en(self, make_app: Any) -> None:
        """Switching to English should call generate_note_sectioned with language='en'."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.run()
            harness.run(upload=pdf_upload)
            harness.set_selectbox("output_language_select", "en", upload=pdf_upload)
            harness.click_button(label="분석 시작")

            _assert_no_exception(harness.app)
            assert harness.generate_note.called, "Note generation should be called"
            _, kwargs = harness.generate_note.call_args
            assert kwargs.get("language") == "en", (
                "generate_note_sectioned should receive language='en' from selectbox"
            )

    def test_image_parse_called_with_language_en(self, make_app: Any) -> None:
        """Image upload with English selectbox should call parse_image with language='en'."""
        img_upload = _image_upload()

        with make_app() as harness:
            harness.run()
            harness.run(upload=img_upload)
            harness.set_selectbox("output_language_select", "en", upload=img_upload)
            harness.click_button(label="분석 시작")

            _assert_no_exception(harness.app)
            assert harness.parse_image.called, "parse_image should be called for image upload"
            _, kwargs = harness.parse_image.call_args
            assert kwargs.get("language") == "en", (
                "parse_image should receive language='en' from selectbox"
            )

    def test_enrich_pdf_figures_called_with_language_en(self, make_app: Any) -> None:
        """PDF enrichment should receive the language param from the selectbox."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.run()
            harness.run(upload=pdf_upload)
            harness.set_selectbox("output_language_select", "en", upload=pdf_upload)
            harness.click_button(label="분석 시작")

            _assert_no_exception(harness.app)
            assert harness.enrich_pdf_figures.called, "enrich_pdf_figures should be called for PDF"
            _, kwargs = harness.enrich_pdf_figures.call_args
            assert kwargs.get("language") == "en", (
                "enrich_pdf_figures should receive language='en' from selectbox"
            )


# ---------------------------------------------------------------------------
# _parse_followup_suggestions unit tests
# ---------------------------------------------------------------------------


def _load_parse_followup() -> Any:
    """Load _parse_followup_suggestions from ui/demo.py without running the app."""
    spec = importlib.util.spec_from_file_location("ui_demo_parse_followup", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.object(st, "set_page_config", side_effect=_StopDemoImport):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass
    return module._parse_followup_suggestions


def test_parse_followup_single_newline() -> None:
    """Standard single-newline-separated block should parse correctly."""
    fn = _load_parse_followup()
    answer = "본문 내용\n---SUGGESTIONS---\n질문1\n질문2\n질문3\n---END---"
    clean, suggestions = fn(answer)
    assert clean == "본문 내용"
    assert suggestions == ["질문1", "질문2", "질문3"]


def test_parse_followup_double_newline_before_marker() -> None:
    """LLM sometimes emits two blank lines before ---SUGGESTIONS---; must still parse."""
    fn = _load_parse_followup()
    answer = "본문 내용\n\n---SUGGESTIONS---\n질문1\n질문2\n질문3\n---END---"
    clean, suggestions = fn(answer)
    assert clean == "본문 내용", "double-newline before marker should be stripped from clean answer"
    assert len(suggestions) == 3


def test_parse_followup_no_block_returns_unchanged() -> None:
    """Answer without SUGGESTIONS block should be returned as-is."""
    fn = _load_parse_followup()
    answer = "그냥 답변입니다."
    clean, suggestions = fn(answer)
    assert clean == answer
    assert suggestions == []


def test_parse_followup_strips_numbering() -> None:
    """Numbered questions (1. / 1) ...) should have their prefix stripped."""
    fn = _load_parse_followup()
    answer = "답변\n---SUGGESTIONS---\n1. 질문1\n2) 질문2\n3. 질문3\n---END---"
    _, suggestions = fn(answer)
    assert suggestions == ["질문1", "질문2", "질문3"]


def test_parse_followup_truncates_to_three() -> None:
    """More than 3 suggestions should be truncated."""
    fn = _load_parse_followup()
    answer = "답변\n---SUGGESTIONS---\nQ1\nQ2\nQ3\nQ4\n---END---"
    _, suggestions = fn(answer)
    assert len(suggestions) == 3


# ---------------------------------------------------------------------------
# _render_concept_connections unit tests (CU-17)
# ---------------------------------------------------------------------------


def _load_render_concept_connections() -> Any:
    """Load _render_concept_connections from ui/demo.py without running the app."""
    spec = importlib.util.spec_from_file_location("ui_demo_concept_conn", DEMO_APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.object(st, "set_page_config", side_effect=_StopDemoImport):
        try:
            spec.loader.exec_module(module)
        except _StopDemoImport:
            pass
    fn = getattr(module, "_render_concept_connections", None)
    assert fn is not None, "_render_concept_connections not found in ui/demo.py"
    return fn


def test_concept_connections_banner_renders() -> None:
    """_render_concept_connections should call st.markdown when connections are provided."""
    fn = _load_render_concept_connections()

    connections = [
        {
            "concept_id_a": 1,
            "concept_id_b": 2,
            "confidence_score": 1.0,
            "relationship_type": "same_concept",
            "relationship_desc": "",
            "source_concept_name": "backpropagation",
            "source_canonical_name": "backpropagation",
            "target_concept_name": "backpropagation",
            "target_canonical_name": "backpropagation",
            "target_document_id": "doc-other",
            "target_document_title": "lecture_b.pdf",
        }
    ]

    rendered_calls: list[str] = []
    with patch.object(st, "markdown", side_effect=lambda html, **_kw: rendered_calls.append(html)):
        fn(connections)

    assert len(rendered_calls) == 1
    html_out = rendered_calls[0]
    assert "backpropagation" in html_out
    assert "lecture_b.pdf" in html_out


def test_concept_connections_empty_no_banner() -> None:
    """_render_concept_connections with empty list should not call st.markdown."""
    fn = _load_render_concept_connections()

    rendered_calls: list[str] = []
    with patch.object(st, "markdown", side_effect=lambda html, **_kw: rendered_calls.append(html)):
        fn([])

    assert rendered_calls == []
