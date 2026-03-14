"""AppTest-based runtime coverage for the Streamlit demo UI."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
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
from models.document import Block, BlockMetadata, BlockType, Document, DocumentFormat, DocumentMetadata
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.testing.v1 import AppTest
from vlm.client import SUPPORTED_MODELS

DEMO_APP_PATH = Path(__file__).resolve().parents[1] / "ui" / "demo.py"
DEFAULT_VLM_MODEL = SUPPORTED_MODELS[0]
DEFAULT_LLM_MODEL = SUPPORTED_LLM_MODELS[0]
TEST_SESSION_ID = "test session id"
UNSET = object()


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
            (widget for widget in widget_states.widgets if widget.id == uploader.proto.id),
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
        return f"result_{upload.file_hash}_{vlm_model}_{llm_model}"

    def doc_cache_key(
        self,
        upload: UploadFixture,
        *,
        vlm_model: str = DEFAULT_VLM_MODEL,
    ) -> str:
        """Return the app's session cache key for a parsed document."""
        return f"doc_{upload.file_hash}_{vlm_model}"

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
    return UploadFixture(file_id=file_id, name=name, mime_type="application/pdf", data=data)


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
) -> Document:
    """Build a library document row like db.sqlite.get_document() returns."""
    return Document(
        id=doc_id,
        source=source,
        format=fmt,
        blocks=[],
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


def _qa_result(answer: str, source_blocks: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    """Build a lightweight rag.query() response object."""
    return SimpleNamespace(answer=answer, source_blocks=source_blocks or [])


def _stored_document_copy(document: Document) -> Document:
    """Store a SQLite-like copy of a document without raw blocks."""
    stored = copy.deepcopy(document)
    stored.blocks = []
    return stored


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

        def _get_note(document_id: str, vlm_model: str, llm_model: str) -> dict[str, Any] | None:
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
            stack.enter_context(
                patch.object(local_script_runner.LocalScriptRunner, "__init__", _patched_init)
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
                        side_effect=lambda _path, model=DEFAULT_VLM_MODEL: _sample_document(
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
                    ),
                )
            )
            generate_note = stack.enter_context(
                patch(
                    "llm.note_generator.generate_note",
                    new=MagicMock(side_effect=lambda doc, model=DEFAULT_LLM_MODEL: _sample_note_result()),
                )
            )
            edit_section = stack.enter_context(
                patch(
                    "llm.note_editor.edit_section",
                    new=MagicMock(side_effect=lambda **kwargs: _sample_edit_result()),
                )
            )
            save_document = stack.enter_context(
                patch("db.sqlite.save_document", new=MagicMock(side_effect=_save_document))
            )
            save_note = stack.enter_context(
                patch("db.sqlite.save_note", new=MagicMock(side_effect=_save_note))
            )
            get_document = stack.enter_context(
                patch("db.sqlite.get_document", new=MagicMock(side_effect=_get_document))
            )
            get_note = stack.enter_context(
                patch("db.sqlite.get_note", new=MagicMock(side_effect=_get_note))
            )
            list_documents = stack.enter_context(
                patch("db.sqlite.list_documents", new=MagicMock(side_effect=_list_documents))
            )
            list_notes_for_document = stack.enter_context(
                patch(
                    "db.sqlite.list_notes_for_document",
                    new=MagicMock(side_effect=_list_notes_for_document),
                )
            )
            delete_document = stack.enter_context(
                patch("db.sqlite.delete_document", new=MagicMock(side_effect=_delete_document))
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
                    new=MagicMock(
                        side_effect=lambda question, **kwargs: _qa_result(
                            f"{question}에 대한 답변",
                        )
                    ),
                )
            )
            rewrite_query = stack.enter_context(
                patch(
                    "rag.rewrite_query",
                    new=MagicMock(side_effect=lambda question, model=DEFAULT_LLM_MODEL: f"rewritten::{question}"),
                )
            )
            pyperclip_copy = stack.enter_context(patch("pyperclip.copy", new=MagicMock()))

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
    assert spec is not None and spec.loader is not None, "ui/demo.py should be importable for testing"
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

    assert recorder.captions == ["참조 블록"], "Source block caption should render exactly once"
    assert len(recorder.expanders) == 2, "Duplicate source/page pairs should collapse into one expander"
    assert recorder.expanders[0]["captions"] == ["block_order: 0"], "First unique block should keep its metadata"
    assert recorder.expanders[1]["texts"] == ["다른 위치 미리보기"], "Second unique source should still render its preview"


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
            assert harness.generate_note.call_count == 0, "generate_note should not run when parsing fails"
            assert harness.save_document.call_count == 0, "save_document should not run for parse-failed documents"
            assert harness.save_note.call_count == 0, "save_note should not run for parse-failed documents"

    def test_successful_pdf_upload_flow_has_no_exceptions(self, make_app: Any) -> None:
        """PDF upload -> parse -> note generation should finish without runtime errors."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            app = _analyze_upload(harness, pdf_upload)
            cache_key = harness.cache_key(pdf_upload)

            _assert_no_exception(app)
            assert harness.parse_pdf.call_count == 1, "PDF parsing should run exactly once for the first analysis"
            assert harness.generate_note.call_count == 1, "Note generation should run for non-image uploads"
            assert harness.save_document.call_count == 1, "Analyzed documents should be persisted"
            assert harness.save_note.call_count == 1, "Generated notes should be persisted"
            assert harness.has_session_key(cache_key), "Successful analysis should populate the session cache"
            assert len(app.chat_input) == 1, "A successful PDF analysis should expose the Q&A chat input"

    def test_same_file_upload_uses_session_cache(self, make_app: Any) -> None:
        """Re-uploading the same file in one session should not call the APIs again."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.run(upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.parse_pdf.call_count == 1, "Cached uploads should skip parse_pdf on rerun"
            assert harness.generate_note.call_count == 1, "Cached uploads should skip generate_note on rerun"


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
            harness.set_selectbox("editor_doc-pdf_edit_section_selectbox", 1, upload=pdf_upload)
            harness.set_radio("active_right_panel", "💬 Q&A", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert len(_session_messages(harness, "doc-pdf")) == 2, "Q&A history should survive a round-trip to the note editor"

            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.set_toggle("note_edit_toggle", True, upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.session_value("active_right_panel") == "✏️ 노트 수정", "Toggling left-panel edit mode should not reset the active right panel"

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
            assert harness.app.selectbox(key="qa_model_select").value == qa_model, "Q&A model selection should persist after visiting the note editor"
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            assert harness.app.selectbox(key="editor_doc-pdf_note_editor_model_select").value == editor_model, "Note editor model selection should stay independent from the Q&A model"


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
            assert messages[-1]["content"], "Emotional queries should still produce a non-empty assistant message"
            assert "같이 정리" in messages[-1]["content"], "The assistant should return the empathetic fallback response"

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
        answer = "답변 본문\n---SUGGESTIONS---\n1. 후속 질문 A\n2. 후속 질문 B\n---END---"

        with make_app() as harness:
            harness.rag_query.side_effect = lambda question, **kwargs: _qa_result(
                answer,
                source_blocks,
            )
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("첫 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rag_query.call_args.kwargs["document_id"] == "doc-pdf", "rag.query should always be filtered to the active document"
            assert any(button.label == "후속 질문 A" for button in harness.app.button), "The first follow-up suggestion pill should render after a RAG answer"


class TestQueryRewriting:
    """Runtime checks for the CU-13 query-rewriting control."""

    def test_query_rewriting_checkbox_renders_with_default_off(self, make_app: Any) -> None:
        """The Q&A panel should render a Query Rewriting checkbox, defaulting to OFF."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            checkbox = harness.app.checkbox(key="qa_query_rewrite_doc-pdf")
            assert checkbox.label == "Query Rewriting", "The Q&A panel should expose the Query Rewriting checkbox"
            assert checkbox.value is False, "Query Rewriting should default to OFF"

    def test_rewrite_is_not_called_when_checkbox_is_off(self, make_app: Any) -> None:
        """With the checkbox OFF, questions should go straight to rag.query."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_chat_input("기본 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 0, "rewrite_query should not run while the checkbox is OFF"
            assert harness.rag_query.call_args.args[0] == "기본 질문", "rag.query should receive the original question when rewriting is disabled"

    def test_rewrite_is_called_once_when_checkbox_is_on(self, make_app: Any) -> None:
        """With the checkbox ON, rewrite_query should run exactly once per submitted question."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_query_rewrite_doc-pdf", True, upload=pdf_upload)
            harness.set_chat_input("재작성 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, "rewrite_query should run once when Query Rewriting is enabled"
            assert harness.rag_query.call_args.args[0] == "rewritten::재작성 질문", "rag.query should receive the rewritten query text"

    def test_toggling_query_rewriting_on_then_off_stays_stable(self, make_app: Any) -> None:
        """The Q&A panel should remain stable after toggling Query Rewriting on and back off."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_query_rewrite_doc-pdf", True, upload=pdf_upload)
            harness.set_chat_input("첫 질문", upload=pdf_upload)
            harness.set_checkbox("qa_query_rewrite_doc-pdf", False, upload=pdf_upload)
            harness.set_chat_input("두 번째 질문", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, "Turning rewriting back OFF should stop additional rewrite calls"
            assert harness.rag_query.call_args.args[0] == "두 번째 질문", "After toggling OFF, rag.query should receive the original question again"

    def test_rewrite_failure_falls_back_to_original_query(self, make_app: Any) -> None:
        """Rewrite failures should fall back to the original user query without crashing."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            harness.rewrite_query.side_effect = RuntimeError("rewrite failed")
            _analyze_upload(harness, pdf_upload)
            harness.set_checkbox("qa_query_rewrite_doc-pdf", True, upload=pdf_upload)
            harness.set_chat_input("원본 쿼리", upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.rewrite_query.call_count == 1, "rewrite_query should still be attempted when the checkbox is ON"
            assert harness.rag_query.call_args.args[0] == "원본 쿼리", "rewrite failures should fall back to the original query text"


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
            assert harness.parse_image.call_count == 1, "Image uploads should route through parse_image"
            assert harness.generate_note.call_count == 0, "Image uploads should bypass note generation"
            assert harness.session_value(f"is_image_{cache_key}") is True, "Image analyses should set the is_image session flag"
            assert len(harness.app.radio) == 0, "Image mode should not render the PDF-only right-panel radio"

            harness.run(upload=image_upload)

            _assert_no_exception(harness.app)
            assert harness.parse_image.call_count == 1, "Rerunning the same image should reuse the cached analysis"
            assert harness.session_value(f"is_image_{cache_key}") is True, "The image workspace should survive reruns while cached"

    def test_image_api_key_error_surfaces_message(self, make_app: Any) -> None:
        """Image parser auth failures should show a user-facing API key error."""
        image_upload = _image_upload(name="slide.webp", file_id="upload-webp")

        with make_app() as harness:
            harness.parse_image.side_effect = RuntimeError("API key missing for image parse")
            harness.run()
            harness.run(upload=image_upload)
            harness.click_button(label="분석 시작")

            _assert_no_exception(harness.app)
            assert any("API 키" in error.value for error in harness.app.error), "API-key failures should surface the dedicated st.error message"
            assert harness.generate_note.call_count == 0, "generate_note should never run when image parsing fails"

    def test_pdf_upload_still_reaches_qna_after_image_branch_changes(
        self,
        make_app: Any,
    ) -> None:
        """The PDF pipeline should keep the original parse -> note -> Q&A flow."""
        pdf_upload = _pdf_upload(name="regression.pdf", file_id="upload-pdf-regression")

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.generate_note.call_count == 1, "PDF uploads should still generate a note after the image branch changes"
            assert harness.app.radio(key="active_right_panel").value == "💬 Q&A", "Successful PDF analysis should still land on the Q&A panel"
            assert len(harness.app.chat_input) == 1, "PDF analyses should still expose the Q&A chat input"


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
            assert harness.edit_section.call_count == 1, "Submitting an edit prompt should call edit_section"
            assert harness.edit_section.call_args.kwargs["document_id"] == "doc-pdf", "Section-edit RAG grounding should be scoped to the current document"

            harness.click_button(label="✅ 적용", upload=pdf_upload)
            cache_key = harness.cache_key(pdf_upload)
            updated_result = harness.session_value(cache_key)

            _assert_no_exception(harness.app)
            assert updated_result["note_markdown"] == edited_markdown, "Apply should replace the pending note markdown"
            assert "핵심 내용" in updated_result["note_markdown"], "Applying one section edit must preserve other sections"
            assert not harness.has_session_key("editor_doc-pdf_edit_pending_markdown"), "Pending preview markdown should be cleared after Apply"
            assert harness.save_note.call_count >= 2, "Applying a note edit should persist the dirty note back to SQLite"

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
            assert not harness.has_session_key("editor_doc-pdf_edit_pending_markdown"), "Cancel should clear edit_pending_markdown"
            assert harness.session_value(harness.cache_key(pdf_upload))["note_markdown"] == original_markdown, "Cancel must leave the saved note markdown untouched"

    def test_empty_edit_input_is_safe(self, make_app: Any) -> None:
        """An empty note-editor chat input should be ignored without crashing."""
        pdf_upload = _pdf_upload()

        with make_app() as harness:
            _analyze_upload(harness, pdf_upload)
            harness.set_radio("active_right_panel", "✏️ 노트 수정", upload=pdf_upload)
            harness.app.chat_input[0].set_value("")
            harness.run(upload=pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.edit_section.call_count == 0, "Empty edit prompts should not call edit_section"

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
            restored_markdown = harness.session_value(harness.cache_key(pdf_upload))["note_markdown"]
            assert "개요 내용" in restored_markdown, "Undo should restore the previous section body"
            assert "개요 수정" not in restored_markdown, "Undo should remove the applied edited body"


class TestLibraryPersistence:
    """Coverage for saved-document restoration and session persistence."""

    def test_analysis_saves_and_sidebar_lists_document(self, make_app: Any) -> None:
        """Completed analyses should persist and show up in the sidebar library."""
        pdf_upload = _pdf_upload(name="library.pdf", file_id="upload-library")
        parsed_doc = _sample_document(doc_id="doc-library", source="library.pdf")

        with make_app() as harness:
            harness.parse_pdf.side_effect = lambda _path: copy.deepcopy(parsed_doc)
            _analyze_upload(harness, pdf_upload)

            _assert_no_exception(harness.app)
            assert harness.save_document.call_count == 1, "Finished analyses should call save_document"
            assert harness.save_note.call_count == 1, "Finished analyses should call save_note once"
            assert harness.app.button(key="lib_doc-library").label.startswith("📄"), "Saved documents should render in the sidebar library"

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
            assert harness.parse_pdf.call_count == 0, "Library restores should not rerun parse_pdf"
            assert harness.generate_note.call_count == 0, "Library restores should not rerun generate_note"
            assert harness.session_value("_library_mode") is True, "Loading from the library should enable library mode"
            assert harness.session_value("_library_doc_id") == document.id, "The active library document ID should be tracked in session state"

            harness.run(upload=upload)

            _assert_no_exception(harness.app)
            assert harness.session_value("_library_mode") is True, "A stale uploader value for the same file must not cancel library mode"
            assert harness.parse_pdf.call_count == 0, "A stale uploader value should not restart the pipeline"

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
            assert deletion_order == ["delete_document_index", "delete_document"], "Sidebar deletion should remove vector index before deleting SQLite rows"
            assert not harness.has_session_key(cache_key), "Deleting a document should evict its cached note result from session_state"

            harness.run(upload=upload)
            harness.click_button(label="분석 시작", upload=upload)

            _assert_no_exception(harness.app)
            assert harness.parse_pdf.call_count == 2, "Re-uploading a deleted document should rerun parse_pdf instead of hitting the old cache"

    def test_library_chats_are_isolated_between_documents(self, make_app: Any) -> None:
        """Loading document B after chatting on document A should start with an empty Q&A history."""
        doc_a = _sample_document_row(doc_id="doc-a", source="alpha.pdf")
        doc_b = _sample_document_row(doc_id="doc-b", source="beta.pdf")
        note_a = _sample_note_row(document_id=doc_a.id, file_hash="hash-a", result=_sample_note_result(title="알파 노트"))
        note_b = _sample_note_row(document_id=doc_b.id, file_hash="hash-b", result=_sample_note_result(title="베타 노트"))

        with make_app() as harness:
            harness.seed_library(doc_a, note_a, has_vectors=True)
            harness.seed_library(doc_b, note_b, has_vectors=True)
            harness.run()
            harness.click_button(key="lib_doc-a", upload=None)
            harness.set_chat_input("문서 A 질문", upload=None)

            _assert_no_exception(harness.app)
            assert len(_session_messages(harness, "doc-a")) == 2, "Document A should record its Q&A chat history"

            harness.click_button(key="lib_doc-b", upload=None)

            _assert_no_exception(harness.app)
            assert _session_messages(harness, "doc-b") == [], "Document B should start with an empty chat history"
            assert len(_session_messages(harness, "doc-a")) == 2, "Document A chat history should remain isolated under its own session_state key"

    def test_qna_is_disabled_when_vectors_are_missing(self, make_app: Any) -> None:
        """Library documents without vectors should not render an active Q&A chat input."""
        document = _sample_document_row(doc_id="doc-no-vectors", source="novectors.pdf")
        note_row = _sample_note_row(document_id=document.id, file_hash="hash-no-vectors")

        with make_app() as harness:
            harness.seed_library(document, note_row, has_vectors=False)
            harness.run()
            harness.click_button(key="lib_doc-no-vectors", upload=None)

            _assert_no_exception(harness.app)
            assert len(harness.app.chat_input) == 0, "Q&A should be disabled when no document vectors are available"
            assert any("벡터" in info.value for info in harness.app.info), "The UI should explain why Q&A is unavailable without vectors"

    def test_library_note_edit_persists_dirty_note(self, make_app: Any) -> None:
        """Editing a library-loaded note should save with the restored file hash and models."""
        document = _sample_document_row(doc_id="doc-library-edit", source="editable.pdf")
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
            harness.click_button(key="editor_doc-library-edit_direct_edit_save", upload=None)

            _assert_no_exception(harness.app)
            last_args = harness.save_note.call_args_list[-1].args
            assert last_args[0] == document.id, "Dirty library edits should save back to the loaded document"
            assert last_args[1] == note_row["file_hash"], "Dirty library edits should preserve the restored file_hash"
            assert last_args[3] == note_row["vlm_model"], "Dirty library edits should preserve the restored VLM model"
            assert last_args[4] == note_row["llm_model"], "Dirty library edits should preserve the restored LLM model"
            assert harness.notes_db[(document.id, note_row["vlm_model"], note_row["llm_model"])]["result"]["note_markdown"] == updated_markdown, "Dirty library edits should update the persisted note body"
