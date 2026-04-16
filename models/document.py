"""
Intermediate format schema for CatchUp pipeline.

All parsers (PDF, ipynb, image) produce Document objects.
All downstream modules (note_generator, concept_extractor, RAG) consume them.

⚠️ Changes to this file affect the ENTIRE pipeline.
   Requires Claude Projects approval before modification.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


class DocumentFormat(str, Enum):
    """Supported input document formats."""

    PDF = "pdf"
    IPYNB = "ipynb"
    IMAGE = "image"


class BlockType(str, Enum):
    """Content block types extracted from documents."""

    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"


class ImageType(str, Enum):
    """Image subtypes for VLM processing strategy."""

    CODE_SCREENSHOT = "code_screenshot"
    DIAGRAM = "diagram"
    CHART = "chart"
    TEXT_CAPTURE = "text_capture"
    EQUATION = "equation"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Pipeline processing stage tracking."""

    PARSED = "parsed"  # parser completed
    NOTE_GENERATED = "note_generated"  # note generation completed
    CONCEPTS_EXTRACTED = "concepts_extracted"  # concept extraction completed
    EMBEDDED = "embedded"  # stored in ChromaDB


class BlockMetadata(BaseModel):
    """Per-block metadata varying by source format."""

    # PDF-specific
    page: Optional[int] = None

    # ipynb-specific
    cell_index: Optional[int] = None
    cell_type: Optional[str] = None  # "code" | "markdown" | "output"

    # image/VLM-specific
    image_type: Optional[ImageType] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preprocess: Optional[dict[str, Any]] = None

    # code-specific
    language: Optional[str] = None

    # figure/table-specific
    caption: Optional[str] = None


class Block(BaseModel):
    """Single content block within a document."""

    type: BlockType
    content: str
    order: int = Field(description="Block position within the document, 0-indexed")
    metadata: BlockMetadata = Field(default_factory=BlockMetadata)

    # optional: raw image bytes path for VLM re-processing
    image_path: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Document-level metadata."""

    title: Optional[str] = None  # LLM-extracted or filename fallback
    tags: list[str] = Field(default_factory=list)  # LLM-extracted keywords
    total_pages: Optional[int] = None  # PDF only
    total_cells: Optional[int] = None  # ipynb only


class ProcessingInfo(BaseModel):
    """Pipeline processing tracking."""

    parser_model: Optional[str] = None  # e.g. "docling", "nbformat", "gpt-4o-mini"
    vlm_model: Optional[str] = None  # VLM used for image blocks
    llm_model: Optional[str] = None  # LLM used for note generation
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class Document(BaseModel):
    """
    Core intermediate format for CatchUp pipeline.

    Every parser outputs this. Every downstream module consumes this.
    ID is derived from file content hash for deduplication and caching.
    """

    id: str = Field(description="SHA256 hash of source file content")
    source: str = Field(description="Original filename")
    format: DocumentFormat
    blocks: list[Block] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)
    status: ProcessingStatus = ProcessingStatus.PARSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def block_count(self) -> int:
        """Total number of blocks in the document."""
        return len(self.blocks)

    def get_blocks_by_type(self, block_type: BlockType) -> list[Block]:
        """Filter blocks by type."""
        return [b for b in self.blocks if b.type == block_type]

    def to_plain_text(self) -> str:
        """Concatenate all block contents for embedding or LLM input."""
        return "\n\n".join(b.content for b in self.blocks)


def generate_document_id(file_path: str) -> str:
    """
    Generate document ID from file content hash.

    Used for:
    - Deduplication: same file -> same ID -> skip re-processing
    - Caching: check if document already exists in DB
    """
    path = Path(file_path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]
