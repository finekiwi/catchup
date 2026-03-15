"""Pydantic schemas for VLM JSON outputs.

These schemas represent the VLM response contract, not the internal Block schema.
The conversion happens in `parsers/image_parser.py`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VLMOutputBase(BaseModel):
    """Common fields shared by all image-type VLM outputs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    has_truncation: bool
    errors: list[str] = Field(default_factory=list)

    @field_validator("errors")
    @classmethod
    def _normalize_errors(cls, value: list[str]) -> list[str]:
        """Normalize empty/whitespace-only error items."""
        return [item.strip() for item in value if item.strip()]


class CodeVLMOutput(VLMOutputBase):
    """Structured payload for code screenshot extraction."""

    language: str = Field(min_length=1)
    code: str
    code_markdown: str
    description: str = ""


class DiagramComponent(BaseModel):
    """Single component/node in a diagram."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    role: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name(cls, v: object) -> str:
        """Coerce None/non-string name to empty string."""
        return str(v) if v is not None else ""


class DiagramRelationship(BaseModel):
    """Directed relationship between two diagram components."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_component: str = Field(alias="from", min_length=1)
    to_component: str = Field(alias="to", min_length=1)
    label: str | None = None


_KNOWN_DIAGRAM_TYPES = {
    "flowchart", "architecture", "sequence", "er", "class", "mindmap", "network", "other"
}

DiagramType = Literal[
    "flowchart",
    "architecture",
    "sequence",
    "er",
    "class",
    "mindmap",
    "network",
    "other",
]


class DiagramVLMOutput(VLMOutputBase):
    """Structured payload for diagram extraction."""

    diagram_type: DiagramType = "other"

    @field_validator("diagram_type", mode="before")
    @classmethod
    def _normalize_diagram_type(cls, v: object) -> str:
        """Map unknown diagram_type values to 'other'."""
        return v if v in _KNOWN_DIAGRAM_TYPES else "other"
    title: str | None = None
    description: str = ""
    components: list[DiagramComponent] = Field(default_factory=list)
    relationships: list[DiagramRelationship] = Field(default_factory=list)
    flow_summary: str = ""


TextType = Literal[
    "lecture_slide",
    "handwritten_notes",
    "textbook_page",
    "article",
    "whiteboard",
    "other",
]


class TextVLMOutput(VLMOutputBase):
    """Structured payload for image text extraction."""

    text_type: TextType
    title: str | None = None
    content: str = ""
    key_points: list[str] = Field(default_factory=list)
    has_math: bool
