"""Pydantic schemas for LLM note-generation outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DifficultyLevel = Literal["beginner", "intermediate", "advanced"]


class NoteGenerationOutput(BaseModel):
    """Validated JSON schema returned by note generation model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    title: str = ""
    summary: str = ""
    note_markdown: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    difficulty_level: DifficultyLevel
    estimated_read_time_min: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)

    @field_validator("key_concepts")
    @classmethod
    def _normalize_key_concepts(cls, value: list[str]) -> list[str]:
        """Strip empty key concepts and keep deterministic ordering."""
        return [item.strip() for item in value if item.strip()]

    @field_validator("errors")
    @classmethod
    def _normalize_errors(cls, value: list[str]) -> list[str]:
        """Normalize empty/whitespace-only error items."""
        return [item.strip() for item in value if item.strip()]

