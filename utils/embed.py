"""Public embedding utility — shared by rag/qa_chain.py and llm/concept_linker.py."""

from __future__ import annotations

EMBED_MODEL = "text-embedding-3-small"


def get_openai_embedding(text: str) -> tuple[list[float], int]:
    """Embed text using OpenAI text-embedding-3-small.

    Args:
        text: Input text to embed.

    Returns:
        Tuple of (embedding vector, total_tokens used).
    """
    import openai  # lazy import

    client = openai.OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding, resp.usage.total_tokens
