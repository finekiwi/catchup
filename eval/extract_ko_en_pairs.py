"""Extract KO-EN vocabulary mismatch pairs from golden-set documents.

Loads all 10 golden documents (5 PDF + 5 ipynb) from the parsed cache,
samples text blocks per document, and calls an LLM to extract technical
term pairs where Korean colloquial usage differs from the indexed English form.

Output: eval/ko_en_pairs.json
Usage:
    python -m eval.extract_ko_en_pairs
    python -m eval.extract_ko_en_pairs --model gpt-4.1-nano --output eval/ko_en_pairs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4.1-nano"
_DEFAULT_OUTPUT = Path("eval/ko_en_pairs.json")
_MAX_BLOCKS_PER_DOC = 60  # sample cap per document to stay within token budget
_MAX_CONTENT_LEN = 300  # truncate each block content

_GOLDEN_DIR = Path("data/golden")
_GOLDEN_FILES = [
    "deep_learning_ch3.pdf",
    "little_book_ch3.pdf",
    "d2l_ch4.pdf",
    "llm_app_ch3.pdf",
    "git_github_ch3.pdf",
    "LLM_034_neo4j_intro.ipynb",
    "LLM_016_ToolCalling_Agent.ipynb",
    "day53_langchain.ipynb",
    "elk_01_basic.ipynb",
    "Logistic_Regression_with_a_Neural_Network_mindset.ipynb",
]

_EXTRACT_PROMPT = """\
You are a retrieval-quality analyst. Given text excerpts from a technical document,
extract vocabulary pairs where a Korean colloquial query would FAIL to retrieve the
English indexed term via embedding similarity.

Focus on:
1. English technical terms that Korean users would naturally say differently
   (e.g. "git log" → Korean users say "커밋로그" or "커밋 기록")
2. English abbreviations whose expansion would aid Korean retrieval
   (e.g. "MLP" → Korean users say "다층 퍼셉트론" or "엠엘피")
3. Compound Korean words that map to spaced English terms
   (e.g. "시그모이드함수" → "sigmoid function")
4. English-only terms in Korean documents where users might query in Korean
   (e.g. "dropout" in a KO notebook → Korean users might say "드롭아웃" or "과적합 방지")

Output ONLY a JSON array (no explanation, no markdown). Each item:
{
  "en_indexed": "the term as it appears in the document",
  "ko_colloquial": ["Korean query variant 1", "Korean query variant 2"],
  "mismatch_reason": "one sentence why vanilla embedding would miss this"
}

Skip pairs where the Korean and English terms are already close enough for embedding
to match (e.g. "API" is fine as-is). Only include genuine retrieval gaps.
If no mismatch pairs exist in these excerpts, return [].
"""


def _load_doc_texts(file_path: Path) -> list[str]:
    """Load text block contents from parsed cache for a golden document."""
    from utils.cache import load_cached_parse

    doc = load_cached_parse(file_path)
    if doc is None:
        LOGGER.warning("No cached parse for %s — skipping", file_path.name)
        return []

    from models.document import BlockType

    text_blocks = [
        b
        for b in doc.blocks
        if b.type in (BlockType.TEXT, BlockType.CODE)
        and b.content
        and len(b.content.strip()) > 20
    ]

    # Sample evenly across document to cover full scope
    if len(text_blocks) > _MAX_BLOCKS_PER_DOC:
        step = len(text_blocks) / _MAX_BLOCKS_PER_DOC
        text_blocks = [text_blocks[int(i * step)] for i in range(_MAX_BLOCKS_PER_DOC)]

    return [b.content[:_MAX_CONTENT_LEN] for b in text_blocks]


def _extract_pairs_for_doc(
    source: str,
    texts: list[str],
    model: str,
    client: openai.OpenAI,
) -> list[dict]:
    """Call LLM to extract KO-EN mismatch pairs from sampled text blocks."""
    if not texts:
        return []

    content = "\n---\n".join(texts)
    user_msg = f"Document: {source}\n\nExcerpts:\n{content}"

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        raw = (response.choices[0].message.content or "").strip()
        LOGGER.info(
            "  %s: %.0fms, %d tokens in / %d out",
            source,
            latency,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        pairs = json.loads(raw)
        if not isinstance(pairs, list):
            LOGGER.warning("Unexpected response type for %s: %s", source, type(pairs))
            return []

        # Attach source to each pair
        for p in pairs:
            p["source"] = source
        return pairs

    except Exception as exc:
        LOGGER.error("LLM call failed for %s: %s", source, exc)
        return []


def run(model: str, output: Path) -> list[dict]:
    """Extract KO-EN mismatch pairs from all golden documents.

    Args:
        model: OpenAI model for extraction.
        output: Output JSON path.

    Returns:
        List of mismatch pair dicts, each with en_indexed, ko_colloquial, mismatch_reason, source.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_pairs: list[dict] = []

    for filename in _GOLDEN_FILES:
        file_path = _GOLDEN_DIR / filename
        LOGGER.info("Processing %s", filename)

        texts = _load_doc_texts(file_path)
        if not texts:
            continue

        LOGGER.info("  Loaded %d text blocks", len(texts))
        pairs = _extract_pairs_for_doc(filename, texts, model, client)
        LOGGER.info("  Extracted %d mismatch pairs", len(pairs))
        all_pairs.extend(pairs)

    # Deduplicate by (en_indexed, source)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for p in all_pairs:
        key = (p.get("en_indexed", "").lower(), p.get("source", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    LOGGER.info("Total pairs: %d (deduped from %d)", len(deduped), len(all_pairs))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Saved to %s", output)

    return deduped


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m scripts.extract_ko_en_pairs."""
    parser = argparse.ArgumentParser(
        description="Extract KO-EN vocabulary mismatch pairs from golden-set documents."
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL, help="OpenAI model for extraction"
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT, help="Output JSON path"
    )
    args = parser.parse_args(argv)

    pairs = run(args.model, args.output)

    print(f"\nExtracted {len(pairs)} KO-EN mismatch pairs.")
    print(f"Output: {args.output}")
    print("\nSample pairs:")
    for p in pairs[:5]:
        print(f"  [{p['source']}] {p['en_indexed']!r} → {p['ko_colloquial']}")


if __name__ == "__main__":
    main()
