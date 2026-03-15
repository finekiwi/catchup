## 2026-03-13

### RAG Q&A scope note

- Fixed in current scope:
  - Q&A retrieval is now restricted to the currently uploaded document.
  - Pure emotional fallback was tightened so it does not suggest unrelated generic topics.

- Confirmed out of scope for the current fix:
  - Colloquial summary/comparison questions such as "가드레일로 젤 효과적인게 뭐임" can still fall back to
    "해당 내용은 문서에서 찾을 수 없습니다. 다른 질문이 있으시면 알려주세요."

- Why that still happens:
  - Retrieval recall is still conservative for colloquial wording and comparative questions.
  - `rag.query()` still uses a small top-k and no query normalization.
  - The RAG prompt still strongly prefers "not found" when context is incomplete or weak.

- Follow-up task candidates:
  - Add colloquial query normalization.
  - Revisit retrieval/top-k settings for summary/comparison questions.
  - Relax the RAG fallback so it synthesizes from partial relevant context before saying "not found".

### Image VLM scope note

- Confirmed out of scope for the current UI/RAG fix:
  - Some image uploads are misclassified as `code_screenshot`, then produce hallucinated code-like blocks
    (for example guardrail material being turned into fake Python code and then indexed into RAG).

- Why that happens:
  - The VLM can misread dense screenshots with mixed text / diagram / code-like layout.
  - The current pipeline trusts the VLM output too directly:
    - classification sends the image down the code path,
    - code extraction output is accepted as a `CODE` block,
    - RAG indexes that block as if it were reliable source content.

- What is not enough:
  - This is not just a UI problem.
  - This is not just a "use a better model" problem.

- Conclusion:
  - This is not mainly a Q&A prompt problem.
  - The bigger issue is that the pipeline trusts image-parser / VLM output too much once it has been classified and structured.

- Proper follow-up scope:
  - `parsers/image_parser.py`
  - `prompts/vlm_classify.py`
  - `prompts/vlm_code.py`
  - related tests for image parsing / downgrade behavior

- Follow-up task candidates:
  - Tighten `code_screenshot` classification criteria.
  - Add post-parse validation so natural-language-heavy or suspicious code extraction is downgraded to `text` or `other`.
  - Block or downgrade low-confidence image parse results before RAG indexing.
  - Avoid generating code-focused starter prompts when the parsed image content is low-confidence or obviously non-code.
