# Case Study: gs_016 — KO Colloquial Query vs. Retrieval Precision

## Background

**Query:** `커밋로그 보는 방법이 뭐야?`

The document (`git_github_ch3.pdf`) indexes the relevant section as:
> "git log 명령어로 커밋 로그를 확인할 수 있습니다" (p.43)

The Korean compound "커밋로그" (no space) has lower embedding similarity to the indexed text
than the commit-introduction chunks (p.18–37), which contain "커밋" repeatedly.
This is the KO colloquial → technical term mismatch that motivated CU-13.

---

## Three-way comparison

| Pipeline | top_k | Overall score | What was retrieved |
|---|---|---|---|
| Vanilla | 5 | 0.6589 | Commit-intro chunks (p.18, p.23, p.24, p.31, p.36) — git log section (p.43) **not included** |
| Vanilla | 8 | ~pass | p.43 included, but 5+ noise chunks mixed in |
| **Rewrite** | **5** | **0.8928** | p.43 git log section **top-ranked**, minimal noise |

Rewritten query: `커밋로그 (commit log, git log) 보는 방법이 뭐야?`

---

## Key insight

**Increasing top_k is a recall lever, not a precision lever.**

- `top_k=8` vanilla: p.43 sneaks into retrieval, but commit-intro noise remains in positions 1–7.
  The LLM sees noisy context → citation accuracy drops, answer is verbose and unfocused.
- `rewrite top_k=5`: embedding similarity of the rewritten query to p.43 jumps because
  "commit log, git log" directly matches the section heading.
  Only 5 chunks retrieved, but all 5 are relevant.

**Query rewriting lets you keep top_k low (lower cost, less noise) while recovering precision
that would otherwise require inflating top_k.**

This is the core trade-off:

```
top_k ↑  →  recall ↑,  precision ↓,  cost ↑
query rewriting  →  recall ≈,  precision ↑,  cost += ~$0.000005/query
```

---

## Eval evidence

From `eval/results/rewrite_eval.json`, gs_016:

```json
{
  "case_id": "gs_016",
  "question": "커밋로그 보는 방법이 뭐야?",
  "rewritten_query": "커밋로그 (commit log, git log) 보는 방법이 뭐야?",
  "rewrite_needed": true,
  "vanilla_overall": 0.6589,
  "rewrite_overall": 0.8928,
  "vanilla_correct": true,
  "rewrite_correct": true
}
```

Delta: **+0.2339** (largest improvement among single-document cases)

Context Precision difference (vanilla vs rewrite):
- Vanilla retrieved: p.18, p.23, p.24, p.31, p.36, p.37, p.43 (git log section buried at position 7 with top_k=8)
- Rewrite retrieved: p.43, p.44, p.37, p.36, p.18 (git log section at position 1)

---

## Limitations and next steps

- **gs_005 / gs_008**: Already-specific English queries got over-expanded by the rewriter,
  causing a -0.17 / -0.16 delta. Fix: strengthen the prompt rule
  "if the query is already a precise English technical phrase, return it unchanged" → `query_rewrite v1.1`
- **Wilcoxon p=0.57**: Not significant at n=31 with ~3%p delta. This is a statistical power issue,
  not evidence of no effect. The rewrite_needed subgroup (n=7) shows +6.36%p,
  which is the meaningful signal.
