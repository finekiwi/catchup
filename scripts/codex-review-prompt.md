# Codex PR Review Prompt Template

아래를 Codex 태스크에 복붙해서 사용.
PR 번호만 바꿔서 입력하면 됨.

---

## 프롬프트 (복붙용)

```
PR #{번호}를 리뷰해줘.

1. `gh pr diff {번호}`로 diff 확인
2. `gh pr view {번호} --json title,body,headRefName`로 PR 정보 확인
3. AGENTS.md의 "절대 규칙"과 "리뷰 컨벤션" 기준으로 분석
4. 아래 포맷으로 `gh pr comment {번호}`로 코멘트 달아줘

포맷:
# PR Review — `{branch_name}`

## P0 (Must Fix)
[findings with `file:line` — or "None"]

## P1 (Should Fix)
[findings — or "None"]

## P2 (Deferred to follow-up PR)
[findings — or "None"]

## P3 (Nit)
[findings — or "None"]

**Verdict: [Fix P0 + P1, then merge. P2 deferred. / Clean — ready to merge.]**

*Reviewed with Codex*

체크 항목:
- prompts/ 수정 시 VERSION_LOG.md 업데이트 여부
- models/document.py 변경 여부
- db/ 스키마 변경 여부
- 새 의존성 추가 여부
- 타입힌트, docstring 누락
- log_api_call() 누락
- 하드코딩된 프롬프트
- import 순서 (stdlib → third-party → local)
- PR description이 AGENTS.md 포맷(Summary → Design Decisions → Test plan → Notes) 준수하는지
```
