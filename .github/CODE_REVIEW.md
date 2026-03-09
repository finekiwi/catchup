# Code Review Convention

## Priority Levels
- `P0`: Merge blocker. Functional defect, data loss risk, security issue, or explicit spec violation.
- `P1`: Must fix before merge. High-confidence bug/risk likely to cause incorrect behavior.
- `P2`: Should fix. Quality, maintainability, or edge-case gap with moderate impact.
- `P3`: Nice to have. Style/readability/nit.

## Reviewer Checklist
- Confirm requirements are fully implemented and interfaces match task spec.
- Check prohibited areas are untouched (`models/document.py`, DB schema, unauthorized `prompts/` edits).
- Validate tests cover happy path + failure/edge path.
- Verify logging/observability requirements are present for API calls.
- Call out concrete evidence with `file:line` and expected vs actual behavior.

## Comment Format
- Prefix findings with `[P0]`, `[P1]`, `[P2]`, or `[P3]`.
- Include:
  - impacted file/line
  - why it is a problem
  - suggested fix direction

## Author Response Rules
- `P0/P1`: must resolve with code changes or provide clear technical rebuttal.
- `P2/P3`: accept/defer allowed, but reason must be stated in thread.
- Close each thread with commit hash or rationale reference.

