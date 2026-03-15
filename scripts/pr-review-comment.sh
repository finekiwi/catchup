#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/pr-review-comment.sh <pr-number> [--dry-run] [--edit-last]

Generate a PR review with Codex using refs/AGENTS.md and either print it
or post it to GitHub with gh.

Options:
  --dry-run    Print the generated review instead of posting it
  --edit-last  Update the last comment from the current GitHub user
  -h, --help   Show this help message
EOF
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

parse_repo_slug() {
  local remote_url
  remote_url=$(git remote get-url origin)

  case "$remote_url" in
    git@github.com:*)
      remote_url=${remote_url#git@github.com:}
      printf '%s\n' "${remote_url%.git}"
      ;;
    https://github.com/*)
      remote_url=${remote_url#https://github.com/}
      printf '%s\n' "${remote_url%.git}"
      ;;
    *)
      die "Unsupported origin URL: $remote_url"
      ;;
  esac
}

pr_number=""
dry_run=0
edit_last=0

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=1
      ;;
    --edit-last)
      edit_last=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      [[ -z "$pr_number" ]] || die "Only one PR number is supported"
      pr_number="$1"
      ;;
  esac
  shift
done

[[ -n "$pr_number" ]] || {
  usage >&2
  exit 1
}

require_command codex
require_command gh
require_command git

[[ -f refs/AGENTS.md ]] || die "Missing refs/AGENTS.md"

if ! gh auth status -h github.com >/dev/null 2>&1; then
  die "GitHub CLI is not authenticated. Run: gh auth login -h github.com"
fi

repo_slug=$(parse_repo_slug)
pr_title=$(gh pr view "$pr_number" --json title --jq .title)
pr_body=$(gh pr view "$pr_number" --json body --jq '.body // ""')
head_ref=$(gh pr view "$pr_number" --json headRefName --jq .headRefName)
base_ref=$(gh pr view "$pr_number" --json baseRefName --jq .baseRefName)
pr_url=$(gh pr view "$pr_number" --json url --jq .url)
current_branch=$(git branch --show-current)

if [[ "$current_branch" != "$head_ref" ]]; then
  die "Current branch is '$current_branch' but PR #$pr_number head is '$head_ref'. Run: gh pr checkout $pr_number"
fi

if ! git rev-parse --verify "origin/$base_ref" >/dev/null 2>&1; then
  die "Missing origin/$base_ref locally. Run: git fetch origin $base_ref"
fi

review_comments=$(
  gh api "repos/$repo_slug/pulls/$pr_number/comments" --paginate \
    --jq '.[] | "- " + (.path // "unknown") + ":" + ((.line // 0) | tostring) + " " + (.body | gsub("\r"; ""))'
)
issue_comments=$(
  gh api "repos/$repo_slug/issues/$pr_number/comments" --paginate \
    --jq '.[] | "- " + (.user.login // "unknown") + ": " + (.body | gsub("\r"; ""))'
)

prompt_file=$(mktemp)
output_file=$(mktemp)
trap 'rm -f "$prompt_file" "$output_file"' EXIT

cat >"$prompt_file" <<__CODEX_PR_PROMPT__
You are reviewing GitHub pull request #$pr_number for $repo_slug.

Follow the review conventions in refs/AGENTS.md exactly.
Important requirements:
- Read refs/AGENTS.md before reviewing.
- Review the checked-out branch \`$head_ref\` against \`origin/$base_ref\`.
- Use the priority buckets P0, P1, P2, P3 exactly as defined there.
- Write the review comment in English.
- Output only the final GitHub comment body. No preamble, no explanations, no code fences.

PR metadata:
- URL: $pr_url
- Title: $pr_title
- Head branch: $head_ref
- Base branch: $base_ref

PR body:
$pr_body

Existing review comments:
${review_comments:-None}

Existing issue comments:
${issue_comments:-None}

Before you finalize:
- Avoid repeating already-resolved findings from existing comments unless they still apply.
- Prefer concrete file:line references when possible.
- If a section has no findings, write "None ✅".
__CODEX_PR_PROMPT__

codex exec -s read-only -o "$output_file" - <"$prompt_file"
[[ -s "$output_file" ]] || die "Codex returned an empty review"

if ((dry_run)); then
  cat "$output_file"
  exit 0
fi

comment_args=(pr comment "$pr_number" --body-file "$output_file")
if ((edit_last)); then
  comment_args+=(--edit-last --create-if-none)
fi

gh "${comment_args[@]}"
printf 'Posted review to %s\n' "$pr_url"
