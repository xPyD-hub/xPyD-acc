<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Author Policy — xPyD-acc

Rules for the bot that writes code and submits PRs.

## Identity

| Role | GitHub Account |
|------|---------------|
| Author | `hlin99` |

## Before Coding

1. Pull latest main: `git pull origin main`
2. Create feature branch: `git checkout -b <type>/<short-description>`
3. Read [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) for architecture constraints.

## Code Quality

1. Run lint: `ruff check xpyd_acc tests`
2. Run tests: `pytest tests/ -q`
3. Rebase before push: `git pull --rebase origin main`

## PR Submission

1. **One PR per task.** Don't bundle unrelated changes.
2. **Descriptive title.** Format: `type: short description` (e.g., `feat: add KV cache comparison`).
3. **PR body must include:** what changed, why, test coverage, breaking changes.
4. **All CI must pass** before requesting review.

## Responding to Review

1. Fix all blockers before re-requesting review.
2. Reply to every comment — "Fixed in <commit>" or explain disagreement with evidence.
3. Push new commits (don't force push over reviewer comments).
4. **Never force push.** If the branch is too messy, close the PR and open a new one.

## Documentation Updates

Every PR must update relevant documentation:

| Change Type | Update |
|---|---|
| New feature / CLI argument | `docs/guide.md` — add usage section |
| Design decision | `bot/DESIGN_PRINCIPLES.md` — append if needed |
| Quick Start affected | `README.md` — update (keep it one screen max, link to guide.md) |
| PR completed | `bot/iterations/current.md` — append summary |

`docs/guide.md` is the source of truth for how to use the tool.

When current iteration is complete, rename `bot/iterations/current.md` to `YYYY-MM-DD-<topic>.md` and create a fresh `current.md`.
