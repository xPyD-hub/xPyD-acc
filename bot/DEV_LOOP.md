<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Development Loop

Autonomous infinite loop. Runs until explicitly stopped.

## Setup (every iteration)
```
git config user.email "hlin99@users.noreply.github.com"
git config user.name "hlin99"
```

## Each Iteration

1. Pull latest code
2. Read `ROADMAP.md` — find the next incomplete milestone
3. Read `DESIGN_PRINCIPLES.md` — follow the rules
4. Check open issues/PRs — handle unmerged PRs first (fix CI failures, address review comments)
5. If no milestone left, create new ones (see Phase 2 below)
6. Create GitHub Issue: problem, solution, acceptance criteria, tests
7. Create branch, implement code + tests
8. Pass lint and tests (see [AUTHOR_POLICY.md](AUTHOR_POLICY.md) for commands)
9. Update `iterations/current.md` with what you did this iteration
10. Create PR (body contains `Closes #N`)
11. Wait for CI green. Fix failures. Never merge red CI.
12. **Wait for reviewer bots** — do NOT self-merge. See [REVIEW_POLICY.md](REVIEW_POLICY.md) for timing and rules.
13. Handle review result:
    - **2 approvals** → auto-merge → update ROADMAP.md → go to step 1
    - **request changes** → fix code, push to same PR → wait for re-review
    - **closed by reviewer** → iteration failed → push update to `iterations/current.md` on main recording the failure → go to step 1 with a different task
14. Go to step 1

## Phase 1: Roadmap-Driven
Follow ROADMAP.md milestones in order.

## Phase 2: Continuous Evolution
When all milestones are done:
1. Review the project — find limitations, improvements, new scenarios
2. Create new milestones in ROADMAP.md
3. Return to Phase 1

## Iteration Tracking

`iterations/current.md` must maintain a running log at the bottom:

```markdown
## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | Added X feature | ✅ merged | Both approved |
| 2 | 2026-04-06 | Refactored Y | ❌ closed | BotX: idea not valuable |
```

This table is the source of truth for iteration success/failure rate.
