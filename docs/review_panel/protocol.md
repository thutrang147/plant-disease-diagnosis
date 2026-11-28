# Review panel protocol (committed before scoring)
- Snapshots: S0 main `da5e1ba`; S1 round 1 at the round-2 branch point `61c3cd1`; S2 round 2 final `bb794d7`. The intermediate pre-fix score at `ab02c7e` (7.45) came from same-model reviewers and is kept for reference only.
- Blind labels, from Python `random.Random(20261128).shuffle` on [S0, S1, S2]: A = S2, B = S0, C = S1. Reviewers see only their label.
- Reviewer model: claude-sonnet-5-5. The work was done by claude-opus-5-5 (worker and leader). Reviewers run in separate sessions with no shared conversation, so they have no access to the worker's or the leader's context.
- Checkout: a depth-1 clone at the pinned commit. Read-only instructions: no pushes, PRs or comments.
- Prompt: identical for every snapshot except the label (verbatim below).
- Reports are relayed unedited to docs/review_panel/s0_main.md, s1_round1.md and s2_round2.md. Scores go into scores.csv, and finding-to-fix links into traceability.csv.
- Known limits:
  - All models are Claude models from one vendor, so the panel is separated from the authors but is not fully independent.
  - The blinding is partial, since repository contents reveal maturity.
  - One run per snapshot, with no reruns to improve a number. A lower score than the previous snapshot is reported as is.
  - A human grader (for example a classmate using the rubric) is the only fully independent check.

## Prompt (verbatim)

```text
You are an independent grader for a university computer-vision course project. You grade ONE snapshot of a repository, called "Snapshot {LABEL}". The repository is already checked out in your working directory, pinned to a single commit. Do not fetch other branches or commits, do not look at GitHub pull requests, and do not try to work out how this snapshot relates to any other version.

STRICTLY READ-ONLY. Never commit, push, open or comment on pull requests or issues, or edit tracked files. You may run read-only commands, lightweight scripts and the test suite. You may download the published checkpoints if the repository provides a downloader. Do not train models, and do not run inference that takes longer than about 10 minutes of CPU time.

Rubric (official weights; the project is assessed independently of publication potential):
- C1 Computer Vision knowledge and technical difficulty: 20%
- C2 Implementation depth and completion: 25%
- C3 Experimental design, controls and correctness: 25%
- C4 Results analysis and identification of limitations: 15%
- C5 Report, demo and reproducibility: 15%

Rules:
1. Treat every statement in the repository's own documents as an unverified claim. This includes README, any review or summary files, and any "done", "confirmed" or "✅" markers. Score only what you can verify in the code, the data files, the committed results and the tests. If a claim cannot be verified, say so; do not give credit for it.
2. Check numbers. For each headline claim in the README, find the file that supports it and check that the value matches. List mismatches.
3. Check honesty: tuning on test sets, claims that go beyond the evidence, nulls written as equivalence, and missing controls. Penalise these under C3/C4.
4. Score each criterion from 0 to 10 in 0.5 steps. Calibrate so that 5 means "adequate", 7 means "good university project" and 9 means "exceptional"; use the whole scale. Compute the weighted total.
5. For every finding give: an id (C<k>-<n>), severity (critical / high / medium / low), file:line or file path, the evidence (a quote or a number), and a one-line suggested fix.
6. After all five criteria, do a cross-check pass. Re-read your findings, drop any you cannot back with evidence, and flag any contradictions between criteria.

Output format (Markdown). Send the whole report as ONE message with send_message to session_id "@parent". Do not write it into the repository.

# Snapshot {LABEL} — grading report
## Scores
| Criterion | Weight | Score | One-line justification |
...
Weighted total: x.xx / 10
## Findings
### C1 ... (per criterion: list of findings in the format above)
## Number checks
| Claim (file:line) | Source file | Stated | Actual | Match? |
## Cross-check notes
## What I could not verify
```
