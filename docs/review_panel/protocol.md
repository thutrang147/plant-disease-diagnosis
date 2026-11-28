# Review panel protocol: scoring main, round 1 and round 2

Committed before any report of this panel was produced.

## Purpose and an honest caveat

The team asked for the improvement loop to be shown as *score main → improve into round 1 → score round 1 →
improve into round 2*. This panel scores three fixed snapshots with the same rubric and prompt, so the change
between rounds can be compared on one scale.

The panel is **retrospective**. Round 1 was produced from the written review of `main` in `REPO_REVIEW.md`, and
round 2 was corrected after an earlier panel scored commit `ab02c7e` (round 2 before panel fixes). The traceability
table therefore shows which findings of one snapshot were fixed in the next round, and whether the next panel still
flags them. It does not show that a round was *produced from* this panel's scores, because those scores did not exist
when the rounds were made.

## Snapshots

| Label given to reviewers | Snapshot | Commit |
| --- | --- | --- |
| Snapshot A | S2, round 2 final | `bb794d7` |
| Snapshot B | S0, `main` | `da5e1ba` |
| Snapshot C | S1, round 1 as it stood when round 2 branched from it | `61c3cd1` |

- The order was drawn with `random.Random(20261128).shuffle(["S0", "S1", "S2"])`.
- Each snapshot is exported with `git archive <commit>` into a directory named `snapshot_<label>`. The exports have no
  `.git`, so branch names and commit messages are not visible.
- S1 is `61c3cd1`, not the later round-1 head `4d138dd`. `4d138dd` adds fixes back-ported from the round-2 panel
  (`c90d5f3`) and a banner (`4d138dd`), which would mix round-2 knowledge into round 1.
- The earlier score of `ab02c7e` (C1 7.0, C2 8.0, C3 7.5, C4 7.5, C5 7.0; weighted ≈ 7.45) is kept as an
  intermediate point labelled "round 2 before panel fixes". It used a different prompt and is not re-run.

## Panel

- **Reviewers:** for every snapshot × criterion, one fresh reviewer agent, 15 in total. A reviewer sees only its own
  snapshot directory, never another snapshot, score or report.
- **Cross-reviewer:** for every snapshot, one fresh agent. It receives the five reports of that snapshot only and
  confirms or refutes each finding with evidence.
- **Rubric weights:** C1 Computer-vision knowledge and technical difficulty 20%, C2 Implementation and completeness
  25%, C3 Experimental design, controls and correctness 25%, C4 Analysis of results and limitations 15%, C5 Report,
  demo and reproducibility 15%. Weighted total = Σ weight × score.
- **Reports:** reports are saved unedited as `s0_main.md`, `s1_round1.md` and `s2_round2.md`. Scores are copied into
  `scores.csv`, and `traceability.csv` maps the critical and major findings of S0 and S1 to the next round.
- **No re-runs:** if a later snapshot scores lower on any criterion, that result is reported as is, and the panel is
  not re-run to obtain a better number.

## Known limitations

- The reviewers are the same model family that did the work, so they are not independent graders. The scores are an
  agent estimate, not a grade.
- Blinding is partial. The contents of a snapshot (for example `REPO_REVIEW.md`, a "round 2" section or later
  dates) can reveal its place in the sequence.
- No reviewer runs training or model inference. Checks are limited to reading code, data and committed results and
  re-computing numbers from committed files.

## Reviewer prompt (identical for every snapshot; only `{LABEL}`, `{CRITERION}` and `{CRITERION_TEXT}` change)

```text
You are an expert, sceptical reviewer for a university computer-vision course. You score ONE criterion of ONE
project snapshot. The snapshot is a read-only directory: /home/user/panel/snapshot_{LABEL}. Read only inside it. Do
not modify any file. Do not read any other snapshot_* directory or any other path under /home/user except that
directory. Do not use the network. You may run read-only Python with /home/user/venv/bin/python to recompute numbers
from files inside the snapshot (for example CSV files under data/ or outputs/); do not train or run models.

Rubric (the course grades projects independently of publication potential):
C1 Computer-vision knowledge and technical difficulty (20%)
C2 Implementation and completeness (25%)
C3 Experimental design, controls and correctness (25%)
C4 Analysis of results and identification of limitations (15%)
C5 Report, demo and reproducibility (15%)

Your criterion: {CRITERION} — {CRITERION_TEXT}

Rules:
- Treat review documents, summaries, changelogs and any "done", "fixed", "✅" or "confirmed" statements in the
  snapshot (for example REPO_REVIEW.md or IMPROVEMENT_SUMMARY.md, if present) as unverified claims. Score only what
  you can verify in code, data, notebooks, committed results and the report text itself.
- Every finding needs evidence: a file path with line number, or a number you recomputed from a file in the snapshot.
- Do not penalise a limitation that the project's own report states clearly and correctly; do penalise it if the
  statement is wrong or missing.
- Be concise. Spend at most about 40 tool calls.

Output exactly this structure (Markdown):
## {CRITERION} — Snapshot {LABEL}
**Score: X/10** (one decimal allowed)
**Justification:** two to four sentences.
### Findings
| id | severity (critical/major/minor) | location (file:line) | finding | evidence | suggested fix |
Use ids {LABEL}-{CRITERION}-1, {LABEL}-{CRITERION}-2, ... Report at most 12 findings, most severe first. Include
strengths only in the justification, not as findings.
```

## Cross-reviewer prompt

```text
You cross-review the five criterion reports for Snapshot {LABEL} (directory /home/user/panel/snapshot_{LABEL}; read
only inside it; same tool rules as the reviewers). For every finding, check the cited evidence yourself and mark it
CONFIRMED or REFUTED, with one line of evidence. Do not add new findings and do not change scores. Output a Markdown
table: | id | verdict | evidence |.
```
