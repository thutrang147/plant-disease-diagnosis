# Run notes

- The three Sonnet reviewer cloud sessions launched by the leader failed to start (`ref_not_found`: an abbreviated
  SHA cannot be fetched with a depth-1 clone). They never ran.
- The reviewers were therefore run as three claude-sonnet-5-5 subagents with no shared conversation, each on a
  depth-1 clone of a neutral branch pinned to the snapshot commit (`grading/snapshot-a|b|c` → `bb794d7`, `da5e1ba`,
  `61c3cd1`; one commit each, remote removed). Each received the committed prompt verbatim with its label, preceded by
  a one-line setup note giving the checkout path and asking for the report as the final answer instead of
  `send_message` (subagents have no such tool).
- Reports `s0_main.md`, `s1_round1.md` and `s2_round2.md` are the reviewers' final answers, extracted
  programmatically and unedited. One run per snapshot; nothing was re-run.
- Scores went down on three criterion steps: C1 and C2 from S0 to S1 (−0.5 each) and C5 from S1 to S2 (−0.5). They
  are reported as is.
- The S2 reviewer could not verify pre-registration order because a depth-1 clone has no history (S2 C3-1, C5-3).
  The full repository keeps that history (`8762743` before `cfc4752`).
