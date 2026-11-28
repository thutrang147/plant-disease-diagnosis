# Provenance timeline

The order below is what makes the pre-registered claims checkable. Each row is a commit of the original working
history, exported with `git log -1 --format='%h %cI %s' <commit>` before that history was squashed into the three
submitted commits; the commit hashes cited in the README and the evaluation ledger refer to the same working
history. `tests/test_provenance.py` checks the order whenever these commits are present and skips otherwise.

The confirmatory run also records the SHA-256 of the protocol it used: `protocol_sha256` in
[`run_record.json`](../data/results/external_v3_plantseg/run_record.json), which must equal the hash of
[`protocol.json`](../data/external_v3_plantseg/protocol.json). That check works without any git history.

| Commit | Committed (UTC) | Event |
| --- | --- | --- |
| `8762743` | 2026-11-27T14:47:47+00:00 | External v3 manifest and protocol locked (P1–P3 declared) |
| `cfc4752` | 2026-11-27T14:51:31+00:00 | External v3 evaluated once (confirmatory run) |
| `828bbfe` | 2026-11-27T23:51:55+00:00 | E3 AdaBN protocol declared |
| `4a4c8ff` | 2026-11-28T00:04:37+00:00 | E3 AdaBN results |
| `0d47ea8` | 2026-11-27T23:47:43+00:00 | E2 (Mixed, 24 epochs) protocol declared |
| `7c1a280` | 2026-11-27T23:53:21+00:00 | E2 amendment declared (seeds 43–44, Field-only arm) |
| `64d749b` | 2026-11-28T00:02:33+00:00 | E2 sanity-check correction, before any amendment run |
| `6a6faca` | 2026-11-28T00:09:51+00:00 | First E2 amendment result |
