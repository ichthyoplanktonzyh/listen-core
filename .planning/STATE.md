# State

> Updated: 2026-07-31 CST

## Position

- Repository: `ichthyoplanktonzyh/listen-core`
- Default implementation owner: Codex
- Consumer: `ichthyoplanktonzyh/listen-app`
- API generation: `1`
- Contract version: `1.1.0` (unreleased)
- Runtime/workspace version: `0.7.0`
- Published split baseline: `v0.7.0-split.1`

## Current Work

Issue #103 now has an internal durable Content Preparation parent behind the
single content-level learner action. It reuses one unambiguous available
Subtitle Text Track or owns a deterministic ASR child, freezes the exact
language/text snapshot, and then delegates to recommended Foundation
Preparation for WordTimeline, ChunkTimeline, SenseGroup, and the derived A/B
readiness views. Child provenance, single-flight, input replacement,
cancellation, retry, and restart recovery are durable. Model and timeline
choices remain internal; SoundLine is independent and observed phone evidence
(view C) still requires a separate learner decision.

## Established Boundaries

- Core owns canonical OpenAPI, Rust/Python backend, schemas, and runtime artifacts.
- App owns Flutter, client compatibility, lock file, and final product assembly.
- Cross-repo synchronization uses immutable releases and explicit lock updates.
- Imported monorepo planning is frozen under `archive/monorepo-baseline/`.
- Root `CHANGELOG.md` is updated only by a release owner from merged PRs.
- Manual timeline editing and generic technical-resource destinations are not
  part of the learner journey. Existing core lifecycle/write contracts remain
  compatibility surfaces until a separately coordinated breaking migration.

## Known Operational Constraint

GitHub-hosted Actions cannot currently start because of account billing/spending
state. Local validation is required and CI red-with-zero-steps is infrastructure,
not code evidence.

## Next

1. Complete the app-originated contract/handoff for the content-level prepare
   action without exposing model, resource, or timeline controls.
2. Coordinate removal of manual timeline editing from the app; deprecate or
   remove published core write routes only through an explicit compatibility
   decision.
3. Split core issue #80 into a production-model slice followed by an
   app-originated contract slice; do not promote the English-centric spike
   variant enums into the multilingual contract.
4. Publish immutable `1.1.0` contract/runtime artifacts only after owner
   approval, then complete the `listen-app` lock and DTO handoff.
5. Run the Apple Silicon short-audio cascade smoke only with explicit model
   download/live-inference authorization.
6. Remove the exact `local-runtime` HTTP route debt allowlist one route module
   at a time.
