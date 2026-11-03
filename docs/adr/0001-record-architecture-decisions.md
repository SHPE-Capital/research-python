# 0001. Record architecture decisions

- Status: Proposed
- Date: 2026-11-02
- Tasks: Wave 0 (lead)

## Context

The repo is moving from a single-runner hello world (C0) to a library used by
three strategy teams, a workshop, and several people and coding agents working
in parallel worktrees. Structural choices (layers, registries, config format,
parity rules) will be questioned repeatedly. Without a record, each question
reopens the debate, and agents have no stable reference for why the code is
shaped the way it is.

## Decision

We will record each significant architectural decision as a numbered Markdown
file in `docs/adr/`, using [0000-template.md](0000-template.md): Context,
Decision, Consequences, Alternatives considered, Status. An accepted ADR is not
edited; a decision is changed by a new ADR that supersedes the old one.

A decision is significant if it constrains more than one package or team, is
expensive to reverse, or changes a public contract (`docs/contracts.md`, a
registry key, a type in the shared-type table of
[../handoff/README.md](../handoff/README.md)).

## Consequences

- Agents and new members read the ADRs instead of reverse-engineering intent
  from code. `AGENTS.md` and the `repo-architect` agent point to them.
- Writing an ADR takes time. The rule above limits them to decisions that
  matter.
- [../README.md](../README.md) treats ADRs as authoritative, so other docs must
  be updated when an ADR is accepted.

## Alternatives considered

- **Design docs only.** Design docs describe the current shape and change with
  it; they lose the reasons and the rejected options. We keep both: design docs
  for the what, ADRs for the why.
- **Decisions in PR descriptions.** Hard to find later, and invisible to an
  agent working offline in a worktree.
