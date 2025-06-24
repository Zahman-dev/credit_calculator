# 0001 – Record architecture decision

Date: 2024-05-25

## Status
Accepted

## Context
We need a single source of truth to document significant architectural decisions. ADRs provide lightweight yet formal documentation.

## Decision
Adopt the Architecture Decision Record (ADR) process (Madr format) and store ADR files under `docs/adr/`.

## Consequences
* Developers document future critical decisions as incremental ADRs.
* Documentation evolves with the codebase. Pipeline builds (MkDocs) will include ADRs for visibility. 