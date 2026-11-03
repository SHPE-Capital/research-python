# C1m: Security master and point-in-time universe

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int–Adv | C1g, D4, D5 | W1 (optional), studies on multi-year daily data | `repo-architect` (review: `leakage-auditor`) |

## Goal

Record which symbols existed, traded and belonged to which sector on each
date, so universes are built as of the decision date and delisted names are
not silently dropped (survivorship bias).

## Owns (exclusive)

- `src/shpe_research/data/security_master.py`
- `src/shpe_research/data/universe.py` (the `as_of` path only; D4's builder unchanged)
- `tests/unit/data/test_security_master.py`

## Reads

- `src/shpe_research/data/universe.py` (D4), `data/snapshot.py` (D5)
- [adr/0011-pluggable-data-sources-security-master.md](../../adr/0011-pluggable-data-sources-security-master.md)

## Interface

- `SecurityMaster` table: `symbol`, `valid_from`, `valid_to`, `sector`,
  `listing_exchange`, `asset_status`, `source`. Stored as a snapshot artifact
  with a checksum (D5).
- `SecurityMaster.members(date, filters) -> list[str]` and
  `SecurityMaster.sector_of(symbols, date) -> pd.Series`.
- Universe config: `universe: {type: point_in_time, master: <path>, filters: {...}}`.

## Behaviour

- `members(d)` uses only rows with `valid_from <= d < valid_to`.
- A symbol delisted mid-sample stays in the universe up to `valid_to`; its
  missing bars after that are not filled.
- Sector labels are the ones valid on `d`, so W8's sector neutralization uses
  as-of sectors.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/data/test_security_master.py` | `test_members_as_of` | Hand table: a symbol listed in March is absent in February and present in April |
| | `test_delisted_kept_until_valid_to` | A delisted symbol appears in members up to its last date |
| | `test_sector_change` | `sector_of` returns the old sector before the change date and the new one after |
| | `test_checksum_verified` | A tampered master raises on load |

## Out of scope

Corporate-action adjustment (D2 audits it), fundamentals.

## Done when

Tests pass and a daily config with `universe: {type: point_in_time}` runs.
