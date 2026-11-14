# E1 benchmark (PROVISIONAL: proxy replay kernel on synthetic bars)

## Machine

- os: Windows-11-10.0.26200-SP0
- python: 3.12.4
- numpy: 2.5.4
- pandas: 3.1.0
- cpu: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD
- cores_physical: 8
- cores_logical: 16
- ram_total_gb: 17.1
- ram_available_gb: 3.4
- scratch_free_gb: 80.7

Peak process RSS: 276.6 MB

## Steps

| step | status | key numbers |
|---|---|---|
| 1 one stock, few sessions (correctness) | OK | cold 0.06 ms/cand, warm 0.05 ms/cand, hit rate 0.969, read 4.8 MB/s, 5 symbol-sessions |
| 2 30 stocks x 20 sessions (load + indicator profile) | OK | cold 6.08 ms/cand, warm 5.78 ms/cand, hit rate 0.984, read 548.2 MB/s, 600 symbol-sessions |
| 3 300-stock early+late calibration (>=256 candidates, finalists, account replay) | BLOCKED | C4 account simulator (shpe_research.trade.account) and A6 panel not implemented |
| 4 consecutive weekly cycles (incremental reuse) | BLOCKED | C4 account simulator (shpe_research.trade.account) and A6 panel not implemented |

## Projection (300 symbols, 2x rerun reserve)

| cache | weekend job (h) | vs 24h | full run (h) |
|---|---|---|---|
| cold | 0.054 | 0.23% | 50.98 |
| warm | 0.052 | 0.22% | 48.48 |

Assumptions: `{"n_symbols": 300, "history_sessions": 252, "candidates": 256, "weeks": 52, "policies": 3, "seeds": 3, "rerun_reserve": 2.0, "note": "weeks/policies/seeds/history are PLACEHOLDERS until 0-F fixes scope; proxy kernel, not C4"}`
