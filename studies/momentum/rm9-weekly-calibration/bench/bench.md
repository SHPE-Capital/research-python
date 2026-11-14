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
- ram_available_gb: 3.5
- scratch_free_gb: 78.3

Peak process RSS: 286.7 MB

## Steps

| step | status | key numbers |
|---|---|---|
| 1 one stock, few sessions (correctness) | OK | cold 0.06 ms/cand, warm 0.05 ms/cand, hit rate 0.969, read 4.5 MB/s, 5 symbol-sessions |
| 2 30 stocks x 20 sessions (load + indicator profile) | OK | cold 9.11 ms/cand, warm 9.39 ms/cand, hit rate 0.984, read 540.5 MB/s, 600 symbol-sessions |
| 3 300-stock early+late calibration (>=256 candidates, finalists, account replay) | BLOCKED | C4 account simulator (shpe_research.trade.account) and A6 panel not implemented |
| 4 consecutive weekly cycles (incremental reuse) | BLOCKED | C4 account simulator (shpe_research.trade.account) and A6 panel not implemented |

## Projection (300 symbols, 2x rerun reserve)

| cache | weekend job (h) | vs 24h | full run (h) |
|---|---|---|---|
| cold | 0.082 | 0.34% | 76.39 |
| warm | 0.084 | 0.35% | 78.72 |

Assumptions: `{"n_symbols": 300, "history_sessions": 252, "candidates": 256, "weeks": 52, "policies": 3, "seeds": 3, "rerun_reserve": 2.0, "note": "weeks/policies/seeds/history are PLACEHOLDERS until 0-F fixes scope; proxy kernel, not C4"}`

## Scale matrix (STAND-IN replay: a FLOOR, not a C4 measurement)

N / years / search budget source: PLACEHOLDERS (0-G protocol config not present). Linear extrapolation from per-symbol-session costs of the proxy kernel and the A6 synthetic builder; real C4 replay will be slower. Pull time needs `--pull-rows-per-s` (no data was pulled); otherwise n/a.

| N | years | threads | panel GB | parquet GB | pull h | RAM/worker MB | RAM total GB | weekly h no cache | weekly h D6 | full run h no cache | full run h D6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 2 | 1 | 1.3 | 3.2 | n/a | 182 | 0.18 | 0.17 | 0.17 | 159.8 | 157.5 |
| 300 | 2 | 8 | 1.3 | 3.2 | n/a | 182 | 1.46 | 0.04 | 0.03 | 36.5 | 30.4 |
| 300 | 2 | 16 | 1.3 | 3.2 | n/a | 185 | 2.95 | 0.04 | 0.04 | 40.1 | 34.1 |
| 300 | 5 | 1 | 3.2 | 8.1 | n/a | 182 | 0.18 | 0.43 | 0.42 | 399.5 | 393.6 |
| 300 | 5 | 8 | 3.2 | 8.1 | n/a | 182 | 1.46 | 0.10 | 0.08 | 91.2 | 76.0 |
| 300 | 5 | 16 | 3.2 | 8.1 | n/a | 185 | 2.95 | 0.11 | 0.09 | 100.2 | 85.3 |
| 300 | 10 | 1 | 6.4 | 16.2 | n/a | 182 | 0.18 | 0.85 | 0.84 | 799.0 | 787.2 |
| 300 | 10 | 8 | 6.4 | 16.2 | n/a | 182 | 1.46 | 0.19 | 0.16 | 182.4 | 151.8 |
| 300 | 10 | 16 | 6.4 | 16.2 | n/a | 185 | 2.95 | 0.21 | 0.18 | 200.5 | 170.5 |
| 500 | 2 | 1 | 2.1 | 5.4 | n/a | 190 | 0.19 | 0.28 | 0.28 | 266.3 | 262.4 |
| 500 | 2 | 8 | 2.1 | 5.4 | n/a | 191 | 1.53 | 0.06 | 0.05 | 60.8 | 50.7 |
| 500 | 2 | 16 | 2.1 | 5.4 | n/a | 193 | 3.09 | 0.07 | 0.06 | 66.8 | 56.9 |
| 500 | 5 | 1 | 5.3 | 13.5 | n/a | 190 | 0.19 | 0.71 | 0.70 | 665.8 | 656.0 |
| 500 | 5 | 8 | 5.3 | 13.5 | n/a | 191 | 1.53 | 0.16 | 0.14 | 152.0 | 126.6 |
| 500 | 5 | 16 | 5.3 | 13.5 | n/a | 193 | 3.09 | 0.18 | 0.15 | 167.1 | 142.1 |
| 500 | 10 | 1 | 10.6 | 26.9 | n/a | 190 | 0.19 | 1.42 | 1.40 | 1331.7 | 1312.0 |
| 500 | 10 | 8 | 10.6 | 26.9 | n/a | 191 | 1.53 | 0.32 | 0.27 | 304.0 | 253.1 |
| 500 | 10 | 16 | 10.6 | 26.9 | n/a | 193 | 3.09 | 0.36 | 0.30 | 334.1 | 284.1 |
| 1000 | 2 | 1 | 4.3 | 10.8 | n/a | 211 | 0.21 | 0.57 | 0.56 | 532.7 | 524.9 |
| 1000 | 2 | 8 | 4.3 | 10.8 | n/a | 212 | 1.69 | 0.13 | 0.11 | 121.6 | 101.4 |
| 1000 | 2 | 16 | 4.3 | 10.8 | n/a | 214 | 3.43 | 0.14 | 0.12 | 133.6 | 113.8 |
| 1000 | 5 | 1 | 10.6 | 26.9 | n/a | 211 | 0.21 | 1.42 | 1.40 | 1331.7 | 1312.1 |
| 1000 | 5 | 8 | 10.6 | 26.9 | n/a | 212 | 1.69 | 0.32 | 0.27 | 304.0 | 253.2 |
| 1000 | 5 | 16 | 10.6 | 26.9 | n/a | 214 | 3.43 | 0.36 | 0.30 | 334.1 | 284.2 |
| 1000 | 10 | 1 | 21.3 | 53.9 | n/a | 211 | 0.21 | 2.85 | 2.80 | 2663.4 | 2624.1 |
| 1000 | 10 | 8 | 21.3 | 53.9 | n/a | 212 | 1.69 | 0.65 | 0.54 | 608.0 | 506.2 |
| 1000 | 10 | 16 | 21.3 | 53.9 | n/a | 214 | 3.43 | 0.71 | 0.61 | 668.2 | 568.2 |

Measured thread sweep (stand-in): `{"1": {"status": "OK", "wall_s": 9.560287699999208, "worker_peak_rss_mb": 169.2, "speedup": 1.0}, "8": {"status": "OK", "wall_s": 1.8433981000007407, "worker_peak_rss_mb": 169.6, "speedup": 5.186230635691426}, "16": {"status": "OK", "wall_s": 2.06955089999974, "worker_peak_rss_mb": 171.9, "speedup": 4.619498703801104}}`
