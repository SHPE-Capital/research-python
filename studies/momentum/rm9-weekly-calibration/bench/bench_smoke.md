# E1 benchmark (real account engine, signal masks, panel loader)

- kernel: `fast`; matrix pool rule: threads (kernel releases GIL); smoke: True
- budgets: RAM 11.0 GB, disk 30.0 GB; reference equity 100000; panel: synthetic (generate_synthetic_panel)
- machine: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD, 16 threads, 17.1 GB RAM

**Step 1** deterministic rerun: True (2 candidates, 0 trades)

**Step 2** 30 x 5 sessions: build 0.037 s, disk 2.51 MB, npy read 25.1 MB/s, mmap open 0.0168 s; replay cold 19.3 / warm 1.2 ms per candidate-week; indicator-cache hit rate cold pass 0.4694, warm pass 1.0; mask CPU cold 0.0588 s vs warm 0.0004 s

## 3 calibration cell N=12, 1-week windows, 4 candidates (1 worker(s), pool serial)

| window | s/cand-window cold | warm | hit rate cold/warm | peak RAM total MB | per worker MB |
|---|---|---|---|---|---|
| early | 0.052 | 0.001 | 0.4694/1.0 | 244.0 | 6.1 |
| late | 0.222 | 0.001 | 0.4694/1.0 | 244.2 | 8.3 |

## 4 consecutive weekly cycles (week-chunk cache)

| cycle | new chunks | reused | reuse | wall s |
|---|---|---|---|---|
| 0 | 8 | 0 | 0.0% | 0.117 |
| 1 | 4 | 4 | 50.0% | 0.099 |
| 2 | 4 | 4 | 50.0% | 0.078 |

## Scale matrix (seconds per candidate-window; window = full history, extrapolated from a sample)

| N | years | build s | panel GB | 1T s/cw (RAM total/worker MB) | 2T s/cw (RAM total/worker MB) | full run h cache / no cache | weekend h cache / no cache |
|---|---|---|---|---|---|---|---|
| 12 | 0.05 | 0.06 | 0.00 | 0.13 (243.7/5.9, serial) | 0.04 (246.2/5.1, thread) | 0.0 / 0.01 | 0.0 / 0.002 |

Projection parameters: `{"weeks": 4, "policies": 3, "seeds": 3, "budget": 16, "carry_over": 0.5, "policies_share_replays": true}`; 2x rerun reserve; 24 h weekend target. With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over candidate (D6). Rates use the fastest measured thread count per cell.
