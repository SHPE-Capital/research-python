# E1 benchmark (real account engine, signal masks, panel loader)

- kernel: `python`; matrix pool rule: processes with panels memmapped read-only (kernel holds the GIL); smoke: True
- budgets: RAM 11.0 GB, disk 30.0 GB; reference equity 100000; panel: synthetic (generate_synthetic_panel)
- machine: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD, 16 threads, 17.1 GB RAM

**Step 1** deterministic rerun: True (2 candidates, 0 trades)

**Step 2** 30 x 5 sessions: build 0.042 s, disk 2.51 MB, npy read 23.6 MB/s, mmap open 0.015 s; replay cold 1465.3 / warm 7.4 ms per candidate-week; indicator-cache hit rate cold pass 0.4694, warm pass 1.0; mask CPU cold 5.8236 s vs warm 0.0026 s

## 3 calibration cell N=12, 1-week windows, 4 candidates (1 worker(s), pool serial)

| window | s/cand-window cold | warm | hit rate cold/warm | peak RAM total MB | per worker MB |
|---|---|---|---|---|---|
| early | 1.885 | 0.008 | 0.4694/1.0 | 173.9 | 5.5 |
| late | 1.520 | 0.002 | 0.4694/1.0 | 174.5 | 6.6 |

## 4 consecutive weekly cycles (week-chunk cache)

| cycle | new chunks | reused | reuse | wall s |
|---|---|---|---|---|
| 0 | 8 | 0 | 0.0% | 7.771 |
| 1 | 4 | 4 | 50.0% | 4.054 |
| 2 | 4 | 4 | 50.0% | 4.463 |

## Scale matrix (seconds per candidate-window; window = full history, extrapolated from a sample)

| N | years | build s | panel GB | 1T s/cw (RAM total/worker MB) | 2T s/cw (RAM total/worker MB) | full run h cache / no cache | weekend h cache / no cache |
|---|---|---|---|---|---|---|---|
| 12 | 0.05 | 0.1 | 0.00 | 4.53 (174.5/5.8, serial) | 5.24 (387.1/109.2, process) | 0.36 / 1.45 | 0.04 / 0.181 |

Projection parameters: `{"weeks": 4, "policies": 3, "seeds": 3, "budget": 16, "carry_over": 0.5, "policies_share_replays": true}`; 2x rerun reserve; 24 h weekend target. With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over candidate (D6). Rates use the fastest measured thread count per cell.
