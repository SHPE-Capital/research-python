# E1 benchmark (real account engine, signal masks, panel loader)

- kernel: `fast`; matrix pool rule: threads (kernel releases GIL); smoke: True
- budgets: RAM 12.0 GB, disk 40.0 GB; reference equity 100000; panel: synthetic (generate_synthetic_panel)
- machine: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD, 16 threads, 17.1 GB RAM

**Step 1** deterministic rerun: True (2 candidates, 0 trades)

**Step 2** 30 x 5 sessions: build 0.043 s, disk 2.51 MB, npy read 27.3 MB/s, mmap open 0.0128 s; replay cold 19.5 / warm 0.2 ms per candidate-week; indicator-cache hit rate cold pass 0.4694, warm pass 1.0; mask CPU cold 0.0632 s vs warm 0.0003 s

## 3 calibration cell N=12, 1-week windows, 4 candidates (1 worker(s), pool serial)

| window | s/cand-window cold | warm | hit rate cold/warm | peak RAM total MB | per worker MB |
|---|---|---|---|---|---|
| early | 0.047 | 0.000 | 0.4694/1.0 | 243.3 | 6.4 |
| late | 0.156 | 0.000 | 0.4694/1.0 | 244.9 | 6.2 |

## 4 consecutive weekly cycles (week-chunk cache)

| cycle | new chunks | reused | reuse | wall s |
|---|---|---|---|---|
| 0 | 8 | 0 | 0.0% | 0.096 |
| 1 | 4 | 4 | 50.0% | 0.077 |
| 2 | 4 | 4 | 50.0% | 0.063 |

## Scale matrix (M = measured on this machine, P = projected from measured cells, linear in N)

Seconds per candidate-window (window = full history; measured on a sample of weeks and extrapolated). RAM is the peak RSS of that run.

| N | years | basis | panel = disk GB | pull h | 1 thread(s) | 2 thread(s) | full run h cache / no cache | weekend h cache / no cache | D6 result cache GB |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.05 | measured | 0.00 | n/a | 0.12 s, 0.24 GB total / 5 MB worker [M] | 0.04 s, 0.24 GB total / 4 MB worker [M] | 0.0 / 0.01 | 0.0 / 0.002 | 0.0 |

Projection parameters: `{"weeks": 4, "policies": 3, "seeds": 3, "budget": 16, "carry_over": 0.5, "policies_share_replays": true}`; 2x rerun reserve; 24 h weekend target. With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over candidate (D6). Rates use the fastest thread count per cell.
