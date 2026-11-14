# E1 benchmark (real account engine, signal masks, panel loader)

- kernel: `fast`; matrix pool rule: threads (kernel releases GIL; signal masks do not); smoke: False
- budgets: RAM 12.0 GB, disk 40.0 GB; reference equity 100000; panel: synthetic (generate_synthetic_panel)
- machine: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD, 16 threads, 17.1 GB RAM

**Step 1** deterministic rerun: True (8 candidates, 19 trades)

**Step 2** 30 x 20 sessions: build 0.173 s, disk 9.53 MB, npy read 20.0 MB/s, mmap open 0.0572 s; replay cold 7.6 / warm 0.5 ms per candidate-week; indicator-cache hit rate cold pass 0.8415, warm pass 1.0; mask CPU cold 1.6585 s vs warm 0.0538 s

## 3 calibration cell N=300, 13-week windows, 64 candidates (1 worker(s), pool serial)

| window | s/cand-window cold | warm | hit rate cold/warm | peak RAM total MB | per worker MB |
|---|---|---|---|---|---|
| early | 1.511 | 1.543 | 0.8415/0.8415 | 1675.0 | 1438.2 |
| late | 1.663 | 1.792 | 0.8415/0.8415 | 1653.3 | 1414.9 |

## 4 consecutive weekly cycles (week-chunk cache)

| cycle | new chunks | reused | reuse | wall s |
|---|---|---|---|---|
| 0 | 832 | 0 | 0.0% | 70.271 |
| 1 | 64 | 768 | 92.3% | 5.639 |
| 2 | 64 | 768 | 92.3% | 5.737 |
| 3 | 64 | 768 | 92.3% | 5.598 |

## Scale matrix (M = measured on this machine, P = projected from measured cells, linear in N)

Seconds per candidate-window (window = full history; measured on a sample of weeks and extrapolated). RAM is the peak RSS of that run.

| N | years | basis | panel = disk GB | pull h | 1 thread(s) | 8 thread(s) | 16 thread(s) | full run h cache / no cache | weekend h cache / no cache | D6 result cache GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 2.0 | measured | 1.28 | n/a | 22.74 s, 1.39 GB total / 1330 MB worker [M] | 30.79 s, 2.29 GB total / 278 MB worker [M] | 46.05 s, 3.63 GB total / 224 MB worker [M] | 2075.72 / 12107.3 | 19.589 / 116.416 | 0.6836 |
| 300 | 5.0 | measured | 3.12 | n/a | 66.09 s, 1.38 GB total / 1348 MB worker [M] | 71.54 s, 2.33 GB total / 287 MB worker [M] | 114.84 s, 3.65 GB total / 226 MB worker [M] | 5999.75 / 35189.06 | 56.61 / 338.356 | 1.6996 |
| 300 | 10.0 | measured | 4.85 | n/a | 101.26 s, 1.39 GB total / 1350 MB worker [M] | 125.39 s, 2.29 GB total / 281 MB worker [M] | 208.63 s, 4.07 GB total / 252 MB worker [M] | 9176.15 / 53918.31 | 86.574 / 518.445 | 3.3928 |
| 500 | 2.0 | measured | 2.13 | n/a | 30.99 s, 2.30 GB total / 2261 MB worker [M] | 40.12 s, 3.89 GB total / 482 MB worker [M] | 69.39 s, 6.34 GB total / 394 MB worker [M] | 2829.39 / 16503.35 | 26.702 / 158.686 | 0.6836 |
| 500 | 5.0 | measured | 5.20 | n/a | 102.01 s, 2.29 GB total / 2256 MB worker [M] | 123.06 s, 3.99 GB total / 495 MB worker [M] | 206.71 s, 5.86 GB total / 364 MB worker [M] | 9261.3 / 54318.3 | 87.383 / 522.291 | 1.6996 |
| 500 | 10.0 | measured | 8.07 | n/a | 178.04 s, 2.28 GB total / 2247 MB worker [M] | 231.01 s, 4.06 GB total / 503 MB worker [M] | 408.62 s, 6.17 GB total / 384 MB worker [M] | 16134.25 / 94803.57 | 152.221 / 911.573 | 3.3928 |
| 1000 | 2.0 | measured | 4.27 | n/a | 69.00 s, 4.53 GB total / 4504 MB worker [M] | 98.18 s, 7.02 GB total / 874 MB worker [M] | 138.78 s, 12.61 GB total / 788 MB worker [P] | 6299.32 / 36742.82 | 59.449 / 353.296 | 0.6836 |
| 1000 | 5.0 | measured | 10.38 | n/a | 142.85 s, 4.54 GB total / 4511 MB worker [M] | 185.88 s, 8.01 GB total / 998 MB worker [M] | 346.94 s, 12.61 GB total / 788 MB worker [P] | 12969.28 / 76065.96 | 122.369 / 731.403 | 1.6996 |
| 1000 | 10.0 | measured | 16.14 | n/a | 289.08 s, 4.64 GB total / 4605 MB worker [M] | 370.54 s, 7.86 GB total / 966 MB worker [M] | 693.88 s, 12.61 GB total / 788 MB worker [P] | 26196.46 / 153928.29 | 247.154 / 1480.08 | 3.3928 |

Projection parameters: `{"weeks": 52, "policies": 3, "seeds": 3, "budget": 2048, "carry_over": 0.5, "policies_share_replays": true}`; 2x rerun reserve; 24 h weekend target. With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over candidate (D6). Rates use the fastest thread count per cell.
