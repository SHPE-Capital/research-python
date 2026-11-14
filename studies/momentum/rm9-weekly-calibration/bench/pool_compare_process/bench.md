# E1 benchmark (real account engine, signal masks, panel loader)

- kernel: `fast`; matrix pool rule: threads (kernel releases GIL; signal masks do not); smoke: False
- budgets: RAM 12.0 GB, disk 40.0 GB; reference equity 100000; panel: not run (--skip-steps)
- machine: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD, 16 threads, 17.1 GB RAM

## Scale matrix (M = measured on this machine, P = projected from measured cells, linear in N)

Seconds per candidate-window (window = full history; measured on a sample of weeks and extrapolated). RAM is the peak RSS of that run.

| N | years | basis | panel = disk GB | pull h | 8 thread(s) | full run h cache / no cache | weekend h cache / no cache | D6 result cache GB |
|---|---|---|---|---|---|---|---|---|
| 300 | 2.0 | measured | 1.28 | n/a | 35.00 s, 8.24 GB total / 1006 MB worker [M] | 3195.3 / 18637.64 | 30.155 / 179.208 | 0.6836 |

Projection parameters: `{"weeks": 52, "policies": 3, "seeds": 3, "budget": 2048, "carry_over": 0.5, "policies_share_replays": true}`; 2x rerun reserve; 24 h weekend target. With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over candidate (D6). Rates use the fastest thread count per cell.
