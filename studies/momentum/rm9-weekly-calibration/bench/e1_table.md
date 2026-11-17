# E1 benchmark: time, RAM and disk per universe size and history length

Real account engine (`trade.account_fast` Numba kernel unless noted), real B5 signal masks, A6 panel chunks, synthetic prices (no data pulled). Rows marked **measured** ran on this machine; rows marked **projected** were extrapolated linearly in N from the largest measured cell because they exceed the RAM or disk budget here. Synthetic prices mean trade counts, and so replay time, may differ on real data: re-run steps 3-4 with `--panel-dir` once real chunks exist.

## Machine profile

- Windows-11-10.0.26200-SP0; AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD; 8 physical / 16 logical cores
- RAM 17.1 GB (8.6 GB free at start), scratch free 75.9 GB
- python 3.12.4, numpy 2.5.4, pandas 3.1.0; kernel `fast`, pool rule: threads (kernel releases GIL; signal masks do not)
- budgets used for the measured/projected split: RAM 12.0 GB, disk 40.0 GB

Plan parameters: `{"weeks": 52, "policies": 3, "seeds": 3, "budget": 2048, "carry_over": 0.5, "policies_share_replays": true}`; time = cutoffs x policies x seeds x budget candidate-windows, 2x rerun reserve (cache: first cutoff full, later cutoffs one new chunk for carried-over candidates, D6).

| N | years | basis | disk GB (panel) | best threads | RAM total GB / per worker MB | full run h, no cache | full run h, D6 cache | weekend job h, no cache | weekend job h, D6 cache | D6 result cache GB | pull h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 2.0 | measured | 1.28 | 1 | 1.39 / 1330 | 12107.3 | 2075.72 | 116.416 | 19.589 | 0.6836 | n/a |
| 300 | 5.0 | measured | 3.12 | 1 | 1.38 / 1348 | 35189.06 | 5999.75 | 338.356 | 56.61 | 1.6996 | n/a |
| 300 | 10.0 | measured | 4.85 | 1 | 1.39 / 1350 | 53918.31 | 9176.15 | 518.445 | 86.574 | 3.3928 | n/a |
| 500 | 2.0 | measured | 2.13 | 1 | 2.30 / 2261 | 16503.35 | 2829.39 | 158.686 | 26.702 | 0.6836 | n/a |
| 500 | 5.0 | measured | 5.20 | 1 | 2.29 / 2256 | 54318.3 | 9261.3 | 522.291 | 87.383 | 1.6996 | n/a |
| 500 | 10.0 | measured | 8.07 | 1 | 2.28 / 2247 | 94803.57 | 16134.25 | 911.573 | 152.221 | 3.3928 | n/a |
| 1000 | 2.0 | measured | 4.27 | 1 | 4.53 / 4504 | 36742.82 | 6299.32 | 353.296 | 59.449 | 0.6836 | n/a |
| 1000 | 5.0 | measured | 10.38 | 1 | 4.54 / 4511 | 76065.96 | 12969.28 | 731.403 | 122.369 | 1.6996 | n/a |
| 1000 | 10.0 | measured | 16.14 | 1 | 4.64 / 4605 | 153928.29 | 26196.46 | 1480.08 | 247.154 | 3.3928 | n/a |

Method: each measured cell replays a sample of candidates over a few week chunks on the real engine and scales to the full window (seconds per candidate-window = seconds per candidate-week x window weeks). Thread counts above one are run on a thread pool unless the table says otherwise; the Numba kernel releases the GIL but the signal-mask step does not, so thread scaling can be flat or negative.

Disk is the on-disk A6 chunk size (plain npy), which equals the panel's RAM size when fully mapped; pull time needs a measured pull rate (`--pull-rows-per-s`) and is n/a otherwise.

Step 4 week-chunk cache: cycle 0 computed 832 (config, chunk) results, later cycles 64, 64, 64 new; reuse 92%, 92%, 92%.

## Findings for the 0-F decision

- Labeling: every number above is measured on the real account engine except the cells tagged projected in
  `bench.md` (the 16-thread runs at N = 1000, projected from N = 500 because estimated RAM exceeded the 12 GB
  budget). Matrix cells were measured on 32 candidates x 4 week chunks and extrapolated; steps 3-4 used 64
  candidates and 13-week windows (not the 256 / 52-week defaults) to keep the run to about an hour. Prices are
  synthetic, so absolute replay seconds are indicative, not a real-data result.
- Time: one candidate-window costs 23 s (N = 300, 2 y) to 289 s (N = 1000, 10 y) on one thread. The full plan
  (52 cutoffs x 3 policies x 3 seeds x 2048 candidates) is out of reach on this machine: 12,107 h (N = 300,
  2 y) to 153,928 h (N = 1000, 10 y) without the D6 cache, and 2,076 h to 26,196 h with it (2x reserve).
  The only weekend-sized job (24 h) is N = 300, 2 y with the cache (19.6 h); every other cell needs a smaller
  budget, fewer seeds or policies, or more cores through processes.
- Cache: D6 cuts the projected total by about 5.8x to 5.9x; in step 4 the weekly cycle drops from 70.3 s (cold
  window) to 5.6 s with 92.3% of (config, chunk) results reused.
- Threads do not help: 8 threads were 1.1x to 1.4x slower than 1 and 16 threads 1.7x to 2.2x slower, because the signal
  mask (indicator and signal arrays, about 95% of replay CPU; the Numba account kernel is under 5%) holds the GIL.
  An 8-process pool (`pool_compare_process/`) was also slower than one thread (35.0 s vs 22.7 s at N = 300, 2 y)
  and used 8.2 GB, since each process rebuilds its own indicator cache. Speeding up the mask path is the lever,
  not more workers.
- RAM: single-thread peak is about 1.4 GB (N = 300), 2.3 GB (N = 500), 4.5 GB (N = 1000) and is flat in history
  length because only one week chunk plus a 20-session prefix is resident. Reference machine free RAM is the
  limit, not total (E6 doc: about 2.4 GB free with usual apps open), so N = 1000 needs the machine otherwise idle.
- Disk: panel 1.3 GB (N = 300, 2 y) to 16.1 GB (N = 1000, 10 y), roughly 1.6 MB per symbol-year,
  scaling linearly; the D6 result cache is under 3.4 GB in every cell. All cells fit in about 76 GB scratch.
- Step 3 "warm" equals "cold" because the harness keeps only 2 week stores, so a 13-week window rebuilds
  indicator caches on the second pass; step 2 (30 stocks x 20 sessions) shows the within-week effect: mask CPU
  1.66 s cold vs 0.05 s warm, hit rate 0.84 vs 1.0.
- Caveat: another bench process was running on this machine (other agent, main checkout) during parts of the
  run, so wall times, especially multi-thread rows, carry some contention. Rerun on an idle machine before
  treating the thread comparison as final.
