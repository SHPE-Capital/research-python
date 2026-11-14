# Momentum golden parity fixtures (format `momentum_golden.v1`)

Research writes these files (`scripts/momentum/make_golden.py`, task B6) and the trading platform replays them
(P1-6). Everything is JSON or JSON Lines, UTF-8, `\n` line ends, so TypeScript reads them with `JSON.parse` per
line and no extra library. Files are deterministic: re-running the generator reproduces them byte for byte
(`tests/unit/test_momentum_golden.py`).

Location: `tests/fixtures/momentum_golden/<scenario>/`. Scenarios today: `s1_trend_gap` (breakout + ADX + EMA),
`s2_rvol_vwap` (breakout + RVOL + VWAP), `s3_rank_size` (breakout + VWAP, ranking and sizing with ties).
All numbers are float64; JSON `null` means NaN or "not available" and must match exactly (a non-null
actual against a null expected, or the reverse, is a failure).

## Files per scenario

| File | One line is | Purpose |
|---|---|---|
| `manifest.json` | (single object) | symbols, sessions, edge cases, per-indicator tolerances, file names, row counts |
| `weekly_config.json` | (single object) | a valid `weekly_config.v1` artifact (`docs/contracts/weekly_config.v1.schema.json`): selected config, account and exit settings. Run metadata (run id, dates outside `activation`, acceptance) is placeholder |
| `bars.jsonl` | one bar event | input: first-seen values |
| `indicators.jsonl` | one `(bar index, symbol)` | expected per-bar indicator values, readiness and component signals, final signal |
| `signals.jsonl` | one true final signal | expected entry signals (subset of `indicators.jsonl`, plus ATR) |
| `orders.jsonl` | one decision bar with a signal | expected ranked and sized orders for a scripted account state |
| `ledger.jsonl` | one closed trade | optional end-to-end check: what the C4 account engine did on these bars (not needed for indicator or order parity) |

### Time and index conventions

- `i` is the global bar-column index: 0-based, one column per minute, contiguous across sessions. Bar `i` is the
  completed minute that starts at `ts`. A session boundary does **not** reset `i`.
- `session` (0-based) and `minute` (offset from the session's first column) are given for convenience.
  Sessions in the fixtures are deliberately short (12 to 40 minutes, not 390) to keep files small; an early
  close is a shorter session (`manifest.sessions[].early_close`). Only column order, session boundaries and
  session length matter to the logic; `ts` is informational (09:30 New York, January, so UTC-5).
- A **missing minute** is a `(asset_id, i)` that has no line in `bars.jsonl` (listed in `manifest.missing_bars`).
  Treat it as a bar whose open, high, low, close and volume are all NaN. It still has a row in
  `indicators.jsonl`.
- Replay bars in file order (by `i`, then `asset_id` ascending), feeding each bar to an incremental
  implementation, and compare to the matching `indicators.jsonl` row after every bar.

### `bars.jsonl`

`{"i":0,"session":0,"minute":0,"ts":"2026-01-05T14:30:00Z","asset_id":"a-0010","o":80.03,"h":80.12,"l":80.02,"c":80.11,"v":26260}`

Prices are on a 0.01 grid, `v` is an integer share count.

### `indicators.jsonl`

One row for every symbol at every `i` (including missing bars), keys in this order:
`i`, `asset_id`, the columns below for the **active components only**, then `signal`.
`manifest.indicator_columns` lists each column with `kind` (`number` or `bool`) and, for numbers,
`abs` and `rel`: the actual passes when `|actual - expected| <= abs + rel * |expected|`. Booleans compare exactly.

| Column | Meaning |
|---|---|
| `atr14`, `ready_atr14` | Wilder ATR(14) and its readiness |
| `prec_high_n{n}`, `sig_breakout`, `ready_breakout_n{n}` | breakout |
| `adx_a{a}`, `pdi_a{a}`, `mdi_a{a}`, `sig_adx`, `ready_adx_a{a}` | ADX component (`pdi`/`mdi` are +DI/-DI) |
| `ema_{fast}`, `ema_{slow}`, `sig_ema`, `ready_ema_f{f}_s{s}` | EMA trend |
| `rvol_w{w}`, `sig_rvol`, `ready_rvol_w{w}` | clock-window relative volume ratio |
| `vwap`, `sig_vwap`, `ready_vwap` | session VWAP band |
| `signal` | final entry signal (see below) |

Component `sig_*` values are the raw component signal; readiness is a separate column and is combined only in
`signal`. Numeric columns are NaN (`null`) wherever the indicator is not yet defined, so they also encode
warm-up (structural validity). `ready_*` additionally encodes the convergence warm-up of rule B4.

**Final signal** (B5): `signal = sig_breakout AND ready_atr14 AND ready_breakout_n{n}`, and for each other active
component `AND sig_<c> AND ready_<c>`. Eligibility is all true in the fixtures.

### `signals.jsonl`

`{"i":182,"session":15,"minute":2,"asset_id":"a-0011","atr":0.1196...}`: exactly the rows with `signal == true`,
with the ATR14 at that bar (used for sizing and stops). Compare the set of `(i, asset_id)` exactly and `atr` with
the `atr14` tolerance.

### `orders.jsonl` (ranked and sized orders)

Each line is one decision bar (`i`) in a session at or after `manifest.warmup_sessions`, where at least one
signal exists and `minute + 1 < session_length - exit.session_exit_minutes_before_close` (later bars never
submit). The account state is **scripted**, not simulated: `profile` cycles `fresh`, `busy`, `tight_cash`,
`entered` (the first decision is always `fresh`), and `state` holds the numbers
(`equity`, `cash_free`, `exposure_used`, `open_positions`, `pending_orders`, `entered_today`).
Account and exit settings come from `weekly_config.json` (`account`).

Algorithm (C4 steps 4 to 6 and C3 sizing, in `trade/account.py` and `trade/funded_sizing.py`):

1. `signals` are the symbols with `signal` at `i`; drop those in `state.entered_today` -> `candidates`.
2. Order candidates by descending **median daily dollar volume** over the previous `account.liquidity_lookback`
   sessions (session `s`: sessions `max(0, s-lookback) .. s-1`; day dollar volume = sum over the session of
   `close * volume`, missing bars counted as 0; median of the days, mean of the two middle values for an even
   count; 0 when `s == 0`), ties broken by ascending `asset_id` (string order). `dollar_volume_median` shows the
   score used.
3. For each candidate in order:
   - if `open_positions + pending_orders >= account.max_positions` -> rejection `max_positions`
     (`pending_orders` counts orders accepted earlier in this same decision);
   - `price = close` of bar `i`, `atr` = ATR14 at `i`; not finite or not positive -> rejection `bad_data`;
   - `limit_price = price * (1 + fill.limit_markup)`; `unit_cost = limit_price * (1 + cost_bps / 1e4)`;
     `stop_distance = exit.initial_stop_atr * atr`;
   - liquidity shares: gather the volume of the same symbol at the same `minute` in each of the previous
     `liquidity_lookback` sessions (skip a session that is shorter than `minute + 1` or whose bar is missing);
     with fewer than `liquidity_min_obs` values the cap is 0, otherwise
     `liquidity_cap_pct * median(values)` (mean of two middle values for an even count);
   - `qty = floor(min(c1..c5) + 1e-9)` with
     `c1 = per_stock_pct * equity / price`, `c2 = risk_pct * equity / stop_distance`,
     `c3 = (cash_free - reserved) / unit_cost`,
     `c4 = (total_exposure_cap * equity - exposure_used - pending_notional) / price`,
     `c5 = liquidity shares` (`size_multiplier`, when not null, multiplies c1 and c2). The binding cap is the
     smallest (first one wins ties) and is reported as `per_stock | risk | cash | exposure | liquidity`.
     `qty <= 0` -> rejection `size_zero:<cap>`;
   - otherwise emit the order and add `qty * unit_cost` to `reserved` and `qty * limit_price` to `pending_notional`
     (both start at 0 for each decision) and 1 to `pending_orders`.
4. Output: `candidates` (ranked), `orders` (`asset_id`, `rank` = 0-based position among candidates, `qty`,
   `signal_close`, `atr`, `stop_distance`, `limit_price`, `unit_cost`, `reserved`, `liquidity_shares`,
   `binding_cap`), `rejections` (`asset_id`, `reason`). Compare `qty`, `asset_id`, order of orders, `binding_cap`
   and reasons exactly; floats with abs 1e-9 plus rel 1e-9.

### `ledger.jsonl` (optional)

Closed trades from the C4 engine run over the same bars (sessions from `warmup_sessions` on, config as in
`weekly_config.json`): `entry_i`, `exit_i`, `qty`, prices, `exit_reason` (`stop | max_hold | session_end | halt |
unresolved`), `net_pnl`. The entry is at the open of bar `entry_i` (the bar after the signal bar), exits per C2.
Use it for a whole-engine replay; it is not needed for indicator or order parity.

## Indicator definitions and seeding (what the TypeScript port must reproduce)

Sources: `features/momentum/{wilder,breakout,ema_trend,vwap_band,clock_rvol,readiness}.py`. Arrays run over the
global bar index `i` per symbol; recursive indicators carry across sessions. Batch (NumPy) and incremental forms
are the same arithmetic; port as the incremental form below.

**Non-finite input resets.** If any input a recursion needs is non-finite at bar `i` (a missing bar, or `TR`
undefined because the previous close is missing), that recursion's state is cleared and its output is NaN until it
has accumulated its seed length of valid inputs again. This is why one missing minute silences a symbol for
roughly 14 (ATR), `2a-1` (ADX) or `length` (EMA) bars afterwards.

**Seeded recursive smoother** `S(x, n, alpha)`: accumulate the sum of the first `n` valid inputs; at the `n`-th
valid input set `state = sum / n` (output is defined from this bar); afterwards
`state = state + alpha * (x - state)`. Wilder uses `alpha = 1/n`; EMA uses `alpha = 2/(n+1)`.

- **True range** `TR_i = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)`; NaN at `i = 0` and whenever
  `H_i`, `L_i`, `C_i` or `C_{i-1}` is non-finite. Continuous across sessions (previous close may be the prior
  session's last bar).
- **ATR14** = `S(TR, 14, 1/14)`. First value at `i = 14` (the 14th valid TR) when no gaps.
- **ADX(a)**, `a` in {7, 14, 21}: `up_i = H_i - H_{i-1}`, `dn_i = L_{i-1} - L_i` (NaN at 0 or if either is
  non-finite); `+DM = up if up > dn and up > 0 else 0`; `-DM = dn if dn > up and dn > 0 else 0`
  (NaN when `up`/`dn` NaN). `sTR = S(TR, a, 1/a)`, `sP = S(+DM, a, 1/a)`, `sM = S(-DM, a, 1/a)`.
  `+DI = 100 * sP / sTR` and `-DI = 100 * sM / sTR` (NaN if `sTR` is not > 0 or NaN).
  `DX = 100 * |+DI - -DI| / (+DI + -DI)` (NaN if the denominator is not > 0).
  `ADX = S(DX, a, 1/a)` (seed = mean of the first `a` valid DX values, so the first ADX is at `i = 2a-1`
  without gaps). Values in 0..100.
- **ADX signal**: `ADX`, `+DI`, `-DI` all finite and `ADX - q > 1e-6 * max(1, |q|)` and `+DI - -DI > 1e-6`.
- **Breakout** `prec_high_n = max(H_{i-n} .. H_{i-1})`, the current bar excluded; NaN if fewer than `n` preceding
  bars or any of the `n` highs is non-finite. `thr = prec_high_n + b * ATR14`. Signal: `C`, `thr`, `ATR14` finite
  and `C - thr > 1e-6 * max(|C|, |thr|)`.
- **EMA(len)** = `S(C, len, 2/(len+1))`: seeded with the mean of the first `len` closes (first value at
  `i = len-1`), carried across sessions, reset by a non-finite close.
  **EMA trend signal**: `norm = (EMAfast - EMAslow) / ATR14` finite, `ATR14 > 0`, `EMAslow_{i-5}` finite,
  `norm - d > 1e-6 * max(1, |d|)` and `EMAslow_i - EMAslow_{i-5} > 1e-6 * |EMAslow_i|`.
- **Session VWAP**: per session, `cumsum(((H+L+C)/3) * V) / cumsum(V)` over bars in that session with all of
  H, L, C, V finite (a bad bar is skipped, it does not poison later bars, and its own VWAP is NaN); NaN while the
  cumulative volume is 0. Resets at each session boundary. **VWAP band signal**: `z = (C - VWAP) / ATR14` finite,
  `ATR14 > 0`, `z >= -1e-6` and `z <= e + 1e-6 * max(1, e)`.
- **Relative volume** (`w` in {3, 5, 10}), minute-of-session `mn`: `ws_i` = sum of `V` over bars `i-w+1 .. i`,
  defined only if all `w` bars are finite, lie in the same session and are consecutive columns
  (`mn_i >= w-1` and `mn_i - mn_{i-w+1} == w-1`); otherwise NaN. The denominator for bar `i` of session `k` at
  minute `mn` is the **median** of `ws` at the same minute `mn` over sessions `max(0, k-20) .. k-1` (only finite
  values; an early-close session simply has no value at its absent minutes), required to have at least
  15 finite values, else NaN (even count: mean of the two middle values). `ratio = ws / median`, NaN when the
  median is not > 0. **Signal**: `ratio` finite and `ratio >= v * (1 - 1e-6)`.

**Readiness (B4)** `ready_*` (bar index `i` counted from the first bar supplied, which is column 0 of the
panel; the platform must warm up on the same history length it was calibrated on):
`ready_atr14 = ATR14 finite and i >= 5*14`; `ready_adx_a{a} = ADX, +DI, -DI finite and i >= 5*a`;
`ready_breakout_n{n} = prec_high_n finite and ready_atr14`;
`ready_ema_f{f}_s{s} = EMAfast finite and EMAslow_{i-5} finite and ready_atr14 and i >= 5*s`;
`ready_vwap = VWAP finite and ready_atr14`; `ready_rvol_w{w} = ratio finite` (which already requires 15 prior
sessions with a complete same-clock window).

**Tolerances.** Booleans exact. Numbers: `abs + rel*|expected|` with 1e-9/1e-9 (ATR, EMA, VWAP, RVOL, preceding high)
and abs 1e-7, rel 1e-9 for ADX, +DI, -DI. The signal comparisons already include the 1e-6 relative guards above;
port the guards exactly.

## Regenerating and extending

`uv run python scripts/momentum/make_golden.py` rewrites the directory; `--check` fails if anything differs.
The generator forces the NumPy reference path (`SHPE_MOMENTUM_NUMBA=0`). A new scenario is a new `Spec` entry in
`make_golden.py`; changing the format bumps `format_version` and needs P1 and B owner sign-off (task 0-D).
