# From a minute close to a reconciled exit

**Why two data frequencies and a quote sample?** Daily bars decide which stocks can participate. Regular-session minute bars determine when each strategy would act. A quote sample measures how far an executable bid or offer could be from a reported price. Minute highs and lows alone cannot demonstrate that a resting limit order filled or that there was enough demand to sell. The program models that uncertainty; it cannot observe every historical queue.

| Component | What has to be true at the completed minute |
|---|---|
| Breakout (required) | Close exceeds the highest high of the **preceding** 10, 20, 40 or 60 bars plus a buffer of 0, 0.10 or 0.25 × 14-bar ATR. ATR is a recent typical price range in dollars. |
| EMA trend | Fast (5/8/12) minus slow (20/35/60) EMA exceeds a buffer (0/0.10/0.25 ATR); the slow EMA is rising. |
| Relative volume | A 3/5/10-minute volume window meets 1/1.5/2 times the prior same-clock baseline. |
| VWAP band | Close is at or above the session volume-weighted average and no more than 1/2/3 ATR above it. Prevents chasing far above the session average. |
| Directional ADX | ADX on 7/14/21 bars exceeds 15/20/25/30 and positive directional movement exceeds negative. |

At, for example, 10:01 ET, the 10:00–10:01 bar has finished. The runner can signal on its close; the **earliest** modeled purchase uses the next minute's opening price and modeled costs. It never buys at a trigger bar's low. A 1-second quote proxy is an assumption about friction, not a promise that a real broker receives an order at the minute boundary. Exchange holidays and early closes use the real trading calendar; missing bars are not forward-filled into trades.

## Buying and sizing

* Rank simultaneous buys by descending median dollar volume over the prior 20 sessions, ties by stable asset ID. No second successful entry into the same stock that session. An unfilled order can be nominated later if another minute independently qualifies.
* At the signal close, reserve cash and a slot for a one-minute entry capped at **close × 1.002**. Set quantity to the integer floor of the smallest cap: **8% of equity / limit price**, **0.25% of equity / (1.5 × signal ATR)**, unreserved cash / limit price, remaining capacity under **80%** gross exposure / limit price, and **1%** of the prior median same-clock share volume. That median needs at least 15 of the previous 20 sessions. A zero quantity skips the order. Up to **10** open + pending slots. An 80% cap leaves room for cash; it does not require investing the other 80%.
* At the next open, buy the whole quantity only if the modeled executable cost fits the limit. Otherwise cancel. Missing next open: reject. No low-touch fills, partial shares, shorting or leverage in this study. A 0.25% starting-stop budget is **planned risk**: a gap or an untradeable stock can cost more.

## What a sampled quote costs

Use the 60 exchange sessions **before** the archive, separated by low/medium/high prior absolute market movement. Full mode draws three dates per group, two stocks per liquidity/volatility cell where available, and one-minute intervals near the opening, midpoint and closing part of the session. Sample once per second to avoid a burst of messages dominating a window. Match the old midpoint with the offer/bid one second later; discard crossed/invalid quotes or observations more than two seconds stale. Require all three market groups in every cell/time bucket. Store every sampled window, date, symbol, success flag, aggregate base friction and displayed depth. If quote coverage is insufficient, stop.

Buy and sell friction differ. The 75th percentile of adverse one-second costs per window, then the 75th percentile across windows, gives a one-side base, floored at 2 basis points. Depth is a conservative displayed-share quartile. Historical lot-sized quote fields use the instrument's lot multiplier before Nov 3, 2025; newer fields are shares. Full-study calibration is entirely in 2016. Documented [Alpaca market data](https://docs.alpaca.markets/us/reference/stockquotesingle-1). This does not reconstruct a full order book.

For quantity q, displayed depth D and median same-clock volume V, modeled one-side **bps** = base + 2 × [max(q/D − 1, 0) + 100q/max(V,1)]. Divide bps by 10,000. For buys multiply the next open by (1 + rate); for sells multiply the raw exit price by (1 − rate). The implementation clips the rate at 95% to bound an extreme stress input, and does not add a duplicate flat spread. The 2× stress doubles both base and impact. Gross reporting adds back these simulated costs on the accepted trades; it is not a separately optimized frictionless portfolio. Statutory fees, queue priority, borrow, broker routing, partial fills and market-impact calibration to actual fills are not claimed. A fixed pre-study spread sample can become stale over ten years.

## Exits and emergency behavior

* Start the stop at paid entry price − 1.5 × entry ATR. Once a completed close reaches entry + 1 ATR, the trailing stop may rise to highest completed close − 1.5 ATR. It never moves down. There is no fixed profit target.
* The order at a tradable minute is: an already triggered portfolio halt, a gap through the existing stop at the open, a scheduled pre-close exit, a 60-minute timeout, then an intrabar low crossing the old stop. A gap uses the worse open, a normal stop uses the trigger, both with modeled sell cost. If a bar touches both the old stop and the trailing activation level, the old stop takes precedence and the ambiguity is counted.
* Compare marked account equity to that session's starting equity. At or below −1%, stop placing entries for the day and liquidate positions at the next available tradable open. Missing prints can delay an exit. A normal session starts mandatory liquidation five minutes before close (15:55 on a 16:00 close); a shortened session acts five minutes before its own close. No overnight strategy, scale-in, or partial sell.
* After a split-ratio change observed at a past close, suppress entries for 20 sessions. A >0.30 absolute log gap visible at today's open also blocks that stock that day. The ratio cannot exclude trades earlier in its own day. These are conservative quarantines, not a complete corporate-actions feed.
* If a position cannot be liquidated by the end of the session, the low-level simulator marks it as unresolved. **The research runner refuses that replay and stops; it will not book the last mark as realized proceeds.** This can stop a real run. Investigate the data and halt as such; a different valuation convention requires a separately labeled protocol.

## How to audit it

The tests include an individually priced 33-share round trip, a missed limit despite a usable bar, a gap-through stop, a missing next open, stock-identity reuse and an IPO, direct comparison of the packed entry masks with the club's original signal engine, a future-P&L perturbation and deliberately damaged output. The prior unmodified fill mode still has independent Python-versus-Numba replay tests. Every completed study reconciles dollar costs to gross/net daily equity and checks continuity of both base and stressed accounts.

For Juan's TypeScript platform, `ledger` exports one week of researched entries and exits with UTC timestamps, share counts, reasons, prices and P&L. Compare it with an **independently** produced platform ledger using the runbook. A matching Python file compared to itself is not platform validation. An unsupported `limit_markup_sampled_sip_v2` contract must not silently fall back to the older flat-bps fill model.
