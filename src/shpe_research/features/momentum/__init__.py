"""Momentum breakout signal indicators (batch/vectorized reference forms).

All functions take 2D arrays shaped (symbols, minutes). Recursive indicators
carry across sessions; VWAP and relative volume reset/segment per session via a
(minutes,) ``session_idx`` array. Invalid states are False (signals) or NaN
(values). The incremental per-bar form is deferred.
"""
