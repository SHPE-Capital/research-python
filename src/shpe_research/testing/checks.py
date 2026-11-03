"""One-call check kits for features, models, labels and strategies (C1k).

Each ``check_*`` runs the standard contract, leakage and reproducibility checks and raises ``AssertionError``
whose message names the failed check and the first offending row. Library code never imports this module.

Timing: a bar is usable at its ``ts_available``. Every kit applies the prefix-truncation idea: a value known at
time t must not change when the history after t is deleted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shpe_research.contracts.base import ContractError, to_utc_ns
from shpe_research.contracts.candidates import validate_candidates
from shpe_research.contracts.dataset import Dataset
from shpe_research.features.base import Feature, assert_no_lookahead
from shpe_research.testing.fixtures import check_bars

__all__ = ["check_feature", "check_label", "check_model", "check_strategy"]

_SESSION_WARMUP_MINUTES = 30


def _cutoffs(n: int) -> list[int]:
    return sorted({k for k in (max(1, n // 7), n // 3, n // 2, (2 * n) // 3, n - 1) if 0 < k < n})


def _one_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    first = sorted(bars["symbol"].unique())[0]
    return bars[bars["symbol"] == first].sort_values("ts").reset_index(drop=True)


# ---- features -------------------------------------------------------------------------------------------------


class _WithContext:
    """Adapts a context-reading feature to the one-argument `compute` that `assert_no_lookahead` calls."""

    def __init__(self, feature: Feature, context: dict[str, pd.DataFrame]):
        self._feature, self._context, self.name = feature, context, feature.name

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return self._feature.compute(bars, self._context)  # type: ignore[call-arg]


def check_feature(feature: Feature, bars: pd.DataFrame | None = None) -> None:
    """Run the feature battery on one symbol's bars (default: five fixture sessions).

    Checks: the output is a float Series aligned to the input index; `assert_no_lookahead` (values at
    each row do not change when later bars are removed); deterministic across calls; NaN only in a warm-up
    (the first session, plus the first 30 minutes of any later session). Context symbols a feature declares are
    taken from the fixture. Timing: the value on a row is known at that row's ts_available.
    """
    name = feature.name
    context = None
    if bars is None:
        symbols = ("AAA", *[s for s in feature.context_symbols if s != "AAA"])
        all_bars = check_bars(symbols)
        bars = _one_symbol(all_bars[all_bars["symbol"] == "AAA"])
        context = {s: all_bars[all_bars["symbol"] == s] for s in feature.context_symbols}
    bars = _one_symbol(bars)
    missing = [c for c in feature.requires if c not in bars.columns]
    if missing:
        raise AssertionError(f"check_feature[{name}] inputs: bars lack required columns {missing}")
    target: Any = _WithContext(feature, context or {}) if feature.context_symbols else feature
    values = target.compute(bars)

    if not isinstance(values, pd.Series) or not values.index.equals(bars.index):
        raise AssertionError(f"check_feature[{name}] aligned: output must be a Series aligned to the input index")
    if not pd.api.types.is_numeric_dtype(values):
        raise AssertionError(f"check_feature[{name}] dtype: expected numeric output, got {values.dtype}")
    arr = values.to_numpy(dtype=float)

    assert_no_lookahead(target, bars)  # type: ignore[arg-type]  (names the feature and the first changed row)

    again = target.compute(bars).to_numpy(dtype=float)
    same = np.isclose(arr, again, rtol=0, atol=0, equal_nan=True)
    if not same.all():
        row = int(np.argmin(same))
        raise AssertionError(
            f"check_feature[{name}] deterministic: row {row} is {arr[row]!r} then {again[row]!r} on a second call"
        )

    session = bars["ts_available"].dt.tz_convert("America/New_York").dt.date
    first_session = (session == session.iloc[0]).to_numpy()
    first_ts = bars.groupby(session, sort=False)["ts_available"].transform("min")
    since_open = (bars["ts_available"] - first_ts) / pd.Timedelta("1min")
    allowed = first_session | (since_open.to_numpy() < _SESSION_WARMUP_MINUTES)
    bad = np.isnan(arr) & ~allowed
    if bad.any():
        row = int(np.argmax(bad))
        raise AssertionError(
            f"check_feature[{name}] NaN after warm-up: row {row} (ts_available {bars['ts_available'].iloc[row]}) "
            f"is NaN though it is past the first session and the first {_SESSION_WARMUP_MINUTES} minutes of its session"
        )


# ---- models ---------------------------------------------------------------------------------------------------


def _dataset(X: pd.DataFrame, y: pd.Series) -> Dataset:
    n = len(X)
    decision = pd.Series(to_utc_ns(pd.date_range("2026-03-02 14:31", periods=n, freq="1min", tz="UTC")), index=X.index)
    meta = pd.DataFrame(
        {
            "symbol": "AAA",
            "decision_ts": decision,
            "label_end_ts": decision + pd.Timedelta(minutes=5),
            "weight": 1.0,
            "split": "train",
            "era": decision.dt.strftime("%Y-%m-%d"),
        },
        index=X.index,
    )
    return Dataset(X=X, y=y, meta=meta, label_name=str(y.name or "y"))


def _fresh(model: Any, seed: int) -> Any:
    return type(model)(name=model.name, seed=seed, **model.params)


def check_model(model: Any, X: pd.DataFrame, y: pd.Series, seed: int = 7) -> None:
    """Run the model battery on features `X` and label `y` (a 0/1 label for classification models).

    `model` is an unfitted model whose `model_type` is registered; fresh copies are built from it. Checks: the
    same seed gives identical predictions; `save` then `load_model` gives identical predictions; predictions are
    finite, and lie in [0, 1] when `task` is "classification" (so `task` agrees with what `predict` returns);
    ONNX export, when implemented, matches `predict` within 1e-5. Timing: the model sees only the rows it is
    given; the check passes it one training split.
    """
    from shpe_research.contracts.model import load_model

    name = model.model_type
    data = _dataset(X, y)
    a = _fresh(model, seed).fit(data)
    b = _fresh(model, seed).fit(data)
    pa, pb = a.predict(X), b.predict(X)
    if pa.shape != (len(X),) or not np.isfinite(pa).all():
        raise AssertionError(f"check_model[{name}] output: predict must return one finite score per row")
    differs = ~np.isclose(pa, pb, rtol=0, atol=1e-12)
    if differs.any():
        row = int(np.argmax(differs))
        raise AssertionError(f"check_model[{name}] same seed: row {row} predicts {pa[row]!r} then {pb[row]!r}")
    if a.task == "classification" and ((pa < 0) | (pa > 1)).any():
        row = int(np.argmax((pa < 0) | (pa > 1)))
        raise AssertionError(
            f"check_model[{name}] task: classification score at row {row} is {pa[row]!r}, not in [0, 1]"
        )
    if a.task == "regression" and y.nunique() <= 2 and ((pa < 0) | (pa > 1)).any():
        row = int(np.argmax((pa < 0) | (pa > 1)))
        raise AssertionError(
            f"check_model[{name}] task: declared 'regression' but trained on a binary label and scores row {row} "
            f"as {pa[row]!r} outside [0, 1]; the task ClassVar and the output disagree"
        )

    with tempfile.TemporaryDirectory() as tmp:
        a.save(Path(tmp) / "model")
        loaded = load_model(Path(tmp) / "model")
        pl = loaded.predict(X)
        off = ~np.isclose(pa, pl, rtol=0, atol=1e-12)
        if off.any():
            row = int(np.argmax(off))
            raise AssertionError(f"check_model[{name}] save/load: row {row} predicts {pa[row]!r}, reloaded {pl[row]!r}")
        _check_onnx(a, X, pa, Path(tmp) / "export")


def _check_onnx(model: Any, X: pd.DataFrame, expected: np.ndarray, out_dir: Path) -> None:
    from shpe_research.models.export_onnx import export_onnx

    try:
        path = export_onnx(model, X.head(50), out_dir)
    except NotImplementedError:
        return  # export not supported yet (M3)
    onnx_file = Path(path) / "model.onnx" if Path(path).is_dir() else Path(path)
    if not onnx_file.exists():
        return
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_file))
    rows = X.head(50)[model.manifest.feature_names].to_numpy(dtype=np.float32)
    got = np.asarray(session.run(None, {session.get_inputs()[0].name: rows})[-1])
    got = got[:, 1] if got.ndim == 2 and got.shape[1] == 2 else got.reshape(-1)
    delta = np.abs(got - expected[:50])
    if (delta > 1e-5).any():
        row = int(np.argmax(delta))
        raise AssertionError(f"check_model[{model.model_type}] onnx: row {row} differs by {delta[row]:.2e} (> 1e-5)")


# ---- labels ---------------------------------------------------------------------------------------------------


def _decision_ts(decisions: Any) -> pd.DatetimeIndex:
    if isinstance(decisions, pd.DataFrame):
        decisions = decisions["decision_ts"]
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(decisions).to_numpy(), utc=True))


def check_label(label: Any, bars: pd.DataFrame, decisions: Any) -> None:
    """Check a label on one symbol's `bars` at the given `decisions` (decision_ts values or a frame with them).

    `label.compute(bars)` returns decision_ts, label_end_ts and a column `label.name`. Checks: on every
    requested decision `label_end_ts > decision_ts`; truncation invariance (computing on bars cut at a cutoff
    gives the same value on every decision whose `label_end_ts` is inside the cut); no truncation at the data end
    (a decision whose `label_end_ts` is past the last available bar is NaN, in the full data and in any cut).
    Timing: the label of a decision uses bars up to `label_end_ts` only.
    """
    bars = _one_symbol(bars)
    wanted = _decision_ts(decisions)
    col = label.name

    def run(frame: pd.DataFrame) -> pd.DataFrame:
        out = label.compute(frame)
        out = out[out["decision_ts"].isin(wanted)] if len(wanted) else out
        return out.drop_duplicates("decision_ts").set_index("decision_ts")

    full = run(bars)
    if full.empty:
        raise AssertionError(f"check_label[{col}] inputs: none of the requested decisions are in the label output")
    bad = full["label_end_ts"] <= full.index
    if bad.any():
        ts = full.index[bad.to_numpy()][0]
        raise AssertionError(f"check_label[{col}] label_end_ts: not after decision_ts at decision {ts} "
                             f"(label_end_ts {full.loc[ts, 'label_end_ts']})")

    def check_end(out: pd.DataFrame, last: pd.Timestamp, where: str) -> None:
        past = (out["label_end_ts"] > last) & out[col].notna()
        if past.any():
            ts = out.index[past.to_numpy()][0]
            raise AssertionError(
                f"check_label[{col}] truncation: decision {ts} has label_end_ts {out.loc[ts, 'label_end_ts']} past "
                f"the last available bar ({last}) {where} but a value {out.loc[ts, col]!r}; it must be NaN"
            )

    check_end(full, bars["ts_available"].max(), "in the full data")
    for k in _cutoffs(len(bars)):
        cut = bars.iloc[:k]
        last = cut["ts_available"].max()
        part = run(cut)
        check_end(part, last, f"when bars after row {k - 1} are removed")
        inside = part[part["label_end_ts"] <= last].index.intersection(full.index)
        a, b = part.loc[inside, col].to_numpy(dtype=float), full.loc[inside, col].to_numpy(dtype=float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            ts = inside[int(np.argmin(same))]
            raise AssertionError(
                f"check_label[{col}] truncation invariance: decision {ts} is {full.loc[ts, col]!r} with the full "
                f"history but {part.loc[ts, col]!r} when bars after row {k - 1} are removed"
            )


# ---- strategies -----------------------------------------------------------------------------------------------

_SORT = ["symbol", "decision_ts", "strategy_id"]


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in [*_SORT, "action"] if c in df.columns]
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def _first_difference(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    if list(a.columns) != list(b.columns):
        return f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return f"{len(a)} rows vs {len(b)} rows"
    for col in a.columns:
        same = (a[col] == b[col]) | (a[col].isna() & b[col].isna())
        if not same.all():
            row = int(np.argmin(same.to_numpy()))
            return f"row {row} column {col!r}: {a[col].iloc[row]!r} vs {b[col].iloc[row]!r}"
    return None


def check_strategy(generator: Any, bars: pd.DataFrame | None = None) -> None:
    """Run the candidate-generator battery on `bars` (default: five fixture sessions, one symbol).

    Checks: `generator.generate(bars)` passes `validate_candidates`; every `decision_ts` is some bar's
    `ts_available` for that symbol; deterministic; truncation invariance (candidates with `decision_ts` up to a
    cutoff do not change when bars after the cutoff are removed). Timing: a candidate is known at its
    `decision_ts`, the `ts_available` of the bar it used.
    """
    name = getattr(generator, "strategy_type", type(generator).__name__)
    bars = check_bars() if bars is None else bars
    bars = bars.sort_values(["symbol", "ts"]).reset_index(drop=True)
    full = generator.generate(bars)
    try:
        validate_candidates(full)
    except ContractError as exc:
        raise AssertionError(f"check_strategy[{name}] validate_candidates: {exc}") from exc

    known = set(zip(bars["symbol"], bars["ts_available"], strict=True))
    for row, (symbol, ts) in enumerate(zip(full["symbol"], full["decision_ts"], strict=True)):
        if (symbol, ts) not in known:
            raise AssertionError(
                f"check_strategy[{name}] decision_ts: row {row} ({symbol} {ts}) is not the ts_available of any "
                f"{symbol} bar"
            )

    canon = _canon(full)
    again = _canon(generator.generate(bars))
    diff = _first_difference(canon, again)
    if diff:
        raise AssertionError(f"check_strategy[{name}] deterministic: a second call differs: {diff}")

    stamps = np.sort(bars["ts_available"].unique())
    for k in _cutoffs(len(stamps)):
        cutoff = stamps[k - 1]
        cut = bars[bars["ts_available"] <= cutoff]
        part = _canon(generator.generate(cut))
        expected = canon[canon["decision_ts"] <= cutoff].reset_index(drop=True)
        got = part[part["decision_ts"] <= cutoff].reset_index(drop=True)
        diff = _first_difference(expected, got)
        if diff:
            raise AssertionError(
                f"check_strategy[{name}] truncation invariance: candidates up to {pd.Timestamp(cutoff)} change when "
                f"later bars are removed (future data used): {diff}"
            )

