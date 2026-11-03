# ruff: noqa: E402  (the sys.path cell must run before the imports below)
# %% [markdown]
# # Example 1: Leakage lab
#
# **Lesson.** A leak makes a result look better than it is and raises no error. Plant each one on purpose,
# measure how much it inflates the result, then catch it with the repo's checks (or learn that no check exists).
#
# **Misconception corrected.** "If the backtest number is good and the code runs, the pipeline is fine."
#
# Four leaks: feature look-ahead (part 1), preprocessing fit on all rows (part 2), snooping on the test split
# (part 3), survivorship in the universe (part 4). Runs offline on the synthetic fixture (planted effect: a weak
# short-horizon reversion). Spec: `docs/workshop/examples/01-leakage-lab.md`.
#
# Every leaky helper below lives in this notebook, never in `src/`, so the library cannot ship a leaky feature.

# %%
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = next((p for p in [_here, *_here.parents] if (p / "workshop" / "_common").is_dir()), None)
if _root is not None:  # a scratch copy (the test harness) gets the repo root from PYTHONPATH instead
    sys.path.insert(0, str(_root))
sys.path.insert(0, str(_here))

import dataclasses
from itertools import combinations

import exercises
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from workshop._common import real_data_enabled, show_table

from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import build_model
from shpe_research.data.calendar import session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.experiments.runner import ExperimentConfig, build_dataset
from shpe_research.features.base import (
    FEATURE_REGISTRY,
    Feature,
    assert_no_lookahead,
    build_feature,
    register_feature,
)
from shpe_research.pipeline import stages
from shpe_research.testing.checks import check_feature

CONFIGS = _here / "configs"
NY = "America/New_York"


# %% [markdown]
# ## Setup: the leaky helpers
#
# Each helper is a normal `Feature`. `fixed=True` gives the causal version, so the same class shows the leak and
# its repair. A feature's value on a row must be computable at that row's `ts_available` from that row and
# earlier rows only; `assert_no_lookahead` tests exactly that by deleting later bars and recomputing.

# %%
def _register_once(cls):
    """Register a notebook feature; re-running the cell (or importing twice) must not raise."""
    if cls.feature_type not in FEATURE_REGISTRY:
        register_feature(cls)
    return FEATURE_REGISTRY[cls.feature_type]


def _session_key(bars: pd.DataFrame) -> pd.Series:
    """New York session date of each bar, from ts_available minus one minute (so the 16:00 end belongs to its day)."""
    return session_date(bars["ts_available"] - pd.Timedelta(minutes=1))


@_register_once
class CenteredMean(Feature):
    """LEAK 1. Rolling mean of the close centred on t, relative to the close. Uses window // 2 FUTURE bars.

    `centered=False` is the fix: the trailing mean. Rows are bars, so a window is a number of bars.
    """

    feature_type = "centered_mean"

    def __init__(self, window: int, centered: bool = True):
        self.window, self.centered = int(window), bool(centered)

    @property
    def name(self) -> str:
        return f"centered_mean_{self.window}" + ("" if self.centered else "_trailing")

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        c = bars["close"]
        return c.rolling(self.window, center=self.centered).mean() / c - 1


@_register_once
class DayZScore(Feature):
    """LEAK 2. Z-score of the close with the WHOLE session's mean and standard deviation (known only at the close).

    `fixed=True` calls your exercise `day_zscore_expanding` (an expanding window that restarts each session).
    """

    feature_type = "day_zscore"
    requires = ("close", "ts_available")

    def __init__(self, fixed: bool = False):
        self.fixed = bool(fixed)

    @property
    def name(self) -> str:
        return "day_zscore_expanding" if self.fixed else "day_zscore_whole_day"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        if self.fixed:
            return exercises.day_zscore_expanding(bars)
        groups = bars["close"].groupby(_session_key(bars))
        return (bars["close"] - groups.transform("mean")) / groups.transform("std")


@_register_once
class NextReturn(Feature):
    """LEAK 3. The label shifted into the features: the return over the NEXT `horizon` bars.

    The hello model's label is the 30-minute forward return, so this feature is (almost) the answer.
    `fixed=True` flips the sign of the shift: the return over the PAST `horizon` bars.
    """

    feature_type = "next_return"

    def __init__(self, horizon: int = 30, fixed: bool = False):
        self.horizon, self.fixed = int(horizon), bool(fixed)

    @property
    def name(self) -> str:
        return f"{'past' if self.fixed else 'next'}_return_{self.horizon}"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        c = bars["close"]
        return c / c.shift(self.horizon) - 1 if self.fixed else c.shift(-self.horizon) / c - 1


@_register_once
class OneBarLookahead(Feature):
    """Core exercise 1: delegates to `exercises.one_bar_lookahead` (the next bar's close over this bar's)."""

    feature_type = "one_bar_lookahead"

    @property
    def name(self) -> str:
        return "one_bar_lookahead"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return exercises.one_bar_lookahead(bars)


@_register_once
class PriceLevel(Feature):
    """Not a leak by itself: log of the raw close, a non-stationary feature used in part 2."""

    feature_type = "price_level"

    @property
    def name(self) -> str:
        return "price_level"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return np.log(bars["close"])


def make_feature(name: str, fixed: bool = False) -> Feature:
    """The helper named `name` ("centered_mean", "day_zscore" or "next_return"), leaky or fixed."""
    if name == "centered_mean":
        return CenteredMean(11, centered=not fixed)
    if name == "day_zscore":
        return DayZScore(fixed=fixed)
    if name == "next_return":
        return NextReturn(30, fixed=fixed)
    raise ValueError(f"unknown helper {name!r}")


# %% [markdown]
# ## Setup: measuring a run
#
# `run_metrics` is the same pipeline as `shpe run`: features, a 30-minute forward-return sign label sampled every
# 15 minutes, a chronological 60/20/20 split by session, and a logistic model whose scaler is fitted on train rows
# inside the model. `seed` sets both the fixture history and the model seed. Nothing here touches the test split
# until part 3, which uses it on purpose to show what that costs.

# %%
def load_config(name: str) -> ExperimentConfig:
    """`configs/<name>.yaml` as an ExperimentConfig ("clean" or "leaky_feature")."""
    return ExperimentConfig(**yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")))


def with_features(cfg: ExperimentConfig, extra: list[dict]) -> ExperimentConfig:
    """A copy of `cfg` with feature entries appended."""
    return dataclasses.replace(cfg, features=[*cfg.features, *extra])


def leaky_config(name: str) -> ExperimentConfig:
    """The clean config plus one leaky helper: `centered_mean` (leaky_feature.yaml), `day_zscore` or `next_return`."""
    if name == "centered_mean":
        return load_config("leaky_feature")
    extra = {"day_zscore": {"type": "day_zscore"}, "next_return": {"type": "next_return", "horizon": 30}}[name]
    return with_features(load_config("clean"), [extra])


def _seeded(cfg: ExperimentConfig, seed: int) -> ExperimentConfig:
    return dataclasses.replace(cfg, seed=seed, data={**cfg.data, "seed": seed})


def run_metrics(cfg: ExperimentConfig, seed: int, bars: pd.DataFrame | None = None):
    """Fit the config's model on train rows. Returns (metrics by split, fitted model, dataset).

    Timing: features use rows up to each decision; the label ends 30 minutes later; the split is by session.
    """
    cfg = _seeded(cfg, seed)
    if bars is None:
        bars = stages.load(cfg.data)
    ds = build_dataset(cfg, bars)
    fitted = stages.model(ds, cfg.model, cfg.seed)
    _, metrics, _ = stages.evaluate(ds, fitted, stages.plan(cfg).sections)
    return metrics, fitted, ds


def validation_auc(cfg: ExperimentConfig, seed: int) -> float:
    """Validation AUC of one run (0.5 is no skill)."""
    return float(run_metrics(cfg, seed)[0]["validation"]["auc"])


# %% [markdown]
# ## Setup: helpers for parts 3 and 4
#
# Part 3's grid, scoring and the snooping gain (exercise 3); part 4's survivorship demo.

# %%
CANDIDATES = [
    {"type": "past_return", "window": 5},
    {"type": "past_return", "window": 15},
    {"type": "past_return", "window": 30},
    {"type": "past_return", "window": 60},
    {"type": "minute_of_day"},
]
C_GRID = (0.01, 0.1, 1.0, 10.0)


def candidate_dataset(seed: int) -> Dataset:
    """Dataset with all five candidate features on fixture `seed` (clean config's label and split)."""
    cfg = _seeded(dataclasses.replace(load_config("clean"), features=CANDIDATES), seed)
    return build_dataset(cfg, stages.load(cfg.data))


def _config_grid() -> list[tuple[tuple[str, ...], float]]:
    """Every non-empty subset of the five candidates (by column name) crossed with C_GRID: 31 x 4 = 124."""
    names = [build_feature(spec).name for spec in CANDIDATES]
    subsets = [c for k in range(1, len(names) + 1) for c in combinations(names, k)]
    return [(subset, C) for subset in subsets for C in C_GRID]


def fit_scores(ds: Dataset, columns: tuple[str, ...], C: float) -> tuple[float, float]:
    """(validation AUC, test AUC) of a logistic model on `columns`, fit on train rows only."""
    sub = dataclasses.replace(ds, X=ds.X[list(columns)])
    model = build_model("logistic", seed=0, C=C).fit(sub.subset("train"))
    val, test = sub.subset("validation"), sub.subset("test")
    return roc_auc_score(val.y, model.predict(val.X)), roc_auc_score(test.y, model.predict(test.X))


def snooping_table(seed: int, ds: Dataset | None = None) -> pd.DataFrame:
    """One row per configuration (124): columns, C, val_auc, test_auc, all measured on fixture `seed`."""
    ds = candidate_dataset(seed) if ds is None else ds
    rows = []
    for columns, C in _config_grid():
        val_auc, test_auc = fit_scores(ds, columns, C)
        rows.append({"columns": columns, "C": C, "val_auc": val_auc, "test_auc": test_auc})
    return pd.DataFrame(rows)


def selection_gain(table: pd.DataFrame, n_configs: int, n_draws: int = 300, seed: int = 0) -> float:
    """Exercise 3 (see exercises.py): test-best configuration's test AUC minus the table's mean test AUC."""
    return exercises.selection_gain(table, n_configs, n_draws=n_draws, seed=seed)




def survivorship_demo(
    seed: int = 7, n_symbols: int = 60, drop_share: float = 0.2, cost_bps_per_day: float = 14.0
) -> dict:
    """Mean daily return of buy-yesterday's-losers on all names versus the survivors, gross and net of costs.

    Timing: names are ranked on day d-1's close-to-close return and credited with day d's return, so the signal
    uses only information known at the d-1 close. Survivors are chosen with the WHOLE window's return, which is
    the leak. Returns are per day, in fractions (0.001 = 10 bps).
    """
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    bars = synthetic_bars(symbols, seed=seed)
    day = session_date(bars["ts_available"] - pd.Timedelta(minutes=1))
    close = bars.assign(day=day).sort_values("ts").groupby(["day", "symbol"])["close"].last().unstack("symbol")
    ret = close.pct_change().iloc[1:]
    total = close.iloc[-1] / close.iloc[0] - 1
    n_drop = int(round(drop_share * n_symbols))
    survivors = total.sort_values().index[n_drop:]

    def strategy(names) -> dict:
        r = ret[list(names)]
        k = max(1, int(round(0.1 * len(names))))
        picks = r.shift(1).rank(axis=1, method="first") <= k  # worst previous-day returns
        daily = r.where(picks).mean(axis=1).dropna()
        return {"gross": float(daily.mean()), "net": float(daily.mean() - cost_bps_per_day / 1e4),
                "days": int(len(daily)), "names_held": k}

    return {"all": strategy(ret.columns), "survivors": strategy(survivors), "n_survivors": len(survivors),
            "dropped": list(total.sort_values().index[:n_drop])}



if __name__ != "__main__":
    # Imported by the tests: they need the helpers above, not the walkthrough below.
    _helpers_only = True
    raise SystemExit

# %% [markdown]
# ## Part 1: feature look-ahead (20 min)
#
# The clean config's validation AUC should be a little above 0.5: the fixture plants a weak reversion and
# nothing else. The centred mean at `t` includes the next five closes, and those five minutes are the first part
# of the 30-minute label.

# %%
SEED = 7
clean_cfg = load_config("clean")
clean_metrics, clean_model, clean_ds = run_metrics(clean_cfg, SEED)
rows = [("clean", clean_metrics["validation"]["auc"], clean_metrics["validation"]["ic_t"])]

for leak in ("centered_mean", "day_zscore", "next_return"):
    m, fitted, _ = run_metrics(leaky_config(leak), SEED)
    rows.append((leak, m["validation"]["auc"], m["validation"]["ic_t"]))
    if leak == "centered_mean":
        coefs = pd.Series(fitted.pipeline_.named_steps["model"].coef_[0], index=fitted.manifest.feature_names)
        show_table(coefs.to_frame("coef (scaled)"), "centered_mean run: logistic coefficients")

table1 = pd.DataFrame(rows, columns=["config", "validation_auc", "daily_ic_t"]).set_index("config")
table1["auc_minus_clean"] = table1["validation_auc"] - table1.loc["clean", "validation_auc"]
show_table(table1, "Part 1: validation metrics, seed 7")

# %% [markdown]
# **Question.** On the fixture the centred-mean jump is large enough that nobody would believe it. A one-bar
# shift on real data would look like a plausible edge. What would make you suspicious of a small jump?

# %% [markdown]
# Now catch each leak. `assert_no_lookahead` recomputes the feature on prefixes of the history; `check_feature`
# runs that plus alignment, determinism and warm-up checks.

# %%
for leak in ("centered_mean", "day_zscore", "next_return"):
    try:
        check_feature(make_feature(leak, fixed=False))
        print(f"{leak:14s} NOT caught")
    except AssertionError as err:
        print(f"{leak:14s} caught: {str(err)[:110]}")

# %% [markdown]
# Fix them and rerun the check **and** the config (dropping a feature without rerunning the check hides the
# problem rather than fixing it). `DayZScore`'s fix is exercise 2.

# %%
for leak in ("centered_mean", "next_return", "day_zscore"):
    try:
        feature = make_feature(leak, fixed=True)
        check_feature(feature)
        print(f"{leak:14s} fixed version passes check_feature")
    except NotImplementedError:
        print(f"EXERCISE pending: {leak} fix (exercise 2, exercises.day_zscore_expanding)")

fixed_cfg = with_features(clean_cfg, [{"type": "centered_mean", "window": 11, "centered": False}])
fixed_auc = validation_auc(fixed_cfg, SEED)
print(f"trailing-mean config validation AUC {fixed_auc:.4f} (clean {clean_metrics['validation']['auc']:.4f})")

# %% [markdown]
# ### Exercise 1 (core): a one-bar look-ahead
#
# Fill in `exercises.one_bar_lookahead` (`close.shift(-1) / close - 1`). How much does AUC rise, and does the
# prefix check catch it at every cutoff?

# %%
try:
    one_bar = with_features(clean_cfg, [{"type": "one_bar_lookahead"}])
    auc = validation_auc(one_bar, SEED)
    print(f"one-bar look-ahead validation AUC {auc:.4f} vs clean {clean_metrics['validation']['auc']:.4f}")
    try:
        assert_no_lookahead(OneBarLookahead(), stages.load(clean_cfg.data).query("symbol == 'AAA'"))
        print("NOT caught")
    except AssertionError as err:
        print(f"caught: {str(err)[:110]}")
except NotImplementedError:
    print("EXERCISE pending: exercise 1 (exercises.one_bar_lookahead)")

# %% [markdown]
# ## Part 2: preprocessing fit on all rows (15 min)
#
# Fit a `StandardScaler` on all rows of `dataset.X` (train, validation and test), then the logistic regression on
# the scaled train rows. Compare with the contract version, whose scaler sees only the train rows
# (`SklearnModel.fit` builds a `Pipeline`, and the scaler only sees `dataset.subset("train")`).

# %%
def scaler_gap(ds: Dataset, C: float = 1.0) -> dict:
    """Validation AUC with the scaler fit on train rows only (the contract) versus fit on every row (the leak)."""
    train, val = ds.subset("train"), ds.subset("validation")
    contract = build_model("logistic", seed=0, C=C).fit(train)
    honest = roc_auc_score(val.y, contract.predict(val.X))
    scaler = StandardScaler().fit(ds.X)  # LEAK: sees validation and test rows
    leaky_model = LogisticRegression(C=C, max_iter=1000, random_state=0).fit(scaler.transform(train.X), train.y)
    leaky = roc_auc_score(val.y, leaky_model.predict_proba(scaler.transform(val.X))[:, 1])
    # A scale-sensitive model feels the leaked mean and spread much more than a logistic regression does.
    def knn_auc(scale_on_all: bool) -> float:
        scaler_ = StandardScaler().fit(ds.X if scale_on_all else train.X)
        knn = KNeighborsClassifier(n_neighbors=50).fit(scaler_.transform(train.X), train.y)
        return roc_auc_score(val.y, knn.predict_proba(scaler_.transform(val.X))[:, 1])

    knn_honest, knn_leaky = knn_auc(False), knn_auc(True)
    return {"honest_auc": honest, "leaky_auc": leaky, "gap": leaky - honest,
            "knn_honest_auc": knn_honest, "knn_leaky_auc": knn_leaky, "knn_gap": knn_leaky - knn_honest}


stationary = scaler_gap(clean_ds)
print(f"stationary fixture, logistic: honest {stationary['honest_auc']:.4f}, leaky {stationary['leaky_auc']:.4f}, "
      f"gap {stationary['gap']:+.4f}; k-nearest neighbours gap {stationary['knn_gap']:+.4f}")

# %% [markdown]
# The gap is near zero here, because the fixture's feature distributions do not drift. The leak is real and
# invisible on this data. Make it visible (**hypothesis:** the gap widens; we measure it, we do not assume it):
# scale each day's returns by `1 + 0.5 * day_index / n_days` (a slow volatility trend; a notebook stand-in, not a
# library function) and add the raw price level as a feature.

# %%
def add_vol_trend(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale each session's close-to-close log returns by 1 + 0.5 * day_index / n_days (OHLC rescaled with close)."""
    out = []
    for _, g in bars.groupby("symbol", sort=True):
        g = g.sort_values("ts").copy()
        day = _session_key(g)
        index = day.rank(method="dense").to_numpy() - 1
        mult = 1 + 0.5 * index / max(index.max(), 1)
        logret = np.log(g["close"]).diff().fillna(0.0).to_numpy() * mult
        close = g["close"].iloc[0] * np.exp(np.cumsum(logret))
        ratio = close / g["close"].to_numpy()
        for col in ("open", "high", "low", "close"):
            g[col] = g[col].to_numpy() * ratio
        out.append(g)
    return pd.concat(out, ignore_index=True)


trend_cfg = with_features(clean_cfg, [{"type": "price_level"}])
gaps = []
for s in (7, 8, 9):
    bars = add_vol_trend(stages.load(_seeded(trend_cfg, s).data))
    _, _, ds_trend = run_metrics(trend_cfg, s, bars=bars)
    gaps.append(scaler_gap(ds_trend))
trend = pd.DataFrame(gaps, index=[7, 8, 9])
show_table(trend, "Part 2: trending fixture + price level, scaler fit on all rows vs train only")
print(f"logistic: mean gap {trend['gap'].mean():+.4f} (stationary seed 7: {stationary['gap']:+.4f}); spread over 3 "
      f"seeds {trend['gap'].min():+.4f} to {trend['gap'].max():+.4f}")
print(f"k-nearest neighbours: mean gap {trend['knn_gap'].mean():+.4f}; spread {trend['knn_gap'].min():+.4f} to "
      f"{trend['knn_gap'].max():+.4f}")

# %% [markdown]
# **Question.** Logistic regression is nearly scale-equivariant, so a leaked mean and standard deviation only
# shift the regularisation and the intercept. Does that make the leak harmless? Which models (k-nearest neighbours,
# PCA, neural networks) would feel it more? Do not conclude that scaling on all rows is harmless: the fixture is
# stationary and markets are not. The repo prevents it by fitting the scaler inside `fit()`.

# %% [markdown]
# ## Part 3: snooping (25 min)
#
# Try many configurations, report the one with the best **test** AUC (A), and compare with the one chosen on
# **validation** (B). Then run A and B unchanged on ten fresh histories (seeds 101-110): same planted effect,
# new noise. The grid below is every non-empty subset of five candidate features crossed with four values of `C`
# (124 configurations); part 3 uses the 62 with `C` in {0.01, 1.0} and cuts them to 40.

# %%
ds7 = candidate_dataset(7)
table7 = snooping_table(7, ds7)
pool = table7[table7["C"].isin([0.01, 1.0])].sample(40, random_state=0)
a, b = pool.loc[pool["test_auc"].idxmax()], pool.loc[pool["val_auc"].idxmax()]
print(f"A (best test AUC): test {a.test_auc:.4f}, val {a.val_auc:.4f}, C={a.C}, {len(a["columns"])} features")
print(f"B (best val AUC):  test {b.test_auc:.4f}, val {b.val_auc:.4f}, C={b.C}, {len(b["columns"])} features")
print(f"typical configuration's test AUC (mean of the 40): {pool.test_auc.mean():.4f}")
print(f"trials counted: {len(pool)}  <- the n_trials a deflated Sharpe ratio would need (E2)")

# %%
fresh = []
for s_ in range(101, 111):
    ds = candidate_dataset(s_)
    test_aucs = [fit_scores(ds, c["columns"], c["C"])[1] for _, c in pool.iterrows()]
    fresh.append({"seed": s_, "A_test_auc": fit_scores(ds, a["columns"], a["C"])[1],
                  "B_test_auc": fit_scores(ds, b["columns"], b["C"])[1], "typical_test_auc": float(np.mean(test_aucs))})
fresh = pd.DataFrame(fresh).set_index("seed")
fresh["A_over_typical"] = fresh["A_test_auc"] - fresh["typical_test_auc"]
show_table(fresh, "Part 3: A and B rerun unchanged on fresh histories (test split)")
gain_seed7 = a["test_auc"] - pool["test_auc"].mean()
print(f"A's advantage over a typical configuration: seed 7 test {gain_seed7:+.4f}; fresh seeds mean "
      f"{fresh['A_over_typical'].mean():+.4f} (sd {fresh['A_over_typical'].std():.4f} over 10 seeds)")
print(f"snooping bias (advantage that does not survive) = {gain_seed7 - fresh['A_over_typical'].mean():+.4f}")
print(f"A minus B on fresh seeds: mean {(fresh.A_test_auc - fresh.B_test_auc).mean():+.4f}")
print("Compare ADVANTAGES, not raw AUC: a fresh history has its own noise level, so A's raw AUC there can be higher "
      "or lower than on seed 7 for reasons unrelated to snooping.")

# %% [markdown]
# **Question.** A teammate says "A's features are also the best on validation". Does that rescue A? (Hint: both
# splits come from the same 125 sessions and the same 40 trials.)
#
# ### Exercise 3 (core): how snooping grows with the number of configurations
#
# Implement `exercises.selection_gain` and call it for 10, 40 and 100 configurations on the 124-row table.

# %%
try:
    gains = {n: selection_gain(table7, n) for n in (10, 40, 100)}
    show_table(pd.Series(gains, name="gain").to_frame(), "Part 3, exercise 3: test-best minus typical, by trials")
except NotImplementedError:
    print("EXERCISE pending: exercise 3 (exercises.selection_gain)")

# %% [markdown]
# ## Part 4: survivorship (25 min)
#
# Generate 60 symbols (`S00`..`S59`) on the default window. Remove the 20% with the worst return over the whole
# window (a stand-in for "no longer listed, or below $10, at the end"). Strategy: each day buy the 10% of names
# with the worst previous-day return and hold one day. **Net of costs:** each name-day pays the platform fill
# defaults, 2 bps half-spread plus 5 bps slippage on entry and again on exit (14 bps per day; commission of
# $0.005 per share is left out, and full daily turnover is assumed, so net is slightly pessimistic). The fixture
# has no planted cross-sectional effect (C1o adds one), so whatever the strategy earns on all 60 names is the
# baseline; the survivors' extra is bias.

# %%
surv = survivorship_demo()
frame = pd.DataFrame({"all 60": surv["all"], f"{surv['n_survivors']} survivors": surv["survivors"]}).T
frame[["gross", "net"]] *= 1e4
show_table(frame.rename(columns={"gross": "gross bps/day", "net": "net bps/day"}), "Part 4: mean daily return, seed 7")
print(f"bias (gross): {1e4 * (surv['survivors']['gross'] - surv['all']['gross']):+.1f} bps/day, all of it from the "
      "choice of names, none from the strategy")

# %% [markdown]
# **Question.** Does the bias depend on how you remove the names ("worst 20% of window return" versus "price
# ends below a threshold" versus removing 20% at random)? Try the random removal and see.
#
# ### The snapshot universe (demonstration, real data only)
#
# `configs/universe/workshop_v0.yaml` measures its screens over the 60 sessions before `as_of`, and the snapshot
# covers the same window, so any backtest on the snapshot uses names chosen with knowledge of that period. This
# cell only reads the universe config; it never opens snapshot data. This lab measures the bias on the fixture
# only; do not treat the discussion of the snapshot as a measured result.

# %%
if real_data_enabled():
    path = Path(_root or ".") / "configs" / "universe" / "workshop_v0.yaml"
    universe = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"universe {universe['name']}: as_of {universe['as_of']}, screens over {universe['window_sessions']} "
          f"sessions before it: {universe['screens']}")
    print("The screens look at the same sessions the backtest uses; the fix is a window that ends before the backtest "
          "starts, plus a point-in-time security master (C1m).")
else:
    print("real-data cell skipped (set SHPE_WORKSHOP_REAL_DATA=1 to run it)")

# %% [markdown]
# ## Wrap-up: fill in the table
#
# | Leak | Clean | Leaky | Difference | Caught by |
# |---|---|---|---|---|
# | Feature look-ahead | part 1 | part 1 | part 1 | `assert_no_lookahead` / `check_feature` |
# | Preprocessing on all rows | part 2 | part 2 | part 2 | the model contract (scaler inside `fit`) |
# | Snooping | fresh-seed mean | seed-7 test AUC | part 3 bias | nothing in code: the protocol, trial count, E2 |
# | Survivorship | all names | survivors | part 4 bias | D4's `as_of`; C1m |
#
# ## What to claim next
#
# - **E2** (deflated Sharpe, already merged): run it on part 3's best configuration with `n_trials = 40`.
# - **V1 / C1f** purged walk-forward: purge overlapping labels and show how much validation AUC falls.
# - **C1m** point-in-time security master (survivorship fix); **M2** trial counting.
# - **C1k** check kits: add `check_feature` to your own feature PR.
