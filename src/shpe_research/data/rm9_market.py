"""Read-only SIP archive and deterministic offline source. Storage is content checked; never places orders."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.data.alpaca import AlpacaDataClient
from shpe_research.data.calendar import get_calendar
from shpe_research.data.rm9_io import digest, file_hash, pin, read, write

FIELDS = ["o", "h", "l", "c", "v"]
MASTER = ["asset_id", "symbol", "start", "end", "kind", "exchange", "lot"]
EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN", "NYSEARCA", "ARCA", "BATS"}
NAMES = re.compile(r"\b(ETF|ETN|fund|trust|warrants?|rights?|units?|preferred|notes?|bonds?)\b|%", re.I)


def ordinal(s):
    return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:4], "big")


def data_id(p):
    keys = (
        "mode",
        "start",
        "end",
        "universe",
        "screen_days",
        "warmup",
        "min_price",
        "min_dollar_volume",
        "min_prior_minute_coverage",
        "seed",
        "quote_days",
        "quote_draws_per_regime",
        "quote_symbol_samples",
        "quote_min_windows",
        "quote_p",
        "impact_bps",
        "turnover_floor_bps",
    )
    return digest({"schema": "rm9.market.2", **{k: p[k] for k in keys}})


def calendar_dates(p):
    cal = get_calendar()
    weeks = cal.sessions(p["start"], p["end"])
    prefix = cal.previous_sessions(weeks[0], p["screen_days"] + p["quote_days"] + 3)
    return prefix.append(weeks)


def demo_master(p):
    dates = calendar_dates(p)
    start = pd.Timestamp(p["start"])
    # Nine initial eligible names; one delists midweek. A replacement IPO seasons late.
    rows = []
    for k in range(10):
        lo = dates[0] if k != 9 else start - pd.Timedelta(days=43)
        hi = pd.Timestamp(p["end"]) if k != 8 else start + pd.Timedelta(days=51)
        rows.append((f"demo-{k:02}", f"DEMO{k:02}", str(lo.date()), str(hi.date()), "common", "NYSE", 100))
    return pd.DataFrame(rows, columns=MASTER)


def candidate_snapshot(client):
    """Alpaca's present directory is an approximate candidate list, not a historical security master."""
    raw = [
        row
        for status in ("active", "inactive")
        for row in client._get(
            "https://paper-api.alpaca.markets/v2/assets", {"asset_class": "us_equity", "status": status}
        )
    ]
    counts = Counter(x["symbol"] for x in raw)
    return pd.DataFrame(
        [
            (x["id"], x["symbol"], "1900-01-01", "2099-12-31", "common", x["exchange"], 100)
            for x in raw
            if x["exchange"] in EXCHANGES
            and counts[x["symbol"]] == 1
            and re.fullmatch(r"[A-Z]{1,5}([.-][A-Z])?", x["symbol"])
            and not NAMES.search(x.get("name", ""))
            and x.get("exchange") not in {"ARCA", "NYSEARCA", "BATS"}
        ],
        columns=MASTER,
    )


def checked_master(df):
    missing = set(MASTER) - set(df)
    if missing:
        raise ValueError(f"master missing {sorted(missing)}")
    df = df.loc[(df.kind == "common") & df.exchange.isin(EXCHANGES), MASTER].copy()
    for name in ("start", "end"):
        df[name] = pd.to_datetime(df[name], errors="raise").dt.strftime("%Y-%m-%d")
    if df.empty or df.isna().any().any() or (df.start > df.end).any() or (df.lot.astype(float) < 1).any():
        raise ValueError("empty or invalid dated common-stock master (inclusive end dates, lot in shares)")
    if not df.symbol.map(lambda v: bool(re.fullmatch(r"[A-Z0-9.\-]{1,24}", str(v)))).all():
        raise ValueError("invalid ticker in master")
    for key in ("symbol", "asset_id"):
        for ident, group in df.sort_values("start").groupby(key):
            if len(group) > 1 and any(a >= b for a, b in zip(group.end[:-1], group.start[1:], strict=True)):
                raise ValueError(f"overlapping {key} intervals: {ident}")
    return df.sort_values(["asset_id", "start"]).reset_index(drop=True)


def _demo_bars(symbol, sessions, minute=False):
    """Fixture prices have deliberate directional intraday structure; never evidence of a market edge."""
    k = int(symbol[-2:])
    parts = []
    for d in sessions:
        serial = d.toordinal()
        base = (15 + 7 * k) * (1 + (0.01 + 0.012 * k) * np.sin(serial * (0.13 + 0.027 * k)))
        vol = 30000 + 4500 * k
        if not minute:
            parts.append(
                (
                    d.tz_localize("America/New_York").tz_convert("UTC"),
                    base,
                    base * 1.04,
                    base * 0.995,
                    base * 1.025,
                    vol * 390,
                )
            )
            continue
        ts = get_calendar().regular_minutes(d)
        rng = np.random.default_rng([serial, k, 91])
        t = np.arange(len(ts))
        ret = 0.00028 + 0.00078 * np.sin(t / 13 + k / 2) + rng.normal(0, 0.00023, len(t))
        c = base * np.exp(np.cumsum(ret))
        o = np.r_[base, c[:-1]]
        parts.append(
            pd.DataFrame(
                {
                    "t": ts,
                    "o": o,
                    "h": np.maximum(c, o) * 1.00009,
                    "l": np.minimum(o, c) * 0.99991,
                    "c": c,
                    "v": vol * (1 + 0.3 * np.cos(t / 17)),
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if minute and parts else pd.DataFrame(parts, columns=["t", *FIELDS])


class Market:
    """Cache each raw symbol-period as parquet plus SHA-256; .offline forbids filling missing units."""

    def __init__(self, root, spec, *, offline=False):
        self.root, self.p, self.offline = Path(root), spec, offline
        self.root.mkdir(exist_ok=True, parents=True)
        self.demo = spec["mode"] == "demo"
        self.client = None
        self.verified = set()
        self.locks = [threading.Lock() for _ in range(64)]
        self.cached = lru_cache(maxsize=48)(self._read_frame)
        dates = calendar_dates(spec)
        self.lo, self.hi = dates[0], dates[-1]
        pin(self.root / "identity.json", {"schema": 2, "mode": spec["mode"], "data_spec": data_id(spec)})

    def api(self):
        if self.offline or self.demo:
            raise RuntimeError("network is disabled for offline replay")
        if self.client is None:
            self.client = AlpacaDataClient()
        return self.client

    def asset_master(self, csv=None, approximate=False):
        path = self.root / "instruments.csv"
        if path.exists() and (self.root / "master-source.json").exists():
            info = read(self.root / "master-source.json")
            if file_hash(path) != info["sha256"] or (csv and info.get("input_sha256") != file_hash(csv)):
                raise ValueError("instrument master changed; use a new archive")
            return checked_master(pd.read_csv(path, dtype={"asset_id": str})), info
        if self.offline:
            raise FileNotFoundError("missing sealed instruments.csv or master-source.json")
        if self.demo:
            df, label = demo_master(self.p), "synthetic"
        elif csv:
            df, label = pd.read_csv(csv, dtype={"asset_id": str}), "operator-dated-master"
        elif approximate:
            df, label = candidate_snapshot(self.api()), "approximate-present-directory-not-survivorship-free"
            if self.p["mode"] == "pilot":
                # This list is expressly a connectivity/engineering test, never the universe for the full experiment.
                seeds = (
                    "AAPL MSFT NVDA ORCL IBM INTC AMD ADBE AMZN WMT COST TGT LOW HD KO PEP MCD SBUX "
                    "UNH ABT MRK PFE JPM GS MS C BAC WFC V MA CAT DE F GM XOM CVX BA GE"
                ).split()
                df = df[df.symbol.isin(seeds)].copy()
                label = "fixed-pilot-candidates-not-for-inference"
        else:
            raise ValueError("provide --master dated-instruments.csv or explicitly --approximate-master")
        df = checked_master(df)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(path)
        info = {
            "source": label,
            "sha256": file_hash(path),
            "input_sha256": file_hash(csv) if csv else None,
            "ids": df.asset_id.nunique(),
        }
        write(self.root / "master-source.json", info)
        return df, info

    def _unit(self, path, build):
        with self.locks[hash(path) % len(self.locks)]:
            self._locked_unit(path, build)

    def _locked_unit(self, path, build):
        side = path.with_suffix(".sha.json")
        if side.exists() and path.exists():
            if path not in self.verified:
                h = read(side)["sha256"]
                if (
                    getattr(self, "expected", None) is not None
                    and self.expected.get(side.relative_to(self.root).as_posix()) != h
                ):
                    raise ValueError(f"raw unit does not match frozen inventory: {path}")
                if file_hash(path) != h:
                    raise ValueError(f"corrupt raw unit: {path}")
                self.verified.add(path)
            return
        if self.offline:
            raise FileNotFoundError(f"unstaged unit: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df = build()
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
        write(side, {"sha256": file_hash(path), "rows": len(df)})
        self.verified.add(path)

    def frame(self, symbol, start, end, kind):
        """Raw/split daily or raw minute data on inclusive local dates, with literal historical ticker mapping."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end < start:
            return pd.DataFrame(columns=["t", *FIELDS])
        freq = "M" if kind == "minute" else "Y"
        parts = []
        for period in pd.period_range(start, end, freq=freq):
            path = self.root / "raw" / kind / symbol / (str(period) + ".parquet")
            lo = max(self.lo, period.start_time.normalize())
            hi = min(self.hi, period.end_time.normalize())

            def make(lo=lo, hi=hi):
                if self.demo:
                    return _demo_bars(symbol, get_calendar().sessions(lo, hi), kind == "minute")
                left = lo.tz_localize("America/New_York").tz_convert("UTC")
                right = (hi + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
                rows = self.api()._records(
                    f"/stocks/{symbol}/bars",
                    "bars",
                    left,
                    right,
                    {
                        "timeframe": "1Min" if kind == "minute" else "1Day",
                        "adjustment": "split" if kind == "split" else "raw",
                        "feed": "sip",
                        "asof": "-",
                    },
                )
                return pd.DataFrame(rows, columns=["t", *FIELDS])

            self._unit(path, make)
            frame = self.cached(path)
            if not frame.empty:
                ix = frame.index
                # All raw timestamps are timezone aware; the day cutoff is always New York.
                left = start.tz_localize("America/New_York").tz_convert("UTC")
                right = (end + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
                parts.append(frame.iloc[ix.searchsorted(left) : ix.searchsorted(right)])
        return pd.concat(parts) if parts else pd.DataFrame(columns=FIELDS, index=pd.DatetimeIndex([], tz="UTC"))

    @staticmethod
    def _read_frame(path):
        f = pd.read_parquet(path)
        f["t"] = pd.to_datetime(f.t, utc=True, format="mixed").astype("datetime64[ns, UTC]")
        if f.t.duplicated().any():
            raise ValueError(f"duplicate bars: {path}")
        if len(f):
            a = f[FIELDS].to_numpy(float)
            if (
                not np.isfinite(a).all()
                or (a[:, :4] <= 0).any()
                or (a[:, 4] < 0).any()
                or (a[:, 1] < a[:, [0, 2, 3]].max(axis=1) - 1e-6).any()
                or (a[:, 2] > a[:, [0, 1, 3]].min(axis=1) + 1e-6).any()
            ):
                raise ValueError(f"invalid OHLCV: {path}")
        return f.set_index("t").sort_index()

    def quotes(self, symbol, left, seconds=60):
        """One sampled SIP interval; preserve quote units and timestamp for calibration."""
        right = left + pd.Timedelta(seconds=seconds)
        path = self.root / "quotes" / symbol / (left.strftime("%Y%m%dT%H%M%S") + ".parquet")

        def make():
            if self.demo:
                k = int(symbol[-2:])
                t = pd.date_range(left, right, freq="1s", inclusive="left")
                mid = (15 + 7 * k) * (1 + 0.00005 * np.sin(np.arange(len(t)) / 5))
                return pd.DataFrame({"t": t, "bp": mid * 0.99984, "ap": mid * 1.00016, "bs": 24, "as": 18})
            data = self.api()._records(f"/stocks/{symbol}/quotes", "quotes", left, right, {"feed": "sip", "asof": "-"})
            return pd.DataFrame(data, columns=["t", "bp", "ap", "bs", "as"])

        self._unit(path, make)
        return pd.read_parquet(path)
