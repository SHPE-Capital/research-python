"""`edgar` source: filings of one form for one or more companies.

One row per filing: `ts` is the period the filing reports on (falling back to the filing date),
`ts_available` is the SEC acceptance time, `series` is "<cik>:<form>" and `value` is 1.0 (an event marker);
the accession number is kept in an extra `accession` column. The SEC requires a contact User-Agent:
set EDGAR_USER_AGENT in the environment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, ClassVar

import httpx
import pandas as pd

from shpe_research.data.sources.base import SERIES_COLUMNS, clip_available, register_source, series_frame

URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def fetch_submissions(cik: int, form: str, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Recent filings of `form` for `cik` as `{accession, form, report_date, acceptance}` dicts."""
    agent = os.environ.get("EDGAR_USER_AGENT")
    if not agent:
        raise RuntimeError("EDGAR_USER_AGENT is not set; the SEC requires a contact string (name and email)")
    with httpx.Client(timeout=30.0, headers={"User-Agent": agent}, transport=transport) as http:
        res = http.get(URL.format(cik=cik))
    res.raise_for_status()
    recent = res.json()["filings"]["recent"]
    return [
        {"accession": recent["accessionNumber"][i], "form": f,
         "report_date": recent["reportDate"][i] or recent["filingDate"][i],
         "acceptance": recent["acceptanceDateTime"][i]}
        for i, f in enumerate(recent["form"])
        if f == form
    ]


@register_source
class EdgarSource:
    source_type: ClassVar[str] = "edgar"

    def __init__(self, form: str, cik: int | str | list[int | str], fetch: Callable | None = None):
        self.form = form
        self.ciks = [int(c) for c in (cik if isinstance(cik, list) else [cik])]
        self._fetch = fetch or fetch_submissions

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Day") -> pd.DataFrame:
        """Filings whose acceptance time is inside the date window; `symbols` and `timeframe` are ignored.

        Raises ValueError if a filing is accepted before the period it reports on.
        """
        rows = []
        for cik in self.ciks:
            for f in self._fetch(cik, self.form):
                rows.append({"ts": f["report_date"], "ts_available": f["acceptance"],
                             "series": f"{cik}:{f['form']}", "value": 1.0, "accession": f["accession"]})
        if not rows:
            return pd.DataFrame(columns=[*SERIES_COLUMNS, "accession"])
        df = series_frame(rows)
        return clip_available(df, start, end)
