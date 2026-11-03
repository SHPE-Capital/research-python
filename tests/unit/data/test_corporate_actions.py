from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from shpe_research.data.alpaca import AlpacaDataClient, Credentials
from shpe_research.data.audit import audit_bars
from shpe_research.data.corporate_actions import (
    explain_split_gaps,
    fetch_corporate_actions,
    load_corporate_actions,
    split_factor,
)
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.ratelimit import RateLimiter

PAGE1 = {
    "corporate_actions": {
        "forward_splits": [
            {"symbol": "AAPL", "new_rate": 4, "old_rate": 1, "ex_date": "2020-08-31", "process_date": "2020-08-31"}
        ],
        "name_changes": [{"old_symbol": "FB", "new_symbol": "META", "process_date": "2020-09-10"}],
    },
    "next_page_token": "p2",
}
PAGE2 = {
    "corporate_actions": {
        "reverse_splits": [
            {"symbol": "AAPL", "new_rate": 1, "old_rate": 10, "ex_date": "2020-09-15", "process_date": "2020-09-15"}
        ]
    },
    "next_page_token": None,
}


def _client(requests: list) -> AlpacaDataClient:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE2 if "page_token=p2" in str(request.url) else PAGE1)

    return AlpacaDataClient(
        Credentials("k", "s"), limiter=RateLimiter(10**9), transport=httpx.MockTransport(handle), sleep=lambda s: None
    )


def test_corporate_actions(tmp_path):
    requests: list = []
    actions = fetch_corporate_actions(["AAPL"], "2020-08-01", "2020-09-30", client=_client(requests), root=tmp_path)
    assert actions is not None
    assert len(requests) == 2  # followed next_page_token
    assert requests[0].url.path == "/v1/corporate-actions"
    assert "forward_split" in str(requests[0].url) and "name_change" in str(requests[0].url)
    assert sorted(actions["type"]) == ["forward_split", "name_change", "reverse_split"]
    fwd = actions[actions["type"] == "forward_split"].iloc[0]
    assert fwd["symbol"] == "AAPL" and fwd["ratio"] == 4.0 and fwd["ex_date"] == pd.Timestamp("2020-08-31")
    rev = actions[actions["type"] == "reverse_split"].iloc[0]
    assert rev["ratio"] == pytest.approx(0.1)
    nc = actions[actions["type"] == "name_change"].iloc[0]
    assert (nc["symbol"], nc["new_symbol"]) == ("FB", "META")
    assert (tmp_path / "corporate_actions").is_dir()
    # stored and reloadable; refetching does not duplicate rows
    fetch_corporate_actions(["AAPL"], "2020-08-01", "2020-09-30", client=_client([]), root=tmp_path)
    assert len(load_corporate_actions(["AAPL"], root=tmp_path)) == 2  # name change is keyed to FB


def test_split_factor(tmp_path):
    fetch_corporate_actions(["AAPL"], "2020-08-01", "2020-09-30", client=_client([]), root=tmp_path)
    f = lambda a, b: split_factor("AAPL", a, b, root=tmp_path)  # noqa: E731
    assert f("2020-08-01", "2020-08-30") == 1.0
    assert f("2020-08-01", "2020-08-31") == 4.0  # ex_date inside (start, end]
    assert f("2020-08-31", "2020-09-14") == 1.0  # start is already post-split
    assert f("2020-08-01", "2020-09-30") == pytest.approx(0.4)
    with pytest.raises(ValueError, match="MSFT"):
        split_factor("MSFT", "2020-08-01", "2020-09-30", root=tmp_path)


def test_empty_fetch_is_remembered(tmp_path):
    def handle(request):
        return httpx.Response(200, json={"corporate_actions": {}, "next_page_token": None})

    client = AlpacaDataClient(
        Credentials("k", "s"), limiter=RateLimiter(10**9), transport=httpx.MockTransport(handle), sleep=lambda s: None
    )
    assert fetch_corporate_actions(["KO"], "2020-01-01", "2020-02-01", client=client, root=tmp_path).empty
    assert split_factor("KO", "2020-01-01", "2020-02-01", root=tmp_path) == 1.0


def test_audit_gaps_are_marked_explained(tmp_path):
    bars = synthetic_bars(symbols=["AAA"], start="2026-11-23", end="2026-11-30")
    after = bars["ts"] >= pd.Timestamp("2026-11-27T00:00Z")
    bars.loc[after, ["open", "high", "low", "close", "vwap"]] /= 2
    report = audit_bars(bars)
    assert (report.issues["kind"] == "possible_split").sum() == 1
    split = pd.DataFrame(
        {"type": ["forward_split"], "symbol": ["AAA"], "new_symbol": [None], "ratio": [2.0],
         "ex_date": [pd.Timestamp("2026-11-27")]}
    )
    explained = explain_split_gaps(report, split)
    row = explained.issues[explained.issues["kind"] == "possible_split"].iloc[0]
    assert row["detail"].startswith("explained") and row["severity"] == "info"
    assert "explained" in explained.to_markdown()
    assert explained.symbols["warnings"].iloc[0] == 0
    # a split of a different size or date does not explain it
    wrong = split.assign(ratio=[10.0])
    assert explain_split_gaps(report, wrong).issues["detail"].str.startswith("unexplained").any()
    none = explain_split_gaps(report, split.iloc[0:0])
    assert none.issues[none.issues["kind"] == "possible_split"]["detail"].str.startswith("unexplained").all()
    assert np.isclose(report.days["overnight_gap"].min(), -np.log(2), atol=0.05)
