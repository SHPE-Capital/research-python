"""C1m: point-in-time security master."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.data.security_master import SecurityMaster
from shpe_research.data.universe import point_in_time_members


def _master() -> SecurityMaster:
    rows = [
        # symbol, valid_from, valid_to, sector, exchange, status
        ("AAA", "2020-01-01", None, "Tech", "NASDAQ", "active"),
        ("LATE", "2024-03-01", None, "Tech", "NYSE", "active"),  # listed in March
        ("GONE", "2020-01-01", "2024-03-15", "Energy", "NYSE", "delisted"),  # last day is 03-14
        ("SWCH", "2020-01-01", "2024-06-01", "Tech", "NASDAQ", "active"),
        ("SWCH", "2024-06-01", None, "Health", "NASDAQ", "active"),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "valid_from", "valid_to", "sector", "listing_exchange",
                                     "asset_status"])
    df["source"] = "hand"
    return SecurityMaster(df)


def test_members_as_of():
    m = _master()
    assert "LATE" not in m.members("2024-02-29")
    assert "LATE" in m.members("2024-04-01")
    assert "LATE" in m.members("2024-03-01")  # valid_from is inclusive


def test_delisted_kept_until_valid_to():
    m = _master()
    assert "GONE" in m.members("2024-03-14")
    assert "GONE" not in m.members("2024-03-15")  # valid_to is exclusive
    assert "GONE" in m.members("2020-01-01")


def test_sector_change():
    m = _master()
    before = m.sector_of(["SWCH", "AAA"], "2024-05-31")
    after = m.sector_of(["SWCH", "AAA"], "2024-06-01")
    assert before["SWCH"] == "Tech" and after["SWCH"] == "Health"
    assert before["AAA"] == after["AAA"] == "Tech"
    assert m.members("2024-06-01").count("SWCH") == 1


def test_checksum_verified(tmp_path):
    path = _master().save(tmp_path / "master.parquet")
    assert SecurityMaster.load(path).members("2024-04-01") == _master().members("2024-04-01")
    other = _master().table.assign(sector="Tampered")
    other.to_parquet(path)
    with pytest.raises(ValueError, match="checksum"):
        SecurityMaster.load(path)


def test_filters_and_unknown_symbol():
    m = _master()
    assert m.members("2024-04-01", {"sector": "Tech"}) == ["AAA", "LATE", "SWCH"]
    assert m.members("2024-04-01", {"listing_exchange": ["NYSE"]}) == ["LATE"]
    assert pd.isna(m.sector_of(["NOPE"], "2024-04-01")["NOPE"])
    with pytest.raises(ValueError, match="unknown filter"):
        m.members("2024-04-01", {"colour": "red"})


def test_overlapping_rows_rejected():
    df = _master().table
    bad = pd.concat([df, df[df["symbol"] == "AAA"].assign(sector="Other")], ignore_index=True)
    with pytest.raises(ValueError, match="overlap"):
        SecurityMaster(bad)


def test_point_in_time_members_from_config(tmp_path):
    path = _master().save(tmp_path / "master.parquet")
    cfg = {"type": "point_in_time", "master": str(path), "filters": {"sector": "Tech"}}
    assert point_in_time_members(cfg, "2024-02-01") == ["AAA", "SWCH"]
    with pytest.raises(ValueError, match="point_in_time"):
        point_in_time_members({"type": "static"}, "2024-02-01")
