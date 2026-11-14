import pytest

from shpe_research.trade.exits import StopState, initial_stop, step_stop


def test_initial_stop():
    assert initial_stop(100.0, 2.0) == 97.0
    assert StopState.from_entry(100.0, 2.0).stop == 97.0


def test_no_trailing_below_one_atr():
    s = StopState.from_entry(100.0, 2.0)
    assert s.on_close(101.9) == 97.0 and not s.trailing


def test_trailing_activation_and_never_down():
    s = StopState.from_entry(100.0, 2.0)
    assert s.on_close(102.0) == 99.0  # 102 - 3
    assert s.trailing
    assert s.on_close(105.0) == 102.0
    assert s.on_close(101.0) == 102.0  # lower close: stop holds


def test_step_stop_scalar_matches():
    assert step_stop(97.0, 100.0, False, 100.0, 2.0, 103.0) == (100.0, 103.0, True)


def test_invalid():
    with pytest.raises(ValueError):
        StopState.from_entry(100.0, 0.0)
