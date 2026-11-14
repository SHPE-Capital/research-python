# ruff: noqa: E741
import numpy as np

from shpe_research.features.momentum.wilder import adx_di, adx_signal, atr, true_range


def _ref_adx(h, l, c, a):
    """Independent scalar 'spreadsheet' reference (Wilder, mean seeds)."""
    n = len(c)
    tr = [None] + [max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1])) for t in range(1, n)]
    pdm, mdm = [None], [None]
    for t in range(1, n):
        up, dn = h[t] - h[t - 1], l[t - 1] - l[t]
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)

    def smooth(x, start):
        out = [None] * n
        s = sum(x[start : start + a]) / a
        out[start + a - 1] = s
        for t in range(start + a, n):
            s = s + (x[t] - s) / a
            out[t] = s
        return out

    st, sp, sm = smooth(tr, 1), smooth(pdm, 1), smooth(mdm, 1)
    pdi = [None if st[t] is None else 100 * sp[t] / st[t] for t in range(n)]
    mdi = [None if st[t] is None else 100 * sm[t] / st[t] for t in range(n)]
    dx = [None if pdi[t] is None else 100 * abs(pdi[t] - mdi[t]) / (pdi[t] + mdi[t]) for t in range(n)]
    adx = smooth(dx, a)  # first DX at index a
    return st, pdi, mdi, adx


def _series(n=30):
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    h = c + rng.uniform(0.2, 1.5, n)
    l = c - rng.uniform(0.2, 1.5, n)
    return h, l, c


def test_atr_hand_worked():
    # close constant 100, H=100+t/2, L=100-t/2 => TR_t = t (t>=1)
    t = np.arange(30, dtype=float)
    h, l, c = 100 + t / 2, 100 - t / 2, np.full(30, 100.0)
    a = atr(h[None], l[None], c[None], 14)[0]
    assert np.isnan(a[:14]).all()
    assert a[14] == 7.5  # mean(1..14)
    assert abs(a[15] - (7.5 + (15 - 7.5) / 14)) < 1e-12  # 8.035714...
    assert abs(a[15] - 8.0357142857) < 1e-9


def test_true_range_uses_previous_close_gap():
    tr = true_range(np.array([[10.0, 15.0]]), np.array([[9.0, 14.0]]), np.array([[9.5, 14.5]]))
    assert np.isnan(tr[0, 0]) and tr[0, 1] == 5.5  # |15-9.5|


def test_adx_hand_worked_trend():
    # C=100, H=100+t, L=100-t/2: +DM=1, -DM=0, TR=1.5t. a=3.
    t = np.arange(12, dtype=float)
    h, l, c = (100 + t)[None], (100 - t / 2)[None], np.full((1, 12), 100.0)
    adx, p, m = adx_di(h, l, c, 3)
    assert np.isnan(p[0, :3]).all()
    assert abs(p[0, 3] - 100 / 3) < 1e-9  # 100*1/mean(1.5,3,4.5)=100/3
    assert m[0, 3] == 0
    assert np.isnan(adx[0, :5]).all()  # first ADX at 2a-1
    assert abs(adx[0, 5] - 100) < 1e-9


def test_adx_matches_scalar_reference_30_bars():
    h, l, c = _series(30)
    st, pdi, mdi, adx = _ref_adx(h, l, c, 7)
    got_adx, got_p, got_m = adx_di(h[None], l[None], c[None], 7)
    for t in range(30):
        for ref, got in ((pdi, got_p), (mdi, got_m), (adx, got_adx)):
            if ref[t] is None:
                assert np.isnan(got[0, t])
            else:
                assert abs(got[0, t] - ref[t]) < 1e-9
    got_atr = atr(h[None], l[None], c[None], 7)[0]
    assert abs(got_atr[-1] - st[-1]) < 1e-9


def test_vectorized_across_symbols_independent():
    h, l, c = _series(40)
    h2, l2, c2 = _series(40)
    both = adx_di(np.stack([h, h2 + 1]), np.stack([l, l2 + 1]), np.stack([c, c2 + 1]), 14)[0]
    alone = adx_di(h[None], l[None], c[None], 14)[0]
    np.testing.assert_allclose(both[0], alone[0], equal_nan=True)


def test_nan_input_invalid_and_resets():
    h, l, c = _series(60)
    c = c.copy()
    c[30] = np.nan
    a = atr(h[None], l[None], c[None], 14)[0]
    # NaN close at 30 poisons TR[31] only; recursion resets there
    assert np.isfinite(a[30]) and np.isnan(a[31]) and np.isnan(a[44])
    assert np.isfinite(a[45])  # 14 valid TRs (32..45) after the reset


def test_zero_denominator_and_signal_invalid():
    flat = np.full((1, 40), 50.0)
    adx, p, m = adx_di(flat, flat, flat, 7)
    assert np.isnan(adx).all() and np.isnan(p).all()
    assert not adx_signal(adx, p, m, 15).any()


def test_adx_signal_thresholds_and_direction():
    adx = np.array([[26.0, 26.0, 25.0]])
    p = np.array([[30.0, 10.0, 30.0]])
    m = np.array([[10.0, 30.0, 10.0]])
    assert adx_signal(adx, p, m, 25).tolist() == [[True, False, False]]
    assert adx_signal(adx, p, m, 30).tolist() == [[False, False, False]]


def test_float32_supported():
    h, l, c = _series(40)
    out = adx_di(h[None].astype(np.float32), l[None].astype(np.float32), c[None].astype(np.float32), 7)[0]
    assert out.dtype == np.float32
