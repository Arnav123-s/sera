import numpy as np

from experiments.sparse_mechanisms.imaging import recover


def test_fft_cost_counter_includes_final_stationarity_check(monkeypatch):
    measured = np.zeros((8, 8), complex)
    measured[0, 0] = 1.
    count = [0]
    for name in ("fft2", "ifft2"):
        original = getattr(np.fft, name)
        def wrapped(*args, _original=original, **kwargs):
            count[0] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(np.fft, name, wrapped)
    _, work = recover(measured, np.ones((8, 8), bool), "haar", 0., iterations=10)
    assert work["fft_calls"] == count[0]
