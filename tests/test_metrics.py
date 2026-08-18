import numpy as np

from surrogategen import metrics


def test_perfect_prediction():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    m = metrics.compute_metrics(y, y, ["a", "b"])
    assert m["n_test"] == 3
    for col in ("a", "b"):
        assert m["per_output"][col]["mae"] == 0.0
        assert m["per_output"][col]["rmse"] == 0.0
        assert m["per_output"][col]["r2"] == 1.0
    assert m["overall"]["mae"] == 0.0


def test_known_errors():
    y_true = np.array([[10.0], [20.0], [30.0]])
    y_pred = np.array([[11.0], [19.0], [30.0]])  # errors: +1, -1, 0
    m = metrics.compute_metrics(y_true, y_pred, ["v"])
    out = m["per_output"]["v"]
    assert abs(out["mae"] - (2.0 / 3.0)) < 1e-9
    assert abs(out["rmse"] - np.sqrt(2.0 / 3.0)) < 1e-9
    assert abs(out["max_abs_err"] - 1.0) < 1e-9


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        metrics.compute_metrics(np.zeros((3, 2)), np.zeros((3, 1)), ["a", "b"])


def test_format_report_contains_columns():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = metrics.compute_metrics(y, y, ["alpha", "beta"])
    text = metrics.format_report(m)
    assert "alpha" in text
    assert "beta" in text
    assert "overall" in text
