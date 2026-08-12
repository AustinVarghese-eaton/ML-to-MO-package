import math

import pytest

from surrogategen.export.formatting import fmt_float, fmt_mat, fmt_vec


def test_fmt_float_roundtrip():
    assert fmt_float(1.5) == "1.5"
    assert float(fmt_float(0.1)) == 0.1
    assert float(fmt_float(-3.14159e-7)) == -3.14159e-7


def test_fmt_vec():
    assert fmt_vec([1.0, 2.0, 3.0]) == "{1.0,2.0,3.0}"


def test_fmt_mat():
    assert fmt_mat([[1.0, 2.0], [3.0, 4.0]]) == "{{1.0,2.0},{3.0,4.0}}"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_reject_non_finite(bad):
    with pytest.raises(ValueError):
        fmt_float(bad)
    with pytest.raises(ValueError):
        fmt_vec([1.0, bad])
