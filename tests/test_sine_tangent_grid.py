from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sine_tangent_grid import (F, PITCH, WAVELENGTH, FOV, fft_check,
                               finite_grid_intensity, map_sines, na_rows, order_rows)


def test_edge_reference_and_monotonicity():
    rows = na_rows()
    assert len(rows) == 9
    assert np.isclose(rows[0]["delta_um"], 25.189076296, atol=1e-7)
    assert np.isclose(rows[-1]["r_tan_mm"], 103.237080242, atol=1e-8)
    assert np.all(np.diff([r["error_percent"] for r in rows]) > 0)


def test_radial_denominator_and_rotation_symmetry():
    xt, yt = map_sines(.355, .355, projection="tan")
    assert np.isclose(xt, F*.355/np.sqrt(1-2*.355**2))
    assert xt == yt
    rx, ry = map_sines(np.hypot(.355, .355), 0, projection="tan")
    assert np.isclose(np.hypot(xt, yt), rx)
    assert ry == 0


def test_invalid_directions_rejected():
    with pytest.raises(ValueError):
        map_sines(.8, .8)
    with pytest.raises(ValueError):
        map_sines(1, 0, projection="tan")


def test_na_admits_orders_but_does_not_move_them():
    lower, higher = order_rows(.3), order_rows(.9)
    assert len(lower) == 9
    assert len(higher) == 81
    assert all(row in higher for row in lower)
    assert len(order_rows(.1)) == 1


def test_finite_grid_peak_and_first_zero():
    m, ell = 5, 0
    expected = (np.sinc(.3*m)*np.sinc(.3*ell))**2
    assert np.isclose(finite_grid_intensity(m/PITCH, ell/PITCH), expected)
    assert finite_grid_intensity(m/PITCH+1/FOV, ell/PITCH) < 1e-20


@pytest.mark.parametrize("pixels", [100, 200])
def test_separable_fft_agrees_with_continuous_grid_coefficients(pixels):
    check = fft_check(order_rows(), pixels)
    assert check["max_absolute_power_coefficient_error"] < 1e-14


def test_jacobian_power_conservation_and_local_stretch():
    s = np.array([.355, .355])
    step = 1e-6
    columns = []
    for ds in (np.array([step, 0]), np.array([0, step])):
        upper = np.array(map_sines(*(s+ds), projection="tan"))
        lower = np.array(map_sines(*(s-ds), projection="tan"))
        columns.append((upper-lower)/(2*step*F))
    jac = np.column_stack(columns)
    area = np.linalg.det(jac)
    predicted = (1-np.sum(s*s))**-2
    assert np.isclose(area, predicted, rtol=1e-8)
    # Coordinate density times area equals original differential power.
    assert np.isclose(area*(1-np.sum(s*s))**2, 1, rtol=1e-8)


def test_twenty_um_original_fundamental_is_not_na_edge():
    s = WAVELENGTH/.020
    xs, _ = map_sines(s, 0)
    xt, _ = map_sines(s, 0, projection="tan")
    assert np.isclose((xt-xs)*1000, .139842029477, atol=1e-10)
