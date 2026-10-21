from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourier_filter_sim import (  # noqa: E402
    fft2c,
    make_case,
    make_filter,
    make_grid,
    make_pupil,
)


def config() -> dict:
    with (ROOT / "config" / "default.yml").open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_user_parameters_are_internally_consistent() -> None:
    cfg = config()
    assert cfg["optics"]["objective_focal_length_mm"] == 50.0
    assert cfg["optics"]["magnification"] == 20.0
    assert cfg["optics"]["tube_lens_focal_length_mm"] == 1000.0
    assert cfg["grid"]["object_fov_mm"] == 0.5
    assert cfg["optics"]["wavelength_mm"] == 0.000355
    assert np.isclose(
        cfg["optics"]["objective_focal_length_mm"]
        * cfg["optics"]["magnification"],
        cfg["optics"]["tube_lens_focal_length_mm"],
    )


def test_fourier_plane_order_spacing() -> None:
    cfg = config()
    grid = make_grid(cfg)
    objective_bfp_spacing = (
        cfg["optics"]["wavelength_mm"]
        * cfg["optics"]["objective_focal_length_mm"]
        / cfg["object"]["lattice_pitch_mm"]
    )
    filter_plane_spacing = objective_bfp_spacing * cfg["optics"]["pupil_relay_magnification"]
    assert np.isclose(objective_bfp_spacing, 0.8875)
    assert np.isclose(filter_plane_spacing, 4.4375)
    assert np.isclose(grid.dxi_mm, 0.1775)


def test_stripe_spectrum_peak_matches_theory() -> None:
    cfg = config()
    grid = make_grid(cfg)
    case = make_case("stripes", grid, cfg)
    spectrum = np.abs(fft2c(case.reference_field)) ** 2
    row = spectrum[len(grid.x_mm) // 2].copy()
    center = len(row) // 2
    row[center - 2 : center + 3] = 0
    positive = np.arange(len(row)) > center
    peak_index = np.argmax(np.where(positive, row, 0))
    measured = grid.XI_mm[center, peak_index]
    assert abs(measured - 4.4375) <= grid.dxi_mm


def test_filter_blocks_theoretical_orders() -> None:
    cfg = config()
    grid = make_grid(cfg)
    case = make_case("stripes", grid, cfg)
    transmission, centers = make_filter(grid, case, cfg, "hard")
    assert len(centers) == 11
    center_row = len(grid.x_mm) // 2
    for cx, _ in centers:
        idx = int(np.argmin(np.abs(grid.XI_mm[center_row] - cx)))
        assert transmission[center_row, idx] == 0


def test_pupil_radius_and_sine_tangent_error() -> None:
    cfg = config()
    grid = make_grid(cfg)
    pupil = np.abs(make_pupil(grid, cfg))
    occupied_radius = np.max(np.hypot(grid.XI_mm, grid.ETA_mm)[pupil > 0])
    assert occupied_radius <= 25.0 + grid.dxi_mm
    s = cfg["optics"]["numerical_aperture"] / cfg["optics"]["object_refractive_index"]
    error = 1 / np.sqrt(1 - s**2) - 1
    assert np.isclose(error * 100, 0.5037815, rtol=1e-5)


def test_reference_filter_dimensions_and_deck_spacing() -> None:
    cfg = config()
    filt = cfg["filter"]
    assert np.isclose(filt["hard_width_mm"], 1.0)
    assert np.isclose(filt["fuzzy_min_width_mm"], 1.1)
    assert np.isclose(filt["fuzzy_max_width_mm"], 4.0)
    assert np.isclose(filt["serration_pitch_mm"], 0.75)
    assert np.isclose(
        filt["serrated_nominal_width_mm"] - 2 * filt["serration_amplitude_mm"],
        filt["fuzzy_min_width_mm"],
    )
    assert np.isclose(
        filt["serrated_nominal_width_mm"] + 2 * filt["serration_amplitude_mm"],
        filt["fuzzy_max_width_mm"],
    )
    order_spacing = 4.4375
    assert 2 * order_spacing >= filt["minimum_same_deck_spacing_mm"]
    assert order_spacing >= filt["minimum_cross_deck_spacing_mm"]
