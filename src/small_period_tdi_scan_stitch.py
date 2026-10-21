"""Matched narrow-line TDI scan-and-stitch simulation for 2--5 um patterns."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from small_period_raw_intensity_simulation import (
    F_MM,
    LAMBDA_MM,
    MY,
    NA,
    PERIODS_UM,
    ft2,
    ift2,
    irregular_particle_field,
    periodic_pattern,
    seven_rod_mask,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "small_period_point_defects"
TDI_STAGES = 25
PIXEL_PITCH_IMG_UM = 5.0
MAGNIFICATION = 20.0
TDI_WIDTH_IMG_UM = TDI_STAGES * PIXEL_PITCH_IMG_UM
LINE_WIDTH_OBJ_UM = TDI_WIDTH_IMG_UM / MAGNIFICATION
OUTPUT_PIXEL_PITCH_OBJ_UM = PIXEL_PITCH_IMG_UM / MAGNIFICATION


def scan_pixel_by_pixel(actual, pupil, rods, coordinate):
    """Scan adjacent object pixels; every output column is written exactly once."""
    line_fwhm_mm = LINE_WIDTH_OBJ_UM / 1000
    output_pitch_mm = OUTPUT_PIXEL_PITCH_OBJ_UM / 1000
    centres = np.arange(-0.025, 0.025 + 0.5 * output_pitch_mm, output_pitch_mm)
    scanned_open = np.zeros((actual.shape[0], centres.size), dtype=np.float64)
    scanned_filtered = np.zeros((actual.shape[0], centres.size), dtype=np.float64)

    for output_index, centre in enumerate(centres):
        offset = coordinate - centre
        illumination_intensity = np.exp(-4 * np.log(2) * (offset / line_fwhm_mm) ** 2)
        illuminated_field = actual * np.sqrt(illumination_intensity)[None, :]
        spectrum = ft2(illuminated_field)
        image_open = np.abs(ift2(spectrum * pupil)) ** 2
        image_filtered = np.abs(ift2(spectrum * pupil * rods)) ** 2
        # One 5 um image pixel corresponds to a non-overlapping 0.25 um object bin.
        pixel_bin = np.abs(offset) < output_pitch_mm / 2
        scanned_open[:, output_index] = np.mean(image_open[:, pixel_bin], axis=1)
        scanned_filtered[:, output_index] = np.mean(image_filtered[:, pixel_bin], axis=1)
    return scanned_open, scanned_filtered, centres


def simulate_period(period_um, grid, coordinate):
    period_mm = period_um / 1000
    spacing_mm = MY * F_MM * LAMBDA_MM / period_mm
    rods = seven_rod_mask(grid["x_hw"], grid["y_hw"], spacing_mm)
    background = periodic_pattern(grid["xx"], grid["yy"], period_mm)
    actual, _ = irregular_particle_field(background, grid["xx"], grid["yy"])
    open_image, filtered_image, output_x = scan_pixel_by_pixel(actual, grid["pupil"], rods, coordinate)
    return {
        "period_um": period_um,
        "bright_width_um": period_um * 4 / 7,
        "dark_width_um": period_um * 3 / 7,
        "spacing_mm": spacing_mm,
        "scan_count": output_x.size,
        "output_x": output_x,
        "open": open_image,
        "filtered": filtered_image,
    }


def plot_results(results, coordinate):
    roi_y = np.abs(coordinate) <= 0.008
    extent = [-25, 25, -8, 8]
    fig, axes = plt.subplots(len(results), 2, figsize=(10.5, 9.8), constrained_layout=True)
    for row, result in enumerate(results):
        reference = max(float(result["open"].max()), 1e-30)
        panels = []
        for key in ("open", "filtered"):
            intensity = result[key][roi_y, :] / reference
            panels.append(10 * np.log10(np.maximum(intensity, 10 ** (-3.5))))
        for column, data, label in [(0, panels[0], "Unfiltered pixel-scanned TDI image"),
                                    (1, panels[1], "Filtered pixel-scanned TDI image")]:
            image = axes[row, column].imshow(
                data, origin="lower", extent=extent, cmap="turbo", vmin=-35, vmax=0, aspect="auto"
            )
            axes[row, column].set_title(
                f"p={result['period_um']:.1f} um "
                f"(bright {result['bright_width_um']:.2f} / dark {result['dark_width_um']:.2f} um)\n{label}",
                fontsize=9.5,
            )
            axes[row, column].set_xlabel("X: scan direction (um)")
            axes[row, column].set_ylabel("Y: camera long edge (um)")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046,
                         label="dB re same scanned-unfiltered peak")
    fig.suptitle(
        "Matched pixel-by-pixel scan: 25-stage TDI, 20x\n"
        "0.25 um adjacent object pixels, no overlap; 6.25 um illumination width; same scale; no subtraction",
        fontsize=13,
    )
    path = OUT / "small_period_tdi_pixel_scan_same_scale.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count, field_width = 512, 0.064
    dx = field_width / count
    coordinate = (np.arange(count) - count // 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    frequency = fft.fftshift(fft.fftfreq(count, d=dx))
    fxx, fyy = np.meshgrid(frequency, frequency)
    pupil = (((LAMBDA_MM * fxx) ** 2 + (LAMBDA_MM * fyy) ** 2) <= NA**2).astype(np.float32)
    grid = {
        "xx": xx,
        "yy": yy,
        "pupil": pupil,
        "x_hw": F_MM * LAMBDA_MM * frequency,
        "y_hw": MY * F_MM * LAMBDA_MM * frequency,
    }
    results = [simulate_period(period, grid, coordinate) for period in PERIODS_UM]
    plot_results(results, coordinate)
    summary = {
        "tdi_stages": TDI_STAGES,
        "image_pixel_pitch_um": PIXEL_PITCH_IMG_UM,
        "magnification": MAGNIFICATION,
        "tdi_width_image_um": TDI_WIDTH_IMG_UM,
        "illumination_and_acceptance_width_object_um": LINE_WIDTH_OBJ_UM,
        "output_pixel_pitch_object_um": OUTPUT_PIXEL_PITCH_OBJ_UM,
        "periods": [
            {key: value for key, value in result.items() if np.isscalar(value)} for result in results
        ],
        "evaluation_rule": "raw pixel-by-pixel TDI scan; adjacent output pixels; no overlap or periodic subtraction",
    }
    (OUT / "tdi_scan_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
