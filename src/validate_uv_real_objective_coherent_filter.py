"""Coherent Fourier filtering driven by the final real Zemax objective pupil.

The optical transfer is not replaced by an ideal lens.  This calculation uses:
1) the 401-ray physical BFP mapping exported from the final .zos model;
2) the 3209-ray two-dimensional OPD pupil exported from that model;
3) the measured 36 mm physical pupil and NA=0.6 support.

Native OpticStudio POP is retained as a separate audit, but its paraxial
surface-to-surface propagation does not reconstruct the 2 um extended object
through this NA=0.6, 4.67 mm-WD objective.  The model-sourced coherent pupil is
therefore the valid Fourier-filter calculation used here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
UV_OUT = ROOT / "results" / "uv_real_objective"
OUT_DEFAULT = ROOT / "results" / "uv_real_objective_fourier_validation"

LAMBDA_MM = 0.000355
NA = 0.6
MAG = 20.0
PERIOD_MM = 0.002
BRIGHT_MM = PERIOD_MM * 4.0 / 7.0
DARK_MM = PERIOD_MM - BRIGHT_MM
ILLUMINATION_WIDTH_MM = 25 * 0.005 / MAG  # 25 TDI stages, 5 um image pixels
CORE_MM = 1.0
DEPTH_MM = 0.5
TOOTH_PITCH_MM = 0.75
ORDERS = tuple(range(-3, 4))
FLOOR_DB = -45.0


def pixel_rectangle(values, centre, width, step):
    return np.clip((width / 2 + step / 2 - np.abs(values - centre)) / step, 0, 1)


def object_fields(n: int, window_mm: float):
    dx = window_mm / n
    coordinate = (np.arange(n) - n / 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    distance = np.abs((yy + PERIOD_MM / 2) % PERIOD_MM - PERIOD_MM / 2)
    stripes = np.clip((BRIGHT_MM / 2 + dx / 2 - distance) / dx, 0, 1)
    illumination = pixel_rectangle(xx, 0.0, ILLUMINATION_WIDTH_MM, dx)
    regular = (stripes * illumination).astype(np.float32)

    angle = np.arctan2(yy - PERIOD_MM / 2, xx)
    radius = np.hypot(xx / 1.10, (yy - PERIOD_MM / 2) / 0.82)
    boundary = 0.00030 * (1.0 + 0.18 * np.sin(3 * angle + 0.4)
                          + 0.10 * np.sin(5 * angle - 0.7))
    particle = (radius <= boundary).astype(np.float32) * illumination
    defect = np.maximum(regular, particle).astype(np.float32)
    return coordinate, regular, defect, particle


def radial_average(radius, values):
    key = np.round(radius, 10)
    unique, inverse = np.unique(key, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique, sums / counts


def zemax_transfer(n: int, window_mm: float):
    fan = np.genfromtxt(
        OUT_DEFAULT / "real_ray_abbe_and_fourier_fan.csv",
        delimiter=",", names=True, encoding="utf-8-sig")
    positive = fan[fan["sin_object"] >= 0]
    positive = np.sort(positive, order="sin_object")
    sin_axis = positive["sin_object"]
    map_axis = np.abs(positive["fourier_y_mm"])

    pupil_data = np.load(UV_OUT / "rays" / "field_0.npz")
    samples = pupil_data["samples"]
    image = pupil_data["image"]
    valid = (image[:, 1] == 0) & (image[:, 2] == 0)
    norm_radius = np.hypot(samples[valid, 0], samples[valid, 1])
    opd_radius, opd_mean = radial_average(norm_radius, image[valid, 12])

    step = window_mm / n
    frequency = fft.fftfreq(n, d=step)
    fx, fy = np.meshgrid(frequency, frequency)
    sine_x = LAMBDA_MM * fx
    sine_y = LAMBDA_MM * fy
    sine_radius = np.hypot(sine_x, sine_y)
    pupil = sine_radius <= NA
    normalized_radius = np.interp(
        sine_radius, sin_axis, positive["normalized_pupil_y"], left=0, right=1)
    normalized_radius = np.abs(normalized_radius)
    opd = np.interp(normalized_radius, opd_radius, opd_mean).astype(np.float32)
    opd -= float(np.mean(opd[pupil]))

    map_radius = np.interp(sine_radius, sin_axis, map_axis, left=0, right=map_axis[-1])
    bfp_x = map_radius * np.divide(sine_x, sine_radius,
                                  out=np.zeros_like(sine_radius), where=sine_radius > 0)
    bfp_y = map_radius * np.divide(sine_y, sine_radius,
                                  out=np.zeros_like(sine_radius), where=sine_radius > 0)
    transfer = pupil.astype(np.complex64) * np.exp(2j * np.pi * opd)
    return transfer.astype(np.complex64), pupil, opd, bfp_x, bfp_y, frequency


def serrated_mask(bfp_x, bfp_y, centres):
    phase = bfp_x / TOOTH_PITCH_MM
    triangle = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    half_width = CORE_MM / 2 + DEPTH_MM * 0.5 * (triangle + 1.0)
    mask = np.ones(bfp_x.shape, dtype=bool)
    for centre in centres:
        mask &= np.abs(bfp_y - centre) > half_width
    return mask


def propagate(field, transfer, mask):
    spectrum = fft.fft2(fft.ifftshift(field), workers=4)
    open_field = fft.fftshift(fft.ifft2(spectrum * transfer, workers=4))
    filtered_field = fft.fftshift(fft.ifft2(spectrum * transfer * mask, workers=4))
    return spectrum, open_field, filtered_field


def metrics(coordinate, regular_open, regular_filtered, defect_open, defect_filtered):
    xx, yy = np.meshgrid(coordinate, coordinate)
    roi = ((np.abs(xx) <= ILLUMINATION_WIDTH_MM / 2) & (np.abs(yy) <= 0.012))
    particle_roi = roi & (xx ** 2 + (yy - PERIOD_MM / 2) ** 2 <= 0.00075 ** 2)
    pattern_roi = roi & ~particle_roi
    step = abs(coordinate[1] - coordinate[0])
    open_background_power = float(np.sum(regular_open[roi]) * step * step)
    filtered_background_power = float(np.sum(regular_filtered[roi]) * step * step)
    open_particle_peak = float(np.max(defect_open[particle_roi]))
    filtered_particle_peak = float(np.max(defect_filtered[particle_roi]))
    open_pattern95 = float(np.percentile(defect_open[pattern_roi], 95))
    filtered_pattern95 = float(np.percentile(defect_filtered[pattern_roi], 95))
    filtered_particle_peak = float(np.max(defect_filtered[particle_roi]))
    ghost_ratios = []
    for order in range(-6, 7):
        if order == 0:
            continue
        ghost_y = PERIOD_MM / 2 + order * PERIOD_MM
        if abs(ghost_y) > 0.012:
            continue
        ghost_roi = roi & (xx ** 2 + (yy - ghost_y) ** 2 <= 0.00075 ** 2)
        ghost_ratios.append(float(np.max(defect_filtered[ghost_roi]) /
                                  max(filtered_particle_peak, 1e-30)))
    worst_ghost = max(ghost_ratios)
    return {
        "evaluation": "raw intensity only; no periodic-background subtraction",
        "background_suppression_db": float(10 * np.log10(
            open_background_power / max(filtered_background_power, 1e-30))),
        "particle_peak_retention_percent": float(
            100 * filtered_particle_peak / max(open_particle_peak, 1e-30)),
        "particle_to_pattern_open_db": float(10 * np.log10(
            open_particle_peak / max(open_pattern95, 1e-30))),
        "particle_to_pattern_filtered_db": float(10 * np.log10(
            filtered_particle_peak / max(filtered_pattern95, 1e-30))),
        "particle_contrast_improvement_db": float(10 * np.log10(
            filtered_particle_peak / max(filtered_pattern95, 1e-30)) -
            10 * np.log10(open_particle_peak / max(open_pattern95, 1e-30))),
        "worst_adjacent_comb_ghost_percent_of_particle_peak": 100 * worst_ghost,
        "worst_adjacent_comb_ghost_db_relative_to_particle": float(10 * np.log10(worst_ghost)),
        "common_unfiltered_peak": float(max(np.max(defect_open), np.max(regular_open))),
    }


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_images(out, coordinate, defect_open, defect_filtered, result):
    scale = result["common_unfiltered_peak"]
    before = defect_open / scale
    after = defect_filtered / scale
    db = [10 * np.log10(np.maximum(values, 10 ** (FLOOR_DB / 10)))
          for values in (before, after)]
    extent = [coordinate[0] * 1000, coordinate[-1] * 1000,
              coordinate[0] * 1000, coordinate[-1] * 1000]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    linear = []
    for ax, values, title in zip(
            axes[0], (before, after),
            ("Unfiltered raw intensity", "Seven aligned serrated rods: raw intensity")):
        im = ax.imshow(values, extent=extent, origin="lower", cmap="magma",
                       vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        linear.append(im)
        ax.set(xlim=(-3.125, 3.125), ylim=(-8, 8),
               xlabel="object X (um): TDI scan / illumination width",
               ylabel="object Y (um): camera long edge", title=title)
        ax.plot(0, 1, marker="o", ms=11, mfc="none", mec="cyan", mew=1.2)
    cb = fig.colorbar(linear[0], ax=axes[0], shrink=0.9)
    cb.set_label("I / same unfiltered raw-image peak")
    logarithmic = []
    for ax, values, title in zip(
            axes[1], db,
            ("Unfiltered, common-reference dB", "Filtered, common-reference dB")):
        im = ax.imshow(values, extent=extent, origin="lower", cmap="turbo",
                       vmin=FLOOR_DB, vmax=0, interpolation="nearest", aspect="auto")
        logarithmic.append(im)
        ax.set(xlim=(-3.125, 3.125), ylim=(-8, 8),
               xlabel="object X (um): TDI scan / illumination width",
               ylabel="object Y (um): camera long edge", title=title)
        ax.plot(0, 1, marker="o", ms=11, mfc="none", mec="white", mew=1.2)
    cb = fig.colorbar(logarithmic[0], ax=axes[1], shrink=0.9)
    cb.set_label("dB relative to the same unfiltered raw-image peak")
    fig.suptitle(
        "Coherent filter result using the final Zemax real-objective OPD and BFP mapping\n"
        "2 um pattern (bright 1.143 / dark 0.857 um) + one irregular bright particle; "
        "25-stage matched TDI; no background subtraction\n"
        f"background suppression {result['background_suppression_db']:.2f} dB | "
        f"raw particle peak retained {result['particle_peak_retention_percent']:.1f}% | "
        f"contrast gain {result['particle_contrast_improvement_db']:.2f} dB | "
        f"worst comb ghost {result['worst_adjacent_comb_ghost_db_relative_to_particle']:.2f} dB",
        fontsize=12.5,
    )
    save_figure(fig, out / "zemax_pupil_filter_same_intensity_scale")


def plot_fourier(out, spectrum, transfer, mask, bfp_x, bfp_y, centres):
    before = np.abs(spectrum * transfer) ** 2
    after = before * mask
    before = fft.fftshift(before)
    after = fft.fftshift(after)
    x = fft.fftshift(bfp_x[0])
    y = fft.fftshift(bfp_y[:, 0])
    # Outside NA the radial interpolation is deliberately clamped, so the
    # coordinate arrays contain flat repeated end values.  Keep only the
    # central strictly monotonic pupil segment before drawing with imshow.
    increasing_x = np.where(np.diff(x) > 1e-10)[0]
    increasing_y = np.where(np.diff(y) > 1e-10)[0]
    mono_x = np.arange(increasing_x[0], increasing_x[-1] + 2)
    mono_y = np.arange(increasing_y[0], increasing_y[-1] + 2)
    select_x = mono_x[np.abs(x[mono_x]) <= 4]
    select_y = mono_y[np.abs(y[mono_y]) <= 18.2]
    before = before[np.ix_(select_y, select_x)]
    after = after[np.ix_(select_y, select_x)]
    reference = float(before.max())
    arrays = [10 * np.log10(np.maximum(values / reference, 10 ** (-60 / 10)))
              for values in (before, after)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), constrained_layout=True)
    images = []
    for ax, values, title in zip(
            axes, arrays, ("Before rods", "After seven physical serrated rods")):
        im = ax.imshow(values, origin="lower",
                       extent=[x[select_x[0]], x[select_x[-1]],
                               y[select_y[0]], y[select_y[-1]]],
                       cmap="turbo", vmin=-60, vmax=0, interpolation="nearest",
                       aspect="auto")
        images.append(im)
        ax.set(xlabel="physical BFP X (mm): rod axis / serration",
               ylabel="physical BFP Y (mm): sample orders / rod position", title=title)
        for order, centre in zip(ORDERS, centres):
            ax.axhline(centre, color="white", lw=0.65, alpha=0.8)
            ax.text(-3.8, centre + 0.15, f"m={order}", color="white", fontsize=7)
    cb = fig.colorbar(images[0], ax=axes, shrink=0.86)
    cb.set_label("spectral intensity, dB to the same pre-filter peak")
    fig.suptitle(
        "Real-Zemax physical BFP mapping: 2 um order peaks aligned to movable rods\n"
        "Rods long in X, independently positioned in Y; true 36 mm circular pupil",
        fontsize=13,
    )
    save_figure(fig, out / "zemax_pupil_fourier_alignment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grid", type=int, default=2048)
    parser.add_argument("--window-mm", type=float, default=0.15)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    coordinate, regular, defect, particle = object_fields(args.grid, args.window_mm)
    transfer, pupil, opd, bfp_x, bfp_y, frequency = zemax_transfer(
        args.grid, args.window_mm)
    fan = np.genfromtxt(OUT_DEFAULT / "real_ray_abbe_and_fourier_fan.csv",
                        delimiter=",", names=True, encoding="utf-8-sig")
    positive = fan[fan["sin_object"] >= 0]
    slope = float(np.sum(positive["sin_object"] * positive["fourier_y_mm"]) /
                  np.sum(positive["sin_object"] ** 2))
    spacing = abs(slope) * LAMBDA_MM / PERIOD_MM
    centres = np.asarray(ORDERS) * spacing
    mask = serrated_mask(bfp_x, bfp_y, centres) & pupil

    regular_spectrum, regular_open_field, regular_filtered_field = propagate(
        regular, transfer, mask)
    defect_spectrum, defect_open_field, defect_filtered_field = propagate(
        defect, transfer, mask)
    regular_open = np.abs(regular_open_field) ** 2
    regular_filtered = np.abs(regular_filtered_field) ** 2
    defect_open = np.abs(defect_open_field) ** 2
    defect_filtered = np.abs(defect_filtered_field) ** 2
    result = metrics(coordinate, regular_open, regular_filtered,
                     defect_open, defect_filtered)
    result.update({
        "engine": "coherent FFT using final Zemax real-ray pupil mapping and 2D OPD",
        "grid": args.grid,
        "window_mm": args.window_mm,
        "object_sample_um": args.window_mm / args.grid * 1000,
        "zemax_opd_rms_waves": float(np.std(opd[pupil])),
        "measured_bfp_sine_slope_mm": slope,
        "two_um_order_spacing_mm": spacing,
        "rod_centres_y_mm": centres.tolist(),
        "rod_core_mm": CORE_MM,
        "one_sided_tooth_depth_mm": DEPTH_MM,
        "tooth_pitch_x_mm": TOOTH_PITCH_MM,
        "pupil_diameter_mm": 36.0,
        "tdi_object_width_um": ILLUMINATION_WIDTH_MM * 1000,
    })
    (out / "zemax_pupil_filter_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "zemax_pupil_filter_metrics.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    if not args.metrics_only:
        np.savez_compressed(
            out / "zemax_pupil_filter_images.npz", coordinate=coordinate,
            object_regular=regular, object_defect=defect, particle=particle,
            regular_open=regular_open, regular_filtered=regular_filtered,
            defect_open=defect_open, defect_filtered=defect_filtered)
        plot_images(out, coordinate, defect_open, defect_filtered, result)
        plot_fourier(out, defect_spectrum, transfer, mask, bfp_x, bfp_y, centres)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
