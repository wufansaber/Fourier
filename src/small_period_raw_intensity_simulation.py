"""Raw-intensity 2--5 um periodic-pattern simulation with irregular bright particles.

No periodic-reference subtraction is used in any image or target metric.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "small_period_point_defects"
LAMBDA_MM, F_MM, NA, MY = 0.000355, 30.0, 0.60, 2.816901408450704
CORE_MM, DEPTH_MM, TOOTH_PITCH_MM = 1.0, 0.5, 0.75
PERIODS_UM = [2.0, 3.0, 4.0, 5.0]


def ft2(field):
    return fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho", workers=4))


def ift2(field):
    return fft.fftshift(fft.ifft2(fft.ifftshift(field), norm="ortho", workers=4))


def seven_rod_mask(x_hw, y_hw, spacing_mm):
    phase = x_hw / TOOTH_PITCH_MM
    triangle = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    half_width = CORE_MM / 2 + DEPTH_MM * 0.5 * (triangle + 1.0)
    mask = np.ones((y_hw.size, x_hw.size), dtype=np.float32)
    for order in range(-3, 4):
        mask[np.abs(y_hw[:, None] - order * spacing_mm) <= half_width[None, :]] = 0.0
    return mask


def periodic_pattern(xx, yy, period_mm):
    wrapped = np.remainder(yy + period_mm / 2, period_mm) - period_mm / 2
    amplitude = np.ones_like(xx, dtype=np.float32)
    # Bright:dark width = 4:3, so neither part becomes an unrealistically
    # narrow line compared with the other.
    amplitude[np.abs(wrapped) <= (3.0 / 14.0) * period_mm] = 0.12
    radius = np.maximum(np.abs(xx), np.abs(yy))
    taper = np.ones_like(radius, dtype=np.float32)
    taper[radius >= 0.060] = 0.0
    transition = (radius > 0.052) & (radius < 0.060)
    taper[transition] = 0.5 + 0.5 * np.cos(np.pi * (radius[transition] - 0.052) / 0.008)
    return amplitude * taper


def irregular_particle_field(background, xx, yy):
    field = background.astype(np.complex64)
    specifications = [(-0.012, 0.0, 0.60, 0.35), (0.000, 0.0, 0.90, -0.55), (0.012, 0.0, 1.20, 0.80)]

    def gaussian(cx, cy, sx, sy, angle):
        dx, dy = xx - cx, yy - cy
        cosine, sine = np.cos(angle), np.sin(angle)
        xr, yr = cosine * dx + sine * dy, -sine * dx + cosine * dy
        return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    for cx, cy, size_um, angle in specifications:
        scale = size_um / 1000
        core = gaussian(cx, cy, 0.25 * scale, 0.16 * scale, angle)
        lobe_a = gaussian(cx + 0.18 * scale * np.cos(angle), cy + 0.18 * scale * np.sin(angle),
                          0.18 * scale, 0.11 * scale, angle + 0.9)
        lobe_b = gaussian(cx - 0.16 * scale * np.sin(angle), cy + 0.16 * scale * np.cos(angle),
                          0.14 * scale, 0.09 * scale, angle - 0.7)
        shape = core + 0.70 * lobe_a + 0.45 * lobe_b
        amplitude = 2.0 * (size_um / 0.90) ** 0.25
        field += amplitude * shape * np.exp(1j * (np.pi / 2 + 0.25 * shape))
    return field, specifications


def simulate(period_um, grid):
    xx, yy = grid["xx"], grid["yy"]
    period_mm = period_um / 1000
    spacing_mm = MY * F_MM * LAMBDA_MM / period_mm
    rods = seven_rod_mask(grid["x_hw"], grid["y_hw"], spacing_mm)
    background = periodic_pattern(xx, yy, period_mm)
    actual, particles = irregular_particle_field(background, xx, yy)
    background_open = np.abs(ift2(ft2(background) * grid["pupil"])) ** 2
    background_filtered = np.abs(ift2(ft2(background) * grid["pupil"] * rods)) ** 2
    actual_open = np.abs(ift2(ft2(actual) * grid["pupil"])) ** 2
    actual_filtered = np.abs(ift2(ft2(actual) * grid["pupil"] * rods)) ** 2

    analysis_region = (np.abs(xx) <= 0.045) & (np.abs(yy) <= 0.045)
    particle_region = np.zeros_like(analysis_region)
    for cx, cy, size_um, _ in particles:
        radius_mm = max(0.0015, 1.5 * size_um / 1000)
        particle_region |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_mm**2
    pattern_region = analysis_region & ~particle_region
    peak_open = float(np.max(actual_open[particle_region]))
    peak_filtered = float(np.max(actual_filtered[particle_region]))
    pattern95_open = float(np.percentile(actual_open[pattern_region], 95))
    pattern95_filtered = float(np.percentile(actual_filtered[pattern_region], 95))
    return {
        "period_um": period_um, "bright_width_um": period_um * 4.0 / 7.0,
        "dark_width_um": period_um * 3.0 / 7.0, "spacing_mm": spacing_mm,
        "actual_open": actual_open, "actual_filtered": actual_filtered,
        "background_suppression_db": float(10 * np.log10(
            np.sum(background_open[analysis_region]) / np.sum(background_filtered[analysis_region]))),
        "raw_particle_peak_retention": peak_filtered / max(peak_open, 1e-30),
        "raw_particle_to_pattern_open_db": float(10 * np.log10(peak_open / max(pattern95_open, 1e-30))),
        "raw_particle_to_pattern_filtered_db": float(10 * np.log10(
            peak_filtered / max(pattern95_filtered, 1e-30))),
        "max_propagating_order": int(np.floor(NA * period_mm / LAMBDA_MM)),
    }


def plot_raw_before_after(results, coordinate):
    roi_x, roi_y = np.abs(coordinate) <= 0.018, np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]
    global_reference = max(float(result["actual_open"].max()) for result in results)
    fig, axes = plt.subplots(len(results), 2, figsize=(9.8, 9.8), constrained_layout=True)
    for row, result in enumerate(results):
        images = []
        for key in ["actual_open", "actual_filtered"]:
            raw = result[key][np.ix_(roi_y, roi_x)] / global_reference
            images.append(10 * np.log10(np.maximum(raw, 10 ** (-3.5))))
        for column, data, title in [(0, images[0], "Unfiltered raw intensity"),
                                    (1, images[1], "Filtered raw intensity")]:
            image = axes[row, column].imshow(data, origin="lower", extent=extent, cmap="turbo",
                                              vmin=-35, vmax=0, aspect="auto")
            axes[row, column].set_title(f"{result['period_um']:.0f} um period | {title}", fontsize=10)
            axes[row, column].set_xlabel("X (um)")
            axes[row, column].set_ylabel("Y (um)")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046,
                         label="dB re one global unfiltered peak")
    fig.suptitle("2--5 um periodic patterns with irregular bright particles\n"
                 "One global reference and -35...0 dB scale for all panels; no background subtraction",
                 fontsize=13)
    path = OUT / "small_period_particle_raw_same_scale.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_global_linear_brightness(results, coordinate):
    """Plot every period and filter state on one honest linear intensity scale."""
    roi_x, roi_y = np.abs(coordinate) <= 0.018, np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]
    global_reference = max(float(result["actual_open"].max()) for result in results)
    fig, axes = plt.subplots(len(results), 2, figsize=(9.8, 9.8), constrained_layout=True)
    shown = None
    for row, result in enumerate(results):
        for column, key, title in (
            (0, "actual_open", "Unfiltered raw intensity"),
            (1, "actual_filtered", "Filtered raw intensity"),
        ):
            normalized = result[key][np.ix_(roi_y, roi_x)] / global_reference
            shown = axes[row, column].imshow(
                normalized, origin="lower", extent=extent, cmap="turbo",
                vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
            )
            axes[row, column].set_title(
                f"{result['period_um']:.0f} um period | {title}", fontsize=10
            )
            axes[row, column].set_xlabel("X (um)")
            axes[row, column].set_ylabel("Y (um)")
    fig.colorbar(
        shown, ax=axes, fraction=0.025, pad=0.02,
        label="Normalized raw intensity I / I_ref (one global unfiltered peak)"
    )
    fig.suptitle(
        "2--5 um periodic patterns with irregular bright particles\n"
        "One global linear intensity scale (0...1) for every panel; no background subtraction",
        fontsize=13,
    )
    path = OUT / "small_period_particle_global_linear_brightness.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_orders(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15.5, 4.8), constrained_layout=True)
    for axis, result in zip(np.atleast_1d(axes).flat, results):
        maximum_order = result["max_propagating_order"]
        orders = np.arange(-maximum_order, maximum_order + 1)
        positions = orders * result["spacing_mm"]
        strengths = np.sinc(0.28 * orders) ** 2
        for order, position, strength in zip(orders, positions, strengths):
            axis.vlines(position, 1e-6, max(strength, 1e-6), color="#2F5AA8", linewidth=1.5)
            if abs(order) <= 3:
                axis.axvspan(position - 1.0, position + 1.0, color="#D48A27", alpha=0.20)
        axis.axvline(-MY * F_MM * NA, color="#333333", linestyle="--")
        axis.axvline(MY * F_MM * NA, color="#333333", linestyle="--")
        axis.set(yscale="log", ylim=(1e-5, 2), xlim=(-55, 55),
                 xlabel="Mechanical Fourier-plane Y (mm)", ylabel="Relative periodic-order intensity")
        axis.set_title(f"p={result['period_um']:.0f} um: spacing={result['spacing_mm']:.1f} mm, "
                       f"propagating |m|<={maximum_order}")
        axis.grid(color="#D5D8DC", linewidth=0.6)
    fig.suptitle("Diffraction-order centres and seven 2 mm serrated-rod envelopes", fontsize=14)
    path = OUT / "small_period_fourier_orders.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_raw_metrics(results):
    periods = [r["period_um"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].plot(periods, [r["background_suppression_db"] for r in results], "o-",
                 label="Raw periodic-background suppression (dB)")
    axes[0].plot(periods, [100 * r["raw_particle_peak_retention"] for r in results], "s-",
                 label="Raw particle peak retention (%)")
    axes[0].set(xlabel="Pattern period (um)", title="Raw-image metrics; no image subtraction")
    axes[0].grid(color="#D5D8DC", linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[1].plot(periods, [r["raw_particle_to_pattern_open_db"] for r in results], "o-", label="Unfiltered")
    axes[1].plot(periods, [r["raw_particle_to_pattern_filtered_db"] for r in results], "s-", label="Filtered")
    axes[1].set(xlabel="Pattern period (um)", ylabel="Particle peak / pattern 95th percentile (dB)",
                title="Contrast measured directly in each raw image")
    axes[1].grid(color="#D5D8DC", linewidth=0.6)
    axes[1].legend(frameon=False)
    path = OUT / "small_period_raw_metrics.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count, field_width = 2048, 0.120
    dx = field_width / count
    coordinate = (np.arange(count) - count // 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    frequency = fft.fftshift(fft.fftfreq(count, d=dx))
    fxx, fyy = np.meshgrid(frequency, frequency)
    pupil = (((LAMBDA_MM * fxx) ** 2 + (LAMBDA_MM * fyy) ** 2) <= NA**2).astype(np.float32)
    grid = {"xx": xx, "yy": yy, "pupil": pupil, "x_hw": F_MM * LAMBDA_MM * frequency,
            "y_hw": MY * F_MM * LAMBDA_MM * frequency}
    results = [simulate(period, grid) for period in PERIODS_UM]
    plot_raw_before_after(results, coordinate)
    plot_global_linear_brightness(results, coordinate)
    plot_orders(results)
    plot_raw_metrics(results)
    rows = [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items() if np.isscalar(v)}
            for r in results]
    with (OUT / "small_period_raw_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "parameters": {"wavelength_nm": 355, "focal_length_mm": F_MM, "na": NA,
                       "fourier_y_magnification": MY, "rod_count": 7,
                       "active_orders": list(range(-3, 4)), "rod_envelope_mm": CORE_MM + 2 * DEPTH_MM,
                       "grid": [count, count], "fov_um": field_width * 1000},
        "results": rows,
        "evaluation_rule": "raw intensity only; no periodic-reference subtraction in images or target metrics",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
