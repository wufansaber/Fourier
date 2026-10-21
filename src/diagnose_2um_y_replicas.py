"""Diagnose and reduce Y-periodic particle replicas for the 2 um pattern."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from small_period_raw_intensity_simulation import (
    CORE_MM,
    DEPTH_MM,
    F_MM,
    LAMBDA_MM,
    MY,
    NA,
    TOOTH_PITCH_MM,
    irregular_particle_field,
    periodic_pattern,
)
from small_period_tdi_scan_stitch import scan_pixel_by_pixel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "small_period_point_defects"
PERIOD_UM = 2.0
ORDER_CASES = {
    "1 soft rod: 0": [0],
    "3 soft rods: 0,+/-1": [-1, 0, 1],
    "5 soft rods: 0,+/-1,+/-2": [-2, -1, 0, 1, 2],
    "7 soft rods: 0...+/-3": [-3, -2, -1, 0, 1, 2, 3],
}


def ft2(field):
    return fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho", workers=4))


def ift2(field):
    return fft.fftshift(fft.ifft2(fft.ifftshift(field), norm="ortho", workers=4))


def rod_mask(x_hw, y_hw, spacing_mm, active_orders):
    phase = x_hw / TOOTH_PITCH_MM
    triangle = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    half_width = CORE_MM / 2 + DEPTH_MM * 0.5 * (triangle + 1.0)
    mask = np.ones((y_hw.size, x_hw.size), dtype=np.float32)
    for order in active_orders:
        mask[np.abs(y_hw[:, None] - order * spacing_mm) <= half_width[None, :]] = 0.0
    return mask


def zero_order_soft_mask(x_hw, y_hw, spacing_mm, active_orders):
    """X-averaged serration transmission seen when TDI accepts only serration order zero."""
    transmission_y = np.ones_like(y_hw, dtype=np.float32)
    for order in active_orders:
        distance = np.abs(y_hw - order * spacing_mm)
        one_rod = np.ones_like(y_hw, dtype=np.float32)
        one_rod[distance <= CORE_MM / 2] = 0.0
        transition = (distance > CORE_MM / 2) & (distance < CORE_MM / 2 + DEPTH_MM)
        one_rod[transition] = (distance[transition] - CORE_MM / 2) / DEPTH_MM
        transmission_y *= one_rod
    return np.repeat(transmission_y[:, None], x_hw.size, axis=1)


def image_metrics(actual_image, background_image, open_actual, open_background, x_out, y, label, order_count):
    analysis_y = np.abs(y) <= 0.008
    main = (np.abs(x_out[None, :] - 0.012) <= 0.00075) & (np.abs(y[:, None]) <= 0.00065)
    echo_2 = (np.abs(x_out[None, :] - 0.012) <= 0.00075) & (
        (np.abs(y[:, None] - 0.002) <= 0.00045) | (np.abs(y[:, None] + 0.002) <= 0.00045)
    )
    all_particles = np.zeros_like(actual_image, dtype=bool)
    for centre_x in (-0.012, 0.0, 0.012):
        all_particles |= (np.abs(x_out[None, :] - centre_x) <= 0.0012) & (np.abs(y[:, None]) <= 0.0012)
    background_region = analysis_y[:, None] & ~all_particles
    main_peak = float(np.max(actual_image[main]))
    open_main_peak = float(np.max(open_actual[main]))
    echo_peak = float(np.max(actual_image[echo_2]))
    pattern95 = float(np.percentile(actual_image[background_region], 95))
    return {
        "case": label,
        "rod_count": order_count,
        "raw_background_suppression_db": float(10 * np.log10(
            np.sum(open_background[analysis_y, :]) / np.sum(background_image[analysis_y, :])
        )),
        "raw_particle_peak_retention": main_peak / max(open_main_peak, 1e-30),
        "raw_particle_to_pattern_db": float(10 * np.log10(main_peak / max(pattern95, 1e-30))),
        "raw_y_echo_at_2um_db": float(10 * np.log10(echo_peak / max(main_peak, 1e-30))),
    }


def plot_images(images, reference, x_out, y):
    roi_y = np.abs(y) <= 0.008
    extent = [x_out[0] * 1000, x_out[-1] * 1000, -8, 8]
    cases = ["Unfiltered"] + list(ORDER_CASES) + ["7 rods: full 2-D serration"]
    fig, axes = plt.subplots(1, len(cases), figsize=(19, 4.8), constrained_layout=True)
    for axis, name in zip(axes, cases):
        data = images[name][roi_y, :] / reference
        data_db = 10 * np.log10(np.maximum(data, 10 ** (-3.5)))
        image = axis.imshow(data_db, origin="lower", extent=extent, cmap="turbo", vmin=-35, vmax=0,
                            aspect="auto")
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("X scan (um)")
        axis.set_ylabel("Y camera long edge (um)")
        fig.colorbar(image, ax=axis, fraction=0.046, label="dB re same unfiltered peak")
    fig.suptitle(
        "2 um pattern: raw pixel-scanned TDI intensity versus active rod orders\n"
        "All panels share one reference and scale; no periodic-reference subtraction",
        fontsize=13,
    )
    path = OUT / "diagnostic_2um_active_order_comparison.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_impulse_profiles(pupil, masks, coordinate):
    centre_x = coordinate.size // 2
    fig, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for name, mask in masks.items():
        psf = np.abs(ift2(pupil * mask)) ** 2
        profile = psf[:, centre_x]
        profile /= max(float(profile.max()), 1e-30)
        axis.plot(coordinate * 1000, 10 * np.log10(np.maximum(profile, 1e-8)), label=name)
    for multiple in range(-4, 5):
        axis.axvline(2 * multiple, color="#999999", linewidth=0.5, linestyle=":" if multiple else "--")
    axis.set_xlim(-10, 10)
    axis.set_ylim(-80, 0)
    axis.set_xlabel("Y offset from particle (um)")
    axis.set_ylabel("Normalized raw PSF intensity (dB)")
    axis.set_title("Y impulse-response replicas align with the 2 um pattern pitch")
    axis.grid(color="#D5D8DC", linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    path = OUT / "diagnostic_2um_y_impulse_profiles.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_fourier_planes(actual, pupil, mask, x_hw, y_hw):
    """Show that the filter removes existing spectrum but creates no new BFP orders."""
    spectrum_open = ft2(actual) * pupil
    spectrum_filtered = spectrum_open * mask
    intensity_open = np.abs(spectrum_open) ** 2
    intensity_filtered = np.abs(spectrum_filtered) ** 2
    reference = max(float(intensity_open.max()), 1e-30)
    open_db = 10 * np.log10(np.maximum(intensity_open / reference, 1e-8))
    filtered_db = 10 * np.log10(np.maximum(intensity_filtered / reference, 1e-8))
    extent = [x_hw[0], x_hw[-1], y_hw[0], y_hw[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    first = axes[0].imshow(open_db, origin="lower", extent=extent, cmap="magma", vmin=-80, vmax=0,
                           aspect="auto")
    axes[0].set_title("Raw Fourier-plane intensity\nbefore rods")
    fig.colorbar(first, ax=axes[0], fraction=0.046, label="dB re same unfiltered spectral peak")
    second = axes[1].imshow(mask, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1,
                            aspect="auto")
    axes[1].set_title("Seven-rod TDI-zero-order\nsoft transmission mask")
    fig.colorbar(second, ax=axes[1], fraction=0.046, label="Amplitude transmission")
    third = axes[2].imshow(filtered_db, origin="lower", extent=extent, cmap="magma", vmin=-80, vmax=0,
                           aspect="auto")
    axes[2].set_title("Raw Fourier-plane intensity\nafter rods")
    fig.colorbar(third, ax=axes[2], fraction=0.046, label="dB re same unfiltered spectral peak")
    profile_open = np.sum(intensity_open, axis=1)
    profile_filtered = np.sum(intensity_filtered, axis=1)
    profile_reference = max(float(profile_open.max()), 1e-30)
    axes[3].plot(y_hw, 10 * np.log10(np.maximum(profile_open / profile_reference, 1e-10)),
                 label="Before rods")
    axes[3].plot(y_hw, 10 * np.log10(np.maximum(profile_filtered / profile_reference, 1e-10)),
                 label="After rods")
    for order in range(-3, 4):
        axes[3].axvline(order * 15.0, color="#999999", linewidth=0.7, linestyle=":")
    axes[3].set(xlim=(-52, 52), ylim=(-100, 3), xlabel="Mechanical Fourier-plane Y (mm)",
                ylabel="X-integrated raw spectral intensity (dB)",
                title="Y spectrum: rods suppress peaks;\nno new order is generated")
    axes[3].grid(color="#D5D8DC", linewidth=0.6)
    axes[3].legend(frameon=False)
    for axis in axes[:3]:
        axis.set_xlabel("Mechanical Fourier-plane X (mm)")
        axis.set_ylabel("Mechanical Fourier-plane Y (mm)")
        axis.set_xlim(-20, 20)
        axis.set_ylim(-52, 52)
    fig.suptitle("2 um pattern Fourier-plane audit | raw spectra, common reference, no subtraction", fontsize=14)
    path = OUT / "diagnostic_2um_fourier_plane_audit.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_metrics(rows):
    labels = [row["case"].split(":")[0] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(x - 0.18, [row["raw_background_suppression_db"] for row in rows], width=0.36,
                label="Background suppression (dB)")
    axes[0].bar(x + 0.18, [100 * row["raw_particle_peak_retention"] for row in rows], width=0.36,
                label="Particle peak retention (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Raw-image trade-off")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#D5D8DC", linewidth=0.6)
    axes[1].plot(x, [row["raw_y_echo_at_2um_db"] for row in rows], "o-", label="Y echo at +/-2 um")
    axes[1].plot(x, [row["raw_particle_to_pattern_db"] for row in rows], "s-", label="Particle/pattern")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Raw intensity ratio (dB)")
    axes[1].set_title("Replica versus useful contrast")
    axes[1].legend(frameon=False)
    axes[1].grid(color="#D5D8DC", linewidth=0.6)
    path = OUT / "diagnostic_2um_order_tradeoff.png"
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
    spacing_mm = MY * F_MM * LAMBDA_MM / (PERIOD_UM / 1000)
    x_hw = F_MM * LAMBDA_MM * frequency
    y_hw = MY * F_MM * LAMBDA_MM * frequency
    background = periodic_pattern(xx, yy, PERIOD_UM / 1000)
    actual, _ = irregular_particle_field(background, xx, yy)
    open_mask = np.ones_like(pupil)
    open_actual, _, x_out = scan_pixel_by_pixel(actual, pupil, open_mask, coordinate)
    open_background, _, _ = scan_pixel_by_pixel(background, pupil, open_mask, coordinate)
    images = {"Unfiltered": open_actual}
    masks = {}
    rows = []
    for label, orders in ORDER_CASES.items():
        mask = zero_order_soft_mask(x_hw, y_hw, spacing_mm, orders)
        masks[label] = mask
        _, filtered_actual, _ = scan_pixel_by_pixel(actual, pupil, mask, coordinate)
        _, filtered_background, _ = scan_pixel_by_pixel(background, pupil, mask, coordinate)
        images[label] = filtered_actual
        rows.append(image_metrics(filtered_actual, filtered_background, open_actual, open_background,
                                  x_out, coordinate, label, len(orders)))
    full_label = "7 rods: full 2-D serration"
    full_orders = ORDER_CASES["7 soft rods: 0...+/-3"]
    full_mask = rod_mask(x_hw, y_hw, spacing_mm, full_orders)
    masks[full_label] = full_mask
    _, full_actual, _ = scan_pixel_by_pixel(actual, pupil, full_mask, coordinate)
    _, full_background, _ = scan_pixel_by_pixel(background, pupil, full_mask, coordinate)
    images[full_label] = full_actual
    rows.append(image_metrics(full_actual, full_background, open_actual, open_background,
                              x_out, coordinate, full_label, len(full_orders)))
    plot_images(images, max(float(open_actual.max()), 1e-30), x_out, coordinate)
    plot_fourier_planes(actual, pupil, masks["7 soft rods: 0...+/-3"], x_hw, y_hw)
    plot_impulse_profiles(pupil, masks, coordinate)
    plot_metrics(rows)
    with (OUT / "diagnostic_2um_order_tradeoff.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "period_um": PERIOD_UM,
        "fourier_order_spacing_mm": spacing_mm,
        "cause": "the Fourier-plane filter creates no new object order; the earlier replicas must be separated into full-serration diffraction and residual soft-notch PSF sidelobes",
        "rows": rows,
    }
    (OUT / "diagnostic_2um_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
