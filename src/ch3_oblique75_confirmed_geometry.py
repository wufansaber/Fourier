"""Ch3 + 75 deg illumination using the user-confirmed detector/rod geometry.

Confirmed axes
--------------
X: camera long edge and illumination-line long axis.
Y: TDI scan/charge-transfer and effective narrow acceptance.
Fourier rods: long/serrated along Y_F, independently positioned along X_F.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ch3_oblique75_confirmed_geometry"

LAMBDA_MM = 0.000355
F_MM = 30.0
NA = 0.60
FOURIER_POSITION_MAG = 2.816901408450704
MAGNIFICATION = 20.0
PERIOD_MM = 0.002
INCIDENCE_DEG = 75.0
FY_INC = np.sin(np.deg2rad(INCIDENCE_DEG)) / LAMBDA_MM
FCUT = NA / LAMBDA_MM

CORE_MM = 1.0
DEPTH_MM = 0.5
TOOTH_PITCH_MM = 0.75
TDI_STAGES = 25
IMAGE_PIXEL_UM = 5.0
TDI_WIDTH_IMAGE_UM = TDI_STAGES * IMAGE_PIXEL_UM
TDI_WIDTH_OBJECT_UM = TDI_WIDTH_IMAGE_UM / MAGNIFICATION
SERRATION_ORDER_OBJECT_UM = 1000 * LAMBDA_MM * F_MM / TOOTH_PITCH_MM
SERRATION_ORDER_IMAGE_UM = MAGNIFICATION * SERRATION_ORDER_OBJECT_UM


def ft2(field):
    return fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho", workers=4))


def ift2(spectrum):
    return fft.fftshift(fft.ifft2(fft.ifftshift(spectrum), norm="ortho", workers=4))


def one_dimensional_pattern(coordinate):
    wrapped = np.remainder(coordinate + PERIOD_MM / 2, PERIOD_MM) - PERIOD_MM / 2
    amplitude = np.ones_like(coordinate, dtype=np.float32)
    amplitude[np.abs(wrapped) <= 3 * PERIOD_MM / 14] = 0.18
    return amplitude


def two_dimensional_pattern(xx, yy):
    """2-D periodic pattern so oblique +Y momentum can be compensated into Ch3."""
    px = one_dimensional_pattern(xx)
    py = one_dimensional_pattern(yy)
    return (0.35 * px + 0.65 * px * py).astype(np.float32)


def collected_lattice():
    values = []
    for m in range(-12, 13):
        for n in range(-12, 13):
            sx = m * LAMBDA_MM / PERIOD_MM
            sy = np.sin(np.deg2rad(INCIDENCE_DEG)) + n * LAMBDA_MM / PERIOD_MM
            if sx * sx + sy * sy <= NA * NA:
                values.append((m, n, sx, sy))
    return values


def rod_x_centres(lattice):
    active_m = sorted({m for m, _, _, _ in lattice})
    # Maximum seven rods and only columns that actually cross the Ch3 pupil.
    active_m = [m for m in active_m if -3 <= m <= 3]
    spacing = FOURIER_POSITION_MAG * F_MM * LAMBDA_MM / PERIOD_MM
    return active_m, np.asarray(active_m, dtype=float) * spacing, spacing


def averaged_serration_k0_mask(x_mechanical, y_bfp, centres):
    """Only the serration k=0 order accepted by the narrow-Y TDI window."""
    transmission_x = np.ones_like(x_mechanical, dtype=np.float32)
    for centre in centres:
        distance = np.abs(x_mechanical - centre)
        one = np.ones_like(x_mechanical, dtype=np.float32)
        one[distance <= CORE_MM / 2] = 0.0
        edge = (distance > CORE_MM / 2) & (distance < CORE_MM / 2 + DEPTH_MM)
        one[edge] = (distance[edge] - CORE_MM / 2) / DEPTH_MM
        transmission_x *= one
    return np.repeat(transmission_x[None, :], y_bfp.size, axis=0)


def physical_serrated_mask(x_mechanical, y_bfp, centres):
    phase = y_bfp / TOOTH_PITCH_MM
    triangle = 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1
    half_width = CORE_MM / 2 + DEPTH_MM * 0.5 * (triangle + 1)
    mask = np.ones((y_bfp.size, x_mechanical.size), dtype=np.float32)
    for centre in centres:
        mask[np.abs(x_mechanical[None, :] - centre) <= half_width[:, None]] = 0
    return mask


def particle_spectrum(fxx, fyy, pupil):
    radial = np.sqrt(fxx**2 + fyy**2) / FCUT
    envelope = np.exp(-0.5 * (radial / 0.82) ** 6) * pupil
    particles = [(-0.010, -0.004, 0.90, 0.15),
                 (0.000, 0.000, 1.00, -0.50),
                 (0.010, 0.004, 1.10, 0.72)]
    offsets = [(0.0, 0.0, 1.0, 0.0),
               (0.00015, 0.00007, 0.48, 0.55),
               (-0.00010, 0.00012, 0.35, -0.70)]
    total = np.zeros_like(fxx, dtype=np.complex64)
    single = []
    for cx, cy, strength, angle in particles:
        ca, sa = np.cos(angle), np.sin(angle)
        one = np.zeros_like(total)
        for ox, oy, weight, phase0 in offsets:
            rx, ry = ca * ox - sa * oy, sa * ox + ca * oy
            phase = np.exp(-2j * np.pi * (fxx * (cx + rx) + fyy * (cy + ry)))
            one += strength * weight * np.exp(1j * phase0) * envelope * phase
        total += one
        single.append(one)
    return total, single, particles


def spot_width(image, coordinate, cx=0.0, cy=0.0):
    xx, yy = np.meshgrid(coordinate, coordinate)
    roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= 0.002**2
    w = image * roi
    total = max(float(w.sum()), 1e-30)
    x0, y0 = float((w * xx).sum() / total), float((w * yy).sum() / total)
    sx = np.sqrt(float((w * (xx - x0) ** 2).sum() / total))
    sy = np.sqrt(float((w * (yy - y0) ** 2).sum() / total))
    factor = 2 * np.sqrt(2 * np.log(2)) * 1000
    return factor * sx, factor * sy


def plot_images(coordinate, open_image, filtered_image):
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    reference = max(float(open_image[np.ix_(roi, roi)].max()), 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4), constrained_layout=True)
    titles = ["No rods: reconstructed TDI image",
              "Seven rods: serration k=0 accepted TDI image"]
    for axis, raw, title in zip(axes, [open_image, filtered_image], titles):
        data = raw[np.ix_(roi, roi)] / reference
        shown = axis.imshow(10 * np.log10(np.maximum(data, 1e-4)), origin="lower",
                            extent=extent, cmap="turbo", vmin=-40, vmax=0, aspect="equal")
        axis.set(title=title, xlabel="X: camera / illumination-line long direction (um)",
                 ylabel="Y: TDI scan direction (um)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(shown, ax=axis, fraction=0.046,
                     label="Stage-normalised irradiance (dB a.u.; common reference)")
    fig.suptitle("Confirmed orientation | Ch3, 75 deg illumination projection +Y, 2 um 2-D pattern\n"
                 "Same raw-intensity scale; no background subtraction; no scan convolution")
    path = OUT / "confirmed_geometry_raw_before_after.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_fourier(pattern_spec, one_particle_spec, combined, filtered, mask,
                 x_mechanical, y_bfp, active_m, centres):
    spectra = [pattern_spec, one_particle_spec, combined, filtered]
    titles = ["2-D pattern only\n(discrete translated peaks)",
              "One particle only\n(continuous spectrum)",
              "Combined before rods", "Combined after k=0 rod mask"]
    reference = max(float(np.abs(combined).max() ** 2), 1e-30)
    extent = [x_mechanical[0], x_mechanical[-1], y_bfp[0], y_bfp[-1]]
    fig, axes = plt.subplots(1, 5, figsize=(20.7, 5.0), constrained_layout=True)
    for axis, spectrum, title in zip(axes[:4], spectra, titles):
        db = 10 * np.log10(np.maximum(np.abs(spectrum) ** 2 / reference, 1e-9))
        shown = axis.imshow(db, origin="lower", extent=extent, cmap="magma",
                            vmin=-90, vmax=0, aspect="auto")
        axis.set(title=title, xlim=(-51, 51), ylim=(-18, 18),
                 xlabel="Mechanical Fourier X (mm)", ylabel="Original BFP Y (mm)")
        for m, centre in zip(active_m, centres):
            axis.axvline(centre, color="#5DE2E7", lw=0.55, alpha=0.7)
            axis.text(centre + 0.6, -17, f"m={m}", color="#5DE2E7", fontsize=7, rotation=90)
    fig.colorbar(shown, ax=axes[:4], fraction=0.016,
                 label="Spectral intensity (dB; common combined-before reference)")
    mask_image = axes[4].imshow(mask, origin="lower", extent=extent, cmap="gray",
                                vmin=0, vmax=1, aspect="auto")
    axes[4].set(title="TDI-accepted k=0\nsoft-edge rod mask", xlim=(-51, 51), ylim=(-18, 18),
                xlabel="Mechanical Fourier X (mm)", ylabel="Original BFP Y (mm)")
    for m, centre in zip(active_m, centres):
        axes[4].axvline(centre, color="#E74C3C", lw=0.7)
        axes[4].text(centre + 0.6, -17, f"m={m}\n{centre:+.1f}", color="#E74C3C",
                     fontsize=7, rotation=90)
    fig.colorbar(mask_image, ax=axes[4], fraction=0.046, label="Amplitude transmission")
    fig.suptitle("Confirmed Fourier-plane directions: rods long Y_F and positioned along X_F")
    path = OUT / "confirmed_geometry_fourier_plane_and_rods.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_tdi_order_acceptance():
    y = np.linspace(-22, 22, 2600)
    sigma = 0.45
    fig, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    for k in range(-1, 2):
        centre = k * SERRATION_ORDER_OBJECT_UM
        profile = np.exp(-0.5 * ((y - centre) / sigma) ** 2)
        axis.plot(y, profile, lw=2.0, label=f"serration k={k}, centre {centre:+.1f} um")
    half = TDI_WIDTH_OBJECT_UM / 2
    axis.axvspan(-half, half, color="#2563EB", alpha=0.16,
                 label=f"25-stage TDI Y window: {TDI_WIDTH_OBJECT_UM:.2f} um object")
    axis.axvline(-half, color="#2563EB", lw=1.2)
    axis.axvline(half, color="#2563EB", lw=1.2)
    axis.set(xlim=(-22, 22), ylim=(-0.03, 1.08), xlabel="Object-side Y (um)",
             ylabel="Relative order envelope",
             title="Y-narrow TDI accepts only the central serration order")
    axis.grid(color="#D5D8DC", lw=0.6)
    axis.legend(frameon=False, ncol=2)
    axis.text(0, 0.62, "accepted k=0", ha="center", color="#1D4ED8")
    path = OUT / "confirmed_geometry_tdi_zero_order_acceptance.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count, field_width = 1024, 0.128
    dx = field_width / count
    coordinate = (np.arange(count) - count // 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    frequency = fft.fftshift(fft.fftfreq(count, d=dx))
    fxx, fyy = np.meshgrid(frequency, frequency)
    pupil = ((fxx**2 + fyy**2) <= FCUT**2).astype(np.float32)

    pattern = two_dimensional_pattern(xx, yy)
    illumination = np.exp(2j * np.pi * FY_INC * yy)
    pattern_spec = ft2(pattern * illumination) * pupil
    particles, single_particles, particle_list = particle_spectrum(fxx, fyy, pupil)
    particle_scale = 0.040 * np.sqrt(np.sum(np.abs(pattern_spec) ** 2) /
                                      max(np.sum(np.abs(particles) ** 2), 1e-30))
    particles *= particle_scale
    single_particles = [item * particle_scale for item in single_particles]
    combined = pattern_spec + particles

    lattice = collected_lattice()
    active_m, centres, spacing = rod_x_centres(lattice)
    x_mechanical = FOURIER_POSITION_MAG * F_MM * LAMBDA_MM * frequency
    y_bfp = F_MM * LAMBDA_MM * frequency
    k0_mask = averaged_serration_k0_mask(x_mechanical, y_bfp, centres)
    physical_mask = physical_serrated_mask(x_mechanical, y_bfp, centres)
    filtered = combined * k0_mask

    open_image = np.abs(ift2(combined)) ** 2
    filtered_image = np.abs(ift2(filtered)) ** 2
    single_particle_image = np.abs(ift2(single_particles[1])) ** 2
    fwhm_x, fwhm_y = spot_width(single_particle_image, coordinate)

    plot_images(coordinate, open_image, filtered_image)
    plot_fourier(pattern_spec, single_particles[1], combined, filtered, k0_mask,
                 x_mechanical, y_bfp, active_m, centres)
    plot_tdi_order_acceptance()

    summary = {
        "confirmed_axes": {
            "X": "camera long edge and illumination-line long axis",
            "Y": "TDI scan, charge transfer and effective narrow acceptance",
            "rod_long_axis": "Fourier Y",
            "rod_independent_positioning": "Fourier X",
            "illumination_sample_projection": "+Y",
        },
        "optics": {"wavelength_nm": 355, "objective_f_mm": F_MM, "NA": NA,
                   "magnification": MAGNIFICATION, "incidence_deg_from_normal": INCIDENCE_DEG},
        "pattern": {"period_um": PERIOD_MM * 1000, "type": "2-D periodic amplitude pattern",
                    "bright_dark_width_ratio": "4:3 in each periodic coordinate"},
        "rods": {"active_m_columns": active_m, "count": len(active_m),
                 "centres_mechanical_X_mm": centres.tolist(), "spacing_mm": spacing,
                 "core_mm": CORE_MM, "tooth_depth_each_side_mm": DEPTH_MM,
                 "tooth_pitch_mm": TOOTH_PITCH_MM},
        "tdi": {"stages": TDI_STAGES, "image_pixel_um": IMAGE_PIXEL_UM,
                "Y_width_image_um": TDI_WIDTH_IMAGE_UM,
                "Y_width_object_um": TDI_WIDTH_OBJECT_UM,
                "serration_order_spacing_object_um": SERRATION_ORDER_OBJECT_UM,
                "serration_order_spacing_image_um": SERRATION_ORDER_IMAGE_UM,
                "accepted_serration_orders": [0]},
        "single_particle_FWHM_um": {"X": fwhm_x, "Y": fwhm_y,
                                    "X_over_Y": fwhm_x / fwhm_y},
        "evaluation": "raw intensity, common before/after reference, no background subtraction",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
