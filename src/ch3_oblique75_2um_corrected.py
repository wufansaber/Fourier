"""Corrected Ch3 / 75-degree illumination / 2 um-pattern Fourier-filter model.

This model deliberately separates two physically different terms:

1. coherent diffraction of the periodic pattern under the oblique illumination;
2. broad angular dark-field scattering from local particles into Ch3.

The 25-stage velocity-matched TDI does not convolve neighbouring object columns.
Its stage-normalised image is therefore the optical intensity itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ch3_oblique75_2um_corrected"

LAMBDA_MM = 0.000355
F_MM = 30.0
NA = 0.60
MY = 2.816901408450704
PERIOD_MM = 0.002
INCIDENCE_DEG = 75.0
INCIDENCE_FY = np.sin(np.deg2rad(INCIDENCE_DEG)) / LAMBDA_MM
FCUT = NA / LAMBDA_MM

CORE_MM = 1.0
DEPTH_MM = 0.5
TOOTH_PITCH_MM = 0.75
TDI_STAGES = 25
MAGNIFICATION = 20.0
# Newly confirmed detector orientation: the effective narrow acceptance is Y.
# This provisional 1.5 um object-side width is smaller than the 2 um adjacent
# order spacing, so it implements the requested central-order-only check.
TDI_Y_ACCEPTANCE_OBJ_UM = 1.5


def ft2(field):
    return fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho", workers=4))


def ift2(spectrum):
    return fft.fftshift(fft.ifft2(fft.ifftshift(spectrum), norm="ortho", workers=4))


def periodic_amplitude(yy):
    """2 um binary pattern with bright:dark width = 4:3."""
    wrapped = np.remainder(yy + PERIOD_MM / 2, PERIOD_MM) - PERIOD_MM / 2
    amplitude = np.ones_like(yy, dtype=np.float32)
    amplitude[np.abs(wrapped) <= 3.0 * PERIOD_MM / 14.0] = 0.12
    return amplitude


def collected_orders():
    """Orders for which |sin(75 deg) + n*lambda/p| <= NA in Ch3."""
    candidates = np.arange(-30, 31)
    sine_coordinate = np.sin(np.deg2rad(INCIDENCE_DEG)) + candidates * LAMBDA_MM / PERIOD_MM
    keep = np.abs(sine_coordinate) <= NA
    orders = candidates[keep]
    fy = INCIDENCE_FY + orders / PERIOD_MM
    y_mechanical = MY * F_MM * LAMBDA_MM * fy
    return orders, fy, y_mechanical, sine_coordinate[keep]


def serrated_rod_mask(x_bfp, y_mechanical, rod_centres):
    """Six independently positioned 1+2x0.5 mm serrated rods."""
    phase = x_bfp / TOOTH_PITCH_MM
    triangle = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    half_width = CORE_MM / 2 + DEPTH_MM * 0.5 * (triangle + 1.0)
    mask = np.ones((y_mechanical.size, x_bfp.size), dtype=np.float32)
    for centre in rod_centres:
        blocked = np.abs(y_mechanical[:, None] - centre) <= half_width[None, :]
        mask[blocked] = 0.0
    return mask


def tdi_zero_order_rod_mask(x_bfp, y_mechanical, rod_centres):
    """X-averaged serration transmission when only its zero order reaches TDI.

    For a triangular binary tooth, the open fraction changes linearly through
    each 0.5 mm edge band.  This is the patent's effective soft edge.
    """
    transmission_y = np.ones_like(y_mechanical, dtype=np.float32)
    for centre in rod_centres:
        distance = np.abs(y_mechanical - centre)
        one_rod = np.ones_like(y_mechanical, dtype=np.float32)
        one_rod[distance <= CORE_MM / 2] = 0.0
        edge = (distance > CORE_MM / 2) & (distance < CORE_MM / 2 + DEPTH_MM)
        one_rod[edge] = (distance[edge] - CORE_MM / 2) / DEPTH_MM
        transmission_y *= one_rod
    return np.repeat(transmission_y[:, None], x_bfp.size, axis=1)


def particle_scatter_spectrum(fxx, fyy, pupil):
    """Broad, smooth Ch3 dark-field spectrum from three irregular particles.

    A local particle is not represented by a translated copy of its real-space
    shape in the Fourier plane.  Position appears only as a phase ramp; the
    spectral magnitude is a broad continuous envelope.
    """
    radial = np.sqrt(fxx**2 + fyy**2) / FCUT
    envelope = np.exp(-0.5 * (radial / 0.82) ** 6) * pupil
    specifications = [(-0.012, 0.0, 0.85, 0.10),
                      (0.000, 0.0, 1.00, -0.55),
                      (0.012, 0.0, 1.15, 0.75)]
    spectrum = np.zeros_like(fxx, dtype=np.complex64)
    individual_spectra = []
    # Each particle is a compact cluster of sub-scatterers.  The cluster makes
    # the image irregular while its Fourier magnitude remains continuous.
    offsets = [(0.0, 0.0, 1.0, 0.0),
               (0.00016, 0.00007, 0.55, 0.65),
               (-0.00011, 0.00013, 0.38, -0.8)]
    for cx, cy, strength, angle in specifications:
        ca, sa = np.cos(angle), np.sin(angle)
        one_particle = np.zeros_like(fxx, dtype=np.complex64)
        for ox, oy, weight, phase0 in offsets:
            rx, ry = ca * ox - sa * oy, sa * ox + ca * oy
            phase = np.exp(-2j * np.pi * (fxx * (cx + rx) + fyy * (cy + ry)))
            one_particle += strength * weight * np.exp(1j * phase0) * envelope * phase
        spectrum += one_particle
        individual_spectra.append(0.0032 * one_particle)
    return 0.0032 * spectrum, individual_spectra, specifications


def local_spot_widths(image, coordinate, cx=0.0, cy=0.0):
    """Second-moment widths in a 2 um radius, reported as Gaussian-equivalent FWHM."""
    xx, yy = np.meshgrid(coordinate, coordinate)
    roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.0020) ** 2
    weights = image * roi
    total = float(weights.sum())
    x0 = float((weights * xx).sum() / total)
    y0 = float((weights * yy).sum() / total)
    sigma_x = np.sqrt(float((weights * (xx - x0) ** 2).sum() / total))
    sigma_y = np.sqrt(float((weights * (yy - y0) ** 2).sum() / total))
    factor = 2 * np.sqrt(2 * np.log(2)) * 1000
    return factor * sigma_x, factor * sigma_y


def plot_images(coordinate, image_open, image_filtered, particle_image):
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    reference = max(float(image_open[np.ix_(roi, roi)].max()), 1e-30)
    panels = [image_open, image_filtered]
    titles = ["No rods: Ch3 optical image",
              "Six rods: complete pre-TDI image plane"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    for axis, raw, title in zip(axes, panels, titles):
        data = raw[np.ix_(roi, roi)] / reference
        db = 10 * np.log10(np.maximum(data, 10 ** (-4.0)))
        shown = axis.imshow(db, origin="lower", extent=extent, cmap="turbo",
                            vmin=-40, vmax=0, aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("X: TDI scan direction (um)")
        axis.set_ylabel("Y: camera long edge (um)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(shown, ax=axis, fraction=0.046,
                     label="Stage-normalised irradiance (dB a.u.; common reference)")
    fig.suptitle("Corrected 2 um case: Ch3 collection, 75 deg illumination along +Y\n"
                 "Ideal velocity-matched 25-stage TDI; no scan blur and no background subtraction")
    path = OUT / "ch3_75deg_2um_raw_before_after_same_scale.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)

    # A separate particle-only input is a diagnostic, not a reference-subtracted image.
    p_ref = max(float(particle_image.max()), 1e-30)
    fig, axis = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    data = particle_image[np.ix_(roi, roi)] / p_ref
    shown = axis.imshow(10 * np.log10(np.maximum(data, 1e-5)), origin="lower", extent=extent,
                        cmap="turbo", vmin=-50, vmax=0, aspect="equal")
    axis.set(title="Particle-scatter input alone: Ch3 image\n25 aligned TDI stages preserve this geometry",
             xlabel="X: TDI scan direction (um)", ylabel="Y: camera long edge (um)")
    axis.set_aspect("equal", adjustable="box")
    fig.colorbar(shown, ax=axis, label="Normalised irradiance (dB a.u.)")
    path = OUT / "particle_only_shape_and_tdi_check.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_tdi_y_acceptance(coordinate, image_open, image_filtered):
    """Apply the newly confirmed Y-narrow TDI acceptance to the full image plane."""
    half_width_mm = 0.5 * TDI_Y_ACCEPTANCE_OBJ_UM / 1000
    gate_1d = (np.abs(coordinate) <= half_width_mm).astype(np.float32)
    gate = gate_1d[:, None]
    detected_open = image_open * gate
    detected_filtered = image_filtered * gate
    roi = np.abs(coordinate) <= 0.012
    extent = [-12, 12, -12, 12]
    reference = max(float(image_open[np.ix_(roi, roi)].max()), 1e-30)
    panels = [(image_filtered, "Complete image plane before TDI acceptance"),
              (detected_open, "No rods: Y-narrow TDI accepted signal"),
              (detected_filtered, "Six rods: Y-narrow TDI accepted signal")]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), constrained_layout=True)
    for axis, (raw, title) in zip(axes, panels):
        db = 10 * np.log10(np.maximum(raw[np.ix_(roi, roi)] / reference, 1e-5))
        shown = axis.imshow(db, origin="lower", extent=extent, cmap="turbo",
                            vmin=-50, vmax=0, aspect="equal")
        axis.axhline(-TDI_Y_ACCEPTANCE_OBJ_UM / 2, color="#FF3B30", lw=1.2)
        axis.axhline(+TDI_Y_ACCEPTANCE_OBJ_UM / 2, color="#FF3B30", lw=1.2)
        axis.set(title=title, xlabel="X: detector long direction (um)",
                 ylabel="Y: effective narrow acceptance (um)")
        axis.set_aspect("equal", adjustable="box")
    fig.colorbar(shown, ax=axes, fraction=0.018,
                 label="Detected irradiance (dB a.u.; common no-rod reference)")
    fig.suptitle(
        f"Corrected detector orientation: TDI accepts only central Y order | "
        f"width={TDI_Y_ACCEPTANCE_OBJ_UM:.1f} um object / "
        f"{TDI_Y_ACCEPTANCE_OBJ_UM*MAGNIFICATION:.0f} um image"
    )
    path = OUT / "ch3_75deg_2um_tdi_y_zero_order_only.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_spectra(pattern_spectrum, single_particle_spectrum, combined_spectrum, filtered_spectrum,
                 mask, x_bfp, y_mechanical, orders, rod_centres):
    spectra = [pattern_spectrum, single_particle_spectrum, combined_spectrum, filtered_spectrum]
    reference = max(float(np.abs(combined_spectrum).max() ** 2), 1e-30)
    extent = [x_bfp[0], x_bfp[-1], y_mechanical[0], y_mechanical[-1]]
    fig, axes = plt.subplots(1, 5, figsize=(20.5, 5.1), constrained_layout=True)
    titles = ["Periodic pattern only\n(discrete diffraction orders)",
              "One particle only\n(broad continuous magnitude)",
              "Combined raw spectrum\nbefore rods", "Combined raw spectrum\nafter rods"]
    for axis, spectrum, title in zip(axes[:4], spectra, titles):
        intensity = np.abs(spectrum) ** 2 / reference
        shown = axis.imshow(10 * np.log10(np.maximum(intensity, 1e-9)), origin="lower",
                            extent=extent, cmap="magma", vmin=-90, vmax=0, aspect="auto")
        axis.set_title(title, fontsize=10)
        axis.set(xlim=(-16, 16), ylim=(-50.7, 50.7),
                 xlabel="Original BFP X (mm)", ylabel="Mechanical Fourier Y (mm)")
        for order, centre in zip(orders, rod_centres):
            axis.axhline(centre, color="#5DE2E7", lw=0.55, alpha=0.65)
            axis.text(-15.2, centre + 0.7, f"m={order}", color="#5DE2E7", fontsize=7)
    fig.colorbar(shown, ax=axes[:4], fraction=0.016,
                 label="Spectral intensity (dB; common combined-before reference)")
    mask_plot = axes[4].imshow(mask, origin="lower", extent=extent, cmap="gray",
                               vmin=0, vmax=1, aspect="auto")
    axes[4].set(title="TDI-accepted serration zero order\n(soft-edge amplitude mask)", xlim=(-16, 16), ylim=(-50.7, 50.7),
                xlabel="Original BFP X (mm)", ylabel="Mechanical Fourier Y (mm)")
    for order, centre in zip(orders, rod_centres):
        axes[4].axhline(centre, color="#E74C3C", lw=0.65)
        axes[4].text(-15.2, centre + 0.7, f"m={order}\n{centre:+.2f} mm",
                     color="#E74C3C", fontsize=7)
    fig.colorbar(mask_plot, ax=axes[4], fraction=0.046, label="Amplitude transmission")
    fig.suptitle("Ch3 Fourier plane under 75 deg illumination: pattern peaks translate; particle spectrum does not copy particle shape")
    path = OUT / "ch3_75deg_2um_fourier_components_and_rods.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_order_geometry(orders, fy, centres, sine_coordinates):
    fig, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    axis.axvspan(-MY * F_MM * NA, MY * F_MM * NA, color="#EAF2F8", label="Ch3 NA=0.6 pupil")
    axis.scatter(centres, np.ones_like(centres), s=68, color="#C0392B", zorder=4)
    for order, centre, sine in zip(orders, centres, sine_coordinates):
        axis.axvspan(centre - 1.0, centre + 1.0, color="#F5B041", alpha=0.24)
        axis.text(centre, 1.035, f"m={order}\n{centre:+.2f} mm\ns={sine:+.3f}",
                  ha="center", fontsize=8)
    axis.set(xlim=(-55, 55), ylim=(0.88, 1.13), yticks=[],
             xlabel="Mechanical Fourier-plane Y (mm)",
             title="Six independently positioned rods follow the actually collected oblique-illumination orders")
    axis.grid(axis="x", color="#D5D8DC", lw=0.7)
    axis.legend(frameon=False, loc="lower left")
    path = OUT / "ch3_75deg_2um_collected_orders_and_rod_positions.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_replica_cause(single_particle_spectrum, x_bfp, y_mechanical, rod_centres, coordinate):
    """Controlled proof that the Y point train is the multi-notch PSF."""
    no_rods = np.ones_like(single_particle_spectrum, dtype=np.float32)
    one_rod = tdi_zero_order_rod_mask(x_bfp, y_mechanical, [rod_centres[3]])
    six_rods = tdi_zero_order_rod_mask(x_bfp, y_mechanical, rod_centres)
    cases = [("No rods", no_rods),
             ("One averaged k=0 notch", one_rod),
             ("Six averaged k=0 notches", six_rods)]
    images = [np.abs(ift2(single_particle_spectrum * mask)) ** 2 for _, mask in cases]
    reference = max(float(images[0].max()), 1e-30)
    roi = np.abs(coordinate) <= 0.012
    extent = [-12, 12, -12, 12]
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.8), constrained_layout=True)
    for axis, (title, _), raw in zip(axes[:3], cases, images):
        db = 10 * np.log10(np.maximum(raw[np.ix_(roi, roi)] / reference, 1e-6))
        shown = axis.imshow(db, origin="lower", extent=extent, cmap="turbo",
                            vmin=-60, vmax=0, aspect="equal")
        axis.set(title=title, xlabel="X scan (um)", ylabel="Y camera edge (um)")
        axis.set_aspect("equal", adjustable="box")
    fig.colorbar(shown, ax=axes[:3], fraction=0.018,
                 label="Single-particle irradiance (dB; common no-rod reference)")
    centre_x = coordinate.size // 2
    for (title, _), raw in zip(cases, images):
        profile = raw[:, centre_x] / reference
        axes[3].plot(coordinate * 1000, 10 * np.log10(np.maximum(profile, 1e-8)), label=title)
    for multiple in range(-5, 6):
        axes[3].axvline(2 * multiple, color="#9E9E9E", lw=0.55, ls=":" if multiple else "--")
    axes[3].set(xlim=(-12, 12), ylim=(-80, 2), xlabel="Y from particle (um)",
                ylabel="Irradiance (dB; common reference)",
                title="Y-notch-comb impulse response\n(not serration k != 0 orders)")
    axes[3].grid(color="#D5D8DC", lw=0.6)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("Controlled cause test: one input particle; serration k != 0 orders already removed\n"
                 "The remaining Y point train comes from six averaged k=0 notches")
    path = OUT / "controlled_single_particle_replica_cause.png"
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

    pattern = periodic_amplitude(yy)
    illumination_carrier = np.exp(2j * np.pi * INCIDENCE_FY * yy)
    pattern_spectrum = ft2(pattern * illumination_carrier) * pupil
    particle_spectrum, individual_particle_spectra, particles = particle_scatter_spectrum(fxx, fyy, pupil)
    # Match useful particle and periodic-pattern levels without removing either term.
    # Keep the unfiltered periodic pattern visibly bright; the particle is not
    # artificially made orders of magnitude stronger than the background.
    particle_scale = 0.035 * np.sqrt(np.sum(np.abs(pattern_spectrum) ** 2) /
                                    max(np.sum(np.abs(particle_spectrum) ** 2), 1e-30))
    particle_spectrum *= particle_scale
    individual_particle_spectra = [item * particle_scale for item in individual_particle_spectra]
    combined_spectrum = pattern_spectrum + particle_spectrum

    orders, fy, rod_centres, sine_coordinates = collected_orders()
    x_bfp = F_MM * LAMBDA_MM * frequency
    y_mechanical = MY * F_MM * LAMBDA_MM * frequency
    physical_rods = serrated_rod_mask(x_bfp, y_mechanical, rod_centres)
    rods = tdi_zero_order_rod_mask(x_bfp, y_mechanical, rod_centres)
    filtered_spectrum = combined_spectrum * rods

    image_open = np.abs(ift2(combined_spectrum)) ** 2
    image_filtered = np.abs(ift2(filtered_spectrum)) ** 2
    particle_image = np.abs(ift2(particle_spectrum)) ** 2
    fwhm_x, fwhm_y = local_spot_widths(particle_image, coordinate)

    plot_images(coordinate, image_open, image_filtered, particle_image)
    plot_tdi_y_acceptance(coordinate, image_open, image_filtered)
    plot_spectra(pattern_spectrum, individual_particle_spectra[1], combined_spectrum, filtered_spectrum,
                 rods, x_bfp, y_mechanical, orders, rod_centres)
    plot_order_geometry(orders, fy, rod_centres, sine_coordinates)
    plot_replica_cause(individual_particle_spectra[1], x_bfp, y_mechanical,
                       rod_centres, coordinate)

    summary = {
        "model": "Ch3 centred collection with 75 deg illumination propagating +Y; scalar periodic diffraction plus broad particle dark-field scattering",
        "wavelength_nm": 355.0,
        "objective_f_mm": F_MM,
        "objective_NA": NA,
        "incidence_from_normal_deg": INCIDENCE_DEG,
        "illumination_tangential_frequency_cycles_per_mm": INCIDENCE_FY,
        "pattern_period_um": PERIOD_MM * 1000,
        "bright_width_um": PERIOD_MM * 1000 * 4 / 7,
        "dark_width_um": PERIOD_MM * 1000 * 3 / 7,
        "collected_orders": orders.tolist(),
        "collected_order_fy_cycles_per_mm": fy.tolist(),
        "rod_centres_mechanical_y_mm": rod_centres.tolist(),
        "rod_count": int(orders.size),
        "rod_core_mm": CORE_MM,
        "rod_tooth_depth_each_side_mm": DEPTH_MM,
        "rod_total_envelope_mm": CORE_MM + 2 * DEPTH_MM,
        "tdi_stages": TDI_STAGES,
        "tdi_rule": "velocity-matched charge packets align the same object point; stage-normalised image has no scan convolution; newly confirmed effective narrow acceptance is Y and only its central order is detected",
        "tdi_effective_narrow_direction": "Y",
        "tdi_long_direction": "X",
        "tdi_y_acceptance_object_um_provisional": TDI_Y_ACCEPTANCE_OBJ_UM,
        "tdi_y_acceptance_image_um_provisional": TDI_Y_ACCEPTANCE_OBJ_UM * MAGNIFICATION,
        "serration_acceptance_rule": "25-stage narrow TDI accepts serration zero order; final image uses X-averaged soft-edge amplitude transmission, while the full binary geometry remains available as physical_rods in the calculation",
        "particle_only_gaussian_equivalent_fwhm_x_um": fwhm_x,
        "particle_only_gaussian_equivalent_fwhm_y_um": fwhm_y,
        "fwhm_aspect_ratio_x_over_y": fwhm_x / fwhm_y,
        "evaluation": "raw combined intensity, common before/after reference, no periodic-background subtraction",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
