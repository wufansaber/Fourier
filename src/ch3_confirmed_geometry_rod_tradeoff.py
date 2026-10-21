"""Rod-count trade study using the user-confirmed camera/TDI coordinates.

X is the camera long edge and illumination-line long direction.
Y is the TDI scan/charge-transfer and narrow-acceptance direction.
The serrated rods are long along Fourier Y and independently positioned in X.

All displayed before/after panels use raw irradiance with one common reference.
No periodic-reference subtraction is used for image evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

import ch3_oblique75_confirmed_geometry as base


OUT = base.ROOT / "results" / "ch3_confirmed_geometry_rod_tradeoff"
ROD_SETS = {
    "3 rods": np.arange(-1, 2),
    "5 rods": np.arange(-2, 3),
    "7 rods": np.arange(-3, 4),
}


def hard_rod_mask(x_mechanical, y_bfp, centres, width_mm):
    transmission_x = np.ones_like(x_mechanical, dtype=np.float32)
    for centre in centres:
        transmission_x[np.abs(x_mechanical - centre) <= width_mm / 2] = 0.0
    return np.repeat(transmission_x[None, :], y_bfp.size, axis=0)


def energy_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.sum(numerator) / max(float(np.sum(denominator)), 1e-30))


def local_particle_contrast_db(image: np.ndarray, coordinate: np.ndarray,
                               particles: list[tuple[float, float, float, float]]) -> float:
    """Median raw-image peak/local-annulus contrast at known particle sites."""
    xx, yy = np.meshgrid(coordinate, coordinate)
    values = []
    for cx, cy, _, _ in particles:
        rr = np.hypot(xx - cx, yy - cy)
        core = image[rr <= 0.00055]
        annulus = image[(rr >= 0.0010) & (rr <= 0.0018)]
        peak = float(np.max(core))
        background = float(np.median(annulus))
        values.append(10 * np.log10(max(peak, 1e-30) / max(background, 1e-30)))
    return float(np.median(values))


def ringing_metrics(single_image: np.ndarray, coordinate: np.ndarray) -> tuple[float, float]:
    """X-replica peak and energy outside a 1 um central disk for one particle."""
    xx, yy = np.meshgrid(coordinate, coordinate)
    centre = np.hypot(xx, yy) <= 0.00065
    x_replica_band = ((np.abs(yy) <= 0.00065) &
                      (np.abs(xx) >= 0.00125) & (np.abs(xx) <= 0.0105))
    central_peak = max(float(np.max(single_image[centre])), 1e-30)
    replica_peak = max(float(np.max(single_image[x_replica_band])), 1e-30)
    replica_peak_db = 10 * np.log10(replica_peak / central_peak)
    roi = (np.abs(xx) <= 0.012) & (np.abs(yy) <= 0.006)
    outside = roi & ~centre
    outside_fraction = energy_ratio(single_image[outside], single_image[roi])
    return float(replica_peak_db), float(outside_fraction)


def build_model():
    count, field_width = 1024, 0.128
    dx = field_width / count
    coordinate = (np.arange(count) - count // 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    frequency = fft.fftshift(fft.fftfreq(count, d=dx))
    fxx, fyy = np.meshgrid(frequency, frequency)
    pupil = ((fxx**2 + fyy**2) <= base.FCUT**2).astype(np.float32)

    pattern = base.two_dimensional_pattern(xx, yy)
    illumination = np.exp(2j * np.pi * base.FY_INC * yy)
    pattern_spec = base.ft2(pattern * illumination) * pupil
    particles_spec, single_specs, particles = base.particle_spectrum(fxx, fyy, pupil)
    particle_scale = 0.040 * np.sqrt(np.sum(np.abs(pattern_spec) ** 2) /
                                      max(np.sum(np.abs(particles_spec) ** 2), 1e-30))
    particles_spec *= particle_scale
    single_specs = [item * particle_scale for item in single_specs]
    combined_spec = pattern_spec + particles_spec

    x_mechanical = base.FOURIER_POSITION_MAG * base.F_MM * base.LAMBDA_MM * frequency
    y_bfp = base.F_MM * base.LAMBDA_MM * frequency
    spacing = base.FOURIER_POSITION_MAG * base.F_MM * base.LAMBDA_MM / base.PERIOD_MM
    return {
        "coordinate": coordinate,
        "frequency": frequency,
        "pattern_spec": pattern_spec,
        "particles_spec": particles_spec,
        "single_specs": single_specs,
        "particles": particles,
        "combined_spec": combined_spec,
        "x_mechanical": x_mechanical,
        "y_bfp": y_bfp,
        "spacing": spacing,
    }


def evaluate(model):
    coordinate = model["coordinate"]
    open_pattern = np.abs(base.ift2(model["pattern_spec"])) ** 2
    open_particle = np.abs(base.ift2(model["particles_spec"])) ** 2
    open_combined = np.abs(base.ift2(model["combined_spec"])) ** 2
    open_single = np.abs(base.ift2(model["single_specs"][1])) ** 2
    results = {
        "No rods": {
            "image": open_combined,
            "single": open_single,
            "mask": np.ones_like(open_combined, dtype=np.float32),
            "background_suppression_db": 0.0,
            "particle_energy_retention": 1.0,
            "local_particle_contrast_db": local_particle_contrast_db(
                open_combined, coordinate, model["particles"]),
            "x_replica_peak_db": ringing_metrics(open_single, coordinate)[0],
            "ringing_outside_core_fraction": ringing_metrics(open_single, coordinate)[1],
        }
    }
    for label, orders in ROD_SETS.items():
        centres = orders.astype(float) * model["spacing"]
        mask = base.averaged_serration_k0_mask(
            model["x_mechanical"], model["y_bfp"], centres)
        pattern_image = np.abs(base.ift2(model["pattern_spec"] * mask)) ** 2
        particle_image = np.abs(base.ift2(model["particles_spec"] * mask)) ** 2
        combined_image = np.abs(base.ift2(model["combined_spec"] * mask)) ** 2
        single_image = np.abs(base.ift2(model["single_specs"][1] * mask)) ** 2
        background_ratio = energy_ratio(pattern_image, open_pattern)
        replica_db, outside_fraction = ringing_metrics(single_image, coordinate)
        results[label] = {
            "image": combined_image,
            "single": single_image,
            "mask": mask,
            "background_suppression_db": -10 * np.log10(max(background_ratio, 1e-30)),
            "particle_energy_retention": energy_ratio(particle_image, open_particle),
            "local_particle_contrast_db": local_particle_contrast_db(
                combined_image, coordinate, model["particles"]),
            "x_replica_peak_db": replica_db,
            "ringing_outside_core_fraction": outside_fraction,
        }
    return results


def plot_raw_comparison(model, results):
    coordinate = model["coordinate"]
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    reference = max(float(results["No rods"]["image"][np.ix_(roi, roi)].max()), 1e-30)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.4), constrained_layout=True)
    for axis, label in zip(axes.flat, results):
        raw = results[label]["image"][np.ix_(roi, roi)] / reference
        shown = axis.imshow(10 * np.log10(np.maximum(raw, 1e-4)), origin="lower",
                            extent=extent, cmap="turbo", vmin=-40, vmax=0,
                            interpolation="nearest")
        axis.set_title(label if label == "No rods" else f"{label}, k=0 accepted")
        axis.set_xlabel("X: camera / illumination-line long direction (um)")
        axis.set_ylabel("Y: TDI scan direction (um)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(shown, ax=axis, fraction=0.046,
                     label="Raw irradiance (dB a.u.; common reference)")
    fig.suptitle("Confirmed coordinates | Ch3 with 75 deg +Y illumination | 2 um pattern\n"
                 "Pixel-matched TDI: no scan blur; same raw scale; no background subtraction")
    path = OUT / "confirmed_coordinates_raw_3_5_7_rods.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_profiles(model, results):
    coordinate_um = model["coordinate"] * 1000
    centre = len(coordinate_um) // 2
    take = np.abs(coordinate_um) <= 12
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    for label, result in results.items():
        image = result["single"]
        xline = image[centre, :]
        yline = image[:, centre]
        xline = xline / max(float(xline[centre]), 1e-30)
        yline = yline / max(float(yline[centre]), 1e-30)
        axes[0].plot(coordinate_um[take], 10 * np.log10(np.maximum(xline[take], 1e-6)),
                     lw=1.5, label=label)
        axes[1].plot(coordinate_um[take], 10 * np.log10(np.maximum(yline[take], 1e-6)),
                     lw=1.5, label=label)
    axes[0].set(title="Single-particle profile along X\nrod-comb ringing direction",
                xlabel="X (um)", ylabel="Relative irradiance (dB)", ylim=(-60, 2))
    axes[1].set(title="Single-particle profile along Y\nTDI narrow-acceptance direction",
                xlabel="Y (um)", ylabel="Relative irradiance (dB)", ylim=(-60, 2))
    for axis in axes:
        axis.grid(color="#D5D8DC", lw=0.6)
        axis.legend(frameon=False)
    for x in (-2, 2):
        axes[0].axvline(x, color="#E74C3C", ls="--", lw=0.9, alpha=0.8)
    axes[0].text(2.15, -8, "2 um comb period", color="#C0392B", fontsize=9)
    fig.suptitle("Residual replicas rotate with the corrected rod orientation: X only, not TDI-Y")
    path = OUT / "confirmed_coordinates_single_particle_profiles.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_controlled_origin_test(model, results):
    """Prove that the X point train comes from the multi-rod k=0 notch comb."""
    coordinate = model["coordinate"]
    one_mask = base.averaged_serration_k0_mask(
        model["x_mechanical"], model["y_bfp"], np.asarray([0.0]))
    no_rods = np.abs(base.ift2(model["single_specs"][1])) ** 2
    one_rod = np.abs(base.ift2(model["single_specs"][1] * one_mask)) ** 2
    seven_rods = results["7 rods"]["single"]
    images = [no_rods, one_rod, seven_rods]
    labels = ["No rods: no periodic replicas",
              "One central rod: continuous sinc-like X tail",
              "Seven equally spaced rods: 2 um X replica train"]
    roi = np.abs(coordinate) <= 0.012
    extent = [-12, 12, -6, 6]
    yroi = np.abs(coordinate) <= 0.006
    reference = max(float(no_rods.max()), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), constrained_layout=True)
    for axis, image, label in zip(axes, images, labels):
        crop = image[np.ix_(yroi, roi)] / reference
        shown = axis.imshow(10 * np.log10(np.maximum(crop, 1e-5)), origin="lower",
                            extent=extent, cmap="turbo", vmin=-50, vmax=0,
                            interpolation="nearest", aspect="equal")
        axis.set(title=label, xlabel="X: camera long direction (um)",
                 ylabel="Y: TDI scan direction (um)")
        fig.colorbar(shown, ax=axis, fraction=0.046,
                     label="Raw single-particle irradiance (dB; common reference)")
    fig.suptitle("Controlled source test: the residual X train is created by the k=0 rod comb\n"
                 "No periodic sample background and no serration k!=0 orders are present")
    path = OUT / "controlled_no_rod_one_rod_seven_rod_origin.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_inserted_rod_spectrum(model, results):
    """Dedicated common-scale spectrum before/after inserting seven rods."""
    before = model["combined_spec"]
    mask = results["7 rods"]["mask"]
    after = before * mask
    reference = max(float(np.abs(before).max() ** 2), 1e-30)
    before_db = 10 * np.log10(np.maximum(np.abs(before) ** 2 / reference, 1e-9))
    after_db = 10 * np.log10(np.maximum(np.abs(after) ** 2 / reference, 1e-9))
    extent = [model["x_mechanical"][0], model["x_mechanical"][-1],
              model["y_bfp"][0], model["y_bfp"][-1]]
    centres = ROD_SETS["7 rods"].astype(float) * model["spacing"]
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0), constrained_layout=True)
    shown = axes[0].imshow(before_db, origin="lower", extent=extent, cmap="magma",
                           vmin=-90, vmax=0, aspect="auto")
    axes[0].contour(model["x_mechanical"], model["y_bfp"], mask,
                    levels=[0.02, 0.98], colors=["#38BDF8", "#E0F2FE"], linewidths=[1.1, 0.7])
    axes[0].set_title("Before rods + inserted-rod outlines")
    mask_shown = axes[1].imshow(mask, origin="lower", extent=extent, cmap="gray",
                                vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Seven inserted rods\nk=0 equivalent amplitude transmission")
    axes[2].imshow(after_db, origin="lower", extent=extent, cmap="magma",
                   vmin=-90, vmax=0, aspect="auto")
    axes[2].contour(model["x_mechanical"], model["y_bfp"], mask,
                    levels=[0.02, 0.98], colors=["#38BDF8", "#E0F2FE"], linewidths=[1.1, 0.7])
    axes[2].set_title("After inserting rods\nactual transmitted spectrum")
    for axis in axes:
        axis.set(xlim=(-51, 51), ylim=(-18, 18), xlabel="Mechanical Fourier X (mm)",
                 ylabel="Original BFP Y (mm)")
        for m, centre in zip(ROD_SETS["7 rods"], centres):
            axis.axvline(centre, color="#22D3EE", lw=0.45, alpha=0.65)
            axis.text(centre + 0.7, -17, f"m={m}", color="#67E8F9", fontsize=7,
                      rotation=90, va="bottom")
    fig.colorbar(shown, ax=[axes[0], axes[2]], fraction=0.025,
                 label="Spectral irradiance (dB; common before-rods reference)")
    fig.colorbar(mask_shown, ax=axes[1], fraction=0.046,
                 label="Amplitude transmission")
    fig.suptitle("Fourier plane after rod insertion | Ch3, 75 deg +Y illumination, 2 um pattern\n"
                 "Rods are long along Y_F and independently positioned along X_F")
    path = OUT / "seven_rods_fourier_spectrum_before_mask_after.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_pattern_and_combined_spectrum(model, results):
    """Separate periodic peaks from continuous particle spectrum for clarity."""
    mask = results["7 rods"]["mask"]
    pattern_before = model["pattern_spec"]
    pattern_after = pattern_before * mask
    combined_before = model["combined_spec"]
    combined_after = combined_before * mask
    rows = [(pattern_before, pattern_after, "Periodic pattern only"),
            (combined_before, combined_after, "Pattern + three particles")]
    extent = [model["x_mechanical"][0], model["x_mechanical"][-1],
              model["y_bfp"][0], model["y_bfp"][-1]]
    centres = ROD_SETS["7 rods"].astype(float) * model["spacing"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), constrained_layout=True)
    for row_index, (before, after, row_label) in enumerate(rows):
        reference = max(float(np.abs(before).max() ** 2), 1e-30)
        for column, (spectrum, state) in enumerate(((before, "Before rods"),
                                                     (after, "After seven rods"))):
            db = 10 * np.log10(np.maximum(np.abs(spectrum) ** 2 / reference, 1e-9))
            shown = axes[row_index, column].imshow(
                db, origin="lower", extent=extent, cmap="magma", vmin=-90, vmax=0,
                aspect="auto")
            axes[row_index, column].contour(
                model["x_mechanical"], model["y_bfp"], mask,
                levels=[0.02, 0.98], colors=["#22D3EE", "#CFFAFE"],
                linewidths=[1.0, 0.6])
            axes[row_index, column].set(
                title=f"{row_label} | {state}", xlim=(-51, 51), ylim=(-18, 18),
                xlabel="Mechanical Fourier X (mm)", ylabel="Original BFP Y (mm)")
            for m, centre in zip(ROD_SETS["7 rods"], centres):
                axes[row_index, column].axvline(centre, color="#22D3EE", lw=0.4,
                                                alpha=0.55)
                axes[row_index, column].text(centre + 0.7, -17, f"m={m}",
                                             color="#67E8F9", fontsize=7,
                                             rotation=90, va="bottom")
        fig.colorbar(shown, ax=axes[row_index, :], fraction=0.025,
                     label=f"{row_label} spectral irradiance (dB; row-common reference)")
    fig.suptitle("Inserted-rod Fourier-plane check | discrete pattern peaks versus continuous defect spectrum\n"
                 "Cyan outlines are the seven Y_F-long rods; after panels are not renormalised")
    path = OUT / "seven_rods_pattern_and_combined_spectrum_before_after.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_serrated_edge_benefit(model):
    """Compare hard rod edges with the serrated k=0 equivalent soft edge."""
    centres = ROD_SETS["7 rods"].astype(float) * model["spacing"]
    masks = {
        "1.0 mm hard-edge rods": hard_rod_mask(
            model["x_mechanical"], model["y_bfp"], centres, 1.0),
        "2.0 mm hard-envelope rods": hard_rod_mask(
            model["x_mechanical"], model["y_bfp"], centres, 2.0),
        "1.0 + 2x0.5 mm serrated k=0": base.averaged_serration_k0_mask(
            model["x_mechanical"], model["y_bfp"], centres),
    }
    coordinate_um = model["coordinate"] * 1000
    centre_index = len(coordinate_um) // 2
    take = np.abs(coordinate_um) <= 30
    open_single = np.abs(base.ift2(model["single_specs"][1])) ** 2
    open_particles = np.abs(base.ift2(model["particles_spec"])) ** 2
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    for label, mask in masks.items():
        single_image = np.abs(base.ift2(model["single_specs"][1] * mask)) ** 2
        all_particle_image = np.abs(base.ift2(model["particles_spec"] * mask)) ** 2
        replica_db, outside_fraction = ringing_metrics(single_image, model["coordinate"])
        retention = 100 * energy_ratio(all_particle_image, open_particles)
        rows.append({"edge": label, "particle_energy_retention_percent": retention,
                     "worst_x_replica_db": replica_db,
                     "outside_core_energy_percent": 100 * outside_fraction})
        xline = single_image[centre_index, :]
        xline /= max(float(xline[centre_index]), 1e-30)
        axes[0].plot(coordinate_um[take],
                     10 * np.log10(np.maximum(xline[take], 1e-7)), lw=1.7, label=label)
    axes[0].set(title="Single-particle X profile", xlabel="X (um)",
                ylabel="Relative irradiance (dB)", ylim=(-70, 2))
    axes[0].grid(color="#D5D8DC", lw=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(len(rows))
    replica_values = [row["worst_x_replica_db"] for row in rows]
    bars = axes[1].bar(x, replica_values, color=["#64748B", "#334155", "#0EA5E9"])
    axes[1].set(title="Worst 2-10 um X replica", ylabel="Replica / main peak (dB)",
                xticks=x, xticklabels=["hard 1.0", "hard 2.0", "serrated k=0"], ylim=(-30, 0))
    axes[1].grid(axis="y", color="#D5D8DC", lw=0.6)
    for bar, row in zip(bars, rows):
        axes[1].text(bar.get_x() + bar.get_width() / 2, row["worst_x_replica_db"] - 1.2,
                     f"{row['worst_x_replica_db']:.1f} dB\nE={row['particle_energy_retention_percent']:.1f}%",
                     ha="center", va="top", color="white", fontsize=8)
    fig.suptitle("Serrated edges suppress the single-rod envelope, but not the equal-spacing comb condition")
    path = OUT / "hard_edge_vs_serrated_k0_ringing.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    (OUT / "hard_edge_vs_serrated_k0_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def plot_complete_filter_same_scale(model, results):
    """Linear and logarithmic before/after views with one absolute reference."""
    coordinate = model["coordinate"]
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    open_image = results["No rods"]["image"][np.ix_(roi, roi)]
    filtered_image = results["7 rods"]["image"][np.ix_(roi, roi)]
    reference = max(float(open_image.max()), 1e-30)
    linear = [open_image / reference, filtered_image / reference]
    logarithmic = [10 * np.log10(np.maximum(item, 1e-5)) for item in linear]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 10.2), constrained_layout=True)
    linear_shown = None
    db_shown = None
    for column, title in enumerate(("No rods", "Seven serrated rods, k=0 accepted")):
        linear_shown = axes[0, column].imshow(
            linear[column], origin="lower", extent=extent, cmap="turbo",
            vmin=0, vmax=1, interpolation="nearest")
        db_shown = axes[1, column].imshow(
            logarithmic[column], origin="lower", extent=extent, cmap="turbo",
            vmin=-50, vmax=0, interpolation="nearest")
        axes[0, column].set_title(f"{title} | linear")
        axes[1, column].set_title(f"{title} | logarithmic")
        for row in range(2):
            axes[row, column].set(
                xlabel="X: camera long direction (um)",
                ylabel="Y: TDI scan direction (um)")
            axes[row, column].set_aspect("equal", adjustable="box")
    fig.colorbar(linear_shown, ax=axes[0, :], fraction=0.025,
                 label="Raw irradiance / unfiltered maximum (linear, common 0-1 scale)")
    fig.colorbar(db_shown, ax=axes[1, :], fraction=0.025,
                 label="Raw irradiance (dB re unfiltered maximum, common -50 to 0 dB scale)")
    fig.suptitle("Complete filtering result on one brightness reference | no subtraction, no renormalisation\n"
                 "2 um matched-frequency demonstration: seven rod centres are 15 mm apart")
    path = OUT / "complete_filter_same_brightness_linear_and_db.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_spacing_consistency_same_scale(model, results):
    """Show that 1-1.5 mm rod pitches do not match a 2 um pattern in this relay."""
    coordinate = model["coordinate"]
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    configurations = [("No rods", None),
                      ("1.0 mm projected pitch", 1.0),
                      ("1.5 mm projected pitch", 1.5),
                      ("15 mm pitch: matched to 2 um", 15.0)]
    images = []
    rows = []
    pattern_open = np.abs(base.ift2(model["pattern_spec"])) ** 2
    particle_open = np.abs(base.ift2(model["particles_spec"])) ** 2
    for label, spacing in configurations:
        if spacing is None:
            mask = np.ones_like(model["pattern_spec"], dtype=np.float32)
        else:
            centres = ROD_SETS["7 rods"].astype(float) * spacing
            mask = base.averaged_serration_k0_mask(
                model["x_mechanical"], model["y_bfp"], centres)
        combined = np.abs(base.ift2(model["combined_spec"] * mask)) ** 2
        pattern = np.abs(base.ift2(model["pattern_spec"] * mask)) ** 2
        particle = np.abs(base.ift2(model["particles_spec"] * mask)) ** 2
        images.append(combined)
        rows.append({
            "configuration": label,
            "rod_pitch_mm": spacing,
            "background_suppression_db": -10 * np.log10(max(
                energy_ratio(pattern, pattern_open), 1e-30)),
            "particle_energy_retention_percent": 100 * energy_ratio(particle, particle_open),
        })
    reference = max(float(images[0][np.ix_(roi, roi)].max()), 1e-30)
    for display, suffix, vmin, vmax, colorbar_label in (
            ("linear", "linear", 0, 1,
             "Raw irradiance / unfiltered maximum (common linear scale)"),
            ("db", "db", -50, 0,
             "Raw irradiance (dB re unfiltered maximum; common scale)")):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.2), constrained_layout=True)
        shown = None
        for axis, (label, spacing), image, row in zip(axes.flat, configurations, images, rows):
            raw = image[np.ix_(roi, roi)] / reference
            data = raw if display == "linear" else 10 * np.log10(np.maximum(raw, 1e-5))
            shown = axis.imshow(data, origin="lower", extent=extent, cmap="turbo",
                                vmin=vmin, vmax=vmax, interpolation="nearest")
            axis.set_title(f"{label}\nS={min(row['background_suppression_db'], 60):.1f} dB, "
                           f"particle E={row['particle_energy_retention_percent']:.1f}%")
            axis.set(xlabel="X: camera long direction (um)",
                     ylabel="Y: TDI scan direction (um)")
            axis.set_aspect("equal", adjustable="box")
        fig.colorbar(shown, ax=axes.ravel().tolist(), fraction=0.025, label=colorbar_label)
        fig.suptitle("Rod-pitch consistency check on the same 2 um object | no subtraction, no renormalisation\n"
                     "1-1.5 mm projected pitches are not aligned to the 2 um diffraction-order columns")
        path = OUT / f"rod_pitch_1_1p5_15mm_same_scale_{suffix}.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)
    (OUT / "rod_pitch_consistency_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def plot_metrics(results):
    labels = list(ROD_SETS)
    suppression_exact = [results[k]["background_suppression_db"] for k in labels]
    suppression = [min(v, 60.0) for v in suppression_exact]
    retention = [100 * results[k]["particle_energy_retention"] for k in labels]
    replica = [results[k]["x_replica_peak_db"] for k in labels]
    contrast = [results[k]["local_particle_contrast_db"] for k in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    bars0 = axes[0].bar(x, suppression, color="#2563EB")
    axes[0].set(title="Periodic-background suppression", ylabel="Suppression (dB)", xticks=x,
                xticklabels=labels, ylim=(0, 66))
    for bar, exact in zip(bars0, suppression_exact):
        label = ">60" if exact > 60 else f"{exact:.1f}"
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                     label, ha="center", fontsize=9)
    bars1 = axes[1].bar(x, retention, width=0.55, color="#16A085")
    axes[1].set(title="Particle energy retention", ylabel="Retention (%)", xticks=x,
                xticklabels=labels, ylim=(0, 100))
    for bar, value, contrast_value in zip(bars1, retention, contrast):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                     f"{value:.1f}%\nraw C={contrast_value:.1f} dB", ha="center", fontsize=8)
    axes[2].bar(x, replica, color="#C0392B")
    axes[2].set(title="Worst X replica relative to centre", ylabel="Replica peak (dB)", xticks=x,
                xticklabels=labels)
    for axis in axes:
        axis.grid(axis="y", color="#D5D8DC", lw=0.6)
    fig.suptitle("Trade-off metrics use physical components; displayed images remain raw and unsubtracted")
    path = OUT / "confirmed_coordinates_rod_count_metrics.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def save_metrics(results, spacing):
    rows = []
    for label, result in results.items():
        count = 0 if label == "No rods" else int(label.split()[0])
        row = {
            "configuration": label,
            "rod_count": count,
            "rod_spacing_mm": spacing if count else "",
            "background_suppression_db": result["background_suppression_db"],
            "particle_energy_retention_percent": 100 * result["particle_energy_retention"],
            "raw_local_particle_contrast_db": result["local_particle_contrast_db"],
            "worst_x_replica_peak_db_relative_centre": result["x_replica_peak_db"],
            "ringing_energy_outside_0p65um_core_percent":
                100 * result["ringing_outside_core_fraction"],
        }
        rows.append(row)
    fieldnames = list(rows[0])
    with (OUT / "rod_count_tradeoff_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "confirmed_coordinates": {
            "X": "camera long edge and illumination-line long axis",
            "Y": "TDI scan/charge transfer/narrow acceptance",
            "rod_long_axis": "Fourier Y",
            "rod_positioning": "Fourier X",
        },
        "simulation": {
            "illumination": "75 deg from normal, projected +Y, Ch3 collection",
            "scan_model": "pixel-matched one-to-one TDI sampling; no overlap and no convolution",
            "image_evaluation": "raw common-scale irradiance; no periodic-background subtraction",
            "accepted_serration_orders": [0],
            "rod_core_mm": base.CORE_MM,
            "tooth_depth_each_side_mm": base.DEPTH_MM,
            "tooth_pitch_mm": base.TOOTH_PITCH_MM,
        },
        "rows": rows,
        "interpretation": (
            "TDI rejects serration k!=0 along Y. Residual replicas are produced along X by the "
            "equally spaced k=0 rod-notch comb; they are not TDI smear or leaked serration orders."
        ),
    }
    (OUT / "rod_count_tradeoff_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    results = evaluate(model)
    plot_raw_comparison(model, results)
    plot_profiles(model, results)
    plot_controlled_origin_test(model, results)
    plot_inserted_rod_spectrum(model, results)
    plot_pattern_and_combined_spectrum(model, results)
    serration_rows = plot_serrated_edge_benefit(model)
    plot_complete_filter_same_scale(model, results)
    spacing_rows = plot_spacing_consistency_same_scale(model, results)
    plot_metrics(results)
    rows = save_metrics(results, model["spacing"])
    print(json.dumps(serration_rows, ensure_ascii=False, indent=2))
    print(json.dumps(spacing_rows, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
