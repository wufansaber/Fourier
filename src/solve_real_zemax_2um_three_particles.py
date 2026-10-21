"""Resolve the 2 um three-particle real-Zemax comparison.

The exact, untransposed Python object field is propagated on a sufficiently
sampled FFT grid.  The coherent pupil phase and the nonlinear physical BFP
coordinate mapping come from the saved real OpticStudio objective.  Two masks
are evaluated separately:

1. ``matched``: the orientation used by the unchanged Python example, with
   rods long in BFP X and positioned in BFP Y;
2. ``engineering``: the final documented orientation, with rods long in BFP Y
   and positioned in BFP X.

This is deliberately not native surface-to-surface POP.  The latter expands a
0.12 mm object grid to 8.61 mm at the first refracting surface, so a 1024 grid
cannot sample the 2 um carrier.  Here Zemax supplies the real lens transfer,
while the Fourier propagation retains the object sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from small_period_raw_intensity_simulation import periodic_pattern
from validate_uv_real_objective_coherent_filter import zemax_transfer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results" / "uv_real_objective_python_case_2um_exact_image"
    / "python_reference_2um.npz"
)
DEFAULT_OUT = (
    ROOT / "results" / "uv_real_objective_python_case_2um_exact_image"
    / "resolved_real_zemax"
)

ORDERS = np.arange(-3, 4)
# Hardware mask dimensions divided by the 2.8169x pupil relay.  The transfer
# coordinate arrays below are at the objective's physical BFP.
PUPIL_RELAY_MAG = 2.816901408450704
CORE_BFP_MM = 1.0 / PUPIL_RELAY_MAG
DEPTH_BFP_MM = 0.5 / PUPIL_RELAY_MAG
PITCH_BFP_MM = 0.75 / PUPIL_RELAY_MAG


def serrated_mask(bfp_x, bfp_y, centres, orientation):
    if orientation == "matched":
        # Old/existing Python example: long X, positioned in Y.
        along, across = bfp_x, bfp_y
    elif orientation == "engineering":
        # Final engineering document: long Y, positioned in X.
        along, across = bfp_y, bfp_x
    else:
        raise ValueError(orientation)
    phase = along / PITCH_BFP_MM
    triangle = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    half_width = CORE_BFP_MM / 2 + DEPTH_BFP_MM * 0.5 * (triangle + 1.0)
    mask = np.ones(bfp_x.shape, dtype=bool)
    for centre in centres:
        mask &= np.abs(across - centre) > half_width
    return mask


def propagate(field, transfer, mask):
    spectrum = fft.fft2(fft.ifftshift(field), norm="ortho", workers=4)
    image_field = fft.fftshift(
        fft.ifft2(spectrum * transfer * mask, norm="ortho", workers=4)
    )
    return spectrum, np.abs(image_field) ** 2


def measure(coordinate, background_open, background_filtered,
            actual_open, actual_filtered, particle_centres, particle_sizes):
    xx, yy = np.meshgrid(coordinate, coordinate)
    analysis = (np.abs(xx) <= 0.045) & (np.abs(yy) <= 0.045)
    particle_region = np.zeros_like(analysis)
    per_particle = []
    for (cx, cy), size_um in zip(particle_centres, particle_sizes):
        radius = max(0.0015, 1.5 * float(size_um) / 1000)
        roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        particle_region |= roi
        per_particle.append({
            "centre_um": [float(cx * 1000), float(cy * 1000)],
            "size_um": float(size_um),
            "open_peak": float(actual_open[roi].max()),
            "filtered_peak": float(actual_filtered[roi].max()),
            "peak_retention_percent": float(
                100 * actual_filtered[roi].max() / max(actual_open[roi].max(), 1e-30)
            ),
        })
    pattern_region = analysis & ~particle_region
    open_p95 = float(np.percentile(actual_open[pattern_region], 95))
    filtered_p95 = float(np.percentile(actual_filtered[pattern_region], 95))
    peak_open = float(actual_open[particle_region].max())
    peak_filtered = float(actual_filtered[particle_region].max())
    return {
        "background_suppression_db": float(10 * np.log10(
            background_open[analysis].sum()
            / max(background_filtered[analysis].sum(), 1e-30)
        )),
        "particle_peak_retention_percent": float(
            100 * peak_filtered / max(peak_open, 1e-30)
        ),
        "particle_to_pattern_open_db": float(
            10 * np.log10(peak_open / max(open_p95, 1e-30))
        ),
        "particle_to_pattern_filtered_db": float(
            10 * np.log10(peak_filtered / max(filtered_p95, 1e-30))
        ),
        "per_particle": per_particle,
    }


def save_images(out, coordinate, arrays, particle_centres):
    labels = [
        "Real Zemax pupil | open",
        "Matched to unchanged Python\nrods long X, positioned Y",
        "Final engineering orientation\nrods long Y, positioned X",
    ]
    keys = ["open", "matched", "engineering"]
    reference = float(arrays["open"].max())
    selection_x = np.abs(coordinate) <= 0.018
    selection_y = np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), constrained_layout=True)
    for column, (key, label) in enumerate(zip(keys, labels)):
        linear = arrays[key][np.ix_(selection_y, selection_x)] / reference
        im0 = axes[0, column].imshow(
            linear, origin="lower", extent=extent, cmap="turbo",
            vmin=0, vmax=1, aspect="auto", interpolation="nearest"
        )
        logarithmic = 10 * np.log10(np.maximum(linear, 1e-5))
        im1 = axes[1, column].imshow(
            logarithmic, origin="lower", extent=extent, cmap="turbo",
            vmin=-50, vmax=0, aspect="auto", interpolation="nearest"
        )
        axes[0, column].set_title(label)
        axes[1, column].set_title(label + " | common-reference dB")
        for row in range(2):
            axes[row, column].set_xlabel("object X (um)")
            axes[row, column].set_ylabel("object Y (um)")
            for cx, cy in particle_centres:
                axes[row, column].plot(cx * 1000, cy * 1000, "o", ms=8,
                                       mfc="none", mec="white", mew=0.8)
    fig.colorbar(im0, ax=axes[0], shrink=0.88,
                 label="raw intensity / same open-image peak")
    fig.colorbar(im1, ax=axes[1], shrink=0.88,
                 label="dB / same open-image peak")
    fig.suptitle(
        "Exact untransposed 2 um horizontal stripes + three particles\n"
        "Real OpticStudio objective OPD and BFP mapping; no background subtraction",
        fontsize=13,
    )
    fig.savefig(out / "real_zemax_resolved_image_comparison.png", dpi=170)
    plt.close(fig)


def save_python_zemax_comparison(out, coordinate, python_open, python_filtered,
                                 zemax_open, zemax_filtered, particle_centres):
    """Direct same-input comparison on one absolute intensity reference."""
    reference = max(float(python_open.max()), float(zemax_open.max()))
    selection_x = np.abs(coordinate) <= 0.018
    selection_y = np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]
    panels = [
        (python_open, "Python ideal pupil | unfiltered"),
        (python_filtered, "Python ideal pupil | filtered"),
        (zemax_open, "Real Zemax pupil/OPD | unfiltered"),
        (zemax_filtered, "Real Zemax pupil/OPD | filtered"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    for ax, (values, title) in zip(axes.flat, panels):
        image = values[np.ix_(selection_y, selection_x)] / reference
        im = ax.imshow(image, origin="lower", extent=extent, cmap="turbo",
                       vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set(title=title, xlabel="object X (um)", ylabel="object Y (um)")
        for cx, cy in particle_centres:
            ax.plot(cx * 1000, cy * 1000, "o", ms=8,
                    mfc="none", mec="white", mew=0.8)
    fig.colorbar(im, ax=axes, shrink=0.88,
                 label="raw intensity / common Python+Zemax open-image peak")
    fig.suptitle(
        "Same exact 2 um three-particle field, no transpose, no background subtraction\n"
        "Filtered panels use the rod orientation of the unchanged Python case",
        fontsize=12.5,
    )
    fig.savefig(out / "python_vs_real_zemax_same_input.png", dpi=180)
    plt.close(fig)


def save_spectra(out, spectrum, transfer, masks, bfp_x, bfp_y, centres):
    before = np.abs(spectrum * transfer) ** 2
    shifted_x = fft.fftshift(bfp_x[0])
    shifted_y = fft.fftshift(bfp_y[:, 0])
    # Outside the NA support the radial mapping is clamped and repeats its end
    # value.  Keep only the central strictly monotonic pupil segment, otherwise
    # imshow would compress the physical coordinate scale and draw the 5.166 mm
    # orders at the wrong apparent locations.
    ix = np.where(np.diff(shifted_x) > 1e-10)[0]
    iy = np.where(np.diff(shifted_y) > 1e-10)[0]
    mono_x = np.arange(ix[0], ix[-1] + 2)
    mono_y = np.arange(iy[0], iy[-1] + 2)
    sx = mono_x[np.abs(shifted_x[mono_x]) <= 18.2]
    sy = mono_y[np.abs(shifted_y[mono_y]) <= 18.2]
    data = {
        "open": fft.fftshift(before)[np.ix_(sy, sx)],
        "matched": fft.fftshift(before * masks["matched"])[np.ix_(sy, sx)],
        "engineering": fft.fftshift(before * masks["engineering"])[np.ix_(sy, sx)],
    }
    reference = float(data["open"].max())
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3), constrained_layout=True)
    titles = ["Before rods", "Matched orientation", "Final engineering orientation"]
    for ax, key, title in zip(axes, ("open", "matched", "engineering"), titles):
        values = 10 * np.log10(np.maximum(data[key] / reference, 1e-7))
        im = ax.imshow(values, origin="lower", cmap="turbo", vmin=-70, vmax=0,
                       extent=[shifted_x[sx[0]], shifted_x[sx[-1]],
                               shifted_y[sy[0]], shifted_y[sy[-1]]],
                       aspect="equal", interpolation="nearest")
        ax.set(title=title, xlabel="real BFP X (mm)", ylabel="real BFP Y (mm)")
        if key == "matched":
            for centre in centres:
                ax.axhline(centre, color="white", lw=0.45, alpha=0.75)
        elif key == "engineering":
            for centre in centres:
                ax.axvline(centre, color="white", lw=0.45, alpha=0.75)
    fig.colorbar(im, ax=axes, shrink=0.86, label="spectral intensity (dB, common ref.)")
    fig.suptitle("Same object spectrum mapped by the real Zemax objective")
    fig.savefig(out / "real_zemax_resolved_spectrum_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = np.load(args.input.resolve())
    coordinate = source["coordinate_mm"]
    actual = source["object_field"]
    centres_particles = source["particle_centres_mm"]
    sizes_particles = source["particle_sizes_um"]
    n = coordinate.size
    window = float((coordinate[1] - coordinate[0]) * n)
    xx, yy = np.meshgrid(coordinate, coordinate)
    background = periodic_pattern(xx, yy, 0.002)

    transfer, pupil, opd, bfp_x, bfp_y, _ = zemax_transfer(n, window)
    fan_path = ROOT / "results" / "uv_real_objective_fourier_validation" / "real_ray_abbe_and_fourier_fan.csv"
    fan = np.genfromtxt(fan_path, delimiter=",", names=True, encoding="utf-8-sig")
    positive = fan[fan["sin_object"] >= 0]
    slope = float(np.sum(positive["sin_object"] * positive["fourier_y_mm"])
                  / np.sum(positive["sin_object"] ** 2))
    spacing = slope * 0.000355 / 0.002
    rod_centres = ORDERS * spacing
    masks = {
        "open": pupil,
        "matched": serrated_mask(bfp_x, bfp_y, rod_centres, "matched") & pupil,
        "engineering": serrated_mask(bfp_x, bfp_y, rod_centres, "engineering") & pupil,
    }

    spectrum, actual_open = propagate(actual, transfer, masks["open"])
    _, actual_matched = propagate(actual, transfer, masks["matched"])
    _, actual_engineering = propagate(actual, transfer, masks["engineering"])
    _, background_open = propagate(background, transfer, masks["open"])
    _, background_matched = propagate(background, transfer, masks["matched"])
    _, background_engineering = propagate(background, transfer, masks["engineering"])

    arrays = {
        "open": actual_open.astype(np.float32),
        "matched": actual_matched.astype(np.float32),
        "engineering": actual_engineering.astype(np.float32),
    }
    python_open = source["open_intensity"]
    python_filtered = source["filtered_intensity"]
    matched_metrics = measure(
        coordinate, background_open, background_matched,
        actual_open, actual_matched, centres_particles, sizes_particles
    )
    engineering_metrics = measure(
        coordinate, background_open, background_engineering,
        actual_open, actual_engineering, centres_particles, sizes_particles
    )

    def cosine_similarity(first, second):
        numerator = float(np.vdot(first.ravel(), second.ravel()).real)
        return numerator / max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-30)

    prior_comparison = json.loads(
        (args.input.resolve().parent / "python_vs_zemax_2um_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    python_metrics = prior_comparison["metrics"][0]
    metrics = {
        "engine": "real OpticStudio objective OPD + measured BFP mapping + sampled Fourier propagation",
        "native_pop_used_for_final_image": False,
        "why": "native POP undersamples the 2 um carrier after the beam expands at the first real refracting surface",
        "input_source": str(args.input.resolve()),
        "input_transposed": False,
        "grid": n,
        "window_mm": window,
        "object_sample_um": window / n * 1000,
        "zemax_opd_rms_waves": float(np.std(opd[pupil])),
        "real_bfp_sine_slope_mm": slope,
        "two_um_order_spacing_mm": spacing,
        "rod_centres_real_bfp_mm": rod_centres.tolist(),
        "rod_dimensions_real_bfp_mm": {
            "core": CORE_BFP_MM,
            "one_sided_depth": DEPTH_BFP_MM,
            "tooth_pitch": PITCH_BFP_MM,
        },
        "direct_python_comparison": {
            "python_background_suppression_db": python_metrics["pattern_region_suppression_db"],
            "zemax_background_suppression_db": matched_metrics["background_suppression_db"],
            "python_three_particle_peak_retention_percent": python_metrics["three_particle_peak_retention_percent"],
            "zemax_three_particle_peak_retention_percent": matched_metrics["particle_peak_retention_percent"],
            "open_image_cosine_similarity": cosine_similarity(python_open, arrays["open"]),
            "filtered_image_cosine_similarity": cosine_similarity(python_filtered, arrays["matched"]),
            "zemax_over_python_open_peak": float(arrays["open"].max() / python_open.max()),
            "zemax_over_python_filtered_peak": float(arrays["matched"].max() / python_filtered.max()),
        },
        "matched_to_unchanged_python": matched_metrics,
        "final_engineering_orientation": engineering_metrics,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out / "resolved_fields.npz", coordinate_mm=coordinate,
        object_field=actual, background_field=background,
        open_intensity=arrays["open"], matched_intensity=arrays["matched"],
        engineering_intensity=arrays["engineering"],
        particle_centres_mm=centres_particles, particle_sizes_um=sizes_particles,
    )
    save_images(out, coordinate, arrays, centres_particles)
    save_python_zemax_comparison(
        out, coordinate, python_open, python_filtered,
        arrays["open"], arrays["matched"], centres_particles
    )
    save_spectra(out, spectrum, transfer, masks, bfp_x, bfp_y, rod_centres)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
