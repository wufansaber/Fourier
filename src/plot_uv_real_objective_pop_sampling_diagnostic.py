"""Document why native surface-to-surface POP loses the 2 um object spectrum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


MAG = 20.0
EXPECTED = np.arange(-3, 4) * 5.165663367412653


def resample_image(zemax):
    x = -zemax["open_x_mm"] / MAG
    y = -zemax["open_y_mm"] / MAG
    values = zemax["open_irradiance_W_mm2"][::-1, ::-1] * MAG**2
    return np.sort(x), np.sort(y), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--start-probe-dir", type=Path, required=True)
    parser.add_argument("--bfp-probe-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    source = np.load(result_dir / "python_reference_2um.npz")
    start = np.load(args.start_probe_dir.resolve() / "zemax_native_pop_2um.npz")
    bfp = np.load(args.bfp_probe_dir.resolve() / "zemax_native_pop_2um.npz")
    final = np.load(result_dir / "zemax_native_pop_2um.npz")

    coordinate = source["coordinate_mm"]
    input_intensity = np.abs(source["object_field"])**2
    start_intensity = start["open_irradiance_W_mm2"]
    # The probes and formal run have different grids; input equality is checked
    # against the source saved beside the start probe.
    start_source = np.load(args.start_probe_dir.resolve() / "python_reference_2um.npz")
    start_reference = np.abs(start_source["object_field"])**2
    start_correlation = float(np.corrcoef(
        start_reference.ravel(), start_intensity.ravel())[0, 1])

    bfp_y = bfp["open_y_mm"]
    bfp_profile = np.sum(bfp["open_irradiance_W_mm2"], axis=1)
    bfp_profile /= max(float(np.max(bfp_profile)), 1e-30)
    order_ratios = {
        str(order): float(bfp_profile[np.argmin(np.abs(bfp_y - position))])
        for order, position in zip(range(-3, 4), EXPECTED)
    }

    zx, zy, zemax_open = resample_image(final)
    interpolator = RegularGridInterpolator((zy, zx), zemax_open,
                                            bounds_error=False, fill_value=0.0)
    xx, yy = np.meshgrid(coordinate, coordinate)
    zemax_on_python = interpolator(
        np.column_stack((yy.ravel(), xx.ravel()))).reshape(xx.shape)
    python_open = source["open_intensity"]

    x_roi = np.abs(coordinate) <= 0.018
    y_roi = np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.8), constrained_layout=True)
    axes[0, 0].imshow(
        input_intensity[np.ix_(y_roi, x_roi)], origin="lower", extent=extent,
        cmap="turbo", interpolation="nearest", aspect="equal")
    axes[0, 0].set_title(f"A. Exact Python field loaded at Zemax S1\n"
                         f"pixel correlation = {start_correlation:.14f}")
    axes[0, 0].set_xlabel("object X (um)")
    axes[0, 0].set_ylabel("object Y (um)")

    axes[0, 1].plot(bfp_y, 10 * np.log10(np.maximum(bfp_profile, 1e-14)),
                    color="#235789", lw=1.2)
    for order, position in zip(range(-3, 4), EXPECTED):
        axes[0, 1].axvline(position, color="#d1495b", lw=0.8, alpha=0.75)
        axes[0, 1].text(position, -136, f"m={order}", rotation=90,
                        ha="center", va="bottom", fontsize=7)
    axes[0, 1].set(xlim=(-18, 18), ylim=(-140, 3),
                   xlabel="real-objective BFP Y (mm)",
                   ylabel="X-integrated irradiance (dB to zero order)",
                   title="B. Native POP at BFP: 2 um orders numerically lost")
    axes[0, 1].grid(alpha=0.2)

    for axis, values, title in (
            (axes[1, 0], python_open, "C. Python reference: 2 um image retained"),
            (axes[1, 1], zemax_on_python,
             "D. Native object-to-image POP: only smooth field remains")):
        crop = values[np.ix_(y_roi, x_roi)]
        db = 10 * np.log10(np.maximum(crop / max(float(crop.max()), 1e-30), 1e-4))
        shown = axis.imshow(db, origin="lower", extent=extent, cmap="turbo",
                            vmin=-40, vmax=0, interpolation="nearest", aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("object-equivalent X (um)")
        axis.set_ylabel("object-equivalent Y (um)")
    fig.colorbar(shown, ax=axes[1], fraction=0.025, pad=0.02,
                 label="dB to each unfiltered peak (diagnostic display only)")
    fig.suptitle(
        "Native OpticStudio POP sampling audit for the same 2 um three-particle object\n"
        "Input is exact, but the 36-40 mm intermediate grid cannot Nyquist-sample a 2 um carrier",
        fontsize=13,
    )
    base = result_dir / "diagnostic_native_pop_sampling_failure"
    fig.savefig(base.with_suffix(".png"), dpi=210, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    required_samples_nyquist = int(np.ceil(40.0 / (PERIOD_MM := 0.002) * 2))
    diagnostic = {
        "status": "native object-to-image POP result is invalid for image comparison",
        "start_surface_intensity_correlation": start_correlation,
        "bfp_expected_order_centres_mm": EXPECTED.tolist(),
        "bfp_intensity_ratios_to_zero_order": order_ratios,
        "manual_bfp_grid_width_mm": 40.0,
        "minimum_square_samples_for_two_points_per_2um_period": required_samples_nyquist,
        "reason": (
            "surface-to-surface POP resampling suppresses the 2 um carrier before the real BFP; "
            "the final smooth image and its filter metrics are therefore not physical"
        ),
        "valid_artifact": "the ZBF input itself is verified against the Python field",
    }
    (base.with_suffix(".json")).write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
