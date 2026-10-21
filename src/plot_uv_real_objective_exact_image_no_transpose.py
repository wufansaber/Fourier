"""Plot the exact, untransposed 2 um object and native Zemax POP grids."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAG = 20.0
FLOOR_DB = -40.0


def crop(values, x, y, x_limit=0.018, y_limit=0.006):
    x_keep = np.abs(x) <= x_limit
    y_keep = np.abs(y) <= y_limit
    return values[np.ix_(y_keep, x_keep)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()

    source = np.load(directory / "python_reference_2um.npz")
    zemax = np.load(directory / "zemax_native_pop_2um.npz")
    coordinate = source["coordinate_mm"]
    input_intensity = np.abs(source["object_field"]) ** 2
    python_open = source["open_intensity"]

    # Zemax image is inverted at -20x. Reverse both axes and express the grid
    # in ascending object-equivalent coordinates. M^2 converts irradiance area.
    zx = np.sort(-zemax["open_x_mm"] / MAG)
    zy = np.sort(-zemax["open_y_mm"] / MAG)
    zemax_open = zemax["open_irradiance_W_mm2"][::-1, ::-1] * MAG**2
    zemax_filtered = zemax["filtered_irradiance_W_mm2"][::-1, ::-1] * MAG**2

    input_crop = crop(input_intensity, coordinate, coordinate)
    python_crop = crop(python_open, coordinate, coordinate)
    zemax_open_crop = crop(zemax_open, zx, zy)
    zemax_filtered_crop = crop(zemax_filtered, zx, zy)
    common = max(float(python_crop.max()), float(zemax_open_crop.max()), 1e-30)

    panels = (
        (input_crop / max(float(input_crop.max()), 1e-30),
         "A. Exact Python object intensity\n(no transpose; display-normalized only)"),
        (python_crop / common, "B. Python unfiltered reference"),
        (zemax_open_crop / common, "C. Zemax real lens | open"),
        (zemax_filtered_crop / common,
         "D. Zemax real lens | final rods Y-long / X-positioned"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    shown = None
    for axis, (values, title) in zip(axes.flat, panels):
        db = 10 * np.log10(np.maximum(values, 10 ** (FLOOR_DB / 10)))
        shown = axis.imshow(
            db, origin="lower", extent=(-18, 18, -6, 6), cmap="turbo",
            vmin=FLOOR_DB, vmax=0, interpolation="nearest", aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("Object X (um)")
        axis.set_ylabel("Object Y (um): 2 um periodic direction")
    fig.colorbar(shown, ax=axes, fraction=0.025, pad=0.02, label="dB")
    fig.suptitle(
        "2 um horizontal-band three-particle case | exact image orientation\n"
        "No transpose and no background subtraction; B-D use one common open-image scale",
        fontsize=12.5,
    )
    output = directory / "exact_image_no_transpose_zemax_pop.png"
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
