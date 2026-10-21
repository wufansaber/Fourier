"""Plot an apples-to-apples Python versus real-objective Zemax 2 um comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


MAG_IMAGE = 20.0
FLOOR_DB = -35.0


def oriented_object_coordinates(x_image_mm, y_image_mm, irradiance):
    """Map the inverted -20x Zemax image back to ascending object coordinates."""
    x_object = -np.asarray(x_image_mm) / MAG_IMAGE
    y_object = -np.asarray(y_image_mm) / MAG_IMAGE
    order_x = np.argsort(x_object)
    order_y = np.argsort(y_object)
    # Multiplication by M^2 expresses irradiance per object-equivalent mm^2.
    values = np.asarray(irradiance)[np.ix_(order_y, order_x)] * MAG_IMAGE**2
    return x_object[order_x], y_object[order_y], values


def resample(x, y, values, target):
    interpolator = RegularGridInterpolator((y, x), values, bounds_error=False,
                                            fill_value=0.0)
    xx, yy = np.meshgrid(target, target)
    return interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(xx.shape)


def particle_and_pattern_masks(coordinate):
    xx, yy = np.meshgrid(coordinate, coordinate)
    particle = np.zeros_like(xx, dtype=bool)
    for cx, size_um in zip((-0.012, 0.0, 0.012), (0.6, 0.9, 1.2)):
        radius = max(0.0015, 1.5 * size_um / 1000)
        particle |= (xx - cx)**2 + yy**2 <= radius**2
    view = (np.abs(xx) <= 0.018) & (np.abs(yy) <= 0.006)
    return view & particle, view & ~particle, view


def metrics(coordinate, python_open, python_filtered, zemax_open, zemax_filtered):
    particle, pattern, view = particle_and_pattern_masks(coordinate)
    rows = []
    for engine, before, after in (
            ("Python unchanged model", python_open, python_filtered),
            ("Zemax real-objective native POP", zemax_open, zemax_filtered)):
        background_suppression = 10 * np.log10(
            max(float(np.sum(before[pattern])), 1e-30) /
            max(float(np.sum(after[pattern])), 1e-30))
        peak_retention = (float(np.max(after[particle])) /
                          max(float(np.max(before[particle])), 1e-30))
        common = max(float(np.max(before[view])), 1e-30)
        rows.append({
            "engine": engine,
            "pattern_region_suppression_db": background_suppression,
            "three_particle_peak_retention_percent": 100 * peak_retention,
            "open_peak_common_units": common,
            "filtered_peak_over_open_peak": float(np.max(after[view])) / common,
        })
    py_before = python_open[view] / max(float(np.linalg.norm(python_open[view])), 1e-30)
    ze_before = zemax_open[view] / max(float(np.linalg.norm(zemax_open[view])), 1e-30)
    py_after = python_filtered[view] / max(float(np.linalg.norm(python_filtered[view])), 1e-30)
    ze_after = zemax_filtered[view] / max(float(np.linalg.norm(zemax_filtered[view])), 1e-30)
    comparison = {
        "open_cosine_similarity": float(np.dot(py_before, ze_before)),
        "filtered_cosine_similarity": float(np.dot(py_after, ze_after)),
    }
    return rows, comparison


def save_figure(images, titles, coordinate, reference, path, logarithmic):
    x_roi = np.abs(coordinate) <= 0.018
    y_roi = np.abs(coordinate) <= 0.006
    extent = [-18, 18, -6, 6]
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 11.0), constrained_layout=True)
    shown = None
    for ax, values, title in zip(axes.flat, images, titles):
        crop = values[np.ix_(y_roi, x_roi)] / reference
        if logarithmic:
            crop = 10 * np.log10(np.maximum(crop, 10 ** (FLOOR_DB / 10)))
            limits = (FLOOR_DB, 0.0)
            label = "Raw irradiance (dB to one global unfiltered peak)"
            cmap = "turbo"
        else:
            limits = (0.0, 1.0)
            label = "Normalized raw irradiance (one global scale)"
            cmap = "turbo"
        shown = ax.imshow(crop, origin="lower", extent=extent, cmap=cmap,
                          vmin=limits[0], vmax=limits[1], interpolation="nearest",
                          aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("X (um): original Python-case horizontal axis")
        ax.set_ylabel("Y (um): 2 um periodic direction")
    fig.colorbar(shown, ax=axes, fraction=0.025, pad=0.02, label=label)
    fig.suptitle(
        "2 um pattern + the same three-particle object | Python vs real-lens Zemax POP\n"
        "Same 1 W input field, object-coordinate crop and global scale; no background subtraction",
        fontsize=12.5,
    )
    fig.savefig(path, dpi=210, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    python = np.load(directory / "python_reference_2um.npz")
    zemax = np.load(directory / "zemax_native_pop_2um.npz")
    coordinate = python["coordinate_mm"]
    python_open = python["open_intensity"]
    python_filtered = python["filtered_intensity"]

    zx, zy, zopen = oriented_object_coordinates(
        zemax["open_x_mm"], zemax["open_y_mm"],
        zemax["open_irradiance_W_mm2"])
    zfx, zfy, zfiltered = oriented_object_coordinates(
        zemax["filtered_x_mm"], zemax["filtered_y_mm"],
        zemax["filtered_irradiance_W_mm2"])
    zemax_open = resample(zx, zy, zopen, coordinate)
    zemax_filtered = resample(zfx, zfy, zfiltered, coordinate)

    view = particle_and_pattern_masks(coordinate)[2]
    global_reference = max(
        float(np.max(python_open[view])), float(np.max(zemax_open[view])), 1e-30)
    images = [python_open, python_filtered, zemax_open, zemax_filtered]
    titles = [
        "Python | Unfiltered raw intensity",
        "Python | Filtered raw intensity",
        "Zemax real lens | Unfiltered native POP",
        "Zemax real lens | Filtered native POP",
    ]
    save_figure(images, titles, coordinate, global_reference,
                directory / "python_vs_zemax_2um_same_global_linear_scale.png", False)
    save_figure(images, titles, coordinate, global_reference,
                directory / "python_vs_zemax_2um_same_global_db_scale.png", True)

    rows, comparison = metrics(
        coordinate, python_open, python_filtered, zemax_open, zemax_filtered)
    with (directory / "python_vs_zemax_2um_metrics.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "metrics": rows,
        "shape_comparison": comparison,
        "global_reference_object_equivalent_W_per_mm2": global_reference,
        "display_crop_um": {"x": [-18, 18], "y": [-6, 6]},
        "evaluation": "raw irradiance, one global reference, no background subtraction",
    }
    (directory / "python_vs_zemax_2um_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
