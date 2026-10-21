"""Plot saved Zemax POP grids in the stable fourier-filter environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER_SPACING_HARDWARE_MM = 4.4375
LINEAR_WIDTH_MM = 1.0
FUZZY_MIN_MM = 1.1
FUZZY_MAX_MM = 4.0
SERRATION_PITCH_MM = 0.75


def load_data(path: Path):
    archive = np.load(path)
    keys = sorted({name.rsplit("_", 1)[0] for name in archive if name.endswith("_x")})
    return {
        key: (
            archive[f"{key}_x"],
            archive[f"{key}_y"],
            archive[f"{key}_irradiance"],
        )
        for key in keys
    }


def display(image, log=False):
    image = np.maximum(np.asarray(image, float), 0)
    peak = float(image.max())
    scaled = image / peak if peak else image
    return np.log10(np.maximum(scaled, 1e-8)) if log else scaled


def interpolate(source_x, source_y, source, target_x, target_y):
    intermediate = np.vstack(
        [np.interp(target_x, source_x, row, left=0, right=0) for row in source]
    )
    return np.vstack(
        [np.interp(target_y, source_y, intermediate[:, i], left=0, right=0) for i in range(len(target_x))]
    ).T


def triangle_wave(y):
    phase = y / SERRATION_PITCH_MM
    return 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1


def plot_filter_geometry(output_dir: Path):
    y = np.linspace(-5, 5, 3001)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, scale, title in (
        (axes[0], 1.0, "Mechanical mask at 5x conjugate pupil"),
        (axes[1], 0.2, "Equivalent mask at objective BFP (Zemax POP)"),
    ):
        for order in range(-5, 6):
            center = order * ORDER_SPACING_HARDWARE_MM * scale
            if order % 2:
                width = np.full_like(y, LINEAR_WIDTH_MM * scale)
                color = "#1b77b3"
            else:
                midpoint = (FUZZY_MIN_MM + FUZZY_MAX_MM) / 2
                excursion = (FUZZY_MAX_MM - FUZZY_MIN_MM) / 2
                width = (midpoint + excursion * triangle_wave(y / scale)) * scale
                color = "#d95f02"
            ax.fill_betweenx(y, center - width / 2, center + width / 2, color=color, alpha=0.86)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_xlim(-26, 26)
    axes[1].set_xlim(-5.2, 5.2)
    fig.savefig(output_dir / "zemax_filter_geometry.png", dpi=180)
    plt.close(fig)


def plot_transform(data, output_dir: Path):
    panels = [
        ("defect_unfiltered_s3", "Objective BFP before mask", True),
        ("defect_hard_s3", "After equivalent straight rods", True),
        ("defect_hybrid_s3", "After equivalent hybrid mask", True),
        ("defect_unfiltered_s5", "Unfiltered 20x image", False),
        ("defect_hard_s5", "Straight-rod filtered image", False),
        ("defect_hybrid_s5", "Two-deck filtered image", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for ax, (key, title, use_log) in zip(axes.flat, panels):
        x, y, image = data[key]
        shown = display(image, use_log)
        im = ax.imshow(
            shown,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            cmap="magma",
            aspect="auto",
            vmin=-6 if use_log else 0,
            vmax=0 if use_log else 1,
        )
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.80)
    fig.savefig(output_dir / "zemax_pop_transform_process.png", dpi=180)
    plt.close(fig)


def plot_image_comparison(data, output_dir: Path):
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12), constrained_layout=True)
    for row, kind in enumerate(("unfiltered", "hard", "hybrid")):
        xr, yr, regular = data[f"regular_{kind}_s5"]
        xd, yd, defect = data[f"defect_{kind}_s5"]
        defect = interpolate(xd, yd, defect, xr, yr)
        scale = max(float(regular.max()), float(defect.max()), 1e-30)
        difference = np.abs(defect - regular)
        for col, (panel, title) in enumerate(
            zip(
                (regular / scale, defect / scale, difference / scale),
                ("regular", "local shifted bar", "absolute difference"),
            )
        ):
            ax = axes[row, col]
            vmax = 1 if col < 2 else max(float(panel.max()), 1e-8)
            im = ax.imshow(
                panel,
                extent=[xr[0], xr[-1], yr[0], yr[-1]],
                origin="lower",
                cmap="magma",
                aspect="auto",
                vmin=0,
                vmax=vmax,
            )
            ax.set_title(f"{kind}: {title}")
            ax.set_xlabel("image x (mm)")
            ax.set_ylabel("image y (mm)")
            fig.colorbar(im, ax=ax, shrink=0.78)
    fig.savefig(output_dir / "zemax_image_comparison.png", dpi=180)
    plt.close(fig)


def plot_profiles(data, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for key, label in (
        ("defect_unfiltered_s3", "before mask"),
        ("defect_hard_s3", "straight rods"),
        ("defect_hybrid_s3", "hybrid rods"),
    ):
        x, _, image = data[key]
        profile = np.max(image, axis=0)
        axes[0].semilogy(x, profile / max(float(profile.max()), 1e-30), label=label)
    for order in range(-5, 6):
        axes[0].axvline(order * 0.8875, color="0.75", lw=0.6, zorder=0)
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(1e-8, 2)
    axes[0].set_title("BFP peak envelope (max over y)")
    axes[0].set_xlabel("objective BFP x (mm)")
    axes[0].set_ylabel("normalized irradiance")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    for kind in ("unfiltered", "hard", "hybrid"):
        x, _, regular = data[f"regular_{kind}_s5"]
        axes[1].plot(x, np.max(regular, axis=0) / max(float(regular.max()), 1e-30), label=kind)
    axes[1].set_xlim(-5, 5)
    axes[1].set_title("Regular-array image envelope")
    axes[1].set_xlabel("image x (mm)")
    axes[1].set_ylabel("normalized irradiance")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.savefig(output_dir / "zemax_profiles.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    data = load_data(output_dir / "zemax_pop_grids.npz")
    plot_filter_geometry(output_dir)
    plot_transform(data, output_dir)
    plot_image_comparison(data, output_dir)
    plot_profiles(data, output_dir)
    print(f"plots={output_dir}")


if __name__ == "__main__":
    main()
