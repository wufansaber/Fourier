"""Plot audited native Zemax POP grids using shared, non-subtracted scales."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


MAG_IMAGE = 20.0
BFP_SPACING_MM = 5.325
M_FX = 2.816901408450704
CORE_BFP_MM = 1.0 / M_FX
DEPTH_BFP_MM = 0.5 / M_FX
ORDERS_A = (-2, 0, 2)
ORDERS_B = (-3, -1, 1, 3)


def extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])


def save_both(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def image_comparison(data, summary, out: Path) -> None:
    x = data["particle_open_s6_x"] / MAG_IMAGE * 1000.0
    y = data["particle_open_s6_y"] / MAG_IMAGE * 1000.0
    before = data["particle_open_s6_irradiance"]
    after = data["particle_filtered_s6_irradiance"]
    common_peak = float(summary["metrics"]["common_peak_W_per_mm2"])
    floor_db = -45.0
    before_db = 10 * np.log10(np.maximum(before / common_peak, 10 ** (floor_db / 10)))
    after_db = 10 * np.log10(np.maximum(after / common_peak, 10 ** (floor_db / 10)))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.3), constrained_layout=True)
    linear_images = []
    for ax, values, title in zip(
        axes[0], (before, after),
        ("Unfiltered raw image", "Seven serrated rods: raw filtered image"),
    ):
        image = ax.imshow(values, origin="lower", extent=extent(x, y), cmap="magma",
                          vmin=0.0, vmax=common_peak, interpolation="nearest", aspect="auto")
        linear_images.append(image)
        ax.set_title(title)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-3.125, 3.125)
        ax.set_xlabel("Object-equivalent X (um), image X / 20")
        ax.set_ylabel("Object-equivalent Y (um), image Y / 20")
        ax.plot(-1.0, 0.0, marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor="cyan", markeredgewidth=1.2)
    cbar = fig.colorbar(linear_images[0], ax=axes[0], shrink=0.90)
    cbar.set_label("Native Zemax irradiance (W/mm^2), one absolute scale")

    db_images = []
    for ax, values, title in zip(
        axes[1], (before_db, after_db),
        ("Unfiltered, common-reference dB", "Filtered, common-reference dB"),
    ):
        image = ax.imshow(values, origin="lower", extent=extent(x, y), cmap="turbo",
                          vmin=floor_db, vmax=0.0, interpolation="nearest", aspect="auto")
        db_images.append(image)
        ax.set_title(title)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-3.125, 3.125)
        ax.set_xlabel("Object-equivalent X (um), image X / 20")
        ax.set_ylabel("Object-equivalent Y (um), image Y / 20")
        ax.plot(-1.0, 0.0, marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor="white", markeredgewidth=1.2)
    cbar = fig.colorbar(db_images[0], ax=axes[1], shrink=0.90)
    cbar.set_label("dB relative to the same unfiltered peak")

    metrics = summary["metrics"]
    fig.suptitle(
        "Native Zemax POP, ideal 30/600 mm 4f system | 355 nm, NA 0.6, 20x\n"
        "2 um bright periodic pattern + one irregular 0.6 um bright particle; "
        "no background subtraction\n"
        f"background suppression {metrics['background_suppression_db']:.2f} dB | "
        f"particle peak retained {metrics['particle_peak_retention_percent']:.1f}% | "
        f"filtered particle/pattern {metrics['particle_to_pattern_filtered_db']:.2f} dB",
        fontsize=12.5,
    )
    save_both(fig, out / "zemax_native_particle_same_absolute_scale")
    plt.close(fig)


def fourier_alignment(data, out: Path) -> None:
    x0 = data["particle_open_s3_x"]
    y0 = data["particle_open_s3_y"]
    before = data["particle_open_s3_irradiance"]
    x1 = data["particle_filtered_s4_x"]
    y1 = data["particle_filtered_s4_y"]
    after = data["particle_filtered_s4_irradiance"]
    bfp_span = float(x0[-1] - x0[0])
    reference = float(before.max())
    floor_db = -60.0
    arrays = [10 * np.log10(np.maximum(values / reference, 10 ** (floor_db / 10)))
              for values in (before, after)]

    fig = plt.figure(figsize=(14.0, 8.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.75, 1.25))
    spectrum_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    dy0 = abs(y0[1] - y0[0])
    dy1 = abs(y1[1] - y1[0])
    line_before = np.sum(before, axis=0) * dy0
    line_after = np.sum(after, axis=0) * dy1
    line_reference = float(line_before.max())
    spectrum_ax.plot(x0, 10 * np.log10(np.maximum(line_before / line_reference, 1e-8)),
                     color="#155f97", linewidth=1.15, label="Before rods")
    spectrum_ax.plot(x1, 10 * np.log10(np.maximum(line_after / line_reference, 1e-8)),
                     color="#d1495b", linewidth=1.15, label="After both decks")
    spectrum_ax.set_xlim(-18.15, 18.15)
    spectrum_ax.set_ylim(-65, 2)
    spectrum_ax.set_ylabel("Y-integrated spectrum\n(dB, one pre-filter reference)")
    spectrum_ax.set_xlabel("Fourier-plane X (mm)")
    spectrum_ax.grid(True, alpha=0.22)
    spectrum_ax.legend(loc="upper right")
    max_width = CORE_BFP_MM + 2 * DEPTH_BFP_MM
    for order in range(-3, 4):
        centre = order * BFP_SPACING_MM
        deck = "A" if order in ORDERS_A else "B"
        color = "#f0b429" if deck == "A" else "#5b4b8a"
        spectrum_ax.axvspan(centre - max_width / 2, centre + max_width / 2,
                           facecolor=color, edgecolor=color, alpha=0.18)
        spectrum_ax.text(centre, -61, f"m={order}\n{deck}", ha="center", va="bottom",
                         fontsize=7.5, color=color)
    spectrum_ax.annotate("5.325 mm", xy=(0, -7), xytext=(BFP_SPACING_MM, -7),
                         arrowprops=dict(arrowstyle="<->", color="black"),
                         ha="center", va="bottom", fontsize=9)

    images = []
    for ax, values, x, y, title in zip(
        axes, arrays, (x0, x1), (y0, y1),
        ("Before rods: native 2D Fourier plane",
         "After Deck A + Deck B: native 2D Fourier plane"),
    ):
        image = ax.imshow(values, origin="lower", extent=extent(x, y), cmap="turbo",
                          vmin=floor_db, vmax=0.0, interpolation="nearest", aspect="auto")
        images.append(image)
        ax.set_xlim(-18.15, 18.15)
        ax.set_ylim(-0.55, 0.55)
        ax.set_ylabel("Fourier-plane Y (mm)")
        ax.set_title(title)
        for order in range(-3, 4):
            centre = order * BFP_SPACING_MM
            deck = "A" if order in ORDERS_A else "B"
            color = "white" if deck == "A" else "black"
            ax.add_patch(Rectangle((centre - max_width / 2, -0.55), max_width, 1.10,
                                   facecolor=color, edgecolor=color, alpha=0.16,
                                   linewidth=0.8))
            ax.axvline(centre, color=color, linewidth=0.75, alpha=0.85)
            ax.text(centre, 0.50, f"m={order}\nDeck {deck}", color=color,
                    ha="center", va="top", fontsize=7.5,
                    bbox=dict(facecolor="gray", edgecolor="none", alpha=0.25, pad=1))
        ax.set_xlabel("Fourier-plane X (mm)")
    cbar = fig.colorbar(images[0], ax=axes, shrink=0.92)
    cbar.set_label("Native Zemax irradiance (dB, same pre-filter peak reference)")
    fig.suptitle(
        "Audited native POP Fourier plane: seven peaks aligned to seven rods\n"
        f"1024 grid, {bfp_span:.3f} mm window, 36 mm pupil | "
        "Equivalent original-BFP rod envelope: 0.355 mm core + 0.178 mm serration per side; "
        "hardware scale is 2.8169x in X",
        fontsize=12.5,
    )
    save_both(fig, out / "zemax_native_fourier_alignment")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    data = np.load(directory / "native_pop_grids.npz")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    image_comparison(data, summary, directory)
    fourier_alignment(data, directory)
    print(directory / "zemax_native_particle_same_absolute_scale.png")
    print(directory / "zemax_native_fourier_alignment.png")


if __name__ == "__main__":
    main()
