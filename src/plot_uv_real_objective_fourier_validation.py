"""Plot real-ray and native-POP validation of the six-element UV objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


MAG = 20.0
FLOOR_DB = -55.0
ORDERS_A = (-2, 0, 2)
ORDERS_B = (-3, -1, 1, 3)


def extent(x, y):
    return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_ray_validation(directory, summary):
    fan = np.genfromtxt(directory / "real_ray_abbe_and_fourier_fan.csv",
                        delimiter=",", names=True, encoding="utf-8-sig")
    order = np.genfromtxt(directory / "real_objective_frequency_order_positions.csv",
                          delimiter=",", names=True, encoding="utf-8-sig")
    ray = summary["ray_validation"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    axes[0, 0].plot(fan["sin_object"], fan["abbe_sine_residual"] * 1e4,
                    color="#235789", lw=1.6)
    axes[0, 0].axhline(0, color="black", lw=0.7)
    axes[0, 0].set(xlabel=r"object ray $\sin u_o$",
                   ylabel=r"$10^4[\sin u_o+20\sin u_i]$",
                   title="1. Abbe sine residual from 401 native real rays")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(fan["sin_object"], fan["fourier_y_mm"],
                    color="#d1495b", lw=2, label="Zemax real-ray intercept")
    axes[0, 1].plot(fan["sin_object"], fan["fourier_fit_mm"], "--",
                    color="black", lw=1, label="best straight line")
    axes[0, 1].set(xlabel=r"object angular spectrum coordinate $\sin u_o$",
                   ylabel="physical Fourier-plane Y (mm)",
                   title="2. Physical BFP coordinate is linear in sine angle")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(fan["sin_object"], fan["fourier_residual_um"],
                    color="#2a9d8f", lw=1.6)
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set(xlabel=r"object ray $\sin u_o$",
                   ylabel="BFP straight-line residual (um)",
                   title="Fourier-coordinate nonlinearity residual")
    axes[1, 0].grid(alpha=0.2)

    p2 = order[np.isclose(order["period_um"], 2.0)]
    axes[1, 1].stem(p2["predicted_fourier_y_mm"], np.ones(len(p2)),
                    linefmt="#7b2cbf", markerfmt="o", basefmt=" ")
    for row in p2:
        axes[1, 1].text(row["predicted_fourier_y_mm"], 1.02,
                        f"{int(row['order'])}", ha="center", va="bottom", fontsize=8)
    axes[1, 1].set(xlim=(-18.5, 18.5), ylim=(0, 1.18), yticks=[],
                   xlabel="physical Fourier-plane Y (mm)",
                   title="2 um sample: accessible diffraction-order centres")
    axes[1, 1].grid(axis="x", alpha=0.2)

    abbe = ray["abbe_sine"]
    linear = ray["fourier_linearity"]
    fig.suptitle(
        "Six-element 355 nm Corning-7980-equivalent objective: native Zemax real-ray validation\n"
        f"Abbe f-sin RMS {abbe['equivalent_fsin_rms_um_at_30mm']:.3f} um, "
        f"max {abbe['equivalent_fsin_max_um_at_30mm']:.3f} um | "
        f"BFP Y={linear['slope_mm']:.6f} sin(u), R2={linear['r_squared']:.9f}, "
        f"max nonlinearity {linear['max_abs_residual_um']:.2f} um",
        fontsize=13,
    )
    save(fig, directory / "real_objective_abbe_and_fourier_linearity")


def plot_fourier(directory, data, summary):
    x0 = data["particle_open_fourier_x"]
    y0 = data["particle_open_fourier_y"]
    before = data["particle_open_fourier_irradiance"]
    x1 = data["particle_filtered_fourier_x"]
    y1 = data["particle_filtered_fourier_y"]
    after = data["particle_filtered_fourier_irradiance"]
    reference = float(before.max())
    values_db = [10 * np.log10(np.maximum(v / reference, 10 ** (FLOOR_DB / 10)))
                 for v in (before, after)]
    centres = {int(k): float(v) for k, v in
               summary["filter"]["rod_centres_y_mm"].items()}

    fig = plt.figure(figsize=(14.5, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.7, 1.3))
    profile_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    profile_before = np.sum(before, axis=1) * abs(x0[1] - x0[0])
    profile_after = np.sum(after, axis=1) * abs(x1[1] - x1[0])
    profile_ref = float(profile_before.max())
    profile_ax.plot(y0, 10 * np.log10(np.maximum(profile_before / profile_ref, 1e-8)),
                    lw=1.2, color="#235789", label="before rods")
    profile_ax.plot(y1, 10 * np.log10(np.maximum(profile_after / profile_ref, 1e-8)),
                    lw=1.2, color="#d1495b", label="after both decks")
    for order, centre in centres.items():
        color = "#f0b429" if order in ORDERS_A else "#6a4c93"
        profile_ax.axvspan(centre - 1.0, centre + 1.0, color=color, alpha=0.13)
        profile_ax.text(centre, -52, f"m={order}", ha="center", va="bottom",
                        fontsize=7, color=color)
    profile_ax.set(xlim=(-18.2, 18.2), ylim=(-58, 2),
                   xlabel="physical Fourier-plane Y (mm)",
                   ylabel="X-integrated irradiance\n(dB, one pre-filter reference)")
    profile_ax.grid(alpha=0.2)
    profile_ax.legend(fontsize=8)

    images = []
    for ax, array, x, y, title in zip(
            axes, values_db, (x0, x1), (y0, y1),
            ("Before rods: real-objective native POP",
             "After two serrated decks + 36 mm stop")):
        image = ax.imshow(array, origin="lower", extent=extent(x, y), cmap="turbo",
                          vmin=FLOOR_DB, vmax=0, interpolation="nearest", aspect="auto")
        images.append(image)
        ax.set(xlim=(-4.0, 4.0), ylim=(-18.2, 18.2),
               xlabel="Fourier X (mm): rod long axis / serration",
               ylabel="Fourier Y (mm): sample orders / rod positions", title=title)
        for order, centre in centres.items():
            color = "white" if order in ORDERS_A else "black"
            ax.add_patch(Rectangle((-4.0, centre - 1.0), 8.0, 2.0,
                                   edgecolor=color, facecolor=color, alpha=0.12, lw=0.8))
            ax.axhline(centre, color=color, lw=0.7, alpha=0.8)
    cbar = fig.colorbar(images[0], ax=axes, shrink=0.9)
    cbar.set_label("Native POP irradiance (dB to one unfiltered BFP peak)")
    fig.suptitle(
        "Measured-frequency alignment in the real 36 mm Fourier pupil\n"
        "Rods are long in X and centred on the real-objective Y order positions; "
        "1.0 mm core + 0.5 mm teeth per side",
        fontsize=13,
    )
    save(fig, directory / "real_objective_fourier_plane_before_after")


def plot_images(directory, data, summary):
    x = data["particle_open_image_x"] / MAG * 1000
    y = data["particle_open_image_y"] / MAG * 1000
    before = data["particle_open_image_irradiance"]
    xf = data["particle_filtered_image_x"] / MAG * 1000
    yf = data["particle_filtered_image_y"] / MAG * 1000
    after = data["particle_filtered_image_irradiance"]
    common = float(summary["pop"]["metrics"]["common_unfiltered_peak_W_per_mm2"])
    db = [10 * np.log10(np.maximum(v / common, 10 ** (FLOOR_DB / 10)))
          for v in (before, after)]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5), constrained_layout=True)
    linear_images = []
    for ax, v, xx, yy, title in zip(
            axes[0], (before, after), (x, xf), (y, yf),
            ("Unfiltered raw TDI image", "Filtered raw TDI image")):
        image = ax.imshow(v, origin="lower", extent=extent(xx, yy), cmap="magma",
                          vmin=0, vmax=common, interpolation="nearest", aspect="auto")
        linear_images.append(image)
        ax.set(xlim=(-3.125, 3.125), ylim=(-8, 8),
               xlabel="object-equivalent X (um): TDI scan",
               ylabel="object-equivalent Y (um): camera long edge", title=title)
        ax.plot(0, -1.0, marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor="cyan", markeredgewidth=1.1)
    cb = fig.colorbar(linear_images[0], ax=axes[0], shrink=0.9)
    cb.set_label("Native Zemax irradiance (W/mm2), identical absolute scale")
    db_images = []
    for ax, v, xx, yy, title in zip(
            axes[1], db, (x, xf), (y, yf),
            ("Unfiltered, common-reference dB", "Filtered, common-reference dB")):
        image = ax.imshow(v, origin="lower", extent=extent(xx, yy), cmap="turbo",
                          vmin=FLOOR_DB, vmax=0, interpolation="nearest", aspect="auto")
        db_images.append(image)
        ax.set(xlim=(-3.125, 3.125), ylim=(-8, 8),
               xlabel="object-equivalent X (um): TDI scan",
               ylabel="object-equivalent Y (um): camera long edge", title=title)
        ax.plot(0, -1.0, marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor="white", markeredgewidth=1.1)
    cb = fig.colorbar(db_images[0], ax=axes[1], shrink=0.9)
    cb.set_label("dB relative to the same unfiltered raw-image peak")
    m = summary["pop"]["metrics"]
    fig.suptitle(
        "Native Zemax POP through six real 7980-equivalent elements + ideal tube lens\n"
        "2 um horizontal pattern + one irregular bright particle; 25-stage TDI crop; "
        "no background subtraction\n"
        f"raw background suppression {m['background_suppression_db']:.3f} dB | "
        f"raw particle peak retention {m['particle_peak_retention_percent']:.1f}%",
        fontsize=12.5,
    )
    save(fig, directory / "real_objective_filter_same_absolute_scale")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    data = np.load(directory / "native_real_objective_pop_grids.npz")
    plot_ray_validation(directory, summary)
    plot_fourier(directory, data, summary)
    plot_images(directory, data, summary)
    print(directory)


if __name__ == "__main__":
    main()
