"""Scientific figures for the six-element Corning-7980-equivalent objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.input_dir.resolve()
    surfaces = pd.read_csv(out / "surface_prescription_6element.csv")
    layout = pd.read_csv(out / "layout_rays_6element.csv")
    sine = pd.read_csv(out / "abbe_sine_fan.csv")
    summary = json.loads((out / "simplified_validation_summary.json").read_text(
        encoding="utf-8"))

    fig = plt.figure(figsize=(14.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    objective = surfaces[surfaces.surface <= 13]
    for _, row in objective.iterrows():
        z0 = row.z_vertex_mm
        if row.material == "F_SILICA" and row.surface + 1 in set(objective.surface):
            z1 = float(objective.loc[objective.surface == row.surface + 1, "z_vertex_mm"].iloc[0])
            ax.axvspan(z0, z1, color="#a8dadc", alpha=0.55)
        if row.surface > 0:
            color = "#0077b6" if row.surface == 11 else "#5b6770"
            width = 1.5 if row.surface == 11 else 0.6
            ax.plot([z0, z0], [-row.semi_diameter_mm, row.semi_diameter_mm],
                    color=color, linewidth=width, alpha=0.9)
    colors = {0.0: "#1d3557", 1.0: "#d1495b"}
    styles = {-0.95: "-", 0.0: "--", 0.95: "-"}
    for (field, pupil), group in layout[layout.surface <= 13].groupby(
            ["field_hy", "pupil_y"]):
        group = group.sort_values("surface")
        ax.plot(group.z_vertex_mm, group.ray_y_mm, color=colors[field],
                linestyle=styles[pupil], linewidth=0.9, alpha=0.9)
    ax.set_title("Six real Corning 7980-equivalent elements and 36 mm stop")
    ax.set_xlabel("Axial vertex coordinate (mm)")
    ax.set_ylabel("Ray height Y (mm)")
    ax.grid(True, alpha=0.18)
    ax.text(0.02, 0.96, "blue: on-axis | red: 0.25 mm object height\n"
            "solid: marginal | dashed: chief",
            transform=ax.transAxes, va="top", fontsize=8.5)

    ax = fig.add_subplot(grid[0, 1])
    so = sine.sin_object.to_numpy()
    mapped = -20.0 * sine.sin_image.to_numpy()
    ax.plot(so, mapped, color="#1d70a2", linewidth=1.3, label="Real ray: -20 sin(u_image)")
    ax.plot([-0.6, 0.6], [-0.6, 0.6], "k--", linewidth=1.0,
            label="Ideal Abbe sine mapping")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("sin(u_object)")
    ax.set_ylabel("-20 sin(u_image)")
    ax.set_title("Finite-conjugate Abbe sine condition")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5)

    ax = fig.add_subplot(grid[1, 0])
    error_um = 30.0 * 1000.0 * sine.residual_nominal_sine_scale_minus20.to_numpy()
    ax.plot(sine.normalized_pupil_y, error_um, color="#9b2226", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhspan(-25, 25, color="#94d2bd", alpha=0.22, label="previous max budget +/-25 um")
    ax.set_xlabel("Normalized meridional pupil coordinate")
    ax.set_ylabel("Equivalent f sin(theta) residual (um)")
    ax.set_title("Abbe residual: RMS 2.20 um, max 11.64 um")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5)

    ax = fig.add_subplot(grid[1, 1])
    field_colors = {0.0: "#005f73", 0.5: "#ee9b00", 1.0: "#ae2012"}
    for field in (0.0, 0.5, 1.0):
        data = np.load(out / "rays" / f"field_{field:g}.npz")["image"]
        valid = (data[:, 1] == 0) & (data[:, 2] == 0)
        x = 1000 * (data[valid, 3] - np.mean(data[valid, 3]))
        y = 1000 * (data[valid, 4] - np.mean(data[valid, 4]))
        ax.scatter(x, y, s=1.5, alpha=0.25, color=field_colors[field],
                   label=f"object Y={field * 0.25:.3f} mm")
    airy_radius_um = 0.61 * 0.355 / (0.6 / 20.0)
    ax.add_patch(Circle((0, 0), airy_radius_um, facecolor="none",
                        edgecolor="black", linewidth=1.2, linestyle="--",
                        label=f"Airy first-zero radius {airy_radius_um:.2f} um"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("Image-plane X relative to centroid (um)")
    ax.set_ylabel("Image-plane Y relative to centroid (um)")
    ax.set_title("Real-ray spots: RMS radii 1.66-2.71 um; Airy radius 7.22 um")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=7.8)

    fields = summary["field_validation"]
    fig.suptitle(
        "Native Zemax real-curvature UV objective | 355 nm, NA 0.6, 0.5 mm field, 20x\n"
        "six F_SILICA elements (Corning HPFS 7980 equivalent) + ideal 600 mm tube lens; "
        "coatings omitted\n"
        f"OPD RMS waves: {fields[0]['opd_rms_waves']:.4f} / "
        f"{fields[1]['opd_rms_waves']:.4f} / {fields[2]['opd_rms_waves']:.4f}",
        fontsize=12.3,
    )
    base = out / "uv355_real_objective_abbe_validation"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
