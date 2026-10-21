"""Large-period (>100 um) defect gallery for the current Fourier-filter study.

The current hardware magnification (2.8169x in Fourier-plane Y) no longer
separates individual diffraction orders when the object period is 100--200 um.
Accordingly this script uses one manufacturable central serrated rod as a dense
low-order stop, rather than drawing physically overlapping independent rods.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "large_period_defects"

WAVELENGTH_MM = 0.000355
F_OBJ_MM = 30.0
NA = 0.60
MY = 2.816901408450704
PUPIL_MM = 36.0
CORE_MM = 1.0
TOOTH_DEPTH_MM = 0.5
TOOTH_PITCH_MM = 0.75
PERIODS_UM = [100.0, 150.0, 200.0]


def ft2(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.fft2(fft.ifftshift(a), norm="ortho", workers=4))


def ift2(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.ifft2(fft.ifftshift(a), norm="ortho", workers=4))


def triangular_fraction(x_mm: np.ndarray) -> np.ndarray:
    phase = x_mm / TOOTH_PITCH_MM
    tri = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    return 0.5 * (tri + 1.0)


def central_serrated_rod(x_hw_mm: np.ndarray, y_hw_mm: np.ndarray) -> np.ndarray:
    half_width = CORE_MM / 2 + TOOTH_DEPTH_MM * triangular_fraction(x_hw_mm)
    return (np.abs(y_hw_mm[:, None]) > half_width[None, :]).astype(np.float32)


def raised_cosine_square(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    r = np.maximum(np.abs(xx), np.abs(yy))
    result = np.ones_like(r, dtype=np.float32)
    edge0, edge1 = 0.54, 0.60
    result[r >= edge1] = 0.0
    transition = (r > edge0) & (r < edge1)
    result[transition] = 0.5 + 0.5 * np.cos(np.pi * (r[transition] - edge0) / (edge1 - edge0))
    return result


def base_pattern(xx: np.ndarray, yy: np.ndarray, period_mm: float) -> np.ndarray:
    wrapped = np.remainder(yy + period_mm / 2, period_mm) - period_mm / 2
    slot_half = 0.11 * period_mm
    field = np.ones_like(xx, dtype=np.float32)
    field[np.abs(wrapped) <= slot_half] = 0.12
    return field * raised_cosine_square(xx, yy)


def apply_defect(base: np.ndarray, xx: np.ndarray, yy: np.ndarray, period_mm: float, kind: str):
    result = base.copy()
    slot_half = 0.11 * period_mm
    if kind == "isolated_dark_hole":
        centre = (-0.33, 0.50 * period_mm)
        region = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= 0.004**2
        result[region] = 0.02
    elif kind == "isolated_bright_point":
        centre = (0.33, 0.0)
        region = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= 0.004**2
        result[region] = 1.0
    elif kind == "elliptical_hole":
        centre = (-0.08, 0.50 * period_mm)
        region = ((xx - centre[0]) / 0.007) ** 2 + ((yy - centre[1]) / 0.004) ** 2 <= 1.0
        result[region] = 0.02
    elif kind == "close_hole_pair":
        centre = (0.05, 0.50 * period_mm)
        left = (xx - (centre[0] - 0.010)) ** 2 + (yy - centre[1]) ** 2 <= 0.003**2
        right = (xx - (centre[0] + 0.010)) ** 2 + (yy - centre[1]) ** 2 <= 0.003**2
        result[left | right] = 0.02
    elif kind == "irregular_hole":
        centre = (0.25, -0.50 * period_mm)
        lobe1 = ((xx - (centre[0] - 0.003)) / 0.009) ** 2 + ((yy - centre[1]) / 0.005) ** 2 <= 1.0
        lobe2 = ((xx - (centre[0] + 0.005)) / 0.006) ** 2 + ((yy - (centre[1] + 0.003)) / 0.008) ** 2 <= 1.0
        result[lobe1 | lobe2] = 0.02
    else:
        raise ValueError(kind)
    return result, centre


def mixed_scene(base: np.ndarray, xx: np.ndarray, yy: np.ndarray, period_mm: float):
    actual = base.copy()
    centres = {}
    for kind in ["isolated_dark_hole", "isolated_bright_point", "elliptical_hole", "close_hole_pair", "irregular_hole"]:
        candidate, centre = apply_defect(base, xx, yy, period_mm, kind)
        changed = np.abs(candidate - base) > 1e-7
        actual[changed] = candidate[changed]
        centres[kind] = centre
    return actual, centres


def r90_um(x: np.ndarray, y: np.ndarray, energy: np.ndarray, centre) -> float:
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx - centre[0], yy - centre[1]).ravel()
    weight = np.maximum(energy, 0).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    if cumulative[-1] <= 0:
        return float("nan")
    index = int(np.searchsorted(cumulative, 0.90 * cumulative[-1]))
    return float(radius[order[index]] * 1000)


def simulate_period(period_um: float, grid: dict):
    x, y, xx, yy = grid["x"], grid["y"], grid["xx"], grid["yy"]
    pupil, rod = grid["pupil"], grid["rod"]
    period_mm = period_um / 1000
    background = base_pattern(xx, yy, period_mm)
    actual, centres = mixed_scene(background, xx, yy, period_mm)
    bg_spec = ft2(background)
    actual_spec = ft2(actual)
    u_bg_open = ift2(bg_spec * pupil)
    u_actual_open = ift2(actual_spec * pupil)
    u_bg_filtered = ift2(bg_spec * pupil * rod)
    u_actual_filtered = ift2(actual_spec * pupil * rod)
    delta_open = u_actual_open - u_bg_open
    delta_filtered = u_actual_filtered - u_bg_filtered
    return {
        "period_um": period_um,
        "background": background,
        "actual": actual,
        "centres": centres,
        "i_open": np.abs(u_actual_open) ** 2,
        "i_filtered": np.abs(u_actual_filtered) ** 2,
        "i_bg_open": np.abs(u_bg_open) ** 2,
        "i_bg_filtered": np.abs(u_bg_filtered) ** 2,
        "signed_open": np.abs(u_actual_open) ** 2 - np.abs(u_bg_open) ** 2,
        "signed_filtered": np.abs(u_actual_filtered) ** 2 - np.abs(u_bg_filtered) ** 2,
        "delta_open_energy": np.abs(delta_open) ** 2,
        "delta_filtered_energy": np.abs(delta_filtered) ** 2,
    }


def plot_period_gallery(results: list[dict], x: np.ndarray, y: np.ndarray):
    extent = [x[0] * 1000, x[-1] * 1000, y[0] * 1000, y[-1] * 1000]
    fig, axes = plt.subplots(3, 4, figsize=(16.2, 12.3), constrained_layout=True)
    for row, result in enumerate(results):
        ref = max(float(result["i_open"].max()), 1e-30)
        filtered_db = 10 * np.log10(np.maximum(result["i_filtered"] / ref, 1e-5))
        panels = [
            (result["actual"], "Object amplitude", "gray", 0.0, 1.0, "Amplitude"),
            (result["i_open"] / ref, "Unfiltered image", "magma", 0.0, 1.0, "I / row unfiltered peak"),
            (filtered_db, "Filtered image (enhanced dB display)", "turbo", -50.0, 0.0, "dB re row unfiltered peak"),
        ]
        for col, (data, title, cmap, vmin, vmax, label) in enumerate(panels):
            im = axes[row, col].imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{result['period_um']:.0f} um period | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, label=label)
        signed = result["signed_filtered"] / ref
        lim = max(float(np.max(np.abs(signed))), 1e-12)
        im = axes[row, 3].imshow(signed, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[row, 3].set_title(f"{result['period_um']:.0f} um period | filtered actual - reference")
        fig.colorbar(im, ax=axes[row, 3], fraction=0.046, label="Delta I / row unfiltered peak")
        for col in range(4):
            axes[row, col].set_xlabel("X: TDI scan / rod axis (um)")
            axes[row, col].set_ylabel("Y: camera long edge (um)")
    fig.suptitle(
        "Large-period compact-defect scenes | dark hole, bright point, ellipse, close pair and irregular hole\n"
        "355 nm, f=30 mm, NA=0.6; filtered intensity uses -50...0 dB enhanced display",
        fontsize=14,
    )
    path = OUT / "large_period_mixed_defect_gallery.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_type_gallery(grid: dict):
    x, y, xx, yy = grid["x"], grid["y"], grid["xx"], grid["yy"]
    pupil, rod = grid["pupil"], grid["rod"]
    period_mm = 0.150
    background = base_pattern(xx, yy, period_mm)
    ub0 = ift2(ft2(background) * pupil)
    ub1 = ift2(ft2(background) * pupil * rod)
    kinds = ["isolated_dark_hole", "isolated_bright_point", "elliptical_hole", "close_hole_pair", "irregular_hole"]
    labels = ["Isolated dark hole", "Isolated bright point", "Elliptical hole", "Close hole pair", "Irregular hole"]
    fig, axes = plt.subplots(5, 3, figsize=(11.8, 16.5), constrained_layout=True)
    rows = []
    for row, (kind, label) in enumerate(zip(kinds, labels)):
        actual, centre = apply_defect(background, xx, yy, period_mm, kind)
        ua0 = ift2(ft2(actual) * pupil)
        ua1 = ift2(ft2(actual) * pupil * rod)
        signed0 = np.abs(ua0) ** 2 - np.abs(ub0) ** 2
        signed1 = np.abs(ua1) ** 2 - np.abs(ub1) ** 2
        ref = max(float(np.abs(ua0).max() ** 2), 1e-30)
        # Show more than one full 150 um period in Y so every compact defect is
        # visibly embedded in the same periodic pattern rather than appearing
        # as a free-space standalone target.
        hx, hy = 0.050, 0.095
        ix = (x >= centre[0] - hx) & (x <= centre[0] + hx)
        iy = (y >= centre[1] - hy) & (y <= centre[1] + hy)
        extent = [(centre[0] - hx) * 1000, (centre[0] + hx) * 1000,
                  (centre[1] - hy) * 1000, (centre[1] + hy) * 1000]
        im = axes[row, 0].imshow(actual[np.ix_(iy, ix)], origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label} | object amplitude")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, label="Amplitude")
        lim0 = max(float(np.max(np.abs(signed0[np.ix_(iy, ix)] / ref))), 1e-12)
        lim1 = max(float(np.max(np.abs(signed1[np.ix_(iy, ix)] / ref))), 1e-12)
        contrast_panels = [
            (signed0, "Unfiltered contrast", lim0, "Delta I / unfiltered peak"),
            (signed1, "Filtered contrast (enhanced range)", lim1, "Delta I / unfiltered peak"),
        ]
        for col, (data, title, lim, color_label) in enumerate(contrast_panels, start=1):
            im = axes[row, col].imshow(data[np.ix_(iy, ix)] / ref, origin="lower", extent=extent,
                                       cmap="coolwarm", vmin=-lim, vmax=lim)
            axes[row, col].set_title(f"{label} | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, label=color_label)
        delta0 = np.abs(ua0 - ub0) ** 2
        delta1 = np.abs(ua1 - ub1) ** 2
        e0, e1 = float(delta0.sum()), float(delta1.sum())
        rows.append({
            "period_um": 150.0,
            "defect_type": kind,
            "defect_field_energy_retention": e1 / max(e0, 1e-30),
            "r90_open_um": r90_um(x, y, delta0, centre),
            "r90_filtered_um": r90_um(x, y, delta1, centre),
        })
        for col in range(3):
            axes[row, col].set_xlabel("X (um)")
            axes[row, col].set_ylabel("Y (um)")
    fig.suptitle(
        "150 um periodic pattern: isolated-point and compact-hole library\n"
        "Signed contrast keeps identical units; filtered column uses a separate symmetric range for visibility",
        fontsize=14,
    )
    path = OUT / "large_period_defect_type_gallery.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return rows


def plot_spacing_and_metrics(period_rows: list[dict]):
    periods = np.linspace(80, 250, 300)
    spacing = MY * F_OBJ_MM * WAVELENGTH_MM / (periods / 1000)
    required_my = 1.5 * (periods / 1000) / (F_OBJ_MM * WAVELENGTH_MM)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(periods, spacing, color="#2F5AA8", lw=2, label="Current expanded order spacing")
    ax.axhline(1.5, color="#D48A27", ls="--", label="Desired 1.5 mm separation")
    ax.axhspan(0, 1.0, color="#D6D9DE", alpha=0.45, label="Inside 2 mm rod half-envelope")
    for p in PERIODS_UM:
        s = MY * F_OBJ_MM * WAVELENGTH_MM / (p / 1000)
        ax.plot(p, s, "o", color="#2F5AA8")
        ax.annotate(f"{p:.0f} um: {s:.3f} mm", (p, s), xytext=(5, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Object period (um)")
    ax.set_ylabel("Adjacent-order spacing at mechanical Fourier plane (mm)")
    ax.set_title("Large periods create densely packed diffraction orders")
    ax.grid(color="#D5D8DC", lw=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = [f"{r['period_um']:.0f} um" for r in period_rows]
    suppression = [r["background_suppression_db"] for r in period_rows]
    retention = [100 * r["mixed_defect_field_energy_retention"] for r in period_rows]
    xp = np.arange(len(names))
    width = 0.37
    ax.bar(xp - width / 2, suppression, width, color="#2F5AA8", label="Background suppression (dB)")
    ax2 = ax.twinx()
    ax2.bar(xp + width / 2, retention, width, color="#D48A27", label="Mixed-defect field retention (%)")
    ax.set_xticks(xp, names)
    ax.set_ylabel("Background suppression (dB)")
    ax2.set_ylabel("Defect field-energy retention (%)")
    ax.set_title("One central rod: suppression/retention trade-off")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#D5D8DC", lw=0.6)
    path = OUT / "large_period_spacing_and_metrics.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 2048
    fov_mm = 1.2
    dx = fov_mm / n
    x = (np.arange(n) - n // 2) * dx
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    fx = fft.fftshift(fft.fftfreq(n, d=dx))
    fy = fx.copy()
    fxx, fyy = np.meshgrid(fx, fy)
    pupil = (((WAVELENGTH_MM * fxx) ** 2 + (WAVELENGTH_MM * fyy) ** 2) <= NA**2).astype(np.float32)
    x_hw = F_OBJ_MM * WAVELENGTH_MM * fx
    y_hw = MY * F_OBJ_MM * WAVELENGTH_MM * fy
    rod = central_serrated_rod(x_hw, y_hw)
    grid = {"x": x, "y": y, "xx": xx, "yy": yy, "pupil": pupil, "rod": rod}

    results = [simulate_period(p, grid) for p in PERIODS_UM]
    period_rows = []
    for result in results:
        e_bg0 = float(result["i_bg_open"].sum())
        e_bg1 = float(result["i_bg_filtered"].sum())
        e_d0 = float(result["delta_open_energy"].sum())
        e_d1 = float(result["delta_filtered_energy"].sum())
        p = result["period_um"]
        period_rows.append({
            "period_um": p,
            "current_order_spacing_mm": MY * F_OBJ_MM * WAVELENGTH_MM / (p / 1000),
            "required_My_for_1p5mm_spacing": 1.5 * (p / 1000) / (F_OBJ_MM * WAVELENGTH_MM),
            "required_expanded_pupil_y_mm": PUPIL_MM * 1.5 * (p / 1000) / (F_OBJ_MM * WAVELENGTH_MM),
            "background_suppression_db": 10 * np.log10(e_bg0 / max(e_bg1, 1e-30)),
            "mixed_defect_field_energy_retention": e_d1 / max(e_d0, 1e-30),
        })
    type_rows = plot_type_gallery(grid)
    plot_period_gallery(results, x, y)
    plot_spacing_and_metrics(period_rows)

    all_rows = period_rows + type_rows
    keys = sorted({key for row in all_rows for key in row})
    with (OUT / "large_period_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "parameters": {
            "wavelength_nm": 355,
            "objective_focal_length_mm": F_OBJ_MM,
            "objective_na": NA,
            "current_fourier_y_magnification": MY,
            "rod_core_mm": CORE_MM,
            "tooth_depth_each_side_mm": TOOTH_DEPTH_MM,
            "tooth_pitch_mm": TOOTH_PITCH_MM,
            "grid": [n, n],
            "object_fov_mm": fov_mm,
        },
        "period_results": period_rows,
        "defect_type_results_at_150um": type_rows,
        "model_scope": "ideal coherent scalar 2-D model; horizontal grooves are the 90-degree-rotated equivalent of the supplied vertical-groove references",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
