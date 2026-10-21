"""Create publication-style before/after figures for the current dual-deck study.

The Python figure is a coherent 2-D scalar Fourier-optics calculation.  The
Zemax figures are drawn only from the native OpticStudio POP grids exported by
``zemax/build_and_run_dual_deck_equivalent.py``; no synthetic replacement is
used for those panels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
PY_OUT = ROOT / "results" / "dual_deck_engineering"
ZMX_OUT = ROOT / "results" / "dual_deck_zemax_equivalent_fullwindow"

WAVELENGTH_MM = 0.000355
F_OBJ_MM = 30.0
NA = 0.60
MY = 2.816901408450704
OBJECT_PITCH_MM = 0.020
CORE_MM = 1.0
DEPTH_MM = 0.5
TOOTH_PITCH_MM = 0.75


def ft2(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.fft2(fft.ifftshift(a), norm="ortho", workers=4))


def ift2(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.ifft2(fft.ifftshift(a), norm="ortho", workers=4))


def triangle_fraction(x_mm: np.ndarray) -> np.ndarray:
    phase = x_mm / TOOTH_PITCH_MM
    tri = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    return 0.5 * (tri + 1.0)


def rod_stack_mask(x_hw: np.ndarray, y_hw: np.ndarray, orders: range) -> np.ndarray:
    """Binary serrated rods: 1 mm core plus 0.5 mm teeth on each side."""
    spacing = MY * F_OBJ_MM * WAVELENGTH_MM / OBJECT_PITCH_MM
    half_width = CORE_MM / 2 + DEPTH_MM * triangle_fraction(x_hw)
    mask = np.ones((y_hw.size, x_hw.size), dtype=np.float32)
    for order in orders:
        blocked = np.abs(y_hw[:, None] - order * spacing) <= half_width[None, :]
        mask[blocked] = 0.0
    return mask


def make_python_figure() -> dict:
    n = 2048
    fov_mm = 0.5
    dx = fov_mm / n
    x = (np.arange(n) - n // 2) * dx
    y = x.copy()
    xx, yy = np.meshgrid(x, y)

    wrapped = np.remainder(yy + OBJECT_PITCH_MM / 2, OBJECT_PITCH_MM) - OBJECT_PITCH_MM / 2
    background = (np.abs(wrapped) <= 0.0035).astype(np.float32)
    # Suppress artificial FFT wrap-around without changing the central 0.42 mm.
    edge = np.maximum(np.abs(xx), np.abs(yy))
    taper = np.ones_like(background)
    transition = (edge > 0.21) & (edge < 0.25)
    taper[edge >= 0.25] = 0.0
    taper[transition] = 0.5 + 0.5 * np.cos(np.pi * (edge[transition] - 0.21) / 0.04)
    background *= taper

    # Three non-periodic bright defects in nominally dark gaps.  The quoted
    # 0.5 um size is intensity FWHM.  Positions are deliberately asymmetric.
    sigma_amp_mm = 0.0005 / (2 * np.sqrt(np.log(2)))
    defects = np.zeros_like(background)
    positions_mm = [(-0.115, 0.010), (0.036, -0.050), (0.142, 0.070)]
    amplitudes = [0.8, 1.0, 0.7]
    for (xc, yc), amp in zip(positions_mm, amplitudes):
        defects += amp * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma_amp_mm**2))
    actual = background + defects

    fx = fft.fftshift(fft.fftfreq(n, d=dx))
    fy = fx.copy()
    fxx, fyy = np.meshgrid(fx, fy)
    pupil = ((WAVELENGTH_MM * fxx) ** 2 + (WAVELENGTH_MM * fyy) ** 2 <= NA**2).astype(np.float32)
    x_hw = F_OBJ_MM * WAVELENGTH_MM * fx
    y_hw = MY * F_OBJ_MM * WAVELENGTH_MM * fy
    mask7 = rod_stack_mask(x_hw, y_hw, range(-3, 4))

    h0 = pupil
    h7 = pupil * mask7
    u0 = ift2(ft2(actual) * h0)
    ub0 = ift2(ft2(background) * h0)
    u7 = ift2(ft2(actual) * h7)
    ub7 = ift2(ft2(background) * h7)
    ud7 = ift2(ft2(defects) * h7)
    i0 = np.abs(u0) ** 2
    ib0 = np.abs(ub0) ** 2
    i7 = np.abs(u7) ** 2
    ib7 = np.abs(ub7) ** 2
    id7 = np.abs(ud7) ** 2
    ref = float(i0.max())
    signed = (i7 - ib7) / ref
    extent_um = [x[0] * 1000, x[-1] * 1000, y[0] * 1000, y[-1] * 1000]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.3), constrained_layout=True)
    panels = [
        (actual, "Object amplitude", "gray", 0.0, 1.05),
        (i0 / ref, "Unfiltered image", "magma", 0.0, 1.0),
        (i7 / ref, "Filtered image: 7 serrated rods", "magma", 0.0, 1.0),
        (ib7 / ref, "Filtered periodic background only", "magma", 0.0, 1.0),
        (id7 / ref, "Isolated defect response", "magma", 0.0, 1.0),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.flat[:5], panels):
        im = ax.imshow(data, origin="lower", extent=extent_um, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X: TDI scan / rod axis (um)")
        ax.set_ylabel("Y: camera long edge (um)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Amplitude" if title == "Object amplitude" else "I / unfiltered peak")
    lim = max(float(np.max(np.abs(signed))), 1e-12)
    ax = axes[1, 2]
    im = ax.imshow(signed, origin="lower", extent=extent_um, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title("Actual - periodic reference (signed)")
    ax.set_xlabel("X: TDI scan / rod axis (um)")
    ax.set_ylabel("Y: camera long edge (um)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Delta I / unfiltered peak")
    fig.suptitle(
        "Python 2-D coherent Fourier optics | 355 nm, f=30 mm, NA=0.6, pupil=36 mm\n"
        "20 um bars + three non-periodic 0.5 um-FWHM targets; common absolute intensity reference",
        fontsize=14,
    )
    path = PY_OUT / "python_2d_before_after.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)

    # Magnified ROIs make the sub-micron targets auditable without changing the
    # intensity definition.  All before/after panels retain the same I/I_ref
    # limits; the signed column uses one common symmetric Delta-I/I_ref limit.
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 11.3), constrained_layout=True)
    signed_global = (i7 - ib7) / ref
    dlim = max(float(np.max(np.abs(signed_global))), 1e-12)
    roi_half_mm = 0.006
    for row, ((xc, yc), amp) in enumerate(zip(positions_mm, amplitudes)):
        ix = (x >= xc - roi_half_mm) & (x <= xc + roi_half_mm)
        iy = (y >= yc - roi_half_mm) & (y <= yc + roi_half_mm)
        ext = [(xc - roi_half_mm) * 1000, (xc + roi_half_mm) * 1000,
               (yc - roi_half_mm) * 1000, (yc + roi_half_mm) * 1000]
        for col, (data, title) in enumerate([
            (i0 / ref, "Unfiltered"), (i7 / ref, "Filtered, 7 rods")
        ]):
            im = axes[row, col].imshow(data[np.ix_(iy, ix)], origin="lower", extent=ext,
                                       cmap="magma", vmin=0, vmax=1)
            axes[row, col].set_title(f"Target {row + 1} | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, label="I / unfiltered peak")
        im = axes[row, 2].imshow(signed_global[np.ix_(iy, ix)], origin="lower", extent=ext,
                                 cmap="coolwarm", vmin=-dlim, vmax=dlim)
        axes[row, 2].set_title(f"Target {row + 1} | actual - reference")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, label="Delta I / unfiltered peak")
        for col in range(3):
            axes[row, col].set_xlabel("X (um), TDI scan")
            axes[row, col].set_ylabel("Y (um), camera long edge")
    fig.suptitle("Python defect ROIs | identical intensity units and before/after scale", fontsize=14)
    zoom_path = PY_OUT / "python_2d_defect_zoom_before_after.png"
    fig.savefig(zoom_path, dpi=200)
    fig.savefig(zoom_path.with_suffix(".svg"))
    plt.close(fig)

    e_bg0 = float(ib0.sum())
    e_bg7 = float(ib7.sum())
    e_d0 = float(np.sum(np.abs(ift2(ft2(defects) * h0)) ** 2))
    e_d7 = float(id7.sum())
    return {
        "background_suppression_db": float(10 * np.log10(e_bg0 / e_bg7)),
        "isolated_defect_energy_retention": float(e_d7 / e_d0),
        "common_reference_peak": ref,
        "active_orders": list(range(-3, 4)),
        "image": str(path),
        "defect_zoom_image": str(zoom_path),
    }


def load_grid(npz, key: str):
    return npz[f"{key}_x"], npz[f"{key}_y"], npz[f"{key}_irradiance"]


def make_zemax_figures() -> dict:
    npz = np.load(ZMX_OUT / "native_pop_grids.npz")
    modes = [("open", "No rods"), ("balanced7", "7 rods (-3...+3)")]
    grids = {}
    for mode, _ in modes:
        grids[(mode, "regular")] = load_grid(npz, f"regular_{mode}_s6")
        grids[(mode, "defect")] = load_grid(npz, f"defect_{mode}_s6")
    ref = max(float(grids[("open", "defect")][2].max()), float(grids[("open", "regular")][2].max()))

    # Figure 1: physical irradiance in the same W/mm^2 range for every panel.
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.5), constrained_layout=True)
    for row, (mode, label) in enumerate(modes):
        xr, yr, regular = grids[(mode, "regular")]
        xd, yd, defect = grids[(mode, "defect")]
        extent = [xr[0], xr[-1], yr[0], yr[-1]]
        diff = defect - regular
        for col, (data, title) in enumerate([(regular, "Periodic reference"), (defect, "With local shifted segment")]):
            im = axes[row, col].imshow(data, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=ref)
            axes[row, col].set_title(f"{label} | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, label="Irradiance (W/mm^2)")
        dlim = max(float(np.max(np.abs(diff))), 1e-15)
        im = axes[row, 2].imshow(diff, origin="lower", extent=extent, cmap="coolwarm", vmin=-dlim, vmax=dlim)
        axes[row, 2].set_title(f"{label} | defect - reference")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, label="Delta irradiance (W/mm^2)")
        for col in range(3):
            axes[row, col].set_xlabel("Image X (mm)")
            axes[row, col].set_ylabel("Image Y (mm)")
    fig.suptitle(
        "Native Zemax OpticStudio POP: current dual-deck equivalent pupil\n"
        "Intensity panels share one W/mm^2 scale; signed-difference panels use row-local symmetric W/mm^2 bounds",
        fontsize=14,
    )
    abs_path = ZMX_OUT / "zemax_before_after_absolute.png"
    fig.savefig(abs_path, dpi=180)
    plt.close(fig)

    # Figure 2: common dB reference exposes morphology after strong attenuation.
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.5), constrained_layout=True)
    for row, (mode, label) in enumerate(modes):
        xr, yr, regular = grids[(mode, "regular")]
        xd, yd, defect = grids[(mode, "defect")]
        extent = [xr[0], xr[-1], yr[0], yr[-1]]
        diff = defect - regular
        for col, (data, title) in enumerate([(regular, "Periodic reference"), (defect, "With local shifted segment")]):
            db = 10 * np.log10(np.maximum(data / ref, 1e-8))
            im = axes[row, col].imshow(db, origin="lower", extent=extent, cmap="magma", vmin=-60, vmax=0)
            axes[row, col].set_title(f"{label} | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, label="dB re unfiltered peak")
        signed_db = np.sign(diff) * 10 * np.log10(1 + np.abs(diff) / max(float(np.max(np.abs(diff))), 1e-15) * 999)
        im = axes[row, 2].imshow(signed_db, origin="lower", extent=extent, cmap="coolwarm", vmin=-30, vmax=30)
        axes[row, 2].set_title(f"{label} | signed defect contrast")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, label="Signed display dB (row-local)")
        for col in range(3):
            axes[row, col].set_xlabel("Image X (mm)")
            axes[row, col].set_ylabel("Image Y (mm)")
    fig.suptitle(
        "Native Zemax POP: common dB display of the same absolute irradiance grids\n"
        "Intensity panels use one unfiltered reference; signed-contrast panels use row-local display bounds",
        fontsize=14,
    )
    db_path = ZMX_OUT / "zemax_before_after_common_db.png"
    fig.savefig(db_path, dpi=180)
    plt.close(fig)

    summary = json.loads((ZMX_OUT / "summary.json").read_text(encoding="utf-8"))
    return {
        "absolute_image": str(abs_path),
        "common_db_image": str(db_path),
        "reference_peak_W_per_mm2": ref,
        "metrics": summary["metrics"],
        "scope": summary["scope"],
    }


def main() -> None:
    PY_OUT.mkdir(parents=True, exist_ok=True)
    ZMX_OUT.mkdir(parents=True, exist_ok=True)
    result = {"python_2d": make_python_figure(), "zemax_native_pop": make_zemax_figures()}
    path = ROOT / "results" / "before_after_comparison_summary.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
