"""Engineering validation for a two-deck, independently movable serrated-rod filter.

The model is intentionally traceable rather than proprietary:

* Abbe-sine Fourier coordinates are used.
* The object-frequency separation is expanded only in Fourier-plane Y.
* Nine rods block relative Y orders -4..+4; even orders are on deck A and
  odd orders are on deck B.
* Every rod has a 1.0 mm opaque core and a 0.5 mm single-side linear
  transition, i.e. a 2.0 mm projected envelope.
* A 1-D coherent scalar calculation validates the direction in which the rods
  act. A symmetric free-space propagation model checks a compact 4 mm
  near-Fourier-plane deck separation.

This is an ideal-system engineering model. It does not replace Zemax POP with
the final relay prescription, measured rod contours, polarization or scatter.
"""
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dual_deck_engineering"


@dataclass(frozen=True)
class Design:
    wavelength_mm: float = 0.000355
    objective_focal_length_mm: float = 30.0
    numerical_aperture: float = 0.60
    object_pitch_mm: float = 0.020
    fourier_y_magnification: float = 2.816901408450704
    fourier_x_magnification: float = 1.0
    mechanical_aperture_mm: float = 50.0
    core_width_mm: float = 1.0
    tooth_depth_each_side_mm: float = 0.5
    tooth_pitch_mm: float = 0.75
    rod_count: int = 9
    deck_gap_mm: float = 4.0
    image_magnification: float = 20.0
    tdi_row_pitch_um: float = 5.0
    tdi_active_rows: int = 25
    tdi_order_fraction: float = 0.9999
    illumination_fwhm_image_um: float = 125.0
    engineering_margin_um: float = 10.0
    field_mm: float = 0.5
    numerical_window_mm: float = 2.048
    numerical_samples: int = 16384

    @property
    def true_pupil_diameter_mm(self) -> float:
        return 2 * self.objective_focal_length_mm * self.numerical_aperture

    @property
    def alpha_y_mm2(self) -> float:
        return self.fourier_y_magnification * self.objective_focal_length_mm * self.wavelength_mm

    @property
    def raw_order_spacing_mm(self) -> float:
        return self.objective_focal_length_mm * self.wavelength_mm / self.object_pitch_mm

    @property
    def expanded_order_spacing_mm(self) -> float:
        return self.fourier_y_magnification * self.raw_order_spacing_mm

    @property
    def rod_envelope_mm(self) -> float:
        return self.core_width_mm + 2 * self.tooth_depth_each_side_mm

    @property
    def expanded_pupil_y_mm(self) -> float:
        return self.true_pupil_diameter_mm * self.fourier_y_magnification

    @property
    def expanded_pupil_x_mm(self) -> float:
        return self.true_pupil_diameter_mm * self.fourier_x_magnification


def ft(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.fft(fft.ifftshift(a), norm="ortho", workers=4))


def ift(a: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.ifft(fft.ifftshift(a), norm="ortho", workers=4))


def propagate_u(field: np.ndarray, du_mm: float, z_mm: float, wavelength_mm: float) -> np.ndarray:
    """Paraxial angular-spectrum propagation over physical mask coordinate u."""
    fu = fft.fftshift(fft.fftfreq(field.size, d=du_mm))
    h = np.exp(-1j * np.pi * wavelength_mm * z_mm * fu**2)
    return ift(ft(field) * h)


def rod_amplitude(u_mm: np.ndarray, centres_mm: np.ndarray, d: Design, hard: bool = False) -> np.ndarray:
    result = np.ones_like(u_mm, dtype=float)
    for centre in centres_mm:
        distance = np.abs(u_mm - centre)
        if hard:
            one = np.where(distance <= d.core_width_mm / 2, 0.0, 1.0)
        else:
            one = np.clip(
                (distance - d.core_width_mm / 2) / d.tooth_depth_each_side_mm,
                0.0,
                1.0,
            )
        result *= one
    return result


def raised_cosine_field(y_mm: np.ndarray, half_field_mm: float, taper_mm: float) -> np.ndarray:
    envelope = np.ones_like(y_mm)
    edge_start = half_field_mm - taper_mm
    outside = np.abs(y_mm) >= half_field_mm
    transition = (np.abs(y_mm) > edge_start) & ~outside
    envelope[outside] = 0.0
    envelope[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (np.abs(y_mm[transition]) - edge_start) / taper_mm
    )
    return envelope


def make_fields(d: Design, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    envelope = raised_cosine_field(y_mm, d.field_mm / 2, 0.030)
    wrapped = np.remainder(y_mm + d.object_pitch_mm / 2, d.object_pitch_mm) - d.object_pitch_mm / 2
    background = (np.abs(wrapped) < 0.35 * d.object_pitch_mm / 2).astype(float) * envelope
    # Field amplitude chosen so the intensity FWHM is 0.5 um.
    defect_fwhm_mm = 0.0005
    defect = np.exp(-2 * np.log(2) * (y_mm / defect_fwhm_mm) ** 2)
    return background, defect


def apply_stack(
    spectrum: np.ndarray,
    pupil: np.ndarray,
    u_mm: np.ndarray,
    d: Design,
    gap_mm: float,
    hard: bool = False,
    selected_orders: tuple[int, ...] | None = None,
) -> np.ndarray:
    orders = np.array(
        selected_orders
        if selected_orders is not None
        else tuple(range(-(d.rod_count // 2), d.rod_count // 2 + 1))
    )
    centres = orders * d.expanded_order_spacing_mm
    # Alternate the sorted active positions between decks. This guarantees that
    # neighbouring projected rods are never on the same physical deck.
    deck_a = centres[np.arange(len(centres)) % 2 == 0]
    deck_b = centres[np.arange(len(centres)) % 2 != 0]
    mask_a = rod_amplitude(u_mm, deck_a, d, hard=hard)
    mask_b = rod_amplitude(u_mm, deck_b, d, hard=hard)
    e = spectrum * pupil
    if gap_mm == 0:
        return e * mask_a * mask_b
    e = propagate_u(e, u_mm[1] - u_mm[0], -gap_mm / 2, d.wavelength_mm)
    e *= mask_a
    e = propagate_u(e, u_mm[1] - u_mm[0], gap_mm, d.wavelength_mm)
    e *= mask_b
    e = propagate_u(e, u_mm[1] - u_mm[0], -gap_mm / 2, d.wavelength_mm)
    return e * pupil


def fwhm(axis_mm: np.ndarray, intensity: np.ndarray) -> float:
    peak_index = int(np.argmax(intensity))
    peak = float(intensity[peak_index])
    if peak <= 0:
        return float("nan")
    threshold = peak / 2
    left_candidates = np.flatnonzero(intensity[:peak_index] <= threshold)
    right_candidates = np.flatnonzero(intensity[peak_index:] <= threshold)
    if not len(left_candidates) or not len(right_candidates):
        return float("nan")
    li = int(left_candidates[-1])
    ri = peak_index + int(right_candidates[0])
    left = np.interp(threshold, [intensity[li], intensity[li + 1]], [axis_mm[li], axis_mm[li + 1]])
    right = np.interp(
        threshold,
        [intensity[ri], intensity[ri - 1]],
        [axis_mm[ri], axis_mm[ri - 1]],
    )
    return float((right - left) * 1000)


def profile_metrics(y_mm: np.ndarray, baseline: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
    i0 = np.abs(baseline) ** 2
    i1 = np.abs(filtered) ** 2
    e0 = float(i0.sum())
    e1 = float(i1.sum())
    guard = 0.0015
    p0 = float(i0.max())
    p1 = float(i1.max())
    h0 = float(i0[np.abs(y_mm) > guard].sum() / max(e0, 1e-30))
    h1 = float(i1[np.abs(y_mm) > guard].sum() / max(e1, 1e-30))
    centre = int(np.argmax(i1))
    h_center = abs(filtered[centre])
    cross = {}
    for separation_um in [2.0, 3.0, 5.0, 10.0]:
        r = float(
            np.interp(separation_um / 1000, y_mm, np.abs(filtered)) / max(h_center, 1e-30)
        )
        cross[f"crosstalk_worst_phase_{separation_um:g}um"] = 2 * r + r**2
    return {
        "energy_retention": e1 / max(e0, 1e-30),
        "peak_retention": p1 / max(p0, 1e-30),
        "fwhm_um": fwhm(y_mm, i1),
        "baseline_fwhm_um": fwhm(y_mm, i0),
        "halo_fraction_outside_1p5um": h1,
        "baseline_halo_fraction_outside_1p5um": h0,
        **cross,
    }


def gaussian_containment_width(fwhm_um: float, fraction: float) -> float:
    from scipy.special import erfinv

    return fwhm_um / np.sqrt(np.log(2)) * erfinv(fraction)


def plot_layout(d: Design, out: Path) -> None:
    orders = np.arange(-4, 5)
    centres = orders * d.expanded_order_spacing_mm
    deck = np.where(orders % 2 == 0, "A", "B")
    colors = {"A": "#2F5AA8", "B": "#D48A27"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True)
    ax = axes[0]
    for order, centre, layer in zip(orders, centres, deck):
        ax.add_patch(
            plt.Rectangle(
                (centre - d.rod_envelope_mm / 2, 0 if layer == "A" else 1),
                d.rod_envelope_mm,
                0.62,
                facecolor=colors[layer],
                edgecolor="#222222",
                alpha=0.45,
            )
        )
        ax.plot(centre, 0.31 if layer == "A" else 1.31, "o", color=colors[layer])
        ax.text(centre, 0.31 if layer == "A" else 1.31, f"  {order:+d}", va="center", fontsize=9)
    ax.set_xlim(-8.2, 8.2)
    ax.set_ylim(-0.2, 1.85)
    ax.set_yticks([0.31, 1.31], ["Deck A: even orders", "Deck B: odd orders"])
    ax.set_xlabel("Expanded local Fourier-plane Y coordinate (mm)")
    ax.set_title("Nine independently movable rods: projected 1.5 mm order pitch, 3.0 mm same-deck pitch")
    ax.grid(axis="x", color="#D5D8DC", lw=0.7)

    ax = axes[1]
    ax.axhline(0, color="#222222", lw=1)
    ax.axhline(d.deck_gap_mm, color="#222222", lw=1)
    for order, centre, layer in zip(orders, centres, deck):
        z = 0 if layer == "A" else d.deck_gap_mm
        circle = plt.Circle((centre, z), d.rod_envelope_mm / 2, facecolor=colors[layer], alpha=0.35, ec="#222222")
        ax.add_patch(circle)
    nearest = np.hypot(d.expanded_order_spacing_mm, d.deck_gap_mm) - d.rod_envelope_mm
    ax.annotate(
        f"nearest cross-deck envelope clearance\n(conservative circular envelope) = {nearest:.2f} mm",
        xy=(0.75, 2.0),
        xytext=(3.0, 2.0),
        arrowprops=dict(arrowstyle="->", color="#444444"),
        fontsize=9,
    )
    ax.set_xlim(-8.2, 8.2)
    ax.set_ylim(-1.35, d.deck_gap_mm + 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Fourier-plane Y coordinate (mm)")
    ax.set_ylabel("Optical-axis Z (mm)")
    ax.set_title("Nominal Y-Z two-deck structure and key dimensions")
    ax.grid(color="#D5D8DC", lw=0.7)
    fig.savefig(out / "dual_deck_layout.png", dpi=190)
    fig.savefig(out / "dual_deck_layout.svg")
    plt.close(fig)


def plot_structure_schematic(d: Design, out: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)

    # Front view: directions and the deliberately anisotropic relayed pupil.
    ax = axes[0]
    ax.add_patch(Ellipse((0, 0), 3.6, 10.14, fc="#EEF3FA", ec="#2F5AA8", lw=2))
    for k in range(-3, 4):
        yv = k * 0.55
        ax.plot([-1.65, 1.65], [yv, yv], color="#D48A27" if k % 2 else "#2F5AA8", lw=5, alpha=0.75)
    ax.annotate("X：档杆长轴、锯齿周期、TDI扫描", xy=(2.2, -4.8), xytext=(-2.2, -4.8), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("Y：相机长边、照明线、档杆排布", xy=(-2.25, 4.6), xytext=(-2.25, -4.1), arrowprops=dict(arrowstyle="->", lw=1.8), rotation=90, va="center")
    ax.text(0, 4.15, "放大后光瞳\nX=36 mm，Y=101.4 mm", ha="center", va="top", fontsize=10)
    ax.annotate("相邻级次 1.5 mm", xy=(1.75, 0), xytext=(2.25, 0.55), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlim(-3.0, 3.4)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("傅里叶滤波面正视示意")

    # Y-Z section showing alternating decks and nominal dimensions only.
    ax = axes[1]
    for yv in [-6, -3, 0, 3, 6]:
        ax.add_patch(plt.Circle((yv, 0), 1.0, fc="#2F5AA8", ec="#222222", alpha=0.45))
    for yv in [-4.5, -1.5, 1.5, 4.5]:
        ax.add_patch(plt.Circle((yv, 4), 1.0, fc="#D48A27", ec="#222222", alpha=0.45))
    ax.annotate("同层中心距 3.0 mm", xy=(-6, -1.3), xytext=(-3, -1.3), ha="center", arrowprops=dict(arrowstyle="<->"))
    ax.annotate("层间距 4.0 mm", xy=(7.0, 0), xytext=(7.0, 4), va="center", ha="left", arrowprops=dict(arrowstyle="<->"))
    ax.annotate("跨层投影中心距 1.5 mm", xy=(0, 0.8), xytext=(1.5, 2.0), arrowprops=dict(arrowstyle="<->"), ha="center", fontsize=9)
    ax.text(-7.7, 0, "A层\n偶数级", ha="right", va="center", color="#2F5AA8")
    ax.text(-7.7, 4, "B层\n奇数级", ha="right", va="center", color="#A76012")
    ax.set_xlim(-9.2, 9.0)
    ax.set_ylim(-2.0, 5.5)
    ax.set_aspect("equal")
    ax.set_xlabel("傅里叶面 Y / mm")
    ax.set_ylabel("光轴 Z / mm")
    ax.grid(color="#D5D8DC", lw=0.6)
    ax.set_title("双层交错名义结构")

    # Rod top view and tooth dimensions.
    ax = axes[2]
    pitch = d.tooth_pitch_mm
    core_half = d.core_width_mm / 2
    depth = d.tooth_depth_each_side_mm
    xs = np.arange(0, 6 * pitch + 1e-12, pitch / 2)
    top = np.where(np.arange(xs.size) % 2 == 0, core_half, core_half + depth)
    bottom = -top
    polygon = np.column_stack([np.r_[xs, xs[::-1]], np.r_[top, bottom[::-1]]])
    ax.add_patch(Polygon(polygon, closed=True, fc="#7EA1D4", ec="#222222", lw=1.4))
    ax.axhline(core_half, color="#555555", ls="--", lw=0.8)
    ax.axhline(-core_half, color="#555555", ls="--", lw=0.8)
    ax.annotate("核心宽度 1.0 mm", xy=(0.25, -core_half), xytext=(0.25, core_half), arrowprops=dict(arrowstyle="<->"), va="center", ha="right", fontsize=9)
    ax.annotate("单侧齿深 0.5 mm", xy=(2.3, core_half), xytext=(2.3, core_half + depth), arrowprops=dict(arrowstyle="<->"), va="center", ha="left", fontsize=9)
    ax.annotate("齿距 0.75 mm", xy=(3.0, -1.25), xytext=(3.75, -1.25), arrowprops=dict(arrowstyle="<->"), ha="center", fontsize=9)
    ax.text(2.25, 1.22, "最大投影包络 2.0 mm", ha="center", fontsize=10)
    ax.annotate("X / 档杆长轴", xy=(4.5, -1.55), xytext=(0, -1.55), arrowprops=dict(arrowstyle="->"), va="center")
    ax.set_xlim(-0.35, 4.95)
    ax.set_ylim(-1.85, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("单根双侧三角锯齿杆")

    fig.suptitle("双层交错、逐杆独立移动的傅里叶滤杆结构示意（非加工工程图）", fontsize=15)
    fig.savefig(out / "dual_deck_structure_schematic.png", dpi=200)
    fig.savefig(out / "dual_deck_structure_schematic.svg")
    plt.close(fig)


def plot_validation(
    d: Design,
    y: np.ndarray,
    u: np.ndarray,
    bg_spec: np.ndarray,
    pupil: np.ndarray,
    images: dict[str, np.ndarray],
    defect_images: dict[str, np.ndarray],
    rows: list[dict[str, float]],
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    valid = pupil > 0
    ax.semilogy(u[valid], np.maximum(np.abs(bg_spec[valid]) ** 2, 1e-12) / np.max(np.abs(bg_spec[valid]) ** 2), color="#2F5AA8")
    for k in range(-4, 5):
        ax.axvline(k * d.expanded_order_spacing_mm, color="#D48A27", lw=0.7, alpha=0.7)
    ax.set_xlim(-8, 8)
    ax.set_ylim(1e-9, 2)
    ax.set_xlabel("Expanded Fourier-plane Y (mm)")
    ax.set_ylabel("Relative spectral intensity (DC = 1)")
    ax.set_title("Periodic-background spectrum and nine rod centres")
    ax.grid(color="#D5D8DC", lw=0.6)

    ax = axes[0, 1]
    view = np.abs(y) <= 0.08
    common_background_peak = max(float(np.abs(images["unfiltered"]).max() ** 2), 1e-30)
    for name, field in images.items():
        intensity = np.abs(field) ** 2
        intensity /= common_background_peak
        ax.plot(y[view] * 1000, intensity[view], label=name)
    ax.set_xlabel("Object/image-equivalent Y (um)")
    ax.set_ylabel("Intensity / unfiltered-background peak")
    ax.set_title("Periodic background after coherent filtering; one common intensity reference")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#D5D8DC", lw=0.6)

    ax = axes[1, 0]
    view = np.abs(y) <= 0.020
    baseline_peak = max(float(np.abs(defect_images["unfiltered"]).max() ** 2), 1e-30)
    for name, field in defect_images.items():
        ax.semilogy(y[view] * 1000, np.maximum(np.abs(field[view]) ** 2 / baseline_peak, 1e-8), label=name)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("Distance from 0.5 um defect (um)")
    ax.set_ylabel("Intensity / unfiltered peak")
    ax.set_title("Defect ringing: linear teeth versus hard edges and 4 mm deck gap")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#D5D8DC", lw=0.6)

    ax = axes[1, 1]
    labels = [r["case"] for r in rows]
    suppression = [r["background_suppression_db"] for r in rows]
    retention = [100 * r["energy_retention"] for r in rows]
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, suppression, width, color="#2F5AA8", label="background suppression (dB)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, retention, width, color="#D48A27", label="defect energy retention (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Background suppression (dB)")
    ax2.set_ylabel("0.5 um defect energy retention (%)")
    ax.set_title("Engineering trade-off metrics; common absolute-intensity normalization")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#D5D8DC", lw=0.6)
    fig.savefig(out / "dual_deck_simulation_validation.png", dpi=190)
    fig.savefig(out / "dual_deck_simulation_validation.svg")
    plt.close(fig)


def plot_tuning(d: Design, out: Path) -> None:
    periods = np.linspace(12, 35, 240)
    spacing = d.fourier_y_magnification * d.objective_focal_length_mm * d.wavelength_mm / (periods / 1000)
    same_deck_clearance = 2 * spacing - d.rod_envelope_mm
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(periods, same_deck_clearance, color="#2F5AA8", lw=2)
    ax.axhline(0.1, color="#D48A27", ls="--", label="minimum desired clearance 0.1 mm")
    ax.axvline(20, color="#222222", ls=":", label="20 um nominal object pitch")
    ax.set_xlabel("Object period (um)")
    ax.set_ylabel("Nominal same-deck clearance (mm)")
    ax.set_title("Independent retuning range is limited by same-deck collision")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#D5D8DC", lw=0.6)
    fig.savefig(out / "dual_deck_period_tuning.png", dpi=190)
    fig.savefig(out / "dual_deck_period_tuning.svg")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def scan_order_subsets(
    d: Design,
    y: np.ndarray,
    u: np.ndarray,
    pupil: np.ndarray,
    bg_spec: np.ndarray,
    defect_spec: np.ndarray,
    base_bg: np.ndarray,
    base_defect: np.ndarray,
) -> list[dict]:
    rows = []
    e_bg0 = float(np.sum(np.abs(base_bg) ** 2))
    for pair_count in range(5):
        for positive in itertools.combinations(range(1, 9), pair_count):
            orders = tuple(sorted((-np.array(positive)).tolist() + [0] + list(positive)))
            bg = ift(apply_stack(bg_spec, pupil, u, d, 4.0, hard=False, selected_orders=orders))
            defect = ift(apply_stack(defect_spec, pupil, u, d, 4.0, hard=False, selected_orders=orders))
            metric = profile_metrics(y, base_defect, defect)
            ratio = float(np.sum(np.abs(bg) ** 2) / max(e_bg0, 1e-30))
            worst_c = max(v for k, v in metric.items() if k.startswith("crosstalk_worst_phase"))
            row = {
                "orders": ",".join(str(v) for v in orders),
                "active_rods": len(orders),
                "background_suppression_db": -10 * np.log10(max(ratio, 1e-30)),
                "energy_retention": metric["energy_retention"],
                "halo_fraction_outside_1p5um": metric["halo_fraction_outside_1p5um"],
                "worst_crosstalk_2_to_10um": worst_c,
                "fwhm_ratio": metric["fwhm_um"] / max(metric["baseline_fwhm_um"], 1e-30),
            }
            row["balanced_gate"] = bool(
                row["background_suppression_db"] >= 10
                and row["energy_retention"] >= 0.60
                and row["halo_fraction_outside_1p5um"] <= 0.12
                and row["worst_crosstalk_2_to_10um"] <= 0.30
                and row["fwhm_ratio"] <= 1.10
            )
            rows.append(row)
    return rows


def main() -> None:
    d = Design()
    OUT.mkdir(parents=True, exist_ok=True)
    n = d.numerical_samples
    dy = d.numerical_window_mm / n
    y = (np.arange(n) - n // 2) * dy
    nu = fft.fftshift(fft.fftfreq(n, d=dy))
    u = d.alpha_y_mm2 * nu
    pupil = (np.abs(nu) <= d.numerical_aperture / d.wavelength_mm).astype(float)
    background, defect = make_fields(d, y)
    bg_spec = ft(background)
    defect_spec = ft(defect)
    base_bg = ift(bg_spec * pupil)
    base_defect = ift(defect_spec * pupil)

    search_rows = scan_order_subsets(d, y, u, pupil, bg_spec, defect_spec, base_bg, base_defect)
    balanced_pool = [r for r in search_rows if r["balanced_gate"]]
    balanced = max(balanced_pool, key=lambda r: r["background_suppression_db"]) if balanced_pool else max(
        search_rows,
        key=lambda r: r["background_suppression_db"]
        + 10 * np.log10(max(r["energy_retention"], 1e-30))
        - 10 * np.log10(1 + r["halo_fraction_outside_1p5um"] / 0.12)
        - 10 * np.log10(1 + r["worst_crosstalk_2_to_10um"] / 0.30),
    )
    balanced_orders = tuple(int(v) for v in balanced["orders"].split(","))

    cases = [
        ("hard, 9 rods, coplanar", 0.0, True, tuple(range(-4, 5))),
        ("serrated, 9 rods, coplanar", 0.0, False, tuple(range(-4, 5))),
        ("serrated, balanced, 2 mm gap", 2.0, False, balanced_orders),
        ("serrated, balanced, 4 mm gap", 4.0, False, balanced_orders),
        ("serrated, balanced, 6 mm gap", 6.0, False, balanced_orders),
        ("serrated, 9 rods, 4 mm gap", 4.0, False, tuple(range(-4, 5))),
    ]
    rows: list[dict[str, float]] = []
    bg_images = {"unfiltered": base_bg}
    defect_images = {"unfiltered": base_defect}
    e_bg0 = float(np.sum(np.abs(base_bg) ** 2))
    for name, gap, hard, orders in cases:
        bg_filtered = ift(apply_stack(bg_spec, pupil, u, d, gap, hard=hard, selected_orders=orders))
        defect_filtered = ift(apply_stack(defect_spec, pupil, u, d, gap, hard=hard, selected_orders=orders))
        metrics = profile_metrics(y, base_defect, defect_filtered)
        e_bg = float(np.sum(np.abs(bg_filtered) ** 2))
        metrics.update(
            case=name,
            deck_gap_mm=gap,
            hard_edge=hard,
            active_orders=",".join(str(v) for v in orders),
            active_rods=len(orders),
            background_energy_ratio=e_bg / max(e_bg0, 1e-30),
            background_suppression_db=-10 * np.log10(max(e_bg / max(e_bg0, 1e-30), 1e-30)),
            fwhm_ratio=metrics["fwhm_um"] / max(metrics["baseline_fwhm_um"], 1e-30),
        )
        rows.append(metrics)
        if name in (
            "hard, 9 rods, coplanar",
            "serrated, 9 rods, coplanar",
            "serrated, balanced, 4 mm gap",
            "serrated, 9 rods, 4 mm gap",
        ):
            bg_images[name] = bg_filtered
            defect_images[name] = defect_filtered

    w = gaussian_containment_width(d.illumination_fwhm_image_um, d.tdi_order_fraction)
    tooth_shift_um = (
        d.image_magnification * d.wavelength_mm * d.objective_focal_length_mm / d.tooth_pitch_mm * 1000
    )
    hmax_um = 2 * (tooth_shift_um - d.engineering_margin_um) - w
    max_rows = int(np.floor(np.nextafter(hmax_um / d.tdi_row_pitch_um, -np.inf)))
    nearest_cross_deck_clearance = np.hypot(d.expanded_order_spacing_mm, d.deck_gap_mm) - d.rod_envelope_mm
    period_limit_um = (
        2
        * d.fourier_y_magnification
        * d.objective_focal_length_mm
        * d.wavelength_mm
        / (d.rod_envelope_mm + 0.1)
        * 1000
    )
    summary = {
        "design": asdict(d),
        "derived": {
            "true_pupil_diameter_mm": d.true_pupil_diameter_mm,
            "raw_order_spacing_mm": d.raw_order_spacing_mm,
            "expanded_order_spacing_mm": d.expanded_order_spacing_mm,
            "same_deck_spacing_mm": 2 * d.expanded_order_spacing_mm,
            "nominal_same_deck_clearance_mm": 2 * d.expanded_order_spacing_mm - d.rod_envelope_mm,
            "rod_envelope_mm": d.rod_envelope_mm,
            "expanded_pupil_x_mm": d.expanded_pupil_x_mm,
            "expanded_pupil_y_mm": d.expanded_pupil_y_mm,
            "minimum_new_filter_clear_opening_x_mm": d.expanded_pupil_x_mm + 4,
            "minimum_new_filter_clear_opening_y_mm": d.expanded_pupil_y_mm + 8,
            "nearest_cross_deck_clearance_mm": float(nearest_cross_deck_clearance),
            "object_period_limit_for_0p1mm_same_deck_clearance_um": float(period_limit_um),
            "tooth_first_order_image_shift_um": float(tooth_shift_um),
            "tdi_99p99_order_width_um": float(w),
            "tdi_hmax_with_10um_margin_um": float(hmax_um),
            "tdi_max_integer_rows": max_rows,
            "tdi_25_row_width_um": d.tdi_active_rows * d.tdi_row_pitch_um,
        },
        "simulation_cases": rows,
        "order_subset_search": {
            "candidate_count": len(search_rows),
            "balanced_gate_count": len(balanced_pool),
            "balanced_selected": balanced,
            "gate_definition": "proposed preliminary gate: S>=10 dB, R>=0.60, halo<=0.12, worst coherent crosstalk for 2/3/5/10 um <=0.30, FWHM ratio<=1.10",
        },
        "model_limitations": [
            "1-D coherent scalar model in the rod-normal Y direction",
            "linear zero-order soft-edge surrogate for the binary triangular teeth",
            "paraxial free-space propagation between compact near-pupil decks",
            "no measured contour, polarization, scatter, coating, aberration, or stage vibration",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUT / "simulation_metrics.csv", rows)
    write_csv(OUT / "order_subset_search.csv", search_rows)
    period_rows = []
    for period_um in [12, 15, 20, 25, 27, 30, 35]:
        spacing = d.alpha_y_mm2 / (period_um / 1000)
        period_rows.append(
            {
                "object_period_um": period_um,
                "expanded_adjacent_order_spacing_mm": spacing,
                "same_deck_spacing_mm": 2 * spacing,
                "nominal_same_deck_clearance_mm": 2 * spacing - d.rod_envelope_mm,
                "collision_free_with_0p1mm_target": 2 * spacing - d.rod_envelope_mm >= 0.1,
            }
        )
    write_csv(OUT / "period_retuning_clearance.csv", period_rows)
    plot_layout(d, OUT)
    plot_structure_schematic(d, OUT)
    plot_validation(d, y, u, bg_spec, pupil, bg_images, defect_images, rows, OUT)
    plot_tuning(d, OUT)
    print(json.dumps(summary["derived"], indent=2, ensure_ascii=False))
    for row in rows:
        print(
            f"{row['case']}: S={row['background_suppression_db']:.3f} dB, "
            f"R={row['energy_retention']:.4f}, halo={row['halo_fraction_outside_1p5um']:.4f}, "
            f"FWHM ratio={row['fwhm_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
