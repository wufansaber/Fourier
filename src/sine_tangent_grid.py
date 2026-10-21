"""Air-side sin/tan Fourier-coordinate comparison; not a Zemax lens design.

Lengths are mm, spatial frequencies cycles/mm. The square-aperture grid is
separable: its exact 2-D Fourier transform is the product of two 1-D transforms.
The FFT check uses exact pixel-area integration; no rasterized 2-D FFT needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "sine_tangent"
F = 50.0
WAVELENGTH = 0.000355
PITCH = 0.002
DUTY = 0.3
CELLS = 250
FOV = CELLS * PITCH
BLUE, ORANGE, GRAY = "#185ABD", "#B85A12", "#7A8189"


def map_sines(sx, sy, focal_length=F, projection="sin"):
    """Same air-side angular support, common *radial* tan denominator."""
    sx, sy = np.broadcast_arrays(np.asarray(sx, float), np.asarray(sy, float))
    s2 = sx * sx + sy * sy
    if np.any(s2 >= 1):
        raise ValueError("Only forward propagating directions with s < 1.")
    if projection not in ("sin", "tan"):
        raise ValueError("projection must be sin or tan")
    denominator = np.sqrt(1 - s2) if projection == "tan" else 1.0
    return focal_length * sx / denominator, focal_length * sy / denominator


def na_rows():
    rows = []
    for na in np.arange(1, 10) / 10:
        rs, rt = F * na, F * na / np.sqrt(1 - na**2)
        rows.append(dict(na=float(na), theta_deg=float(np.degrees(np.arcsin(na))),
                         r_sin_mm=float(rs), r_tan_mm=float(rt),
                         delta_um=float((rt-rs)*1000), error_percent=float(100*(rt/rs-1))))
    return rows


def order_rows(na=0.9, pitch=PITCH):
    rows = []
    limit = int(np.floor(na * pitch / WAVELENGTH))
    for m in range(-limit, limit + 1):
        for ell in range(-limit, limit + 1):
            sx, sy = WAVELENGTH * np.array([m, ell]) / pitch
            s = np.hypot(sx, sy)
            if s > na + 1e-12:
                continue
            xs, ys = map_sines(sx, sy)
            xt, yt = map_sines(sx, sy, projection="tan")
            coeff_power = DUTY**4 * (np.sinc(DUTY*m)*np.sinc(DUTY*ell))**2
            rows.append(dict(m=m, ell=ell, na_required=float(s),
                             nu_x_per_mm=m/pitch, nu_y_per_mm=ell/pitch,
                             x_sin_mm=float(xs), y_sin_mm=float(ys),
                             x_tan_mm=float(xt), y_tan_mm=float(yt),
                             delta_r_mm=float(np.hypot(xt-xs, yt-ys)),
                             error_percent=float(100*(1/np.sqrt(1-s*s)-1)),
                             relative_coefficient_power=float(coeff_power/DUTY**4),
                             squared_fourier_series_coefficient=float(coeff_power),
                             radial_stretch=float((1-s*s)**-1.5),
                             azimuthal_stretch=float((1-s*s)**-0.5),
                             area_stretch=float((1-s*s)**-2),
                             density_ratio=float((1-s*s)**2)))
    return rows


def finite_grid_intensity(nu_x, nu_y):
    """Exact scalar finite-array FT intensity normalized by its DC value."""
    result = 1.0
    for nu in (nu_x, nu_y):
        u = PITCH * np.asarray(nu)
        v = u - np.rint(u)
        # Integer-period sign cancels in intensity; sinc removes 0/0 at orders.
        factor = np.sinc(CELLS*v) / np.sinc(v)
        result = result * (np.sinc(DUTY*u)*factor)**2
    return result


def fft_check(rows, pixels_per_period=100):
    """Separable 2-D FFT coefficients; exact integration of constant pixels."""
    count = CELLS * pixels_per_period
    cell = np.zeros(pixels_per_period)
    open_count = int(round(DUTY*pixels_per_period))
    first = (pixels_per_period-open_count)//2
    cell[first:first+open_count] = 1
    obj = np.tile(cell, CELLS)
    fft = np.fft.fft(obj) / count
    dx = PITCH / pixels_per_period

    def coefficient(order):
        # Converts the point-sampled DFT into the FT of the actual square pixels.
        return fft[(order*CELLS) % count] * np.sinc(order/PITCH*dx)

    errors = []
    for row in rows:
        calculated = abs(coefficient(row["m"])*coefficient(row["ell"]))**2
        errors.append(abs(calculated-row["squared_fourier_series_coefficient"]))
    return dict(pixels_per_period=pixels_per_period, dx_um=dx*1000,
                one_dimensional_samples=count,
                equivalent_two_dimensional_shape=[count, count],
                computation="separable outer product, not allocated as 2-D array",
                max_absolute_power_coefficient_error=float(max(errors)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_na(rows):
    s = np.linspace(0, 0.9, 501)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    axs[0].plot(s, F*s, color=BLUE, label=r"$f\sin\theta$")
    axs[0].plot(s, F*s/np.sqrt(1-s*s), "--", color=ORANGE, label=r"$f\tan\theta$")
    axs[0].set(title="孔径边缘半径", ylabel="半径 / mm", ylim=(0, 110))
    axs[0].legend(frameon=False)
    axs[1].plot(s, 100*(1/np.sqrt(1-s*s)-1), color=ORANGE)
    axs[1].scatter([r["na"] for r in rows], [r["error_percent"] for r in rows], color=ORANGE)
    axs[1].set(title="tan 相对 sin 的径向位置差", ylabel="相对差值 / %", ylim=(0, 140))
    for ax in axs:
        ax.set(xlabel="NA（空气，n = 1）", xlim=(0, 0.92))
        ax.grid(alpha=.18)
    fig.suptitle("sin / tan 映射的孔径边缘比较 | f = 50 mm", fontsize=14)
    fig.savefig(OUTPUT / "na_mapping.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def plot_grid(rows):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained")
    ax = axs[0]
    for j in range(-3, 4):
        for k in range(-3, 4):
            ax.add_patch(Rectangle((2*j-.3, 2*k-.3), .6, .6, facecolor=BLUE, edgecolor="none"))
    ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="物面 x / µm", ylabel="物面 y / µm",
           title="方孔周期网格（局部）\np = 2 µm；孔宽 0.6 µm；250 × 250 单元")
    ax.set_aspect("equal")
    values = 10*np.log10([r["relative_coefficient_power"] for r in rows])
    norm = Normalize(-40, 0)
    for ax, projection, title in zip(axs[1:], ("sin", "tan"), ("sin：等间距频谱网格", "tan：径向展开的频谱网格")):
        # Row/column guides are mapped continuously, not incorrectly kept straight.
        for fixed in range(-5, 6):
            sf = fixed*WAVELENGTH/PITCH
            if abs(sf) > .9:
                continue
            sv = np.linspace(-np.sqrt(.9**2-sf**2), np.sqrt(.9**2-sf**2), 401)
            for sx, sy in ((sv, np.full_like(sv, sf)), (np.full_like(sv, sf), sv)):
                x, y = map_sines(sx, sy, projection=projection)
                ax.plot(x, y, color=GRAY, alpha=.23, lw=.65, zorder=0)
        im = ax.scatter([r[f"x_{projection}_mm"] for r in rows],
                        [r[f"y_{projection}_mm"] for r in rows],
                        c=values, norm=norm, cmap="cividis", s=38, edgecolors="#464C53", linewidths=.4)
        radius = F*.9/(np.sqrt(1-.9**2) if projection == "tan" else 1)
        ax.add_patch(Circle((0, 0), radius, fill=False, color=GRAY, ls="--", lw=1.2))
        row = next(r for r in rows if r["m"] == 5 and r["ell"] == 0)
        ax.annotate("(5, 0)", (row[f"x_{projection}_mm"], 0), xytext=(-20, 12),
                    textcoords="offset points", fontsize=9)
        ax.set(title=title, xlim=(-110, 110), ylim=(-110, 110),
               xlabel="物理频谱面 X / mm", ylabel="物理频谱面 Y / mm")
        ax.set_aspect("equal")
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_yticks([-100, -50, 0, 50, 100])
    cb = fig.colorbar(im, ax=list(axs[1:]), shrink=.82, pad=.018)
    cb.set_label("傅里叶系数模平方 / DC（dB）；不是 W/mm²")
    fig.suptitle("同一网格的衍射级次中心 | 355 nm，f = 50 mm，NA = 0.9\n两张频谱图坐标和色标一致；点大小仅作标记，不代表光斑宽度", fontsize=13)
    fig.savefig(OUTPUT / "periodic_grid_spectra.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def plot_orders(rows):
    axial = sorted((r for r in rows if r["ell"] == 0 and r["m"] >= 0), key=lambda r:r["m"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for projection, color, marker in (("sin", BLUE, "o"), ("tan", ORANGE, "s")):
        axs[0].plot([r["m"] for r in axial], [r[f"x_{projection}_mm"] for r in axial],
                    marker=marker, color=color, label=projection)
    axs[0].set(title="x 轴衍射级次的位置", xlabel="衍射级次 m（ℓ = 0）", ylabel="X / mm", ylim=(0, 105))
    axs[0].legend(frameon=False)
    nu = 5/PITCH + np.linspace(-4/FOV, 4/FOV, 8001)
    intensity = finite_grid_intensity(nu, 0)
    s = WAVELENGTH*nu
    xs, _ = map_sines(s, 0)
    xt, _ = map_sines(s, 0, projection="tan")
    center = next(r for r in axial if r["m"] == 5)
    peak = center["relative_coefficient_power"]
    axs[1].plot(xs-center["x_sin_mm"], intensity/peak, color=BLUE, label="sin")
    axs[1].plot(xt-center["x_tan_mm"], intensity/peak, color=ORANGE, ls="--", label="tan")
    axs[1].set(title="(5, 0) 主峰径向剖面（分别归一化）", xlabel="相对各自中心的距离 / mm",
               ylabel="归一化强度（只比较形状）", xlim=(-.85, .85), ylim=(0, 1.06))
    axs[1].legend(frameon=False)
    for ax in axs:
        ax.grid(alpha=.18)
    fig.suptitle("周期网格：峰位与有限阵列峰宽 | p = 2 µm，有限宽度 0.5 mm", fontsize=13)
    fig.savefig(OUTPUT / "order_position_width.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def make_tables(na, rows, counts):
    lines = ["# 自动生成的数值表", "", "## NA 孔径边缘差值", "",
             "| NA | 半角 θ / ° | sin 半径 / mm | tan 半径 / mm | 差值 / µm | 相对差值 / % |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in na:
        lines.append(f'| {r["na"]:.1f} | {r["theta_deg"]:.4f} | {r["r_sin_mm"]:.4f} | {r["r_tan_mm"]:.4f} | {r["delta_um"]:.2f} | {r["error_percent"]:.4f} |')
    lines += ["", "## 2 µm 周期的代表级次", "",
              "| 级次 (m,ℓ) | 最低 NA | sin 坐标 (X,Y) / mm | tan 坐标 (X,Y) / mm | 径向差 / mm | 相对差 / % | 系数模平方 / DC |",
              "|---|---:|---|---|---:|---:|---:|"]
    for m, ell in [(1,0),(1,1),(2,0),(2,2),(3,0),(3,3),(4,0),(4,3),(5,0)]:
        r = next(r for r in rows if (r["m"], r["ell"]) == (m,ell))
        lines.append(f'| ({m},{ell}) | {r["na_required"]:.6f} | ({r["x_sin_mm"]:.4f}, {r["y_sin_mm"]:.4f}) | ({r["x_tan_mm"]:.4f}, {r["y_tan_mm"]:.4f}) | {r["delta_r_mm"]:.4f} | {r["error_percent"]:.4f} | {r["relative_coefficient_power"]:.6f} |')
    lines += ["", "## 可通过的级次中心数量（含 DC）", "", "| NA | 数量 | 最大轴向级次 |", "|---:|---:|---:|"]
    for r in counts:
        lines.append(f'| {r["na"]:.1f} | {r["order_centers_including_dc"]} | {r["largest_axial_order"]} |')
    return "\n".join(lines)+"\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus":False, "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False})
    na, rows = na_rows(), order_rows()
    counts = [dict(na=r["na"], order_centers_including_dc=len(order_rows(r["na"])),
                   largest_axial_order=int(np.floor(r["na"]*PITCH/WAVELENGTH))) for r in na]
    checks = [fft_check(rows, p) for p in (100, 200)]
    edge = next(r for r in rows if (r["m"], r["ell"]) == (5,0))
    widths = dict(sin_center_to_first_zero_mm=F*WAVELENGTH/FOV,
                  tan_radial_center_to_first_zero_local_mm=F*WAVELENGTH/FOV*edge["radial_stretch"],
                  tan_azimuthal_center_to_first_zero_local_mm=F*WAVELENGTH/FOV*edge["azimuthal_stretch"])
    report = dict(parameters=dict(focal_length_mm=F, wavelength_mm=WAVELENGTH,
                  refractive_index=1, pitch_mm=PITCH, aperture_width_mm=PITCH*DUTY,
                  fov_mm=FOV, cells_per_axis=CELLS, maximum_na=.9, pupil_relay_magnification=1),
                  na_edge_rows=na, orders=rows, counts=counts, fft_checks=checks,
                  edge_order_5_0=edge, finite_grid_widths=widths,
                  original_20um_first_order_delta_um=float(F*WAVELENGTH/.02*(1/np.sqrt(1-(WAVELENGTH/.02)**2)-1)*1000),
                  limitations=["Analytic scalar coordinate comparison, not Zemax output.",
                               "Same angular support; different physical pupil radii.",
                               "No real-lens aberrations, vector polarization or efficiency calibration.",
                               "Marker colors are coefficient powers, not irradiance densities."])
    write_csv(OUTPUT/"na_edge_comparison.csv", na)
    write_csv(OUTPUT/"periodic_grid_orders.csv", rows)
    write_csv(OUTPUT/"na_order_counts.csv", counts)
    (OUTPUT/"calculation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT/"数值表.md").write_text(make_tables(na, rows, counts), encoding="utf-8")
    plot_na(na)
    plot_grid(rows)
    plot_orders(rows)
    print(json.dumps(dict(output=str(OUTPUT), order_count=len(rows), fft_checks=checks,
                          counts=counts, edge_order=edge, widths=widths), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
