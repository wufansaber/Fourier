"""Draw the currently confirmed TDI / rod / illumination direction convention."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "direction_geometry"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def arrow(ax, start, end, color, text, text_xy=None, width=2.6, style="-|>"):
    item = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                           linewidth=width, color=color, zorder=6)
    ax.add_patch(item)
    if text_xy is None:
        text_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    ax.text(*text_xy, text, color=color, fontsize=11, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1.5), zorder=8)


def serrated_rod(cx, y0=-3.5, y1=3.5, core=0.34, depth=0.22, pitch=0.72):
    ys = np.arange(y0, y1 + pitch / 2, pitch / 2)
    ys[-1] = min(ys[-1], y1)
    left, right = [], []
    for index, y in enumerate(ys):
        tooth = depth if index % 2 else 0.0
        left.append((cx - core / 2 - tooth, y))
        right.append((cx + core / 2 + tooth, y))
    return Polygon(left + right[::-1], closed=True, facecolor="#4B5563",
                   edgecolor="#991B1B", linewidth=1.2, zorder=4)


def sample_top_view(ax):
    ax.set_title("A  样品面 / TDI俯视图", fontsize=14, pad=10)
    ax.add_patch(Rectangle((-4.3, -3.5), 8.6, 7.0, facecolor="#F3F4F6",
                           edgecolor="#6B7280", linewidth=1.3))
    # Line illumination: long X, narrow Y.
    ax.add_patch(Rectangle((-3.8, -0.38), 7.6, 0.76, facecolor="#FBBF24",
                           edgecolor="#B45309", linewidth=1.3, alpha=0.78, zorder=3))
    ax.text(0, 0, "线照明光斑", ha="center", va="center", fontsize=11, zorder=5)
    arrow(ax, (-3.6, -2.8), (3.2, -2.8), "#1D4ED8",
          "相机长边 / 光斑长轴：X", (0, -3.18))
    arrow(ax, (-3.25, -2.2), (-3.25, 2.65), "#047857",
          "TDI扫描与电荷转移：+Y", (-2.15, 1.95))
    arrow(ax, (3.25, -2.2), (3.25, 2.45), "#9D174D",
          "75°入射在样品面的投影：+Y", (2.05, 1.45))
    ax.annotate("TDI有效窄边：Y", xy=(0, 0.38), xytext=(0.6, 1.3), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8), color="#DC2626")
    ax.text(0, -4.12, "当前坐标：相机/照明线沿X；扫描及窄接收方向沿Y",
            ha="center", fontsize=10, color="#374151")
    ax.set(xlim=(-5, 5), ylim=(-4.5, 4.2), aspect="equal")
    ax.axis("off")


def fourier_plane_view(ax):
    ax.set_title("B  傅里叶面档杆俯视图", fontsize=14, pad=10)
    ax.add_patch(Circle((0, 0), 4.05, facecolor="#EFF6FF", edgecolor="#6B7280",
                        linewidth=1.3, alpha=0.9))
    centres = [-2.7, -1.35, 0, 1.35, 2.7]
    for cx in centres:
        ax.add_patch(serrated_rod(cx))
    arrow(ax, (-3.35, -3.9), (3.35, -3.9), "#7C3AED",
          "各档杆独立定位 / 调节：X_F", (0, -4.28))
    arrow(ax, (3.65, -3.1), (3.65, 3.1), "#047857",
          "档杆长轴：Y_F", (3.0, 2.35))
    ax.text(-3.95, 4.28, "锯齿沿档杆长轴Y_F周期重复", ha="left", fontsize=10,
            color="#991B1B")
    ax.text(-3.95, 3.83, "锯齿非零级沿像面Y分开，由Y向窄TDI窗排除",
            ha="left", fontsize=10, color="#991B1B")
    ax.text(0, -4.85, "档杆长轴与相机长边X垂直", ha="center", fontsize=10,
            color="#374151")
    ax.set(xlim=(-5, 5), ylim=(-5.1, 4.9), aspect="equal")
    ax.axis("off")


def illumination_side_view(ax):
    ax.set_title("C  75°倾斜照明侧视图（沿X观察）", fontsize=14, pad=10)
    # Sample surface and normal.
    ax.plot([-4.6, 4.6], [0, 0], color="#111827", linewidth=4, solid_capstyle="butt")
    ax.text(3.6, -0.48, "样品面", fontsize=11)
    arrow(ax, (0, 0), (0, 4.1), "#374151", "样品法线 +Z", (0.85, 3.5), width=2.0)
    theta = np.deg2rad(75)
    source = (-4.4 * np.sin(theta), 4.4 * np.cos(theta))
    ax.add_patch(Circle(source, 0.22, facecolor="#DC2626", edgecolor="#7F1D1D", zorder=7))
    arrow(ax, source, (0, 0.08), "#DC2626", "入射光传播方向", (-2.25, 1.15), width=3.2)
    # Projection along +Y.
    arrow(ax, (0.15, 0.35), (3.5, 0.35), "#9D174D",
          "样品面投影：+Y", (2.05, 0.78), width=2.4)
    # Angle arc from +Z toward source ray.
    angles = np.linspace(np.pi / 2, np.pi - np.deg2rad(15), 80)
    radius = 1.3
    ax.plot(radius * np.cos(angles), radius * np.sin(angles), color="#2563EB", lw=1.8)
    ax.text(-0.78, 1.45, "75°（相对法线）", fontsize=11, color="#2563EB")
    ax.text(-4.55, 1.95, "光源位于−Y侧", fontsize=10, color="#7F1D1D")
    ax.text(0, -1.05, "线光斑长轴沿X（垂直本侧视图画面）", ha="center", fontsize=10,
            color="#B45309")
    ax.set(xlim=(-5, 5), ylim=(-1.5, 4.8), aspect="equal")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.1]})
    sample_top_view(axes[0])
    fourier_plane_view(axes[1])
    illumination_side_view(axes[2])
    fig.suptitle("当前待确认的系统方向关系（TDI有效窄边已调整为Y）", fontsize=17, y=0.975)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.90), w_pad=2.0)
    path = OUT / "tdi_rod_illumination_direction_schematic.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
