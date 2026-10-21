"""Build and run a sequential POP proof model in local OpticStudio.

The model is intentionally paraxial: its purpose is to verify the physical
Fourier-transform / mask / inverse-transform chain and all mechanical scales.
It does not claim to be a manufacturable 355 nm objective prescription.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from zosapi_utils import ZOSApplication


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "zemax"

WAVELENGTH_UM = 0.355
OBJECTIVE_F_MM = 50.0
TUBE_F_MM = 1000.0
NA = 0.10
OBJECT_FOV_MM = 0.50
PATTERN_PITCH_MM = 0.020
PATTERN_BAR_WIDTH_MM = 0.010
PUPIL_RELAY_MAG = 5.0
RELAY_F1_MM = 50.0
RELAY_F2_MM = 250.0
FILTER_PUPIL_DIAMETER_MM = 50.0
ORDER_SPACING_MM = (
    PUPIL_RELAY_MAG * (WAVELENGTH_UM * 1e-3) * OBJECTIVE_F_MM / PATTERN_PITCH_MM
)
LINEAR_ROD_WIDTH_MM = 1.0
FUZZY_MIN_WIDTH_MM = 1.1
FUZZY_MAX_WIDTH_MM = 4.0
SERRATION_PITCH_MM = 0.75
POP_START_WIDTH_SETTING_MM = 40.0
SURFACE_SEMI_DIAMETERS_MM = {1: 0.30, 2: 5.0, 3: 5.0, 4: 6.0, 5: 6.0}


@dataclass(frozen=True)
class Rod:
    order: int
    center_mm: float
    kind: str


SURFACES = [
    (1, "PERIODIC OBJECT UDA", "standard", None, 50.0),
    (2, "OBJECTIVE f=50 mm, NA=0.10", "paraxial", 50.0, 50.0),
    (3, "OBJECTIVE BFP: EQUIVALENT FILTER UDA", "standard", None, 1000.0),
    (4, "TUBE LENS f=1000 mm", "paraxial", 1000.0, 1000.0),
    (5, "20X IMAGE", "image", None, 0.0),
]


def rectangle(x0: float, x1: float, y0: float, y1: float):
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def write_uda(path: Path, polygons: Iterable[Sequence[Tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("! Generated for the Fourier filtering proof model\n")
        for polygon in polygons:
            for x, y in polygon:
                stream.write(f"{x:.12g} {y:.12g}\n")
            stream.write("0 0\n")


def object_polygons(defective: bool) -> List[List[Tuple[float, float]]]:
    half = OBJECT_FOV_MM / 2
    centers = np.arange(-0.24, 0.2400001, PATTERN_PITCH_MM)
    polygons: List[List[Tuple[float, float]]] = []
    for center in centers:
        x0 = float(center - PATTERN_BAR_WIDTH_MM / 2)
        x1 = float(center + PATTERN_BAR_WIDTH_MM / 2)
        if defective and abs(center) < 1e-10:
            polygons.append(rectangle(x0, x1, -half, -0.050))
            polygons.append(rectangle(x0, x1, 0.050, half))
            shifted = center + 0.006
            polygons.append(
                rectangle(
                    float(shifted - PATTERN_BAR_WIDTH_MM / 2),
                    float(shifted + PATTERN_BAR_WIDTH_MM / 2),
                    -0.050,
                    0.050,
                )
            )
        else:
            polygons.append(rectangle(x0, x1, -half, half))
    return polygons


def triangle_wave(y_mm: float) -> float:
    """Unit triangular wave in [-1, 1], with its valley at y=0."""
    phase = y_mm / SERRATION_PITCH_MM
    return 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0


def rod_half_width(rod: Rod, y_mm: float) -> float:
    if rod.kind == "linear":
        return LINEAR_ROD_WIDTH_MM / 2
    midpoint = (FUZZY_MIN_WIDTH_MM + FUZZY_MAX_WIDTH_MM) / 2
    excursion = (FUZZY_MAX_WIDTH_MM - FUZZY_MIN_WIDTH_MM) / 2
    return (midpoint + excursion * triangle_wave(y_mm)) / 2


def rod_set(filter_kind: str) -> List[Rod]:
    if filter_kind == "unfiltered":
        return []
    if filter_kind == "hard":
        return [
            Rod(order=m, center_mm=m * ORDER_SPACING_MM, kind="linear")
            for m in range(-5, 6)
        ]
    if filter_kind == "hybrid":
        return [
            Rod(
                order=m,
                center_mm=m * ORDER_SPACING_MM,
                kind="fuzzy" if m % 2 == 0 else "linear",
            )
            for m in range(-5, 6)
        ]
    raise ValueError(filter_kind)


def filter_polygons(filter_kind: str) -> List[List[Tuple[float, float]]]:
    half_y = FILTER_PUPIL_DIAMETER_MM / 2
    outer_x = 30.0
    if filter_kind == "unfiltered":
        return [rectangle(-outer_x, outer_x, -half_y, half_y)]

    # Include all triangular extrema exactly; no sinusoidal approximation is used.
    extrema = np.arange(-half_y, half_y + SERRATION_PITCH_MM / 2, SERRATION_PITCH_MM / 2)
    y_values = sorted(set([-half_y, half_y] + [float(y) for y in extrema if -half_y < y < half_y]))
    rods = rod_set(filter_kind)

    def left(rod: Rod, y: float) -> float:
        return rod.center_mm - rod_half_width(rod, y)

    def right(rod: Rod, y: float) -> float:
        return rod.center_mm + rod_half_width(rod, y)

    polygons: List[List[Tuple[float, float]]] = []
    first = rods[0]
    polygons.append(
        [(outer_x * -1, y) for y in y_values]
        + [(left(first, y), y) for y in reversed(y_values)]
    )
    for rod_a, rod_b in zip(rods[:-1], rods[1:]):
        polygons.append(
            [(right(rod_a, y), y) for y in y_values]
            + [(left(rod_b, y), y) for y in reversed(y_values)]
        )
    last = rods[-1]
    polygons.append(
        [(right(last, y), y) for y in y_values]
        + [(outer_x, y) for y in reversed(y_values)]
    )
    return polygons


def scale_polygons(polygons, scale: float):
    return [[(scale * x, scale * y) for x, y in polygon] for polygon in polygons]


def check_mechanics() -> None:
    assert math.isclose(ORDER_SPACING_MM, 4.4375, abs_tol=1e-12)
    assert 2 * ORDER_SPACING_MM >= 4.0
    assert ORDER_SPACING_MM >= 1.0
    ys = np.linspace(-25, 25, 10001)
    widths = np.asarray([2 * rod_half_width(Rod(0, 0, "fuzzy"), float(y)) for y in ys])
    assert abs(widths.min() - FUZZY_MIN_WIDTH_MM) < 1e-6
    assert abs(widths.max() - FUZZY_MAX_WIDTH_MM) < 1e-6


def install_udas(zos: ZOSApplication, output_dir: Path) -> Dict[str, str]:
    uda_dir = output_dir / "uda"
    definitions = {
        "object_regular": object_polygons(False),
        "object_defect": object_polygons(True),
        # The POP mask is at the original objective BFP, so all mechanical
        # dimensions are divided by the 5x pupil-relay magnification.  It is
        # optically equivalent to the full-size hardware mask at its conjugate.
        "filter_unfiltered": scale_polygons(filter_polygons("unfiltered"), 1 / PUPIL_RELAY_MAG),
        "filter_hard": scale_polygons(filter_polygons("hard"), 1 / PUPIL_RELAY_MAG),
        "filter_hybrid": scale_polygons(filter_polygons("hybrid"), 1 / PUPIL_RELAY_MAG),
    }
    installed: Dict[str, str] = {}
    destination_dir = Path(zos.application.ObjectsDir) / "Apertures"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for key, polygons in definitions.items():
        filename = f"FOURIER_{key.upper()}.UDA"
        source = uda_dir / filename
        write_uda(source, polygons)
        shutil.copy2(source, destination_dir / filename)
        installed[key] = filename
    zos.application.UpdateFileLists()
    return installed


def set_user_aperture(zos: ZOSApplication, surface, filename: str) -> None:
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(
        api.Editors.LDE.SurfaceApertureTypes.UserAperture
    )
    data._S_UserAperture.ApertureFile = filename
    data._S_UserAperture.UDAScale = 1.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_circular_aperture(zos: ZOSApplication, surface, radius_mm: float) -> None:
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(
        api.Editors.LDE.SurfaceApertureTypes.CircularAperture
    )
    data._S_CircularAperture.MaximumRadius = radius_mm
    data._S_CircularAperture.MinimumRadius = 0.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_paraxial_lens(zos: ZOSApplication, surface, focal_length_mm: float) -> None:
    api = zos.ZOSAPI
    surface.ChangeType(surface.GetSurfaceTypeSettings(api.Editors.LDE.SurfaceType.Paraxial))
    surface.GetSurfaceCell(api.Editors.LDE.SurfaceColumn.Par1).DoubleValue = focal_length_mm


def build_model(
    zos: ZOSApplication,
    output_dir: Path,
    udas: Dict[str, str],
    defective: bool,
    filter_kind: str,
) -> Path:
    api = zos.ZOSAPI
    system = zos.system
    system.New(False)
    system.SystemData.TitleNotes.Title = (
        f"355 nm Fourier filtering: {'defect' if defective else 'regular'}, {filter_kind}"
    )
    system.SystemData.TitleNotes.Notes = (
        "Paraxial POP proof model; f_obj=50 mm, f_tube=1000 mm, M=20x, "
        "NA=0.10, object FOV=0.5 mm, pupil relay=5x."
    )
    system.SystemData.Aperture.ApertureValue = 2 * OBJECTIVE_F_MM * NA
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = WAVELENGTH_UM
    field = system.SystemData.Fields.GetField(1)
    field.X = 0.0
    field.Y = 0.0

    lde = system.LDE
    while lde.NumberOfSurfaces < 6:
        lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)

    for number, comment, surface_type, focal_length, thickness in SURFACES:
        surface = lde.GetSurfaceAt(number)
        surface.Comment = comment
        surface.Thickness = thickness
        surface.SemiDiameter = SURFACE_SEMI_DIAMETERS_MM[number]
        if surface_type == "paraxial":
            set_paraxial_lens(zos, surface, float(focal_length))

    set_user_aperture(
        zos,
        lde.GetSurfaceAt(1),
        udas["object_defect" if defective else "object_regular"],
    )
    set_circular_aperture(zos, lde.GetSurfaceAt(2), OBJECTIVE_F_MM * NA)
    set_user_aperture(zos, lde.GetSurfaceAt(3), udas[f"filter_{filter_kind}"])
    # Keep the scalar grid governed by diffraction sampling, not by the on-axis
    # geometrical pilot rays.  The latter contain no information about the
    # +/-0.8875 mm grating orders and can collapse the focal-plane window.
    for number in range(1, 6):
        pop_data = lde.GetSurfaceAt(number).PhysicalOpticsData
        pop_data.DoNotRescaleBeamSizeUsingRayData = True

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    name = f"fourier_{'defect' if defective else 'regular'}_{filter_kind}.zos"
    model_path = model_dir / name
    system.SaveAs(str(model_path))
    return model_path


def net_grid_to_numpy(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = int(grid.Nx)
    ny = int(grid.Ny)
    values = np.asarray(list(grid.Values), dtype=float)
    if values.size != nx * ny:
        raise RuntimeError(f"Unexpected POP grid size: {values.size} != {nx}*{ny}")
    # .NET enumerates this POP grid row-by-row (Y outer, X inner).
    image = values.reshape((ny, nx))
    x = float(grid.MinX) + np.arange(nx) * float(grid.Dx)
    y = float(grid.MinY) + np.arange(ny) * float(grid.Dy)
    return x, y, image


def run_pop(zos: ZOSApplication, model_path: Path, end_surface: int, sampling_code: int):
    system = zos.system
    api = zos.ZOSAPI
    system.LoadFile(str(model_path), False)
    # Re-apply this switch after loading because OpticStudio 2022 does not always
    # persist it in a binary .zos save made by a standalone API instance.
    for number in range(1, system.LDE.NumberOfSurfaces):
        pop_data = system.LDE.GetSurfaceAt(number).PhysicalOpticsData
        pop_data.DoNotRescaleBeamSizeUsingRayData = True
        pop_data.AutoResample = False
        pop_data.ResampleAfterRefraction = False
    pop = system.Analyses.New_Analysis(api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    settings = pop.GetSettings()
    settings.StartSurface.SetSurfaceNumber(1)
    settings.EndSurface.SetSurfaceNumber(end_surface)
    settings.Wavelength.SetWavelengthNumber(1)
    settings.Field.SetFieldNumber(1)
    settings.BeamType = 2  # uniform rectangular field
    settings.Update()
    settings.SetParameterValue(0, 0.55)
    settings.SetParameterValue(1, 0.55)
    settings.SetParameterValue(2, 0.0)
    settings.SetParameterValue(3, 0.0)
    settings.SetParameterValue(4, 0.0)
    settings.SetParameterValue(5, 0.0)
    settings.XSampling = sampling_code
    settings.YSampling = sampling_code
    # In OpticStudio 2022 POP this setting is four times the returned full grid
    # span for a uniform beam.  40.0 therefore produces a 10 mm computational
    # window.  The UDA object occupies only its central 0.50 mm; the large empty
    # guard band preserves the +/-5 mm pupil and its discrete diffraction orders.
    settings.XWidth = POP_START_WIDTH_SETTING_MM
    settings.YWidth = POP_START_WIDTH_SETTING_MM
    settings.UsePolarization = False
    settings.UseTotalPower = True
    settings.TotalPower = 1.0
    settings.SeparateXY = False
    pop.ApplyAndWaitForCompletion()
    results = pop.GetResults()
    messages = [str(results.GetMessageAt(i)) for i in range(results.NumberOfMessages)]
    if results.NumberOfDataGrids < 1:
        pop.Close()
        raise RuntimeError(f"POP returned no grid for {model_path.name}, S{end_surface}: {messages}")
    x, y, image = net_grid_to_numpy(results.GetDataGrid(0))
    pop.Close()
    return x, y, image, messages


def normalized_for_display(image: np.ndarray, log: bool = False) -> np.ndarray:
    image = np.maximum(np.asarray(image, dtype=float), 0)
    peak = image.max()
    scaled = image / peak if peak > 0 else image
    return np.log10(np.maximum(scaled, 1e-8)) if log else scaled


def grid_power(x: np.ndarray, y: np.ndarray, image: np.ndarray) -> float:
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else 1.0
    return float(np.sum(np.maximum(image, 0)) * dx * dy)


def resample_to(source_x, source_y, source_image, target_x, target_y):
    intermediate = np.vstack(
        [np.interp(target_x, source_x, row, left=0.0, right=0.0) for row in source_image]
    )
    return np.vstack(
        [np.interp(target_y, source_y, intermediate[:, i], left=0.0, right=0.0) for i in range(len(target_x))]
    ).T


def save_prescription(output_dir: Path) -> None:
    z = 0.0
    with (output_dir / "zemax_surface_prescription.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["surface", "comment", "type", "focal_length_mm", "thickness_mm", "z_mm"])
        for number, comment, surface_type, focal_length, thickness in SURFACES:
            writer.writerow([number, comment, surface_type, focal_length or "", thickness, z])
            z += thickness


def plot_filter_geometry(output_dir: Path) -> None:
    y = np.linspace(-5, 5, 2001)
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for rod in rod_set("hybrid"):
        left = np.asarray([rod.center_mm - rod_half_width(rod, float(v)) for v in y])
        right = np.asarray([rod.center_mm + rod_half_width(rod, float(v)) for v in y])
        color = "#d95f02" if rod.kind == "fuzzy" else "#1b77b3"
        ax.fill_betweenx(y, left, right, color=color, alpha=0.85)
        ax.text(rod.center_mm, 5.15, f"m={rod.order}", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-26, 26)
    ax.set_ylim(-5.7, 5.7)
    ax.set_xlabel("5x relayed Fourier-plane x (mm)")
    ax.set_ylabel("y (mm; central 10 mm shown)")
    ax.set_title("Two-deck filter geometry: even fuzzy rods / odd linear rods")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "zemax_filter_geometry.png", dpi=180)
    plt.close(fig)


def plot_results(output_dir: Path, data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    # Fourier-transform chain for the defective object.
    keys = [
        ("defect_unfiltered_s3", "Objective BFP before mask", True),
        ("defect_hard_s3", "After equivalent straight rods", True),
        ("defect_hybrid_s3", "After equivalent hybrid mask", True),
        ("defect_unfiltered_s5", "Unfiltered 20x image", False),
        ("defect_hard_s5", "Straight-rod filtered image", False),
        ("defect_hybrid_s5", "Two-deck filtered image", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for ax, (key, title, use_log) in zip(axes.flat, keys):
        x, y, image = data[key]
        shown = normalized_for_display(image, log=use_log)
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
        fig.colorbar(im, ax=ax, shrink=0.82, label="log10 normalized" if use_log else "normalized")
    fig.savefig(output_dir / "zemax_pop_transform_process.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14.6, 12.0), constrained_layout=True)
    for row, filter_kind in enumerate(("unfiltered", "hard", "hybrid")):
        xr, yr, regular = data[f"regular_{filter_kind}_s5"]
        xd, yd, defect = data[f"defect_{filter_kind}_s5"]
        defect_on_regular = resample_to(xd, yd, defect, xr, yr)
        scale = max(float(regular.max()), float(defect_on_regular.max()), 1e-30)
        difference = np.abs(defect_on_regular - regular)
        panels = [regular / scale, defect_on_regular / scale, difference / scale]
        titles = ["regular", "with local shifted bar", "absolute difference"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            im = ax.imshow(
                panel,
                extent=[xr[0], xr[-1], yr[0], yr[-1]],
                origin="lower",
                cmap="magma",
                aspect="auto",
                vmin=0,
                vmax=1 if col < 2 else max(float(panel.max()), 1e-6),
            )
            ax.set_title(f"{filter_kind}: {title}")
            ax.set_xlabel("image x (mm)")
            ax.set_ylabel("image y (mm)")
            fig.colorbar(im, ax=ax, shrink=0.78)
    fig.savefig(output_dir / "zemax_image_comparison.png", dpi=180)
    plt.close(fig)


def calculate_metrics(output_dir: Path, data) -> List[Dict[str, float]]:
    metrics = []
    base_x, base_y, base_regular = data["regular_unfiltered_s5"]
    _, _, base_defect = data["defect_unfiltered_s5"]
    base_defect = resample_to(
        data["defect_unfiltered_s5"][0],
        data["defect_unfiltered_s5"][1],
        base_defect,
        base_x,
        base_y,
    )
    base_difference_rms = float(np.sqrt(np.mean((base_defect - base_regular) ** 2)))
    base_background_rms = float(np.sqrt(np.mean(base_regular**2)))
    base_visibility = base_difference_rms / max(base_background_rms, 1e-30)
    base_power = grid_power(base_x, base_y, base_regular)

    for filter_kind in ("unfiltered", "hard", "hybrid"):
        xr, yr, regular = data[f"regular_{filter_kind}_s5"]
        xd, yd, defect = data[f"defect_{filter_kind}_s5"]
        defect = resample_to(xd, yd, defect, xr, yr)
        difference_rms = float(np.sqrt(np.mean((defect - regular) ** 2)))
        background_rms = float(np.sqrt(np.mean(regular**2)))
        visibility = difference_rms / max(background_rms, 1e-30)
        power = grid_power(xr, yr, regular)
        metrics.append(
            {
                "filter": filter_kind,
                "regular_power": power,
                "periodic_suppression_db": 10 * math.log10(max(base_power, 1e-30) / max(power, 1e-30)),
                "defect_difference_rms": difference_rms,
                "intensity_defect_retention_percent": 100 * difference_rms / max(base_difference_rms, 1e-30),
                "relative_visibility": visibility,
                "visibility_gain": visibility / max(base_visibility, 1e-30),
            }
        )

    with (output_dir / "zemax_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def save_summary(output_dir: Path, metrics, messages) -> None:
    lines = [
        "Zemax POP Fourier-filter proof run",
        f"wavelength_um={WAVELENGTH_UM}",
        f"objective_focal_length_mm={OBJECTIVE_F_MM}",
        f"tube_lens_focal_length_mm={TUBE_F_MM}",
        f"magnification={TUBE_F_MM / OBJECTIVE_F_MM}",
        f"objective_na={NA}",
        f"object_fov_mm={OBJECT_FOV_MM}",
        f"pupil_relay_magnification={PUPIL_RELAY_MAG}",
        f"filter_pupil_diameter_mm={FILTER_PUPIL_DIAMETER_MM}",
        f"order_spacing_mm={ORDER_SPACING_MM}",
        f"linear_rod_width_mm={LINEAR_ROD_WIDTH_MM}",
        f"fuzzy_width_range_mm={FUZZY_MIN_WIDTH_MM}..{FUZZY_MAX_WIDTH_MM}",
        f"serration_pitch_mm={SERRATION_PITCH_MM}",
        "zemax_filter_plane=original objective BFP",
        "zemax_mask_dimensions=hardware dimensions divided by 5",
        "",
        "Metrics use Zemax irradiance grids. Defect retention is based on RMS intensity difference,",
        "so it is not numerically identical to the complex-field metric used by the Python FFT model.",
    ]
    for metric in metrics:
        lines.append(
            "metric=" + ",".join(f"{key}:{value:.9g}" if isinstance(value, float) else f"{key}:{value}" for key, value in metric.items())
        )
    if any(messages.values()):
        lines.append("")
        lines.append("POP messages:")
        for key, values in messages.items():
            for value in values:
                lines.append(f"{key}: {value}")
    (output_dir / "zemax_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sampling-code",
        type=int,
        default=4,
        help="OpticStudio POP per-axis sampling code; 4 is the baseline run.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib output (recommended in the legacy ZOS-API environment).",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Generate UDA and ZOS files without running POP.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_mechanics()
    save_prescription(output_dir)
    if not args.no_plots:
        plot_filter_geometry(output_dir)

    data = {}
    all_messages = {}
    with ZOSApplication() as zos:
        print(f"OpticStudio={zos.application.OpticStudioVersion}")
        print(f"ObjectsDir={zos.application.ObjectsDir}")
        udas = install_udas(zos, output_dir)
        models = {}
        for defective in (False, True):
            for filter_kind in ("unfiltered", "hard", "hybrid"):
                key = f"{'defect' if defective else 'regular'}_{filter_kind}"
                models[key] = build_model(zos, output_dir, udas, defective, filter_kind)
                print(f"built={models[key]}")

        if args.build_only:
            print(f"output={output_dir}")
            return

        jobs = []
        for filter_kind in ("unfiltered", "hard", "hybrid"):
            jobs.append((f"regular_{filter_kind}_s5", models[f"regular_{filter_kind}"], 5))
            jobs.append((f"defect_{filter_kind}_s5", models[f"defect_{filter_kind}"], 5))
        jobs.extend(
            [
                ("defect_unfiltered_s3", models["defect_unfiltered"], 3),
                ("defect_hard_s3", models["defect_hard"], 3),
                ("defect_hybrid_s3", models["defect_hybrid"], 3),
            ]
        )
        for index, (key, model, end_surface) in enumerate(jobs, 1):
            print(f"POP {index}/{len(jobs)}: {key}", flush=True)
            x, y, image, messages = run_pop(zos, model, end_surface, args.sampling_code)
            data[key] = (x, y, image)
            all_messages[key] = messages
            print(
                f"  grid={len(x)}x{len(y)}, x=[{x[0]:.6g},{x[-1]:.6g}], "
                f"y=[{y[0]:.6g},{y[-1]:.6g}], power={grid_power(x,y,image):.6g}",
                flush=True,
            )

    np.savez_compressed(
        output_dir / "zemax_pop_grids.npz",
        **{f"{key}_x": value[0] for key, value in data.items()},
        **{f"{key}_y": value[1] for key, value in data.items()},
        **{f"{key}_irradiance": value[2] for key, value in data.items()},
    )
    metrics = calculate_metrics(output_dir, data)
    if not args.no_plots:
        plot_results(output_dir, data)
    save_summary(output_dir, metrics, all_messages)
    print("metrics=")
    for metric in metrics:
        print(metric)
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
