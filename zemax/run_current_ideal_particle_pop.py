"""Native OpticStudio POP for the current ideal 355 nm Fourier-filter design.

Target system: f_obj=30 mm, NA=0.6, 36 mm pupil, f_tube=600 mm, 20x.
Object: 2 um vertical periodic amplitude bars plus one irregular bright particle
in a dark gap.  Seven serrated rods use a 1.0 mm hardware core, 0.5 mm
one-sided A/B depth and 0.75 mm tooth pitch.  The X relay is represented at
the original BFP by dividing X positions and widths by M_FX=2.8169.

This is an ideal-paraxial native Zemax POP baseline, not the final real-lens
prescription and not a Python FFT substitute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

from zosapi_utils import ZOSApplication


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "zemax_current_ideal_particle"

WAVELENGTH_UM = 0.355
WAVELENGTH_MM = WAVELENGTH_UM * 1e-3
F_OBJ_MM = 30.0
F_TUBE_MM = 600.0
NA = 0.60
MAG_IMAGE = 20.0
PUPIL_DIAMETER_MM = 2 * F_OBJ_MM * NA
OBJECT_FOV_MM = 0.50
OBJECT_PERIOD_MM = 0.002
BRIGHT_WIDTH_MM = OBJECT_PERIOD_MM * 4.0 / 7.0
M_FX = 2.816901408450704
M_FY = 1.0
HARDWARE_SPACING_MM = M_FX * F_OBJ_MM * WAVELENGTH_MM / OBJECT_PERIOD_MM
BFP_SPACING_MM = HARDWARE_SPACING_MM / M_FX
CORE_HARDWARE_MM = 1.0
DEPTH_A_HARDWARE_MM = 0.5
DEPTH_B_HARDWARE_MM = 0.5
CORE_BFP_MM = CORE_HARDWARE_MM / M_FX
DEPTH_A_BFP_MM = DEPTH_A_HARDWARE_MM / M_FX
DEPTH_B_BFP_MM = DEPTH_B_HARDWARE_MM / M_FX
TOOTH_PITCH_BFP_Y_MM = 0.75
TDI_IMAGE_HEIGHT_MM = 25 * 0.005
ACTIVE_ORDERS = np.arange(-3, 4)
# The POP start window is a local 0.15 mm patch inside the 0.50 mm system FOV.
# With an explicit 1024 x 1024 grid the object-equivalent sample is 0.146 um
# and the ideal Fourier-plane span is about N*lambda*f/L = 72.70 mm.  Thus the
# 0.6 um particle has about four samples while the 36 mm pupil and the outer
# m=+/-3 rod centres at +/-15.975 mm remain fully included.  No display zoom
# is permitted.
POP_START_WIDTH_MM = 0.15


def rectangle(x0: float, x1: float, y0: float, y1: float):
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def irregular_particle(cx: float, cy: float, radius_mm: float, count: int = 40):
    points = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        radius = radius_mm * (1.0 + 0.18 * math.sin(3 * angle + 0.4)
                              + 0.10 * math.sin(5 * angle - 0.7))
        points.append((cx + 1.10 * radius * math.cos(angle),
                       cy + 0.82 * radius * math.sin(angle)))
    return points


def object_polygons(defective: bool):
    half = OBJECT_FOV_MM / 2
    centres = np.arange(-half, half + OBJECT_PERIOD_MM / 2, OBJECT_PERIOD_MM)
    polygons = []
    for centre in centres:
        x0 = max(-half, float(centre - BRIGHT_WIDTH_MM / 2))
        x1 = min(half, float(centre + BRIGHT_WIDTH_MM / 2))
        if x1 > x0:
            polygons.append(rectangle(x0, x1, -half, half))
    if defective:
        # The dark gap is centred at x=p/2.  The selected irregular particle is
        # 0.6 um-class and remains an isolated local clear defect, not a bridge.
        polygons.append(irregular_particle(OBJECT_PERIOD_MM / 2, 0.0, 0.00030))
    return polygons


def triangle_fraction(y_mm: float) -> float:
    phase = y_mm / TOOTH_PITCH_BFP_Y_MM
    triangle = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
    return 0.5 * (triangle + 1.0)


def pass_polygons(orders, depth_bfp_mm: float):
    outer_x = 22.0
    outer_y = 22.0
    if not orders:
        return [rectangle(-outer_x, outer_x, -outer_y, outer_y)]
    samples = np.arange(-outer_y, outer_y + TOOTH_PITCH_BFP_Y_MM / 4,
                        TOOTH_PITCH_BFP_Y_MM / 4)
    ys = sorted(set([-outer_y, outer_y] +
                    [float(value) for value in samples if -outer_y < value < outer_y]))
    centres = [float(order) * BFP_SPACING_MM for order in sorted(orders)]

    def half_width(y):
        return CORE_BFP_MM / 2 + depth_bfp_mm * triangle_fraction(y)

    def left(centre, y):
        return centre - half_width(y)

    def right(centre, y):
        return centre + half_width(y)

    polygons = []
    first = centres[0]
    polygons.append([(-outer_x, y) for y in ys] +
                    [(left(first, y), y) for y in reversed(ys)])
    for first_centre, second_centre in zip(centres[:-1], centres[1:]):
        polygons.append([(right(first_centre, y), y) for y in ys] +
                        [(left(second_centre, y), y) for y in reversed(ys)])
    last = centres[-1]
    polygons.append([(right(last, y), y) for y in ys] +
                    [(outer_x, y) for y in reversed(ys)])
    return polygons


def write_uda(path: Path, polygons) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("! Current 355 nm ideal-particle Fourier-filter POP model\n")
        for polygon in polygons:
            for x, y in polygon:
                stream.write(f"{x:.12g} {y:.12g}\n")
            stream.write("0 0\n")


def install_udas(zos: ZOSApplication, out: Path):
    definitions = {
        "object_regular": object_polygons(False),
        "object_particle": object_polygons(True),
        "open": pass_polygons([], 0.0),
        "deck_a": pass_polygons([-2, 0, 2], DEPTH_A_BFP_MM),
        "deck_b": pass_polygons([-3, -1, 1, 3], DEPTH_B_BFP_MM),
    }
    destination = Path(zos.application.ObjectsDir) / "Apertures"
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, polygons in definitions.items():
        filename = f"CURRENT_IDEAL_{key.upper()}.UDA"
        source = out / "uda" / filename
        write_uda(source, polygons)
        shutil.copy2(source, destination / filename)
        result[key] = filename
    zos.application.UpdateFileLists()
    return result


def set_user_aperture(zos, surface, filename: str) -> None:
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(
        api.Editors.LDE.SurfaceApertureTypes.UserAperture)
    data._S_UserAperture.ApertureFile = filename
    data._S_UserAperture.UDAScale = 1.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_circle(zos, surface, radius: float) -> None:
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(
        api.Editors.LDE.SurfaceApertureTypes.CircularAperture)
    data._S_CircularAperture.MaximumRadius = radius
    data._S_CircularAperture.MinimumRadius = 0.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_paraxial(zos, surface, focal_mm: float) -> None:
    api = zos.ZOSAPI
    surface.ChangeType(surface.GetSurfaceTypeSettings(api.Editors.LDE.SurfaceType.Paraxial))
    surface.GetSurfaceCell(api.Editors.LDE.SurfaceColumn.Par1).DoubleValue = focal_mm


def build_model(zos: ZOSApplication, out: Path, udas: dict,
                defective: bool, filtered: bool) -> Path:
    system = zos.system
    system.New(False)
    state = "particle" if defective else "regular"
    mode = "seven_rods" if filtered else "open"
    system.SystemData.TitleNotes.Title = f"Current ideal POP: {state}, {mode}"
    system.SystemData.TitleNotes.Notes = (
        "Native Zemax ideal-paraxial baseline: 355 nm, f=30/600 mm, NA=0.6, 20x, "
        "2 um vertical pattern plus one irregular clear particle; X-relay-equivalent rods."
    )
    system.SystemData.Aperture.ApertureValue = PUPIL_DIAMETER_MM
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = WAVELENGTH_UM
    field = system.SystemData.Fields.GetField(1)
    field.X = 0.0
    field.Y = 0.0
    lde = system.LDE
    while lde.NumberOfSurfaces < 7:
        lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)
    definitions = [
        (1, "2 um PERIODIC OBJECT + OPTIONAL PARTICLE", "standard", 0.0, F_OBJ_MM, 0.30),
        (2, "IDEAL OBJECTIVE f=30 mm; 36 mm PUPIL", "paraxial", F_OBJ_MM, F_OBJ_MM, 18.0),
        (3, "FOURIER DECK A: m=-2,0,+2", "standard", 0.0, 0.0, 22.0),
        (4, "FOURIER DECK B: m=-3,-1,+1,+3", "standard", 0.0, F_TUBE_MM, 22.0),
        (5, "IDEAL TUBE LENS f=600 mm", "paraxial", F_TUBE_MM, F_TUBE_MM, 30.0),
        (6, "20X IMAGE / TDI PLANE", "image", 0.0, 0.0, 6.0),
    ]
    for number, comment, surface_type, focal, thickness, semi_diameter in definitions:
        surface = lde.GetSurfaceAt(number)
        surface.Comment = comment
        surface.Thickness = thickness
        surface.SemiDiameter = semi_diameter
        if surface_type == "paraxial":
            set_paraxial(zos, surface, focal)
    set_user_aperture(zos, lde.GetSurfaceAt(1),
                      udas["object_particle" if defective else "object_regular"])
    set_circle(zos, lde.GetSurfaceAt(2), PUPIL_DIAMETER_MM / 2)
    set_user_aperture(zos, lde.GetSurfaceAt(3), udas["deck_a" if filtered else "open"])
    set_user_aperture(zos, lde.GetSurfaceAt(4), udas["deck_b" if filtered else "open"])
    for number in range(1, 7):
        data = lde.GetSurfaceAt(number).PhysicalOpticsData
        data.DoNotRescaleBeamSizeUsingRayData = True
        data.AutoResample = False
        data.ResampleAfterRefraction = False
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"current_ideal_{state}_{mode}.zos"
    system.SaveAs(str(path))
    return path


def net_grid(grid):
    nx, ny = int(grid.Nx), int(grid.Ny)
    values = np.asarray(list(grid.Values), dtype=float).reshape((ny, nx))
    x = float(grid.MinX) + np.arange(nx) * float(grid.Dx)
    y = float(grid.MinY) + np.arange(ny) * float(grid.Dy)
    return x, y, values


def run_pop(zos: ZOSApplication, path: Path, end_surface: int, grid_size: int):
    system = zos.system
    api = zos.ZOSAPI
    system.LoadFile(str(path), False)
    pop = system.Analyses.New_Analysis_SettingsFirst(
        api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    settings = api.Analysis.PhysicalOptics.IAS_PhysicalOpticsPropagation(
        pop.GetSettings())
    settings.StartSurface.SetSurfaceNumber(1)
    settings.EndSurface.SetSurfaceNumber(end_surface)
    settings.Wavelength.SetWavelengthNumber(1)
    settings.Field.SetFieldNumber(1)
    settings.BeamType = api.Analysis.PhysicalOptics.POPBeamTypes.TopHat
    settings.SetParameterValue(0, 0.55)
    settings.SetParameterValue(1, 0.55)
    for index in range(2, 6):
        settings.SetParameterValue(index, 0.0)
    sampling = getattr(api.Analysis.SampleSizes, f"S_{grid_size}x{grid_size}")
    settings.XSampling = sampling
    settings.YSampling = sampling
    settings.XWidth = POP_START_WIDTH_MM
    settings.YWidth = POP_START_WIDTH_MM
    settings.UsePolarization = False
    settings.UseTotalPower = True
    settings.TotalPower = 1.0
    settings.SeparateXY = False
    settings.DataType = api.Analysis.PhysicalOptics.POPDataTypes.Irradiance
    settings.ZoomIn = api.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
    pop.ApplyAndWaitForCompletion()
    results = pop.GetResults()
    messages = [str(results.GetMessageAt(i)) for i in range(results.NumberOfMessages)]
    if results.NumberOfDataGrids < 1:
        pop.Close()
        raise RuntimeError(f"No POP grid: {path.name}, S{end_surface}, {messages}")
    x, y, irradiance = net_grid(results.GetDataGrid(0))
    pop.Close()
    return x, y, irradiance, messages


def resample(source_x, source_y, values, target_x, target_y):
    intermediate = np.vstack([
        np.interp(target_x, source_x, row, left=0, right=0) for row in values
    ])
    return np.vstack([
        np.interp(target_y, source_y, intermediate[:, index], left=0, right=0)
        for index in range(len(target_x))
    ]).T


def calculate_metrics(data: dict):
    x, y, regular_open = data["regular_open_s6"]
    _, _, particle_open = data["particle_open_s6"]
    xf, yf, regular_filtered = data["regular_filtered_s6"]
    xpf, ypf, particle_filtered = data["particle_filtered_s6"]
    regular_filtered = resample(xf, yf, regular_filtered, x, y)
    particle_filtered = resample(xpf, ypf, particle_filtered, x, y)
    xx, yy = np.meshgrid(x, y)
    tdi_roi = (np.abs(xx) <= 0.60) & (np.abs(yy) <= TDI_IMAGE_HEIGHT_MM / 2)
    particle_centre_x = -MAG_IMAGE * OBJECT_PERIOD_MM / 2
    particle_roi = tdi_roi & ((xx - particle_centre_x) ** 2 + yy**2 <= 0.018**2)
    pattern_roi = tdi_roi & ~particle_roi
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    open_background_power = float(np.sum(regular_open[tdi_roi]) * dx * dy)
    filtered_background_power = float(np.sum(regular_filtered[tdi_roi]) * dx * dy)
    open_particle_peak = float(np.max(particle_open[particle_roi]))
    filtered_particle_peak = float(np.max(particle_filtered[particle_roi]))
    open_pattern95 = float(np.percentile(particle_open[pattern_roi], 95))
    filtered_pattern95 = float(np.percentile(particle_filtered[pattern_roi], 95))
    return {
        "tdi_image_height_mm": TDI_IMAGE_HEIGHT_MM,
        "background_suppression_db": float(10 * np.log10(
            max(open_background_power, 1e-30) / max(filtered_background_power, 1e-30))),
        "particle_peak_retention_percent": float(100 * filtered_particle_peak /
                                                   max(open_particle_peak, 1e-30)),
        "particle_to_pattern_open_db": float(10 * np.log10(
            max(open_particle_peak, 1e-30) / max(open_pattern95, 1e-30))),
        "particle_to_pattern_filtered_db": float(10 * np.log10(
            max(filtered_particle_peak, 1e-30) / max(filtered_pattern95, 1e-30))),
        "common_peak_W_per_mm2": float(max(np.max(regular_open), np.max(particle_open))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grid", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument("--job-key", help="Run only one named POP job for sampling audit")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    messages = {}
    with ZOSApplication() as zos:
        print(f"OpticStudio={zos.application.OpticStudioVersion}")
        udas = install_udas(zos, out)
        models = {}
        for defective in (False, True):
            for filtered in (False, True):
                state = "particle" if defective else "regular"
                mode = "filtered" if filtered else "open"
                key = f"{state}_{mode}"
                models[key] = build_model(zos, out, udas, defective, filtered)
                print(f"built={models[key]}")
        jobs = [
            ("regular_open_s6", models["regular_open"], 6),
            ("particle_open_s6", models["particle_open"], 6),
            ("regular_filtered_s6", models["regular_filtered"], 6),
            ("particle_filtered_s6", models["particle_filtered"], 6),
            ("particle_open_s3", models["particle_open"], 3),
            ("particle_filtered_s4", models["particle_filtered"], 4),
        ]
        if args.job_key:
            jobs = [job for job in jobs if job[0] == args.job_key]
            if not jobs:
                raise ValueError(f"Unknown --job-key {args.job_key!r}")
        for index, (key, path, end_surface) in enumerate(jobs, 1):
            print(f"POP {index}/{len(jobs)} {key}", flush=True)
            x, y, irradiance, pop_messages = run_pop(
                zos, path, end_surface, args.grid)
            data[key] = (x, y, irradiance)
            messages[key] = pop_messages
            power = float(np.sum(np.maximum(irradiance, 0)) *
                          abs(x[1] - x[0]) * abs(y[1] - y[0]))
            print(f" grid={len(x)}x{len(y)} span={x[-1]-x[0]:.6g}x{y[-1]-y[0]:.6g} mm "
                  f"power={power:.6g} W", flush=True)
    np.savez_compressed(
        out / "native_pop_grids.npz",
        **{f"{key}_x": value[0] for key, value in data.items()},
        **{f"{key}_y": value[1] for key, value in data.items()},
        **{f"{key}_irradiance": value[2] for key, value in data.items()},
    )
    metrics = calculate_metrics(data) if not args.job_key else {}
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    summary = {
        "engine": "Local OpticStudio 2022 R2.02 native POP",
        "scope": "ideal paraxial 4f baseline under the current final optical and filter conditions",
        "parameters": {
            "wavelength_nm": 355,
            "objective_focal_length_mm": F_OBJ_MM,
            "objective_na": NA,
            "true_pupil_diameter_mm": PUPIL_DIAMETER_MM,
            "tube_lens_focal_length_mm": F_TUBE_MM,
            "image_magnification": MAG_IMAGE,
            "object_period_um": OBJECT_PERIOD_MM * 1000,
            "bright_width_um": BRIGHT_WIDTH_MM * 1000,
            "dark_width_um": (OBJECT_PERIOD_MM - BRIGHT_WIDTH_MM) * 1000,
            "defect": "single irregular approximately 0.6 um bright particle in a dark periodic gap",
            "hardware_fourier_x_magnification": M_FX,
            "hardware_order_spacing_mm": HARDWARE_SPACING_MM,
            "equivalent_original_bfp_spacing_mm": BFP_SPACING_MM,
            "rod_core_hardware_mm": CORE_HARDWARE_MM,
            "deck_A_depth_hardware_mm": DEPTH_A_HARDWARE_MM,
            "deck_B_depth_hardware_mm": DEPTH_B_HARDWARE_MM,
            "tooth_pitch_y_mm": TOOTH_PITCH_BFP_Y_MM,
            "active_orders": ACTIVE_ORDERS.tolist(),
            "tdi_stages": 25,
            "tdi_pixel_pitch_um": 5.0,
            "pop_grid_size": args.grid,
            "pop_start_window_mm": POP_START_WIDTH_MM,
        },
        "metrics": metrics,
        "messages": messages,
        "evaluation_rule": "native raw irradiance; one W/mm^2 scale; no periodic-background subtraction",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
