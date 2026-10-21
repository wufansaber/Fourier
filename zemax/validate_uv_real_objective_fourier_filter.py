"""Validate the final six-element UV objective with native Zemax rays and POP.

The confirmed camera coordinates are used throughout:
  image X = TDI scan / charge-transfer direction;
  image Y = camera long edge and illumination-line direction;
  rods are long along Fourier X and independently positioned along Fourier Y.

The sample is a 2 um horizontal-line pattern (periodic in Y), with one
irregular bright particle in a dark gap.  All image metrics use raw native POP
irradiance; no periodic-reference subtraction is used.
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
BASE_MODEL = (ROOT / "results" / "uv_real_objective" / "models" /
              "uv355_corning7980_6element_ideal_tube.zos")
OUT_DEFAULT = ROOT / "results" / "uv_real_objective_fourier_validation"

WAVELENGTH_UM = 0.355
WAVELENGTH_MM = WAVELENGTH_UM * 1e-3
F_OBJ_NOMINAL_MM = 30.0
MAG_IMAGE = 20.0
OBJECT_PERIOD_MM = 0.002
BRIGHT_WIDTH_MM = OBJECT_PERIOD_MM * 4.0 / 7.0
OBJECT_FOV_MM = 0.50
PUPIL_RADIUS_MM = 18.0
CORE_MM = 1.0
DEPTH_A_MM = 0.5
DEPTH_B_MM = 0.5
TOOTH_PITCH_MM = 0.75
ORDERS_A = (-2, 0, 2)
ORDERS_B = (-3, -1, 1, 3)
POP_START_WIDTH_MM = 0.15
TDI_STAGES = 25
TDI_PIXEL_PITCH_MM = 0.005
TDI_ACTIVE_WIDTH_MM = TDI_STAGES * TDI_PIXEL_PITCH_MM


def rectangle(x0: float, x1: float, y0: float, y1: float):
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def irregular_particle(cx: float, cy: float, radius_mm: float, count: int = 48):
    points = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        radius = radius_mm * (1.0 + 0.18 * math.sin(3 * angle + 0.4)
                              + 0.10 * math.sin(5 * angle - 0.7))
        points.append((cx + 1.10 * radius * math.cos(angle),
                       cy + 0.82 * radius * math.sin(angle)))
    return points


def object_polygons(defective: bool):
    """Horizontal bright lines: periodic frequency and rod centres are in Y."""
    half = OBJECT_FOV_MM / 2
    centres = np.arange(-half, half + OBJECT_PERIOD_MM / 2, OBJECT_PERIOD_MM)
    polygons = []
    for centre in centres:
        y0 = max(-half, float(centre - BRIGHT_WIDTH_MM / 2))
        y1 = min(half, float(centre + BRIGHT_WIDTH_MM / 2))
        if y1 > y0:
            polygons.append(rectangle(-half, half, y0, y1))
    if defective:
        # The first dark gap is centred at y=p/2.  The clear particle is local,
        # irregular and isolated; it is not a bridge or a background subtraction.
        polygons.append(irregular_particle(0.0, OBJECT_PERIOD_MM / 2, 0.00030))
    return polygons


def triangle_fraction(x_mm: float) -> float:
    phase = x_mm / TOOTH_PITCH_MM
    triangle = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
    return 0.5 * (triangle + 1.0)


def pass_polygons(centres, depth_mm: float):
    """Pass corridors around rods long in X, centred at independently set Y."""
    outer = 22.0
    if not centres:
        return [rectangle(-outer, outer, -outer, outer)]
    samples = np.arange(-outer, outer + TOOTH_PITCH_MM / 4,
                        TOOTH_PITCH_MM / 4)
    xs = sorted(set([-outer, outer] +
                    [float(value) for value in samples if -outer < value < outer]))
    centres = sorted(float(value) for value in centres)

    def half_width(x):
        return CORE_MM / 2 + depth_mm * triangle_fraction(x)

    def lower(centre, x):
        return centre - half_width(x)

    def upper(centre, x):
        return centre + half_width(x)

    polygons = []
    first = centres[0]
    polygons.append([(x, -outer) for x in xs] +
                    [(x, lower(first, x)) for x in reversed(xs)])
    for first_centre, second_centre in zip(centres[:-1], centres[1:]):
        polygons.append([(x, upper(first_centre, x)) for x in xs] +
                        [(x, lower(second_centre, x)) for x in reversed(xs)])
    last = centres[-1]
    polygons.append([(x, upper(last, x)) for x in xs] +
                    [(x, outer) for x in reversed(xs)])
    return polygons


def write_uda(path: Path, polygons, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"! {title}\n")
        for polygon in polygons:
            for x, y in polygon:
                stream.write(f"{x:.12g} {y:.12g}\n")
            stream.write("0 0\n")


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


def batch(zos, surface: int, samples, hy: float = 0.0):
    tool = zos.system.Tools.OpenBatchRayTrace()
    try:
        data = zos.ZOSAPI.Tools.RayTrace.IRayTraceNormUnpolData(
            tool.CreateNormUnpol(len(samples), zos.ZOSAPI.Tools.RayTrace.RaysType.Real,
                                 surface))
        mode = getattr(zos.ZOSAPI.Tools.RayTrace.OPDMode, "None")
        for px, py in samples:
            data.AddRay(1, 0.0, hy, float(px), float(py), mode)
        tool.RunAndWaitForCompletion()
        data.StartReadingResults()
        rows = []
        while True:
            row = data.ReadNextResult(0, 0, 0, *([0.0] * 11))
            if not row[0]:
                break
            rows.append([float(value) for value in row[1:]])
        return np.asarray(rows)
    finally:
        tool.Close()


def ray_validation(zos, out: Path):
    """Independent real-ray checks of Abbe sine and physical BFP linearity."""
    system = zos.system
    print(f"ray_validation: loading {BASE_MODEL}", flush=True)
    system.LoadFile(str(BASE_MODEL), False)
    print(f"ray_validation: loaded surfaces={system.LDE.NumberOfSurfaces}", flush=True)
    meridional = [(0.0, float(py)) for py in np.linspace(-1.0, 1.0, 401)]
    print("ray_validation: trace object", flush=True)
    at_object = batch(zos, 0, meridional)
    print("ray_validation: trace BFP S11", flush=True)
    at_bfp = batch(zos, 11, meridional)
    print("ray_validation: trace image S15", flush=True)
    at_image = batch(zos, 15, meridional)
    print("ray_validation: fitting", flush=True)
    valid = ((at_object[:, 1] == 0) & (at_object[:, 2] == 0) &
             (at_bfp[:, 1] == 0) & (at_bfp[:, 2] == 0) &
             (at_image[:, 1] == 0) & (at_image[:, 2] == 0))
    py = np.asarray([sample[1] for sample in meridional])[valid]
    sin_object = at_object[valid, 7]
    sin_image = at_image[valid, 7]
    bfp_y = at_bfp[valid, 4]

    sine_residual = sin_object + 20.0 * sin_image
    # Avoid NumPy LAPACK here: the legacy Python environment paired with
    # OpticStudio 2022 can fault while loading a secondary MKL DLL.  The
    # closed-form straight-line fit is identical for this one-regressor case.
    sin_mean = float(np.mean(sin_object))
    y_mean = float(np.mean(bfp_y))
    slope = float(np.sum((sin_object - sin_mean) * (bfp_y - y_mean)) /
                  np.sum((sin_object - sin_mean) ** 2))
    intercept = float(y_mean - slope * sin_mean)
    fitted = slope * sin_object + intercept
    bfp_residual = bfp_y - fitted
    ss_res = float(np.sum(bfp_residual ** 2))
    ss_tot = float(np.sum((bfp_y - np.mean(bfp_y)) ** 2))

    fan_rows = []
    for values in zip(py, sin_object, sin_image, bfp_y, fitted,
                      bfp_residual, sine_residual):
        fan_rows.append(dict(
            normalized_pupil_y=values[0], sin_object=values[1],
            sin_image=values[2], fourier_y_mm=values[3],
            fourier_fit_mm=values[4], fourier_residual_um=1000 * values[5],
            abbe_sine_residual=values[6]))
    with (out / "real_ray_abbe_and_fourier_fan.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fan_rows[0]))
        writer.writeheader()
        writer.writerows(fan_rows)

    order_rows = []
    for period_um in (2.0, 3.0, 4.0, 5.0):
        for order in range(-8, 9):
            sine = order * WAVELENGTH_UM / period_um
            if abs(sine) <= 0.6000001:
                coordinate = slope * sine + intercept
                order_rows.append(dict(
                    period_um=period_um, order=order, sin_theta=sine,
                    spatial_frequency_cycles_per_mm=order / (period_um * 1e-3),
                    predicted_fourier_y_mm=coordinate,
                    adjacent_order_spacing_mm=abs(slope) * WAVELENGTH_UM / period_um))
    with (out / "real_objective_frequency_order_positions.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(order_rows[0]))
        writer.writeheader()
        writer.writerows(order_rows)

    result = {
        "abbe_sine": {
            "equation": "sin(u_object) + 20*sin(u_image) = 0",
            "rms_dimensionless": float(np.sqrt(np.mean(sine_residual ** 2))),
            "max_abs_dimensionless": float(np.max(np.abs(sine_residual))),
            "equivalent_fsin_rms_um_at_30mm": float(
                1000 * F_OBJ_NOMINAL_MM * np.sqrt(np.mean(sine_residual ** 2))),
            "equivalent_fsin_max_um_at_30mm": float(
                1000 * F_OBJ_NOMINAL_MM * np.max(np.abs(sine_residual))),
        },
        "fourier_linearity": {
            "equation": "Y_BFP = slope*sin(u_object) + intercept",
            "slope_mm": float(slope),
            "effective_sine_focal_length_mm": float(abs(slope)),
            "intercept_mm": float(intercept),
            "r_squared": float(1.0 - ss_res / ss_tot),
            "rms_residual_um": float(1000 * np.sqrt(np.mean(bfp_residual ** 2))),
            "max_abs_residual_um": float(1000 * np.max(np.abs(bfp_residual))),
            "two_um_adjacent_order_spacing_mm": float(
                abs(slope) * WAVELENGTH_UM / 2.0),
        },
    }
    return result, float(slope), float(intercept)


def install_udas(zos, out: Path, slope: float, intercept: float):
    spacing = abs(slope) * WAVELENGTH_UM / 2.0
    # The fitted slope can be negative depending on the ray-coordinate sign.
    # Only physical peak centres matter; symmetric normal illumination is used.
    centres = {order: intercept + order * spacing for order in range(-3, 4)}
    definitions = {
        "object_regular": object_polygons(False),
        "object_particle": object_polygons(True),
        "open": pass_polygons([], 0.0),
        "deck_a": pass_polygons([centres[o] for o in ORDERS_A], DEPTH_A_MM),
        "deck_b": pass_polygons([centres[o] for o in ORDERS_B], DEPTH_B_MM),
    }
    destination = Path(zos.application.ObjectsDir) / "Apertures"
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, polygons in definitions.items():
        filename = f"UVREAL_{key.upper()}.UDA"
        source = out / "uda" / filename
        write_uda(source, polygons, f"UV real objective {key}")
        shutil.copy2(source, destination / filename)
        result[key] = filename
    zos.application.UpdateFileLists()
    return result, centres


def build_variant(zos, out: Path, udas: dict, defective: bool, filtered: bool,
                  grid_size: int):
    system = zos.system
    system.LoadFile(str(BASE_MODEL), False)
    lde = system.LDE

    # Add an amplitude-object screen exactly at the original object plane.
    working_distance = float(lde.GetSurfaceAt(0).Thickness)
    lde.InsertNewSurfaceAt(1)
    lde.GetSurfaceAt(0).Thickness = 0.0
    screen = lde.GetSurfaceAt(1)
    screen.Thickness = working_distance
    screen.Comment = "2 um Y-PERIODIC OBJECT + OPTIONAL PARTICLE"
    screen.SemiDiameter = 0.30

    # After the object-screen insertion the physical stop is S12.  Add two
    # zero-power coplanar mask planes immediately before it; the original
    # circular stop remains S14 and therefore retains the true 36 mm pupil.
    lde.InsertNewSurfaceAt(12)
    deck_a = lde.GetSurfaceAt(12)
    deck_a.Thickness = 0.0
    deck_a.SemiDiameter = 22.0
    deck_a.Comment = "FOURIER DECK A: m=-2,0,+2"
    lde.InsertNewSurfaceAt(13)
    deck_b = lde.GetSurfaceAt(13)
    deck_b.Thickness = 0.0
    deck_b.SemiDiameter = 22.0
    deck_b.Comment = "FOURIER DECK B: m=-3,-1,+1,+3"

    set_user_aperture(zos, screen,
                      udas["object_particle" if defective else "object_regular"])
    set_user_aperture(zos, deck_a, udas["deck_a" if filtered else "open"])
    set_user_aperture(zos, deck_b, udas["deck_b" if filtered else "open"])
    set_circle(zos, lde.GetSurfaceAt(14), PUPIL_RADIUS_MM)
    sampling_code = int(round(math.log2(grid_size))) - 5  # 32->0, ..., 2048->6
    for number in range(1, lde.NumberOfSurfaces):
        data = lde.GetSurfaceAt(number).PhysicalOpticsData
        # Explicit per-surface windows are required for this high-NA finite-
        # conjugate objective.  Leaving every surface at the Zemax 1 mm default
        # crops the +/-5.166 mm grating orders before the 36 mm BFP.
        data.DoNotRescaleBeamSizeUsingRayData = True
        data.AutoResample = False
        data.ResampleAfterRefraction = True
        data.XSampling = sampling_code
        data.YSampling = sampling_code
        if number == 1:
            width = POP_START_WIDTH_MM
        elif number in (12, 13, 14, 15, 16, 17):
            width = 40.0
        elif number == 18:
            width = 3.0
        else:
            width = max(1.0, 2.05 * float(lde.GetSurfaceAt(number).SemiDiameter))
        data.XWidth = width
        data.YWidth = width

    state = "particle" if defective else "regular"
    mode = "filtered" if filtered else "open"
    system.SystemData.TitleNotes.Title = f"UV real objective POP: {state}, {mode}"
    system.SystemData.TitleNotes.Notes = (
        "Native POP through six F_SILICA/Corning-7980-equivalent elements and "
        "ideal 600 mm tube.  Horizontal 2 um pattern; rods long in X and centred in Y.")
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"uvreal_{state}_{mode}.zos"
    system.SaveAs(str(path))
    return path


def net_grid(grid):
    nx, ny = int(grid.Nx), int(grid.Ny)
    values = np.asarray(list(grid.Values), dtype=float).reshape((ny, nx))
    x = float(grid.MinX) + np.arange(nx) * float(grid.Dx)
    y = float(grid.MinY) + np.arange(ny) * float(grid.Dy)
    return x, y, values


def run_pop(zos, path: Path, end_surface: int, grid_size: int):
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


def calculate_metrics(data):
    x, y, regular_open = data["regular_open_image"]
    _, _, particle_open = data["particle_open_image"]
    xf, yf, regular_filtered = data["regular_filtered_image"]
    xpf, ypf, particle_filtered = data["particle_filtered_image"]
    regular_filtered = resample(xf, yf, regular_filtered, x, y)
    particle_filtered = resample(xpf, ypf, particle_filtered, x, y)
    xx, yy = np.meshgrid(x, y)
    tdi_roi = ((np.abs(xx) <= TDI_ACTIVE_WIDTH_MM / 2) &
               (np.abs(yy) <= 0.18))
    particle_centre_y = -MAG_IMAGE * OBJECT_PERIOD_MM / 2
    particle_roi = (tdi_roi &
                    (xx ** 2 + (yy - particle_centre_y) ** 2 <= 0.018 ** 2))
    pattern_roi = tdi_roi & ~particle_roi
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    open_background_power = float(np.sum(regular_open[tdi_roi]) * dx * dy)
    filtered_background_power = float(np.sum(regular_filtered[tdi_roi]) * dx * dy)
    open_particle_peak = float(np.max(particle_open[particle_roi]))
    filtered_particle_peak = float(np.max(particle_filtered[particle_roi]))
    open_pattern95 = float(np.percentile(particle_open[pattern_roi], 95))
    filtered_pattern95 = float(np.percentile(particle_filtered[pattern_roi], 95))
    return {
        "evaluation": "raw native irradiance; no periodic-background subtraction",
        "tdi_active_width_mm": TDI_ACTIVE_WIDTH_MM,
        "background_suppression_db": float(10 * np.log10(
            max(open_background_power, 1e-30) / max(filtered_background_power, 1e-30))),
        "particle_peak_retention_percent": float(
            100 * filtered_particle_peak / max(open_particle_peak, 1e-30)),
        "particle_to_pattern_open_db": float(10 * np.log10(
            max(open_particle_peak, 1e-30) / max(open_pattern95, 1e-30))),
        "particle_to_pattern_filtered_db": float(10 * np.log10(
            max(filtered_particle_peak, 1e-30) / max(filtered_pattern95, 1e-30))),
        "common_unfiltered_peak_W_per_mm2": float(
            max(np.max(regular_open), np.max(particle_open))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grid", type=int, choices=(512, 1024, 2048), default=1024)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    messages = {}
    with ZOSApplication() as zos:
        print(f"OpticStudio={zos.application.OpticStudioVersion}", flush=True)
        ray_result, slope, intercept = ray_validation(zos, out)
        print(json.dumps(ray_result, indent=2), flush=True)
        udas, centres = install_udas(zos, out, slope, intercept)
        models = {}
        for defective in (False, True):
            for filtered in (False, True):
                key = ("particle" if defective else "regular") + "_" + (
                    "filtered" if filtered else "open")
                models[key] = build_variant(
                    zos, out, udas, defective, filtered, args.grid)
                print(f"built={models[key]}", flush=True)
        jobs = [
            ("regular_open_image", models["regular_open"], 18),
            ("particle_open_image", models["particle_open"], 18),
            ("regular_filtered_image", models["regular_filtered"], 18),
            ("particle_filtered_image", models["particle_filtered"], 18),
            ("particle_open_fourier", models["particle_open"], 12),
            ("particle_filtered_fourier", models["particle_filtered"], 14),
        ]
        for index, (key, path, end_surface) in enumerate(jobs, 1):
            print(f"POP {index}/{len(jobs)} {key}", flush=True)
            x, y, irradiance, pop_messages = run_pop(
                zos, path, end_surface, args.grid)
            data[key] = (x, y, irradiance)
            messages[key] = pop_messages
            power = float(np.sum(np.maximum(irradiance, 0)) *
                          abs(x[1] - x[0]) * abs(y[1] - y[0]))
            print(f"grid={len(x)} span={x[-1]-x[0]:.6g}x{y[-1]-y[0]:.6g} mm "
                  f"power={power:.6g} W messages={pop_messages}", flush=True)

    np.savez_compressed(
        out / "native_real_objective_pop_grids.npz",
        **{f"{key}_x": value[0] for key, value in data.items()},
        **{f"{key}_y": value[1] for key, value in data.items()},
        **{f"{key}_irradiance": value[2] for key, value in data.items()},
    )
    metrics = calculate_metrics(data)
    summary = {
        "engine": "Local OpticStudio 2022 R2.02 native real-ray + POP",
        "base_model": str(BASE_MODEL),
        "material": "six F_SILICA surfaces groups, Corning HPFS 7980 equivalent at 355 nm",
        "ray_validation": ray_result,
        "filter": {
            "coordinates": {
                "X": "TDI scan; rod long axis; serration repetition",
                "Y": "camera long edge; illumination line; sample orders; rod positioning",
            },
            "object_period_um": 2.0,
            "bright_width_um": BRIGHT_WIDTH_MM * 1000,
            "dark_width_um": (OBJECT_PERIOD_MM - BRIGHT_WIDTH_MM) * 1000,
            "particle": "one irregular approximately 0.6 um bright clear particle in a dark gap",
            "rod_core_mm": CORE_MM,
            "one_sided_serration_depth_mm": DEPTH_A_MM,
            "tooth_pitch_along_x_mm": TOOTH_PITCH_MM,
            "orders_A": list(ORDERS_A),
            "orders_B": list(ORDERS_B),
            "rod_centres_y_mm": {str(key): value for key, value in centres.items()},
            "true_circular_pupil_diameter_mm": 36.0,
            "tdi_stages": TDI_STAGES,
            "tdi_pixel_pitch_um": TDI_PIXEL_PITCH_MM * 1000,
        },
        "pop": {
            "grid": args.grid,
            "start_window_mm": POP_START_WIDTH_MM,
            "messages": messages,
            "metrics": metrics,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
