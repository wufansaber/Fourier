"""Run a native Zemax POP proof for the current dual-deck equivalent pupil.

This model uses ideal paraxial lenses at 355 nm, f_obj=30 mm, NA=0.6 and
f_tube=600 mm.  The physical Y-magnified hardware is mapped back to the
original objective BFP: transverse Y dimensions are divided by 2.8169 and the
4 mm physical deck gap is divided by M_y**2.  It validates the equivalent
pupil filtering chain, not a manufactured anamorphic relay prescription.
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
OUT_DEFAULT = ROOT / "results" / "dual_deck_zemax_equivalent"

WAVELENGTH_UM = 0.355
WAVELENGTH_MM = WAVELENGTH_UM * 1e-3
F_OBJ_MM = 30.0
F_TUBE_MM = 600.0
NA = 0.60
MAG_IMAGE = 20.0
OBJECT_FOV_MM = 0.50
OBJECT_PITCH_MM = 0.020
OBJECT_BAR_WIDTH_MM = 0.007
MY = 2.816901408450704
MX = 1.0
HARDWARE_ORDER_SPACING_MM = 1.5
BFP_ORDER_SPACING_MM = HARDWARE_ORDER_SPACING_MM / MY
CORE_HARDWARE_MM = 1.0
DEPTH_HARDWARE_MM = 0.5
CORE_BFP_MM = CORE_HARDWARE_MM / MY
DEPTH_BFP_MM = DEPTH_HARDWARE_MM / MY
TOOTH_PITCH_X_MM = 0.75 / MX
HARDWARE_DECK_GAP_MM = 4.0
BFP_DECK_GAP_MM = HARDWARE_DECK_GAP_MM / MY**2
# OpticStudio 2022 returns one quarter of this value as the POP full span.
# A 40 mm setting therefore supplies a 10 mm guard-band window at every
# surface.  This is required to include the seven equivalent-BFP rod centres
# through +/-1.5975 mm and the 20x image of the 0.5 mm object field.
POP_START_WIDTH_SETTING_MM = 40.0


def rectangle(x0, x1, y0, y1):
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def write_uda(path: Path, polygons) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("! Current dual-deck equivalent-pupil POP proof\n")
        for polygon in polygons:
            for x, y in polygon:
                stream.write(f"{x:.12g} {y:.12g}\n")
            stream.write("0 0\n")


def object_polygons(defective: bool):
    half = OBJECT_FOV_MM / 2
    centers = np.arange(-0.24, 0.2400001, OBJECT_PITCH_MM)
    polygons = []
    for centre in centers:
        y0 = float(centre - OBJECT_BAR_WIDTH_MM / 2)
        y1 = float(centre + OBJECT_BAR_WIDTH_MM / 2)
        if defective and abs(centre) < 1e-12:
            polygons.append(rectangle(-half, -0.050, y0, y1))
            polygons.append(rectangle(0.050, half, y0, y1))
            shifted = centre + 0.006
            polygons.append(
                rectangle(-0.050, 0.050, shifted - OBJECT_BAR_WIDTH_MM / 2, shifted + OBJECT_BAR_WIDTH_MM / 2)
            )
        else:
            polygons.append(rectangle(-half, half, y0, y1))
    return polygons


def triangle_fraction(x_mm: float) -> float:
    phase = x_mm / TOOTH_PITCH_X_MM
    tri = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
    return 0.5 * (tri + 1.0)


def half_height(x_mm: float) -> float:
    return CORE_BFP_MM / 2 + DEPTH_BFP_MM * triangle_fraction(x_mm)


def pass_polygons(orders):
    outer_x = 20.0
    outer_y = 20.0
    if not orders:
        return [rectangle(-outer_x, outer_x, -outer_y, outer_y)]
    extrema = np.arange(-outer_x, outer_x + TOOTH_PITCH_X_MM / 2, TOOTH_PITCH_X_MM / 2)
    xs = sorted(set([-outer_x, outer_x] + [float(v) for v in extrema if -outer_x < v < outer_x]))
    centres = [order * BFP_ORDER_SPACING_MM for order in sorted(orders)]

    def lower(c, x):
        return c - half_height(x)

    def upper(c, x):
        return c + half_height(x)

    polygons = []
    first = centres[0]
    polygons.append([(x, -outer_y) for x in xs] + [(x, lower(first, x)) for x in reversed(xs)])
    for a, b in zip(centres[:-1], centres[1:]):
        polygons.append([(x, upper(a, x)) for x in xs] + [(x, lower(b, x)) for x in reversed(xs)])
    last = centres[-1]
    polygons.append([(x, upper(last, x)) for x in xs] + [(x, outer_y) for x in reversed(xs)])
    return polygons


def install_udas(zos: ZOSApplication, out: Path):
    uda = out / "uda"
    definitions = {
        "object_regular": object_polygons(False),
        "object_defect": object_polygons(True),
        "open": pass_polygons([]),
        "a_balanced": pass_polygons([-2, 0, 2]),
        "b_balanced": pass_polygons([-3, -1, 1, 3]),
        "a_all9": pass_polygons([-4, -2, 0, 2, 4]),
        "b_all9": pass_polygons([-3, -1, 1, 3]),
    }
    destination = Path(zos.application.ObjectsDir) / "Apertures"
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, polygons in definitions.items():
        filename = f"DUALDECK_{key.upper()}.UDA"
        source = uda / filename
        write_uda(source, polygons)
        shutil.copy2(source, destination / filename)
        result[key] = filename
    zos.application.UpdateFileLists()
    return result


def set_user_aperture(zos, surface, filename):
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(api.Editors.LDE.SurfaceApertureTypes.UserAperture)
    data._S_UserAperture.ApertureFile = filename
    data._S_UserAperture.UDAScale = 1.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_circle(zos, surface, radius):
    api = zos.ZOSAPI
    data = surface.ApertureData.CreateApertureTypeSettings(api.Editors.LDE.SurfaceApertureTypes.CircularAperture)
    data._S_CircularAperture.MaximumRadius = radius
    data._S_CircularAperture.MinimumRadius = 0.0
    surface.ApertureData.ChangeApertureTypeSettings(data)


def set_paraxial(zos, surface, focal):
    api = zos.ZOSAPI
    surface.ChangeType(surface.GetSurfaceTypeSettings(api.Editors.LDE.SurfaceType.Paraxial))
    surface.GetSurfaceCell(api.Editors.LDE.SurfaceColumn.Par1).DoubleValue = focal


def build_model(zos, out: Path, udas, defective: bool, mode: str) -> Path:
    system = zos.system
    system.New(False)
    system.SystemData.TitleNotes.Title = f"Dual-deck equivalent POP: {'defect' if defective else 'regular'}, {mode}"
    system.SystemData.TitleNotes.Notes = (
        "355 nm ideal equivalent-pupil proof; f_obj=30 mm, NA=0.6, f_tube=600 mm, 20x; "
        "Y hardware magnification=2.8169; not a real anamorphic relay prescription."
    )
    system.SystemData.Aperture.ApertureValue = 2 * F_OBJ_MM * NA
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = WAVELENGTH_UM
    field = system.SystemData.Fields.GetField(1)
    field.X = 0.0
    field.Y = 0.0
    lde = system.LDE
    while lde.NumberOfSurfaces < 7:
        lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)
    zhalf = BFP_DECK_GAP_MM / 2
    surfaces = [
        (1, "PERIODIC OBJECT", "standard", None, F_OBJ_MM, 0.30),
        (2, "PARAXIAL OBJECTIVE f=30 mm, NA=0.6", "paraxial", F_OBJ_MM, F_OBJ_MM - zhalf, 18.0),
        (3, "DECK A equivalent at BFP-z/2", "standard", None, BFP_DECK_GAP_MM, 20.0),
        (4, "DECK B equivalent at BFP+z/2", "standard", None, F_TUBE_MM - zhalf, 20.0),
        (5, "PARAXIAL TUBE LENS f=600 mm", "paraxial", F_TUBE_MM, F_TUBE_MM, 20.0),
        (6, "20X IMAGE", "image", None, 0.0, 12.0),
    ]
    for number, comment, stype, focal, thickness, semi in surfaces:
        s = lde.GetSurfaceAt(number)
        s.Comment = comment
        s.Thickness = thickness
        s.SemiDiameter = semi
        if stype == "paraxial":
            set_paraxial(zos, s, focal)
    set_user_aperture(zos, lde.GetSurfaceAt(1), udas["object_defect" if defective else "object_regular"])
    set_circle(zos, lde.GetSurfaceAt(2), F_OBJ_MM * NA)
    if mode == "open":
        a_name = b_name = udas["open"]
    elif mode == "balanced7":
        a_name, b_name = udas["a_balanced"], udas["b_balanced"]
    elif mode == "all9":
        a_name, b_name = udas["a_all9"], udas["b_all9"]
    else:
        raise ValueError(mode)
    set_user_aperture(zos, lde.GetSurfaceAt(3), a_name)
    set_user_aperture(zos, lde.GetSurfaceAt(4), b_name)
    for number in range(1, 7):
        data = lde.GetSurfaceAt(number).PhysicalOpticsData
        data.DoNotRescaleBeamSizeUsingRayData = True
        data.AutoResample = False
        data.ResampleAfterRefraction = False
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"dual_{'defect' if defective else 'regular'}_{mode}.zos"
    system.SaveAs(str(path))
    return path


def net_grid(grid):
    nx, ny = int(grid.Nx), int(grid.Ny)
    values = np.asarray(list(grid.Values), dtype=float).reshape((ny, nx))
    x = float(grid.MinX) + np.arange(nx) * float(grid.Dx)
    y = float(grid.MinY) + np.arange(ny) * float(grid.Dy)
    return x, y, values


def run_pop(zos, path: Path, end_surface: int, sampling_code: int):
    system = zos.system
    api = zos.ZOSAPI
    system.LoadFile(str(path), False)
    pop = system.Analyses.New_Analysis(api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    settings = pop.GetSettings()
    settings.StartSurface.SetSurfaceNumber(1)
    settings.EndSurface.SetSurfaceNumber(end_surface)
    settings.Wavelength.SetWavelengthNumber(1)
    settings.Field.SetFieldNumber(1)
    settings.BeamType = 2
    settings.Update()
    settings.SetParameterValue(0, 0.55)
    settings.SetParameterValue(1, 0.55)
    for i in range(2, 6):
        settings.SetParameterValue(i, 0.0)
    settings.XSampling = sampling_code
    settings.YSampling = sampling_code
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
        raise RuntimeError(f"No POP grid: {path.name}, S{end_surface}, {messages}")
    x, y, a = net_grid(results.GetDataGrid(0))
    pop.Close()
    return x, y, a, messages


def grid_power(x, y, a):
    return float(np.maximum(a, 0).sum() * abs(x[1] - x[0]) * abs(y[1] - y[0]))


def resample(source_x, source_y, a, target_x, target_y):
    intermediate = np.vstack([np.interp(target_x, source_x, row, left=0, right=0) for row in a])
    return np.vstack([np.interp(target_y, source_y, intermediate[:, i], left=0, right=0) for i in range(len(target_x))]).T


def metrics(data, modes):
    x0, y0, regular0 = data["regular_open_s6"]
    xd0, yd0, defect0 = data["defect_open_s6"]
    defect0 = resample(xd0, yd0, defect0, x0, y0)
    base_power = grid_power(x0, y0, regular0)
    base_diff = float(np.sqrt(np.mean((defect0 - regular0) ** 2)))
    rows = []
    for mode in modes:
        x, y, regular = data[f"regular_{mode}_s6"]
        xd, yd, defect = data[f"defect_{mode}_s6"]
        defect = resample(xd, yd, defect, x, y)
        power = grid_power(x, y, regular)
        diff = float(np.sqrt(np.mean((defect - regular) ** 2)))
        rows.append({
            "mode": mode,
            "regular_power_W": power,
            "background_suppression_db": 10 * np.log10(max(base_power, 1e-30) / max(power, 1e-30)),
            "defect_difference_rms_W_per_mm2": diff,
            "defect_difference_retention_percent": 100 * diff / max(base_diff, 1e-30),
            "regular_peak_W_per_mm2": float(regular.max()),
            "defect_peak_W_per_mm2": float(defect.max()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--sampling-code", type=int, default=5)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    messages = {}
    with ZOSApplication() as zos:
        print(f"OpticStudio={zos.application.OpticStudioVersion}")
        udas = install_udas(zos, out)
        models = {}
        modes = ["open", "balanced7"]
        for defective in [False, True]:
            for mode in modes:
                key = f"{'defect' if defective else 'regular'}_{mode}"
                models[key] = build_model(zos, out, udas, defective, mode)
                print(f"built={models[key]}")
        jobs = []
        for mode in modes:
            jobs.append((f"regular_{mode}_s6", models[f"regular_{mode}"], 6))
            jobs.append((f"defect_{mode}_s6", models[f"defect_{mode}"], 6))
        for i, (key, path, end) in enumerate(jobs, 1):
            print(f"POP {i}/{len(jobs)} {key}", flush=True)
            x, y, a, msg = run_pop(zos, path, end, args.sampling_code)
            data[key] = (x, y, a)
            messages[key] = msg
            print(f" grid={len(x)}x{len(y)} span=({x[0]:.4g},{x[-1]:.4g})x({y[0]:.4g},{y[-1]:.4g}) power={grid_power(x,y,a):.6g}", flush=True)
    np.savez_compressed(
        out / "native_pop_grids.npz",
        **{f"{k}_x": v[0] for k, v in data.items()},
        **{f"{k}_y": v[1] for k, v in data.items()},
        **{f"{k}_irradiance": v[2] for k, v in data.items()},
    )
    rows = metrics(data, modes)
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "engine": "Local OpticStudio 2022 R2.02 native POP",
        "scope": "ideal equivalent pupil of current dual-deck scheme; not a real anamorphic relay prescription",
        "parameters": {
            "wavelength_nm": 355,
            "objective_focal_length_mm": F_OBJ_MM,
            "objective_na": NA,
            "tube_lens_focal_length_mm": F_TUBE_MM,
            "image_magnification": MAG_IMAGE,
            "hardware_y_pupil_magnification": MY,
            "hardware_order_spacing_mm": HARDWARE_ORDER_SPACING_MM,
            "equivalent_bfp_order_spacing_mm": BFP_ORDER_SPACING_MM,
            "hardware_deck_gap_mm": HARDWARE_DECK_GAP_MM,
            "equivalent_bfp_deck_gap_mm": BFP_DECK_GAP_MM,
            "core_hardware_mm": CORE_HARDWARE_MM,
            "tooth_depth_each_side_hardware_mm": DEPTH_HARDWARE_MM,
            "tooth_pitch_x_mm": TOOTH_PITCH_X_MM,
            "pop_full_window_mm": POP_START_WIDTH_SETTING_MM / 4,
            "pop_sampling_code": args.sampling_code,
        },
        "metrics": rows,
        "messages": messages,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
