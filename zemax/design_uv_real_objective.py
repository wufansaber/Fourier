"""Design and validate a real-curvature fused-silica 355 nm microscope objective.

The objective is initialized from the reversed geometry family of the supplied
10X objective, scaled to 30 mm focal length.  All powered objective elements
are replaced by Corning HPFS 7980-equivalent fused silica (`F_SILICA` in the
local Zemax catalog) and re-optimized at 355 nm.  A 600 mm
Paraxial Lens is used only as the permitted ideal tube lens.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from zosapi_utils import ZOSApplication


ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = ROOT / "results" / "real_objective" / "source_audit.json"
OUT_DEFAULT = ROOT / "results" / "uv_real_objective"
WAVE_UM = 0.355
F_OBJ_MM = 30.0
F_TUBE_MM = 600.0
MAG_TARGET = -20.0
# Signed Abbe sine scale after the objective is collimated and the image is
# focused by the ideal tube lens.
SINE_MAG_TARGET = -20.0
OBJECT_NA = 0.60
OBJECT_FIELD_MM = 0.25
START_SCALE = 1.681
WORKING_DISTANCE_MM = 4.79
TUBE_GAP_MM = 50.0
OBJECTIVE_LAST_SURFACE = 17
TUBE_SURFACE = 18
IMAGE_SURFACE = 19
STOP_SURFACE = 14


def add_operand(zos, name, params=(), target=0.0, weight=0.0):
    api = zos.ZOSAPI
    row = zos.system.MFE.AddOperand()
    row.ChangeType(getattr(api.Editors.MFE.MeritOperandType, name))
    for index, value in enumerate(params, 1):
        cell = row.GetOperandCell(
            getattr(api.Editors.MFE.MeritColumn, f"Param{index}"))
        if cell.DataType == api.Editors.CellDataType.Integer:
            cell.IntegerValue = int(value)
        else:
            cell.DoubleValue = float(value)
    row.Target = float(target)
    row.Weight = float(weight)
    return row.OperandNumber


def set_variable(api, cell) -> None:
    cell.SetSolveData(cell.CreateSolveType(api.Editors.SolveType.Variable))


def set_paraxial(zos, surface, focal_mm: float) -> None:
    api = zos.ZOSAPI
    surface.ChangeType(surface.GetSurfaceTypeSettings(api.Editors.LDE.SurfaceType.Paraxial))
    surface.GetSurfaceCell(api.Editors.LDE.SurfaceColumn.Par1).DoubleValue = focal_mm


def build_system(zos: ZOSApplication, out: Path) -> Path:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    source = audit["surfaces"]
    system, api = zos.system, zos.ZOSAPI
    system.New(False)
    system.SystemData.TitleNotes.Title = "355 nm Corning 7980 silica objective f=30 mm NA=0.6 + ideal tube"
    system.SystemData.TitleNotes.Notes = (
        "Real-curvature UV objective. All objective glass volumes use F_SILICA, "
        "the local catalog equivalent of Corning HPFS 7980 at 355 nm. "
        "Only the 600 mm tube lens is Paraxial. Coatings are intentionally omitted.")
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = WAVE_UM
    system.SystemData.Aperture.ApertureType = api.SystemData.ZemaxApertureType.ObjectSpaceNA
    system.SystemData.Aperture.ApertureValue = OBJECT_NA
    fields = system.SystemData.Fields
    fields.SetFieldType(api.SystemData.FieldType.ObjectHeight)
    fields.GetField(1).X = 0.0
    fields.GetField(1).Y = 0.0
    fields.AddField(0.0, OBJECT_FIELD_MM / 2, 1.0)
    fields.AddField(0.0, OBJECT_FIELD_MM, 1.0)

    while system.LDE.NumberOfSurfaces < IMAGE_SURFACE + 1:
        system.LDE.InsertNewSurfaceAt(system.LDE.NumberOfSurfaces - 1)
    lde = system.LDE
    lde.GetSurfaceAt(0).Thickness = WORKING_DISTANCE_MM
    lde.GetSurfaceAt(0).Comment = "OBJECT: 0.5 mm full field in air"

    # Powered portion of the supplied objective family, reversed and scaled.
    # Original glasses are intentionally replaced by UV fused silica.
    for new_surface, old_surface in enumerate(range(18, 1, -1), 1):
        surface = lde.GetSurfaceAt(new_surface)
        old_radius = float(source[old_surface]["radius_mm"])
        surface.Radius = -old_radius * START_SCALE
        if old_surface > 2:
            surface.Thickness = float(source[old_surface - 1]["thickness_mm"]) * START_SCALE
        else:
            surface.Thickness = TUBE_GAP_MM
        surface.Material = "F_SILICA" if source[old_surface - 1]["material"] else ""
        surface.SemiDiameter = 22.0
        surface.Comment = f"UV OBJECTIVE real surface {new_surface}; seed reverse S{old_surface}"
    lde.GetSurfaceAt(STOP_SURFACE).IsStop = True
    lde.GetSurfaceAt(STOP_SURFACE).SemiDiameter = 18.0
    lde.GetSurfaceAt(OBJECTIVE_LAST_SURFACE).Thickness = TUBE_GAP_MM

    tube = lde.GetSurfaceAt(TUBE_SURFACE)
    tube.Comment = "IDEAL TUBE LENS f=600 mm (permitted ideal element)"
    tube.Thickness = F_TUBE_MM
    tube.SemiDiameter = 30.0
    set_paraxial(zos, tube, F_TUBE_MM)
    image = lde.GetSurfaceAt(IMAGE_SURFACE)
    image.Comment = "20X IMAGE / TDI PLANE"
    image.SemiDiameter = 7.0

    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "uv355_real_objective_initial.zos"
    system.SaveAs(str(path))
    return path


def build_merit_function(zos: ZOSApplication, enable_conics: bool) -> None:
    system, api = zos.system, zos.ZOSAPI
    fixed_radius_surfaces = (14, 16)
    for surface_number in range(1, OBJECTIVE_LAST_SURFACE + 1):
        surface = system.LDE.GetSurfaceAt(surface_number)
        if (surface_number not in fixed_radius_surfaces and
                math.isfinite(float(surface.Radius))):
            set_variable(api, surface.RadiusCell)
    set_variable(api, system.LDE.GetSurfaceAt(0).ThicknessCell)
    conic_surfaces = (1, 2, 5, 6, 10, 11, 17)
    if enable_conics:
        for surface_number in conic_surfaces:
            set_variable(api, system.LDE.GetSurfaceAt(surface_number).ConicCell)

    mfe = system.MFE
    if mfe.NumberOfOperands:
        mfe.RemoveOperandsAt(1, mfe.NumberOfOperands)

    pupil_samples = [(0.0, 0.0)]
    for radius in (0.25, 0.50, 0.75, 0.95):
        for azimuth in range(8):
            angle = azimuth * math.pi / 4
            pupil_samples.append((radius * math.cos(angle), radius * math.sin(angle)))
    for hy in (0.0, 0.5, 1.0):
        target_y = MAG_TARGET * OBJECT_FIELD_MM * hy
        for px, py in pupil_samples:
            add_operand(zos, "REAX", (IMAGE_SURFACE, 1, 0.0, hy, px, py),
                        target=0.0, weight=25.0)
            add_operand(zos, "REAY", (IMAGE_SURFACE, 1, 0.0, hy, px, py),
                        target=target_y, weight=25.0)
            add_operand(zos, "OPDX", (0, 1, 0.0, hy, px, py),
                        target=0.0, weight=5.0)

    # Explicit axial finite-conjugate Abbe sine condition:
    # sin(u_object) - M_sine*sin(u_image) = 0, with M_sine=-20.
    for py in (-0.95, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 0.95):
        object_sine = add_operand(zos, "REAB", (0, 1, 0.0, 0.0, 0.0, py))
        image_sine = add_operand(
            zos, "REAB", (IMAGE_SURFACE, 1, 0.0, 0.0, 0.0, py))
        scaled_image_sine = add_operand(zos, "PROB", (image_sine, 0, SINE_MAG_TARGET))
        add_operand(zos, "DIFF", (object_sine, scaled_image_sine),
                    target=0.0, weight=20000.0)

    if enable_conics:
        for surface_number in conic_surfaces:
            add_operand(zos, "COGT", (surface_number,), target=-8.0, weight=50.0)
            add_operand(zos, "COLT", (surface_number,), target=0.5, weight=50.0)


def optimize(zos: ZOSApplication, cycles, cores=4) -> tuple[float, float]:
    system, api = zos.system, zos.ZOSAPI
    before = float(system.MFE.CalculateMeritFunction())
    optimizer = system.Tools.OpenLocalOptimization()
    try:
        optimizer.Algorithm = api.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
        optimizer.Cycles = cycles
        optimizer.NumberOfCores = cores
        optimizer.RunAndWaitForCompletion()
    finally:
        optimizer.Close()
    after = float(system.MFE.CalculateMeritFunction())
    return before, after


def batch(zos: ZOSApplication, surface: int, samples, hy=0.0):
    tool = zos.system.Tools.OpenBatchRayTrace()
    try:
        data = zos.ZOSAPI.Tools.RayTrace.IRayTraceNormUnpolData(
            tool.CreateNormUnpol(len(samples), zos.ZOSAPI.Tools.RayTrace.RaysType.Real, surface))
        opd_mode = (zos.ZOSAPI.Tools.RayTrace.OPDMode.CurrentAndChief
                    if surface == IMAGE_SURFACE
                    else getattr(zos.ZOSAPI.Tools.RayTrace.OPDMode, "None"))
        for px, py in samples:
            data.AddRay(1, 0.0, hy, float(px), float(py), opd_mode)
        tool.RunAndWaitForCompletion()
        data.StartReadingResults()
        values = []
        while True:
            row = data.ReadNextResult(0, 0, 0, *([0.0] * 11))
            if not row[0]:
                break
            values.append([float(value) for value in row[1:]])
        return np.asarray(values)
    finally:
        tool.Close()


def validate(zos: ZOSApplication, out: Path, optimization_history) -> dict:
    system, api = zos.system, zos.ZOSAPI
    grid = np.linspace(-1.0, 1.0, 65)
    samples = [(float(px), float(py)) for py in grid for px in grid
               if px * px + py * py <= 1.0]
    fields = []
    ray_dir = out / "rays"
    ray_dir.mkdir(exist_ok=True)
    for hy in (0.0, 0.5, 1.0):
        image = batch(zos, IMAGE_SURFACE, samples, hy)
        # Batch rows after the success flag are:
        # ray_number,error,vignette,x,y,z,l,m,n,ln,mn,nn,opd,intensity.
        valid = (image[:, 1] == 0) & (image[:, 2] == 0)
        x = image[valid, 3]
        y = image[valid, 4]
        opd = image[valid, 12]
        fields.append({
            "field_hy": hy,
            "rays_total": len(samples),
            "rays_valid": int(valid.sum()),
            "chief_target_y_mm": MAG_TARGET * OBJECT_FIELD_MM * hy,
            "centroid_x_mm": float(np.mean(x)),
            "centroid_y_mm": float(np.mean(y)),
            "rms_spot_um": float(1000 * np.sqrt(np.mean((x - np.mean(x)) ** 2 +
                                                        (y - np.mean(y)) ** 2))),
            "opd_rms_waves": float(np.std(opd)),
            "opd_pv_waves": float(np.ptp(opd)),
        })
        np.savez_compressed(ray_dir / f"field_{hy:g}.npz", samples=np.asarray(samples),
                            image=image)

    meridional = [(0.0, float(py)) for py in np.linspace(-1.0, 1.0, 401)]
    object_rays = batch(zos, 0, meridional, 0.0)
    image_rays = batch(zos, IMAGE_SURFACE, meridional, 0.0)
    valid = ((object_rays[:, 1] == 0) & (object_rays[:, 2] == 0) &
             (image_rays[:, 1] == 0) & (image_rays[:, 2] == 0))
    sin_object = object_rays[valid, 7]
    sin_image = image_rays[valid, 7]
    py_values = np.linspace(-1.0, 1.0, 401)[valid]
    residual_nominal = sin_object - SINE_MAG_TARGET * sin_image
    best_magnification = float(np.dot(sin_image, sin_object) /
                               max(np.dot(sin_image, sin_image), 1e-30))
    residual_best = sin_object - best_magnification * sin_image
    sine_rows = [
        {"normalized_pupil_y": float(py), "sin_object": float(so),
         "sin_image": float(si), "residual_nominal_sine_scale_minus20": float(rn),
         "residual_best_fit": float(rb)}
        for py, so, si, rn, rb in zip(py_values, sin_object, sin_image,
                                      residual_nominal, residual_best)
    ]
    with (out / "abbe_sine_fan.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sine_rows[0]))
        writer.writeheader()
        writer.writerows(sine_rows)

    surface_rows = []
    for number in range(system.LDE.NumberOfSurfaces):
        surface = system.LDE.GetSurfaceAt(number)
        surface_rows.append({
            "surface": number,
            "type": str(surface.TypeName),
            "radius_mm": float(surface.Radius),
            "thickness_mm": float(surface.Thickness),
            "material": str(surface.Material),
            "semi_diameter_mm": float(surface.SemiDiameter),
            "conic": float(surface.Conic),
            "comment": str(surface.Comment),
        })
    with (out / "surface_prescription.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(surface_rows[0]))
        writer.writeheader()
        writer.writerows(surface_rows)
    (out / "surface_prescription.json").write_text(
        json.dumps(surface_rows, indent=2), encoding="utf-8")

    first_order = {}
    operands = api.Editors.MFE.MeritOperandType
    for name in ("EFFL", "ENPP", "EXPP", "EPDI", "EXPD", "ISNA", "PMAG"):
        try:
            first_order[name] = float(system.MFE.GetOperandValue(
                getattr(operands, name), 0, 1, 0, 0, 0, 0, 0, 0))
        except Exception as exc:
            first_order[name] = {"error": str(exc)}

    summary = {
        "engine": f"Local OpticStudio {zos.application.OpticStudioVersion}",
        "wavelength_nm": 355.0,
        "objective_material": "Corning HPFS 7980 equivalent; Zemax F_SILICA",
        "objective_index_at_355nm": 1.4760718797565597,
        "objective_surfaces": OBJECTIVE_LAST_SURFACE,
        "tube_lens": "ideal Paraxial Lens, f=600 mm",
        "coatings": "not modeled by request",
        "target": {"f_objective_mm": F_OBJ_MM, "object_na": OBJECT_NA,
                   "full_object_field_mm": 2 * OBJECT_FIELD_MM,
                   "magnification": MAG_TARGET},
        "first_order_operands": first_order,
        "field_validation": fields,
        "abbe_sine": {
            "definition": "sin(u_object) - M*sin(u_image)",
            "nominal_sine_scale": SINE_MAG_TARGET,
            "best_fit_magnification": best_magnification,
            "valid_meridional_rays": int(valid.sum()),
            "nominal_residual_rms": float(np.sqrt(np.mean(residual_nominal ** 2))),
            "nominal_residual_max_abs": float(np.max(np.abs(residual_nominal))),
            "nominal_equivalent_fsin_rms_um": float(
                1000 * F_OBJ_MM * np.sqrt(np.mean(residual_nominal ** 2))),
            "nominal_equivalent_fsin_max_um": float(
                1000 * F_OBJ_MM * np.max(np.abs(residual_nominal))),
            "best_fit_residual_rms": float(np.sqrt(np.mean(residual_best ** 2))),
            "best_fit_residual_max_abs": float(np.max(np.abs(residual_best))),
            "best_fit_equivalent_fsin_rms_um": float(
                1000 * F_OBJ_MM * np.sqrt(np.mean(residual_best ** 2))),
            "best_fit_equivalent_fsin_max_um": float(
                1000 * F_OBJ_MM * np.max(np.abs(residual_best))),
        },
        "optimization": optimization_history,
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--skip-optimize", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ZOSApplication() as zos:
        initial = build_system(zos, out)
        print(f"initial={initial}", flush=True)
        history = []
        if not args.skip_optimize:
            build_merit_function(zos, enable_conics=False)
            before, after = optimize(
                zos, zos.ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_50_Cycles)
            history.append({"stage": "real radii", "merit_before": before,
                            "merit_after": after})
            print("stage1", history[-1], flush=True)
            build_merit_function(zos, enable_conics=True)
            before, after = optimize(
                zos, zos.ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_50_Cycles)
            history.append({"stage": "radii plus selected conics", "merit_before": before,
                            "merit_after": after})
            print("stage2", history[-1], flush=True)
        final_path = out / "models" / "uv355_real_objective_ideal_tube_optimized.zos"
        zos.system.SaveAs(str(final_path))
        zos.system.SaveAs(str(final_path.with_suffix(".zmx")))
        summary = validate(zos, out, history)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
