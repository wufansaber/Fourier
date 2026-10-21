"""Real-glass Fourier relay based on the supplied 10X objective, no ideal lenses."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from zosapi_utils import ZOSApplication

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_objective"
WAVE_UM = 0.584601


def paraxial_from_audit(audit):
    indices = {r["surface"]: r["index"] for r in audit["material_indices"]}
    matrix = [[1., 0.], [0., 1.]]
    n_before = 1.0
    length = 0.0
    for s in audit["surfaces"][2:]:
        n_after = indices.get(s["surface"], 1.0)
        radius = float(s["radius_mm"])
        curvature = 0.0 if math.isinf(radius) else 1.0 / radius
        power = -(n_after - n_before) * curvature
        matrix[1] = [matrix[1][j] + power * matrix[0][j] for j in range(2)]
        n_before = n_after
        d = float(s["thickness_mm"]) if s["surface"] != 21 else 0.
        matrix[0] = [matrix[0][j] + d / n_after * matrix[1][j] for j in range(2)]
        length += d
    return {"matrix": matrix, "efl_mm": -1. / matrix[1][0],
            "collector_external_bfp_distance_mm": -matrix[1][1] / matrix[1][0],
            "length_mm": length, "determinant": matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]}


def ray(zos, to_surface, hx=0., hy=0., px=0., py=0., opd=False, wave=1):
    tool = zos.system.Tools.OpenBatchRayTrace()
    try:
        result = tool.SingleRayNormUnpol(
            zos.ZOSAPI.Tools.RayTrace.RaysType.Real, to_surface, wave,
            hx, hy, px, py, opd, 0, 0, *([0.] * 11))
        names = ("success", "error", "vignette", "x", "y", "z", "l", "m", "n",
                 "ln", "mn", "nn", "opd", "intensity")
        return {key: float(value) for key, value in zip(names, result)}
    finally:
        tool.Close()


def singlet_matrix(radius, thickness, index):
    p = (index - 1.) / radius
    return [[1. - thickness * p / index, thickness / index],
            [-2. * p + thickness * p * p / index, 1. - thickness * p / index]]


def create_real_system(zos, audit, output_path, bfp_offset):
    system, api = zos.system, zos.ZOSAPI
    system.New(False)
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = WAVE_UM
    system.SystemData.Aperture.ApertureType = api.SystemData.ZemaxApertureType.ObjectSpaceNA
    system.SystemData.Aperture.ApertureValue = 0.4998
    fields = system.SystemData.Fields
    fields.SetFieldType(api.SystemData.FieldType.ObjectHeight)
    fields.GetField(1).X = 0.
    fields.GetField(1).Y = 0.
    fields.AddField(0., 0.125, 1.)
    fields.AddField(0., 0.25, 1.)
    source = audit["surfaces"]
    specs = []
    # Real symmetric singlets; principal-plane locations used for conjugates.
    n_bk7 = 1.516923
    relay_matrix = singlet_matrix(103.0, 4.0, n_bk7)
    fr = -1. / relay_matrix[1][0]
    hr = (1. - relay_matrix[0][0]) * fr
    tube_matrix = singlet_matrix(206.0, 5.0, n_bk7)
    ft = -1. / tube_matrix[1][0]
    ht = (1. - tube_matrix[0][0]) * ft
    print("relay_f", fr, "tube_f", ft, "internal_bfp_offset", bfp_offset, flush=True)
    # Collect from sample through the original objective in reverse.
    for old in range(21, 1, -1):
        specs.append({"radius": -float(source[old]["radius_mm"]),
                      "thickness": float(source[old - 1]["thickness_mm"]) if old > 2 else bfp_offset + fr - hr,
                      "material": source[old - 1]["material"],
                      "semi": float(source[old]["semi_diameter_mm"]),
                      "comment": "COLLECT reverse source S%d" % old,
                      "stop": old == 5})
    for radius, d, mat, name in (
        (103., 4., "N-BK7", "PUPIL RELAY L1 FRONT"),
        (-103., 2. * (fr - hr), "", "PUPIL RELAY L1 BACK"),
        (103., 4., "N-BK7", "PUPIL RELAY L2 FRONT"),
        (-103., fr - hr, "", "PUPIL RELAY L2 BACK"),
        (math.inf, ft - ht, "", "ACCESSIBLE FOURIER PLANE / FILTER"),
        (206., 5., "N-BK7", "REAL TUBE LENS FRONT"),
        (-206., ft - ht, "", "REAL TUBE LENS BACK"),
    ):
        specs.append({"radius": radius, "thickness": d, "material": mat,
                      "semi": 16., "comment": name, "stop": False})
    while system.LDE.NumberOfSurfaces < len(specs) + 2:
        system.LDE.InsertNewSurfaceAt(system.LDE.NumberOfSurfaces - 1)
    system.LDE.GetSurfaceAt(0).Thickness = 0.
    for i, spec in enumerate(specs, 1):
        s = system.LDE.GetSurfaceAt(i)
        s.Radius = spec["radius"]
        s.Thickness = spec["thickness"]
        s.Material = spec["material"]
        s.SemiDiameter = spec["semi"]
        s.Comment = spec["comment"]
        if spec["stop"]:
            s.IsStop = True
    system.LDE.GetSurfaceAt(len(specs) + 1).Comment = "REAL MICROSCOPE IMAGE"
    system.SystemData.TitleNotes.Title = "Real 10X OBJ with pupil relay and tube lens, 584.601 nm"
    system.SystemData.TitleNotes.Notes = (
        "Source objective physical prescription preserved; reverse collector + external "
        "real pupil relay and tube lens. All refracting surfaces are real glass. "
        "Temporary validation at native wavelength. No Paraxial lens surfaces.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    system.SaveAs(str(output_path))
    return 25, len(specs) + 1


def add_operand(zos, name, params=(), target=0., weight=0.):
    api = zos.ZOSAPI
    row = zos.system.MFE.AddOperand()
    row.ChangeType(getattr(api.Editors.MFE.MeritOperandType, name))
    for i, val in enumerate(params, 1):
        cell = row.GetOperandCell(getattr(api.Editors.MFE.MeritColumn, "Param%d" % i))
        if cell.DataType == api.Editors.CellDataType.Integer:
            cell.IntegerValue = int(val)
        else:
            cell.DoubleValue = float(val)
    row.Target = target
    row.Weight = weight
    return row.OperandNumber


def optimize_added_optics(zos, out):
    system, api = zos.system, zos.ZOSAPI
    for num in (21, 22, 23, 24, 26, 27):
        cell = system.LDE.GetSurfaceAt(num).ConicCell
        cell.SetSolveData(cell.CreateSolveType(api.Editors.SolveType.Variable))
    cell = system.LDE.GetSurfaceAt(27).ThicknessCell
    cell.SetSolveData(cell.CreateSolveType(api.Editors.SolveType.Variable))
    mfe = system.MFE
    if mfe.NumberOfOperands:
        mfe.RemoveOperandsAt(1, mfe.NumberOfOperands)
    for hy in (0., 0.5, 1.):
        for r in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95):
            for az in range(8):
                theta = az * math.pi / 4.
                add_operand(zos, "OPDX", (0, 1, 0., hy, r*math.cos(theta), r*math.sin(theta)),
                            weight=1.)
        add_operand(zos, "REAY", (28, 1, 0., hy, 0., 0.), target=2.5*hy, weight=100.)
    # Explicit axial direction-sine to external Fourier-plane mapping.
    for p in (0.15, 0.3, 0.5, 0.7, 0.9):
        a = add_operand(zos, "REAB", (0, 1, 0., 0., 0., p))
        # 1x pupil relay inverts the transverse frequency coordinate.
        b = add_operand(zos, "PROB", (a, 0, -20.000029217895))
        c = add_operand(zos, "REAY", (25, 1, 0., 0., 0., p))
        add_operand(zos, "DIFF", (c, b), target=0., weight=10000.)
    for i in (21, 22, 23, 24, 26, 27):
        add_operand(zos, "COGT", (i,), target=-20., weight=100.)
        add_operand(zos, "COLT", (i,), target=20., weight=100.)
    before = mfe.CalculateMeritFunction()
    print("merit_before", before, flush=True)
    for i in range(173, 177):
        r = mfe.GetOperandAt(i)
        print("operand_check", i, r.Type, r.Value, flush=True)
    opt = system.Tools.OpenLocalOptimization()
    try:
        opt.Algorithm = api.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
        opt.Cycles = api.Tools.Optimization.OptimizationCycles.Fixed_50_Cycles
        opt.NumberOfCores = 4
        opt.RunAndWaitForCompletion()
    finally:
        opt.Close()
    after = mfe.CalculateMeritFunction()
    print("merit_after", after, flush=True)
    model = out / "models" / "real_collector_relay_tube_optimized.zos"
    system.SaveAs(str(model))
    system.SaveAs(str(model.with_suffix(".zmx")))
    results = {"merit_before": before, "merit_after": after,
               "conics": {str(i): float(system.LDE.GetSurfaceAt(i).Conic)
                          for i in (21,22,23,24,26,27)},
               "image_distance_mm": float(system.LDE.GetSurfaceAt(27).Thickness)}
    print("optimized", results, flush=True)
    (out / "optimization.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    audit = json.loads((out / "source_audit.json").read_text(encoding="utf-8"))
    cardinal = paraxial_from_audit(audit)
    print("cardinal", cardinal, flush=True)
    (out / "paraxial_transfer.json").write_text(json.dumps(cardinal, indent=2), encoding="utf-8")
    if not args.build:
        return
    with ZOSApplication() as zos:
        bfp, image_surface = create_real_system(
            zos, audit, out / "models" / "real_collector_relay_tube_initial.zos",
            cardinal["collector_external_bfp_distance_mm"])
        print("surfaces", bfp, image_surface, flush=True)
        if args.optimize:
            optimize_added_optics(zos, out)
        rays = []
        for hy in (0., 0.5, 1.):
            for py in (-0.9, -0.5, 0., 0.5, 0.9):
                im = ray(zos, image_surface, hy=hy, py=py, opd=True)
                fp = ray(zos, bfp, hy=hy, py=py)
                rays.append({"hy": hy, "py": py, "image": im, "fourier": fp})
                print("ray", hy, py, "image", im, "bfp_y", fp["y"], flush=True)
        (out / "relay_ray_audit.json").write_text(json.dumps(rays, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
