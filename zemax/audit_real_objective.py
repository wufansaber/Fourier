"""Read-only optical audit; original Zemax sample is never saved or modified."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from zosapi_utils import ZOSApplication


def finite(value):
    number = float(value)
    return number if math.isfinite(number) else str(number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    preserved = out / "source" / source.name
    preserved.parent.mkdir(exist_ok=True)
    if preserved.exists() and preserved.read_bytes() != source.read_bytes():
        raise RuntimeError("Preserved source already exists with different contents")
    if not preserved.exists():
        shutil.copy2(source, preserved)

    info = {"source": str(source), "source_sha256": source_hash}
    with ZOSApplication() as zos:
        system, api = zos.system, zos.ZOSAPI
        system.LoadFile(str(preserved), False)
        print("Loaded", preserved, flush=True)
        info["version"] = str(zos.application.OpticStudioVersion)
        info["surface_count"] = system.LDE.NumberOfSurfaces
        info["aperture_type"] = str(system.SystemData.Aperture.ApertureType)
        info["aperture_value"] = float(system.SystemData.Aperture.ApertureValue)
        info["field_type"] = str(system.SystemData.Fields.GetFieldType())
        waves = system.SystemData.Wavelengths
        info["wavelengths"] = [
            {"number": i, "um": float(waves.GetWavelength(i).Wavelength),
             "weight": float(waves.GetWavelength(i).Weight)}
            for i in range(1, waves.NumberOfWavelengths + 1)
        ]
        print("waves", info["wavelengths"], flush=True)
        rows = []
        for i in range(system.LDE.NumberOfSurfaces):
            s = system.LDE.GetSurfaceAt(i)
            row = {"surface": i, "type": str(s.TypeName), "comment": str(s.Comment),
                   "radius_mm": finite(s.Radius), "thickness_mm": finite(s.Thickness),
                   "semi_diameter_mm": finite(s.SemiDiameter), "material": str(s.Material),
                   "material_solve": str(s.MaterialCell.GetSolveData().Type)}
            rows.append(row)
        info["surfaces"] = rows
        print("materials", [(r["surface"], r["material"], r["material_solve"])
                            for r in rows if r["material"]], flush=True)
        operands = api.Editors.MFE.MeritOperandType
        info["first_order"] = {}
        for name in ("EFFL", "ENPP", "EXPP", "EPDI", "EXPD", "ISNA", "PMAG"):
            try:
                info["first_order"][name] = finite(system.MFE.GetOperandValue(
                    getattr(operands, name), 0, 2, 0, 0, 0, 0, 0, 0))
            except Exception as exc:
                info["first_order"][name] = {"error": str(exc)}
        print("first_order", info["first_order"], flush=True)
        info["material_indices"] = []
        for row in rows:
            if row["material"]:
                val = system.MFE.GetOperandValue(operands.INDX, row["surface"], 2,
                                                  0, 0, 0, 0, 0, 0)
                info["material_indices"].append({"surface": row["surface"],
                                                 "name": row["material"], "index": finite(val)})
        traced = []
        tool = system.Tools.OpenBatchRayTrace()
        try:
            for hy in (0.0, 0.25, 1.0):
                for py in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
                    result = tool.SingleRayNormUnpol(
                        api.Tools.RayTrace.RaysType.Real, system.LDE.NumberOfSurfaces - 1,
                        2, 0.0, hy, 0.0, py, True,
                        0, 0, *([0.0] * 11))
                    traced.append({"hy": hy, "py": py, "raw": [finite(x) for x in result]})
        finally:
            tool.Close()
        info["rays"] = traced
        print("first_ray", traced[0], flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    (out / "source_audit.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "source_surfaces.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Audit saved", out, flush=True)


if __name__ == "__main__":
    main()
