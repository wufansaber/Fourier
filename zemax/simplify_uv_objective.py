"""Merge zero-index-step silica boundaries into a six-element UV objective."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from zosapi_utils import ZOSApplication

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uv_real_objective"
SOURCE = OUT / "models" / "uv355_real_objective_ideal_tube_optimized.zos"
DESTINATION = OUT / "models" / "uv355_corning7980_6element_ideal_tube.zos"
IMAGE_SURFACE = 15
F_OBJ_MM = 30.0
SINE_SCALE = -20.0


def batch(zos, surface, samples, hy=0.0):
    tool = zos.system.Tools.OpenBatchRayTrace()
    try:
        data = zos.ZOSAPI.Tools.RayTrace.IRayTraceNormUnpolData(
            tool.CreateNormUnpol(len(samples), zos.ZOSAPI.Tools.RayTrace.RaysType.Real, surface))
        mode = (zos.ZOSAPI.Tools.RayTrace.OPDMode.CurrentAndChief
                if surface == IMAGE_SURFACE
                else getattr(zos.ZOSAPI.Tools.RayTrace.OPDMode, "None"))
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


def main() -> None:
    with ZOSApplication() as zos:
        system = zos.system
        system.LoadFile(str(SOURCE), False)
        lde = system.LDE
        # Descending indices keep all remaining lower indices stable.  Each
        # removed surface separates F_SILICA from F_SILICA and has zero power.
        for number in (16, 12, 9, 6):
            previous = lde.GetSurfaceAt(number - 1)
            removed = lde.GetSurfaceAt(number)
            if str(previous.Material) != "F_SILICA" or str(removed.Material) != "F_SILICA":
                raise RuntimeError(f"S{number} is not a same-material boundary")
            previous.Thickness = float(previous.Thickness) + float(removed.Thickness)
            lde.RemoveSurfaceAt(number)
        lde.GetSurfaceAt(11).Comment = "PHYSICAL 36 mm APERTURE STOP / SINE PUPIL"
        lde.GetSurfaceAt(14).Comment = "IDEAL TUBE LENS f=600 mm"
        lde.GetSurfaceAt(15).Comment = "20X IMAGE / TDI PLANE"
        if system.MFE.NumberOfOperands:
            system.MFE.RemoveOperandsAt(1, system.MFE.NumberOfOperands)
        system.SystemData.TitleNotes.Title = (
            "355 nm six-element Corning 7980 objective NA=0.6 + ideal 600 mm tube")
        system.SystemData.TitleNotes.Notes = (
            "Six real fused-silica elements. Zemax F_SILICA is used as the "
            "Corning HPFS 7980 equivalent at 355 nm. Four zero-index-step seed "
            "boundaries were exactly merged; coatings are not modeled.")
        grid = np.linspace(-1.0, 1.0, 65)
        samples = [(float(px), float(py)) for py in grid for px in grid
                   if px * px + py * py <= 1.0]
        footprint_rows = []
        for surface_number in range(1, 14):
            maximum = 0.0
            for hy in (0.0, 0.5, 1.0):
                rays = batch(zos, surface_number, samples, hy)
                valid = (rays[:, 1] == 0) & (rays[:, 2] == 0)
                maximum = max(maximum, float(np.max(np.hypot(
                    rays[valid, 3], rays[valid, 4]))))
            surface = lde.GetSurfaceAt(surface_number)
            if surface_number == 11:
                clear_radius = 18.0
            else:
                clear_radius = math.ceil((maximum + 0.5) * 10) / 10
                surface.SemiDiameter = clear_radius
            radius = float(surface.Radius)
            conic = float(surface.Conic)
            domain_radius = (abs(radius) / math.sqrt(1 + conic)
                             if math.isfinite(radius) and 1 + conic > 0 else math.inf)
            if clear_radius >= domain_radius:
                raise RuntimeError(
                    f"S{surface_number} clear radius {clear_radius} exceeds conic domain {domain_radius}")
            footprint_rows.append({
                "surface": surface_number,
                "max_three_field_ray_radius_mm": maximum,
                "assigned_clear_semi_diameter_mm": clear_radius,
                "radial_clearance_mm": clear_radius - maximum,
                "conic_domain_radius_mm": domain_radius,
                "conic_domain_margin_mm": domain_radius - clear_radius,
            })
        with (OUT / "footprint_clear_apertures.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(footprint_rows[0]))
            writer.writeheader()
            writer.writerows(footprint_rows)

        system.SaveAs(str(DESTINATION))
        system.SaveAs(str(DESTINATION.with_suffix(".zmx")))

        fields = []
        for hy in (0.0, 0.5, 1.0):
            rays = batch(zos, IMAGE_SURFACE, samples, hy)
            valid = (rays[:, 1] == 0) & (rays[:, 2] == 0)
            x, y, opd = rays[valid, 3], rays[valid, 4], rays[valid, 12]
            fields.append({
                "field_hy": hy,
                "rays_total": len(samples),
                "rays_valid": int(valid.sum()),
                "centroid_y_mm": float(np.mean(y)),
                "rms_spot_um": float(1000 * np.sqrt(np.mean(
                    (x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))),
                "opd_rms_waves": float(np.std(opd)),
                "opd_pv_waves": float(np.ptp(opd)),
            })

        meridional = [(0.0, float(py)) for py in np.linspace(-1.0, 1.0, 401)]
        object_rays = batch(zos, 0, meridional)
        image_rays = batch(zos, IMAGE_SURFACE, meridional)
        valid = ((object_rays[:, 1] == 0) & (object_rays[:, 2] == 0) &
                 (image_rays[:, 1] == 0) & (image_rays[:, 2] == 0))
        sin_object = object_rays[valid, 7]
        sin_image = image_rays[valid, 7]
        residual = sin_object - SINE_SCALE * sin_image
        best_scale = float(np.dot(sin_image, sin_object) /
                           np.dot(sin_image, sin_image))
        best_residual = sin_object - best_scale * sin_image

        surfaces = []
        z_vertex = 0.0
        for number in range(lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(number)
            if number > 0:
                z_vertex += float(lde.GetSurfaceAt(number - 1).Thickness)
            surfaces.append({
                "surface": number,
                "z_vertex_mm": z_vertex,
                "type": str(surface.TypeName),
                "radius_mm": float(surface.Radius),
                "thickness_mm": float(surface.Thickness),
                "material": str(surface.Material),
                "semi_diameter_mm": float(surface.SemiDiameter),
                "conic": float(surface.Conic),
                "comment": str(surface.Comment),
            })
        with (OUT / "surface_prescription_6element.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(surfaces[0]))
            writer.writeheader()
            writer.writerows(surfaces)
        (OUT / "surface_prescription_6element.json").write_text(
            json.dumps(surfaces, indent=2), encoding="utf-8")

        layout = []
        for surface in range(lde.NumberOfSurfaces):
            for hy in (0.0, 1.0):
                ray_samples = [(0.0, -0.95), (0.0, 0.0), (0.0, 0.95)]
                rays = batch(zos, surface, ray_samples, hy)
                for (px, py), row in zip(ray_samples, rays):
                    layout.append({
                        "surface": surface, "z_vertex_mm": surfaces[surface]["z_vertex_mm"],
                        "field_hy": hy, "pupil_y": py, "ray_y_mm": float(row[4]),
                        "error": int(row[1]), "vignette": int(row[2]),
                    })
        with (OUT / "layout_rays_6element.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(layout[0]))
            writer.writeheader()
            writer.writerows(layout)

        summary = {
            "model": str(DESTINATION),
            "real_elements": 6,
            "real_material": "Corning HPFS 7980 equivalent / Zemax F_SILICA",
            "field_validation": fields,
            "abbe_sine": {
                "nominal_scale": SINE_SCALE,
                "best_fit_scale": best_scale,
                "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                "residual_max_abs": float(np.max(np.abs(residual))),
                "equivalent_fsin_rms_um": float(
                    1000 * F_OBJ_MM * np.sqrt(np.mean(residual ** 2))),
                "equivalent_fsin_max_um": float(
                    1000 * F_OBJ_MM * np.max(np.abs(residual))),
                "best_fit_equivalent_fsin_rms_um": float(
                    1000 * F_OBJ_MM * np.sqrt(np.mean(best_residual ** 2))),
                "best_fit_equivalent_fsin_max_um": float(
                    1000 * F_OBJ_MM * np.max(np.abs(best_residual))),
            },
        }
        (OUT / "simplified_validation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
