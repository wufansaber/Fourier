"""Run the existing Python 2 um three-particle case through the real Zemax lens.

The Python source model is imported without modification.  Its complete complex
object field is preserved pixel-for-pixel in the original case coordinates,
normalized to one watt, written to a Zemax beam file, and propagated by native
OpticStudio POP.  The original horizontal bands are periodic in object Y, while
the final engineering rods remain long in Fourier Y and positioned in Fourier X.

Only two native POP cases are run: open pupil and the seven-rod filter.  The
filter is placed at the real objective BFP using the equivalent dimensions of
the final 2.816901 X relay, so the physical hardware remains 1.0 mm core plus
0.5 mm teeth per side at 15 mm pitch.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

from zosapi_utils import ZOSApplication


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import small_period_raw_intensity_simulation as python_case  # noqa: E402


BASE_MODEL = (ROOT / "results" / "uv_real_objective" / "models" /
              "uv355_corning7980_6element_ideal_tube.zos")
OUT_DEFAULT = ROOT / "results" / "uv_real_objective_python_case_2um"

WAVELENGTH_MM = 0.000355
PERIOD_MM = 0.002
MAG_IMAGE = 20.0
PUPIL_RADIUS_MM = 18.0
FIELD_WIDTH_MM = 0.120
FOURIER_POSITION_MAG = 2.816901408450704
REAL_BFP_SINE_SLOPE_MM = 29.102328830493818
REAL_BFP_ORDER_SPACING_MM = REAL_BFP_SINE_SLOPE_MM * WAVELENGTH_MM / PERIOD_MM
HARDWARE_CORE_MM = 1.0
HARDWARE_DEPTH_MM = 0.5
HARDWARE_TOOTH_PITCH_MM = 0.75
EQUIVALENT_BFP_CORE_MM = HARDWARE_CORE_MM / FOURIER_POSITION_MAG
EQUIVALENT_BFP_DEPTH_MM = HARDWARE_DEPTH_MM / FOURIER_POSITION_MAG
ORDERS_A = (-2, 0, 2)
ORDERS_B = (-3, -1, 1, 3)


def rectangle(x0: float, x1: float, y0: float, y1: float):
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def triangle_fraction(y_mm: float) -> float:
    phase = y_mm / HARDWARE_TOOTH_PITCH_MM
    triangle = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
    return 0.5 * (triangle + 1.0)


def pass_polygons(centres_x_mm, depth_mm: float):
    """Clear corridors around rods long in BFP Y and positioned in BFP X."""
    outer = 22.0
    if not centres_x_mm:
        return [rectangle(-outer, outer, -outer, outer)]
    samples = np.arange(-outer, outer + HARDWARE_TOOTH_PITCH_MM / 4,
                        HARDWARE_TOOTH_PITCH_MM / 4)
    ys = sorted(set([-outer, outer] +
                    [float(value) for value in samples if -outer < value < outer]))
    centres = sorted(float(value) for value in centres_x_mm)

    def half_width(y):
        return EQUIVALENT_BFP_CORE_MM / 2 + depth_mm * triangle_fraction(y)

    polygons = []
    first = centres[0]
    polygons.append([(-outer, y) for y in ys] +
                    [(first - half_width(y), y) for y in reversed(ys)])
    for lower_centre, upper_centre in zip(centres[:-1], centres[1:]):
        polygons.append([(lower_centre + half_width(y), y) for y in ys] +
                        [(upper_centre - half_width(y), y) for y in reversed(ys)])
    last = centres[-1]
    polygons.append([(last + half_width(y), y) for y in ys] +
                    [(outer, y) for y in reversed(ys)])
    return polygons


def write_uda(path: Path, polygons, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"! {title}\n")
        for polygon in polygons:
            for x, y in polygon:
                stream.write(f"{x:.12g} {y:.12g}\n")
            stream.write("0 0\n")


def write_zbf(path: Path, field: np.ndarray, dx_mm: float, dy_mm: float) -> None:
    """Write Zemax beam-file format 1 with one scalar polarization component."""
    ny, nx = field.shape
    waist_x = nx * dx_mm / 2
    waist_y = ny * dy_mm / 2
    header_i = (1, nx, ny, 0, 0, 0, 0, 0, 0)
    header_d = (
        dx_mm, dy_mm, 0.0, math.pi * waist_x**2 / WAVELENGTH_MM, waist_x,
        0.0, math.pi * waist_y**2 / WAVELENGTH_MM, waist_y,
        WAVELENGTH_MM, 1.0, 0.0, 0.0, *([0.0] * 8),
    )
    ordered = np.asarray(field, dtype=np.complex128).ravel(order="C")
    interleaved = np.empty(2 * ordered.size, dtype="<f8")
    interleaved[0::2] = ordered.real
    interleaved[1::2] = ordered.imag
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<9i", *header_i))
        stream.write(struct.pack("<20d", *header_d))
        stream.write(interleaved.tobytes())


def prepare_python_reference(grid_size: int, out: Path):
    """Evaluate the unchanged Python model and preserve its original axes."""
    dx = FIELD_WIDTH_MM / grid_size
    coordinate = (np.arange(grid_size) - grid_size // 2) * dx
    xx, yy = np.meshgrid(coordinate, coordinate)
    frequency = np.fft.fftshift(np.fft.fftfreq(grid_size, d=dx))
    fxx, fyy = np.meshgrid(frequency, frequency)
    pupil = (((WAVELENGTH_MM * fxx) ** 2 +
              (WAVELENGTH_MM * fyy) ** 2) <= python_case.NA**2).astype(np.float32)
    grid = {
        "xx": xx,
        "yy": yy,
        "pupil": pupil,
        "x_hw": python_case.F_MM * WAVELENGTH_MM * frequency,
        "y_hw": python_case.MY * python_case.F_MM * WAVELENGTH_MM * frequency,
    }
    background = python_case.periodic_pattern(xx, yy, PERIOD_MM)
    object_field, particles = python_case.irregular_particle_field(background, xx, yy)
    input_power = float(np.sum(np.abs(object_field) ** 2) * dx * dx)
    normalized_field = object_field / math.sqrt(input_power)

    # Re-evaluate with the original unmodified functions and preserve its axes.
    spectrum = python_case.ft2(normalized_field)
    rods = python_case.seven_rod_mask(
        grid["x_hw"], grid["y_hw"], python_case.MY * python_case.F_MM *
        WAVELENGTH_MM / PERIOD_MM)
    open_intensity = np.abs(python_case.ift2(spectrum * pupil)) ** 2
    filtered_intensity = np.abs(python_case.ift2(spectrum * pupil * rods)) ** 2
    zbf_path = out / "python_case_2um_three_particles_original_axes.ZBF"
    write_zbf(zbf_path, normalized_field, dx, dx)
    np.savez_compressed(
        out / "python_reference_2um.npz",
        coordinate_mm=coordinate,
        object_field=normalized_field,
        open_intensity=open_intensity,
        filtered_intensity=filtered_intensity,
        particle_centres_mm=np.asarray([[-0.012, 0.0], [0.0, 0.0], [0.012, 0.0]]),
        particle_sizes_um=np.asarray([item[2] for item in particles]),
    )
    return zbf_path, input_power


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


def install_files(zos, out: Path, zbf_path: Path):
    centres = {order: order * REAL_BFP_ORDER_SPACING_MM for order in range(-3, 4)}
    definitions = {
        "open": pass_polygons([], 0.0),
        "deck_a": pass_polygons([centres[o] for o in ORDERS_A], EQUIVALENT_BFP_DEPTH_MM),
        "deck_b": pass_polygons([centres[o] for o in ORDERS_B], EQUIVALENT_BFP_DEPTH_MM),
    }
    aperture_destination = Path(zos.application.ObjectsDir) / "Apertures"
    aperture_destination.mkdir(parents=True, exist_ok=True)
    names = {}
    for key, polygons in definitions.items():
        filename = f"UVREAL_PYCASE2UM_{key.upper()}.UDA"
        source = out / "uda" / filename
        write_uda(source, polygons, f"UV real objective Python-case 2 um {key}")
        shutil.copy2(source, aperture_destination / filename)
        names[key] = filename
    # In OpticStudio 2022 R2 POPDir already resolves to ...\POP\BEAMFILES.
    beam_destination = Path(zos.application.POPDir)
    beam_destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zbf_path, beam_destination / zbf_path.name)
    zos.application.UpdateFileLists()
    return names, centres


def build_variant(zos, out: Path, apertures: dict, filtered: bool,
                  grid_size: int, surface_sampling: str,
                  resample_after_refraction: bool) -> Path:
    system = zos.system
    system.LoadFile(str(BASE_MODEL), False)
    lde = system.LDE

    working_distance = float(lde.GetSurfaceAt(0).Thickness)
    lde.InsertNewSurfaceAt(1)
    lde.GetSurfaceAt(0).Thickness = 0.0
    start = lde.GetSurfaceAt(1)
    start.Thickness = working_distance
    start.SemiDiameter = 0.30
    start.Comment = "PYTHON 2 UM THREE-PARTICLE ZBF OBJECT PLANE"

    lde.InsertNewSurfaceAt(12)
    deck_a = lde.GetSurfaceAt(12)
    deck_a.Thickness = 0.0
    deck_a.SemiDiameter = 22.0
    deck_a.Comment = "EQUIVALENT DECK A: RODS LONG Y, m=-2,0,+2"
    lde.InsertNewSurfaceAt(13)
    deck_b = lde.GetSurfaceAt(13)
    deck_b.Thickness = 0.0
    deck_b.SemiDiameter = 22.0
    deck_b.Comment = "EQUIVALENT DECK B: RODS LONG Y, m=-3,-1,+1,+3"

    set_user_aperture(zos, deck_a, apertures["deck_a" if filtered else "open"])
    set_user_aperture(zos, deck_b, apertures["deck_b" if filtered else "open"])
    set_circle(zos, lde.GetSurfaceAt(14), PUPIL_RADIUS_MM)

    sampling_code = int(round(math.log(grid_size, 2))) - 5
    for number in range(1, lde.NumberOfSurfaces):
        data = lde.GetSurfaceAt(number).PhysicalOpticsData
        data.DoNotRescaleBeamSizeUsingRayData = surface_sampling == "manual"
        data.AutoResample = surface_sampling == "auto"
        data.ResampleAfterRefraction = resample_after_refraction
        data.XSampling = sampling_code
        data.YSampling = sampling_code
        if surface_sampling == "manual":
            if number == 1:
                width = FIELD_WIDTH_MM
            elif number in (12, 13, 14, 15, 16, 17):
                width = 40.0
            elif number == 18:
                width = 3.0
            else:
                width = max(1.0, 2.05 * float(lde.GetSurfaceAt(number).SemiDiameter))
            data.XWidth = width
            data.YWidth = width

    mode = "seven_rods" if filtered else "open"
    system.SystemData.TitleNotes.Title = f"UV real objective: Python 2 um case, {mode}"
    system.SystemData.TitleNotes.Notes = (
        "Native POP through six real 7980-equivalent elements and ideal tube lens. "
        "Input is the existing Python 2 um three-particle complex field with no image "
        "transpose.  Horizontal bands are periodic in object Y; final engineering rods "
        "remain long in Fourier Y and positioned in Fourier X.  This exact-image case "
        "therefore exercises primarily the central X-positioned rod."
    )
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"uvreal_python_case_2um_{mode}.zos"
    system.SaveAs(str(path))
    return path


def net_grid(grid):
    nx, ny = int(grid.Nx), int(grid.Ny)
    values = np.asarray(list(grid.Values), dtype=float).reshape((ny, nx))
    x = float(grid.MinX) + np.arange(nx) * float(grid.Dx)
    y = float(grid.MinY) + np.arange(ny) * float(grid.Dy)
    return x, y, values


def run_pop(zos, path: Path, beam_filename: str, grid_size: int,
            end_surface: int = 18):
    system = zos.system
    api = zos.ZOSAPI
    system.LoadFile(str(path), False)
    pop = system.Analyses.New_Analysis_SettingsFirst(
        api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    try:
        # Keep the concrete settings object.  The interface cast used by older
        # TopHat scripts drops Update(), which is needed after switching to a
        # file beam in OpticStudio 2022 R2.
        settings = pop.GetSettings()
        settings.StartSurface.SetSurfaceNumber(1)
        settings.EndSurface.SetSurfaceNumber(end_surface)
        settings.Wavelength.SetWavelengthNumber(1)
        settings.Field.SetFieldNumber(1)
        settings.SurfaceToBeam = 0
        settings.BeamType = api.Analysis.PhysicalOptics.POPBeamTypes.File
        settings.Update()
        sampling = getattr(api.Analysis.SampleSizes, f"S_{grid_size}x{grid_size}")
        settings.XSampling = sampling
        settings.YSampling = sampling
        settings.XWidth = FIELD_WIDTH_MM
        settings.YWidth = FIELD_WIDTH_MM
        settings.UsePolarization = False
        settings.UseTotalPower = True
        settings.TotalPower = 1.0
        settings.SeparateXY = False
        settings.UseDiskStorage = True
        settings.DataType = api.Analysis.PhysicalOptics.POPDataTypes.Irradiance
        settings.ZoomIn = api.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
        settings.BeamTypeFilename = beam_filename
        settings.Update()
        available = []
        try:
            available = [str(settings.GetBeamTypeFileName(i))
                         for i in range(int(settings.NumberOfBeamTypeFiles))]
        except Exception as exc:
            available = [f"file-list query failed: {exc}"]
        print(f"beam_type={settings.BeamType} beam_file={settings.BeamTypeFilename} "
              f"listed={beam_filename in available} file_count={len(available)}", flush=True)
        pop.ApplyAndWaitForCompletion()
        results = pop.GetResults()
        messages = [str(results.GetMessageAt(i)) for i in range(results.NumberOfMessages)]
        if results.NumberOfDataGrids < 1:
            raise RuntimeError(f"No POP grid for {path.name}: {messages}")
        x, y, irradiance = net_grid(results.GetDataGrid(0))
        return x, y, irradiance, messages
    finally:
        pop.Close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grid", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument("--job", choices=("both", "open", "filtered"), default="both")
    parser.add_argument("--end-surface", type=int, default=18)
    parser.add_argument("--surface-sampling", choices=("manual", "auto"), default="manual")
    parser.add_argument("--refraction-resampling", choices=("on", "off"), default="on")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    zbf_path, input_power = prepare_python_reference(args.grid, out)
    data = {}
    messages = {}
    with ZOSApplication() as zos:
        print(f"OpticStudio={zos.application.OpticStudioVersion}", flush=True)
        apertures, centres = install_files(zos, out, zbf_path)
        models = {
            "open": build_variant(zos, out, apertures, False, args.grid,
                                  args.surface_sampling,
                                  args.refraction_resampling == "on"),
            "filtered": build_variant(zos, out, apertures, True, args.grid,
                                      args.surface_sampling,
                                      args.refraction_resampling == "on"),
        }
        job_keys = ("open", "filtered") if args.job == "both" else (args.job,)
        for index, key in enumerate(job_keys, 1):
            print(f"POP {index}/{len(job_keys)}: {key} to S{args.end_surface}", flush=True)
            x, y, irradiance, pop_messages = run_pop(
                zos, models[key], zbf_path.name, args.grid, args.end_surface)
            data[key] = (x, y, irradiance)
            messages[key] = pop_messages
            power = float(np.sum(np.maximum(irradiance, 0)) *
                          abs(x[1] - x[0]) * abs(y[1] - y[0]))
            print(f"grid={len(x)} span={x[-1]-x[0]:.6g} mm power={power:.8g} W "
                  f"messages={pop_messages}", flush=True)

    np.savez_compressed(
        out / "zemax_native_pop_2um.npz",
        **{f"{key}_x_mm": value[0] for key, value in data.items()},
        **{f"{key}_y_mm": value[1] for key, value in data.items()},
        **{f"{key}_irradiance_W_mm2": value[2] for key, value in data.items()},
    )
    summary = {
        "engine": "Local OpticStudio 2022 R2.02 native POP",
        "base_model": str(BASE_MODEL),
        "comparison_scope": "2 um existing Python three-particle case only",
        "python_model_changed": False,
        "coordinate_operation": "none; original Python case pixels and axes preserved",
        "input": {
            "beam_file": zbf_path.name,
            "grid": args.grid,
            "window_mm": FIELD_WIDTH_MM,
            "input_power_before_1W_normalization": input_power,
            "input_power_W_in_POP": 1.0,
            "particle_centres_um": [[-12.0, 0.0], [0.0, 0.0], [12.0, 0.0]],
            "particle_sizes_um": [0.6, 0.9, 1.2],
            "note": "particles are image features; no separate particle coherence sweep is performed",
        },
        "optics": {
            "wavelength_nm": 355.0,
            "objective_na": 0.60,
            "objective_nominal_focal_length_mm": 30.0,
            "real_bfp_sine_slope_mm": REAL_BFP_SINE_SLOPE_MM,
            "image_magnification": MAG_IMAGE,
            "pupil_diameter_mm": 36.0,
        },
        "filter": {
            "rod_count": 7,
            "rod_long_axis": "Fourier Y",
            "rod_position_axis": "Fourier X",
            "real_bfp_order_spacing_mm": REAL_BFP_ORDER_SPACING_MM,
            "real_bfp_centres_mm": {str(k): v for k, v in centres.items()},
            "hardware_order_spacing_mm": 15.0,
            "hardware_core_mm": HARDWARE_CORE_MM,
            "hardware_one_sided_depth_mm": HARDWARE_DEPTH_MM,
            "hardware_tooth_pitch_mm": HARDWARE_TOOTH_PITCH_MM,
            "equivalent_real_bfp_core_mm": EQUIVALENT_BFP_CORE_MM,
            "equivalent_real_bfp_one_sided_depth_mm": EQUIVALENT_BFP_DEPTH_MM,
        },
        "messages": messages,
        "pop_end_surface": args.end_surface,
        "surface_sampling": args.surface_sampling,
        "resample_after_refraction": args.refraction_resampling == "on",
        "evaluation_rule": "raw irradiance; common scale; no background subtraction",
        "geometry_scope_note": (
            "The unchanged Python example is periodic in object Y, so its orders run along "
            "Fourier Y. With the final rods long in Fourier Y and positioned in Fourier X, "
            "the example is an exact-image consistency case, not a seven-column utilization test."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
