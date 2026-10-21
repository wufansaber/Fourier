"""Run one saved POP model to verify the Fourier-plane sampling window."""

import argparse
from pathlib import Path

from build_and_run_zemax import grid_power, run_pop
from zosapi_utils import ZOSApplication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--end-surface", type=int, default=3)
    parser.add_argument("--sampling-code", type=int, default=7)
    args = parser.parse_args()
    with ZOSApplication() as zos:
        zos.system.LoadFile(str(args.model.resolve()), False)
        api = zos.ZOSAPI
        for number in range(1, zos.system.LDE.NumberOfSurfaces):
            surface = zos.system.LDE.GetSurfaceAt(number)
            try:
                value = surface.GetSurfaceCell(api.Editors.LDE.SurfaceColumn.Par1).DoubleValue
            except Exception:
                value = 0.0
            if value != 0:
                print(
                    f"surface={number};comment={surface.Comment};"
                    f"par1={value};thickness={surface.Thickness}"
                )
        pop_data = zos.system.LDE.GetSurfaceAt(3).PhysicalOpticsData
        print(
            "physical_optics_members="
            + ",".join(sorted(name for name in dir(pop_data) if not name.startswith("_")))
        )
        for name in sorted(name for name in dir(pop_data) if not name.startswith(("_", "get_", "set_"))):
            try:
                value = getattr(pop_data, name)
                if isinstance(value, (bool, int, float, str)):
                    print(f"physical_optics.{name}={value!r}")
            except Exception:
                pass
        x, y, irradiance, messages = run_pop(
            zos, args.model.resolve(), args.end_surface, args.sampling_code
        )
    print(
        f"grid={len(x)}x{len(y)};x={x[0]}..{x[-1]};"
        f"y={y[0]}..{y[-1]};power={grid_power(x,y,irradiance)}"
    )
    profile = irradiance[len(y) // 2]
    local = (profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:])
    indices = (local.nonzero()[0] + 1).tolist()
    indices.sort(key=lambda index: profile[index], reverse=True)
    selected = []
    for index in indices:
        if all(abs(x[index] - position) > 0.2 for position in selected):
            selected.append(float(x[index]))
        if len(selected) == 11:
            break
    print("peak_x_mm=" + ",".join(f"{value:.6f}" for value in sorted(selected)))
    for label, spacing in (("original", 0.8875), ("relayed", 4.4375)):
        samples = []
        for order in range(-5, 6):
            expected = order * spacing
            index = int(abs(x - expected).argmin())
            samples.append(profile[index] / max(float(profile.max()), 1e-300))
        print(
            f"{label}_order_relative_intensity="
            + ",".join(f"{value:.4g}" for value in samples)
        )
    for message in messages:
        print(f"message={message}")


if __name__ == "__main__":
    main()
