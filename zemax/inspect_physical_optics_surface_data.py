"""Print OpticStudio 2022 physical-optics surface-data members and defaults."""
from zosapi_utils import ZOSApplication
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results" / "uv_real_objective" / "models" / "uv355_corning7980_6element_ideal_tube.zos"

with ZOSApplication() as zos:
    zos.system.LoadFile(str(MODEL), False)
    for number in (1, 5, 10, 11, 14, 15):
        data = zos.system.LDE.GetSurfaceAt(number).PhysicalOpticsData
        names = [name for name in dir(data) if not name.startswith("_")]
        print("SURFACE", number, "MEMBERS", ",".join(names), flush=True)
        for name in names:
            try:
                value = getattr(data, name)
                if isinstance(value, (str, int, float, bool)):
                    print(" ", name, repr(value), flush=True)
            except Exception as exc:
                print(" ", name, "ERROR", exc, flush=True)
