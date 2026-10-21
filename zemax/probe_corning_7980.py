"""Read-only check of local OpticStudio names for Corning HPFS 7980 silica."""
from zosapi_utils import ZOSApplication

CANDIDATES = ("CORNING 7980", "HPFS 7980", "7980", "F_SILICA", "SILICA")

with ZOSApplication() as zos:
    system, api = zos.system, zos.ZOSAPI
    system.New(False)
    system.SystemData.Wavelengths.GetWavelength(1).Wavelength = 0.355
    surface = system.LDE.GetSurfaceAt(1)
    for name in CANDIDATES:
        try:
            surface.Material = name
            stored = str(surface.Material)
            index = system.MFE.GetOperandValue(
                api.Editors.MFE.MeritOperandType.INDX, 1, 1, 0, 0, 0, 0, 0, 0)
            print(name, "->", stored, "n355=", float(index), flush=True)
        except Exception as exc:
            print(name, "-> ERROR", repr(exc), flush=True)
