"""Read-only dump of selected merit operands in the UV objective model."""
from pathlib import Path
from zosapi_utils import ZOSApplication

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results" / "uv_real_objective" / "models" / "uv355_real_objective_ideal_tube_optimized.zos"

with ZOSApplication() as zos:
    zos.system.LoadFile(str(MODEL), False)
    mfe = zos.system.MFE
    print("merit", mfe.CalculateMeritFunction(), "rows", mfe.NumberOfOperands)
    for number in range(295, min(mfe.NumberOfOperands, 340) + 1):
        row = mfe.GetOperandAt(number)
        print(number, row.Type, "target", row.Target, "weight", row.Weight,
              "value", row.Value)
