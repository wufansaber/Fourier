"""Validate the frozen Python and native-Zemax before/after deliverables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "before_after_comparison_summary.json"
ZMX = ROOT / "results" / "dual_deck_zemax_equivalent_fullwindow"
REPORT = ROOT / "双层交错傅里叶档杆_工程实现与仿真验证.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required = [
        ROOT / "results/dual_deck_engineering/python_2d_before_after.png",
        ROOT / "results/dual_deck_engineering/python_2d_defect_zoom_before_after.png",
        ZMX / "zemax_before_after_absolute.png",
        ZMX / "zemax_before_after_common_db.png",
        ZMX / "native_pop_grids.npz",
        ZMX / "summary.json",
        ZMX / "metrics.csv",
        REPORT,
    ]
    for path in required:
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty: {path}")

    combined = json.loads(SUMMARY.read_text(encoding="utf-8"))
    py = combined["python_2d"]
    require(py["background_suppression_db"] >= 10.0, "Python suppression below 10 dB")
    require(py["isolated_defect_energy_retention"] >= 0.60, "Python defect retention below 60%")
    require(py["active_orders"] == list(range(-3, 4)), "Python active orders are not -3...+3")

    zsummary = json.loads((ZMX / "summary.json").read_text(encoding="utf-8"))
    p = zsummary["parameters"]
    require(p["wavelength_nm"] == 355, "Zemax wavelength mismatch")
    require(abs(p["objective_focal_length_mm"] - 30.0) < 1e-12, "Zemax focal length mismatch")
    require(abs(p["objective_na"] - 0.6) < 1e-12, "Zemax NA mismatch")
    require(abs(p["pop_full_window_mm"] - 10.0) < 1e-12, "Zemax POP window mismatch")
    require(p["pop_sampling_code"] == 8, "Zemax sampling code mismatch")
    rows = {row["mode"]: row for row in zsummary["metrics"]}
    require(rows["balanced7"]["background_suppression_db"] >= 20.0, "Zemax suppression below 20 dB")

    grids = np.load(ZMX / "native_pop_grids.npz")
    for key in ["regular_open_s6", "defect_open_s6", "regular_balanced7_s6", "defect_balanced7_s6"]:
        x = grids[f"{key}_x"]
        y = grids[f"{key}_y"]
        a = grids[f"{key}_irradiance"]
        require(a.shape == (1024, 1024), f"{key}: wrong grid shape {a.shape}")
        require(x[-1] - x[0] >= 9.8 and y[-1] - y[0] >= 9.8, f"{key}: truncated POP window")
        require(np.isfinite(a).all() and a.min() >= 0, f"{key}: invalid irradiance values")

    for path in required[:4]:
        with Image.open(path) as image:
            require(image.width >= 1600 and image.height >= 900, f"image too small: {path}")

    report = REPORT.read_text(encoding="utf-8")
    for token in [
        "python_2d_before_after.png",
        "python_2d_defect_zoom_before_after.png",
        "zemax_before_after_absolute.png",
        "zemax_before_after_common_db.png",
        "28.206 dB",
        "等效瞳面",
    ]:
        require(token in report, f"report missing token: {token}")

    print("PASS: before/after Python and Zemax deliverables are complete and internally consistent")


if __name__ == "__main__":
    main()
