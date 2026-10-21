"""Validate large-period defect figures, metrics and report integration."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "large_period_defects"
REPORT = ROOT / "双层交错傅里叶档杆_工程实现与仿真验证.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def main() -> None:
    images = [
        OUT / "large_period_mixed_defect_gallery.png",
        OUT / "large_period_defect_type_gallery.png",
        OUT / "large_period_spacing_and_metrics.png",
    ]
    for path in images:
        require(path.is_file() and path.stat().st_size > 100_000, f"missing/small image: {path}")
        with Image.open(path) as image:
            require(image.width >= 2000 and image.height >= 900, f"insufficient image size: {path}")
    require((OUT / "large_period_metrics.csv").is_file(), "missing metrics CSV")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = {row["period_um"]: row for row in summary["period_results"]}
    expected_spacing = {100.0: 0.300, 150.0: 0.200, 200.0: 0.150}
    for period, spacing in expected_spacing.items():
        require(abs(rows[period]["current_order_spacing_mm"] - spacing) < 1e-12, f"spacing mismatch: {period}")
        require(rows[period]["background_suppression_db"] >= 10.0, f"suppression below 10 dB: {period}")
    types = {row["defect_type"]: row for row in summary["defect_type_results_at_150um"]}
    for kind in ["isolated_dark_hole", "isolated_bright_point", "elliptical_hole", "close_hole_pair", "irregular_hole"]:
        require(kind in types, f"missing defect type: {kind}")
    point = types["isolated_dark_hole"]
    require(point["defect_field_energy_retention"] >= 0.60, "isolated-point retention below 60%")
    require(point["r90_filtered_um"] > 2.0 * point["r90_open_um"], "point ringing broadening not represented")
    require(types["close_hole_pair"]["defect_field_energy_retention"] >= 0.50, "close-pair retention unexpectedly low")
    report = REPORT.read_text(encoding="utf-8")
    for token in [
        "large_period_mixed_defect_gallery.png",
        "large_period_defect_type_gallery.png",
        "large_period_spacing_and_metrics.png",
        "10.98 µm",
        "密集低级次带阻模式",
    ]:
        require(token in report, f"report missing: {token}")
    print("PASS: 100/150/200 um periods, five compact point/hole types, spacing limits and ringing metrics checked")


if __name__ == "__main__":
    main()
