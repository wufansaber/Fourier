"""Validate the 2--5 um periodic-pattern simulation deliverables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "small_period_point_defects"
REPORT = ROOT / "双层交错傅里叶档杆_工程实现与仿真验证.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    image_names = [
        "small_period_particle_global_linear_brightness.png",
        "small_period_particle_raw_same_scale.png",
        "small_period_fourier_orders.png",
        "small_period_raw_metrics.png",
    ]
    for name in image_names:
        path = OUT / name
        require(path.is_file(), f"missing image: {path}")
        with Image.open(path) as image:
            require(image.width >= 1800, f"image too narrow: {name}")
            require(image.height >= 700, f"image too short: {name}")

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    parameters = summary["parameters"]
    require(parameters["rod_count"] == 7, "rod count must be seven")
    require(parameters["active_orders"] == list(range(-3, 4)), "active orders changed")
    require(abs(parameters["rod_envelope_mm"] - 2.0) < 1e-12, "rod envelope changed")

    results = summary["results"]
    require([row["period_um"] for row in results] == [2.0, 3.0, 4.0, 5.0], "period set changed")
    expected_spacing = [15.0, 10.0, 7.5, 6.0]
    expected_orders = [3, 5, 6, 8]
    for row, spacing, max_order in zip(results, expected_spacing, expected_orders):
        require(abs(row["spacing_mm"] - spacing) < 1e-9, "Fourier spacing mismatch")
        require(row["max_propagating_order"] == max_order, "propagating-order count mismatch")
        require(row["background_suppression_db"] > 15.0, "background suppression too small")
        require(0.0 < row["raw_particle_peak_retention"] <= 1.0, "invalid particle retention")
        require(row["raw_particle_to_pattern_filtered_db"] >
                row["raw_particle_to_pattern_open_db"], "particle contrast did not improve")

    with (OUT / "small_period_raw_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4, "metrics CSV row count mismatch")

    report = REPORT.read_text(encoding="utf-8")
    require("### 7.2 同强度滤波前后对比" in report, "small-period report section missing")
    for name in image_names[:2]:
        require(name in report, f"report does not link {name}")
    require("results/large_period_defects" not in report, "large-period result is still linked in report")
    print("PASS: 2--5 um simulation, metrics, images, and report links are consistent.")


if __name__ == "__main__":
    main()
