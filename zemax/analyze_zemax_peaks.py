"""Extract Zemax POP diffraction peaks and compare sin/tan mappings."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


WAVELENGTH_MM = 0.000355
FOCAL_LENGTH_MM = 50.0
PITCH_MM = 0.020
PUPIL_RELAY_MAG = 5.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    with np.load(output_dir / "zemax_pop_grids.npz") as archive:
        x = archive["defect_unfiltered_s3_x"]
        image = archive["defect_unfiltered_s3_irradiance"]
    # Integrate over y.  A 50% duty-cycle square wave ideally has only odd
    # diffraction orders, so even orders are deliberately excluded.
    profile = np.sum(image, axis=0)
    rows = []
    for order in (-5, -3, -1, 0, 1, 3, 5):
        sin_theta = order * WAVELENGTH_MM / PITCH_MM
        theory_sin = FOCAL_LENGTH_MM * sin_theta
        theta = math.asin(sin_theta)
        theory_tan = FOCAL_LENGTH_MM * math.tan(theta)
        window = np.flatnonzero(np.abs(x - theory_sin) <= 0.18)
        index = int(window[np.argmax(profile[window])])
        zemax = float(x[index])
        rows.append(
            {
                "order": order,
                "sin_theta": sin_theta,
                "zemax_bfp_peak_mm": zemax,
                "sin_prediction_mm": theory_sin,
                "tan_prediction_mm": theory_tan,
                "sin_error_um": 1000 * (zemax - theory_sin),
                "tan_error_um": 1000 * (zemax - theory_tan),
                "hardware_5x_peak_mm": PUPIL_RELAY_MAG * zemax,
            }
        )
    with (output_dir / "zemax_peak_positions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sin_rms = math.sqrt(np.mean([row["sin_error_um"] ** 2 for row in rows]))
    tan_rms = math.sqrt(np.mean([row["tan_error_um"] ** 2 for row in rows]))
    slope = np.polyfit(
        [row["order"] for row in rows],
        [row["zemax_bfp_peak_mm"] for row in rows],
        1,
    )[0]
    pixel_pitch_um = 1000 * float(abs(x[1] - x[0]))
    max_sin_tan_separation_um = max(
        abs(row["tan_prediction_mm"] - row["sin_prediction_mm"]) * 1000
        for row in rows
    )
    summary = (
        f"fitted_order_spacing_mm={slope:.9f}\n"
        f"theoretical_order_spacing_mm={WAVELENGTH_MM * FOCAL_LENGTH_MM / PITCH_MM:.9f}\n"
        f"sin_mapping_rms_error_um={sin_rms:.6f}\n"
        f"tan_mapping_rms_error_um={tan_rms:.6f}\n"
        f"pop_pixel_pitch_um={pixel_pitch_um:.6f}\n"
        f"max_sin_tan_separation_um={max_sin_tan_separation_um:.6f}\n"
        "interpretation=The 1024-point UDA sampling bias exceeds the sin/tan separation; "
        "this POP run validates order blocking but is not a discriminating sin-vs-tan calibration.\n"
    )
    (output_dir / "zemax_peak_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
