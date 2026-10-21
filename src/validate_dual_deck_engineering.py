"""Fast consistency checks for dual_deck_engineering outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dual_deck_engineering"


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    derived = summary["derived"]
    assert abs(derived["true_pupil_diameter_mm"] - 36.0) < 1e-9
    assert abs(derived["raw_order_spacing_mm"] - 0.5325) < 1e-9
    assert abs(derived["expanded_order_spacing_mm"] - 1.5) < 1e-9
    assert abs(derived["same_deck_spacing_mm"] - 3.0) < 1e-9
    assert abs(derived["nominal_same_deck_clearance_mm"] - 1.0) < 1e-9
    assert derived["expanded_pupil_y_mm"] > 100
    assert derived["tdi_max_integer_rows"] == 26
    cases = {row["case"]: row for row in summary["simulation_cases"]}
    assert "serrated, balanced, 4 mm gap" in cases
    assert cases["serrated, balanced, 4 mm gap"]["background_suppression_db"] >= 10
    assert cases["serrated, balanced, 4 mm gap"]["energy_retention"] >= 0.6
    assert summary["order_subset_search"]["candidate_count"] == 163
    assert summary["order_subset_search"]["balanced_gate_count"] >= 1
    with (OUT / "period_retuning_clearance.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row20 = next(row for row in rows if float(row["object_period_um"]) == 20)
    assert abs(float(row20["expanded_adjacent_order_spacing_mm"]) - 1.5) < 1e-9
    for name in [
        "dual_deck_layout.png",
        "dual_deck_layout.svg",
        "dual_deck_structure_schematic.png",
        "dual_deck_structure_schematic.svg",
        "dual_deck_simulation_validation.png",
        "dual_deck_simulation_validation.svg",
        "dual_deck_period_tuning.png",
        "dual_deck_period_tuning.svg",
        "simulation_metrics.csv",
        "order_subset_search.csv",
    ]:
        assert (OUT / name).exists(), name
    print("PASS: dual-deck 1.5 mm spacing, nominal clearance, TDI limit, simulation metrics and figures checked")


if __name__ == "__main__":
    main()
