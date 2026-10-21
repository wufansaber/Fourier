"""Optimize A/B-deck one-sided serration depths for the confirmed 2 um case.

The opaque core remains 1.0 mm wide.  Deck A carries the even active orders
(-2, 0, +2), deck B carries the odd active orders (-3, -1, +1, +3).
Only the serration k=0 equivalent amplitude is propagated because the confirmed
25-stage TDI acceptance rejects k != 0 along image Y.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ch3_confirmed_geometry_rod_tradeoff as trade
import ch3_oblique75_confirmed_geometry as base


OUT = base.ROOT / "results" / "ab_depth_optimization"
DEPTHS_MM = np.arange(0.5, 2.0001, 0.25)
CORE_MM = 1.0
TOOTH_PITCH_MM = 0.75
ACTIVE_ORDERS = np.arange(-3, 4)


def layered_k0_mask(x_mechanical: np.ndarray, y_bfp: np.ndarray, spacing_mm: float,
                    depth_a_mm: float, depth_b_mm: float) -> np.ndarray:
    """Return the TDI-accepted k=0 mask for independent A/B edge depths."""
    transmission_x = np.ones_like(x_mechanical, dtype=np.float32)
    for order in ACTIVE_ORDERS:
        depth = depth_a_mm if order % 2 == 0 else depth_b_mm
        centre = float(order) * spacing_mm
        distance = np.abs(x_mechanical - centre)
        one = np.ones_like(x_mechanical, dtype=np.float32)
        one[distance <= CORE_MM / 2] = 0.0
        edge = (distance > CORE_MM / 2) & (distance < CORE_MM / 2 + depth)
        one[edge] = (distance[edge] - CORE_MM / 2) / depth
        transmission_x *= one
    return np.repeat(transmission_x[None, :], y_bfp.size, axis=0)


def evaluate_case(model: dict, depth_a: float, depth_b: float,
                  references: dict) -> tuple[dict, np.ndarray]:
    mask = layered_k0_mask(model["x_mechanical"], model["y_bfp"], model["spacing"],
                           depth_a, depth_b)
    pattern_image = np.abs(base.ift2(model["pattern_spec"] * mask)) ** 2
    particle_image = np.abs(base.ift2(model["particles_spec"] * mask)) ** 2
    combined_image = np.abs(base.ift2(model["combined_spec"] * mask)) ** 2
    single_image = np.abs(base.ift2(model["single_specs"][1] * mask)) ** 2
    replica_db, outside_fraction = trade.ringing_metrics(single_image, model["coordinate"])
    row = {
        "depth_a_mm": float(depth_a),
        "depth_b_mm": float(depth_b),
        "envelope_a_mm": float(CORE_MM + 2 * depth_a),
        "envelope_b_mm": float(CORE_MM + 2 * depth_b),
        "background_suppression_db": float(-10 * np.log10(max(
            trade.energy_ratio(pattern_image, references["pattern"]), 1e-30))),
        "particle_energy_retention_percent": float(100 * trade.energy_ratio(
            particle_image, references["particle"])),
        "raw_local_particle_contrast_db": float(trade.local_particle_contrast_db(
            combined_image, model["coordinate"], model["particles"])),
        "worst_x_replica_db": float(replica_db),
        "ringing_energy_outside_core_percent": float(100 * outside_fraction),
    }
    return row, combined_image


def normalise(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.ones_like(values)
    result = (values - low) / (high - low)
    return result if higher_is_better else 1.0 - result


def assign_balanced_score(rows: list[dict]) -> None:
    """Equal-distance knee score across suppression, retention and two ringing metrics."""
    suppression = normalise(np.asarray([min(r["background_suppression_db"], 60.0)
                                         for r in rows]))
    retention = normalise(np.asarray([r["particle_energy_retention_percent"] for r in rows]))
    replica = normalise(np.asarray([r["worst_x_replica_db"] for r in rows]), False)
    outside = normalise(np.asarray([r["ringing_energy_outside_core_percent"] for r in rows]), False)
    # Background is already strongly suppressed in the matched 2 um case.  Give
    # the two defect-preservation/ringing axes most of the selection weight.
    weights = np.asarray([0.15, 0.35, 0.30, 0.20])
    desirabilities = np.column_stack([suppression, retention, replica, outside])
    scores = 1.0 - np.sqrt(np.sum(weights * (1.0 - desirabilities) ** 2, axis=1) /
                           np.sum(weights))
    for row, score in zip(rows, scores):
        row["balanced_score"] = float(score)


def plot_heatmaps(rows: list[dict], best: dict) -> None:
    lookup = {(r["depth_a_mm"], r["depth_b_mm"]): r for r in rows}
    quantities = [
        ("particle_energy_retention_percent", "Particle energy retention (%)", "viridis"),
        ("worst_x_replica_db", "Worst X replica (dB; lower is better)", "viridis_r"),
        ("ringing_energy_outside_core_percent", "Ringing energy outside core (%)", "viridis_r"),
        ("balanced_score", "Balanced desirability score", "viridis"),
    ]
    extent = [DEPTHS_MM[0] - 0.125, DEPTHS_MM[-1] + 0.125,
              DEPTHS_MM[0] - 0.125, DEPTHS_MM[-1] + 0.125]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.4), constrained_layout=True)
    for axis, (key, title, cmap) in zip(axes.flat, quantities):
        matrix = np.asarray([[lookup[(float(a), float(b))][key] for a in DEPTHS_MM]
                             for b in DEPTHS_MM])
        shown = axis.imshow(matrix, origin="lower", extent=extent, cmap=cmap,
                            interpolation="nearest", aspect="equal")
        axis.scatter(best["depth_a_mm"], best["depth_b_mm"], marker="*", s=220,
                     facecolor="#FF3B30", edgecolor="white", linewidth=1.0, zorder=5)
        axis.set(title=title, xlabel="Deck A one-sided tooth depth b_A (mm)",
                 ylabel="Deck B one-sided tooth depth b_B (mm)",
                 xticks=DEPTHS_MM, yticks=DEPTHS_MM)
        axis.tick_params(axis="x", rotation=45)
        fig.colorbar(shown, ax=axis, fraction=0.046)
    fig.suptitle("1.0 mm opaque core | A/B serration-depth scan on the same 2 um particle case\n"
                 "Red star: balanced optimum; 0.25 mm grid; no background subtraction")
    path = OUT / "ab_depth_metric_heatmaps.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_best_comparison(model: dict, images: dict[str, np.ndarray], best: dict) -> None:
    coordinate = model["coordinate"]
    roi = np.abs(coordinate) <= 0.016
    extent = [-16, 16, -16, 16]
    reference = max(float(images["open"][np.ix_(roi, roi)].max()), 1e-30)
    cases = [
        ("No rods", images["open"]),
        ("Largest depth: b_A=b_B=2.00 mm", images["largest"]),
        (f"Balanced optimum: b_A={best['depth_a_mm']:.2f}, b_B={best['depth_b_mm']:.2f} mm",
         images["best"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), constrained_layout=True)
    linear_shown = db_shown = None
    for column, (label, image) in enumerate(cases):
        raw = image[np.ix_(roi, roi)] / reference
        linear_shown = axes[0, column].imshow(
            raw, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=1,
            interpolation="nearest")
        db_shown = axes[1, column].imshow(
            10 * np.log10(np.maximum(raw, 1e-5)), origin="lower", extent=extent,
            cmap="turbo", vmin=-50, vmax=0, interpolation="nearest")
        axes[0, column].set_title(label + " | linear")
        axes[1, column].set_title(label + " | dB")
        for row in range(2):
            axes[row, column].set(xlabel="X: camera long direction (um)",
                                  ylabel="Y: TDI scan direction (um)")
            axes[row, column].set_aspect("equal", adjustable="box")
    fig.colorbar(linear_shown, ax=axes[0, :], fraction=0.025,
                 label="Raw irradiance / one unfiltered reference (common 0-1 scale)")
    fig.colorbar(db_shown, ax=axes[1, :], fraction=0.025,
                 label="Raw irradiance (dB re same unfiltered reference; -50...0 dB)")
    fig.suptitle("A/B serration-depth optimization on exactly the same defect field\n"
                 "One brightness reference for every panel; no subtraction or filtered-image renormalisation")
    path = OUT / "ab_depth_best_same_brightness.png"
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = trade.build_model()
    references = {
        "pattern": np.abs(base.ift2(model["pattern_spec"])) ** 2,
        "particle": np.abs(base.ift2(model["particles_spec"])) ** 2,
    }
    open_combined = np.abs(base.ift2(model["combined_spec"])) ** 2
    rows: list[dict] = []
    image_lookup: dict[tuple[float, float], np.ndarray] = {}
    for depth_a in DEPTHS_MM:
        for depth_b in DEPTHS_MM:
            row, image = evaluate_case(model, float(depth_a), float(depth_b), references)
            rows.append(row)
            image_lookup[(float(depth_a), float(depth_b))] = image
    assign_balanced_score(rows)
    best = max(rows, key=lambda row: row["balanced_score"])
    baseline = next(row for row in rows
                    if row["depth_a_mm"] == 0.5 and row["depth_b_mm"] == 0.5)
    plot_heatmaps(rows, best)
    plot_best_comparison(
        model,
        {"open": open_combined,
         "largest": image_lookup[(2.0, 2.0)],
         "best": image_lookup[(best["depth_a_mm"], best["depth_b_mm"])]},
        best,
    )
    with (OUT / "ab_depth_scan.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "definition": {
            "core_full_width_mm": CORE_MM,
            "deck_A_orders": [-2, 0, 2],
            "deck_B_orders": [-3, -1, 1, 3],
            "tooth_pitch_mm": TOOTH_PITCH_MM,
            "depth_scan_mm": DEPTHS_MM.tolist(),
            "object_case": "same confirmed Ch3 75-degree illumination, 2 um periodic pattern and particle field",
            "display_rule": "one unfiltered reference; no subtraction; no filtered-image renormalisation",
            "balanced_score_weights": {
                "background_suppression": 0.15,
                "particle_energy_retention": 0.35,
                "worst_X_replica": 0.30,
                "ringing_energy_outside_core": 0.20,
            },
        },
        "baseline": baseline,
        "balanced_optimum": best,
        "cases_evaluated": len(rows),
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
