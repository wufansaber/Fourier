from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class Grid:
    x_mm: np.ndarray
    y_mm: np.ndarray
    X_mm: np.ndarray
    Y_mm: np.ndarray
    nu_x_cpm: np.ndarray
    nu_y_cpm: np.ndarray
    NU_X_cpm: np.ndarray
    NU_Y_cpm: np.ndarray
    XI_mm: np.ndarray
    ETA_mm: np.ndarray
    dx_mm: float
    dnu_cpm: float
    dxi_mm: float
    fov_mm: float


@dataclass(frozen=True)
class ObjectCase:
    name: str
    reference_field: np.ndarray
    defective_field: np.ndarray
    defect_roi: np.ndarray
    filter_geometry: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_grid(cfg: dict[str, Any]) -> Grid:
    n_samples = int(cfg["grid"]["samples"])
    fov_mm = float(cfg["grid"]["object_fov_mm"])
    dx_mm = fov_mm / n_samples
    axis = (np.arange(n_samples) - n_samples / 2) * dx_mm
    X_mm, Y_mm = np.meshgrid(axis, axis, indexing="xy")

    freq = np.fft.fftshift(np.fft.fftfreq(n_samples, d=dx_mm))
    NU_X_cpm, NU_Y_cpm = np.meshgrid(freq, freq, indexing="xy")
    dnu_cpm = float(freq[1] - freq[0])

    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    f_obj_mm = float(cfg["optics"]["objective_focal_length_mm"])
    pupil_mag = float(cfg["optics"]["pupil_relay_magnification"])
    XI_mm = pupil_mag * wavelength_mm * f_obj_mm * NU_X_cpm
    ETA_mm = pupil_mag * wavelength_mm * f_obj_mm * NU_Y_cpm
    dxi_mm = pupil_mag * wavelength_mm * f_obj_mm * dnu_cpm

    return Grid(
        x_mm=axis,
        y_mm=axis,
        X_mm=X_mm,
        Y_mm=Y_mm,
        nu_x_cpm=freq,
        nu_y_cpm=freq,
        NU_X_cpm=NU_X_cpm,
        NU_Y_cpm=NU_Y_cpm,
        XI_mm=XI_mm,
        ETA_mm=ETA_mm,
        dx_mm=dx_mm,
        dnu_cpm=dnu_cpm,
        dxi_mm=dxi_mm,
        fov_mm=fov_mm,
    )


def _field_from_binary(binary: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    background = float(cfg["object"]["background_amplitude"])
    feature = float(cfg["object"]["feature_amplitude"])
    return (background + (feature - background) * binary).astype(np.complex128)


def make_stripe_case(grid: Grid, cfg: dict[str, Any]) -> ObjectCase:
    pitch = float(cfg["object"]["lattice_pitch_mm"])
    duty = float(cfg["object"]["stripe_duty_cycle"])
    wrapped_x = np.mod(grid.X_mm + pitch / 2, pitch) - pitch / 2
    binary_ref = (np.abs(wrapped_x) <= duty * pitch / 2).astype(float)

    binary_def = binary_ref.copy()
    missing_segment = (
        (np.abs(wrapped_x) <= duty * pitch / 2)
        & (np.abs(grid.X_mm) < pitch / 2)
        & (np.abs(grid.Y_mm) < 0.045)
    )
    binary_def[missing_segment] = 0.0

    extra_bar = (
        (np.abs(grid.X_mm - 0.073) < 0.004)
        & (np.abs(grid.Y_mm + 0.061) < 0.025)
    )
    binary_def[extra_bar] = 1.0
    defect_roi = missing_segment | extra_bar

    return ObjectCase(
        name="stripes",
        reference_field=_field_from_binary(binary_ref, cfg),
        defective_field=_field_from_binary(binary_def, cfg),
        defect_roi=defect_roi,
        filter_geometry="vertical_rods",
    )


def make_dot_case(grid: Grid, cfg: dict[str, Any]) -> ObjectCase:
    pitch = float(cfg["object"]["lattice_pitch_mm"])
    radius = float(cfg["object"]["dot_diameter_mm"]) / 2
    wrapped_x = np.mod(grid.X_mm + pitch / 2, pitch) - pitch / 2
    wrapped_y = np.mod(grid.Y_mm + pitch / 2, pitch) - pitch / 2
    binary_ref = ((wrapped_x**2 + wrapped_y**2) <= radius**2).astype(float)

    binary_def = binary_ref.copy()
    original_dot = (grid.X_mm**2 + grid.Y_mm**2) <= radius**2
    binary_def[original_dot] = 0.0

    shifted_x = 0.007
    shifted_y = 0.005
    shifted_dot = (
        (grid.X_mm - shifted_x) ** 2 + (grid.Y_mm - shifted_y) ** 2
    ) <= radius**2
    binary_def[shifted_dot] = 1.0

    extra_dot = (
        (grid.X_mm - 0.073) ** 2 + (grid.Y_mm + 0.061) ** 2
    ) <= (0.0045**2)
    binary_def[extra_dot] = 1.0
    defect_roi = original_dot | shifted_dot | extra_dot

    return ObjectCase(
        name="dots",
        reference_field=_field_from_binary(binary_ref, cfg),
        defective_field=_field_from_binary(binary_def, cfg),
        defect_roi=defect_roi,
        filter_geometry="local_spots",
    )


def make_case(name: str, grid: Grid, cfg: dict[str, Any]) -> ObjectCase:
    if name == "stripes":
        return make_stripe_case(grid, cfg)
    if name == "dots":
        return make_dot_case(grid, cfg)
    raise ValueError(f"Unsupported case: {name}")


def illumination_phase(grid: Grid, cfg: dict[str, Any]) -> np.ndarray:
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    index = float(cfg["optics"]["object_refractive_index"])
    angle_x = math.radians(float(cfg["optics"]["illumination_angle_x_deg"]))
    angle_y = math.radians(float(cfg["optics"]["illumination_angle_y_deg"]))
    nu_sx = index * math.sin(angle_x) / wavelength_mm
    nu_sy = index * math.sin(angle_y) / wavelength_mm
    return np.exp(1j * 2 * np.pi * (nu_sx * grid.X_mm + nu_sy * grid.Y_mm))


def make_pupil(grid: Grid, cfg: dict[str, Any]) -> np.ndarray:
    f_obj = float(cfg["optics"]["objective_focal_length_mm"])
    na = float(cfg["optics"]["numerical_aperture"])
    index = float(cfg["optics"]["object_refractive_index"])
    pupil_mag = float(cfg["optics"]["pupil_relay_magnification"])
    rho = np.hypot(grid.XI_mm, grid.ETA_mm)
    pupil_radius = pupil_mag * f_obj * na
    pupil = (rho <= pupil_radius).astype(float)

    if bool(cfg["optics"].get("use_aplanatic_apodization", False)):
        sin_theta = np.clip(rho / (pupil_mag * f_obj * index), 0, 1)
        cos_theta = np.sqrt(np.maximum(1 - sin_theta**2, 0))
        pupil *= np.sqrt(cos_theta)
    return pupil.astype(np.complex128)


def _raised_cosine_notch(distance: np.ndarray, width: float, transition: float) -> np.ndarray:
    inner = width / 2
    if transition <= 0:
        return (distance > inner).astype(float)
    result = np.ones_like(distance, dtype=float)
    result[distance <= inner] = 0.0
    ramp = (distance > inner) & (distance < inner + transition)
    q = (distance[ramp] - inner) / transition
    result[ramp] = 0.5 - 0.5 * np.cos(np.pi * q)
    return result


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    """Return a unit-amplitude triangular wave for straight-sided teeth."""
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _vertical_rod(
    grid: Grid,
    center: float,
    width: float,
    edge: str,
    cfg: dict[str, Any],
) -> np.ndarray:
    if edge == "hard":
        return (np.abs(grid.XI_mm - center) > width / 2).astype(float)
    if edge == "soft":
        transition = float(cfg["filter"]["soft_transition_mm"])
        return _raised_cosine_notch(np.abs(grid.XI_mm - center), width, transition)
    if edge == "serrated":
        pitch = float(cfg["filter"]["serration_pitch_mm"])
        amplitude = float(cfg["filter"]["serration_amplitude_mm"])
        half_width = width / 2 + amplitude * _triangle_wave(2 * np.pi * grid.ETA_mm / pitch)
        return (np.abs(grid.XI_mm - center) > np.maximum(half_width, 0)).astype(float)
    raise ValueError(f"Unsupported edge: {edge}")


def _local_stop(
    grid: Grid,
    cx: float,
    cy: float,
    diameter: float,
    edge: str,
    cfg: dict[str, Any],
) -> np.ndarray:
    radial = np.hypot(grid.XI_mm - cx, grid.ETA_mm - cy)
    if edge == "hard":
        return (radial > diameter / 2).astype(float)
    if edge == "soft":
        transition = float(cfg["filter"]["soft_transition_mm"])
        return _raised_cosine_notch(radial, diameter, transition)
    if edge == "serrated":
        angle = np.arctan2(grid.ETA_mm - cy, grid.XI_mm - cx)
        teeth = max(8, int(round(np.pi * diameter / float(cfg["filter"]["serration_pitch_mm"]))))
        amplitude = float(cfg["filter"]["serration_amplitude_mm"])
        boundary = diameter / 2 + amplitude * _triangle_wave(teeth * angle)
        return (radial > np.maximum(boundary, 0)).astype(float)
    raise ValueError(f"Unsupported edge: {edge}")


def make_filter(
    grid: Grid,
    case: ObjectCase,
    cfg: dict[str, Any],
    edge: str,
    width_mm: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    wavelength = float(cfg["optics"]["wavelength_mm"])
    f_obj = float(cfg["optics"]["objective_focal_length_mm"])
    pupil_mag = float(cfg["optics"]["pupil_relay_magnification"])
    pitch = float(cfg["object"]["lattice_pitch_mm"])
    spacing = pupil_mag * wavelength * f_obj / pitch
    pupil_radius = pupil_mag * f_obj * float(cfg["optics"]["numerical_aperture"])
    max_order_cfg = int(cfg["filter"]["max_order"])
    max_order = min(max_order_cfg, int(math.floor(pupil_radius / spacing)))
    block_zero = bool(cfg["filter"]["block_zero_order"])
    if width_mm is not None:
        width = float(width_mm)
    elif edge == "serrated":
        width = float(cfg["filter"]["serrated_nominal_width_mm"])
    else:
        width = float(cfg["filter"]["hard_width_mm"])

    transmission = np.ones_like(grid.XI_mm, dtype=float)
    centers: list[tuple[float, float]] = []

    if case.filter_geometry == "vertical_rods":
        for m in range(-max_order, max_order + 1):
            if m == 0 and not block_zero:
                continue
            cx = m * spacing
            centers.append((cx, 0.0))
            if edge == "serrated" and bool(cfg["filter"].get("use_two_deck_serrated_rods", False)):
                # Reference assembly: five fuzzy rods and six linear rods.
                # For orders -5...+5, even orders (including zero/specular)
                # form the five-rod fuzzy deck; odd orders form the six-rod
                # linear deck.  This doubles the within-deck center spacing.
                if m % 2 == 0:
                    transmission *= _vertical_rod(grid, cx, width, "serrated", cfg)
                else:
                    transmission *= _vertical_rod(
                        grid,
                        cx,
                        float(cfg["filter"]["hard_width_mm"]),
                        "hard",
                        cfg,
                    )
            else:
                transmission *= _vertical_rod(grid, cx, width, edge, cfg)
    elif case.filter_geometry == "local_spots":
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                if m == 0 and n == 0 and not block_zero:
                    continue
                cx = m * spacing
                cy = n * spacing
                if math.hypot(cx, cy) <= pupil_radius:
                    centers.append((cx, cy))

        nearest_m = np.rint(grid.XI_mm / spacing).astype(int)
        nearest_n = np.rint(grid.ETA_mm / spacing).astype(int)
        center_x = nearest_m * spacing
        center_y = nearest_n * spacing
        valid_center = (
            (np.abs(nearest_m) <= max_order)
            & (np.abs(nearest_n) <= max_order)
            & (np.hypot(center_x, center_y) <= pupil_radius)
        )
        if not block_zero:
            valid_center &= ~((nearest_m == 0) & (nearest_n == 0))
        local_x = grid.XI_mm - center_x
        local_y = grid.ETA_mm - center_y
        radial = np.hypot(local_x, local_y)

        if edge == "hard":
            local_transmission = (radial > width / 2).astype(float)
        elif edge == "soft":
            local_transmission = _raised_cosine_notch(
                radial,
                width,
                float(cfg["filter"]["soft_transition_mm"]),
            )
        elif edge == "serrated":
            angle = np.arctan2(local_y, local_x)
            teeth = max(
                8,
                int(
                    round(
                        np.pi
                        * width
                        / float(cfg["filter"]["serration_pitch_mm"])
                    )
                ),
            )
            boundary = width / 2 + float(
                cfg["filter"]["serration_amplitude_mm"]
            ) * _triangle_wave(teeth * angle)
            local_transmission = (radial > np.maximum(boundary, 0)).astype(float)
        else:
            raise ValueError(f"Unsupported edge: {edge}")
        transmission = np.where(valid_center, local_transmission, 1.0)
    else:
        raise ValueError(case.filter_geometry)

    return transmission.astype(np.complex128), centers


def fft2c(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho"))


def ifft2c(spectrum: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho"))


def simulate_field(
    field: np.ndarray,
    phase: np.ndarray,
    pupil: np.ndarray,
    transmission: np.ndarray,
) -> dict[str, np.ndarray]:
    object_field = field * phase
    spectrum = fft2c(object_field)
    pupil_spectrum = spectrum * pupil
    filtered_spectrum = pupil_spectrum * transmission
    image_field = ifft2c(filtered_spectrum)
    return {
        "object_field": object_field,
        "spectrum": spectrum,
        "pupil_spectrum": pupil_spectrum,
        "filtered_spectrum": filtered_spectrum,
        "image_field": image_field,
        "intensity": np.abs(image_field) ** 2,
    }


def compute_metrics(
    ref_unfiltered: np.ndarray,
    defect_unfiltered: np.ndarray,
    ref_filtered: np.ndarray,
    defect_filtered: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float]:
    eps = np.finfo(float).eps
    rms_ref_0 = float(np.sqrt(np.mean(ref_unfiltered**2)))
    rms_ref_h = float(np.sqrt(np.mean(ref_filtered**2)))
    delta_0 = defect_unfiltered - ref_unfiltered
    delta_h = defect_filtered - ref_filtered
    signal_0 = float(np.sqrt(np.mean(delta_0[roi] ** 2)))
    signal_h = float(np.sqrt(np.mean(delta_h[roi] ** 2)))
    visibility_0 = signal_0 / (rms_ref_0 + eps)
    visibility_h = signal_h / (rms_ref_h + eps)
    return {
        "periodic_rms_unfiltered": rms_ref_0,
        "periodic_rms_filtered": rms_ref_h,
        "periodic_suppression_db": 20 * math.log10((rms_ref_0 + eps) / (rms_ref_h + eps)),
        "defect_signal_unfiltered": signal_0,
        "defect_signal_filtered": signal_h,
        "defect_signal_retention": signal_h / (signal_0 + eps),
        "visibility_unfiltered": visibility_0,
        "visibility_filtered": visibility_h,
        "visibility_gain": visibility_h / (visibility_0 + eps),
        "defective_energy_throughput": float(np.sum(defect_filtered) / (np.sum(defect_unfiltered) + eps)),
    }


def _imshow(ax: plt.Axes, data: np.ndarray, extent: list[float], title: str, cmap: str = "viridis") -> None:
    # Matplotlib keeps an internal copy of every image.  A 3x4 diagnostic
    # figure made directly from twelve 1024x1024 arrays can therefore exceed
    # 1 GB during savefig.  Display-only decimation leaves the numerical
    # simulation and exported metrics at full resolution while keeping the
    # report figure reliable on ordinary engineering workstations.
    stride = max(1, int(math.ceil(max(data.shape) / 512)))
    display_data = data[::stride, ::stride]
    image = ax.imshow(display_data, extent=extent, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def plot_case(
    output_path: Path,
    grid: Grid,
    case: ObjectCase,
    pupil: np.ndarray,
    transmission: np.ndarray,
    ref_0: dict[str, np.ndarray],
    defect_0: dict[str, np.ndarray],
    ref_h: dict[str, np.ndarray],
    defect_h: dict[str, np.ndarray],
    metrics: dict[str, float],
    edge: str,
) -> None:
    obj_extent = [
        grid.x_mm[0] * 1e3,
        grid.x_mm[-1] * 1e3,
        grid.y_mm[0] * 1e3,
        grid.y_mm[-1] * 1e3,
    ]
    bfp_extent = [
        grid.XI_mm[0, 0],
        grid.XI_mm[0, -1],
        grid.ETA_mm[0, 0],
        grid.ETA_mm[-1, 0],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    _imshow(axes[0, 0], np.abs(case.reference_field), obj_extent, "Reference amplitude", "gray")
    _imshow(axes[0, 1], np.abs(case.defective_field), obj_extent, "Defective amplitude", "gray")
    _imshow(
        axes[0, 2],
        np.abs(case.defective_field - case.reference_field),
        obj_extent,
        "Non-periodic target |delta U|",
        "magma",
    )
    log_spectrum = np.log10(np.abs(defect_0["spectrum"]) ** 2 + 1e-12)
    _imshow(axes[0, 3], log_spectrum, bfp_extent, "Objective Fourier plane (log I)")

    _imshow(axes[1, 0], np.abs(pupil), bfp_extent, "Objective pupil", "gray")
    _imshow(axes[1, 1], np.abs(transmission), bfp_extent, f"Fourier filter H ({edge})", "gray")
    log_filtered = np.log10(np.abs(defect_h["filtered_spectrum"]) ** 2 + 1e-12)
    _imshow(axes[1, 2], log_filtered, bfp_extent, "Spectrum after filtering (log I)")
    center = len(grid.x_mm) // 2
    axes[1, 3].plot(grid.XI_mm[center, :], log_spectrum[center, :], label="before")
    axes[1, 3].plot(grid.XI_mm[center, :], log_filtered[center, :], label="after")
    axes[1, 3].set_xlim(-6, 6)
    axes[1, 3].set_xlabel("Fourier-plane x (mm)")
    axes[1, 3].set_ylabel("log10 intensity")
    axes[1, 3].set_title("Central Fourier-plane profile", fontsize=9)
    axes[1, 3].legend()
    axes[1, 3].grid(alpha=0.25)

    _imshow(axes[2, 0], ref_0["intensity"], obj_extent, "Reference image, no rods")
    _imshow(
        axes[2, 1],
        np.abs(defect_0["intensity"] - ref_0["intensity"]),
        obj_extent,
        "Defect difference, no rods",
        "magma",
    )
    _imshow(axes[2, 2], ref_h["intensity"], obj_extent, "Reference image, filtered")
    _imshow(
        axes[2, 3],
        np.abs(defect_h["intensity"] - ref_h["intensity"]),
        obj_extent,
        "Defect difference, filtered",
        "magma",
    )

    for ax in axes[0, :3]:
        ax.set_xlabel("object x (µm)")
        ax.set_ylabel("object y (µm)")
    axes[0, 3].set_xlabel("Fourier-plane x (mm)")
    axes[0, 3].set_ylabel("Fourier-plane y (mm)")
    for ax in axes[1, :3]:
        ax.set_xlabel("Fourier-plane x (mm)")
        ax.set_ylabel("Fourier-plane y (mm)")
    for ax in axes[2, :]:
        ax.set_xlabel("object x (µm)")
        ax.set_ylabel("object y (µm)")

    fig.suptitle(
        f"{case.name} / {edge}: suppression={metrics['periodic_suppression_db']:.1f} dB, "
        f"visibility gain={metrics['visibility_gain']:.2f}x, "
        f"throughput={metrics['defective_energy_throughput']:.3f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_mapping(output_path: Path, cfg: dict[str, Any]) -> None:
    f_obj = float(cfg["optics"]["objective_focal_length_mm"])
    index = float(cfg["optics"]["object_refractive_index"])
    na = float(cfg["optics"]["numerical_aperture"])
    wavelength = float(cfg["optics"]["wavelength_mm"])
    pitch = float(cfg["object"]["lattice_pitch_mm"])
    s = np.linspace(0, na / index, 500)
    theta = np.arcsin(s)
    rho_sin = f_obj * index * s
    rho_tan = f_obj * index * np.tan(theta)
    relative_error = np.zeros_like(s)
    relative_error[1:] = rho_tan[1:] / rho_sin[1:] - 1

    max_order = int(math.floor((na / wavelength) * pitch))
    orders = np.arange(-max_order, max_order + 1)
    order_s = orders * wavelength / (index * pitch)
    valid = np.abs(order_s) <= na / index
    orders = orders[valid]
    order_s = order_s[valid]
    order_rho = f_obj * index * order_s

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(np.degrees(theta), rho_sin, label=r"$f n\sin\theta$")
    axes[0].plot(np.degrees(theta), rho_tan, "--", label=r"$f n\tan\theta$")
    axes[0].scatter(np.degrees(np.arcsin(np.abs(order_s))), np.abs(order_rho), s=22, label="array orders")
    axes[0].set_xlabel("Diffraction angle (deg)")
    axes[0].set_ylabel("Fourier-plane radius (mm)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(s, relative_error * 100)
    axes[1].axvline(na / index, color="r", linestyle="--", label=f"NA/n={na/index:.3f}")
    axes[1].set_xlabel(r"$\sin\theta=NA_{local}/n$")
    axes[1].set_ylabel("tan-vs-sin position error (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("Sine-condition Fourier mapping versus geometric tangent projection")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sweep(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    cases = sorted({str(row["case"]) for row in rows})
    for case_name in cases:
        subset = [row for row in rows if row["case"] == case_name]
        widths = [float(row["width_mm"]) for row in subset]
        axes[0].plot(widths, [float(row["periodic_suppression_db"]) for row in subset], "o-", label=case_name)
        axes[1].plot(widths, [float(row["visibility_gain"]) for row in subset], "o-", label=case_name)
        axes[2].plot(widths, [float(row["defective_energy_throughput"]) for row in subset], "o-", label=case_name)
    axes[0].set_ylabel("Periodic suppression (dB)")
    axes[1].set_ylabel("Defect visibility gain (x)")
    axes[2].set_ylabel("Energy throughput")
    for ax in axes:
        ax.set_xlabel("Stop/rod width (mm)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Fourier-filter width trade-off (hard edge)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, output_dir: Path) -> None:
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(cfg)
    order_spacing = (
        float(cfg["optics"]["pupil_relay_magnification"])
        * float(cfg["optics"]["wavelength_mm"])
        * float(cfg["optics"]["objective_focal_length_mm"])
        / float(cfg["object"]["lattice_pitch_mm"])
    )
    same_deck_spacing = 2 * order_spacing
    if bool(cfg["filter"].get("use_two_deck_serrated_rods", False)):
        if same_deck_spacing < float(cfg["filter"]["minimum_same_deck_spacing_mm"]):
            raise ValueError("Two-deck rod layout violates the same-deck minimum spacing")
        if order_spacing < float(cfg["filter"]["minimum_cross_deck_spacing_mm"]):
            raise ValueError("Two-deck rod layout violates the cross-deck minimum spacing")
    phase = illumination_phase(grid, cfg)
    pupil = make_pupil(grid, cfg)
    unity_filter = np.ones_like(pupil)

    all_metrics: list[dict[str, Any]] = []
    cached_cases: dict[str, tuple[ObjectCase, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    for case_name in cfg["run"]["cases"]:
        print(f"[case] preparing {case_name}", flush=True)
        case = make_case(str(case_name), grid, cfg)
        ref_0 = simulate_field(case.reference_field, phase, pupil, unity_filter)
        defect_0 = simulate_field(case.defective_field, phase, pupil, unity_filter)
        # Only these products are required by later plots/metrics.  Releasing
        # redundant complex arrays prevents the diagnostic figures from
        # competing with the FFT work arrays for memory.
        ref_0 = {"intensity": ref_0["intensity"]}
        defect_0 = {
            "spectrum": defect_0["spectrum"],
            "intensity": defect_0["intensity"],
        }
        cached_cases[case.name] = (case, ref_0, defect_0)

        for edge in cfg["run"]["filter_edges"]:
            print(f"[case] {case.name} / {edge}", flush=True)
            transmission, centers = make_filter(grid, case, cfg, str(edge))
            ref_h = simulate_field(case.reference_field, phase, pupil, transmission)
            defect_h = simulate_field(case.defective_field, phase, pupil, transmission)
            metrics = compute_metrics(
                ref_0["intensity"],
                defect_0["intensity"],
                ref_h["intensity"],
                defect_h["intensity"],
                case.defect_roi,
            )
            record = {
                "case": case.name,
                "edge": str(edge),
                "width_mm": float(
                    cfg["filter"]["serrated_nominal_width_mm"]
                    if str(edge) == "serrated"
                    else cfg["filter"]["hard_width_mm"]
                ),
                "blocked_elements": len(centers),
                **metrics,
            }
            all_metrics.append(record)
            ref_h = {"intensity": ref_h["intensity"]}
            defect_h = {
                "filtered_spectrum": defect_h["filtered_spectrum"],
                "intensity": defect_h["intensity"],
            }
            plot_case(
                output_dir / f"case_{case.name}_{edge}.png",
                grid,
                case,
                pupil,
                transmission,
                ref_0,
                defect_0,
                ref_h,
                defect_h,
                metrics,
                str(edge),
            )
            del transmission, ref_h, defect_h
            gc.collect()

    sweep_rows: list[dict[str, Any]] = []
    for case_name, (case, ref_0, defect_0) in cached_cases.items():
        print(f"[sweep] {case_name}", flush=True)
        for width in cfg["run"]["sweep_widths_mm"]:
            transmission, centers = make_filter(grid, case, cfg, "hard", float(width))
            ref_h = simulate_field(case.reference_field, phase, pupil, transmission)
            defect_h = simulate_field(case.defective_field, phase, pupil, transmission)
            metrics = compute_metrics(
                ref_0["intensity"],
                defect_0["intensity"],
                ref_h["intensity"],
                defect_h["intensity"],
                case.defect_roi,
            )
            sweep_rows.append(
                {
                    "case": case_name,
                    "edge": "hard",
                    "width_mm": float(width),
                    "blocked_elements": len(centers),
                    **metrics,
                }
            )
            del transmission, ref_h, defect_h
            gc.collect()

    print("[plots] mapping and width sweep", flush=True)
    plot_mapping(output_dir / "sin_tan_mapping.png", cfg)
    plot_sweep(output_dir / "filter_width_sweep.png", sweep_rows)
    write_csv(output_dir / "case_metrics.csv", all_metrics)
    write_csv(output_dir / "width_sweep_metrics.csv", sweep_rows)

    derived = {
        "object_sampling_mm": grid.dx_mm,
        "frequency_sampling_cycles_per_mm": grid.dnu_cpm,
        "fourier_plane_sampling_mm": grid.dxi_mm,
        "pupil_radius_mm": float(cfg["optics"]["objective_focal_length_mm"])
        * float(cfg["optics"]["numerical_aperture"])
        * float(cfg["optics"]["pupil_relay_magnification"]),
        "coherent_cutoff_cycles_per_mm": float(cfg["optics"]["numerical_aperture"])
        / float(cfg["optics"]["wavelength_mm"]),
        "image_fov_mm": float(cfg["grid"]["object_fov_mm"])
        * float(cfg["optics"]["magnification"]),
        "tube_lens_focal_length_check_mm": float(cfg["optics"]["objective_focal_length_mm"])
        * float(cfg["optics"]["magnification"]),
        "order_spacing_mm": order_spacing,
        "same_deck_order_spacing_mm": same_deck_spacing,
        "linear_rod_width_mm": float(cfg["filter"]["hard_width_mm"]),
        "fuzzy_rod_min_width_mm": float(cfg["filter"]["fuzzy_min_width_mm"]),
        "fuzzy_rod_max_width_mm": float(cfg["filter"]["fuzzy_max_width_mm"]),
        "serration_pitch_mm": float(cfg["filter"]["serration_pitch_mm"]),
        "minimum_same_deck_spacing_mm": float(cfg["filter"]["minimum_same_deck_spacing_mm"]),
        "minimum_cross_deck_spacing_mm": float(cfg["filter"]["minimum_cross_deck_spacing_mm"]),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"derived": derived, "cases": all_metrics}, stream, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(output_dir), "derived": derived}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coherent microscope Fourier-filter simulation")
    parser.add_argument("--config", type=Path, default=Path("config/default.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/python"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.output_dir)
