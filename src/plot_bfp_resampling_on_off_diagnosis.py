"""Compare real-objective BFP profiles with refraction resampling on and off."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPACING_MM = 5.165663367412653
SINE_SLOPE_MM = 29.102328830493818
WAVELENGTH_MM = 0.000355
NA = 0.60


def profile(path: Path):
    data = np.load(path / "zemax_native_pop_2um.npz")
    y = data["open_y_mm"]
    line = np.sum(data["open_irradiance_W_mm2"], axis=1)
    return y, line / max(float(line.max()), 1e-300)


def separated_peaks(y, values, count=8, separation_mm=0.5):
    candidates = np.where(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    candidates = candidates[np.argsort(values[candidates])[::-1]]
    selected = []
    for index in candidates:
        if all(abs(y[index] - y[other]) > separation_mm for other in selected):
            selected.append(int(index))
        if len(selected) >= count:
            break
    return selected


def python_spectrum(path: Path):
    data = np.load(path / "python_reference_2um.npz")
    field = data["object_field"]
    coordinate = data["coordinate_mm"]
    step = float(coordinate[1] - coordinate[0])
    frequency = np.fft.fftshift(np.fft.fftfreq(field.shape[0], d=step))
    fx, fy = np.meshgrid(frequency, frequency)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(field), norm="ortho"))) ** 2
    spectrum *= (WAVELENGTH_MM * np.hypot(fx, fy) <= NA)
    line = np.sum(spectrum, axis=1)
    line /= max(float(line.max()), 1e-300)
    return SINE_SLOPE_MM * WAVELENGTH_MM * frequency, line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--off-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    y_on, p_on = profile(args.on_dir.resolve())
    y_off, p_off = profile(args.off_dir.resolve())
    y_python, p_python = python_spectrum(args.on_dir.resolve())
    expected = np.arange(-3, 4) * SPACING_MM
    ratios_on = [float(p_on[np.argmin(abs(y_on - value))]) for value in expected]
    ratios_off = [float(p_off[np.argmin(abs(y_off - value))]) for value in expected]
    ratios_python = [float(p_python[np.argmin(abs(y_python - value))])
                     for value in expected]
    off_peaks = separated_peaks(y_off, p_off)

    fig, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    axis.plot(y_python, 10 * np.log10(np.maximum(p_python, 1e-14)), lw=1.35,
              color="#2a9d8f", label="Python input spectrum: correct diffraction orders")
    axis.plot(y_on, 10 * np.log10(np.maximum(p_on, 1e-14)), lw=1.5,
              label="Resample after refraction ON: high frequency erased")
    axis.plot(y_off, 10 * np.log10(np.maximum(p_off, 1e-14)), lw=1.25,
              label="Resample after refraction OFF: aliased peaks")
    for order, position in zip(range(-3, 4), expected):
        axis.axvline(position, color="#d1495b", lw=0.8, alpha=0.7)
        axis.text(position, -137, f"m={order}", rotation=90,
                  ha="center", va="bottom", fontsize=8)
    axis.set(xlim=(-18, 18), ylim=(-140, 3),
             xlabel="Real-objective BFP Y (mm)",
             ylabel="X-integrated irradiance (dB to each zero order)",
             title="2 um exact horizontal-band object: real six-element POP sampling diagnosis")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = output / "bfp_refraction_resampling_on_off_diagnosis"
    fig.savefig(base.with_suffix(".png"), dpi=210, bbox_inches="tight")
    plt.close(fig)

    result = {
        "expected_order_centres_mm": expected.tolist(),
        "python_input_ratios_at_expected_centres": ratios_python,
        "resampling_on_ratios_at_expected_centres": ratios_on,
        "resampling_off_ratios_at_expected_centres": ratios_off,
        "largest_separated_peaks_with_resampling_off": [
            {"y_mm": float(y_off[index]), "relative_intensity": float(p_off[index])}
            for index in off_peaks
        ],
        "conclusion": (
            "ON erases the undersampled carrier; OFF retains aliased energy at incorrect "
            "BFP coordinates. Neither setting is a valid reconstruction of the 2 um orders."
        ),
    }
    base.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
