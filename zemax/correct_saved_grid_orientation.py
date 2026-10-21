"""Convert the first completed POP archive from X-first to image[Y, X] arrays."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with np.load(args.source) as archive:
        corrected = {
            key: (archive[key].T if key.endswith("_irradiance") else archive[key])
            for key in archive.files
        }
    np.savez_compressed(args.destination, **corrected)
    print(f"corrected={args.destination};arrays={len(corrected)}")


if __name__ == "__main__":
    main()
