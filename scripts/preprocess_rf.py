"""Prepare one same-acquisition raw NPZ for the existing A3 tensor-manifest runner."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a3_ultrasound.data.preprocessing import preprocess_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--signal-key", default="x_raw")
    args = parser.parse_args()
    print(json.dumps(preprocess_npz(args.input, args.output, args.config,
                                   signal_key=args.signal_key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
