"""PyCharm and CLI entrypoint for common V1.3 training/exact evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.training import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=HEAD_TYPES)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate only; load this checkpoint.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()
    result = run_from_config(args.config.resolve(), args.output.resolve(), head_type=args.head,
                             checkpoint_path=args.checkpoint, evaluate_split=args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
