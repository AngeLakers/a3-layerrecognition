"""Validate M2-06 runtime evidence and write its deterministic acceptance JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from a3_ultrasound.runtime_contract import build_runtime_acceptance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_runtime_acceptance(args.config)
    result["source_sha256"] = {
        "runtime_contract.py": _sha256(
            SOURCE_ROOT / "a3_ultrasound" / "runtime_contract.py"
        ),
        "build_m2_06_runtime_acceptance.py": _sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "frozen_batch": result["resolved_runtime"]["micro_batch_size"],
                "minimum_free_fraction": result["evidence_summary"][
                    "minimum_required_free_fraction_at_peak"
                ],
                "output": args.output.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
