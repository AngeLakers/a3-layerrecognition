from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
M2_ENTRYPOINTS = (
    "build_m2_01_real_rf_batch.py",
    "m2_real_rf_forward_smoke.py",
    "m2_attention_support_diagnostic.py",
    "m2_04_gpu_profile.py",
    "m2_05_safe_batch_search.py",
    "build_m2_06_runtime_acceptance.py",
    "m2_07_synthetic_tiny_overfit.py",
    "m2_label_free_real_rf_backward_diagnostic.py",
    "build_m2_label_free_acceptance.py",
)


@pytest.mark.parametrize("script_name", M2_ENTRYPOINTS)
def test_m2_cli_help_works_without_pythonpath_from_other_directory(
    script_name: str,
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / script_name),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr
