from __future__ import annotations

import json
from pathlib import Path

import pytest

from a3_ultrasound.review.sanitizer import (
    CHECK_IDS,
    PREFLIGHT_SCHEMA_VERSION,
    main,
    scan_external_review_packet,
)


FIXED_TIME = "2026-07-30T00:00:00+00:00"


def _write_packet(root: Path, text: str, *, name: str = "packet.md") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_deidentified_abstract_packet_passes_all_six_checks(tmp_path: Path) -> None:
    _write_packet(
        tmp_path,
        """# Frozen review packet

Timestamp: 2026-07-23T15:55:41+08:00

This packet excludes raw RF values, source paths, identities, reversible
mappings, scan coordinates, proprietary geometry, and credentials.

The model receives normalized interface fields:
`[k/K_i, tau/T, z/zmax, Delta-/T, Delta+/T]`.
The evidence is structural/API-only and does not establish performance.
""",
    )

    report = scan_external_review_packet(tmp_path, generated_at=FIXED_TIME)

    assert report["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["release_allowed_by_preflight"] is True
    assert report["human_checklist_required"] is True
    assert [check["check_id"] for check in report["checks"]] == list(CHECK_IDS)
    assert all(check["status"] == "PASS" for check in report["checks"])
    assert report["findings"] == []


@pytest.mark.parametrize(
    ("content", "filename", "expected_check"),
    [
        ("values: " + " ".join(str(i) for i in range(80)), "packet.md", "raw_rf_or_raw_files"),
        ("source: C:\\private\\specimen\\trace.txt", "packet.md", "local_absolute_paths"),
        ("supplier: ExampleManufacturingPartner", "packet.md", "real_identity_or_reversible_mapping"),
        ("contact: +86 138-1234-5678", "packet.md", "real_identity_or_reversible_mapping"),
        ("scan_coordinate: [12,34]", "packet.md", "scan_coordinates"),
        ("layer thickness is 3.0 mm", "packet.md", "exact_proprietary_geometry"),
        ("api_key: sk-examplecredential123456789", "packet.md", "secrets_or_private_endpoints"),
        ("abstract", "x120y004.txt", "scan_coordinates"),
    ],
    ids=[
        "raw-rf-values",
        "local-absolute-path",
        "identity-mapping",
        "contact-identifier",
        "scan-coordinate-content",
        "proprietary-geometry",
        "credential-pattern",
        "coordinate-filename",
    ],
)
def test_each_sensitive_category_fails_closed(
    tmp_path: Path,
    content: str,
    filename: str,
    expected_check: str,
) -> None:
    _write_packet(tmp_path, content, name=filename)

    report = scan_external_review_packet(tmp_path, generated_at=FIXED_TIME)

    assert report["status"] == "FAIL"
    assert report["release_allowed_by_preflight"] is False
    failed = {
        check["check_id"]
        for check in report["checks"]
        if check["status"] == "FAIL"
    }
    assert expected_check in failed


def test_binary_or_unsupported_attachment_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "waveform.npy").write_bytes(b"\x00\x01\x02")

    report = scan_external_review_packet(tmp_path, generated_at=FIXED_TIME)

    assert report["status"] == "FAIL"
    assert report["findings"] == [
        {
            "check_id": "raw_rf_or_raw_files",
            "relative_file": "waveform.npy",
            "rule_id": "unsupported_attachment_type",
        }
    ]


def test_report_never_echoes_matched_secret_or_absolute_root(tmp_path: Path) -> None:
    fake_secret = "sk-examplecredential123456789"
    _write_packet(tmp_path, f"api_key: {fake_secret}")

    report = scan_external_review_packet(tmp_path, generated_at=FIXED_TIME)
    serialized = json.dumps(report, ensure_ascii=False)

    assert fake_secret not in serialized
    assert str(tmp_path) not in serialized
    assert report["findings"][0]["relative_file"] == "packet.md"


def test_empty_packet_fails_closed(tmp_path: Path) -> None:
    report = scan_external_review_packet(tmp_path, generated_at=FIXED_TIME)

    assert report["status"] == "FAIL"
    assert report["scanned_file_count"] == 0
    assert report["findings"][0]["rule_id"] == "empty_packet"


def test_cli_writes_json_but_does_not_release_packet(tmp_path: Path) -> None:
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    _write_packet(packet_root, "# abstract packet")
    output = tmp_path / "preflight.json"

    exit_code = main(
        [
            "--packet-root",
            str(packet_root),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["human_checklist_required"] is True
