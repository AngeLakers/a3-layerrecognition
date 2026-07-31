"""Fail-closed local preflight for A3 external-review text packets.

The scanner never sends data. It inventories a packet directory, accepts only
small UTF-8 text artifacts, and reports rule identifiers without copying the
matched sensitive text into its JSON output. A human still has to complete the
project checklist because proprietary identity and geometry cannot be fully
recognized by regular expressions.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PREFLIGHT_SCHEMA_VERSION = "a3-external-review-preflight-v1"

CHECK_IDS = (
    "raw_rf_or_raw_files",
    "local_absolute_paths",
    "real_identity_or_reversible_mapping",
    "scan_coordinates",
    "exact_proprietary_geometry",
    "secrets_or_private_endpoints",
)

_ALLOWED_TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
_MAX_TEXT_BYTES = 2 * 1024 * 1024
_SAFE_PLACEHOLDERS = {
    "",
    "false",
    "forbidden",
    "none",
    "null",
    "redacted",
    "removed",
    "unknown",
}

_RAW_FILE_REFERENCE = re.compile(
    r"(?i)(?:^|[\\/])[^\\/\s]+\.(?:bin|csv|dat|h5|hdf5|mat|npy|npz|tdms|wav|zarr)\b"
)
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)

_WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])")
_UNC_PATH = re.compile(r"(?<!\\)\\\\[A-Za-z0-9._$-]+\\")
_POSIX_PRIVATE_PATH = re.compile(r"(?i)(?<!\w)/(?:home|users|private|mnt/[a-z])/")
_FILE_URI = re.compile(r"(?i)\bfile://")

_IDENTITY_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(
        supplier(?:_or_owner)?|vendor|owner|manufacturing_partner|
        original_(?:specimen|dataset|batch)_?(?:identity|name)?|
        specimen_(?:id|name|serial)|dataset_id|scan_id|
        acquisition_batch_id|batch_id|source_root|local_source_subpath|
        defect_(?:design_identity|region_mapping_ref)|
        reversible_(?:id|mapping)
    )\b
    \s*[:=]\s*
    ["']?([^,\s}\]"']+)
    """
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:1[3-9]\d{9}|"
    r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)|\d{3,4})"
    r"[\s-]\d{3,4}[\s-]\d{3,4})(?!\d)"
)

_COORDINATE_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(
        scan_coordinates?|grid_coordinates?|pixel_coordinates?|
        coordinate_mapping_ref|scan_coordinate_mapping_ref|
        grid_x|grid_y|index_x|index_y
    )\b
    \s*[:=]\s*
    ["']?([^,\s}\]"']+)
    """
)
_COORDINATE_FILENAME = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])x\d{1,8}y\d{1,8}(?:\.[A-Za-z0-9]+)?(?:$|[^A-Za-z0-9])"
)

_EXACT_LENGTH_UNIT = re.compile(
    r"(?i)(?<![A-Za-z0-9_.])\d+(?:\.\d+)?\s*"
    r"(?:mm|cm|micrometers?|microns?|um|µm|μm|inches?|degrees?|deg|°)\b"
)
_EXACT_GRID_GEOMETRY = re.compile(
    r"(?i)\b(?:grid|scan|specimen|coupon|layer|geometry)\b.*"
    r"\b\d{2,}\s*[x×]\s*\d{2,}\b"
)

_SECRET_LITERAL = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|"
    r"AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~+/-]{16,})\b"
)
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(
        api_?key|access_?token|refresh_?token|token|secret|password|
        authorization|base_?url|api_?url|private_?endpoint|endpoint
    )\b
    \s*[:=]\s*
    ["']?([^,\s}\]"']+)
    """
)
_PRIVATE_ENDPOINT = re.compile(
    r"(?i)\bhttps?://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)


def _normalized_value(value: str) -> str:
    return value.strip().strip("\"'<>").lower()


def _has_nonplaceholder_assignment(line: str, pattern: re.Pattern[str]) -> bool:
    return any(
        _normalized_value(match.group(2)) not in _SAFE_PLACEHOLDERS
        for match in pattern.finditer(line)
    )


def _finding(
    check_id: str,
    *,
    relative_file: str,
    line: int | None,
    rule_id: str,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "check_id": check_id,
        "relative_file": relative_file,
        "rule_id": rule_id,
    }
    if line is not None:
        finding["line"] = line
    return finding


def _scan_line(
    line: str,
    *,
    relative_file: str,
    line_number: int,
) -> Iterable[dict[str, Any]]:
    if _RAW_FILE_REFERENCE.search(line):
        yield _finding(
            "raw_rf_or_raw_files",
            relative_file=relative_file,
            line=line_number,
            rule_id="raw_file_reference",
        )
    if len(_NUMERIC_TOKEN.findall(line)) >= 64:
        yield _finding(
            "raw_rf_or_raw_files",
            relative_file=relative_file,
            line=line_number,
            rule_id="long_numeric_sequence",
        )

    if (
        _WINDOWS_ABSOLUTE.search(line)
        or _UNC_PATH.search(line)
        or _POSIX_PRIVATE_PATH.search(line)
        or _FILE_URI.search(line)
    ):
        yield _finding(
            "local_absolute_paths",
            relative_file=relative_file,
            line=line_number,
            rule_id="absolute_or_private_path",
        )

    if (
        _has_nonplaceholder_assignment(line, _IDENTITY_ASSIGNMENT)
        or _EMAIL.search(line)
        or _PHONE.search(line)
    ):
        yield _finding(
            "real_identity_or_reversible_mapping",
            relative_file=relative_file,
            line=line_number,
            rule_id="identity_or_mapping_value",
        )

    if (
        _has_nonplaceholder_assignment(line, _COORDINATE_ASSIGNMENT)
        or _COORDINATE_FILENAME.search(line)
    ):
        yield _finding(
            "scan_coordinates",
            relative_file=relative_file,
            line=line_number,
            rule_id="coordinate_value_or_filename",
        )

    if _EXACT_LENGTH_UNIT.search(line) or _EXACT_GRID_GEOMETRY.search(line):
        yield _finding(
            "exact_proprietary_geometry",
            relative_file=relative_file,
            line=line_number,
            rule_id="dimensional_geometry_value",
        )

    if (
        _SECRET_LITERAL.search(line)
        or _has_nonplaceholder_assignment(line, _SECRET_ASSIGNMENT)
        or _PRIVATE_ENDPOINT.search(line)
    ):
        yield _finding(
            "secrets_or_private_endpoints",
            relative_file=relative_file,
            line=line_number,
            rule_id="credential_or_private_endpoint",
        )


def scan_external_review_packet(
    packet_root: str | Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a privacy preflight report without echoing matched text."""

    root = Path(packet_root)
    if not root.is_dir():
        raise ValueError("packet_root must be an existing directory.")
    root = root.resolve()

    findings: dict[str, list[dict[str, Any]]] = {
        check_id: [] for check_id in CHECK_IDS
    }
    scanned_files: list[str] = []

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_dir():
            continue
        relative_file = path.relative_to(root).as_posix()
        resolved_path = path.resolve()
        if path.is_symlink() or not resolved_path.is_relative_to(root):
            findings["local_absolute_paths"].append(
                _finding(
                    "local_absolute_paths",
                    relative_file=relative_file,
                    line=None,
                    rule_id="symlink_or_path_escape",
                )
            )
            continue

        scanned_files.append(relative_file)
        if _COORDINATE_FILENAME.search(relative_file):
            findings["scan_coordinates"].append(
                _finding(
                    "scan_coordinates",
                    relative_file=relative_file,
                    line=None,
                    rule_id="coordinate_value_or_filename",
                )
            )
        if path.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            findings["raw_rf_or_raw_files"].append(
                _finding(
                    "raw_rf_or_raw_files",
                    relative_file=relative_file,
                    line=None,
                    rule_id="unsupported_attachment_type",
                )
            )
            continue
        if path.stat().st_size > _MAX_TEXT_BYTES:
            findings["raw_rf_or_raw_files"].append(
                _finding(
                    "raw_rf_or_raw_files",
                    relative_file=relative_file,
                    line=None,
                    rule_id="oversized_text_attachment",
                )
            )
            continue

        payload = path.read_bytes()
        if b"\x00" in payload:
            findings["raw_rf_or_raw_files"].append(
                _finding(
                    "raw_rf_or_raw_files",
                    relative_file=relative_file,
                    line=None,
                    rule_id="binary_payload",
                )
            )
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            findings["raw_rf_or_raw_files"].append(
                _finding(
                    "raw_rf_or_raw_files",
                    relative_file=relative_file,
                    line=None,
                    rule_id="non_utf8_payload",
                )
            )
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for finding in _scan_line(
                line,
                relative_file=relative_file,
                line_number=line_number,
            ):
                findings[finding["check_id"]].append(finding)

    checks = []
    all_findings = []
    for check_id in CHECK_IDS:
        check_findings = findings[check_id]
        all_findings.extend(check_findings)
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if not check_findings else "FAIL",
                "finding_count": len(check_findings),
            }
        )

    status = "PASS" if not all_findings and scanned_files else "FAIL"
    if not scanned_files:
        all_findings.append(
            _finding(
                "raw_rf_or_raw_files",
                relative_file=".",
                line=None,
                rule_id="empty_packet",
            )
        )
        checks[0] = {
            "check_id": "raw_rf_or_raw_files",
            "status": "FAIL",
            "finding_count": 1,
        }

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "release_allowed_by_preflight": status == "PASS",
        "human_checklist_required": True,
        "scanned_file_count": len(scanned_files),
        "scanned_files": scanned_files,
        "checks": checks,
        "findings": all_findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locally scan an A3 external-review text packet.",
    )
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = scan_external_review_packet(args.packet_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "finding_count": len(report["findings"]),
                "human_checklist_required": True,
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
