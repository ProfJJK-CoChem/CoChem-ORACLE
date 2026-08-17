import pytest
from cochem_log_scrubber import TelemetrySanitizer, export_telemetry

def test_telemetry_sanitizer_xyz() -> None:
    sanitizer = TelemetrySanitizer()
    text = "C  -1.23456  0.12345  1.11111"
    cleaned = sanitizer.sanitize_text(text)
    assert "[REDACTED_COORD]" in cleaned

def test_telemetry_sanitizer_orca() -> None:
    sanitizer = TelemetrySanitizer()
    text = "* xyz 0 1\nC 0 0 0\n*"
    cleaned = sanitizer.sanitize_text(text)
    assert "[REDACTED_GEOMETRY_BLOCK]" in cleaned

def test_export_telemetry(tmp_path) -> None:
    input_chat = [{"role": "user", "content": "Hello world"}]
    archive = export_telemetry(input_chat, export_dir=str(tmp_path))
    assert archive != ""
