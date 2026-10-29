"""
Unit tests for data_logger/logger.py

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_logger"))

import pytest
from logger import parse_line, validate_ranges, Reading, ValidationError, DataLogger


GOOD_LINE = "123456,13.2,2.1,27.72,28.5,45.0,80.0,OK"


class TestParseLine:
    def test_parses_valid_line(self):
        reading = parse_line(GOOD_LINE)
        assert reading.voltage_v == 13.2
        assert reading.current_a == 2.1
        assert reading.status == "OK"

    def test_rejects_wrong_field_count(self):
        with pytest.raises(ValidationError, match="expected 8 fields"):
            parse_line("123456,13.2,2.1,27.72,28.5,45.0,OK")

    def test_rejects_non_numeric_field(self):
        with pytest.raises(ValidationError, match="non-numeric field"):
            parse_line("123456,not_a_number,2.1,27.72,28.5,45.0,80.0,OK")

    def test_passes_through_firmware_fault_status(self):
        # SENSOR_FAULT rows use -999 sentinel values and should parse without
        # raising, so the caller can route them to the rejects log instead.
        reading = parse_line("123456,13.2,2.1,27.72,-999,-999,80.0,SENSOR_FAULT")
        assert reading.status == "SENSOR_FAULT"

    def test_rejects_out_of_range_voltage_when_status_ok(self):
        with pytest.raises(ValidationError, match="voltage_v"):
            parse_line("123456,99.9,2.1,27.72,28.5,45.0,80.0,OK")


class TestValidateRanges:
    @pytest.mark.parametrize("field,value", [
        ("voltage_v", -5.0),
        ("voltage_v", 30.0),
        ("current_a", 15.0),
        ("temp_c", 100.0),
        ("humidity_pct", 150.0),
        ("light_pct", -10.0),
    ])
    def test_out_of_range_raises(self, field, value):
        kwargs = dict(
            timestamp_utc="2026-01-01T00:00:00Z", timestamp_ms=1,
            voltage_v=12.0, current_a=1.0, power_w=12.0,
            temp_c=25.0, humidity_pct=50.0, light_pct=50.0, status="OK",
        )
        kwargs[field] = value
        with pytest.raises(ValidationError):
            validate_ranges(Reading(**kwargs))

    def test_in_range_passes(self):
        reading = Reading(
            timestamp_utc="2026-01-01T00:00:00Z", timestamp_ms=1,
            voltage_v=12.0, current_a=1.0, power_w=12.0,
            temp_c=25.0, humidity_pct=50.0, light_pct=50.0, status="OK",
        )
        validate_ranges(reading)  # should not raise


class TestDataLogger:
    def test_creates_output_files(self, tmp_path):
        dl = DataLogger(tmp_path)
        assert dl.good_path.exists()
        assert dl.reject_path.exists()

    def test_logs_good_and_reject_rows(self, tmp_path):
        dl = DataLogger(tmp_path)
        reading = parse_line(GOOD_LINE)
        dl.log_good(reading)
        dl.log_reject("garbage,line", "bad schema")

        good_lines = dl.good_path.read_text().strip().splitlines()
        reject_lines = dl.reject_path.read_text().strip().splitlines()

        assert len(good_lines) == 2  # header + 1 row
        assert len(reject_lines) == 2  # header + 1 row

    def test_summary_reports_accurate_counts(self, tmp_path):
        dl = DataLogger(tmp_path)
        for _ in range(3):
            dl.log_good(parse_line(GOOD_LINE))
        dl.log_reject("bad", "reason")

        summary = dl.summary()
        assert "3 accepted" in summary
        assert "1 rejected" in summary
        assert "4 rows" in summary
