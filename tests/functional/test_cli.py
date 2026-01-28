import os
import re
import subprocess
import sys

import pytest

SAMPLE_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "data", "sample_ride.gpx"
)
LOMA_PRIETA_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "data", "Lexington_OSC_Loma_Prieta.gpx"
)


class TestCli:
    def test_run_with_sample_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout
        assert "GPX Ride Analysis" in output
        assert "Distance:" in output
        assert "Elevation Gain:" in output
        assert "Elevation Loss:" in output
        assert "Duration:" in output
        assert "Moving Time:" in output
        assert "Avg Speed:" in output
        assert "Max Speed:" in output
        assert "Est. Work:" in output
        assert "Est. Avg Power:" in output
        assert "Est. Time @150W:" in output

    def test_run_with_custom_mass(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "--mass", "75", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "GPX Ride Analysis" in result.stdout

    def test_run_with_custom_power(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "--power", "200", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Est. Time @200W:" in result.stdout

    def test_nonexistent_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "/nonexistent/file.gpx"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Error" in result.stderr

    def test_no_arguments(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_loma_prieta_ride(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "gpx_analyzer",
                "--mass", "84",
                "--power", "120",
                LOMA_PRIETA_GPX_PATH,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout

        def parse_value(label):
            match = re.search(rf"{re.escape(label)}\s+([\d.]+)", output)
            assert match, f"Could not find '{label}' in output"
            return float(match.group(1))

        def parse_duration_hours(label):
            match = re.search(rf"{re.escape(label)}\s+(\d+)h\s+(\d+)m\s+(\d+)s", output)
            assert match, f"Could not find '{label}' in output"
            return int(match.group(1)) + int(match.group(2)) / 60 + int(match.group(3)) / 3600

        work_kj = parse_value("Est. Work:")
        moving_hours = parse_duration_hours("Moving Time:")
        est_time_hours = parse_duration_hours("Est. Time @120W:")

        assert 1900 < work_kj < 2500, f"Expected work ~2100 kJ, got {work_kj}"
        assert 4.5 < moving_hours < 6.0, f"Expected moving time ~5.5h, got {moving_hours:.2f}h"
        assert 4.5 < est_time_hours < 6.0, f"Expected est. time ~5.5h at 120W, got {est_time_hours:.2f}h"

    def test_output_has_units(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        output = result.stdout
        assert "km" in output
        assert "km/h" in output
        assert "kJ" in output
        assert "W" in output
