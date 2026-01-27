import os
import subprocess
import sys

import pytest

SAMPLE_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "data", "sample_ride.gpx"
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

    def test_run_with_custom_mass(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "--mass", "75", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "GPX Ride Analysis" in result.stdout

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
