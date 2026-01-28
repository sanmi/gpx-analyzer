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
CROIX_DE_FER_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "data", "col_de_la_croix_de_fer.gpx"
)
RIDEWITHGPS_URL = "https://ridewithgps.com/routes/53835626?privacy_code=Z5O4f4AuMpY9ylt1Ht4o3XvAyd1zmzji"


class TestCli:
    def test_run_with_sample_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "--power", "150", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout
        assert "GPX Route Analysis" in output
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
        assert "GPX Route Analysis" in result.stdout

    def test_run_with_custom_power(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "--power", "200", SAMPLE_GPX_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Est. Time @200W:" in result.stdout

    def test_run_with_coasting_params(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "gpx_analyzer",
                "--coasting-grade", "-3",
                "--max-coast-speed", "40",
                SAMPLE_GPX_PATH,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "GPX Route Analysis" in result.stdout

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

    def test_col_de_la_croix_de_fer_ride(self):
        """Validate Col de la Croix de Fer against known actuals."""
        result = subprocess.run(
            [
                sys.executable, "-m", "gpx_analyzer",
                "--mass", "98",
                "--power", "120",
                "--cda", "0.35",
                "--crr", "0.005",
                CROIX_DE_FER_GPX_PATH,
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

        elevation_gain = parse_value("Elevation Gain:")
        work_kj = parse_value("Est. Work:")
        moving_hours = parse_duration_hours("Moving Time:")

        assert 1760 < elevation_gain < 1840, f"Expected elevation gain ~1800m, got {elevation_gain}"
        assert 1950 < work_kj < 2250, f"Expected work ~2100 kJ, got {work_kj}"
        assert 4.6 < moving_hours < 5.2, f"Expected moving time ~4:50h, got {moving_hours:.2f}h"

    def test_loma_prieta_elevation_gain_with_smoothing(self):
        """Default smoothing (50m radius) should yield ~1420m elevation gain."""
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
        match = re.search(r"Elevation Gain:\s+([\d.]+)", result.stdout)
        assert match, "Could not find 'Elevation Gain:' in output"
        gain = float(match.group(1))
        assert 1380 < gain < 1460, f"Expected elevation gain ~1420m, got {gain}"

    def test_no_smoothing_flag(self):
        """--no-smoothing should produce higher (raw) elevation gain."""
        result = subprocess.run(
            [
                sys.executable, "-m", "gpx_analyzer",
                "--no-smoothing",
                "--mass", "84",
                "--power", "120",
                LOMA_PRIETA_GPX_PATH,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        match = re.search(r"Elevation Gain:\s+([\d.]+)", result.stdout)
        assert match, "Could not find 'Elevation Gain:' in output"
        gain = float(match.group(1))
        assert 1490 < gain < 1570, f"Expected raw elevation gain ~1528m, got {gain}"

    def test_custom_smoothing_radius(self):
        """--smoothing with a custom radius should work without error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "gpx_analyzer",
                "--smoothing", "25",
                SAMPLE_GPX_PATH,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "GPX Route Analysis" in result.stdout

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

    @pytest.mark.network
    def test_ridewithgps_url(self):
        """Test fetching and analyzing a route from RideWithGPS URL."""
        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", RIDEWITHGPS_URL],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = result.stdout
        assert "GPX Route Analysis" in output
        assert "Distance:" in output
        assert "Elevation Gain:" in output

    @pytest.mark.network
    def test_ridewithgps_authenticated_route(self):
        """Test fetching a route that requires authentication.

        Skipped if RideWithGPS credentials are not configured.
        Configure via gpx-analyzer.json or environment variables.
        """
        from gpx_analyzer.ridewithgps import _get_auth_headers

        if not _get_auth_headers():
            pytest.skip("RideWithGPS credentials not configured")

        result = subprocess.run(
            [sys.executable, "-m", "gpx_analyzer", "https://ridewithgps.com/routes/53835558"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = result.stdout
        assert "GPX Route Analysis" in output
        assert "Distance:" in output
        assert "Elevation Gain:" in output
