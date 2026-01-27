import os
import tempfile

import pytest

from gpx_analyzer.parser import parse_gpx

SAMPLE_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "functional", "data", "sample_ride.gpx"
)


class TestParseGpx:
    def test_parse_sample_file(self):
        points = parse_gpx(SAMPLE_GPX_PATH)
        assert len(points) == 20
        assert points[0].lat == pytest.approx(37.7749)
        assert points[0].lon == pytest.approx(-122.4194)
        assert points[0].elevation == pytest.approx(10.0)
        assert points[0].time is not None

    def test_all_points_have_elevation(self):
        points = parse_gpx(SAMPLE_GPX_PATH)
        for pt in points:
            assert pt.elevation is not None

    def test_all_points_have_time(self):
        points = parse_gpx(SAMPLE_GPX_PATH)
        for pt in points:
            assert pt.time is not None

    def test_missing_elevation(self):
        gpx_content = """<?xml version="1.0"?>
        <gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
          <trk><trkseg>
            <trkpt lat="37.0" lon="-122.0">
              <time>2024-06-15T08:00:00Z</time>
            </trkpt>
          </trkseg></trk>
        </gpx>"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gpx", delete=False) as f:
            f.write(gpx_content)
            f.flush()
            points = parse_gpx(f.name)
        os.unlink(f.name)
        assert len(points) == 1
        assert points[0].elevation is None

    def test_missing_time(self):
        gpx_content = """<?xml version="1.0"?>
        <gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
          <trk><trkseg>
            <trkpt lat="37.0" lon="-122.0">
              <ele>100</ele>
            </trkpt>
          </trkseg></trk>
        </gpx>"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gpx", delete=False) as f:
            f.write(gpx_content)
            f.flush()
            points = parse_gpx(f.name)
        os.unlink(f.name)
        assert len(points) == 1
        assert points[0].time is None

    def test_empty_gpx(self):
        gpx_content = """<?xml version="1.0"?>
        <gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
          <trk><trkseg></trkseg></trk>
        </gpx>"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gpx", delete=False) as f:
            f.write(gpx_content)
            f.flush()
            points = parse_gpx(f.name)
        os.unlink(f.name)
        assert len(points) == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_gpx("/nonexistent/path/file.gpx")
