"""Tests for climb detection."""

import pytest

from gpx_analyzer.models import TrackPoint
from gpx_analyzer.climb import ClimbInfo, detect_climbs, slider_to_sensitivity


def make_track_points(elevations: list[float], spacing_m: float = 100.0) -> list[TrackPoint]:
    """Create track points with given elevations along a straight line."""
    base_lat, base_lon = 45.0, 6.0
    lat_delta = spacing_m / 111000
    return [
        TrackPoint(lat=base_lat + i * lat_delta, lon=base_lon, elevation=elev, time=None)
        for i, elev in enumerate(elevations)
    ]


class TestDetectClimbs:
    """Tests for climb detection algorithm."""

    def test_no_climb_flat_route(self):
        """Flat route should have no climbs."""
        elevations = [100.0] * 100
        points = make_track_points(elevations)
        result = detect_climbs(points, sensitivity_m=30.0)
        assert result.climbs == []

    def test_single_climb(self):
        """Should detect a single continuous climb."""
        # 500m at 10% grade = 50m gain per 500m
        # Create 51 points for 5km climb gaining 500m total
        elevations = [100.0 + i * 10 for i in range(51)]  # 0 to 500m, gaining 500m
        points = make_track_points(elevations, spacing_m=100.0)  # 5km climb
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=40.0)
        assert len(result.climbs) == 1
        climb = result.climbs[0]
        assert climb.elevation_gain >= 450  # Allow some tolerance
        assert climb.avg_grade > 5.0

    def test_sensitivity_splits_climbs(self):
        """High sensitivity should split climbs at dips; low should merge."""
        # Climb 1: +100m, dip: -20m, Climb 2: +100m
        elevations = (
            [100.0 + i * 5 for i in range(21)]      # +100m over 2km
            + [200.0 - i * 2 for i in range(11)]    # -20m dip over 1km
            + [180.0 + i * 5 for i in range(21)]    # +100m over 2km
        )
        points = make_track_points(elevations, spacing_m=100.0)

        # High sensitivity (10m) - should see 2 climbs
        result_high = detect_climbs(points, sensitivity_m=10.0, min_climb_gain=40.0)
        assert len(result_high.climbs) == 2

        # Low sensitivity (50m) - should see 1 climb (20m dip tolerated)
        result_low = detect_climbs(points, sensitivity_m=50.0, min_climb_gain=40.0)
        assert len(result_low.climbs) == 1

    def test_ignores_small_climbs(self):
        """Should ignore climbs below minimum thresholds."""
        # Small 30m climb
        elevations = [100.0 + i * 3 for i in range(11)]  # 30m gain over 1km
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=50.0)
        assert result.climbs == []

    def test_climb_ends_at_peak(self):
        """Climb should end at peak, not where descent is detected."""
        # Climb to 200m, descend to 150m
        elevations = (
            [100.0 + i * 5 for i in range(21)]  # up to 200m over 2km
            + [200.0 - i * 2.5 for i in range(21)]  # down to 150m over 2km
        )
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=40.0)
        assert len(result.climbs) == 1
        climb = result.climbs[0]
        assert abs(climb.peak_elevation - 200.0) < 5  # Peak should be ~200m

    def test_climb_metrics_calculated(self):
        """Should calculate all metrics for detected climb."""
        elevations = [100.0 + i * 5 for i in range(21)]  # 100m gain over 2km
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=40.0)
        assert len(result.climbs) == 1
        climb = result.climbs[0]

        assert climb.climb_id == 1
        assert climb.distance_m > 1500  # ~2km
        assert climb.elevation_gain >= 95  # ~100m
        assert climb.avg_grade > 0
        assert climb.max_grade > 0
        assert climb.label is not None

    def test_multiple_climbs_numbered_sequentially(self):
        """Multiple climbs should be numbered 1, 2, 3..."""
        # Three separate climbs with large descents between
        elevations = (
            [100.0 + i * 10 for i in range(11)]   # Climb 1: +100m
            + [200.0 - i * 10 for i in range(11)]  # Descent to 100m
            + [100.0 + i * 10 for i in range(11)]   # Climb 2: +100m
            + [200.0 - i * 10 for i in range(11)]  # Descent to 100m
            + [100.0 + i * 10 for i in range(11)]   # Climb 3: +100m
        )
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=40.0)

        assert len(result.climbs) >= 2  # At least 2 climbs
        for i, climb in enumerate(result.climbs):
            assert climb.climb_id == i + 1  # 1-indexed

    def test_climb_with_minor_dip(self):
        """Climb with minor dip should be treated as single climb with sufficient sensitivity."""
        # Climb with 5m dip in the middle
        elevations = (
            [100.0 + i * 5 for i in range(11)]      # +50m
            + [150.0 - i * 0.5 for i in range(11)]  # -5m dip
            + [145.0 + i * 5 for i in range(11)]    # +55m more
        )
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=40.0)
        # With 30m sensitivity, 5m dip should be tolerated
        assert len(result.climbs) == 1


class TestSliderToSensitivity:
    """Tests for slider value to sensitivity conversion."""

    def test_slider_0_high_sensitivity(self):
        """Slider 0 should give high sensitivity (low tolerance)."""
        assert slider_to_sensitivity(0) == 10.0

    def test_slider_100_low_sensitivity(self):
        """Slider 100 should give low sensitivity (high tolerance)."""
        assert slider_to_sensitivity(100) == 100.0

    def test_slider_50_middle(self):
        """Slider 50 should give middle value."""
        result = slider_to_sensitivity(50)
        assert 50.0 <= result <= 60.0  # Approximately middle

    def test_monotonic_increase(self):
        """Higher slider values should give higher tolerance."""
        prev = 0
        for v in range(0, 101, 10):
            curr = slider_to_sensitivity(v)
            assert curr >= prev
            prev = curr
