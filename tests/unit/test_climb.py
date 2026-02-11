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

    def test_climb_start_resets_to_true_low_point(self):
        """Climb should start from true low point, not a false start with brief uphill."""
        # False start: brief uphill, then descend more, then actual climb
        # This simulates a road that has a brief uphill grade then descends to valley
        elevations = (
            [150.0]                                  # Starting point
            + [150.0 - i * 2 for i in range(1, 6)]  # Descend to 140m
            + [140.0 + i * 3 for i in range(1, 4)]  # Brief rise (false start trigger)
            + [149.0 - i * 5 for i in range(1, 13)] # Descend to valley at 89m
            + [89.0 + i * 8 for i in range(1, 21)]  # Actual climb: +160m
        )
        points = make_track_points(elevations, spacing_m=100.0)
        result = detect_climbs(points, sensitivity_m=30.0, min_climb_gain=50.0, grade_threshold=2.0)

        assert len(result.climbs) == 1
        climb = result.climbs[0]
        # Climb should start near the true low point (~89m), not the false start (~140m)
        assert climb.start_elevation < 100.0, f"Start elevation {climb.start_elevation} should be near valley floor"
        # Average grade should reflect the steep actual climb, not be diluted by descent
        assert climb.avg_grade > 6.0, f"Avg grade {climb.avg_grade} should be high (steep climb)"


class TestClimbConsistencyWithProfile:
    """Tests for climb detection consistency with elevation profile data."""

    def test_max_grade_uses_rolling_grades_when_provided(self):
        """Max grade should use rolling_grades when provided, not point-to-point grades."""
        # Create a climb with spiky point-to-point grades but smooth rolling grades
        elevations = [100.0 + i * 5 for i in range(21)]  # 100m gain over 2km
        points = make_track_points(elevations, spacing_m=100.0)

        # Point-to-point grades would be ~5%, but we provide lower rolling grades
        rolling_grades = [3.0] * 20  # Smooth 3% everywhere

        result = detect_climbs(
            points,
            sensitivity_m=30.0,
            min_climb_gain=40.0,
            rolling_grades=rolling_grades,
        )
        assert len(result.climbs) == 1
        # Max grade should use rolling_grades (3%), not point-to-point (~5%)
        assert result.climbs[0].max_grade == pytest.approx(3.0, abs=0.1)

    def test_max_grade_falls_back_to_point_grades_without_rolling(self):
        """Without rolling_grades, max grade should use point-to-point grades."""
        elevations = [100.0 + i * 5 for i in range(21)]  # 100m gain over 2km
        points = make_track_points(elevations, spacing_m=100.0)

        result = detect_climbs(
            points,
            sensitivity_m=30.0,
            min_climb_gain=40.0,
            rolling_grades=None,
        )
        assert len(result.climbs) == 1
        # Should use point-to-point grade (~5%)
        assert result.climbs[0].max_grade >= 4.5

    def test_work_sums_segment_works_when_provided(self):
        """Work should be sum of segment_works when provided."""
        elevations = [100.0 + i * 5 for i in range(11)]  # 50m gain over 1km
        points = make_track_points(elevations, spacing_m=100.0)

        # Provide known segment_works (in joules)
        # 10 segments, each 1000J = 10000J = 10kJ total
        segment_works = [1000.0] * 10

        result = detect_climbs(
            points,
            sensitivity_m=30.0,
            min_climb_gain=40.0,
            segment_works=segment_works,
        )
        assert len(result.climbs) == 1
        # Work should be sum of segment_works converted to kJ
        assert result.climbs[0].work_kj == pytest.approx(10.0, abs=0.5)

    def test_avg_power_from_segment_powers(self):
        """Avg power should be calculated from segment_powers when provided."""
        elevations = [100.0 + i * 5 for i in range(11)]  # 50m gain over 1km
        points = make_track_points(elevations, spacing_m=100.0)
        times_hours = [i * 0.01 for i in range(11)]  # 0.1 hours total

        # Provide known segment_powers (in watts)
        segment_powers = [200.0] * 10  # 200W for each segment

        result = detect_climbs(
            points,
            times_hours=times_hours,
            sensitivity_m=30.0,
            min_climb_gain=40.0,
            segment_powers=segment_powers,
        )
        assert len(result.climbs) == 1
        # Avg power should be average of segment_powers
        assert result.climbs[0].avg_power == pytest.approx(200.0, abs=5.0)


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
