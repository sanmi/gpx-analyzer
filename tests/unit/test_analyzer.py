from datetime import datetime, timedelta, timezone

import pytest

from gpx_analyzer.analyzer import analyze, calculate_hilliness
from gpx_analyzer.models import RiderParams, TrackPoint


@pytest.fixture
def high_power_params():
    return RiderParams(climbing_power=300.0, flat_power=240.0)


class TestAnalyze:
    def test_fewer_than_two_points(self, rider_params):
        result = analyze([], rider_params)
        assert result.total_distance == 0.0
        assert result.elevation_gain == 0.0

        result = analyze(
            [TrackPoint(lat=0, lon=0, elevation=0, time=None)], rider_params
        )
        assert result.total_distance == 0.0

    def test_flat_track(self, simple_track_points, rider_params):
        result = analyze(simple_track_points, rider_params)
        assert result.total_distance > 0
        assert result.elevation_gain == 0.0
        assert result.elevation_loss == 0.0
        assert result.avg_speed > 0
        assert result.estimated_work > 0

    def test_uphill_track(self, uphill_track_points, rider_params):
        result = analyze(uphill_track_points, rider_params)
        assert result.elevation_gain == pytest.approx(25.0)
        assert result.elevation_loss == 0.0
        assert result.estimated_work > 0

    def test_downhill_track(self, downhill_track_points, rider_params):
        result = analyze(downhill_track_points, rider_params)
        assert result.elevation_gain == 0.0
        assert result.elevation_loss == pytest.approx(40.0)

    def test_uphill_more_work_than_flat(
        self, simple_track_points, uphill_track_points, rider_params
    ):
        flat_result = analyze(simple_track_points, rider_params)
        uphill_result = analyze(uphill_track_points, rider_params)
        assert uphill_result.estimated_work > flat_result.estimated_work

    def test_duration(self, simple_track_points, rider_params):
        result = analyze(simple_track_points, rider_params)
        assert result.duration == timedelta(seconds=40)

    def test_moving_time(self, simple_track_points, rider_params):
        result = analyze(simple_track_points, rider_params)
        assert result.moving_time.total_seconds() > 0
        assert result.moving_time <= result.duration

    def test_max_speed(self, simple_track_points, rider_params):
        result = analyze(simple_track_points, rider_params)
        assert result.max_speed > 0
        assert result.max_speed >= result.avg_speed

    def test_estimated_time_at_power(self, simple_track_points, rider_params):
        result = analyze(simple_track_points, rider_params)
        assert result.estimated_moving_time_at_power.total_seconds() > 0
        expected_seconds = result.estimated_work / rider_params.climbing_power
        assert result.estimated_moving_time_at_power.total_seconds() == pytest.approx(
            expected_seconds
        )

    def test_higher_power_shorter_time(self, simple_track_points, rider_params, high_power_params):
        result_low = analyze(simple_track_points, rider_params)
        result_high = analyze(simple_track_points, high_power_params)
        assert result_high.estimated_moving_time_at_power < result_low.estimated_moving_time_at_power

    def test_fewer_than_two_points_zero_time_at_power(self, rider_params):
        result = analyze([], rider_params)
        assert result.estimated_moving_time_at_power == timedelta()


class TestCalculateHilliness:
    """Tests for the calculate_hilliness function, especially steep_time and very_steep_time."""

    @pytest.fixture
    def flat_track_points(self):
        """Flat track - no steep segments. ~500m total."""
        base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        # Create a longer flat track to work with rolling grade window
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=10.0, time=base_time + timedelta(seconds=20)),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=10.0, time=base_time + timedelta(seconds=40)),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=10.0, time=base_time + timedelta(seconds=60)),
            TrackPoint(lat=37.7785, lon=-122.4150, elevation=10.0, time=base_time + timedelta(seconds=80)),
            TrackPoint(lat=37.7794, lon=-122.4139, elevation=10.0, time=base_time + timedelta(seconds=100)),
        ]

    @pytest.fixture
    def steep_track_points(self):
        """Track with ~12% grade - above 10% threshold but below 15%. ~700m total.

        Points are ~139m apart, so need 17m elevation gain per segment for ~12% grade.
        """
        base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=27.0, time=base_time + timedelta(seconds=30)),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=44.0, time=base_time + timedelta(seconds=60)),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=61.0, time=base_time + timedelta(seconds=90)),
            TrackPoint(lat=37.7785, lon=-122.4150, elevation=78.0, time=base_time + timedelta(seconds=120)),
            TrackPoint(lat=37.7794, lon=-122.4139, elevation=95.0, time=base_time + timedelta(seconds=150)),
        ]

    @pytest.fixture
    def very_steep_track_points(self):
        """Track with ~20% grade - well above 15% threshold. ~700m total.

        Points are ~139m apart, so need 28m elevation gain per segment for 20% grade.
        """
        base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=38.0, time=base_time + timedelta(seconds=30)),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=66.0, time=base_time + timedelta(seconds=60)),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=94.0, time=base_time + timedelta(seconds=90)),
            TrackPoint(lat=37.7785, lon=-122.4150, elevation=122.0, time=base_time + timedelta(seconds=120)),
            TrackPoint(lat=37.7794, lon=-122.4139, elevation=150.0, time=base_time + timedelta(seconds=150)),
        ]

    @pytest.fixture
    def mixed_grade_track_points(self):
        """Track with mixed grades: flat section then very steep section.

        Points are ~139m apart. Flat section first, then ~20% grade section.
        """
        base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        return [
            # Flat section (~420m, 3 segments)
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=10.0, time=base_time + timedelta(seconds=20)),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=10.0, time=base_time + timedelta(seconds=40)),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=10.0, time=base_time + timedelta(seconds=60)),
            # Very steep section ~20% grade (~420m, 3 segments, 28m gain each)
            TrackPoint(lat=37.7785, lon=-122.4150, elevation=38.0, time=base_time + timedelta(seconds=90)),
            TrackPoint(lat=37.7794, lon=-122.4139, elevation=66.0, time=base_time + timedelta(seconds=120)),
            TrackPoint(lat=37.7803, lon=-122.4128, elevation=94.0, time=base_time + timedelta(seconds=150)),
        ]

    def test_flat_track_no_steep_time(self, flat_track_points, rider_params):
        """Flat track should have zero steep time."""
        result = calculate_hilliness(flat_track_points, rider_params)
        assert result.steep_time == 0.0
        assert result.very_steep_time == 0.0
        assert result.steep_distance == 0.0
        assert result.very_steep_distance == 0.0

    def test_steep_track_has_steep_time(self, steep_track_points, rider_params):
        """Track at ~12% grade should accumulate steep_time but not very_steep_time."""
        result = calculate_hilliness(steep_track_points, rider_params)
        # Should have steep time (>= 10%)
        assert result.steep_time > 0
        # Should NOT have very steep time (grade is ~12%, below 15%)
        assert result.very_steep_time == 0.0
        # Steep distance should also be tracked
        assert result.steep_distance > 0
        assert result.very_steep_distance == 0.0
        # Max grade should be around 12%
        assert 10 < result.max_grade < 15

    def test_very_steep_track_has_both_times(self, very_steep_track_points, rider_params):
        """Track at ~20% grade should accumulate both steep_time and very_steep_time."""
        result = calculate_hilliness(very_steep_track_points, rider_params)
        # Should have steep time (>= 10%)
        assert result.steep_time > 0
        # Should have very steep time (>= 15%)
        assert result.very_steep_time > 0
        # Distances should be positive
        assert result.steep_distance > 0
        assert result.very_steep_distance > 0
        # Max grade should be around 20%
        assert result.max_grade > 15

    def test_mixed_grades_accumulate_correctly(self, mixed_grade_track_points, rider_params):
        """Mixed grade track should accumulate times correctly for each threshold."""
        result = calculate_hilliness(mixed_grade_track_points, rider_params)
        # steep_time should include steep segments
        assert result.steep_time > 0
        # very_steep_time should include very steep segments
        assert result.very_steep_time > 0
        # steep_time should be >= very_steep_time (steep includes all >= 10%)
        assert result.steep_time >= result.very_steep_time

    def test_fewer_than_two_points_returns_zero(self, rider_params):
        """Empty or single-point track should return zero times."""
        result = calculate_hilliness([], rider_params)
        assert result.steep_time == 0.0
        assert result.very_steep_time == 0.0

        single_point = [TrackPoint(lat=0, lon=0, elevation=0, time=None)]
        result = calculate_hilliness(single_point, rider_params)
        assert result.steep_time == 0.0
        assert result.very_steep_time == 0.0

    def test_steep_time_matches_histogram_sum(self, very_steep_track_points, rider_params):
        """steep_time should approximately match the sum of steep_time_histogram values."""
        result = calculate_hilliness(very_steep_track_points, rider_params)
        histogram_sum = sum(result.steep_time_histogram.values())
        # They should be approximately equal (histogram only includes >= 10%)
        assert result.steep_time == pytest.approx(histogram_sum, rel=0.01)
