from datetime import timedelta

import pytest

from gpx_analyzer.analyzer import analyze
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
