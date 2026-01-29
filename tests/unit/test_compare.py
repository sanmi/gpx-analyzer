"""Tests for route vs trip comparison."""

import pytest

from gpx_analyzer.compare import (
    ComparisonResult,
    GradeBucket,
    compare_route_with_trip,
    format_comparison_report,
)
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.ridewithgps import TripPoint


class TestGradeBucket:
    def test_avg_actual_speed(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[5.0, 6.0, 7.0],
            predicted_speeds=[6.0, 6.0, 6.0],
            actual_powers=[100.0, 110.0, 120.0],
            point_count=3,
        )
        assert bucket.avg_actual_speed == 6.0

    def test_avg_predicted_speed(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[5.0, 6.0, 7.0],
            predicted_speeds=[6.0, 7.0, 8.0],
            actual_powers=[],
            point_count=3,
        )
        assert bucket.avg_predicted_speed == 7.0

    def test_avg_actual_power(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[5.0],
            predicted_speeds=[6.0],
            actual_powers=[100.0, 150.0, 200.0],
            point_count=1,
        )
        assert bucket.avg_actual_power == 150.0

    def test_speed_error_pct_positive_when_predicted_faster(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[10.0],
            predicted_speeds=[12.0],
            actual_powers=[],
            point_count=1,
        )
        assert bucket.speed_error_pct == pytest.approx(20.0)

    def test_speed_error_pct_negative_when_predicted_slower(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[10.0],
            predicted_speeds=[8.0],
            actual_powers=[],
            point_count=1,
        )
        assert bucket.speed_error_pct == pytest.approx(-20.0)

    def test_empty_bucket_returns_zeros(self):
        bucket = GradeBucket(
            grade_pct=0,
            actual_speeds=[],
            predicted_speeds=[],
            actual_powers=[],
            point_count=0,
        )
        assert bucket.avg_actual_speed == 0.0
        assert bucket.avg_predicted_speed == 0.0
        assert bucket.avg_actual_power == 0.0


class TestCompareRouteWithTrip:
    @pytest.fixture
    def params(self):
        return RiderParams(
            assumed_avg_power=100.0,
            max_coasting_speed=15.0,
        )

    @pytest.fixture
    def route_points(self):
        return [
            TrackPoint(lat=37.0, lon=-122.0, elevation=100.0, time=None),
            TrackPoint(lat=37.001, lon=-122.0, elevation=100.0, time=None),
            TrackPoint(lat=37.002, lon=-122.0, elevation=100.0, time=None),
        ]

    @pytest.fixture
    def trip_points(self):
        # Create trip with flat terrain and consistent speed
        points = []
        for i in range(100):
            points.append(
                TripPoint(
                    lat=37.0 + i * 0.0001,
                    lon=-122.0,
                    elevation=100.0,
                    distance=i * 10.0,
                    speed=5.0 if i > 0 else 0.0,  # 5 m/s = 18 km/h
                    timestamp=1000000 + i,
                    power=100.0,
                    heart_rate=120,
                    cadence=80,
                )
            )
        return points

    def test_returns_comparison_result(self, route_points, trip_points, params):
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        assert isinstance(result, ComparisonResult)

    def test_calculates_trip_distance(self, route_points, trip_points, params):
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        # 99 segments * ~11.1m (0.0001° latitude) ≈ 1099m, calculated from geodesic
        assert result.trip_distance == pytest.approx(1099, rel=0.01)

    def test_calculates_moving_time(self, route_points, trip_points, params):
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        # Points with speed > 0.5 m/s (all except first)
        assert result.actual_moving_time == 99

    def test_detects_power_data(self, route_points, trip_points, params):
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        assert result.has_power_data is True
        assert result.actual_avg_power == pytest.approx(100.0)

    def test_calculates_actual_work(self, route_points, trip_points, params):
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        # 99 segments * 1 second * 100W = 9900 joules
        assert result.actual_work == pytest.approx(9900.0)
        assert result.predicted_work == 50000.0

    def test_no_power_data(self, route_points, params):
        trip_points = [
            TripPoint(
                lat=37.0, lon=-122.0, elevation=100.0,
                distance=i * 10.0, speed=5.0, timestamp=1000000 + i,
                power=None, heart_rate=None, cadence=None,
            )
            for i in range(100)
        ]
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        assert result.has_power_data is False
        assert result.actual_avg_power is None
        assert result.actual_work is None

    def test_time_error_positive_when_predicted_slower(self, route_points, trip_points, params):
        # Predicted 200s, actual moving time is 99s
        result = compare_route_with_trip(route_points, trip_points, params, 200.0, 50000.0)
        assert result.time_error_pct > 0

    def test_builds_grade_buckets(self, route_points, params):
        # Create trip with varying elevation (climbing)
        trip_points = []
        for i in range(100):
            trip_points.append(
                TripPoint(
                    lat=37.0 + i * 0.0001,
                    lon=-122.0,
                    elevation=100.0 + i * 0.5,  # 5% grade
                    distance=i * 10.0,
                    speed=3.0 if i > 0 else 0.0,
                    timestamp=1000000 + i,
                    power=150.0,
                    heart_rate=140,
                    cadence=70,
                )
            )
        result = compare_route_with_trip(route_points, trip_points, params, 120.0, 50000.0)
        assert len(result.grade_buckets) > 0
        # Should have a positive grade bucket
        grades = [b.grade_pct for b in result.grade_buckets]
        assert any(g > 0 for g in grades)


class TestFormatComparisonReport:
    def test_includes_distance(self):
        result = ComparisonResult(
            route_distance=10000.0,
            trip_distance=10500.0,
            predicted_time=3600.0,
            actual_moving_time=3500.0,
            time_error_pct=2.9,
            predicted_work=500000.0,
            actual_work=480000.0,
            grade_buckets=[],
            has_power_data=False,
            actual_avg_power=None,
        )
        params = RiderParams(assumed_avg_power=100.0)
        report = format_comparison_report(result, params)
        assert "10.0 km" in report
        assert "10.5 km" in report

    def test_includes_time_comparison(self):
        result = ComparisonResult(
            route_distance=10000.0,
            trip_distance=10000.0,
            predicted_time=7200.0,
            actual_moving_time=6000.0,
            time_error_pct=20.0,
            predicted_work=500000.0,
            actual_work=400000.0,
            grade_buckets=[],
            has_power_data=False,
            actual_avg_power=None,
        )
        params = RiderParams(assumed_avg_power=150.0)
        report = format_comparison_report(result, params)
        assert "2.00 hours" in report
        assert "+20 minutes" in report or "+20.0%" in report

    def test_includes_power_when_available(self):
        result = ComparisonResult(
            route_distance=10000.0,
            trip_distance=10000.0,
            predicted_time=3600.0,
            actual_moving_time=3600.0,
            time_error_pct=0.0,
            predicted_work=400000.0,
            actual_work=432000.0,
            grade_buckets=[],
            has_power_data=True,
            actual_avg_power=120.0,
        )
        params = RiderParams(assumed_avg_power=100.0)
        report = format_comparison_report(result, params)
        assert "120W" in report
        assert "100W" in report

    def test_includes_work_comparison(self):
        result = ComparisonResult(
            route_distance=10000.0,
            trip_distance=10000.0,
            predicted_time=3600.0,
            actual_moving_time=3600.0,
            time_error_pct=0.0,
            predicted_work=500000.0,  # 500 kJ
            actual_work=400000.0,  # 400 kJ
            grade_buckets=[],
            has_power_data=True,
            actual_avg_power=100.0,
        )
        params = RiderParams(assumed_avg_power=100.0)
        report = format_comparison_report(result, params)
        assert "500 kJ" in report
        assert "400 kJ" in report
        assert "+25%" in report  # (500-400)/400 = 25%

    def test_includes_grade_buckets(self):
        bucket = GradeBucket(
            grade_pct=4,
            actual_speeds=[3.0] * 20,
            predicted_speeds=[2.5] * 20,
            actual_powers=[150.0] * 20,
            point_count=20,
        )
        result = ComparisonResult(
            route_distance=10000.0,
            trip_distance=10000.0,
            predicted_time=3600.0,
            actual_moving_time=3600.0,
            time_error_pct=0.0,
            predicted_work=500000.0,
            actual_work=540000.0,
            grade_buckets=[bucket],
            has_power_data=True,
            actual_avg_power=150.0,
        )
        params = RiderParams(assumed_avg_power=100.0)
        report = format_comparison_report(result, params)
        assert "+4%" in report
        assert "150W" in report
