"""Tests for training data management."""

import json
import tempfile
from pathlib import Path

import pytest

from gpx_analyzer.training import (
    TrainingRoute,
    TrainingSummary,
    TrainingResult,
    load_training_data,
    format_training_summary,
)
from gpx_analyzer.compare import ComparisonResult, GradeBucket
from gpx_analyzer.models import RiderParams


class TestLoadTrainingData:
    def test_loads_valid_json(self):
        data = {
            "routes": [
                {
                    "name": "Test Route",
                    "route_url": "https://ridewithgps.com/routes/123",
                    "trip_url": "https://ridewithgps.com/trips/456",
                    "tags": ["road", "flat"],
                    "notes": "A test route",
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        routes = load_training_data(path)
        path.unlink()

        assert len(routes) == 1
        assert routes[0].name == "Test Route"
        assert routes[0].route_url == "https://ridewithgps.com/routes/123"
        assert routes[0].trip_url == "https://ridewithgps.com/trips/456"
        assert routes[0].tags == ["road", "flat"]
        assert routes[0].notes == "A test route"

    def test_loads_multiple_routes(self):
        data = {
            "routes": [
                {
                    "name": "Route 1",
                    "route_url": "https://ridewithgps.com/routes/1",
                    "trip_url": "https://ridewithgps.com/trips/1",
                    "tags": ["road"],
                },
                {
                    "name": "Route 2",
                    "route_url": "https://ridewithgps.com/routes/2",
                    "trip_url": "https://ridewithgps.com/trips/2",
                    "tags": ["gravel"],
                },
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        routes = load_training_data(path)
        path.unlink()

        assert len(routes) == 2
        assert routes[0].name == "Route 1"
        assert routes[1].name == "Route 2"

    def test_handles_missing_optional_fields(self):
        data = {
            "routes": [
                {
                    "name": "Minimal Route",
                    "route_url": "https://ridewithgps.com/routes/123",
                    "trip_url": "https://ridewithgps.com/trips/456",
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        routes = load_training_data(path)
        path.unlink()

        assert len(routes) == 1
        assert routes[0].tags == []
        assert routes[0].notes == ""
        assert routes[0].avg_watts is None

    def test_loads_avg_watts(self):
        data = {
            "routes": [
                {
                    "name": "Power Route",
                    "route_url": "https://ridewithgps.com/routes/123",
                    "trip_url": "https://ridewithgps.com/trips/456",
                    "avg_watts": 150,
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        routes = load_training_data(path)
        path.unlink()

        assert routes[0].avg_watts == 150

    def test_empty_routes_array(self):
        data = {"routes": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        routes = load_training_data(path)
        path.unlink()

        assert routes == []


class TestTrainingRoute:
    def test_dataclass_construction(self):
        route = TrainingRoute(
            name="Test",
            route_url="https://ridewithgps.com/routes/123",
            trip_url="https://ridewithgps.com/trips/456",
            tags=["road"],
            notes="Test notes",
        )
        assert route.name == "Test"
        assert route.tags == ["road"]


class TestFormatTrainingSummary:
    def test_includes_model_params(self):
        params = RiderParams(total_mass=80.0, cda=0.30, crr=0.010, climbing_power=150.0, flat_power=120.0)
        summary = TrainingSummary(
            total_routes=0,
            total_distance_km=0,
            total_elevation_m=0,
            avg_time_error_pct=0,
            avg_work_error_pct=0,
            results_by_tag={},
            road_avg_time_error=None,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )
        report = format_training_summary([], summary, params)
        assert "mass=80.0kg" in report
        assert "cda=0.3" in report
        assert "crr=0.01" in report

    def test_includes_route_count(self):
        params = RiderParams()
        summary = TrainingSummary(
            total_routes=5,
            total_distance_km=250,
            total_elevation_m=5000,
            avg_time_error_pct=10.5,
            avg_work_error_pct=8.2,
            results_by_tag={},
            road_avg_time_error=None,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )
        report = format_training_summary([], summary, params)
        assert "Routes analyzed: 5" in report
        assert "250 km" in report
        assert "5000 m" in report

    def test_includes_error_summary(self):
        params = RiderParams()
        summary = TrainingSummary(
            total_routes=1,
            total_distance_km=50,
            total_elevation_m=1000,
            avg_time_error_pct=15.5,
            avg_work_error_pct=12.3,
            results_by_tag={},
            road_avg_time_error=None,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )
        report = format_training_summary([], summary, params)
        assert "+15.5%" in report
        assert "+12.3%" in report

    def test_includes_terrain_breakdown(self):
        params = RiderParams()

        # Create mock results
        route = TrainingRoute(
            name="Road Route",
            route_url="https://ridewithgps.com/routes/1",
            trip_url="https://ridewithgps.com/trips/1",
            tags=["road"],
        )
        comparison = ComparisonResult(
            route_distance=50000,
            trip_distance=50000,
            predicted_time=7200,
            actual_moving_time=6000,
            time_error_pct=20.0,
            predicted_work=500000,
            actual_work=400000,
            grade_buckets=[],
            has_power_data=True,
            actual_avg_power=100.0,
        )
        result = TrainingResult(
            route=route,
            comparison=comparison,
            route_elevation_gain=1000,
            trip_elevation_gain=900,
            route_distance=50000,
            unpaved_pct=0,
            power_used=100.0,
        )

        summary = TrainingSummary(
            total_routes=1,
            total_distance_km=50,
            total_elevation_m=1000,
            avg_time_error_pct=20.0,
            avg_work_error_pct=25.0,
            results_by_tag={"road": [result]},
            road_avg_time_error=20.0,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )
        report = format_training_summary([result], summary, params)
        assert "Road (1 routes)" in report
        assert "+20.0%" in report

    def test_includes_per_route_breakdown(self):
        params = RiderParams()

        route = TrainingRoute(
            name="My Test Route",
            route_url="https://ridewithgps.com/routes/1",
            trip_url="https://ridewithgps.com/trips/1",
            tags=["road"],
        )
        comparison = ComparisonResult(
            route_distance=75000,
            trip_distance=75000,
            predicted_time=10800,
            actual_moving_time=9000,
            time_error_pct=20.0,
            predicted_work=600000,
            actual_work=500000,
            grade_buckets=[],
            has_power_data=True,
            actual_avg_power=100.0,
        )
        result = TrainingResult(
            route=route,
            comparison=comparison,
            route_elevation_gain=1500,
            trip_elevation_gain=1400,
            route_distance=75000,
            unpaved_pct=0,
            power_used=120.0,
        )

        summary = TrainingSummary(
            total_routes=1,
            total_distance_km=75,
            total_elevation_m=1500,
            avg_time_error_pct=20.0,
            avg_work_error_pct=20.0,
            results_by_tag={},
            road_avg_time_error=None,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )
        report = format_training_summary([result], summary, params)
        assert "My Test Route" in report
        assert "75k" in report
        assert "1500m" in report


class TestVerboseMetrics:
    """Tests for verbose metrics calculation."""

    def test_estimated_verbose_metrics_imports_correctly(self):
        """Ensure the estimated verbose metrics function can be imported and called."""
        from gpx_analyzer.training import _calculate_estimated_verbose_metrics
        from gpx_analyzer.models import TrackPoint

        # Create simple track points with varying grades
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.001, lon=6.0, elevation=110.0, time=None),  # Uphill
            TrackPoint(lat=45.002, lon=6.0, elevation=110.0, time=None),  # Flat
            TrackPoint(lat=45.003, lon=6.0, elevation=100.0, time=None),  # Downhill
        ]

        params = RiderParams(
            total_mass=80.0,
            climbing_power=225.0,  # Direct climbing power
            flat_power=150.0,  # Direct flat power
        )

        # This should not raise an ImportError
        result = _calculate_estimated_verbose_metrics(points, params)

        # Basic checks
        assert result is not None
        assert result.avg_power_climbing == 225.0  # climbing_power
        assert result.avg_power_flat == 150.0  # flat_power
        assert result.avg_power_descending == 0.0  # Model assumes coasting

    def test_flat_power_applied(self):
        """Ensure flat_power is used correctly in calculations."""
        from gpx_analyzer.training import _calculate_estimated_verbose_metrics
        from gpx_analyzer.models import TrackPoint

        # Create simple track points with varying grades
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.001, lon=6.0, elevation=110.0, time=None),  # Uphill
            TrackPoint(lat=45.002, lon=6.0, elevation=110.0, time=None),  # Flat
            TrackPoint(lat=45.003, lon=6.0, elevation=100.0, time=None),  # Downhill
        ]

        params = RiderParams(
            total_mass=80.0,
            climbing_power=225.0,  # 225W on climbs
            flat_power=135.0,  # 135W on flats
        )

        result = _calculate_estimated_verbose_metrics(points, params)

        # Verify direct power values are used
        assert result.avg_power_climbing == 225.0
        assert result.avg_power_flat == 135.0
        assert result.avg_power_descending == 0.0

    def test_actual_verbose_metrics_calculation(self):
        """Test calculation of verbose metrics from trip data."""
        from gpx_analyzer.training import _calculate_verbose_metrics
        from gpx_analyzer.ridewithgps import TripPoint

        # Create trip points with power and speed data
        points = [
            TripPoint(lat=45.0, lon=6.0, elevation=100.0, distance=0, speed=5.0,
                      timestamp=0.0, power=150.0, heart_rate=None, cadence=None),
            TripPoint(lat=45.001, lon=6.0, elevation=110.0, distance=111, speed=3.0,
                      timestamp=30.0, power=200.0, heart_rate=None, cadence=None),  # Climbing
            TripPoint(lat=45.002, lon=6.0, elevation=110.0, distance=222, speed=5.0,
                      timestamp=60.0, power=100.0, heart_rate=None, cadence=None),  # Flat
            TripPoint(lat=45.003, lon=6.0, elevation=100.0, distance=333, speed=10.0,
                      timestamp=90.0, power=50.0, heart_rate=None, cadence=None),  # Descending
        ]

        max_coasting_speed_ms = 15.0  # 54 km/h

        result = _calculate_verbose_metrics(points, max_coasting_speed_ms)

        assert result is not None
        # Check that time percentages sum to ~100%
        total_pct = result.time_climbing_pct + result.time_flat_pct + result.time_descending_pct
        assert 99.0 <= total_pct <= 101.0
