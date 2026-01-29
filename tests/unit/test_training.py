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
        params = RiderParams(total_mass=80.0, cda=0.30, crr=0.010, assumed_avg_power=150.0)
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
