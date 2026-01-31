"""Unit tests for web interface."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gpx_analyzer import web
from gpx_analyzer.models import TrackPoint


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    web.app.config["TESTING"] = True
    with web.app.test_client() as client:
        yield client


@pytest.fixture
def no_config(tmp_path, monkeypatch):
    """Ensure no config files exist."""
    from gpx_analyzer import ridewithgps
    local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
    monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
    global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
    monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)


class TestIndexGet:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_form(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert "<form" in html
        assert 'name="url"' in html
        assert 'name="power"' in html
        assert 'name="mass"' in html
        assert 'name="headwind"' in html

    def test_contains_title(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert "Reality Check my Route" in html

    def test_contains_mode_indicator(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'name="mode"' in html  # Hidden input for mode
        assert 'class="mode-indicator"' in html
        assert "route or collection" in html

    def test_contains_imperial_toggle(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'name="imperial"' in html
        assert "Imperial units" in html


class TestIndexPostSingleRoute:
    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points for testing."""
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=15.0, time=None),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=20.0, time=None),
        ]

    def test_invalid_url_shows_error(self, client, no_config):
        response = client.post("/", data={
            "url": "https://example.com/routes/123",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Invalid RideWithGPS route URL" in html

    def test_empty_url_shows_error(self, client, no_config):
        response = client.post("/", data={
            "url": "",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Please enter a RideWithGPS URL" in html

    def test_invalid_power_shows_error(self, client, no_config):
        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "not-a-number",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Invalid number" in html

    @patch.object(web, "get_route_with_surface")
    def test_valid_route_shows_results(self, mock_get_route, client, no_config, mock_route_points):
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 10},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert response.status_code == 200
        assert "Test Route" in html
        assert "Distance" in html
        assert "Estimated Time" in html
        assert "Estimated Work" in html

    @patch.object(web, "get_route_with_surface")
    def test_result_has_data_attributes_for_unit_conversion(self, mock_get_route, client, no_config, mock_route_points):
        """Results should have data attributes with metric values for JS unit conversion."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Check data attributes exist for client-side unit conversion
        assert 'data-km="' in html
        assert 'data-m="' in html
        assert 'data-kmh="' in html
        assert 'id="singleDistance"' in html
        assert 'id="singleElevGain"' in html
        assert 'id="singleSpeed"' in html

    @patch.object(web, "get_route_with_surface")
    def test_imperial_toggle_js_function_exists(self, mock_get_route, client, no_config, mock_route_points):
        """JavaScript function to update units should exist."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "updateSingleRouteUnits" in html
        assert "isImperial()" in html

    @patch.object(web, "get_route_with_surface")
    def test_metric_units_displayed(self, mock_get_route, client, no_config, mock_route_points):
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "km" in html

    @patch.object(web, "get_route_with_surface")
    def test_route_error_displayed(self, mock_get_route, client, no_config):
        mock_get_route.side_effect = Exception("Network error")

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Error analyzing route" in html
        assert "Network error" in html


class TestIndexPostCollection:
    def test_collection_mode_returns_form(self, client, no_config):
        """Collection mode should return the form (handled by JS/SSE)."""
        response = client.post("/", data={
            "url": "https://ridewithgps.com/collections/12345",
            "mode": "collection",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Collection mode doesn't do server-side analysis
        assert response.status_code == 200
        assert "<form" in html


class TestCollectionStreamEndpoint:
    def test_invalid_collection_url_returns_error(self, client, no_config):
        response = client.get("/analyze-collection-stream", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })

        assert response.status_code == 200
        assert response.content_type.startswith("text/event-stream")

        # Read the SSE data
        data = response.data.decode()
        assert "error" in data
        assert "Invalid collection URL" in data

    def test_invalid_parameters_returns_error(self, client, no_config):
        response = client.get("/analyze-collection-stream", query_string={
            "url": "https://ridewithgps.com/collections/12345",
            "power": "not-a-number",
            "mass": "85",
            "headwind": "0",
        })

        data = response.data.decode()
        assert "error" in data
        assert "Invalid parameters" in data

    @patch.object(web, "get_collection_route_ids")
    def test_empty_collection_returns_error(self, mock_get_ids, client, no_config):
        mock_get_ids.return_value = ([], "Empty Collection")

        response = client.get("/analyze-collection-stream", query_string={
            "url": "https://ridewithgps.com/collections/12345",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })

        data = response.data.decode()
        assert "error" in data
        assert "No routes found" in data

    @patch.object(web, "get_collection_route_ids")
    @patch.object(web, "analyze_single_route")
    def test_streams_route_results(self, mock_analyze, mock_get_ids, client, no_config):
        mock_get_ids.return_value = ([111, 222], "Test Collection")
        mock_analyze.side_effect = [
            {
                "name": "Route 1",
                "distance_km": 50,
                "elevation_m": 500,
                "time_seconds": 7200,
                "time_str": "2h 00m",
                "work_kj": 1000,
                "avg_speed_kmh": 25,
                "unpaved_pct": 0,
                "elevation_scale": 1.0,
            },
            {
                "name": "Route 2",
                "distance_km": 30,
                "elevation_m": 300,
                "time_seconds": 3600,
                "time_str": "1h 00m",
                "work_kj": 600,
                "avg_speed_kmh": 30,
                "unpaved_pct": 10,
                "elevation_scale": 1.05,
            },
        ]

        response = client.get("/analyze-collection-stream", query_string={
            "url": "https://ridewithgps.com/collections/12345",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })

        data = response.data.decode()

        # Should contain start event
        assert '"type": "start"' in data
        assert "Test Collection" in data

        # Should contain progress events
        assert '"type": "progress"' in data

        # Should contain route events
        assert '"type": "route"' in data
        assert "Route 1" in data
        assert "Route 2" in data

        # Should contain complete event
        assert '"type": "complete"' in data

    @patch.object(web, "get_collection_route_ids")
    @patch.object(web, "analyze_single_route")
    def test_continues_after_route_error(self, mock_analyze, mock_get_ids, client, no_config):
        """Should continue processing if one route fails."""
        mock_get_ids.return_value = ([111, 222, 333], "Test Collection")
        mock_analyze.side_effect = [
            {
                "name": "Route 1",
                "distance_km": 50,
                "elevation_m": 500,
                "time_seconds": 7200,
                "time_str": "2h 00m",
                "work_kj": 1000,
                "avg_speed_kmh": 25,
                "unpaved_pct": 0,
                "elevation_scale": 1.0,
            },
            Exception("Route 222 failed"),  # Second route fails
            {
                "name": "Route 3",
                "distance_km": 40,
                "elevation_m": 400,
                "time_seconds": 5000,
                "time_str": "1h 23m",
                "work_kj": 800,
                "avg_speed_kmh": 28,
                "unpaved_pct": 5,
                "elevation_scale": 0.98,
            },
        ]

        response = client.get("/analyze-collection-stream", query_string={
            "url": "https://ridewithgps.com/collections/12345",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })

        data = response.data.decode()

        # Should complete despite error
        assert '"type": "complete"' in data
        # Should have both successful routes
        assert "Route 1" in data
        assert "Route 3" in data


class TestHelperFunctions:
    def test_format_duration_minutes_only(self):
        assert web.format_duration(1800) == "30m"
        assert web.format_duration(300) == "5m"

    def test_format_duration_hours_and_minutes(self):
        assert web.format_duration(3600) == "1h 00m"
        assert web.format_duration(3720) == "1h 02m"
        assert web.format_duration(7200) == "2h 00m"
        assert web.format_duration(9000) == "2h 30m"

    def test_format_duration_long(self):
        assert web.format_duration_long(3661) == "1h 01m 01s"
        assert web.format_duration_long(7325) == "2h 02m 05s"

    def test_get_defaults_uses_config(self, tmp_path, monkeypatch):
        from gpx_analyzer import ridewithgps

        config_path = tmp_path / "gpx-analyzer.json"
        config_path.write_text('{"power": 150, "mass": 90, "headwind": 5}')
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", config_path)
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent.json")

        defaults = web.get_defaults()

        assert defaults["power"] == 150
        assert defaults["mass"] == 90
        assert defaults["headwind"] == 5

    def test_get_defaults_fallback(self, no_config):
        from gpx_analyzer.cli import DEFAULTS

        defaults = web.get_defaults()

        assert defaults["power"] == DEFAULTS["power"]
        assert defaults["mass"] == DEFAULTS["mass"]
        assert defaults["headwind"] == DEFAULTS["headwind"]

    def test_build_params(self, no_config):
        params = web.build_params(power=120, mass=80, headwind=10)

        assert params.assumed_avg_power == 120
        assert params.total_mass == 80
        assert params.headwind == 10 / 3.6  # Converted from km/h to m/s


class TestInfoModals:
    def test_power_modal_exists(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'id="powerModal"' in html
        assert "Average Power" in html

    def test_mass_modal_exists(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'id="massModal"' in html
        assert "Total Mass" in html

    def test_headwind_modal_exists(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'id="headwindModal"' in html
        assert "Headwind" in html

    def test_physics_modal_exists(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'id="physicsModal"' in html
        assert "Physics Model" in html
        assert "CdA" in html
        assert "Rolling resistance" in html


class TestElevationScaling:
    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points with elevation."""
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=0.0, time=None),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=50.0, time=None),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=100.0, time=None),
        ]

    @patch.object(web, "get_route_with_surface")
    def test_elevation_scaled_note_displayed(self, mock_get_route, client, no_config, mock_route_points):
        """When elevation is significantly scaled, show a note."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 150},  # Higher than computed
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "power": "100",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Elevation scaled" in html
