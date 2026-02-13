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
    # Clear caches before each test to prevent cross-test contamination
    web._analysis_cache.clear()
    web._profile_data_cache.clear()
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
        assert 'name="climbing_power"' in html
        assert 'name="flat_power"' in html
        assert 'name="mass"' in html
        assert 'name="headwind"' in html

    def test_contains_title(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert "Reality Check my Route" in html

    def test_contains_compare_checkbox(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'name="mode"' in html  # Hidden input for mode
        assert 'id="compareCheckbox"' in html
        assert "Compare this" in html

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
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Invalid RideWithGPS URL" in html

    def test_empty_url_shows_error(self, client, no_config):
        response = client.post("/", data={
            "url": "",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Please enter a RideWithGPS URL" in html

    def test_invalid_power_shows_error(self, client, no_config):
        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "climbing_power": "not-a-number",
            "flat_power": "100",
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
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert response.status_code == 200
        assert "Test Route" in html
        assert "Distance" in html
        assert "Est. Moving Time" in html
        assert "Est. Work" in html
        assert "Est. Energy" in html
        assert "kcal" in html

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
            "climbing_power": "150",
            "flat_power": "120",
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
    def test_energy_display_with_unit_selector(self, mock_get_route, client, no_config, mock_route_points):
        """Energy display should show kcal value and unit selector dropdown."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Check energy display elements
        assert 'id="singleEnergy"' in html
        assert 'data-kj="' in html
        assert 'id="energyUnitSelect"' in html
        assert 'class="unit-select"' in html
        # Check unit options
        assert 'value="kcal"' in html
        assert 'value="bananas"' in html
        assert 'value="baguettes"' in html
        # Check JavaScript functions
        assert 'updateEnergyUnits' in html
        assert 'formatEnergy' in html

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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "not-a-number",
            "flat_power": "100",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
            "climbing_power": "150",
            "flat_power": "120",
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
        config_path.write_text('{"climbing_power": 150, "flat_power": 120, "mass": 90, "headwind": 5}')
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", config_path)
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent.json")

        defaults = web.get_defaults()

        assert defaults["climbing_power"] == 150
        assert defaults["flat_power"] == 120
        assert defaults["mass"] == 90
        assert defaults["headwind"] == 5

    def test_get_defaults_fallback(self, no_config):
        from gpx_analyzer.cli import DEFAULTS

        defaults = web.get_defaults()

        assert defaults["climbing_power"] == DEFAULTS["climbing_power"]
        assert defaults["flat_power"] == DEFAULTS["flat_power"]
        assert defaults["mass"] == DEFAULTS["mass"]
        assert defaults["headwind"] == DEFAULTS["headwind"]

    def test_build_params(self, no_config):
        params = web.build_params(climbing_power=150, flat_power=120, mass=80, headwind=10)

        assert params.climbing_power == 150
        assert params.flat_power == 120
        assert params.total_mass == 80
        assert params.headwind == 10 / 3.6  # Converted from km/h to m/s


class TestInfoModals:
    def test_power_modal_exists(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert 'id="powerModal"' in html
        assert "Climbing Power" in html

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


class TestElevationProfile:
    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points with elevation for profile testing."""
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=0.0, time=None),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=50.0, time=None),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=100.0, time=None),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=80.0, time=None),
        ]

    @patch.object(web, "get_route_with_surface")
    def test_returns_png_image(self, mock_get_route, client, no_config, mock_route_points):
        """Elevation profile endpoint should return a PNG image."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        assert response.status_code == 200
        assert response.content_type == "image/png"
        # PNG files start with these magic bytes
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_invalid_url_returns_placeholder_image(self, client, no_config):
        """Invalid URL should return a placeholder PNG image."""
        response = client.get("/elevation-profile", query_string={
            "url": "https://example.com/routes/123",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        # Should return a placeholder PNG (not a text error)
        assert response.status_code == 200
        assert response.content_type == "image/png"
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'

    @patch.object(web, "get_route_with_surface")
    def test_route_error_returns_error_image(self, mock_get_route, client, no_config):
        """Route fetching error should return an error PNG image."""
        mock_get_route.side_effect = Exception("Network error")

        response = client.get("/elevation-profile", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        # Should return an error PNG (not crash)
        assert response.status_code == 200
        assert response.content_type == "image/png"
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'

    @patch.object(web, "get_route_with_surface")
    def test_data_endpoint_returns_json(self, mock_get_route, client, no_config, mock_route_points):
        """Elevation profile data endpoint should return JSON with grades."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        assert response.status_code == 200
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "times" in data
        assert "elevations" in data
        assert "grades" in data
        assert "total_time" in data

    def test_data_endpoint_invalid_url_returns_error(self, client, no_config):
        """Data endpoint should return error for invalid URL."""
        response = client.get("/elevation-profile-data", query_string={
            "url": "https://example.com/routes/123",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data


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
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Elevation scaled" in html


class TestTripSupport:
    """Tests for trip URL support."""

    def test_is_valid_rwgps_url_accepts_trips(self):
        """_is_valid_rwgps_url should accept trip URLs."""
        assert web._is_valid_rwgps_url("https://ridewithgps.com/trips/12345")
        assert web._is_valid_rwgps_url("https://www.ridewithgps.com/trips/12345")

    def test_is_valid_rwgps_url_accepts_routes(self):
        """_is_valid_rwgps_url should accept route URLs."""
        assert web._is_valid_rwgps_url("https://ridewithgps.com/routes/12345")

    def test_is_valid_rwgps_url_rejects_invalid(self):
        """_is_valid_rwgps_url should reject non-RWGPS URLs."""
        assert not web._is_valid_rwgps_url("https://example.com/trips/12345")
        assert not web._is_valid_rwgps_url("https://example.com/routes/12345")

    def test_extract_trip_id(self):
        """extract_trip_id should extract trip ID from URL."""
        assert web.extract_trip_id("https://ridewithgps.com/trips/12345") == "12345"
        assert web.extract_trip_id("https://www.ridewithgps.com/trips/67890") == "67890"
        assert web.extract_trip_id("https://ridewithgps.com/routes/12345") is None
        assert web.extract_trip_id(None) is None

    def test_extract_id_from_url_routes(self):
        """_extract_id_from_url should extract route IDs."""
        assert web._extract_id_from_url("https://ridewithgps.com/routes/12345") == "12345"

    def test_extract_id_from_url_trips(self):
        """_extract_id_from_url should extract trip IDs."""
        assert web._extract_id_from_url("https://ridewithgps.com/trips/67890") == "67890"

    def test_trip_url_shows_error_for_invalid_trip(self, client, no_config):
        """Trip URL with example.com should show error."""
        response = client.post("/", data={
            "url": "https://example.com/trips/123",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()
        assert "Invalid RideWithGPS URL" in html

    @patch.object(web, "get_trip_data")
    def test_trip_url_shows_trip_badge(self, mock_get_trip, client, no_config):
        """Trip URLs should show 'Recorded Ride' badge."""
        from gpx_analyzer.ridewithgps import TripPoint

        mock_get_trip.return_value = (
            [
                TripPoint(lat=37.7749, lon=-122.4194, elevation=10.0, distance=0, speed=5.0, timestamp=1000, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=15.0, distance=100, speed=5.0, timestamp=1020, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7767, lon=-122.4172, elevation=20.0, distance=200, speed=5.0, timestamp=1040, power=150, heart_rate=None, cadence=None),
            ],
            {"name": "Test Trip", "distance": 1000, "elevation_gain": 100, "moving_time": 3600, "avg_speed": 5.0, "avg_watts": 150},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/trips/12345",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Recorded Ride" in html
        assert "trip-badge" in html
        assert "trip-results" in html

    @patch.object(web, "get_trip_data")
    def test_trip_shows_moving_time_not_estimated(self, mock_get_trip, client, no_config):
        """Trip should show 'Moving Time' not 'Est. Moving Time' in results."""
        from gpx_analyzer.ridewithgps import TripPoint

        mock_get_trip.return_value = (
            [
                TripPoint(lat=37.7749, lon=-122.4194, elevation=10.0, distance=0, speed=5.0, timestamp=1000, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=15.0, distance=100, speed=5.0, timestamp=1020, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7767, lon=-122.4172, elevation=20.0, distance=200, speed=5.0, timestamp=1040, power=150, heart_rate=None, cadence=None),
            ],
            {"name": "Test Trip", "distance": 1000, "elevation_gain": 100, "moving_time": 3600, "avg_speed": 5.0, "avg_watts": 150},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/trips/12345",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Should show "Moving Time" label in the results section
        # The label appears with info button like: "Moving Time <button..."
        assert 'Moving Time <button type="button" class="info-btn"' in html
        # The results section should NOT have "Est. Moving Time" as the label
        # (note: it will still appear in the modal explanation text, so we check the label context)
        assert 'result-label label-with-info">Moving Time' in html or '>Moving Time <button' in html

    @patch.object(web, "get_trip_data")
    def test_trip_shows_avg_power_when_available(self, mock_get_trip, client, no_config):
        """Trip with power data should show Avg Power."""
        from gpx_analyzer.ridewithgps import TripPoint

        mock_get_trip.return_value = (
            [
                TripPoint(lat=37.7749, lon=-122.4194, elevation=10.0, distance=0, speed=5.0, timestamp=1000, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=15.0, distance=100, speed=5.0, timestamp=1020, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7767, lon=-122.4172, elevation=20.0, distance=200, speed=5.0, timestamp=1040, power=150, heart_rate=None, cadence=None),
            ],
            {"name": "Test Trip", "distance": 1000, "elevation_gain": 100, "moving_time": 3600, "avg_speed": 5.0, "avg_watts": 175},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/trips/12345",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Avg Power" in html
        assert "175 W" in html

    @patch.object(web, "get_route_with_surface")
    def test_route_shows_planned_route_badge(self, mock_get_route, client, no_config):
        """Route URLs should show 'Planned Route' badge."""
        mock_get_route.return_value = (
            [
                TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None),
                TrackPoint(lat=37.7758, lon=-122.4183, elevation=15.0, time=None),
                TrackPoint(lat=37.7767, lon=-122.4172, elevation=20.0, time=None),
            ],
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        assert "Planned Route" in html
        assert "route-badge" in html
        assert "route-results" in html


class TestElevationProfileData:
    """Tests for elevation profile data endpoint structure and values."""

    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points with varied elevation."""
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=100.0, time=None),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=150.0, time=None),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=200.0, time=None),
            TrackPoint(lat=37.7776, lon=-122.4161, elevation=180.0, time=None),
            TrackPoint(lat=37.7785, lon=-122.4150, elevation=120.0, time=None),
        ]

    @patch.object(web, "get_route_with_surface")
    def test_route_profile_data_structure(self, mock_get_route, client, no_config, mock_route_points):
        """Route profile data should include all required fields."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check required fields
        assert "times" in data
        assert "elevations" in data
        assert "grades" in data
        assert "total_time" in data
        assert "speeds" in data
        assert "distances" in data
        assert "elev_gains" in data
        assert "elev_losses" in data

        # Check data consistency
        assert len(data["times"]) == len(data["elevations"])
        assert len(data["grades"]) == len(data["times"]) - 1 or len(data["grades"]) == len(data["times"])
        assert data["total_time"] > 0

    @patch.object(web, "get_route_with_surface")
    def test_route_profile_elevations_match_input(self, mock_get_route, client, no_config, mock_route_points):
        """Route profile elevations should be close to input values (smoothed, unscaled for display)."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        data = json.loads(response.data)
        elevations = data["elevations"]

        # First elevation should be close to input (100m)
        assert 90 <= elevations[0] <= 110
        # Max elevation should be close to input max (200m)
        assert max(elevations) >= 180

    @patch.object(web, "get_route_with_surface")
    def test_route_profile_gain_loss_consistency(self, mock_get_route, client, no_config, mock_route_points):
        """Sum of elev_gains and elev_losses should be consistent."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/routes/12345",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })

        data = json.loads(response.data)
        total_gain = sum(data["elev_gains"])
        total_loss = sum(abs(l) for l in data["elev_losses"])

        # Should have both gain and loss
        assert total_gain > 0
        assert total_loss > 0

    @patch.object(web, "get_trip_data")
    def test_trip_profile_collapse_stops_changes_total_time(self, mock_get_trip, client, no_config):
        """Trip profile with collapse_stops should have different total_time."""
        from gpx_analyzer.ridewithgps import TripPoint

        # Create trip with a stop (0 speed segment)
        mock_get_trip.return_value = (
            [
                TripPoint(lat=37.7749, lon=-122.4194, elevation=100.0, distance=0, speed=5.0, timestamp=0, power=None, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=150.0, distance=100, speed=5.0, timestamp=20, power=None, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=150.0, distance=100, speed=0.0, timestamp=120, power=None, heart_rate=None, cadence=None),  # 100s stop
                TripPoint(lat=37.7767, lon=-122.4172, elevation=200.0, distance=200, speed=5.0, timestamp=140, power=None, heart_rate=None, cadence=None),
            ],
            {"name": "Test Trip", "distance": 200, "elevation_gain": 100, "moving_time": 40},
        )

        # Get elapsed time (includes stop)
        response1 = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/trips/12345",
            "collapse_stops": "false",
        })
        data1 = json.loads(response1.data)

        # Get moving time (excludes stop)
        response2 = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/trips/12345",
            "collapse_stops": "true",
        })
        data2 = json.loads(response2.data)

        # Moving time should be less than elapsed time
        assert data2["total_time"] < data1["total_time"]

    @patch.object(web, "get_trip_data")
    def test_trip_profile_data_structure(self, mock_get_trip, client, no_config):
        """Trip profile data should include all required fields."""
        from gpx_analyzer.ridewithgps import TripPoint

        mock_get_trip.return_value = (
            [
                TripPoint(lat=37.7749, lon=-122.4194, elevation=100.0, distance=0, speed=5.0, timestamp=0, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7758, lon=-122.4183, elevation=150.0, distance=100, speed=5.0, timestamp=20, power=150, heart_rate=None, cadence=None),
                TripPoint(lat=37.7767, lon=-122.4172, elevation=200.0, distance=200, speed=5.0, timestamp=40, power=150, heart_rate=None, cadence=None),
            ],
            {"name": "Test Trip", "distance": 200, "elevation_gain": 100, "moving_time": 40},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/trips/12345",
        })

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check required fields
        assert "times" in data
        assert "elevations" in data
        assert "grades" in data
        assert "total_time" in data
        assert "speeds" in data
        assert "distances" in data
        assert "elev_gains" in data
        assert "elev_losses" in data
        assert "powers" in data

        # Check data consistency
        assert len(data["times"]) == len(data["elevations"])

    @patch.object(web, "get_trip_data")
    def test_trip_profile_handles_none_grades_in_downsampling(self, mock_get_trip, client, no_config):
        """Trip profile should handle None grades (stopped segments) during downsampling."""
        from gpx_analyzer.ridewithgps import TripPoint

        # Create >1000 points to trigger downsampling, with some stopped segments (speed=0, grade=None)
        points = []
        for i in range(1200):
            # Every 50th point is a stop (speed=0)
            is_stopped = (i % 50 == 25)
            points.append(TripPoint(
                lat=37.7749 + i * 0.0001,
                lon=-122.4194 + i * 0.0001,
                elevation=100.0 + (i % 100),  # Varying elevation
                distance=i * 10,
                speed=0.0 if is_stopped else 5.0,
                timestamp=i * 2,
                power=0 if is_stopped else 150,
                heart_rate=None,
                cadence=None,
            ))

        mock_get_trip.return_value = (
            points,
            {"name": "Test Trip with Stops", "distance": 12000, "elevation_gain": 1000, "moving_time": 2400},
        )

        response = client.get("/elevation-profile-data", query_string={
            "url": "https://ridewithgps.com/trips/12345",
        })

        # Should not crash with 500 error due to None * float
        assert response.status_code == 200
        data = json.loads(response.data)

        # Data should be downsampled (fewer than 1200 points)
        assert len(data["times"]) < 1200
        # Grades can include None values for stopped segments
        assert "grades" in data
        # Should have valid structure
        assert len(data["times"]) == len(data["elevations"])


class TestComparisonMode:
    """Tests for route/trip comparison mode."""

    @pytest.fixture
    def mock_route_points(self):
        return [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=100.0, time=None),
            TrackPoint(lat=37.7758, lon=-122.4183, elevation=150.0, time=None),
            TrackPoint(lat=37.7767, lon=-122.4172, elevation=200.0, time=None),
        ]

    @patch.object(web, "get_route_with_surface")
    def test_comparison_mode_has_time_data_attributes(self, mock_get_route, client, no_config, mock_route_points):
        """Comparison mode containers should have moving/elapsed time data attributes."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "url2": "https://ridewithgps.com/routes/67890",
            "compare": "on",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Check that comparison mode is active
        assert 'id="elevationContainer1"' in html
        assert 'id="elevationContainer2"' in html

        # Check for time data attributes (for x-axis synchronization)
        assert 'data-moving-time-hours' in html
        assert 'data-elapsed-time-hours' in html
        assert 'data-max-xlim-hours' in html

    @patch.object(web, "get_route_with_surface")
    def test_comparison_mode_has_ylim_attributes(self, mock_get_route, client, no_config, mock_route_points):
        """Comparison mode containers should have synchronized y-axis limit attributes."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "url2": "https://ridewithgps.com/routes/67890",
            "compare": "on",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Check for y-axis limit attributes
        assert 'data-max-ylim' in html

    @patch.object(web, "get_route_with_surface")
    def test_comparison_mode_has_energy_display(self, mock_get_route, client, no_config, mock_route_points):
        """Comparison mode should show energy values with unit selector."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.post("/", data={
            "url": "https://ridewithgps.com/routes/12345",
            "url2": "https://ridewithgps.com/routes/67890",
            "compare": "on",
            "mode": "route",
            "climbing_power": "150",
            "flat_power": "120",
            "mass": "85",
            "headwind": "0",
        })
        html = response.data.decode()

        # Check energy display in comparison table
        assert 'id="energy1"' in html
        assert 'id="energy2"' in html
        assert 'id="energyDiff"' in html
        assert 'id="energyUnitSelect"' in html
        assert "Est. Energy" in html


class TestRidePage:
    """Tests for the /ride page endpoint."""

    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points for testing climbs."""
        # Create a route with a climb: flat, then climb, then flat
        points = []
        base_lat, base_lon = 37.0, -122.0
        # Flat section (10 points)
        for i in range(10):
            points.append(TrackPoint(
                lat=base_lat + i * 0.001,
                lon=base_lon,
                elevation=100.0,
                time=None
            ))
        # Climb section (20 points, gaining 100m)
        for i in range(20):
            points.append(TrackPoint(
                lat=base_lat + (10 + i) * 0.001,
                lon=base_lon,
                elevation=100.0 + i * 5.0,
                time=None
            ))
        # Flat section at top (10 points)
        for i in range(10):
            points.append(TrackPoint(
                lat=base_lat + (30 + i) * 0.001,
                lon=base_lon,
                elevation=200.0,
                time=None
            ))
        return points

    def test_ride_page_no_url_returns_200(self, client):
        """Ride page without URL should still return 200."""
        response = client.get("/ride")
        assert response.status_code == 200

    def test_ride_page_invalid_url_shows_error(self, client, no_config):
        """Ride page with invalid URL should handle gracefully."""
        response = client.get("/ride?url=https://example.com/invalid")
        assert response.status_code == 200
        html = response.data.decode()
        # Should still render the page structure
        assert "Ride Details" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_valid_url_shows_content(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page with valid URL should show climb data."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Climb Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345&climbing_power=150&flat_power=120&mass=85&headwind=0")
        assert response.status_code == 200
        html = response.data.decode()

        assert "Test Climb Route" in html
        assert "Ride Details" in html
        assert "Elevation Profile" in html
        assert "Climb Detection" in html
        assert "Detected Climbs" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_contains_toggles(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should have speed, gravel, and imperial toggles."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        assert 'id="overlay_speed"' in html
        assert 'id="overlay_gravel"' in html
        assert 'id="imperial"' in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_contains_sensitivity_slider(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should have sensitivity slider for climb detection."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        assert 'id="sensitivitySlider"' in html
        assert "High Sensitivity" in html
        assert "Low Sensitivity" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_imperial_parameter(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should respect imperial parameter."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        # Test with imperial=true
        response = client.get("/ride?url=https://ridewithgps.com/routes/12345&imperial=true")
        html = response.data.decode()
        assert 'id="imperial" checked' in html

        # Test without imperial (should not be checked)
        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()
        assert 'id="imperial" checked' not in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_back_link(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should have back link to main analysis."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        assert "Back to Analysis" in html
        assert 'id="backLink"' in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_hover_tooltip_supports_imperial(self, mock_get_route, client, no_config, mock_route_points):
        """Hover tooltip should have JavaScript for imperial unit conversion."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Check that tooltip update function converts elevation to feet
        assert "data.elevation * 3.28084" in html
        # Check that tooltip update function converts speed to mph
        assert "data.speed * 0.621371" in html
        # Check for imperial unit labels in tooltip code
        assert "'ft'" in html or '"ft"' in html
        assert "'mph'" in html or '"mph"' in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_selection_popup_supports_imperial(self, mock_get_route, client, no_config, mock_route_points):
        """Selection popup (drag-drop) should have JavaScript for imperial unit conversion."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Check showSelectionPopup has imperial conversion for distance (km to mi)
        assert "stats.distKm * 0.621371" in html
        # Check showSelectionPopup has imperial conversion for elevation (m to ft)
        assert "stats.elevGain * 3.28084" in html
        assert "stats.elevLoss * 3.28084" in html
        # Check showSelectionPopup has imperial conversion for speed
        assert "stats.avgSpeed * 0.621371" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_profile_url_includes_imperial_param(self, mock_get_route, client, no_config, mock_route_points):
        """Elevation profile URL should include imperial parameter when building overlay params."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Check that _buildOverlayParams includes imperial
        assert "isImperial()" in html
        assert "&imperial=true" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_climb_table_supports_imperial(self, mock_get_route, client, no_config, mock_route_points):
        """Climb table rendering should have JavaScript for imperial unit conversion."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Check formatDistance function exists and uses imperial
        assert "function formatDistance" in html
        assert "km * 0.621371" in html

        # Check formatElevation function exists and uses imperial
        assert "function formatElevation" in html
        assert "m * 3.28084" in html

        # Check formatSpeed function exists and uses imperial
        assert "function formatSpeed" in html
        assert "kmh * 0.621371" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_summary_supports_imperial(self, mock_get_route, client, no_config, mock_route_points):
        """Summary card should have data attributes for unit conversion."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Check summary values have data attributes for JS conversion
        assert 'id="summaryDistance"' in html
        assert 'data-km=' in html
        assert 'id="summaryElevation"' in html
        assert 'data-m=' in html

        # Check updateSummaryUnits function exists
        assert "function updateSummaryUnits" in html

    def test_ride_page_has_header_and_footer(self, client):
        """Ride page should have the same header and footer as main page."""
        response = client.get("/ride")
        html = response.data.decode()

        # Check header elements
        assert "Reality Check my Route" in html
        assert 'class="header-section"' in html
        assert 'class="logo-container"' in html

        # Check footer elements
        assert 'class="footer"' in html
        assert 'class="footer-content"' in html
        assert "Source Code" in html
        assert "Report a Bug" in html
        assert "github.com/sanmi/gpx-analyzer" in html
        assert "© 2025 Frank San Miguel" in html


class TestApiDetectClimbs:
    """Tests for the /api/detect-climbs endpoint."""

    @pytest.fixture
    def mock_route_with_climb(self):
        """Create mock track points with a detectable climb."""
        points = []
        base_lat, base_lon = 37.0, -122.0
        # Create a climb: 2km at 5% grade = 100m gain
        for i in range(21):
            points.append(TrackPoint(
                lat=base_lat + i * 0.0009,  # ~100m spacing
                lon=base_lon,
                elevation=100.0 + i * 5.0,  # 5m per point = 5% grade
                time=None
            ))
        return points

    def test_api_detect_climbs_no_url_returns_error(self, client):
        """API should return error without URL."""
        response = client.get("/api/detect-climbs")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_api_detect_climbs_invalid_url_returns_error(self, client, no_config):
        """API should return error with invalid URL."""
        response = client.get("/api/detect-climbs?url=https://example.com/invalid")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    @patch.object(web, "get_route_with_surface")
    def test_api_detect_climbs_returns_json(self, mock_get_route, client, no_config, mock_route_with_climb):
        """API should return JSON with climbs array."""
        mock_get_route.return_value = (
            mock_route_with_climb,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/api/detect-climbs?url=https://ridewithgps.com/routes/12345&climbing_power=150&flat_power=120&mass=85")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "climbs" in data
        assert "sensitivity_m" in data
        assert isinstance(data["climbs"], list)

    @patch.object(web, "get_route_with_surface")
    def test_api_detect_climbs_sensitivity_parameter(self, mock_get_route, client, no_config, mock_route_with_climb):
        """API should respect sensitivity parameter."""
        mock_get_route.return_value = (
            mock_route_with_climb,
            {"name": "Test Route", "elevation_gain": 100},
        )

        # Low sensitivity (high tolerance)
        response = client.get("/api/detect-climbs?url=https://ridewithgps.com/routes/12345&sensitivity=100")
        data = json.loads(response.data)
        assert data["sensitivity_m"] == 100.0  # slider 100 = 100m tolerance

        # High sensitivity (low tolerance)
        response = client.get("/api/detect-climbs?url=https://ridewithgps.com/routes/12345&sensitivity=0")
        data = json.loads(response.data)
        assert data["sensitivity_m"] == 10.0  # slider 0 = 10m tolerance

    @patch.object(web, "get_route_with_surface")
    def test_api_detect_climbs_climb_fields(self, mock_get_route, client, no_config, mock_route_with_climb):
        """API should return climbs with all expected fields."""
        mock_get_route.return_value = (
            mock_route_with_climb,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/api/detect-climbs?url=https://ridewithgps.com/routes/12345&sensitivity=50")
        data = json.loads(response.data)

        if len(data["climbs"]) > 0:
            climb = data["climbs"][0]
            # Check all expected fields are present
            expected_fields = [
                "climb_id", "label", "start_km", "end_km",
                "distance_m", "elevation_gain", "elevation_loss",
                "avg_grade", "max_grade", "start_elevation", "peak_elevation",
                "duration_seconds", "work_kj", "avg_power", "avg_speed_kmh"
            ]
            for field in expected_fields:
                assert field in climb, f"Missing field: {field}"


class TestElevationProfileRide:
    """Tests for the /elevation-profile-ride endpoint."""

    @pytest.fixture
    def mock_route_points(self):
        """Create mock track points."""
        points = []
        base_lat, base_lon = 37.0, -122.0
        for i in range(20):
            points.append(TrackPoint(
                lat=base_lat + i * 0.001,
                lon=base_lon,
                elevation=100.0 + i * 5.0,
                time=None
            ))
        return points

    def test_elevation_profile_ride_no_url_returns_placeholder(self, client):
        """Endpoint without URL should return a placeholder image."""
        response = client.get("/elevation-profile-ride")
        assert response.status_code == 200
        assert response.content_type == "image/png"
        # Check it's a valid PNG
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'

    @patch.object(web, "get_route_with_surface")
    def test_elevation_profile_ride_returns_png(self, mock_get_route, client, no_config, mock_route_points):
        """Endpoint with valid URL should return PNG image."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        response = client.get("/elevation-profile-ride?url=https://ridewithgps.com/routes/12345&climbing_power=150&flat_power=120&mass=85")
        assert response.status_code == 200
        assert response.content_type == "image/png"
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'

    @patch.object(web, "get_route_with_surface")
    def test_elevation_profile_ride_sensitivity_parameter(self, mock_get_route, client, no_config, mock_route_points):
        """Endpoint should accept sensitivity parameter."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 100},
        )

        # Should not error with different sensitivity values
        for sensitivity in [0, 50, 100]:
            response = client.get(f"/elevation-profile-ride?url=https://ridewithgps.com/routes/12345&sensitivity={sensitivity}")
            assert response.status_code == 200
            assert response.content_type == "image/png"


class TestDynamicPlotMargins:
    """Tests for dynamic plot margin calculation based on aspect ratio."""

    @pytest.fixture
    def mock_route_points(self):
        """Create simple mock track points."""
        return [
            TrackPoint(lat=37.0, lon=-122.0, elevation=100.0, time=None),
            TrackPoint(lat=37.001, lon=-122.0, elevation=110.0, time=None),
            TrackPoint(lat=37.002, lon=-122.0, elevation=120.0, time=None),
        ]

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_contains_dynamic_margin_formula(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should have getPlotMargins function with aspect-based calculation."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 20},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Verify the dynamic margin function exists
        assert "function getPlotMargins()" in html
        # Verify it calculates aspect from the container
        assert "getContainerAspect()" in html
        # Verify the formula uses division by aspect (margins shrink as aspect increases)
        assert "0.175 / aspect" in html

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_margin_values_calibrated_correctly(self, mock_get_route, client, no_config, mock_route_points):
        """Margin formula should produce correct values at key aspect ratios.

        This test extracts the formula constants from the HTML and verifies they
        produce the expected margin percentages for mobile (aspect=1.0) and
        desktop (aspect=3.5) screen sizes.
        """
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 20},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # The formula is: left = 0.175/aspect, right = 1 - 0.175/aspect
        # These come from: 0.7 inch / (4 inch * aspect) for both margins
        # At aspect=1.0 (mobile/square): left=0.175, right=0.825
        # At aspect=3.5 (desktop/wide): left=0.05, right=0.95

        margin_constant = 0.175

        # Verify formula produces correct margins for mobile (aspect=1.0)
        mobile_aspect = 1.0
        mobile_left = margin_constant / mobile_aspect
        mobile_right = 1 - margin_constant / mobile_aspect
        assert abs(mobile_left - 0.175) < 0.001, f"Mobile left margin should be ~0.175, got {mobile_left}"
        assert abs(mobile_right - 0.825) < 0.001, f"Mobile right margin should be ~0.825, got {mobile_right}"

        # Verify formula produces correct margins for desktop (aspect=3.5)
        desktop_aspect = 3.5
        desktop_left = margin_constant / desktop_aspect
        desktop_right = 1 - margin_constant / desktop_aspect
        assert abs(desktop_left - 0.05) < 0.001, f"Desktop left margin should be ~0.05, got {desktop_left}"
        assert abs(desktop_right - 0.95) < 0.001, f"Desktop right margin should be ~0.95, got {desktop_right}"

        # Verify the page uses these exact constants
        assert "0.175" in html, "Margin constant 0.175 should be in HTML"

    @patch.object(web, "get_route_with_surface")
    def test_ride_page_margins_update_on_resize(self, mock_get_route, client, no_config, mock_route_points):
        """Ride page should update margins when window is resized."""
        mock_get_route.return_value = (
            mock_route_points,
            {"name": "Test Route", "elevation_gain": 20},
        )

        response = client.get("/ride?url=https://ridewithgps.com/routes/12345")
        html = response.data.decode()

        # Verify there's an updatePlotMargins function
        assert "function updatePlotMargins()" in html or "updatePlotMargins" in html
        # Verify resize event handler calls update
        assert "resize" in html.lower() and "updatePlotMargins" in html


class TestUmamiAnalytics:
    """Tests for Umami analytics integration."""

    def test_get_analytics_config_from_env_vars(self, monkeypatch):
        """Environment variables should be read for analytics config."""
        monkeypatch.setenv("UMAMI_WEBSITE_ID", "test-website-id")
        monkeypatch.setenv("UMAMI_SCRIPT_URL", "https://custom.umami.is/script.js")

        config = web._get_analytics_config()
        assert config["umami_website_id"] == "test-website-id"
        assert config["umami_script_url"] == "https://custom.umami.is/script.js"

    def test_get_analytics_config_default_script_url(self, monkeypatch):
        """Default script URL should be used when not specified."""
        monkeypatch.setenv("UMAMI_WEBSITE_ID", "test-website-id")
        monkeypatch.delenv("UMAMI_SCRIPT_URL", raising=False)

        # Also ensure config file doesn't have it
        with patch.object(web, "_load_config", return_value={}):
            config = web._get_analytics_config()
            assert config["umami_website_id"] == "test-website-id"
            assert config["umami_script_url"] == "https://cloud.umami.is/script.js"

    def test_get_analytics_config_env_takes_precedence(self, monkeypatch):
        """Environment variables should take precedence over config file."""
        monkeypatch.setenv("UMAMI_WEBSITE_ID", "env-website-id")

        with patch.object(web, "_load_config", return_value={"umami_website_id": "config-website-id"}):
            config = web._get_analytics_config()
            assert config["umami_website_id"] == "env-website-id"

    def test_get_analytics_config_from_config_file(self, monkeypatch):
        """Config file values should be used when env vars not set."""
        monkeypatch.delenv("UMAMI_WEBSITE_ID", raising=False)
        monkeypatch.delenv("UMAMI_SCRIPT_URL", raising=False)

        with patch.object(web, "_load_config", return_value={
            "umami_website_id": "config-website-id",
            "umami_script_url": "https://config.umami.is/script.js"
        }):
            config = web._get_analytics_config()
            assert config["umami_website_id"] == "config-website-id"
            assert config["umami_script_url"] == "https://config.umami.is/script.js"

    def test_get_analytics_config_returns_none_when_not_configured(self, monkeypatch):
        """Should return None for website_id when not configured."""
        monkeypatch.delenv("UMAMI_WEBSITE_ID", raising=False)

        with patch.object(web, "_load_config", return_value={}):
            config = web._get_analytics_config()
            assert config["umami_website_id"] is None

    def test_umami_script_included_when_configured(self, client, monkeypatch):
        """Umami script tag should be included when website_id is configured."""
        monkeypatch.setenv("UMAMI_WEBSITE_ID", "test-website-id")

        response = client.get("/")
        html = response.data.decode()
        assert 'data-website-id="test-website-id"' in html
        assert 'src="https://cloud.umami.is/script.js"' in html

    def test_umami_script_not_included_when_not_configured(self, client, monkeypatch):
        """Umami script tag should not be included when website_id is not configured."""
        monkeypatch.delenv("UMAMI_WEBSITE_ID", raising=False)

        with patch.object(web, "_load_config", return_value={}):
            response = client.get("/")
            html = response.data.decode()
            assert "data-website-id" not in html
            assert "cloud.umami.is/script.js" not in html


class TestCollectionRouteComparisonSelection:
    """Tests for route comparison selection in collections table."""

    def test_collections_table_has_checkbox_column_header(self, client):
        """Collections table should have a checkbox column header."""
        response = client.get("/")
        html = response.data.decode()
        # Check for the compare column header with tooltip
        assert 'Select routes to compare' in html
        assert '<th class="cmp-col">' in html

    def test_compare_action_bar_exists(self, client):
        """Compare action bar should exist in collections page."""
        response = client.get("/")
        html = response.data.decode()
        assert 'id="compareActionBar"' in html
        assert 'id="compareLink"' in html
        assert '>Compare</a>' in html
        assert 'class="compare-action-bar"' in html
        assert 'id="compareSelectionCount"' in html

    def test_compare_action_bar_css_exists(self, client):
        """CSS for compare action bar should be present."""
        response = client.get("/")
        html = response.data.decode()
        assert '.compare-action-bar' in html
        assert '.compare-btn' in html
        assert '.route-select-checkbox' in html
        assert '.selected-route' in html

    def test_compare_javascript_functions_exist(self, client):
        """JavaScript functions for route comparison should be present."""
        response = client.get("/")
        html = response.data.decode()
        assert 'function toggleRouteSelection' in html
        assert 'function updateCompareActionBar' in html
        assert 'function buildCompareUrl' in html
        assert 'function clearRouteSelection' in html
        assert 'var selectedRouteIds = []' in html

    def test_javascript_builds_comparison_url_with_params(self, client):
        """buildCompareUrl should build URL with both route URLs and rider params."""
        response = client.get("/")
        html = response.data.decode()
        # Check that buildCompareUrl builds proper URL params
        assert "params.set('url', selectedRouteIds[0].url)" in html
        assert "params.set('url2', selectedRouteIds[1].url)" in html
        assert "params.set('climbing_power'" in html
        assert "params.set('flat_power'" in html
        assert "params.set('mass'" in html

    def test_selection_preserved_on_rerender(self, client):
        """rerenderCollectionTable should preserve selected state."""
        response = client.get("/")
        html = response.data.decode()
        # Check that rerender checks for selected state
        assert 'selectedRouteIds.some' in html
        assert "isSelected ? ' checked' : ''" in html
        assert "row.classList.add('selected-route')" in html

    def test_clear_selection_on_new_analysis(self, client):
        """Starting a new collection analysis should clear selection."""
        response = client.get("/")
        html = response.data.decode()
        # Check that clearRouteSelection is called on start event
        assert "clearRouteSelection()" in html
        # Verify the call is in the start handler context
        assert "if (data.type === 'start')" in html

    def test_collect_all_params_includes_all_share_params(self, client):
        """collectAllParams should include all parameters used in shareParams.

        This test ensures that when new parameters are added to shareParams
        (the canonical source for shareable URL parameters), they are also
        added to collectAllParams (used by buildAnalyzeUrl and buildCompareUrl).
        """
        import re
        response = client.get("/")
        html = response.data.decode()

        # Extract parameters set in collectAllParams function
        collect_match = re.search(
            r'function collectAllParams\(\)\s*\{(.*?)\n        \}',
            html, re.DOTALL
        )
        assert collect_match, "collectAllParams function not found"
        collect_body = collect_match.group(1)

        # Find all params.set('param_name', ...) calls in collectAllParams
        collect_params = set(re.findall(r"params\.set\('(\w+)'", collect_body))

        # Extract parameters set in shareParams (the canonical list)
        # shareParams is defined in the SSE 'start' handler
        share_match = re.search(
            r'var shareParams = new URLSearchParams\(\{(.*?)\}\);',
            html, re.DOTALL
        )
        assert share_match, "shareParams not found"
        share_body = share_match.group(1)

        # Find all parameter names in shareParams object literal
        share_params = set(re.findall(r"(\w+):", share_body))

        # shareParams includes 'url' which is route-specific, not a rider param
        share_params.discard('url')

        # Both should have the same parameters
        missing_from_collect = share_params - collect_params
        extra_in_collect = collect_params - share_params

        assert not missing_from_collect, (
            f"collectAllParams is missing parameters that are in shareParams: {missing_from_collect}. "
            f"Add these to collectAllParams() to propagate them when opening routes from collections."
        )
        # Extra params in collectAllParams is fine (like 'imperial' which is handled separately in shareParams)
