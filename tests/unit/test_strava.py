"""Unit tests for Strava integration."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from gpx_analyzer import strava
from gpx_analyzer.strava import (
    is_strava_route_url,
    is_strava_activity_url,
    is_strava_url,
    extract_strava_route_id,
    extract_strava_activity_id,
    TripPoint,
)


class TestUrlDetection:
    """Tests for Strava URL detection functions."""

    def test_is_strava_route_url_valid(self):
        assert is_strava_route_url("https://www.strava.com/routes/123456")
        assert is_strava_route_url("https://strava.com/routes/123456")
        assert is_strava_route_url("http://www.strava.com/routes/123456")
        assert is_strava_route_url("http://strava.com/routes/789")

    def test_is_strava_route_url_invalid(self):
        assert not is_strava_route_url("https://www.strava.com/activities/123456")
        assert not is_strava_route_url("https://ridewithgps.com/routes/123456")
        assert not is_strava_route_url("https://strava.com/segments/123456")
        assert not is_strava_route_url("not a url")

    def test_is_strava_activity_url_valid(self):
        assert is_strava_activity_url("https://www.strava.com/activities/123456")
        assert is_strava_activity_url("https://strava.com/activities/123456")
        assert is_strava_activity_url("http://www.strava.com/activities/123456")
        assert is_strava_activity_url("http://strava.com/activities/789")

    def test_is_strava_activity_url_invalid(self):
        assert not is_strava_activity_url("https://www.strava.com/routes/123456")
        assert not is_strava_activity_url("https://ridewithgps.com/trips/123456")
        assert not is_strava_activity_url("https://strava.com/segments/123456")
        assert not is_strava_activity_url("not a url")

    def test_is_strava_url(self):
        assert is_strava_url("https://www.strava.com/routes/123456")
        assert is_strava_url("https://www.strava.com/activities/123456")
        assert not is_strava_url("https://ridewithgps.com/routes/123456")


class TestIdExtraction:
    """Tests for Strava ID extraction functions."""

    def test_extract_strava_route_id(self):
        assert extract_strava_route_id("https://www.strava.com/routes/123456") == 123456
        assert extract_strava_route_id("https://strava.com/routes/789") == 789

    def test_extract_strava_route_id_invalid(self):
        with pytest.raises(ValueError):
            extract_strava_route_id("https://www.strava.com/activities/123456")
        with pytest.raises(ValueError):
            extract_strava_route_id("not a url")

    def test_extract_strava_activity_id(self):
        assert extract_strava_activity_id("https://www.strava.com/activities/123456") == 123456
        assert extract_strava_activity_id("https://strava.com/activities/789") == 789

    def test_extract_strava_activity_id_invalid(self):
        with pytest.raises(ValueError):
            extract_strava_activity_id("https://www.strava.com/routes/123456")
        with pytest.raises(ValueError):
            extract_strava_activity_id("not a url")


class TestAuthentication:
    """Tests for Strava authentication."""

    @patch.object(strava, "_load_config")
    def test_get_strava_credentials_from_config(self, mock_load_config):
        mock_load_config.return_value = {
            "strava_client_id": "test_id",
            "strava_client_secret": "test_secret",
            "strava_refresh_token": "test_token",
        }

        result = strava._get_strava_credentials()
        assert result == ("test_id", "test_secret", "test_token")

    @patch.object(strava, "_load_config")
    def test_get_strava_credentials_missing(self, mock_load_config):
        mock_load_config.return_value = {}
        assert strava._get_strava_credentials() is None

    @patch.object(strava, "_load_config")
    def test_get_strava_credentials_partial(self, mock_load_config):
        mock_load_config.return_value = {
            "strava_client_id": "test_id",
            # Missing secret and token
        }
        assert strava._get_strava_credentials() is None

    @patch("requests.post")
    @patch.object(strava, "_get_strava_credentials")
    def test_get_access_token_refresh(self, mock_creds, mock_post):
        mock_creds.return_value = ("client_id", "client_secret", "refresh_token")
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "access_token": "new_access_token",
                "expires_at": 9999999999,
            },
            headers={},
        )

        # Clear token cache
        strava._token_cache["access_token"] = None
        strava._token_cache["expires_at"] = 0

        token = strava._get_access_token()
        assert token == "new_access_token"

    @patch.object(strava, "_get_strava_credentials")
    def test_get_access_token_no_credentials(self, mock_creds):
        mock_creds.return_value = None

        # Clear token cache
        strava._token_cache["access_token"] = None
        strava._token_cache["expires_at"] = 0

        with pytest.raises(ValueError, match="Strava credentials not configured"):
            strava._get_access_token()


class TestActivityParsing:
    """Tests for parsing Strava activity data."""

    def test_parse_activity_streams_basic(self):
        data = {
            "metadata": {
                "name": "Morning Ride",
                "distance": 50000,
                "total_elevation_gain": 500,
                "moving_time": 7200,
                "elapsed_time": 8000,
                "average_speed": 6.94,
                "average_watts": 150,
                "start_date_local": "2024-01-15T10:30:00Z",
            },
            "streams": [
                {"type": "latlng", "data": [[37.7749, -122.4194], [37.7750, -122.4195]]},
                {"type": "altitude", "data": [100.0, 105.0]},
                {"type": "time", "data": [0, 60]},
                {"type": "distance", "data": [0, 100]},
                {"type": "velocity_smooth", "data": [5.0, 6.0]},
                {"type": "watts", "data": [140, 160]},
                {"type": "heartrate", "data": [120, 130]},
                {"type": "cadence", "data": [80, 85]},
            ],
        }

        points, metadata = strava._parse_activity_streams(data)

        assert len(points) == 2
        assert metadata["name"] == "Morning Ride"
        assert metadata["distance"] == 50000
        assert metadata["elevation_gain"] == 500
        assert metadata["moving_time"] == 7200
        assert metadata["avg_watts"] == 150
        assert metadata["source"] == "strava"

        # Check first point
        assert points[0].lat == 37.7749
        assert points[0].lon == -122.4194
        assert points[0].elevation == 100.0
        assert points[0].distance == 0
        assert points[0].speed == 5.0
        assert points[0].power == 140
        assert points[0].heart_rate == 120
        assert points[0].cadence == 80

    def test_parse_activity_streams_missing_optional_data(self):
        """Activity without power/HR data should still parse."""
        data = {
            "metadata": {
                "name": "Ride",
                "distance": 10000,
            },
            "streams": [
                {"type": "latlng", "data": [[37.7749, -122.4194]]},
                {"type": "distance", "data": [0]},
            ],
        }

        points, metadata = strava._parse_activity_streams(data)

        assert len(points) == 1
        assert points[0].power is None
        assert points[0].heart_rate is None
        assert points[0].cadence is None
        assert points[0].elevation is None


class TestTripPointDataclass:
    """Tests for TripPoint dataclass."""

    def test_trip_point_creation(self):
        point = TripPoint(
            lat=37.7749,
            lon=-122.4194,
            elevation=100.0,
            distance=500.0,
            speed=5.5,
            timestamp=1705323000.0,
            power=150.0,
            heart_rate=130,
            cadence=85,
        )

        assert point.lat == 37.7749
        assert point.lon == -122.4194
        assert point.elevation == 100.0
        assert point.distance == 500.0
        assert point.speed == 5.5
        assert point.power == 150.0
        assert point.heart_rate == 130
        assert point.cadence == 85

    def test_trip_point_with_none_values(self):
        point = TripPoint(
            lat=37.7749,
            lon=-122.4194,
            elevation=None,
            distance=500.0,
            speed=None,
            timestamp=None,
            power=None,
            heart_rate=None,
            cadence=None,
        )

        assert point.elevation is None
        assert point.speed is None
        assert point.power is None
