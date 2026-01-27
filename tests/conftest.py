import os
from datetime import datetime, timedelta, timezone

import pytest

from gpx_analyzer.models import RiderParams, TrackPoint

SAMPLE_GPX_PATH = os.path.join(
    os.path.dirname(__file__), "functional", "data", "sample_ride.gpx"
)


@pytest.fixture
def rider_params():
    return RiderParams()


@pytest.fixture
def simple_track_points():
    """A short list of track points for unit testing: flat, ~100m apart."""
    base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    return [
        TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
        TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base_time + timedelta(seconds=20),
        ),
        TrackPoint(
            lat=37.7767,
            lon=-122.4172,
            elevation=10.0,
            time=base_time + timedelta(seconds=40),
        ),
    ]


@pytest.fixture
def uphill_track_points():
    """Track points going uphill."""
    base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    return [
        TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base_time),
        TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=20.0,
            time=base_time + timedelta(seconds=30),
        ),
        TrackPoint(
            lat=37.7767,
            lon=-122.4172,
            elevation=35.0,
            time=base_time + timedelta(seconds=60),
        ),
    ]


@pytest.fixture
def downhill_track_points():
    """Track points going downhill."""
    base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    return [
        TrackPoint(lat=37.7749, lon=-122.4194, elevation=50.0, time=base_time),
        TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=30.0,
            time=base_time + timedelta(seconds=15),
        ),
        TrackPoint(
            lat=37.7767,
            lon=-122.4172,
            elevation=10.0,
            time=base_time + timedelta(seconds=30),
        ),
    ]
