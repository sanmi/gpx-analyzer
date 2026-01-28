from datetime import datetime, timedelta, timezone

from gpx_analyzer.models import RideAnalysis, RiderParams, TrackPoint


class TestTrackPoint:
    def test_construction(self):
        pt = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None)
        assert pt.lat == 37.7749
        assert pt.lon == -122.4194
        assert pt.elevation == 10.0
        assert pt.time is None

    def test_with_time(self):
        t = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        pt = TrackPoint(lat=0.0, lon=0.0, elevation=None, time=t)
        assert pt.time == t
        assert pt.elevation is None


class TestRiderParams:
    def test_defaults(self):
        params = RiderParams()
        assert params.total_mass == 85.0
        assert params.cda == 0.35
        assert params.crr == 0.005
        assert params.air_density == 1.225
        assert params.assumed_avg_power == 150.0

    def test_custom_values(self):
        params = RiderParams(total_mass=75.0, cda=0.30, crr=0.004, air_density=1.1, assumed_avg_power=200.0)
        assert params.total_mass == 75.0
        assert params.cda == 0.30
        assert params.assumed_avg_power == 200.0


class TestRideAnalysis:
    def test_construction(self):
        analysis = RideAnalysis(
            total_distance=10000.0,
            elevation_gain=150.0,
            elevation_loss=100.0,
            duration=timedelta(hours=1),
            moving_time=timedelta(minutes=55),
            avg_speed=5.0,
            max_speed=10.0,
            estimated_work=300000.0,
            estimated_avg_power=150.0,
            estimated_moving_time_at_power=timedelta(seconds=2000),
        )
        assert analysis.total_distance == 10000.0
        assert analysis.elevation_gain == 150.0
        assert analysis.estimated_avg_power == 150.0
        assert analysis.estimated_moving_time_at_power == timedelta(seconds=2000)
