import math
from datetime import datetime, timedelta, timezone

import pytest

from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work, estimate_speed_from_power


@pytest.fixture
def params():
    return RiderParams()


class TestCalculateSegmentWork:
    def test_flat_segment(self, params):
        """Flat segment should have rolling resistance + aero drag work."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        work, dist, elapsed = calculate_segment_work(a, b, params)
        assert work > 0
        assert dist > 0
        assert elapsed == 20.0

    def test_uphill_segment(self, params):
        """Uphill segment should produce more work than flat."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        flat_a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        flat_b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        up_a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        up_b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=30.0,
            time=base + timedelta(seconds=20),
        )
        work_flat, _, _ = calculate_segment_work(flat_a, flat_b, params)
        work_up, _, _ = calculate_segment_work(up_a, up_b, params)
        assert work_up > work_flat

    def test_downhill_segment_zero_work(self, params):
        """Steep downhill should clamp to zero work (coasting)."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=100.0, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=15),
        )
        work, dist, elapsed = calculate_segment_work(a, b, params)
        assert work == 0.0

    def test_zero_distance(self, params):
        """Same point should produce zero work."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b = TrackPoint(
            lat=37.7749,
            lon=-122.4194,
            elevation=10.0,
            time=base + timedelta(seconds=10),
        )
        work, dist, elapsed = calculate_segment_work(a, b, params)
        assert work == 0.0
        assert dist < 0.1

    def test_missing_time_estimates_speed(self, params):
        """Missing time should estimate speed from assumed power."""
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None)
        b = TrackPoint(lat=37.7758, lon=-122.4183, elevation=15.0, time=None)
        work, dist, elapsed = calculate_segment_work(a, b, params)
        assert elapsed > 0, "Should estimate elapsed time from power"
        assert work > 0

    def test_missing_time_includes_aero(self, params):
        """Missing time with speed estimation should include aero drag."""
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None)
        b = TrackPoint(lat=37.7758, lon=-122.4183, elevation=10.0, time=None)
        work_no_time, _, _ = calculate_segment_work(a, b, params)

        # Compare with zero-aero case (params with zero CdA)
        no_aero_params = RiderParams(cda=0.0, assumed_avg_power=params.assumed_avg_power)
        work_no_aero, _, _ = calculate_segment_work(a, b, no_aero_params)
        assert work_no_time > work_no_aero

    def test_missing_elevation_uses_zero(self, params):
        """Missing elevation defaults to 0."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=None, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=None,
            time=base + timedelta(seconds=20),
        )
        work, dist, elapsed = calculate_segment_work(a, b, params)
        # Flat at elevation 0, should be rolling + aero
        assert work > 0


class TestEstimateSpeedFromPower:
    def test_flat_returns_positive_speed(self, params):
        speed = estimate_speed_from_power(0.0, params)
        assert speed > 0

    def test_uphill_slower_than_flat(self, params):
        flat_speed = estimate_speed_from_power(0.0, params)
        uphill_speed = estimate_speed_from_power(math.radians(5), params)
        assert uphill_speed < flat_speed

    def test_downhill_faster_than_flat(self, params):
        flat_speed = estimate_speed_from_power(0.0, params)
        downhill_speed = estimate_speed_from_power(math.radians(-5), params)
        assert downhill_speed > flat_speed

    def test_higher_power_faster(self):
        low = RiderParams(assumed_avg_power=100.0)
        high = RiderParams(assumed_avg_power=250.0)
        assert estimate_speed_from_power(0.0, high) > estimate_speed_from_power(0.0, low)

    def test_zero_power_downhill_coasts(self):
        params = RiderParams(assumed_avg_power=0.0)
        speed = estimate_speed_from_power(math.radians(-5), params)
        assert speed > 0, "Should coast downhill even with zero power"
