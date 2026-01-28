import math
from datetime import datetime, timedelta, timezone

import pytest

from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work, effective_power, estimate_speed_from_power


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


class TestEffectivePower:
    def test_flat_full_power(self, params):
        assert effective_power(0.0, params) == params.assumed_avg_power

    def test_uphill_full_power(self, params):
        assert effective_power(math.radians(5), params) == params.assumed_avg_power

    def test_at_threshold_zero_power(self, params):
        threshold_rad = math.radians(params.coasting_grade_threshold)
        assert effective_power(threshold_rad, params) == pytest.approx(0.0)

    def test_beyond_threshold_zero_power(self, params):
        beyond_rad = math.radians(params.coasting_grade_threshold - 3.0)
        assert effective_power(beyond_rad, params) == 0.0

    def test_mid_downhill_partial_power(self, params):
        # Halfway between 0 and threshold (-5°) is -2.5°
        mid_rad = math.radians(-2.5)
        power = effective_power(mid_rad, params)
        assert 0 < power < params.assumed_avg_power
        assert power == pytest.approx(params.assumed_avg_power * 0.5, rel=0.01)

    def test_gentle_downhill_mostly_full_power(self, params):
        # -1° on a -5° threshold → 80% power
        gentle_rad = math.radians(-1.0)
        power = effective_power(gentle_rad, params)
        assert power == pytest.approx(params.assumed_avg_power * 0.8, rel=0.01)


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

    def test_max_coasting_speed_cap(self):
        params = RiderParams()
        # Very steep descent should be capped at max_coasting_speed
        speed = estimate_speed_from_power(math.radians(-15), params)
        assert speed == pytest.approx(params.max_coasting_speed)

    def test_custom_max_coasting_speed(self):
        params = RiderParams(max_coasting_speed=10.0)
        speed = estimate_speed_from_power(math.radians(-15), params)
        assert speed == pytest.approx(10.0)

    def test_headwind_reduces_speed(self):
        """Headwind should reduce estimated speed."""
        no_wind = RiderParams(headwind=0.0)
        headwind = RiderParams(headwind=5.0)
        speed_no_wind = estimate_speed_from_power(0.0, no_wind)
        speed_headwind = estimate_speed_from_power(0.0, headwind)
        assert speed_headwind < speed_no_wind

    def test_tailwind_increases_speed(self):
        """Tailwind (negative headwind) should increase estimated speed."""
        no_wind = RiderParams(headwind=0.0)
        tailwind = RiderParams(headwind=-5.0)
        speed_no_wind = estimate_speed_from_power(0.0, no_wind)
        speed_tailwind = estimate_speed_from_power(0.0, tailwind)
        assert speed_tailwind > speed_no_wind


class TestHeadwindWork:
    def test_headwind_increases_work(self):
        """Headwind should increase work due to higher aero drag."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        no_wind = RiderParams(headwind=0.0)
        headwind = RiderParams(headwind=5.0)
        work_no_wind, _, _ = calculate_segment_work(a, b, no_wind)
        work_headwind, _, _ = calculate_segment_work(a, b, headwind)
        assert work_headwind > work_no_wind

    def test_tailwind_decreases_work(self):
        """Tailwind should decrease work due to lower aero drag."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        no_wind = RiderParams(headwind=0.0)
        tailwind = RiderParams(headwind=-3.0)
        work_no_wind, _, _ = calculate_segment_work(a, b, no_wind)
        work_tailwind, _, _ = calculate_segment_work(a, b, tailwind)
        assert work_tailwind < work_no_wind

    def test_zero_headwind_unchanged(self):
        """Zero headwind should give same result as default params."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        default_params = RiderParams()
        explicit_zero = RiderParams(headwind=0.0)
        work_default, _, _ = calculate_segment_work(a, b, default_params)
        work_explicit, _, _ = calculate_segment_work(a, b, explicit_zero)
        assert work_default == pytest.approx(work_explicit)
