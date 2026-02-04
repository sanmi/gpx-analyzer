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
        no_aero_params = RiderParams(cda=0.0, climbing_power=params.climbing_power, flat_power=params.flat_power)
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
    def test_flat_base_power(self, params):
        # On flat terrain, power should equal flat_power
        assert effective_power(0.0, params) == params.flat_power

    def test_steep_climb_increased_power(self, params):
        # Beyond climb threshold should have full climbing_power
        steep_rad = math.radians(params.climb_threshold_grade + 2.0)
        assert effective_power(steep_rad, params) == pytest.approx(params.climbing_power)

    def test_moderate_climb_partial_increase(self, params):
        # Halfway to climb threshold should have partial increase
        # Default climb_threshold_grade is 4°, so 2° is halfway
        mid_rad = math.radians(2.0)
        power = effective_power(mid_rad, params)
        # Should be between flat and max climb power
        assert power > params.flat_power
        assert power < params.climbing_power
        # At halfway, should be: flat_power + 0.5 * (climbing_power - flat_power)
        expected = params.flat_power + 0.5 * (params.climbing_power - params.flat_power)
        assert power == pytest.approx(expected, rel=0.01)

    def test_at_coasting_threshold_zero_power(self, params):
        threshold_rad = math.radians(params.coasting_grade_threshold)
        assert effective_power(threshold_rad, params) == pytest.approx(0.0)

    def test_beyond_coasting_threshold_zero_power(self, params):
        beyond_rad = math.radians(params.coasting_grade_threshold - 3.0)
        assert effective_power(beyond_rad, params) == 0.0

    def test_mid_downhill_uses_descending_power(self, params):
        # Any descent between 0 and threshold uses descending_power
        mid_rad = math.radians(-2.5)
        power = effective_power(mid_rad, params)
        assert power == pytest.approx(params.descending_power, rel=0.01)

    def test_gentle_downhill_uses_descending_power(self, params):
        # Gentle descent also uses descending_power
        gentle_rad = math.radians(-1.0)
        power = effective_power(gentle_rad, params)
        assert power == pytest.approx(params.descending_power, rel=0.01)

    def test_custom_climbing_power(self):
        params = RiderParams(
            climbing_power=200.0,
            flat_power=100.0,
            climb_threshold_grade=5.0,
        )
        steep_rad = math.radians(6.0)  # Beyond threshold
        assert effective_power(steep_rad, params) == pytest.approx(200.0)

    def test_custom_flat_power(self):
        params = RiderParams(
            climbing_power=150.0,
            flat_power=90.0,
        )
        # On flat terrain, power should be flat_power
        assert effective_power(0.0, params) == pytest.approx(90.0)

    def test_flat_power_ramp_to_climb(self):
        """Power should ramp from flat_power to climbing_power on climbs."""
        params = RiderParams(
            flat_power=80.0,
            climbing_power=150.0,
            climb_threshold_grade=4.0,
        )
        # At 2° (halfway to threshold), should interpolate
        mid_rad = math.radians(2.0)
        power = effective_power(mid_rad, params)
        # Expected: 80 + 0.5 * (150 - 80) = 80 + 35 = 115
        expected = 115.0
        assert power == pytest.approx(expected, rel=0.01)


class TestEstimateSpeedFromPower:
    def test_flat_returns_positive_speed(self, params):
        speed = estimate_speed_from_power(0.0, params)
        assert speed > 0

    def test_uphill_slower_than_flat(self, params):
        flat_speed = estimate_speed_from_power(0.0, params)
        uphill_speed = estimate_speed_from_power(math.radians(5), params)
        assert uphill_speed < flat_speed

    def test_gentle_downhill_faster_than_flat(self, params):
        flat_speed = estimate_speed_from_power(0.0, params)
        # Gentle descent (-1°) should be faster than flat
        downhill_speed = estimate_speed_from_power(math.radians(-1), params)
        assert downhill_speed > flat_speed

    def test_higher_power_faster(self):
        low = RiderParams(climbing_power=100.0, flat_power=80.0)
        high = RiderParams(climbing_power=250.0, flat_power=200.0)
        assert estimate_speed_from_power(0.0, high) > estimate_speed_from_power(0.0, low)

    def test_zero_power_downhill_coasts(self):
        params = RiderParams(climbing_power=0.0, flat_power=0.0)
        speed = estimate_speed_from_power(math.radians(-5), params)
        assert speed > 0, "Should coast downhill even with zero power"

    def test_steep_descent_speed_cap(self):
        """Very steep descents should be capped at steep_descent_speed."""
        params = RiderParams(steep_descent_speed=5.0, steep_descent_grade=-8.0)
        # Very steep descent (-15°) should be capped at steep_descent_speed
        speed = estimate_speed_from_power(math.radians(-15), params)
        assert speed == pytest.approx(params.steep_descent_speed)

    def test_custom_steep_descent_speed(self):
        """Custom steep descent speed should be used."""
        params = RiderParams(steep_descent_speed=6.0, steep_descent_grade=-10.0)
        speed = estimate_speed_from_power(math.radians(-15), params)
        assert speed == pytest.approx(6.0)

    def test_descent_speed_factor_scales_descent(self):
        """descent_speed_factor should scale descent speeds."""
        params_full = RiderParams(descent_speed_factor=1.0)
        params_half = RiderParams(descent_speed_factor=0.5)
        # On descent, factor should scale speed
        descent_full = estimate_speed_from_power(math.radians(-5), params_full)
        descent_half = estimate_speed_from_power(math.radians(-5), params_half)
        assert descent_half == pytest.approx(descent_full * 0.5, rel=0.01)

    def test_descent_speed_factor_no_effect_on_flat(self):
        """descent_speed_factor should not affect flat or uphill speeds."""
        params_full = RiderParams(descent_speed_factor=1.0)
        params_half = RiderParams(descent_speed_factor=0.5)
        # On flat, factor should have no effect
        flat_full = estimate_speed_from_power(0.0, params_full)
        flat_half = estimate_speed_from_power(0.0, params_half)
        assert flat_half == pytest.approx(flat_full)
        # On uphill, factor should have no effect
        uphill_full = estimate_speed_from_power(math.radians(5), params_full)
        uphill_half = estimate_speed_from_power(math.radians(5), params_half)
        assert uphill_half == pytest.approx(uphill_full)

    def test_gradient_limited_speed_unpaved(self):
        """Unpaved surfaces should have proportionally lower descent speed."""
        params = RiderParams(
            max_coasting_speed=15.0,
            max_coasting_speed_unpaved=10.0,
            steep_descent_speed=6.0,
            steep_descent_grade=-10.0,
        )
        # Very steep descent on paved should cap at steep_descent_speed
        speed_paved = estimate_speed_from_power(math.radians(-15), params, unpaved=False)
        assert speed_paved == pytest.approx(6.0)
        # Unpaved should be proportionally lower (10/15 ratio)
        speed_unpaved = estimate_speed_from_power(math.radians(-15), params, unpaved=True)
        expected_unpaved = 6.0 * (10.0 / 15.0)
        assert speed_unpaved == pytest.approx(expected_unpaved)

    def test_unpaved_coasting_speed_slower(self):
        """On descents, unpaved should result in slower coasting speed."""
        params = RiderParams()
        speed_paved = estimate_speed_from_power(math.radians(-10), params, unpaved=False)
        speed_unpaved = estimate_speed_from_power(math.radians(-10), params, unpaved=True)
        assert speed_unpaved < speed_paved

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


class TestSurfaceCrr:
    """Tests for per-segment rolling resistance coefficient (crr)."""

    def test_segment_crr_used_in_work_calculation(self, params):
        """When point has crr, it should be used instead of params.crr."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        # Standard crr (params.crr = 0.005)
        a_std = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b_std = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        # High crr segment (gravel)
        a_gravel = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b_gravel = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
            crr=0.010,
        )
        work_std, _, _ = calculate_segment_work(a_std, b_std, params)
        work_gravel, _, _ = calculate_segment_work(a_gravel, b_gravel, params)
        assert work_gravel > work_std

    def test_none_crr_uses_params_default(self, params):
        """When point.crr is None, params.crr should be used."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base, crr=None)
        b = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
            crr=None,
        )
        work, _, _ = calculate_segment_work(a, b, params)
        # Should be same as without crr field
        a2 = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b2 = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
        )
        work2, _, _ = calculate_segment_work(a2, b2, params)
        assert work == pytest.approx(work2)

    def test_lower_crr_less_work(self, params):
        """Lower crr (quality pavement) should result in less work."""
        base = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=base)
        b_quality = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
            crr=0.004,
        )
        b_standard = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=base + timedelta(seconds=20),
            crr=0.005,
        )
        work_quality, _, _ = calculate_segment_work(a, b_quality, params)
        work_standard, _, _ = calculate_segment_work(a, b_standard, params)
        assert work_quality < work_standard

    def test_crr_used_in_speed_estimation(self):
        """Per-segment crr should affect speed estimation when time is missing."""
        params = RiderParams()
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=10.0, time=None)
        b_paved = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=None,
            crr=0.005,
        )
        b_gravel = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=10.0,
            time=None,
            crr=0.010,
        )
        _, _, elapsed_paved = calculate_segment_work(a, b_paved, params)
        _, _, elapsed_gravel = calculate_segment_work(a, b_gravel, params)
        # Higher crr means slower speed, so longer elapsed time
        assert elapsed_gravel > elapsed_paved

    def test_unpaved_uses_lower_max_coasting_speed(self):
        """Unpaved segments should use max_coasting_speed_unpaved on descents."""
        params = RiderParams(max_coasting_speed=15.0, max_coasting_speed_unpaved=10.0)
        # Steep descent without time data - speed should be estimated and capped
        a = TrackPoint(lat=37.7749, lon=-122.4194, elevation=100.0, time=None)
        b_paved = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=0.0,
            time=None,
            unpaved=False,
        )
        b_unpaved = TrackPoint(
            lat=37.7758,
            lon=-122.4183,
            elevation=0.0,
            time=None,
            unpaved=True,
        )
        _, dist_paved, elapsed_paved = calculate_segment_work(a, b_paved, params)
        _, dist_unpaved, elapsed_unpaved = calculate_segment_work(a, b_unpaved, params)
        # Same distance, but unpaved should have longer elapsed time (slower speed)
        assert dist_paved == pytest.approx(dist_unpaved)
        assert elapsed_unpaved > elapsed_paved


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
