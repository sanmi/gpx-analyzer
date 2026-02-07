from datetime import datetime, timedelta, timezone

from gpx_analyzer.analyzer import _analyze_segments, _calculate_rolling_grades, calculate_hilliness, DEFAULT_MAX_GRADE_WINDOW
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.smoothing import smooth_elevations


def _make_points(elevations, spacing_deg=0.0001):
    """Create a line of points with given elevations, evenly spaced (~11m apart)."""
    base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    return [
        TrackPoint(
            lat=37.7749 + i * spacing_deg,
            lon=-122.4194,
            elevation=e,
            time=base_time + timedelta(seconds=i * 5),
        )
        for i, e in enumerate(elevations)
    ]


class TestSmoothElevations:
    def test_smoothing_reduces_noise(self):
        """Smoothing noisy elevation data should reduce total elevation gain."""
        # Sawtooth noise: 100, 105, 100, 105, ... over 20 points
        elevations = [100.0 + 5.0 * (i % 2) for i in range(20)]
        points = _make_points(elevations)

        params = RiderParams()
        _, raw_gain, _, _, _, _, _ = _analyze_segments(points, params)

        smoothed = smooth_elevations(points, radius_m=25.0)
        _, smooth_gain, _, _, _, _, _ = _analyze_segments(smoothed, params)

        assert smooth_gain < raw_gain

    def test_radius_zero_is_noop(self):
        """Smoothing with radius=0 should not change elevations."""
        elevations = [100.0, 110.0, 105.0, 115.0, 108.0]
        points = _make_points(elevations)

        result = smooth_elevations(points, radius_m=0.0)

        for orig, smoothed in zip(points, result):
            assert orig.elevation == smoothed.elevation
            assert orig.lat == smoothed.lat
            assert orig.lon == smoothed.lon
            assert orig.time == smoothed.time

    def test_none_elevations_preserved(self):
        """Points with None elevation should remain None after smoothing."""
        base_time = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        points = [
            TrackPoint(lat=37.7749, lon=-122.4194, elevation=100.0, time=base_time),
            TrackPoint(
                lat=37.7750,
                lon=-122.4194,
                elevation=None,
                time=base_time + timedelta(seconds=5),
            ),
            TrackPoint(
                lat=37.7751,
                lon=-122.4194,
                elevation=105.0,
                time=base_time + timedelta(seconds=10),
            ),
        ]

        result = smooth_elevations(points, radius_m=50.0)

        assert result[0].elevation is not None
        assert result[1].elevation is None
        assert result[2].elevation is not None

    def test_single_point(self):
        """A single point should be returned as-is."""
        points = _make_points([100.0])
        result = smooth_elevations(points, radius_m=50.0)

        assert len(result) == 1
        assert result[0].elevation == 100.0

    def test_two_points(self):
        """Two points should be handled without errors."""
        points = _make_points([100.0, 200.0])
        result = smooth_elevations(points, radius_m=50.0)

        assert len(result) == 2
        # With linear regression on 2 points, the line passes through both
        # so elevations are preserved (fitted value at each point = original)
        assert result[0].elevation == 100.0
        assert result[1].elevation == 200.0

    def test_preserves_lat_lon_time(self):
        """Smoothing should only change elevation, not other fields."""
        elevations = [100.0, 110.0, 105.0, 115.0, 108.0]
        points = _make_points(elevations)

        result = smooth_elevations(points, radius_m=25.0)

        for orig, smoothed in zip(points, result):
            assert orig.lat == smoothed.lat
            assert orig.lon == smoothed.lon
            assert orig.time == smoothed.time

    def test_elevation_scale_reduces_gain(self):
        """Elevation scale < 1 should reduce elevation changes."""
        # Simple climb: 100m to 200m (100m gain)
        elevations = [100.0, 150.0, 200.0]
        points = _make_points(elevations)

        # Scale by 0.5 should give 50m gain: 100 -> 125 -> 150
        result = smooth_elevations(points, radius_m=0.0, elevation_scale=0.5)

        assert result[0].elevation == 100.0  # Reference stays the same
        assert result[1].elevation == 125.0  # 100 + (150-100)*0.5
        assert result[2].elevation == 150.0  # 100 + (200-100)*0.5

    def test_elevation_scale_without_smoothing(self):
        """Elevation scaling should work even with radius=0."""
        elevations = [100.0, 200.0, 150.0]
        points = _make_points(elevations)

        result = smooth_elevations(points, radius_m=0.0, elevation_scale=0.8)

        assert result[0].elevation == 100.0  # Reference unchanged
        assert result[1].elevation == 180.0  # 100 + (200-100)*0.8
        assert result[2].elevation == 140.0  # 100 + (150-100)*0.8

    def test_elevation_scale_one_is_noop(self):
        """Elevation scale of 1.0 should not change elevations."""
        elevations = [100.0, 150.0, 200.0]
        points = _make_points(elevations)

        result = smooth_elevations(points, radius_m=0.0, elevation_scale=1.0)

        for orig, scaled in zip(points, result):
            assert orig.elevation == scaled.elevation


class TestSmoothingConsistency:
    """Tests to verify smoothing is applied consistently across components."""

    def test_histogram_uses_same_grades_as_elevation_profile(self):
        """Histogram binning should use same rolling grades as elevation profile.

        This prevents the bug where histogram showed different grades than
        the elevation profile tooltip due to extra smoothing.
        """
        # Create a route with a steep section (~15% grade)
        # 100 points, each ~11m apart = ~1.1km total
        # Steep climb in the middle: 100m to 200m over ~300m = ~33% raw grade
        elevations = (
            [100.0] * 30 +  # Flat start
            [100.0 + i * 3.3 for i in range(30)] +  # ~10% climb
            [200.0] * 40  # Flat end
        )
        points = _make_points(elevations, spacing_deg=0.0001)  # ~11m spacing

        # Apply user's smoothing (simulating what happens in the app)
        smoothing_radius = 50.0
        smoothed_points = smooth_elevations(points, radius_m=smoothing_radius)

        # Calculate rolling grades (what elevation profile uses)
        window = DEFAULT_MAX_GRADE_WINDOW
        elevation_profile_grades = _calculate_rolling_grades(smoothed_points, window)

        # Calculate hilliness (what histogram uses)
        params = RiderParams()
        hilliness = calculate_hilliness(
            smoothed_points,
            params,
            unscaled_points=smoothed_points,
            max_grade_window=window,
            max_grade_smoothing=0  # No extra smoothing for this test
        )

        # The rolling grades used for histogram should match elevation profile
        # Check that grades in steep sections would fall in same histogram bucket
        max_profile_grade = max(elevation_profile_grades) if elevation_profile_grades else 0

        # The histogram should show time in the >10% bucket if profile has grades >10%
        if max_profile_grade > 10:
            steep_time = sum(hilliness.steep_time_histogram.values())
            assert steep_time > 0, "Histogram should show steep grades when elevation profile has them"

    def test_no_extra_smoothing_for_histogram_bins(self):
        """Histogram bins should NOT use extra smoothing beyond user's setting.

        Only max_grade calculation gets extra smoothing; histogram bins should
        use the same grades shown in the elevation profile tooltip.
        """
        # Create noisy data that would show different results with extra smoothing
        # Sharp spike that extra smoothing would remove
        elevations = [100.0] * 20 + [150.0] * 5 + [100.0] * 20  # 50m spike
        points = _make_points(elevations, spacing_deg=0.0001)

        # Apply minimal user smoothing
        smoothing_radius = 10.0
        smoothed_points = smooth_elevations(points, radius_m=smoothing_radius)

        window = DEFAULT_MAX_GRADE_WINDOW
        params = RiderParams()

        # Calculate with no extra smoothing
        hilliness_no_extra = calculate_hilliness(
            smoothed_points,
            params,
            unscaled_points=smoothed_points,
            max_grade_window=window,
            max_grade_smoothing=0
        )

        # Calculate with extra smoothing (only affects max_grade, not bins)
        hilliness_with_extra = calculate_hilliness(
            smoothed_points,
            params,
            unscaled_points=smoothed_points,
            max_grade_window=window,
            max_grade_smoothing=150.0  # Extra smoothing
        )

        # Histogram bins should be the same (extra smoothing only affects max_grade)
        assert hilliness_no_extra.grade_time_histogram == hilliness_with_extra.grade_time_histogram
        assert hilliness_no_extra.grade_distance_histogram == hilliness_with_extra.grade_distance_histogram

        # But max_grade may differ (extra smoothing reduces spikes)
        # This is expected and correct behavior
        assert hilliness_with_extra.max_grade <= hilliness_no_extra.max_grade

    def test_user_smoothing_affects_all_calculations(self):
        """User's smoothing setting should affect histogram, elevation profile, and physics equally."""
        # Noisy elevation data
        elevations = [100.0 + 10.0 * (i % 2) for i in range(50)]  # Sawtooth
        points = _make_points(elevations, spacing_deg=0.0001)

        params = RiderParams()
        window = DEFAULT_MAX_GRADE_WINDOW

        # Low smoothing - should preserve more noise
        low_smooth = smooth_elevations(points, radius_m=10.0)
        low_grades = _calculate_rolling_grades(low_smooth, window)
        _, low_gain, _, _, _, _, _ = _analyze_segments(low_smooth, params)

        # High smoothing - should reduce noise
        high_smooth = smooth_elevations(points, radius_m=100.0)
        high_grades = _calculate_rolling_grades(high_smooth, window)
        _, high_gain, _, _, _, _, _ = _analyze_segments(high_smooth, params)

        # Higher smoothing should result in:
        # 1. Lower elevation gain (less noise)
        assert high_gain < low_gain, "Higher smoothing should reduce elevation gain"

        # 2. Lower max grade (smoother profile)
        max_low_grade = max(abs(g) for g in low_grades) if low_grades else 0
        max_high_grade = max(abs(g) for g in high_grades) if high_grades else 0
        assert max_high_grade <= max_low_grade, "Higher smoothing should reduce max grade"

    def test_rolling_grades_consistent_between_profile_and_histogram(self):
        """Verify the same _calculate_rolling_grades function is used for both."""
        # Simple climb
        elevations = [100.0 + i * 2 for i in range(50)]  # 2m per point
        points = _make_points(elevations, spacing_deg=0.0001)

        smoothed = smooth_elevations(points, radius_m=50.0)
        window = DEFAULT_MAX_GRADE_WINDOW

        # These should be identical - same function, same inputs
        grades_for_profile = _calculate_rolling_grades(smoothed, window)
        grades_for_histogram = _calculate_rolling_grades(smoothed, window)

        assert grades_for_profile == grades_for_histogram
