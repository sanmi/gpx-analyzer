from datetime import datetime, timedelta, timezone

from gpx_analyzer.analyzer import _analyze_segments
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
        _, raw_gain, _, _, _, _ = _analyze_segments(points, params)

        smoothed = smooth_elevations(points, radius_m=25.0)
        _, smooth_gain, _, _, _, _ = _analyze_segments(smoothed, params)

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
        # With only 2 points ~11m apart and 50m radius, both see both points
        # so both should average to 150.0
        assert result[0].elevation == result[1].elevation

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
