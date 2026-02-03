"""Tests for tunnel detection and correction."""

import pytest

from gpx_analyzer.models import TrackPoint
from gpx_analyzer.tunnel import (
    TunnelCorrection,
    correct_tunnel_elevations,
    detect_and_correct_tunnels,
    detect_tunnels,
)


def make_track_points(elevations: list[float], spacing_m: float = 10.0) -> list[TrackPoint]:
    """Create track points with given elevations along a straight line."""
    # Create points along a north-south line
    base_lat = 45.0
    base_lon = 6.0
    # Approximate: 1 degree lat = 111km, so spacing_m / 111000 degrees per point
    lat_delta = spacing_m / 111000

    return [
        TrackPoint(
            lat=base_lat + i * lat_delta,
            lon=base_lon,
            elevation=elev,
            time=None,
        )
        for i, elev in enumerate(elevations)
    ]


class TestDetectTunnels:
    """Tests for tunnel detection."""

    def test_no_tunnel_flat_route(self):
        """Flat route should have no tunnels."""
        elevations = [100.0] * 100
        points = make_track_points(elevations)
        tunnels = detect_tunnels(points)
        assert tunnels == []

    def test_no_tunnel_gradual_climb(self):
        """Gradual climb should not be detected as tunnel."""
        # 5% grade over 1km = 50m gain
        elevations = [100.0 + i * 0.5 for i in range(100)]
        points = make_track_points(elevations)
        tunnels = detect_tunnels(points)
        assert tunnels == []

    def test_detects_lambda_pattern(self):
        """Should detect Λ-shaped elevation spike (tunnel artifact)."""
        # Create: flat -> steep up -> steep down -> flat
        elevations = (
            [100.0] * 20  # flat entry
            + [100.0 + i * 5 for i in range(1, 16)]  # steep climb +75m
            + [175.0 - i * 5 for i in range(1, 16)]  # steep descent -75m
            + [100.0] * 20  # flat exit
        )
        points = make_track_points(elevations)
        tunnels = detect_tunnels(
            points,
            min_spike_height=30.0,
            max_span_m=500.0,
            min_grade_pct=10.0,
        )
        assert len(tunnels) == 1
        tunnel = tunnels[0]
        assert tunnel.artificial_gain >= 50  # Should detect significant spike
        assert abs(tunnel.entry_elev - tunnel.exit_elev) < 30  # Returns to similar elevation

    def test_ignores_one_way_climb(self):
        """Should not detect a climb that doesn't return to original elevation."""
        # Steep climb that stays high
        elevations = (
            [100.0] * 20
            + [100.0 + i * 5 for i in range(1, 16)]  # steep climb
            + [175.0] * 30  # stays at top
        )
        points = make_track_points(elevations)
        tunnels = detect_tunnels(points)
        assert tunnels == []

    def test_ignores_long_span(self):
        """Should not detect artifacts longer than max_span_m."""
        # Very long gradual spike
        elevations = (
            [100.0] * 50
            + [100.0 + i * 2 for i in range(1, 51)]  # slow climb
            + [200.0 - i * 2 for i in range(1, 51)]  # slow descent
            + [100.0] * 50
        )
        points = make_track_points(elevations, spacing_m=20.0)  # Long route
        tunnels = detect_tunnels(points, max_span_m=500.0)
        assert tunnels == []

    def test_respects_min_spike_height(self):
        """Should not detect spikes smaller than min_spike_height."""
        # Small spike
        elevations = (
            [100.0] * 10
            + [100.0 + i * 2 for i in range(1, 11)]  # +20m
            + [120.0 - i * 2 for i in range(1, 11)]  # -20m
            + [100.0] * 10
        )
        points = make_track_points(elevations)
        tunnels = detect_tunnels(points, min_spike_height=30.0)
        assert tunnels == []


class TestCorrectTunnelElevations:
    """Tests for tunnel elevation correction."""

    def test_no_correction_without_tunnels(self):
        """Should return same points when no tunnels."""
        elevations = [100.0] * 10
        points = make_track_points(elevations)
        corrected = correct_tunnel_elevations(points, [])
        assert len(corrected) == len(points)
        for orig, corr in zip(points, corrected):
            assert orig.elevation == corr.elevation

    def test_linear_interpolation(self):
        """Should linearly interpolate through tunnel."""
        elevations = [100.0, 100.0, 150.0, 200.0, 150.0, 100.0, 100.0]
        points = make_track_points(elevations)

        tunnel = TunnelCorrection(
            start_idx=1,
            end_idx=5,
            start_km=0.01,
            end_km=0.05,
            peak_elev=200.0,
            entry_elev=100.0,
            exit_elev=100.0,
            artificial_gain=100.0,
        )

        corrected = correct_tunnel_elevations(points, [tunnel])

        # Entry and exit should be unchanged
        assert corrected[1].elevation == 100.0
        assert corrected[5].elevation == 100.0

        # Middle points should be interpolated (approximately 100)
        for i in range(2, 5):
            assert 95.0 <= corrected[i].elevation <= 105.0

    def test_preserves_coordinates(self):
        """Correction should preserve lat/lon/time."""
        elevations = [100.0, 100.0, 200.0, 100.0, 100.0]
        points = make_track_points(elevations)

        tunnel = TunnelCorrection(
            start_idx=1,
            end_idx=3,
            start_km=0.01,
            end_km=0.03,
            peak_elev=200.0,
            entry_elev=100.0,
            exit_elev=100.0,
            artificial_gain=100.0,
        )

        corrected = correct_tunnel_elevations(points, [tunnel])

        for orig, corr in zip(points, corrected):
            assert orig.lat == corr.lat
            assert orig.lon == corr.lon
            assert orig.time == corr.time


class TestDetectAndCorrectTunnels:
    """Tests for combined detect and correct function."""

    def test_returns_tuple(self):
        """Should return (points, tunnels) tuple."""
        elevations = [100.0] * 10
        points = make_track_points(elevations)
        result = detect_and_correct_tunnels(points)
        assert isinstance(result, tuple)
        assert len(result) == 2
        corrected, tunnels = result
        assert isinstance(corrected, list)
        assert isinstance(tunnels, list)

    def test_end_to_end(self):
        """Should detect and correct tunnel artifact."""
        # Create Λ pattern
        elevations = (
            [100.0] * 20
            + [100.0 + i * 5 for i in range(1, 16)]
            + [175.0 - i * 5 for i in range(1, 16)]
            + [100.0] * 20
        )
        points = make_track_points(elevations)

        corrected, tunnels = detect_and_correct_tunnels(
            points,
            min_spike_height=30.0,
            min_grade_pct=10.0,
        )

        assert len(tunnels) == 1
        # Corrected points should have lower max elevation
        orig_max = max(p.elevation for p in points)
        corr_max = max(p.elevation for p in corrected)
        assert corr_max < orig_max
