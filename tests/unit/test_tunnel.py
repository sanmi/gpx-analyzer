"""Tests for elevation anomaly detection and correction."""

import pytest

from gpx_analyzer.models import TrackPoint
from gpx_analyzer.tunnel import (
    AnomalyType,
    ElevationCorrection,
    TunnelCorrection,
    correct_elevation_anomalies,
    correct_tunnel_elevations,
    detect_and_correct_elevation_anomalies,
    detect_and_correct_tunnels,
    detect_dropouts,
    detect_elevation_anomalies,
    detect_spikes,
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


class TestDetectSpikes:
    """Tests for spike (Λ pattern) detection."""

    def test_no_tunnel_flat_route(self):
        """Flat route should have no tunnels."""
        elevations = [100.0] * 100
        points = make_track_points(elevations)
        tunnels = detect_spikes(points)
        assert tunnels == []

    def test_no_tunnel_gradual_climb(self):
        """Gradual climb should not be detected as tunnel."""
        # 5% grade over 1km = 50m gain
        elevations = [100.0 + i * 0.5 for i in range(100)]
        points = make_track_points(elevations)
        tunnels = detect_spikes(points)
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
        spikes = detect_spikes(
            points,
            min_spike_height=30.0,
            max_span_m=500.0,
            min_grade_pct=10.0,
        )
        assert len(spikes) == 1
        spike = spikes[0]
        assert spike.anomaly_type == AnomalyType.SPIKE
        assert spike.artificial_gain >= 50  # Should detect significant spike
        assert abs(spike.entry_elev - spike.exit_elev) < 30  # Returns to similar elevation

    def test_ignores_one_way_climb(self):
        """Should not detect a climb that doesn't return to original elevation."""
        # Steep climb that stays high
        elevations = (
            [100.0] * 20
            + [100.0 + i * 5 for i in range(1, 16)]  # steep climb
            + [175.0] * 30  # stays at top
        )
        points = make_track_points(elevations)
        tunnels = detect_spikes(points)
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
        tunnels = detect_spikes(points, max_span_m=500.0)
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
        tunnels = detect_spikes(points, min_spike_height=30.0)
        assert tunnels == []


class TestDetectDropouts:
    """Tests for dropout (V pattern) detection."""

    def test_no_dropout_flat_route(self):
        """Flat route should have no dropouts."""
        elevations = [100.0] * 100
        points = make_track_points(elevations)
        dropouts = detect_dropouts(points)
        assert dropouts == []

    def test_detects_v_pattern(self):
        """Should detect V-shaped elevation dropout."""
        # Create: flat -> steep down -> steep up -> flat
        elevations = (
            [100.0] * 20  # flat entry
            + [100.0 - i * 5 for i in range(1, 16)]  # steep drop -75m
            + [25.0 + i * 5 for i in range(1, 16)]  # steep climb back +75m
            + [100.0] * 20  # flat exit
        )
        points = make_track_points(elevations)
        dropouts = detect_dropouts(
            points,
            min_drop_depth=30.0,
            max_span_m=500.0,
            min_grade_pct=10.0,
        )
        assert len(dropouts) == 1
        dropout = dropouts[0]
        assert dropout.anomaly_type == AnomalyType.DROPOUT
        assert dropout.artificial_gain >= 50  # Should detect significant drop
        assert abs(dropout.entry_elev - dropout.exit_elev) < 30  # Returns to similar elevation

    def test_ignores_one_way_descent(self):
        """Should not detect a descent that doesn't return to original elevation."""
        # Steep descent that stays low
        elevations = (
            [100.0] * 20
            + [100.0 - i * 5 for i in range(1, 16)]  # steep descent
            + [25.0] * 30  # stays at bottom
        )
        points = make_track_points(elevations)
        dropouts = detect_dropouts(points)
        assert dropouts == []

    def test_respects_min_drop_depth(self):
        """Should not detect drops smaller than min_drop_depth."""
        # Small drop
        elevations = (
            [100.0] * 10
            + [100.0 - i * 2 for i in range(1, 11)]  # -20m
            + [80.0 + i * 2 for i in range(1, 11)]  # +20m
            + [100.0] * 10
        )
        points = make_track_points(elevations)
        dropouts = detect_dropouts(points, min_drop_depth=30.0)
        assert dropouts == []


class TestDetectOutlierSequences:
    """Tests for outlier sequence detection."""

    def test_detects_none_elevation(self):
        """Should detect single point with None elevation as dropout."""
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0001, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0002, lon=6.0, elevation=None, time=None),  # Missing!
            TrackPoint(lat=45.0003, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0004, lon=6.0, elevation=100.0, time=None),
        ]
        from gpx_analyzer.tunnel import detect_outlier_sequences
        outliers = detect_outlier_sequences(points, min_deviation=50.0)
        assert len(outliers) == 1
        assert outliers[0].anomaly_type == AnomalyType.DROPOUT
        # The algorithm finds the widest valid range
        assert 2 in range(outliers[0].start_idx, outliers[0].end_idx + 1)  # None point is included

    def test_detects_zero_elevation_dropout(self):
        """Should detect single point at 0m as dropout."""
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0001, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0002, lon=6.0, elevation=0.0, time=None),  # Dropout!
            TrackPoint(lat=45.0003, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0004, lon=6.0, elevation=300.0, time=None),
        ]
        from gpx_analyzer.tunnel import detect_outlier_sequences
        outliers = detect_outlier_sequences(points, min_deviation=50.0)
        assert len(outliers) == 1
        assert outliers[0].anomaly_type == AnomalyType.DROPOUT

    def test_detects_single_spike(self):
        """Should detect single point spike."""
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0001, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0002, lon=6.0, elevation=200.0, time=None),  # Spike!
            TrackPoint(lat=45.0003, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0004, lon=6.0, elevation=100.0, time=None),
        ]
        from gpx_analyzer.tunnel import detect_outlier_sequences
        outliers = detect_outlier_sequences(points, min_deviation=50.0)
        assert len(outliers) == 1
        assert outliers[0].anomaly_type == AnomalyType.SPIKE

    def test_detects_multi_point_dropout(self):
        """Should detect multi-point dropout sequence."""
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0001, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0002, lon=6.0, elevation=100.0, time=None),  # Drop
            TrackPoint(lat=45.0003, lon=6.0, elevation=50.0, time=None),   # Lower
            TrackPoint(lat=45.0004, lon=6.0, elevation=100.0, time=None),  # Rising
            TrackPoint(lat=45.0005, lon=6.0, elevation=300.0, time=None),
            TrackPoint(lat=45.0006, lon=6.0, elevation=300.0, time=None),
        ]
        from gpx_analyzer.tunnel import detect_outlier_sequences
        outliers = detect_outlier_sequences(points, min_deviation=50.0)
        assert len(outliers) == 1
        assert outliers[0].anomaly_type == AnomalyType.DROPOUT
        # Should capture the full dropout sequence
        assert outliers[0].end_idx - outliers[0].start_idx >= 3

    def test_ignores_small_deviation(self):
        """Should ignore deviations smaller than threshold."""
        points = [
            TrackPoint(lat=45.0, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0001, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0002, lon=6.0, elevation=120.0, time=None),  # Only 20m off
            TrackPoint(lat=45.0003, lon=6.0, elevation=100.0, time=None),
            TrackPoint(lat=45.0004, lon=6.0, elevation=100.0, time=None),
        ]
        from gpx_analyzer.tunnel import detect_outlier_sequences
        outliers = detect_outlier_sequences(points, min_deviation=50.0)
        assert len(outliers) == 0


class TestDetectElevationAnomalies:
    """Tests for combined anomaly detection."""

    def test_detects_both_spikes_and_dropouts(self):
        """Should detect both Λ and V patterns in same route."""
        # Create route with spike then dropout
        elevations = (
            [100.0] * 10  # flat
            # Spike
            + [100.0 + i * 5 for i in range(1, 11)]  # climb +50m
            + [150.0 - i * 5 for i in range(1, 11)]  # descent -50m
            + [100.0] * 20  # flat
            # Dropout
            + [100.0 - i * 5 for i in range(1, 11)]  # drop -50m
            + [50.0 + i * 5 for i in range(1, 11)]  # climb back +50m
            + [100.0] * 10  # flat
        )
        points = make_track_points(elevations)
        anomalies = detect_elevation_anomalies(
            points,
            min_spike_height=30.0,
            min_drop_depth=30.0,
            max_span_m=500.0,
            min_grade_pct=10.0,
        )
        assert len(anomalies) == 2
        types = {a.anomaly_type for a in anomalies}
        assert AnomalyType.SPIKE in types
        assert AnomalyType.DROPOUT in types

    def test_sorted_by_start_index(self):
        """Anomalies should be sorted by start index."""
        # Create route with dropout first, then spike
        elevations = (
            [100.0] * 10  # flat
            # Dropout first
            + [100.0 - i * 5 for i in range(1, 11)]  # drop
            + [50.0 + i * 5 for i in range(1, 11)]  # climb back
            + [100.0] * 20  # flat
            # Spike second
            + [100.0 + i * 5 for i in range(1, 11)]  # climb
            + [150.0 - i * 5 for i in range(1, 11)]  # descent
            + [100.0] * 10  # flat
        )
        points = make_track_points(elevations)
        anomalies = detect_elevation_anomalies(
            points,
            min_spike_height=30.0,
            min_drop_depth=30.0,
            max_span_m=500.0,
            min_grade_pct=10.0,
        )
        assert len(anomalies) == 2
        # Should be sorted by start_idx
        assert anomalies[0].start_idx < anomalies[1].start_idx


class TestCorrectElevationAnomalies:
    """Tests for elevation anomaly correction."""

    def test_no_correction_without_anomalies(self):
        """Should return same points when no anomalies."""
        elevations = [100.0] * 10
        points = make_track_points(elevations)
        corrected = correct_elevation_anomalies(points, [])
        assert len(corrected) == len(points)
        for orig, corr in zip(points, corrected):
            assert orig.elevation == corr.elevation

    def test_linear_interpolation_spike(self):
        """Should linearly interpolate through spike."""
        elevations = [100.0, 100.0, 150.0, 200.0, 150.0, 100.0, 100.0]
        points = make_track_points(elevations)

        anomaly = ElevationCorrection(
            start_idx=1,
            end_idx=5,
            start_km=0.01,
            end_km=0.05,
            peak_elev=200.0,
            trough_elev=100.0,
            entry_elev=100.0,
            exit_elev=100.0,
            artificial_gain=100.0,
            anomaly_type=AnomalyType.SPIKE,
        )

        corrected = correct_elevation_anomalies(points, [anomaly])

        # Entry and exit should be unchanged
        assert corrected[1].elevation == 100.0
        assert corrected[5].elevation == 100.0

        # Middle points should be interpolated (approximately 100)
        for i in range(2, 5):
            assert 95.0 <= corrected[i].elevation <= 105.0

    def test_linear_interpolation_dropout(self):
        """Should linearly interpolate through dropout."""
        elevations = [100.0, 100.0, 50.0, 0.0, 50.0, 100.0, 100.0]
        points = make_track_points(elevations)

        anomaly = ElevationCorrection(
            start_idx=1,
            end_idx=5,
            start_km=0.01,
            end_km=0.05,
            peak_elev=100.0,
            trough_elev=0.0,
            entry_elev=100.0,
            exit_elev=100.0,
            artificial_gain=100.0,
            anomaly_type=AnomalyType.DROPOUT,
        )

        corrected = correct_elevation_anomalies(points, [anomaly])

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

        anomaly = ElevationCorrection(
            start_idx=1,
            end_idx=3,
            start_km=0.01,
            end_km=0.03,
            peak_elev=200.0,
            trough_elev=100.0,
            entry_elev=100.0,
            exit_elev=100.0,
            artificial_gain=100.0,
            anomaly_type=AnomalyType.SPIKE,
        )

        corrected = correct_elevation_anomalies(points, [anomaly])

        for orig, corr in zip(points, corrected):
            assert orig.lat == corr.lat
            assert orig.lon == corr.lon
            assert orig.time == corr.time


class TestDetectAndCorrectElevationAnomalies:
    """Tests for combined detect and correct function."""

    def test_returns_tuple(self):
        """Should return (points, anomalies) tuple."""
        elevations = [100.0] * 10
        points = make_track_points(elevations)
        result = detect_and_correct_elevation_anomalies(points)
        assert isinstance(result, tuple)
        assert len(result) == 2
        corrected, anomalies = result
        assert isinstance(corrected, list)
        assert isinstance(anomalies, list)

    def test_end_to_end_spike(self):
        """Should detect and correct spike artifact."""
        # Create Λ pattern
        elevations = (
            [100.0] * 20
            + [100.0 + i * 5 for i in range(1, 16)]
            + [175.0 - i * 5 for i in range(1, 16)]
            + [100.0] * 20
        )
        points = make_track_points(elevations)

        corrected, anomalies = detect_and_correct_elevation_anomalies(
            points,
            min_spike_height=30.0,
            min_grade_pct=10.0,
        )

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.SPIKE
        # Corrected points should have lower max elevation
        orig_max = max(p.elevation for p in points)
        corr_max = max(p.elevation for p in corrected)
        assert corr_max < orig_max

    def test_end_to_end_dropout(self):
        """Should detect and correct dropout artifact."""
        # Create V pattern
        elevations = (
            [100.0] * 20
            + [100.0 - i * 5 for i in range(1, 16)]
            + [25.0 + i * 5 for i in range(1, 16)]
            + [100.0] * 20
        )
        points = make_track_points(elevations)

        corrected, anomalies = detect_and_correct_elevation_anomalies(
            points,
            min_drop_depth=30.0,
            min_grade_pct=10.0,
        )

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.DROPOUT
        # Corrected points should have higher min elevation
        orig_min = min(p.elevation for p in points)
        corr_min = min(p.elevation for p in corrected)
        assert corr_min > orig_min


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_tunnel_correction_alias(self):
        """TunnelCorrection should be an alias for ElevationCorrection."""
        assert TunnelCorrection is ElevationCorrection

    def test_detect_tunnels_alias(self):
        """detect_tunnels should be an alias for detect_spikes."""
        assert detect_tunnels is detect_spikes

    def test_correct_tunnel_elevations_alias(self):
        """correct_tunnel_elevations should be an alias for correct_elevation_anomalies."""
        assert correct_tunnel_elevations is correct_elevation_anomalies

    def test_detect_and_correct_tunnels_alias(self):
        """detect_and_correct_tunnels should be an alias for detect_and_correct_elevation_anomalies."""
        assert detect_and_correct_tunnels is detect_and_correct_elevation_anomalies
