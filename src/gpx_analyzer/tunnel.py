"""Elevation anomaly detection and correction.

Detects and corrects two types of elevation artifacts:

1. Spikes (Λ pattern): DEM artifacts from tunnels/bridges where elevation shows
   the surface above instead of the actual path. Creates artificial climbs.

2. Dropouts (V pattern): GPS/sensor errors where elevation drops to zero or
   unrealistically low values, then returns to normal. Creates artificial descents.

Both are corrected by linear interpolation between entry and exit elevations.
"""

from dataclasses import dataclass
from enum import Enum

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import TrackPoint


class AnomalyType(Enum):
    """Type of elevation anomaly."""
    SPIKE = "spike"      # Λ pattern - elevation spikes up (tunnel/bridge DEM artifact)
    DROPOUT = "dropout"  # V pattern - elevation drops down (GPS/sensor error)


@dataclass
class ElevationCorrection:
    """A detected and corrected elevation anomaly."""
    start_idx: int
    end_idx: int
    start_km: float
    end_km: float
    peak_elev: float      # For spikes: max elevation; for dropouts: entry elevation
    trough_elev: float    # For spikes: entry elevation; for dropouts: min elevation
    entry_elev: float
    exit_elev: float
    artificial_gain: float  # Absolute elevation change removed (always positive)
    anomaly_type: AnomalyType


# Backwards compatibility alias
TunnelCorrection = ElevationCorrection


def _calculate_grades(
    points: list[TrackPoint],
    cum_dist: list[float],
    window_m: float,
) -> list[tuple[int, float]]:
    """Calculate rolling grades over a window.

    Returns list of (index, grade_pct) tuples.
    """
    grades = []
    for i in range(len(points)):
        # Find points window_m ahead
        target = cum_dist[i] + window_m
        j = i
        while j < len(points) - 1 and cum_dist[j] < target:
            j += 1

        dist = cum_dist[j] - cum_dist[i]
        if dist > 0 and points[i].elevation is not None and points[j].elevation is not None:
            grade = (points[j].elevation - points[i].elevation) / dist * 100
            grades.append((i, grade))
        else:
            grades.append((i, 0))
    return grades


def _calculate_cumulative_distance(points: list[TrackPoint]) -> list[float]:
    """Calculate cumulative distance along track."""
    cum_dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon
        )
        cum_dist.append(cum_dist[-1] + d)
    return cum_dist


def detect_spikes(
    points: list[TrackPoint],
    min_spike_height: float = 40.0,
    max_span_m: float = 800.0,
    max_elev_return: float = 30.0,
    window_m: float = 100.0,
    min_grade_pct: float = 15.0,
    cum_dist: list[float] | None = None,
    grades: list[tuple[int, float]] | None = None,
) -> list[ElevationCorrection]:
    """Detect elevation spike artifacts (Λ pattern).

    Spikes create "Λ" (lambda) shaped artifacts in DEM data:
    - Sharp elevation increase (GPS track "enters" the mountain surface)
    - Sharp elevation decrease (GPS track "exits" the mountain surface)
    - Returns to near-original elevation (actual tunnel is nearly flat)

    Args:
        points: Track points with elevation data
        min_spike_height: Minimum elevation spike to consider (meters)
        max_span_m: Maximum horizontal distance for anomaly (meters)
        max_elev_return: How close exit elevation must be to entry (meters)
        window_m: Window size for grade calculation (meters)
        min_grade_pct: Minimum grade to trigger detection (percent)
        cum_dist: Pre-calculated cumulative distances (optional)
        grades: Pre-calculated grades (optional)

    Returns:
        List of detected spike corrections
    """
    if len(points) < 10:
        return []

    if cum_dist is None:
        cum_dist = _calculate_cumulative_distance(points)
    if grades is None:
        grades = _calculate_grades(points, cum_dist, window_m)

    spikes = []
    i = 0
    while i < len(grades) - 1:
        idx, grade = grades[i]

        # Look for steep uphill (start of Λ pattern)
        if grade > min_grade_pct:
            entry_idx = idx
            entry_elev = points[entry_idx].elevation
            if entry_elev is None:
                i += 1
                continue
            entry_dist = cum_dist[entry_idx]

            # Find the peak (where steep up transitions to steep down)
            peak_idx = idx
            peak_elev = entry_elev
            j = i + 1
            found_spike = False

            while j < len(grades):
                jdx, jgrade = grades[j]
                if points[jdx].elevation and points[jdx].elevation > peak_elev:
                    peak_idx = jdx
                    peak_elev = points[jdx].elevation

                # Check if we've gone too far
                if cum_dist[jdx] - entry_dist > max_span_m:
                    break

                # Look for steep downhill (end of peak)
                if jgrade < -min_grade_pct:
                    # Found steep down, now find where it levels off
                    k = j + 1
                    while k < len(grades):
                        kdx, kgrade = grades[k]
                        if cum_dist[kdx] - entry_dist > max_span_m:
                            break
                        if abs(kgrade) < min_grade_pct / 2:  # Grade stabilizes
                            exit_idx = kdx
                            exit_elev = points[exit_idx].elevation
                            if exit_elev is None:
                                k += 1
                                continue
                            exit_dist = cum_dist[exit_idx]

                            # Check spike criteria
                            spike_height = peak_elev - entry_elev
                            elev_return = abs(exit_elev - entry_elev)
                            span = exit_dist - entry_dist

                            if (
                                spike_height >= min_spike_height
                                and elev_return <= max_elev_return
                                and span <= max_span_m
                            ):
                                spikes.append(ElevationCorrection(
                                    start_idx=entry_idx,
                                    end_idx=exit_idx,
                                    start_km=entry_dist / 1000,
                                    end_km=exit_dist / 1000,
                                    peak_elev=peak_elev,
                                    trough_elev=entry_elev,
                                    entry_elev=entry_elev,
                                    exit_elev=exit_elev,
                                    artificial_gain=spike_height,
                                    anomaly_type=AnomalyType.SPIKE,
                                ))
                                # Skip past this spike
                                i = k
                                found_spike = True
                            break
                        k += 1
                    break
                j += 1

            if found_spike:
                continue

        i += 1

    return spikes


def detect_dropouts(
    points: list[TrackPoint],
    min_drop_depth: float = 40.0,
    max_span_m: float = 800.0,
    max_elev_return: float = 30.0,
    window_m: float = 100.0,
    min_grade_pct: float = 15.0,
    cum_dist: list[float] | None = None,
    grades: list[tuple[int, float]] | None = None,
) -> list[ElevationCorrection]:
    """Detect elevation dropout artifacts (V pattern).

    Dropouts create "V" shaped artifacts from GPS/sensor errors:
    - Sharp elevation decrease (sensor loses signal or reports bad data)
    - Sharp elevation increase (sensor recovers)
    - Returns to near-original elevation

    Args:
        points: Track points with elevation data
        min_drop_depth: Minimum elevation drop to consider (meters)
        max_span_m: Maximum horizontal distance for anomaly (meters)
        max_elev_return: How close exit elevation must be to entry (meters)
        window_m: Window size for grade calculation (meters)
        min_grade_pct: Minimum grade to trigger detection (percent)
        cum_dist: Pre-calculated cumulative distances (optional)
        grades: Pre-calculated grades (optional)

    Returns:
        List of detected dropout corrections
    """
    if len(points) < 10:
        return []

    if cum_dist is None:
        cum_dist = _calculate_cumulative_distance(points)
    if grades is None:
        grades = _calculate_grades(points, cum_dist, window_m)

    dropouts = []
    i = 0
    while i < len(grades) - 1:
        idx, grade = grades[i]

        # Look for steep downhill (start of V pattern)
        if grade < -min_grade_pct:
            entry_idx = idx
            entry_elev = points[entry_idx].elevation
            if entry_elev is None:
                i += 1
                continue
            entry_dist = cum_dist[entry_idx]

            # Find the trough (where steep down transitions to steep up)
            trough_idx = idx
            trough_elev = entry_elev
            j = i + 1
            found_dropout = False

            while j < len(grades):
                jdx, jgrade = grades[j]
                if points[jdx].elevation is not None and points[jdx].elevation < trough_elev:
                    trough_idx = jdx
                    trough_elev = points[jdx].elevation

                # Check if we've gone too far
                if cum_dist[jdx] - entry_dist > max_span_m:
                    break

                # Look for steep uphill (end of trough)
                if jgrade > min_grade_pct:
                    # Found steep up, now find where it levels off
                    k = j + 1
                    while k < len(grades):
                        kdx, kgrade = grades[k]
                        if cum_dist[kdx] - entry_dist > max_span_m:
                            break
                        if abs(kgrade) < min_grade_pct / 2:  # Grade stabilizes
                            exit_idx = kdx
                            exit_elev = points[exit_idx].elevation
                            if exit_elev is None:
                                k += 1
                                continue
                            exit_dist = cum_dist[exit_idx]

                            # Check dropout criteria
                            drop_depth = entry_elev - trough_elev
                            elev_return = abs(exit_elev - entry_elev)
                            span = exit_dist - entry_dist

                            if (
                                drop_depth >= min_drop_depth
                                and elev_return <= max_elev_return
                                and span <= max_span_m
                            ):
                                dropouts.append(ElevationCorrection(
                                    start_idx=entry_idx,
                                    end_idx=exit_idx,
                                    start_km=entry_dist / 1000,
                                    end_km=exit_dist / 1000,
                                    peak_elev=entry_elev,
                                    trough_elev=trough_elev,
                                    entry_elev=entry_elev,
                                    exit_elev=exit_elev,
                                    artificial_gain=drop_depth,
                                    anomaly_type=AnomalyType.DROPOUT,
                                ))
                                # Skip past this dropout
                                i = k
                                found_dropout = True
                            break
                        k += 1
                    break
                j += 1

            if found_dropout:
                continue

        i += 1

    return dropouts


def detect_outlier_sequences(
    points: list[TrackPoint],
    min_deviation: float = 50.0,
    max_span_m: float = 500.0,
    max_elev_return: float = 30.0,
    cum_dist: list[float] | None = None,
) -> list[ElevationCorrection]:
    """Detect sequences of points that deviate drastically from surrounding elevation.

    This detects anomalies where elevation drops or spikes and then returns to normal,
    whether it's a single point or multiple points. Works by finding sequences where:
    - Entry and exit elevations are similar (within max_elev_return)
    - Points in between deviate significantly (>= min_deviation from interpolated line)
    - The sequence spans at most max_span_m distance

    Optimized to only check points that show significant elevation changes.

    Args:
        points: Track points with elevation data
        min_deviation: Minimum deviation from expected elevation to flag (meters)
        max_span_m: Maximum distance for an anomaly sequence (meters)
        max_elev_return: How close exit elevation must be to entry (meters)
        cum_dist: Pre-calculated cumulative distances (optional)

    Returns:
        List of detected outlier corrections
    """
    if len(points) < 3:
        return []

    if cum_dist is None:
        cum_dist = _calculate_cumulative_distance(points)

    outliers = []

    # First pass: find candidate anomaly start points
    # These are points followed by a significant elevation change
    candidates = []
    for i in range(len(points) - 2):
        curr_elev = points[i].elevation
        if curr_elev is None:
            continue

        # Look at next few points for sudden change
        for k in range(i + 1, min(i + 10, len(points))):
            next_elev = points[k].elevation
            if next_elev is None:
                # None elevation is always suspicious
                candidates.append(i)
                break
            change = abs(next_elev - curr_elev)
            if change >= min_deviation * 0.5:  # Half threshold as trigger
                candidates.append(i)
                break

    # Second pass: for each candidate, look for return to normal
    processed_up_to = -1
    for i in candidates:
        if i <= processed_up_to:
            continue

        entry_elev = points[i].elevation
        if entry_elev is None:
            continue

        entry_dist = cum_dist[i]
        best_anomaly = None

        # Limit search to reasonable number of points (max ~100 points ahead)
        max_j = min(len(points), i + 150)
        for j in range(i + 2, max_j):
            exit_dist = cum_dist[j]
            span = exit_dist - entry_dist

            if span > max_span_m:
                break

            exit_elev = points[j].elevation
            if exit_elev is None:
                continue

            elev_return = abs(exit_elev - entry_elev)
            if elev_return > max_elev_return:
                continue

            # Quick check: is there any significant deviation between i and j?
            max_positive_dev = 0.0
            max_negative_dev = 0.0
            has_none = False
            peak_elev = entry_elev
            trough_elev = entry_elev

            for k in range(i + 1, j):
                actual_elev = points[k].elevation
                if actual_elev is None:
                    has_none = True
                    trough_elev = 0.0
                    max_negative_dev = max(max_negative_dev, entry_elev)
                else:
                    # Simple deviation from entry (not interpolated, for speed)
                    deviation = actual_elev - entry_elev
                    if deviation > 0:
                        max_positive_dev = max(max_positive_dev, deviation)
                        peak_elev = max(peak_elev, actual_elev)
                    else:
                        max_negative_dev = max(max_negative_dev, -deviation)
                        trough_elev = min(trough_elev, actual_elev)

            # Check if this qualifies as an anomaly
            if max_positive_dev >= min_deviation and max_positive_dev > max_negative_dev:
                anomaly = ElevationCorrection(
                    start_idx=i,
                    end_idx=j,
                    start_km=entry_dist / 1000,
                    end_km=exit_dist / 1000,
                    peak_elev=peak_elev,
                    trough_elev=entry_elev,
                    entry_elev=entry_elev,
                    exit_elev=exit_elev,
                    artificial_gain=max_positive_dev,
                    anomaly_type=AnomalyType.SPIKE,
                )
                if best_anomaly is None or j > best_anomaly.end_idx:
                    best_anomaly = anomaly

            elif max_negative_dev >= min_deviation or has_none:
                anomaly = ElevationCorrection(
                    start_idx=i,
                    end_idx=j,
                    start_km=entry_dist / 1000,
                    end_km=exit_dist / 1000,
                    peak_elev=entry_elev,
                    trough_elev=trough_elev,
                    entry_elev=entry_elev,
                    exit_elev=exit_elev,
                    artificial_gain=max_negative_dev if not has_none else entry_elev,
                    anomaly_type=AnomalyType.DROPOUT,
                )
                if best_anomaly is None or j > best_anomaly.end_idx:
                    best_anomaly = anomaly

        if best_anomaly is not None:
            outliers.append(best_anomaly)
            processed_up_to = best_anomaly.end_idx

    return outliers


# Keep old function name as alias for backwards compatibility
detect_single_point_outliers = detect_outlier_sequences


def detect_elevation_anomalies(
    points: list[TrackPoint],
    min_spike_height: float = 40.0,
    min_drop_depth: float = 40.0,
    max_span_m: float = 800.0,
    max_elev_return: float = 30.0,
    window_m: float = 100.0,
    min_grade_pct: float = 15.0,
    min_outlier_deviation: float = 50.0,
) -> list[ElevationCorrection]:
    """Detect all elevation anomalies (spikes, dropouts, and single-point outliers).

    Args:
        points: Track points with elevation data
        min_spike_height: Minimum elevation spike to consider (meters)
        min_drop_depth: Minimum elevation drop to consider (meters)
        max_span_m: Maximum horizontal distance for anomaly (meters)
        max_elev_return: How close exit elevation must be to entry (meters)
        window_m: Window size for grade calculation (meters)
        min_grade_pct: Minimum grade to trigger detection (percent)
        min_outlier_deviation: Minimum deviation for single-point outliers (meters)

    Returns:
        List of detected anomaly corrections, sorted by start index
    """
    if len(points) < 10:
        return []

    # Pre-calculate shared data
    cum_dist = _calculate_cumulative_distance(points)
    grades = _calculate_grades(points, cum_dist, window_m)

    # Detect pattern-based anomalies
    spikes = detect_spikes(
        points,
        min_spike_height=min_spike_height,
        max_span_m=max_span_m,
        max_elev_return=max_elev_return,
        window_m=window_m,
        min_grade_pct=min_grade_pct,
        cum_dist=cum_dist,
        grades=grades,
    )
    dropouts = detect_dropouts(
        points,
        min_drop_depth=min_drop_depth,
        max_span_m=max_span_m,
        max_elev_return=max_elev_return,
        window_m=window_m,
        min_grade_pct=min_grade_pct,
        cum_dist=cum_dist,
        grades=grades,
    )

    # Detect outlier sequences (single or multi-point anomalies)
    outliers = detect_outlier_sequences(
        points,
        min_deviation=min_outlier_deviation,
        max_span_m=max_span_m,
        max_elev_return=max_elev_return,
        cum_dist=cum_dist,
    )

    # Combine all detected anomalies
    # Priority: pattern-based detections (spikes/dropouts) over outlier sequences
    # because pattern-based are more precise
    pattern_anomalies = spikes + dropouts
    pattern_anomalies.sort(key=lambda x: x.start_idx)

    # Filter outliers to only include those not overlapping with pattern-based detections
    filtered_outliers = []
    for outlier in outliers:
        overlaps_pattern = False
        for pattern in pattern_anomalies:
            # Check if there's any overlap between outlier and pattern
            if (outlier.start_idx <= pattern.end_idx and
                outlier.end_idx >= pattern.start_idx):
                overlaps_pattern = True
                break
        if not overlaps_pattern:
            filtered_outliers.append(outlier)

    # Combine and sort
    all_anomalies = pattern_anomalies + filtered_outliers
    all_anomalies.sort(key=lambda x: x.start_idx)

    # Final pass: remove any remaining duplicates where one is contained in another
    filtered = []
    for anomaly in all_anomalies:
        is_contained = False
        for existing in filtered:
            if (anomaly.start_idx >= existing.start_idx and
                anomaly.end_idx <= existing.end_idx):
                is_contained = True
                break
        if not is_contained:
            filtered.append(anomaly)

    return filtered


# Backwards compatibility alias
detect_tunnels = detect_spikes


def correct_elevation_anomalies(
    points: list[TrackPoint],
    anomalies: list[ElevationCorrection],
) -> list[TrackPoint]:
    """Correct elevation anomalies by linear interpolation.

    For each detected anomaly, replaces the artificial elevation change with
    a linear interpolation between entry and exit elevations.

    Args:
        points: Original track points
        anomalies: Detected anomaly corrections

    Returns:
        New list of track points with corrected elevations
    """
    if not anomalies:
        return points

    corrected = list(points)

    for anomaly in anomalies:
        # Linear interpolate between entry and exit
        entry_elev = anomaly.entry_elev
        exit_elev = anomaly.exit_elev

        # Calculate cumulative distance for interpolation
        anomaly_cum_dist = [0.0]
        for i in range(anomaly.start_idx + 1, anomaly.end_idx + 1):
            d = haversine_distance(
                points[i - 1].lat, points[i - 1].lon,
                points[i].lat, points[i].lon
            )
            anomaly_cum_dist.append(anomaly_cum_dist[-1] + d)

        total_dist = anomaly_cum_dist[-1]

        for i, idx in enumerate(range(anomaly.start_idx, anomaly.end_idx + 1)):
            if total_dist > 0:
                t = anomaly_cum_dist[i] / total_dist
            else:
                t = 0
            new_elev = entry_elev + t * (exit_elev - entry_elev)

            pt = corrected[idx]
            corrected[idx] = TrackPoint(
                lat=pt.lat,
                lon=pt.lon,
                elevation=new_elev,
                time=pt.time,
                crr=pt.crr,
                unpaved=pt.unpaved,
                curvature=pt.curvature,
            )

    return corrected


# Backwards compatibility alias
correct_tunnel_elevations = correct_elevation_anomalies


def detect_and_correct_elevation_anomalies(
    points: list[TrackPoint],
    **kwargs,
) -> tuple[list[TrackPoint], list[ElevationCorrection]]:
    """Detect elevation anomalies and return corrected points.

    Convenience function that combines detection and correction for both
    spike (Λ) and dropout (V) patterns.

    Args:
        points: Track points with elevation data
        **kwargs: Additional arguments passed to detect_elevation_anomalies

    Returns:
        Tuple of (corrected points, list of corrections applied)
    """
    anomalies = detect_elevation_anomalies(points, **kwargs)
    corrected = correct_elevation_anomalies(points, anomalies)
    return corrected, anomalies


# Backwards compatibility alias
detect_and_correct_tunnels = detect_and_correct_elevation_anomalies
