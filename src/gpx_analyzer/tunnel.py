"""Tunnel detection and correction for DEM elevation artifacts.

RideWithGPS uses Digital Elevation Model (DEM) data which doesn't account for
tunnels - it shows the mountain surface elevation instead of the tunnel floor.
This creates artificial elevation spikes that look like steep climbs.

This module detects these artifacts and corrects them by linear interpolation,
assuming tunnels are approximately flat (or have a slight grade for drainage).
"""

from dataclasses import dataclass

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import TrackPoint


@dataclass
class TunnelCorrection:
    """A detected and corrected tunnel artifact."""
    start_idx: int
    end_idx: int
    start_km: float
    end_km: float
    peak_elev: float
    entry_elev: float
    exit_elev: float
    artificial_gain: float  # meters of fake elevation gain removed


def detect_tunnels(
    points: list[TrackPoint],
    min_spike_height: float = 40.0,
    max_span_m: float = 800.0,
    max_elev_return: float = 30.0,
    window_m: float = 100.0,
    min_grade_pct: float = 15.0,
) -> list[TunnelCorrection]:
    """Detect tunnel artifacts in elevation data.

    Tunnels create "Λ" (lambda) shaped artifacts in DEM data:
    - Sharp elevation increase (GPS track "enters" the mountain surface)
    - Sharp elevation decrease (GPS track "exits" the mountain surface)
    - Returns to near-original elevation (actual tunnel is nearly flat)

    Args:
        points: Track points with elevation data
        min_spike_height: Minimum elevation spike to consider (meters)
        max_span_m: Maximum horizontal distance for a tunnel (meters)
        max_elev_return: How close exit elevation must be to entry (meters)
        window_m: Window size for grade calculation (meters)
        min_grade_pct: Minimum grade to trigger detection (percent)

    Returns:
        List of detected tunnel corrections
    """
    if len(points) < 10:
        return []

    # Calculate cumulative distance
    cum_dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon
        )
        cum_dist.append(cum_dist[-1] + d)

    # Calculate rolling grades over window_m
    grades = []  # (index, grade_pct)
    for i in range(len(points)):
        # Find points window_m ahead
        target = cum_dist[i] + window_m
        j = i
        while j < len(points) - 1 and cum_dist[j] < target:
            j += 1

        dist = cum_dist[j] - cum_dist[i]
        if dist > 0 and points[i].elevation and points[j].elevation:
            grade = (points[j].elevation - points[i].elevation) / dist * 100
            grades.append((i, grade))
        else:
            grades.append((i, 0))

    tunnels = []
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
            found_tunnel = False

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

                            # Check tunnel criteria
                            spike_height = peak_elev - entry_elev
                            elev_return = abs(exit_elev - entry_elev)
                            span = exit_dist - entry_dist

                            if (
                                spike_height >= min_spike_height
                                and elev_return <= max_elev_return
                                and span <= max_span_m
                            ):
                                tunnels.append(TunnelCorrection(
                                    start_idx=entry_idx,
                                    end_idx=exit_idx,
                                    start_km=entry_dist / 1000,
                                    end_km=exit_dist / 1000,
                                    peak_elev=peak_elev,
                                    entry_elev=entry_elev,
                                    exit_elev=exit_elev,
                                    artificial_gain=spike_height,
                                ))
                                # Skip past this tunnel
                                i = k
                                found_tunnel = True
                            break
                        k += 1
                    break
                j += 1

            if found_tunnel:
                continue

        i += 1

    return tunnels


def correct_tunnel_elevations(
    points: list[TrackPoint],
    tunnels: list[TunnelCorrection],
) -> list[TrackPoint]:
    """Correct tunnel artifacts by linear interpolation.

    For each detected tunnel, replaces the artificial elevation spike with
    a linear interpolation between entry and exit elevations. This models
    a flat or slightly graded tunnel.

    Args:
        points: Original track points
        tunnels: Detected tunnel corrections

    Returns:
        New list of track points with corrected elevations
    """
    if not tunnels:
        return points

    corrected = list(points)

    for tunnel in tunnels:
        # Linear interpolate between entry and exit
        entry_elev = tunnel.entry_elev
        exit_elev = tunnel.exit_elev

        # Calculate cumulative distance for interpolation
        tunnel_cum_dist = [0.0]
        for i in range(tunnel.start_idx + 1, tunnel.end_idx + 1):
            d = haversine_distance(
                points[i - 1].lat, points[i - 1].lon,
                points[i].lat, points[i].lon
            )
            tunnel_cum_dist.append(tunnel_cum_dist[-1] + d)

        total_dist = tunnel_cum_dist[-1]

        for i, idx in enumerate(range(tunnel.start_idx, tunnel.end_idx + 1)):
            if total_dist > 0:
                t = tunnel_cum_dist[i] / total_dist
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


def detect_and_correct_tunnels(
    points: list[TrackPoint],
    **kwargs,
) -> tuple[list[TrackPoint], list[TunnelCorrection]]:
    """Detect tunnels and return corrected points with correction info.

    Convenience function that combines detection and correction.

    Args:
        points: Track points with elevation data
        **kwargs: Additional arguments passed to detect_tunnels

    Returns:
        Tuple of (corrected points, list of corrections applied)
    """
    tunnels = detect_tunnels(points, **kwargs)
    corrected = correct_tunnel_elevations(points, tunnels)
    return corrected, tunnels
