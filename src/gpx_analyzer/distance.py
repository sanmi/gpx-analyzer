"""Fast distance and geometry calculations.

Haversine is ~10x faster than geopy.geodesic and accurate enough for cycling
(< 0.5% error at typical distances).
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpx_analyzer.models import TrackPoint

# Earth's mean radius in meters
EARTH_RADIUS_M = 6_371_000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees

    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_M * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees

    Returns:
        Bearing in degrees (0-360, where 0=North, 90=East)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def calculate_curvature(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    lat3: float, lon3: float
) -> float:
    """Calculate curvature at point 2 given three consecutive points.

    Curvature is the rate of heading change per unit distance, measured
    in degrees per meter. Higher values indicate sharper turns.

    Args:
        lat1, lon1: Previous point
        lat2, lon2: Current point (where curvature is measured)
        lat3, lon3: Next point

    Returns:
        Curvature in degrees per meter (always positive)
    """
    # Calculate bearings
    bearing1 = calculate_bearing(lat1, lon1, lat2, lon2)
    bearing2 = calculate_bearing(lat2, lon2, lat3, lon3)

    # Heading change (handle wrap-around)
    heading_change = abs(bearing2 - bearing1)
    if heading_change > 180:
        heading_change = 360 - heading_change

    # Distance over which the turn occurs (use average of both segments)
    dist1 = haversine_distance(lat1, lon1, lat2, lon2)
    dist2 = haversine_distance(lat2, lon2, lat3, lon3)
    avg_dist = (dist1 + dist2) / 2

    if avg_dist < 1.0:  # Avoid division by very small distances
        return 0.0

    return heading_change / avg_dist


def compute_route_curvature(points: list[TrackPoint], smoothing_window: float = 50.0) -> None:
    """Compute curvature for all points in a route, modifying points in place.

    Uses a rolling window to smooth out GPS noise. The curvature at each point
    is calculated by looking at heading change over the smoothing window distance.

    Args:
        points: List of TrackPoints to compute curvature for
        smoothing_window: Distance in meters over which to measure heading change
    """
    if len(points) < 3:
        return

    # Build cumulative distance array for efficient range lookups
    cum_dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            points[i-1].lat, points[i-1].lon,
            points[i].lat, points[i].lon
        )
        cum_dist.append(cum_dist[-1] + d)

    total_dist = cum_dist[-1]
    half_window = smoothing_window / 2

    for i in range(len(points)):
        current_dist = cum_dist[i]

        # Find point approximately half_window behind
        target_behind = max(0, current_dist - half_window)
        j_behind = i
        while j_behind > 0 and cum_dist[j_behind] > target_behind:
            j_behind -= 1

        # Find point approximately half_window ahead
        target_ahead = min(total_dist, current_dist + half_window)
        j_ahead = i
        while j_ahead < len(points) - 1 and cum_dist[j_ahead] < target_ahead:
            j_ahead += 1

        # Need at least 3 distinct points
        if j_behind == i or j_ahead == i:
            points[i].curvature = 0.0
            continue

        # Calculate curvature using the window endpoints
        actual_dist = cum_dist[j_ahead] - cum_dist[j_behind]
        if actual_dist < 10.0:  # Minimum distance for meaningful curvature
            points[i].curvature = 0.0
            continue

        # Calculate heading change over the window
        bearing_start = calculate_bearing(
            points[j_behind].lat, points[j_behind].lon,
            points[i].lat, points[i].lon
        )
        bearing_end = calculate_bearing(
            points[i].lat, points[i].lon,
            points[j_ahead].lat, points[j_ahead].lon
        )

        heading_change = abs(bearing_end - bearing_start)
        if heading_change > 180:
            heading_change = 360 - heading_change

        points[i].curvature = heading_change / actual_dist
