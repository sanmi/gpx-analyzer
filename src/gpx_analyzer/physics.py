import math

from geopy.distance import geodesic

from gpx_analyzer.models import RiderParams, TrackPoint

G = 9.81  # m/s²


def calculate_segment_work(
    point_a: TrackPoint, point_b: TrackPoint, params: RiderParams
) -> tuple[float, float, float]:
    """Calculate work done by rider between two consecutive track points.

    Returns (work_joules, distance_m, elapsed_seconds).
    Work is clamped to zero minimum (rider doesn't do negative work when coasting/braking).
    """
    distance = geodesic(
        (point_a.lat, point_a.lon), (point_b.lat, point_b.lon)
    ).meters

    if distance < 0.1:
        return 0.0, distance, _elapsed(point_a, point_b)

    elapsed = _elapsed(point_a, point_b)
    speed = distance / elapsed if elapsed > 0 else 0.0

    # Elevation change
    elev_a = point_a.elevation if point_a.elevation is not None else 0.0
    elev_b = point_b.elevation if point_b.elevation is not None else 0.0
    delta_elev = elev_b - elev_a

    # Slope angle
    slope_angle = math.atan2(delta_elev, distance) if distance > 0 else 0.0

    # Gravitational work
    work_gravity = params.total_mass * G * delta_elev

    # Rolling resistance work
    work_rolling = params.crr * params.total_mass * G * math.cos(slope_angle) * distance

    # Aerodynamic drag work
    work_aero = 0.5 * params.air_density * params.cda * speed**2 * distance

    total_work = work_gravity + work_rolling + work_aero

    # Only count positive work (rider pedaling, not braking)
    return max(0.0, total_work), distance, elapsed


def _elapsed(point_a: TrackPoint, point_b: TrackPoint) -> float:
    """Return elapsed seconds between two points, or 0 if times are missing."""
    if point_a.time is not None and point_b.time is not None:
        return (point_b.time - point_a.time).total_seconds()
    return 0.0
