import math

from geopy.distance import geodesic

from gpx_analyzer.models import RiderParams, TrackPoint

G = 9.81  # m/s²
MAX_ESTIMATED_SPEED = 20.0  # m/s (~72 km/h) cap for descent speed estimation


def estimate_speed_from_power(slope_angle: float, params: RiderParams) -> float:
    """Estimate rider speed by solving the power balance equation.

    Solves: P = (F_grade + F_roll) * v + 0.5 * rho * CdA * v^3
    where F_grade = m*g*sin(theta), F_roll = Crr*m*g*cos(theta).

    On steep descents where coasting speed exceeds pedaling speed,
    returns the coasting speed (rider does zero work).
    """
    A = 0.5 * params.air_density * params.cda
    B = params.total_mass * G * (
        math.sin(slope_angle) + params.crr * math.cos(slope_angle)
    )
    P = params.assumed_avg_power

    # Coasting speed on descents (where gravity exceeds rolling resistance)
    if B < 0:
        v_coast = math.sqrt(-B / A)
    else:
        v_coast = 0.0

    if P <= 0:
        return min(v_coast, MAX_ESTIMATED_SPEED)

    # Solve A*v^3 + B*v - P = 0 using Newton's method
    v = max(5.0, v_coast)
    for _ in range(50):
        f = A * v**3 + B * v - P
        fp = 3 * A * v**2 + B
        if abs(fp) < 1e-12:
            break
        v_new = v - f / fp
        if v_new <= 0:
            v_new = v / 2
        if abs(v_new - v) < 1e-8:
            break
        v = v_new

    return min(max(v, v_coast), MAX_ESTIMATED_SPEED)


def calculate_segment_work(
    point_a: TrackPoint, point_b: TrackPoint, params: RiderParams
) -> tuple[float, float, float]:
    """Calculate work done by rider between two consecutive track points.

    Returns (work_joules, distance_m, elapsed_seconds).
    Work is clamped to zero minimum (rider doesn't do negative work when coasting/braking).

    When time data is missing, speed is estimated from the rider's assumed
    average power using the power balance equation.
    """
    distance = geodesic(
        (point_a.lat, point_a.lon), (point_b.lat, point_b.lon)
    ).meters

    if distance < 0.1:
        return 0.0, distance, _elapsed(point_a, point_b)

    # Elevation change
    elev_a = point_a.elevation if point_a.elevation is not None else 0.0
    elev_b = point_b.elevation if point_b.elevation is not None else 0.0
    delta_elev = elev_b - elev_a

    # Slope angle
    slope_angle = math.atan2(delta_elev, distance) if distance > 0 else 0.0

    # Determine speed and elapsed time
    elapsed = _elapsed(point_a, point_b)
    if elapsed > 0:
        speed = distance / elapsed
    else:
        speed = estimate_speed_from_power(slope_angle, params)
        elapsed = distance / speed if speed > 0 else 0.0

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
