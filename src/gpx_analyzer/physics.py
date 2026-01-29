import math

from geopy.distance import geodesic

from gpx_analyzer.models import RiderParams, TrackPoint

G = 9.81  # m/s²


def effective_power(slope_angle: float, params: RiderParams) -> float:
    """Compute effective rider power output adjusted for grade.

    On flat or uphill (grade >= 0): full assumed power.
    On downhill: linearly reduces from full power at 0 degrees to zero
    at the coasting grade threshold. Beyond the threshold: zero power.
    """
    threshold_rad = math.radians(params.coasting_grade_threshold)
    if slope_angle >= 0:
        return params.assumed_avg_power
    if slope_angle <= threshold_rad:
        return 0.0
    # Linear interpolation: full power at 0, zero at threshold
    fraction = slope_angle / threshold_rad  # 0 at flat, 1 at threshold
    return params.assumed_avg_power * (1.0 - fraction)


def estimate_speed_from_power(
    slope_angle: float, params: RiderParams, crr: float | None = None, unpaved: bool = False
) -> float:
    """Estimate rider speed by solving the power balance equation.

    Solves: P_eff = (F_grade + F_roll) * v + 0.5 * rho * CdA * (v + headwind)^2 * v
    where F_grade = m*g*sin(theta), F_roll = Crr*m*g*cos(theta),
    and P_eff is the effective power adjusted for downhill coasting.
    Headwind is positive when riding into the wind.

    On steep descents where coasting speed exceeds pedaling speed,
    returns the coasting speed capped at max_coasting_speed (or max_coasting_speed_unpaved
    for unpaved surfaces).
    """
    effective_crr = crr if crr is not None else params.crr
    A = 0.5 * params.air_density * params.cda
    B = params.total_mass * G * (
        math.sin(slope_angle) + effective_crr * math.cos(slope_angle)
    )
    P = effective_power(slope_angle, params)
    max_speed = params.max_coasting_speed_unpaved if unpaved else params.max_coasting_speed
    w = params.headwind

    # Coasting speed on descents (where gravity exceeds rolling resistance)
    # Simplified: ignore headwind for coasting estimate
    if B < 0:
        v_coast = min(math.sqrt(-B / A), max_speed)
    else:
        v_coast = 0.0

    if P <= 0:
        return v_coast

    # Solve A*(v+w)^2*v + B*v - P = 0 using Newton's method
    v = max(5.0, v_coast)
    for _ in range(50):
        airspeed = v + w
        f = A * airspeed**2 * v + B * v - P
        fp = A * (airspeed**2 + 2 * airspeed * v) + B
        if abs(fp) < 1e-12:
            break
        v_new = v - f / fp
        if v_new <= 0:
            v_new = v / 2
        if abs(v_new - v) < 1e-8:
            break
        v = v_new

    return min(max(v, v_coast), max_speed)


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

    # Use per-segment crr if available, otherwise use params default
    segment_crr = point_b.crr if point_b.crr is not None else params.crr

    # Determine speed and elapsed time
    elapsed = _elapsed(point_a, point_b)
    if elapsed > 0:
        speed = distance / elapsed
    else:
        speed = estimate_speed_from_power(slope_angle, params, segment_crr, point_b.unpaved)
        elapsed = distance / speed if speed > 0 else 0.0

    # Gravitational work
    work_gravity = params.total_mass * G * delta_elev

    # Rolling resistance work
    work_rolling = segment_crr * params.total_mass * G * math.cos(slope_angle) * distance

    # Aerodynamic drag work (based on airspeed = ground speed + headwind)
    airspeed = speed + params.headwind
    work_aero = 0.5 * params.air_density * params.cda * airspeed**2 * distance

    total_work = work_gravity + work_rolling + work_aero

    # Only count positive work (rider pedaling, not braking)
    return max(0.0, total_work), distance, elapsed


def _elapsed(point_a: TrackPoint, point_b: TrackPoint) -> float:
    """Return elapsed seconds between two points, or 0 if times are missing."""
    if point_a.time is not None and point_b.time is not None:
        return (point_b.time - point_a.time).total_seconds()
    return 0.0
