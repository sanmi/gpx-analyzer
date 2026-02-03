import math

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import RiderParams, TrackPoint

G = 9.81  # m/s²


def effective_power(slope_angle: float, params: RiderParams) -> float:
    """Compute effective rider power output adjusted for grade.

    Models how riders modulate power based on terrain:
    - Steep descents (beyond coasting threshold): zero power (coasting/braking)
    - Gentle descents (threshold to 0): linearly ramps from 0 to flat power
    - Flat (0 degrees): base power * flat_power_factor
    - Climbs (0 to climb threshold): linearly increases from flat power to climb power
    - Steep climbs (beyond climb threshold): plateau at climb_power_factor * base
    """
    coasting_threshold_rad = math.radians(params.coasting_grade_threshold)
    climb_threshold_rad = math.radians(params.climb_threshold_grade)
    base_power = params.assumed_avg_power
    flat_power = base_power * params.flat_power_factor

    # Steep descent: coasting/braking
    if slope_angle <= coasting_threshold_rad:
        return 0.0

    # Gentle descent: ramp from 0 to flat power
    if slope_angle < 0:
        # Linear interpolation: 0 at coasting threshold, flat power at 0
        fraction = slope_angle / coasting_threshold_rad  # 1 at threshold, 0 at flat
        return flat_power * (1.0 - fraction)

    # Steep climb: plateau at max climb power
    if slope_angle >= climb_threshold_rad:
        return base_power * params.climb_power_factor

    # Moderate climb: ramp from flat power to climb power
    # Linear interpolation: flat_power at 0, climb_factor * base at climb_threshold
    fraction = slope_angle / climb_threshold_rad  # 0 at flat, 1 at threshold
    power_factor = params.flat_power_factor + (params.climb_power_factor - params.flat_power_factor) * fraction
    return base_power * power_factor


def _curvature_limited_speed(curvature: float, params: RiderParams, unpaved: bool) -> float:
    """Calculate max descent speed based on road curvature.

    Riders brake more through tight turns. This models the relationship:
    - Straight sections (low curvature): straight_descent_speed
    - Hairpin turns (high curvature): hairpin_speed
    - Linear interpolation between

    Args:
        curvature: Heading change rate in degrees per meter
        params: Rider parameters
        unpaved: Whether the surface is unpaved

    Returns:
        Maximum speed in m/s based on curvature
    """
    straight_speed = params.straight_descent_speed
    hairpin_speed = params.hairpin_speed

    # For unpaved, scale down speeds proportionally
    if unpaved:
        unpaved_ratio = params.max_coasting_speed_unpaved / params.max_coasting_speed
        straight_speed *= unpaved_ratio
        hairpin_speed *= unpaved_ratio

    # Straight section: use straight descent speed
    if curvature <= params.straight_curvature:
        return straight_speed

    # Hairpin: use hairpin speed
    if curvature >= params.hairpin_curvature:
        return hairpin_speed

    # Interpolate between straight and hairpin speeds
    fraction = (curvature - params.straight_curvature) / (params.hairpin_curvature - params.straight_curvature)
    return straight_speed + (hairpin_speed - straight_speed) * fraction


def _gradient_limited_speed(slope_angle: float, params: RiderParams, unpaved: bool, curvature: float = 0.0) -> float:
    """Calculate max descent speed based on gradient and curvature.

    Combines two braking models:
    1. Gradient-based: steeper descents require more braking
    2. Curvature-based: tighter turns require more braking

    The final speed limit is the minimum of both models.

    For unpaved surfaces, applies an additional reduction factor.
    """
    # Base speeds for paved
    gentle_speed = params.max_coasting_speed
    steep_speed = params.steep_descent_speed
    steep_grade_rad = math.radians(params.steep_descent_grade)

    # For unpaved, scale down both speeds proportionally
    if unpaved:
        unpaved_ratio = params.max_coasting_speed_unpaved / params.max_coasting_speed
        gentle_speed *= unpaved_ratio
        steep_speed *= unpaved_ratio

    # Calculate gradient-based speed limit
    if slope_angle >= 0:
        gradient_speed = gentle_speed
    elif slope_angle <= steep_grade_rad:
        gradient_speed = steep_speed
    else:
        # Interpolate between gentle and steep speeds
        fraction = slope_angle / steep_grade_rad
        gradient_speed = gentle_speed + (steep_speed - gentle_speed) * fraction

    # Calculate curvature-based speed limit (only applies on descents)
    if slope_angle < 0 and curvature > 0:
        curvature_speed = _curvature_limited_speed(curvature, params, unpaved)
        # Use the more restrictive limit
        return min(gradient_speed, curvature_speed)

    return gradient_speed


def estimate_speed_from_power(
    slope_angle: float, params: RiderParams, crr: float | None = None, unpaved: bool = False,
    curvature: float = 0.0
) -> float:
    """Estimate rider speed by solving the power balance equation.

    Solves: P_wheel = (F_grade + F_roll) * v + 0.5 * rho * CdA * (v + headwind)^2 * v
    where F_grade = m*g*sin(theta), F_roll = Crr*m*g*cos(theta),
    P_wheel = P_eff * drivetrain_efficiency (power at wheel after drivetrain losses),
    and P_eff is the effective power adjusted for downhill coasting.
    Headwind is positive when riding into the wind.

    On descents, speed is limited by gradient and curvature-dependent braking,
    then scaled by descent_speed_factor to model rider caution/preference.
    """
    effective_crr = crr if crr is not None else params.crr
    A = 0.5 * params.air_density * params.cda
    B = params.total_mass * G * (
        math.sin(slope_angle) + effective_crr * math.cos(slope_angle)
    )
    # Power at wheel after drivetrain losses
    P = effective_power(slope_angle, params) * params.drivetrain_efficiency
    max_speed = _gradient_limited_speed(slope_angle, params, unpaved, curvature)
    w = params.headwind

    # Coasting speed on descents (where gravity exceeds rolling resistance)
    # Simplified: ignore headwind for coasting estimate
    if B < 0:
        v_coast = min(math.sqrt(-B / A), max_speed)
    else:
        v_coast = 0.0

    if P <= 0:
        # Pure coasting descent - apply descent_speed_factor
        return v_coast * params.descent_speed_factor

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

    final_speed = min(max(v, v_coast), max_speed)

    # Apply descent_speed_factor on descents (slope_angle < 0)
    if slope_angle < 0:
        final_speed *= params.descent_speed_factor

    return final_speed


def calculate_segment_work(
    point_a: TrackPoint, point_b: TrackPoint, params: RiderParams
) -> tuple[float, float, float]:
    """Calculate work done by rider between two consecutive track points.

    Returns (work_joules, distance_m, elapsed_seconds).
    Work is clamped to zero minimum (rider doesn't do negative work when coasting/braking).

    When time data is missing, speed is estimated from the rider's assumed
    average power using the power balance equation.
    """
    distance = haversine_distance(
        point_a.lat, point_a.lon, point_b.lat, point_b.lon
    )

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
        speed = estimate_speed_from_power(slope_angle, params, segment_crr, point_b.unpaved, point_b.curvature)
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
