import math

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import RiderParams, TrackPoint

G = 9.81  # m/s²


def effective_power(slope_angle: float, params: RiderParams) -> float:
    """Compute effective rider power output adjusted for grade.

    Models how riders modulate power based on terrain:
    - Steep descents (beyond coasting threshold): zero power (coasting/braking)
    - Moderate descents (descent_transition to coasting threshold): descending_power
    - Near-flat descents (0 to descent_transition): ramps from flat_power to descending_power
    - Flat (0 degrees): flat_power
    - Climbs (0 to climb threshold): linearly increases from flat power to climbing power
    - Steep climbs (beyond climb threshold): plateau at climbing_power
    """
    coasting_threshold_rad = math.radians(params.coasting_grade_threshold)
    climb_threshold_rad = math.radians(params.climb_threshold_grade)
    # Transition from flat pedaling to descent pedaling over ~1.75% grade
    descent_transition_rad = math.radians(-1.0)
    flat_power = params.flat_power
    climbing_power = params.climbing_power
    descending_power = params.descending_power

    # Steep descent: coasting/braking
    if slope_angle <= coasting_threshold_rad:
        return 0.0

    # Moderate descent: use descending_power
    if slope_angle <= descent_transition_rad:
        return descending_power

    # Near-flat descent: ramp from flat_power (at 0) to descending_power (at transition)
    if slope_angle < 0:
        fraction = slope_angle / descent_transition_rad  # 0 at flat, 1 at transition
        return flat_power + (descending_power - flat_power) * fraction

    # Steep climb: plateau at max climbing power
    if slope_angle >= climb_threshold_rad:
        return climbing_power

    # Moderate climb: ramp from flat power to climbing power
    # Linear interpolation: flat_power at 0, climbing_power at climb_threshold
    fraction = slope_angle / climb_threshold_rad  # 0 at flat, 1 at threshold
    return flat_power + (climbing_power - flat_power) * fraction


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

    # For unpaved, scale down speeds based on gravel grade
    if unpaved:
        straight_speed *= params.gravel_coast_speed_pct
        hairpin_speed *= params.gravel_coast_speed_pct

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

    # For unpaved, scale down speeds based on gravel grade
    if unpaved:
        gentle_speed *= params.gravel_coast_speed_pct
        steep_speed *= params.gravel_coast_speed_pct

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

    On descents, speed is limited by:
    1. Gradient-dependent braking (steep = slower)
    2. Curvature-dependent braking (turns = slower)
    3. Hard cap at max_descent_speed
    4. descent_braking_factor to model rider caution/preference
    """
    effective_crr = crr if crr is not None else params.crr
    A = 0.5 * params.air_density * params.cda
    B = params.total_mass * G * (
        math.sin(slope_angle) + effective_crr * math.cos(slope_angle)
    )
    # Power at wheel after drivetrain losses
    P = effective_power(slope_angle, params) * params.drivetrain_efficiency
    if unpaved:
        P *= params.unpaved_power_factor
    max_speed = _gradient_limited_speed(slope_angle, params, unpaved, curvature)

    # Apply hard cap on descent speed
    if slope_angle < 0:
        hard_cap = params.max_descent_speed
        if unpaved:
            # Scale down for unpaved surfaces based on gravel grade
            hard_cap *= params.gravel_coast_speed_pct
        max_speed = min(max_speed, hard_cap)

    w = params.headwind

    # Coasting speed on descents (where gravity exceeds rolling resistance)
    # Simplified: ignore headwind for coasting estimate
    if B < 0:
        v_coast = min(math.sqrt(-B / A), max_speed)
    else:
        v_coast = 0.0

    if P <= 0:
        # Pure coasting descent - apply descent_braking_factor
        return v_coast * params.descent_braking_factor

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

    # Apply descent_braking_factor on descents (slope_angle < 0)
    if slope_angle < 0:
        final_speed *= params.descent_braking_factor

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
        physics_speed = speed  # Use recorded speed for work calculation
    else:
        # Estimate speed from power model
        speed = estimate_speed_from_power(slope_angle, params, segment_crr, point_b.unpaved, point_b.curvature)
        elapsed = distance / speed if speed > 0 else 0.0
        # For work calculation, use physics speed WITHOUT descent_braking_factor
        # This keeps work independent of rider descent preference for calibration
        if slope_angle < 0 and params.descent_braking_factor != 1.0:
            # Reverse the descent_braking_factor to get physics-based speed
            physics_speed = speed / params.descent_braking_factor if params.descent_braking_factor > 0 else speed
        else:
            physics_speed = speed

    # Gravitational work
    work_gravity = params.total_mass * G * delta_elev

    # Rolling resistance work
    work_rolling = segment_crr * params.total_mass * G * math.cos(slope_angle) * distance

    # Aerodynamic drag work (based on physics speed, not preference-adjusted speed)
    # This keeps work calculation independent of descent_braking_factor for calibration
    airspeed = physics_speed + params.headwind
    work_aero = 0.5 * params.air_density * params.cda * airspeed**2 * distance

    total_work = work_gravity + work_rolling + work_aero

    # Apply gravel work multiplier for unpaved segments
    # This accounts for suspension losses and inefficiencies on rough surfaces
    if point_b.unpaved and total_work > 0:
        total_work *= params.gravel_work_multiplier

    # Only count positive work (rider pedaling, not braking)
    return max(0.0, total_work), distance, elapsed


def _elapsed(point_a: TrackPoint, point_b: TrackPoint) -> float:
    """Return elapsed seconds between two points, or 0 if times are missing."""
    if point_a.time is not None and point_b.time is not None:
        return (point_b.time - point_a.time).total_seconds()
    return 0.0
