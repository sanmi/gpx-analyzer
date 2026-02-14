"""Training data management and batch comparison."""

import json
from dataclasses import dataclass
from pathlib import Path

from gpx_analyzer.analyzer import analyze, calculate_hilliness, DEFAULT_MAX_GRADE_WINDOW, MAX_REALISTIC_GRADE, DEFAULT_MAX_GRADE_SMOOTHING
from gpx_analyzer.compare import ComparisonResult, compare_route_with_trip, _calculate_actual_work
from gpx_analyzer.models import RiderParams
from gpx_analyzer.ridewithgps import (
    TripPoint,
    get_route_with_surface,
    get_trip_data,
    is_ridewithgps_trip_url,
    is_ridewithgps_url,
)
from gpx_analyzer.smoothing import smooth_elevations
from gpx_analyzer.distance import haversine_distance


# Grade thresholds for terrain classification
CLIMB_THRESHOLD = 2.0    # Grade > 2% is climbing
DESCENT_THRESHOLD = -2.0  # Grade < -2% is descending
MAX_COASTING_SPEED_MS = 48.0 / 3.6  # Default 48 km/h in m/s

# Default gravel grades - matches cli.py DEFAULT_GRAVEL_GRADES
DEFAULT_GRAVEL_GRADES = {
    "1": {"power_factor": 0.95, "work_multiplier": 1.05, "coast_speed_pct": 0.90},
    "2": {"power_factor": 0.90, "work_multiplier": 1.12, "coast_speed_pct": 0.75},
    "3": {"power_factor": 0.82, "work_multiplier": 1.25, "coast_speed_pct": 0.60},
    "4": {"power_factor": 0.72, "work_multiplier": 1.40, "coast_speed_pct": 0.45},
}


def _get_gravel_grade_params(grade: int) -> dict:
    """Get gravel parameters for a given grade (1-4).

    Returns dict with: power_factor, work_multiplier, coast_speed_pct
    """
    grade_key = str(grade)
    if grade_key in DEFAULT_GRAVEL_GRADES:
        return DEFAULT_GRAVEL_GRADES[grade_key]
    # Fall back to default for grade 2
    return DEFAULT_GRAVEL_GRADES["2"]


@dataclass
class TrainingRoute:
    """A route/trip pair for training."""

    name: str
    route_url: str
    trip_url: str
    tags: list[str]
    notes: str = ""
    avg_watts: float | None = None  # Override default power for this ride
    headwind: float | None = None  # Override headwind in km/h (negative = tailwind)
    mass: float | None = None  # Override total mass (rider + bike) in kg
    smoothing: float | None = None  # Override smoothing radius in meters
    gravel_grade: int | None = None  # Override gravel grade (1-4) for unpaved sections


@dataclass
class VerboseMetrics:
    """Verbose metrics calculated from trip data."""

    avg_power_climbing: float | None  # Avg power when grade > 2%
    avg_power_flat: float | None  # Avg power when -2% <= grade <= 2%
    avg_power_descending: float | None  # Avg power when grade < -2%
    braking_score: float | None  # Ratio of actual descent time vs theoretical (>1 = more braking)
    time_climbing_pct: float  # % of time spent climbing (grade > 2%)
    time_flat_pct: float  # % of time on flats (-2% to 2%)
    time_descending_pct: float  # % of time descending (grade < -2%)


@dataclass
class TrainingResult:
    """Result of analyzing a training route."""

    route: TrainingRoute
    comparison: ComparisonResult
    route_elevation_gain: float
    trip_elevation_gain: float | None
    route_distance: float
    unpaved_pct: float
    power_used: float  # The power value used for prediction
    mass_used: float = 84.0  # The mass value used for prediction
    elevation_scale_used: float = 1.0  # Scale factor applied to match trip elevation
    route_max_grade: float = 0.0  # Max grade from route (%)
    trip_max_grade: float | None = None  # Max grade from trip (%)
    verbose_metrics: VerboseMetrics | None = None  # Verbose metrics from trip data (actual)
    estimated_verbose_metrics: VerboseMetrics | None = None  # Verbose metrics from route (estimated)
    inferred_braking_factor: float | None = None  # actual_braking_score / estimated_braking_score


@dataclass
class TrainingSummary:
    """Aggregate summary across all training data."""

    total_routes: int
    total_distance_km: float
    total_elevation_m: float
    avg_time_error_pct: float
    avg_work_error_pct: float
    results_by_tag: dict[str, list[TrainingResult]]

    # Breakdown by terrain type
    road_avg_time_error: float | None
    gravel_avg_time_error: float | None
    hilly_avg_time_error: float | None
    flat_avg_time_error: float | None


def _calculate_elevation_gain(points: list) -> float:
    """Calculate total elevation gain from a list of points."""
    if len(points) < 2:
        return 0.0
    gain = 0.0
    for i in range(1, len(points)):
        elev_a = points[i - 1].elevation or 0.0
        elev_b = points[i].elevation or 0.0
        delta = elev_b - elev_a
        if delta > 0:
            gain += delta
    return gain


# Adaptive smoothing constants for high-noise DEM detection
HIGH_NOISE_RATIO_THRESHOLD = 1.8  # raw_gain / api_gain ratio indicating noisy DEM
HIGH_NOISE_SMOOTHING_RADIUS = 300.0  # meters, used when DEM is noisy


def _is_high_noise_dem(raw_gain: float, api_gain: float) -> bool:
    """Detect if DEM data has high noise based on raw/API elevation ratio."""
    if api_gain <= 0 or raw_gain <= 0:
        return False
    ratio = raw_gain / api_gain
    return ratio > HIGH_NOISE_RATIO_THRESHOLD


def _calculate_trip_max_grade(points: list[TripPoint], window: float = 50.0) -> float:
    """Calculate max grade from trip points using rolling average over distance window.

    Uses the cumulative distance from trip points for more accurate measurement.
    If cumulative distance is not available, calculates it from lat/lon coordinates.
    Skips segments with missing elevation data.
    Uses 50m window (shorter than route's 150m) to capture actual steep sections.
    """
    from gpx_analyzer.distance import haversine_distance

    if len(points) < 2:
        return 0.0

    # Check if cumulative distance data is available
    has_distance = any(p.distance is not None and p.distance > 0 for p in points)

    # Build cumulative distance array
    cum_dist = [0.0] * len(points)
    if has_distance:
        for i, p in enumerate(points):
            cum_dist[i] = p.distance
    else:
        # Calculate cumulative distance from lat/lon
        for i in range(1, len(points)):
            d = haversine_distance(
                points[i - 1].lat, points[i - 1].lon,
                points[i].lat, points[i].lon
            )
            cum_dist[i] = cum_dist[i - 1] + d

    max_grade = 0.0

    for i in range(len(points) - 1):
        # Skip if start point has no elevation
        if points[i].elevation is None:
            continue

        start_dist = cum_dist[i]
        start_elev = points[i].elevation

        # Find point approximately 'window' meters ahead
        target_dist = start_dist + window
        j = i + 1
        while j < len(points) - 1 and cum_dist[j] < target_dist:
            j += 1

        # Skip if end point has no elevation
        if points[j].elevation is None:
            continue

        end_dist = cum_dist[j]
        end_elev = points[j].elevation

        dist_delta = end_dist - start_dist

        # Skip segments shorter than half the window (avoids edge effects)
        if dist_delta < window / 2:
            continue
        if dist_delta > 0:
            grade = (end_elev - start_elev) / dist_delta * 100
            # Cap at realistic max to filter GPS/elevation errors
            grade = min(grade, MAX_REALISTIC_GRADE)
            if grade > max_grade:
                max_grade = grade

    return max_grade


def calculate_verbose_metrics(
    trip_points: list[TripPoint],
    max_coasting_speed_ms: float,
) -> VerboseMetrics:
    """Calculate verbose metrics from trip data.

    Args:
        trip_points: Trip points with power, speed, elevation data
        max_coasting_speed_ms: Model's max coasting speed in m/s for braking score

    Returns:
        VerboseMetrics with power by terrain type and braking score
    """

    # Accumulate time and power by terrain type
    climb_time = 0.0
    climb_power_sum = 0.0
    climb_power_count = 0

    flat_time = 0.0
    flat_power_sum = 0.0
    flat_power_count = 0

    descent_time = 0.0
    descent_power_sum = 0.0
    descent_power_count = 0
    descent_speed_sum = 0.0  # For avg descent speed

    total_time = 0.0

    for i in range(1, len(trip_points)):
        prev = trip_points[i - 1]
        curr = trip_points[i]

        # Skip if missing timestamps
        if prev.timestamp is None or curr.timestamp is None:
            continue

        time_delta = curr.timestamp - prev.timestamp
        if time_delta <= 0:
            continue

        # Calculate distance between points
        # Use cumulative distance if available, otherwise calculate from lat/lon
        dist = curr.distance - prev.distance
        if dist <= 0:
            # Fallback to haversine distance from coordinates
            dist = haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)

        # Calculate grade between points
        if dist > 0 and prev.elevation is not None and curr.elevation is not None:
            grade = (curr.elevation - prev.elevation) / dist * 100
        else:
            grade = 0.0

        # Clamp extreme grades
        grade = max(-30, min(30, grade))

        total_time += time_delta

        # Categorize by terrain and accumulate power
        if grade > CLIMB_THRESHOLD:
            climb_time += time_delta
            if curr.power is not None and curr.power > 0:
                climb_power_sum += curr.power * time_delta
                climb_power_count += time_delta
        elif grade < DESCENT_THRESHOLD:
            descent_time += time_delta
            if curr.power is not None:
                descent_power_sum += curr.power * time_delta
                descent_power_count += time_delta
            if curr.speed is not None and curr.speed > 0:
                descent_speed_sum += curr.speed * time_delta
        else:
            flat_time += time_delta
            if curr.power is not None and curr.power > 0:
                flat_power_sum += curr.power * time_delta
                flat_power_count += time_delta

    # Calculate averages
    avg_power_climbing = climb_power_sum / climb_power_count if climb_power_count > 0 else None
    avg_power_flat = flat_power_sum / flat_power_count if flat_power_count > 0 else None
    avg_power_descending = descent_power_sum / descent_power_count if descent_power_count > 0 else None

    # Calculate time percentages
    if total_time > 0:
        time_climbing_pct = climb_time / total_time * 100
        time_flat_pct = flat_time / total_time * 100
        time_descending_pct = descent_time / total_time * 100
    else:
        time_climbing_pct = time_flat_pct = time_descending_pct = 0.0

    # Braking score: ratio of actual avg descent speed to max coasting speed
    # < 1 means they descended slower than max (more braking/caution)
    # = 1 means they descended at max coasting speed
    # > 1 means they descended faster than max coasting speed (unlikely)
    if descent_time > 0 and max_coasting_speed_ms > 0:
        avg_descent_speed = descent_speed_sum / descent_time
        braking_score = avg_descent_speed / max_coasting_speed_ms
    else:
        braking_score = None

    return VerboseMetrics(
        avg_power_climbing=avg_power_climbing,
        avg_power_flat=avg_power_flat,
        avg_power_descending=avg_power_descending,
        braking_score=braking_score,
        time_climbing_pct=time_climbing_pct,
        time_flat_pct=time_flat_pct,
        time_descending_pct=time_descending_pct,
    )


def _calculate_estimated_verbose_metrics(
    route_points: list,
    params: RiderParams,
) -> VerboseMetrics:
    """Calculate estimated verbose metrics from route using physics model.

    Args:
        route_points: Smoothed route track points
        params: Rider parameters including power and climb factors

    Returns:
        VerboseMetrics with estimated power by terrain and braking score
    """
    import math
    from gpx_analyzer.physics import estimate_speed_from_power

    CLIMB_THRESHOLD = 2.0  # Grade > 2% is climbing
    DESCENT_THRESHOLD = -2.0  # Grade < -2% is descending

    # For estimated power:
    # - Climbing: climbing_power (direct input)
    # - Flat: flat_power (direct input)
    # - Descending: minimal pedaling (model assumes coasting)
    flat_power = params.flat_power
    climb_power = params.climbing_power

    # Calculate time at each terrain type using physics model
    climb_time = 0.0
    flat_time = 0.0
    descent_time = 0.0
    descent_speed_sum = 0.0

    for i in range(1, len(route_points)):
        prev = route_points[i - 1]
        curr = route_points[i]

        # Calculate distance and grade
        dist = haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)
        if dist <= 0:
            continue

        if prev.elevation is not None and curr.elevation is not None:
            grade_pct = (curr.elevation - prev.elevation) / dist * 100
        else:
            grade_pct = 0.0

        grade_pct = max(-30, min(30, grade_pct))

        # Calculate speed using physics model (convert grade % to slope angle in radians)
        slope_angle = math.atan(grade_pct / 100)
        speed = estimate_speed_from_power(slope_angle, params)
        if speed <= 0:
            continue

        segment_time = dist / speed

        if grade_pct > CLIMB_THRESHOLD:
            climb_time += segment_time
        elif grade_pct < DESCENT_THRESHOLD:
            descent_time += segment_time
            descent_speed_sum += speed * segment_time
        else:
            flat_time += segment_time

    total_time = climb_time + flat_time + descent_time

    # Calculate time percentages
    if total_time > 0:
        time_climbing_pct = climb_time / total_time * 100
        time_flat_pct = flat_time / total_time * 100
        time_descending_pct = descent_time / total_time * 100
    else:
        time_climbing_pct = time_flat_pct = time_descending_pct = 0.0

    # Estimated braking score: model's avg descent speed / max coasting speed
    if descent_time > 0 and params.max_coasting_speed > 0:
        avg_descent_speed = descent_speed_sum / descent_time
        braking_score = avg_descent_speed / params.max_coasting_speed
    else:
        braking_score = None

    return VerboseMetrics(
        avg_power_climbing=climb_power,
        avg_power_flat=flat_power,
        avg_power_descending=params.descending_power,  # Light pedaling on gentle descents
        braking_score=braking_score,
        time_climbing_pct=time_climbing_pct,
        time_flat_pct=time_flat_pct,
        time_descending_pct=time_descending_pct,
    )


def load_training_data(path: Path) -> list[TrainingRoute]:
    """Load training data from JSON file."""
    with path.open() as f:
        data = json.load(f)

    routes = []
    for entry in data.get("routes", []):
        routes.append(
            TrainingRoute(
                name=entry["name"],
                route_url=entry["route_url"],
                trip_url=entry["trip_url"],
                tags=entry.get("tags", []),
                notes=entry.get("notes", ""),
                avg_watts=entry.get("avg_watts"),
                headwind=entry.get("headwind"),
                mass=entry.get("mass"),
                smoothing=entry.get("smoothing"),
                gravel_grade=entry.get("gravel_grade"),
            )
        )
    return routes


def analyze_training_route(
    route: TrainingRoute,
    params: RiderParams,
    smoothing_radius: float = 50.0,
    elevation_scale: float = 1.0,
    use_trip_elevation: bool = True,
    max_grade_window_route: float = DEFAULT_MAX_GRADE_WINDOW,
    max_grade_window_trip: float = 50.0,
    max_grade_smoothing: float = DEFAULT_MAX_GRADE_SMOOTHING,
) -> TrainingResult | None:
    """Analyze a single training route and return comparison results.

    Args:
        route: Training route to analyze
        params: Rider parameters
        smoothing_radius: Elevation smoothing radius in meters
        elevation_scale: Manual elevation scale factor
        use_trip_elevation: If True, scale route elevation to match trip elevation
        max_grade_window_route: Rolling window size for route max grade (meters)
        max_grade_window_trip: Rolling window size for trip max grade (meters)
        max_grade_smoothing: Smoothing radius for max grade calculation (meters)
    """
    # Validate URLs
    if not is_ridewithgps_url(route.route_url):
        print(f"  Skipping {route.name}: invalid route URL")
        return None
    if not is_ridewithgps_trip_url(route.trip_url):
        print(f"  Skipping {route.name}: invalid trip URL")
        return None

    try:
        # Get route data
        route_points, route_metadata = get_route_with_surface(route.route_url, params.crr)

        # Get trip data
        trip_points, trip_metadata = get_trip_data(route.trip_url)

        if len(route_points) < 2 or len(trip_points) < 2:
            print(f"  Skipping {route.name}: insufficient data points")
            return None

        # Use per-route mass if specified, otherwise use default
        mass_used = route.mass if route.mass is not None else params.total_mass
        headwind_used = route.headwind / 3.6 if route.headwind is not None else params.headwind

        # Get gravel parameters - use per-route grade if specified, otherwise use params defaults
        if route.gravel_grade is not None:
            gravel_params = _get_gravel_grade_params(route.gravel_grade)
            unpaved_power_factor = gravel_params["power_factor"]
            gravel_work_multiplier = gravel_params["work_multiplier"]
            gravel_coast_speed_pct = gravel_params["coast_speed_pct"]
        else:
            unpaved_power_factor = params.unpaved_power_factor
            gravel_work_multiplier = params.gravel_work_multiplier
            gravel_coast_speed_pct = params.gravel_coast_speed_pct

        # Calculate actual power by terrain type from trip data
        # This gives us climbing_power and flat_power separately
        actual_metrics = calculate_verbose_metrics(trip_points, params.max_coasting_speed)

        # Use per-route power if specified, otherwise use actual climbing/flat/descending power from trip
        if route.avg_watts is not None:
            # If avg_watts specified, use it for climbing and derive others proportionally
            climbing_power_used = route.avg_watts
            flat_power_ratio = params.flat_power / params.climbing_power if params.climbing_power > 0 else 0.8
            flat_power_used = route.avg_watts * flat_power_ratio
            descending_power_used = params.descending_power
        elif actual_metrics.avg_power_climbing is not None:
            # Use actual climbing, flat, and descending power from trip
            climbing_power_used = actual_metrics.avg_power_climbing
            flat_power_used = actual_metrics.avg_power_flat if actual_metrics.avg_power_flat is not None else climbing_power_used * 0.8
            descending_power_used = actual_metrics.avg_power_descending if actual_metrics.avg_power_descending is not None else params.descending_power
        else:
            # Fall back to params defaults
            climbing_power_used = params.climbing_power
            flat_power_used = params.flat_power
            descending_power_used = params.descending_power

        # Get elevation gains from metadata
        api_elevation_gain = route_metadata.get("elevation_gain")  # DEM-derived from RWGPS
        trip_elevation_gain = trip_metadata.get("elevation_gain")

        # Calculate raw elevation gain to detect high-noise GPS elevation
        raw_elevation_gain = _calculate_elevation_gain(route_points)

        # Check for high-noise GPS elevation data - use adaptive smoothing radius
        effective_smoothing = smoothing_radius
        if api_elevation_gain and _is_high_noise_dem(raw_elevation_gain, api_elevation_gain):
            effective_smoothing = max(smoothing_radius, HIGH_NOISE_SMOOTHING_RADIUS)

        # Always apply API scaling if available
        effective_scale = elevation_scale
        if use_trip_elevation and api_elevation_gain and api_elevation_gain > 0:
            # Smooth first to calculate gain, then scale to match API elevation
            smoothed = smooth_elevations(route_points, effective_smoothing, elevation_scale)
            route_gain_smoothed = _calculate_elevation_gain(smoothed)
            if route_gain_smoothed > 0:
                correction_factor = api_elevation_gain / route_gain_smoothed
                effective_scale = elevation_scale * correction_factor
                smoothed = smooth_elevations(route_points, effective_smoothing, effective_scale)
        else:
            # No API elevation - just smooth
            smoothed = smooth_elevations(route_points, effective_smoothing, elevation_scale)

        # Build params with descent_braking_factor=1.0 to get raw physics predictions
        physics_params = RiderParams(
            total_mass=mass_used,
            cda=params.cda,
            crr=params.crr,
            air_density=params.air_density,
            climbing_power=climbing_power_used,
            flat_power=flat_power_used,
            descending_power=descending_power_used,
            coasting_grade_threshold=params.coasting_grade_threshold,
            max_coasting_speed=params.max_coasting_speed,
            max_coasting_speed_unpaved=params.max_coasting_speed_unpaved,
            max_descent_speed=params.max_descent_speed,
            headwind=headwind_used,
            climb_threshold_grade=params.climb_threshold_grade,
            steep_descent_speed=params.steep_descent_speed,
            steep_descent_grade=params.steep_descent_grade,
            straight_descent_speed=params.straight_descent_speed,
            hairpin_speed=params.hairpin_speed,
            straight_curvature=params.straight_curvature,
            hairpin_curvature=params.hairpin_curvature,
            descent_braking_factor=1.0,  # Raw physics, no adjustment
            drivetrain_efficiency=params.drivetrain_efficiency,
            unpaved_power_factor=unpaved_power_factor,
            gravel_work_multiplier=gravel_work_multiplier,
            gravel_coast_speed_pct=gravel_coast_speed_pct,
        )

        # Calculate verbose metrics for reporting (no longer used for descent factor inference)
        physics_metrics = _calculate_estimated_verbose_metrics(smoothed, physics_params)

        # Calculate unpaved percentage
        unpaved_pct = route_metadata.get("unpaved_pct", 0) or 0

        # Calculate max grades
        trip_max_grade = _calculate_trip_max_grade(trip_points, max_grade_window_trip)

        # Step 1: Calculate verbose metrics with braking_factor=1.0 to infer actual braking behavior
        # Build physics params (factor=1.0) for braking score calculation
        physics_params = RiderParams(
            total_mass=mass_used,
            cda=params.cda,
            crr=params.crr,
            air_density=params.air_density,
            climbing_power=climbing_power_used,
            flat_power=flat_power_used,
            descending_power=descending_power_used,
            coasting_grade_threshold=params.coasting_grade_threshold,
            max_coasting_speed=params.max_coasting_speed,
            max_coasting_speed_unpaved=params.max_coasting_speed_unpaved,
            max_descent_speed=params.max_descent_speed,
            headwind=headwind_used,
            climb_threshold_grade=params.climb_threshold_grade,
            steep_descent_speed=params.steep_descent_speed,
            steep_descent_grade=params.steep_descent_grade,
            straight_descent_speed=params.straight_descent_speed,
            hairpin_speed=params.hairpin_speed,
            straight_curvature=params.straight_curvature,
            hairpin_curvature=params.hairpin_curvature,
            descent_braking_factor=1.0,  # Pure physics for braking inference
            drivetrain_efficiency=params.drivetrain_efficiency,
            unpaved_power_factor=unpaved_power_factor,
            gravel_work_multiplier=gravel_work_multiplier,
            gravel_coast_speed_pct=gravel_coast_speed_pct,
        )

        # Calculate verbose metrics from trip (actual) and route (estimated with factor=1.0)
        verbose_metrics = calculate_verbose_metrics(trip_points, physics_params.max_coasting_speed)
        estimated_verbose_metrics = _calculate_estimated_verbose_metrics(smoothed, physics_params)

        # Step 2: Infer braking factor from ratio of actual to estimated braking scores
        # inferred_braking_factor < 1.0 means rider descends slower than physics predicts (more cautious)
        inferred_braking_factor = None
        if (verbose_metrics and estimated_verbose_metrics and
            verbose_metrics.braking_score is not None and
            estimated_verbose_metrics.braking_score is not None and
            estimated_verbose_metrics.braking_score > 0):
            inferred_braking_factor = verbose_metrics.braking_score / estimated_verbose_metrics.braking_score
            # Clamp to reasonable range [0.3, 1.5]
            inferred_braking_factor = max(0.3, min(1.5, inferred_braking_factor))

        # Step 3: Build final route_params
        # Note: We don't apply the inferred braking factor to the model because:
        # - The physics model already predicts times that are too slow (rider is faster)
        # - Applying braking factor < 1.0 would make descent predictions even slower
        # - The inferred factor is kept for reporting purposes only
        route_params = RiderParams(
            total_mass=mass_used,
            cda=params.cda,
            crr=params.crr,
            air_density=params.air_density,
            climbing_power=climbing_power_used,
            flat_power=flat_power_used,
            descending_power=descending_power_used,
            coasting_grade_threshold=params.coasting_grade_threshold,
            max_coasting_speed=params.max_coasting_speed,
            max_coasting_speed_unpaved=params.max_coasting_speed_unpaved,
            max_descent_speed=params.max_descent_speed,
            headwind=headwind_used,
            climb_threshold_grade=params.climb_threshold_grade,
            steep_descent_speed=params.steep_descent_speed,
            steep_descent_grade=params.steep_descent_grade,
            straight_descent_speed=params.straight_descent_speed,
            hairpin_speed=params.hairpin_speed,
            straight_curvature=params.straight_curvature,
            hairpin_curvature=params.hairpin_curvature,
            descent_braking_factor=1.0,  # Use physics-based model (inferred factor for reporting only)
            drivetrain_efficiency=params.drivetrain_efficiency,
            unpaved_power_factor=unpaved_power_factor,
            gravel_work_multiplier=gravel_work_multiplier,
            gravel_coast_speed_pct=gravel_coast_speed_pct,
        )

        # Step 4: Analyze with inferred braking factor
        analysis = analyze(smoothed, route_params)

        # Calculate hilliness with final params
        hilliness = calculate_hilliness(smoothed, route_params, route_points, max_grade_window_route, max_grade_smoothing)
        route_max_grade = hilliness.max_grade

        # Compare with trip - use physics-based moving_time with inferred braking factor
        comparison = compare_route_with_trip(
            smoothed,
            trip_points,
            route_params,
            analysis.moving_time.total_seconds(),
            analysis.estimated_work,
            route_elevation_gain=analysis.elevation_gain,
            trip_elevation_gain=trip_elevation_gain,
        )

        return TrainingResult(
            route=route,
            comparison=comparison,
            route_elevation_gain=analysis.elevation_gain,
            trip_elevation_gain=trip_elevation_gain,
            route_distance=analysis.total_distance,
            unpaved_pct=unpaved_pct,
            power_used=climbing_power_used,  # Use climbing power (from trip) for display
            mass_used=mass_used,
            elevation_scale_used=effective_scale,
            route_max_grade=route_max_grade,
            trip_max_grade=trip_max_grade,
            verbose_metrics=verbose_metrics,
            estimated_verbose_metrics=estimated_verbose_metrics,
            inferred_braking_factor=inferred_braking_factor,
        )

    except Exception as e:
        print(f"  Error analyzing {route.name}: {e}")
        return None


def run_training_analysis(
    training_data: list[TrainingRoute],
    params: RiderParams,
    smoothing_radius: float = 50.0,
    elevation_scale: float = 1.0,
    max_grade_window_route: float = DEFAULT_MAX_GRADE_WINDOW,
    max_grade_window_trip: float = 50.0,
    max_grade_smoothing: float = DEFAULT_MAX_GRADE_SMOOTHING,
) -> tuple[list[TrainingResult], TrainingSummary]:
    """Run analysis on all training routes and compute summary statistics."""
    results: list[TrainingResult] = []

    for route in training_data:
        print(f"Analyzing: {route.name}...")
        # Use per-route smoothing if specified, otherwise use default
        route_smoothing = route.smoothing if route.smoothing is not None else smoothing_radius
        result = analyze_training_route(
            route, params, route_smoothing, elevation_scale,
            max_grade_window_route=max_grade_window_route,
            max_grade_window_trip=max_grade_window_trip,
            max_grade_smoothing=max_grade_smoothing,
        )
        if result:
            results.append(result)

    if not results:
        return results, TrainingSummary(
            total_routes=0,
            total_distance_km=0,
            total_elevation_m=0,
            avg_time_error_pct=0,
            avg_work_error_pct=0,
            results_by_tag={},
            road_avg_time_error=None,
            gravel_avg_time_error=None,
            hilly_avg_time_error=None,
            flat_avg_time_error=None,
        )

    # Compute aggregate statistics
    total_distance = sum(r.route_distance for r in results)
    total_elevation = sum(r.route_elevation_gain for r in results)
    avg_time_error = sum(r.comparison.time_error_pct for r in results) / len(results)

    # Work error (only for routes with power data)
    work_errors = []
    for r in results:
        if r.comparison.actual_work and r.comparison.actual_work > 0:
            err = (r.comparison.predicted_work - r.comparison.actual_work) / r.comparison.actual_work * 100
            work_errors.append(err)
    avg_work_error = sum(work_errors) / len(work_errors) if work_errors else 0

    # Group by tags
    results_by_tag: dict[str, list[TrainingResult]] = {}
    for r in results:
        for tag in r.route.tags:
            if tag not in results_by_tag:
                results_by_tag[tag] = []
            results_by_tag[tag].append(r)

    def avg_time_error_for_tag(tag: str) -> float | None:
        tagged = results_by_tag.get(tag, [])
        if not tagged:
            return None
        return sum(r.comparison.time_error_pct for r in tagged) / len(tagged)

    summary = TrainingSummary(
        total_routes=len(results),
        total_distance_km=total_distance / 1000,
        total_elevation_m=total_elevation,
        avg_time_error_pct=avg_time_error,
        avg_work_error_pct=avg_work_error,
        results_by_tag=results_by_tag,
        road_avg_time_error=avg_time_error_for_tag("road"),
        gravel_avg_time_error=avg_time_error_for_tag("gravel"),
        hilly_avg_time_error=avg_time_error_for_tag("hilly"),
        flat_avg_time_error=avg_time_error_for_tag("flat"),
    )

    return results, summary


def format_training_summary(
    results: list[TrainingResult], summary: TrainingSummary, params: RiderParams,
    imperial: bool = False,
    verbose: bool = False,
) -> str:
    """Format training analysis results as a human-readable report."""
    lines = []

    # Unit conversion factors
    dist_factor = 0.621371 if imperial else 1.0
    dist_unit = "mi" if imperial else "km"
    elev_factor = 3.28084 if imperial else 1.0
    elev_unit = "ft" if imperial else "m"
    speed_factor = 0.621371 if imperial else 1.0
    speed_unit = "mph" if imperial else "km/h"

    lines.append("=" * 60)
    lines.append("TRAINING DATA ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Config
    coast_speed = params.max_coasting_speed * 3.6 * speed_factor
    lines.append(f"Model params: mass={params.total_mass}kg cda={params.cda} crr={params.crr}")
    lines.append(f"              max_coast={coast_speed:.0f}{speed_unit} (power from trip data per route)")
    lines.append("")

    # Overall stats
    total_dist = summary.total_distance_km * dist_factor
    total_elev = summary.total_elevation_m * elev_factor
    lines.append(f"Routes analyzed: {summary.total_routes}")
    lines.append(f"Total distance:  {total_dist:.0f} {dist_unit}")
    lines.append(f"Total elevation: {total_elev:.0f} {elev_unit}")
    lines.append("")

    # Calculate avg max grade error
    grade_errors = []
    for r in results:
        if r.trip_max_grade and r.trip_max_grade > 0:
            grade_err = (r.route_max_grade - r.trip_max_grade) / r.trip_max_grade * 100
            grade_errors.append(grade_err)
    avg_grade_error = sum(grade_errors) / len(grade_errors) if grade_errors else 0

    # Calculate avg inferred braking factor
    braking_factors = [r.inferred_braking_factor for r in results if r.inferred_braking_factor is not None]
    avg_braking_factor = sum(braking_factors) / len(braking_factors) if braking_factors else None

    # Error summary
    lines.append("PREDICTION ERRORS (positive = predicted too slow/high)")
    lines.append("-" * 50)
    lines.append(f"  Avg time error:       {summary.avg_time_error_pct:+.1f}%")
    lines.append(f"  Avg work error:       {summary.avg_work_error_pct:+.1f}%")
    lines.append(f"  Avg max grade error:  {avg_grade_error:+.1f}%")
    if avg_braking_factor is not None:
        lines.append(f"  Avg braking factor:   {avg_braking_factor:.2f} (1.0 = physics, <1 = cautious, n={len(braking_factors)})")
    lines.append("")

    # By terrain type
    lines.append("BY TERRAIN TYPE:")
    if summary.road_avg_time_error is not None:
        n = len(summary.results_by_tag.get("road", []))
        lines.append(f"  Road ({n} routes):   {summary.road_avg_time_error:+.1f}% time error")
    if summary.gravel_avg_time_error is not None:
        n = len(summary.results_by_tag.get("gravel", []))
        lines.append(f"  Gravel ({n} routes): {summary.gravel_avg_time_error:+.1f}% time error")
    if summary.hilly_avg_time_error is not None:
        n = len(summary.results_by_tag.get("hilly", []))
        lines.append(f"  Hilly ({n} routes):  {summary.hilly_avg_time_error:+.1f}% time error")
    if summary.flat_avg_time_error is not None:
        n = len(summary.results_by_tag.get("flat", []))
        lines.append(f"  Flat ({n} routes):   {summary.flat_avg_time_error:+.1f}% time error")
    lines.append("")

    # Per-route breakdown
    dist_suffix = "m" if imperial else "k"  # miles or km
    elev_suffix = "'" if imperial else "m"  # feet or meters
    lines.append("PER-ROUTE BREAKDOWN:")

    if verbose:
        # Verbose mode: same as standard plus power by terrain and braking score
        lines.append("-" * 280)
        lines.append(f"{'':34} {'---------------- Estimated ----------------':^52}{'----------------- Actual -----------------':^52}{'------------------ Diff ------------------':^58}")
        lines.append(f"{'Route':<22} {'Unpvd':>5} {'Pwr':>4}  {'Dist':>6} {'Elev':>5} {'Time':>4} {'Work':>4} {'Grade':>5} {'Climb':>5} {'Flat':>4} {'Desc':>4} {'Brk':>4}  {'Dist':>6} {'Elev':>5} {'Time':>4} {'Work':>4} {'Grade':>5} {'Climb':>5} {'Flat':>4} {'Desc':>4} {'Brk':>4}  {'Time':>6} {'Work':>6} {'Elev':>6} {'Grade':>6} {'Climb':>6} {'Flat':>6} {'Desc':>6} {'BrkF':>6}")
        lines.append("-" * 280)
    else:
        # Standard mode
        lines.append("-" * 164)
        lines.append(f"{'':34} {'------- Estimated -------':^34}{'-------- Actual --------':^34}{'--------- Diff ---------':^32}")
        lines.append(f"{'Route':<22} {'Unpvd':>5} {'Pwr':>4}  {'Dist':>6} {'Elev':>5} {'Time':>4} {'Work':>4} {'Grade':>5}  {'Dist':>6} {'Elev':>5} {'Time':>4} {'Work':>4} {'Grade':>5}  {'Time':>6} {'Work':>6} {'Elev':>6} {'Grade':>6}")
        lines.append("-" * 164)

    for r in results:
        # Estimated values
        est_dist = r.route_distance / 1000 * dist_factor
        est_elev = r.route_elevation_gain * elev_factor
        est_time = r.comparison.predicted_time / 3600
        est_work = r.comparison.predicted_work / 1000
        est_max_grade = r.route_max_grade

        # Actual values
        act_dist = r.comparison.trip_distance / 1000 * dist_factor
        act_elev = (r.trip_elevation_gain if r.trip_elevation_gain else 0) * elev_factor
        act_time = r.comparison.actual_moving_time / 3600
        act_work = r.comparison.actual_work / 1000 if r.comparison.actual_work else 0
        act_max_grade = r.trip_max_grade if r.trip_max_grade is not None else 0

        # Differences
        time_diff = r.comparison.time_error_pct
        work_diff = ((r.comparison.predicted_work - r.comparison.actual_work) / r.comparison.actual_work * 100
                     if r.comparison.actual_work and r.comparison.actual_work > 0 else 0)
        elev_diff = ((est_elev - act_elev) / act_elev * 100
                     if act_elev > 0 else 0)
        grade_diff = ((est_max_grade - act_max_grade) / act_max_grade * 100
                      if act_max_grade > 0 else 0)

        name = r.route.name[:21]

        if verbose:
            # Get verbose metrics (actual from trip, estimated from route)
            vm = r.verbose_metrics
            evm = r.estimated_verbose_metrics

            # Estimated power values (shorter format to fit)
            est_pwr_climb = f"{evm.avg_power_climbing:>4.0f}" if evm and evm.avg_power_climbing else "   -"
            est_pwr_flat = f"{evm.avg_power_flat:>3.0f}" if evm and evm.avg_power_flat else "  -"
            est_pwr_desc = f"{evm.avg_power_descending:>3.0f}" if evm and evm.avg_power_descending is not None else "  -"
            est_brake = f"{evm.braking_score:>4.2f}" if evm and evm.braking_score else "   -"

            # Actual power values
            act_pwr_climb = f"{vm.avg_power_climbing:>4.0f}" if vm and vm.avg_power_climbing else "   -"
            act_pwr_flat = f"{vm.avg_power_flat:>3.0f}" if vm and vm.avg_power_flat else "  -"
            act_pwr_desc = f"{vm.avg_power_descending:>3.0f}" if vm and vm.avg_power_descending is not None else "  -"
            act_brake = f"{vm.braking_score:>4.2f}" if vm and vm.braking_score else "   -"

            # Diff for verbose metrics (actual - estimated, as percentage)
            if evm and vm and evm.avg_power_climbing and vm.avg_power_climbing:
                climb_diff = (vm.avg_power_climbing - evm.avg_power_climbing) / evm.avg_power_climbing * 100
                climb_diff_str = f"{climb_diff:>+5.0f}%"
            else:
                climb_diff_str = "     -"

            if evm and vm and evm.avg_power_flat and vm.avg_power_flat:
                flat_diff = (vm.avg_power_flat - evm.avg_power_flat) / evm.avg_power_flat * 100
                flat_diff_str = f"{flat_diff:>+5.0f}%"
            else:
                flat_diff_str = "     -"

            if evm and vm and evm.avg_power_descending is not None and vm.avg_power_descending is not None and evm.avg_power_descending > 0:
                desc_diff = (vm.avg_power_descending - evm.avg_power_descending) / evm.avg_power_descending * 100
                desc_diff_str = f"{desc_diff:>+5.0f}%"
            else:
                desc_diff_str = "     -"

            # Show inferred braking factor (actual/estimated) instead of percentage
            # <1.0 means rider descends slower than physics predicts (more braking)
            if r.inferred_braking_factor is not None:
                brake_diff_str = f"{r.inferred_braking_factor:>5.2f}x"
            else:
                brake_diff_str = "     -"

            lines.append(
                f"{name:<22} {r.unpaved_pct:>4.0f}% {r.power_used:>3.0f}W "
                f" {est_dist:>5.0f}{dist_suffix} {est_elev:>4.0f}{elev_suffix} {est_time:>4.1f}h {est_work:>4.0f}k {est_max_grade:>4.1f}% {est_pwr_climb}W {est_pwr_flat}W {est_pwr_desc}W {est_brake} "
                f" {act_dist:>5.0f}{dist_suffix} {act_elev:>4.0f}{elev_suffix} {act_time:>4.1f}h {act_work:>4.0f}k {act_max_grade:>4.1f}% {act_pwr_climb}W {act_pwr_flat}W {act_pwr_desc}W {act_brake} "
                f" {time_diff:>+5.1f}% {work_diff:>+5.1f}% {elev_diff:>+5.1f}% {grade_diff:>+5.1f}% {climb_diff_str} {flat_diff_str} {desc_diff_str} {brake_diff_str}"
            )
        else:
            lines.append(
                f"{name:<22} {r.unpaved_pct:>4.0f}% {r.power_used:>3.0f}W "
                f" {est_dist:>5.0f}{dist_suffix} {est_elev:>4.0f}{elev_suffix} {est_time:>4.1f}h {est_work:>4.0f}k {est_max_grade:>4.1f}% "
                f" {act_dist:>5.0f}{dist_suffix} {act_elev:>4.0f}{elev_suffix} {act_time:>4.1f}h {act_work:>4.0f}k {act_max_grade:>4.1f}% "
                f" {time_diff:>+5.1f}% {work_diff:>+5.1f}% {elev_diff:>+5.1f}% {grade_diff:>+5.1f}%"
            )

    lines.append("")

    # Tags summary
    all_tags = set()
    for r in results:
        all_tags.update(r.route.tags)

    if all_tags:
        lines.append(f"Tags in dataset: {', '.join(sorted(all_tags))}")
        lines.append("")

    return "\n".join(lines)
