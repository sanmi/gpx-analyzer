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
    elevation_scale_used: float = 1.0  # Scale factor applied to match trip elevation
    route_max_grade: float = 0.0  # Max grade from route (%)
    trip_max_grade: float | None = None  # Max grade from trip (%)


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
    has_distance = any(p.distance > 0 for p in points)

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

        # Use per-route power if specified, otherwise calculate from trip, otherwise use default
        if route.avg_watts is not None:
            power_used = route.avg_watts
        else:
            # Calculate avg power from trip data points
            _, trip_avg_power, has_power = _calculate_actual_work(trip_points)
            if has_power and trip_avg_power:
                power_used = trip_avg_power
            else:
                power_used = params.assumed_avg_power

        headwind_used = route.headwind / 3.6 if route.headwind is not None else params.headwind
        route_params = RiderParams(
            total_mass=params.total_mass,
            cda=params.cda,
            crr=params.crr,
            air_density=params.air_density,
            assumed_avg_power=power_used,
            coasting_grade_threshold=params.coasting_grade_threshold,
            max_coasting_speed=params.max_coasting_speed,
            max_coasting_speed_unpaved=params.max_coasting_speed_unpaved,
            headwind=headwind_used,
            climb_power_factor=params.climb_power_factor,
            climb_threshold_grade=params.climb_threshold_grade,
            steep_descent_speed=params.steep_descent_speed,
            steep_descent_grade=params.steep_descent_grade,
        )

        if len(route_points) < 2 or len(trip_points) < 2:
            print(f"  Skipping {route.name}: insufficient data points")
            return None

        # Get trip elevation gain from metadata
        trip_elevation_gain = trip_metadata.get("elevation_gain")

        # First pass: smooth with base elevation_scale to get route elevation
        smoothed = smooth_elevations(route_points, smoothing_radius, elevation_scale)

        # Calculate elevation correction factor if we have trip data
        effective_scale = elevation_scale
        if use_trip_elevation and trip_elevation_gain and trip_elevation_gain > 0:
            # Calculate route elevation gain after smoothing
            route_gain_smoothed = _calculate_elevation_gain(smoothed)
            if route_gain_smoothed > 0:
                # Scale factor to make route elevation match trip elevation
                correction_factor = trip_elevation_gain / route_gain_smoothed
                # Re-smooth with corrected scale
                effective_scale = elevation_scale * correction_factor
                smoothed = smooth_elevations(route_points, smoothing_radius, effective_scale)

        analysis = analyze(smoothed, route_params)

        # Compare with trip
        comparison = compare_route_with_trip(
            smoothed,
            trip_points,
            route_params,
            analysis.estimated_moving_time_at_power.total_seconds(),
            analysis.estimated_work,
            route_elevation_gain=analysis.elevation_gain,
            trip_elevation_gain=trip_elevation_gain,
        )

        # Calculate unpaved percentage
        unpaved_pct = route_metadata.get("unpaved_pct", 0) or 0

        # Calculate max grades
        hilliness = calculate_hilliness(smoothed, route_params, route_points, max_grade_window_route, max_grade_smoothing)
        route_max_grade = hilliness.max_grade
        trip_max_grade = _calculate_trip_max_grade(trip_points, max_grade_window_trip)

        return TrainingResult(
            route=route,
            comparison=comparison,
            route_elevation_gain=analysis.elevation_gain,
            trip_elevation_gain=trip_elevation_gain,
            route_distance=analysis.total_distance,
            unpaved_pct=unpaved_pct,
            power_used=power_used,
            elevation_scale_used=effective_scale,
            route_max_grade=route_max_grade,
            trip_max_grade=trip_max_grade,
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
        result = analyze_training_route(
            route, params, smoothing_radius, elevation_scale,
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
    lines.append(f"              power={params.assumed_avg_power}W max_coast={coast_speed:.0f}{speed_unit}")
    lines.append("")

    # Overall stats
    total_dist = summary.total_distance_km * dist_factor
    total_elev = summary.total_elevation_m * elev_factor
    lines.append(f"Routes analyzed: {summary.total_routes}")
    lines.append(f"Total distance:  {total_dist:.0f} {dist_unit}")
    lines.append(f"Total elevation: {total_elev:.0f} {elev_unit}")
    lines.append("")

    # Error summary
    lines.append("PREDICTION ERRORS (positive = predicted too slow/high)")
    lines.append("-" * 50)
    lines.append(f"  Avg time error:  {summary.avg_time_error_pct:+.1f}%")
    lines.append(f"  Avg work error:  {summary.avg_work_error_pct:+.1f}%")
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
    lines.append("-" * 130)
    lines.append(f"{'':28} {'------- Estimated -------':>27} {'-------- Actual --------':>27} {'--- Diff ---':>14} {'- Max Grade -':>15}")
    lines.append(f"{'Route':<22} {'Unpvd':>5} {'Dist':>7} {'Elev':>6} {'Time':>6} {'Work':>6} {'Dist':>7} {'Elev':>6} {'Time':>6} {'Work':>6} {'Time':>7} {'Work':>7} {'Est':>7} {'Act':>7}")
    lines.append("-" * 130)

    for r in results:
        # Estimated values
        est_dist = r.route_distance / 1000 * dist_factor
        est_elev = r.route_elevation_gain * elev_factor
        est_time = r.comparison.predicted_time / 3600
        est_work = r.comparison.predicted_work / 1000

        # Actual values
        act_dist = r.comparison.trip_distance / 1000 * dist_factor
        act_elev = (r.trip_elevation_gain if r.trip_elevation_gain else 0) * elev_factor
        act_time = r.comparison.actual_moving_time / 3600
        act_work = r.comparison.actual_work / 1000 if r.comparison.actual_work else 0

        # Differences
        time_diff = r.comparison.time_error_pct
        work_diff = ((r.comparison.predicted_work - r.comparison.actual_work) / r.comparison.actual_work * 100
                     if r.comparison.actual_work and r.comparison.actual_work > 0 else 0)

        # Max grades
        est_max_grade = r.route_max_grade
        act_max_grade = r.trip_max_grade if r.trip_max_grade is not None else 0

        name = r.route.name[:21]
        lines.append(
            f"{name:<22} {r.unpaved_pct:>4.0f}% "
            f"{est_dist:>6.0f}{dist_suffix} {est_elev:>5.0f}{elev_suffix} {est_time:>5.1f}h {est_work:>5.0f}k "
            f"{act_dist:>6.0f}{dist_suffix} {act_elev:>5.0f}{elev_suffix} {act_time:>5.1f}h {act_work:>5.0f}k "
            f"{time_diff:>+6.1f}% {work_diff:>+6.1f}% "
            f"{est_max_grade:>6.1f}% {act_max_grade:>6.1f}%"
        )

    lines.append("")

    # Tags summary
    all_tags = set()
    for r in results:
        all_tags.update(r.route.tags)

    if all_tags:
        lines.append(f"Tags in dataset: {', '.join(sorted(all_tags))}")
        lines.append("")

    # Elevation quality analysis
    routes_with_scale = [r for r in results if r.trip_elevation_gain and r.trip_elevation_gain > 0]
    if routes_with_scale:
        lines.append("ELEVATION DATA QUALITY:")
        lines.append("-" * 50)

        scales = [r.elevation_scale_used for r in routes_with_scale]
        avg_scale = sum(scales) / len(scales)
        lines.append(f"  Avg elevation scale: {avg_scale:.2f} (1.00 = perfect match)")

        # Flag routes with significant elevation discrepancy (>15% adjustment)
        inaccurate_routes = [r for r in routes_with_scale if abs(r.elevation_scale_used - 1.0) > 0.15]
        if inaccurate_routes:
            lines.append(f"  Routes with >15% elevation discrepancy: {len(inaccurate_routes)}/{len(routes_with_scale)}")
            for r in inaccurate_routes:
                pct_off = (r.elevation_scale_used - 1.0) * 100
                direction = "inflated" if pct_off < 0 else "understated"
                lines.append(f"    - {r.route.name}: scale={r.elevation_scale_used:.2f} (route elevation {direction} by {abs(pct_off):.0f}%)")
            lines.append("")
            lines.append("  TIP: Route elevation is auto-scaled using RideWithGPS API data in single-route mode.")
        else:
            lines.append("  All routes have consistent elevation data (<15% discrepancy)")

    return "\n".join(lines)
