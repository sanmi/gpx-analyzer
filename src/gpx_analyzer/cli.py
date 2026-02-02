import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from gpx_analyzer.distance import haversine_distance

from gpx_analyzer import __version_date__, get_git_hash
from gpx_analyzer.analyzer import analyze, calculate_hilliness, GRADE_LABELS, DEFAULT_MAX_GRADE_WINDOW, DEFAULT_MAX_GRADE_SMOOTHING
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.parser import parse_gpx
from gpx_analyzer.compare import compare_route_with_trip, format_comparison_report
from gpx_analyzer.ridewithgps import (
    _load_config,
    get_collection_route_ids,
    get_gpx,
    get_route_with_surface,
    get_trip_data,
    is_ridewithgps_collection_url,
    is_ridewithgps_trip_url,
    is_ridewithgps_url,
)
from gpx_analyzer.smoothing import smooth_elevations
from gpx_analyzer.training import (
    format_training_summary,
    load_training_data,
    run_training_analysis,
)
from gpx_analyzer.optimize import optimize_parameters, save_optimized_config

# Default values for CLI options
DEFAULTS = {
    "mass": 85.0,
    "cda": 0.35,
    "crr": 0.005,
    "power": 150.0,
    "coasting_grade": -5.0,
    "max_coast_speed": 48.0,
    "max_coast_speed_unpaved": 24.0,
    "climb_power_factor": 1.5,
    "climb_threshold_grade": 4.0,
    "steep_descent_speed": 18.0,
    "steep_descent_grade": -8.0,
    "straight_descent_speed": 45.0,
    "hairpin_speed": 18.0,
    "straight_curvature": 0.3,
    "hairpin_curvature": 3.0,
    "smoothing": 50.0,
    "elevation_scale": 1.0,
    "headwind": 0.0,
    "max_grade_window_route": 150.0,
    "max_grade_window_trip": 50.0,
    "max_grade_smoothing": 150.0,
}


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    """Build argument parser with defaults from config file."""
    if config is None:
        config = {}

    def get_default(key: str) -> float:
        return config.get(key, DEFAULTS[key])

    parser = argparse.ArgumentParser(
        description="Analyze GPX bike routes using physics-based power estimation.",
        epilog="""Examples:
  gpx-analyzer route.gpx                      Analyze a local GPX file
  gpx-analyzer https://ridewithgps.com/routes/123  Analyze a RideWithGPS route
  gpx-analyzer --collection URL               Analyze all routes in a collection
  gpx-analyzer --training data.json           Batch compare predictions vs actual rides
  gpx-analyzer --optimize data.json           Tune model parameters from training data

Config file: ~/.config/gpx-analyzer/gpx-analyzer.json or ./gpx-analyzer.json
  Set default parameters and RideWithGPS API credentials.

See README.md for detailed parameter descriptions.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"gpx-analyzer {__version_date__} ({get_git_hash()})"
    )
    parser.add_argument("gpx_file", nargs="?", help="Path to GPX file or RideWithGPS route URL")
    parser.add_argument(
        "--training",
        type=str,
        default=None,
        metavar="FILE",
        help="Run batch analysis on training data JSON file instead of single route.",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default=None,
        metavar="FILE",
        help="Run parameter optimization on training data JSON file.",
    )
    parser.add_argument(
        "--optimize-mode",
        type=str,
        default="physics",
        choices=["physics", "grade"],
        help="Optimization mode: 'physics' (time/work) or 'grade' (max grade). Default: physics",
    )
    parser.add_argument(
        "--optimize-output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save optimized parameters to JSON file (use with --optimize).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        metavar="URL",
        help="Analyze all routes in a RideWithGPS collection.",
    )
    parser.add_argument(
        "--imperial",
        action="store_true",
        help="Display output in imperial units (miles, feet) instead of metric.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=get_default("mass"),
        help=f"Total mass of rider + bike in kg (default: {DEFAULTS['mass']})",
    )
    parser.add_argument(
        "--cda",
        type=float,
        default=get_default("cda"),
        help=f"Drag coefficient * frontal area in m² (default: {DEFAULTS['cda']})",
    )
    parser.add_argument(
        "--crr",
        type=float,
        default=get_default("crr"),
        help=f"Rolling resistance coefficient (default: {DEFAULTS['crr']})",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=get_default("power"),
        help=f"Assumed average power output in watts (default: {DEFAULTS['power']})",
    )
    parser.add_argument(
        "--coasting-grade",
        type=float,
        default=get_default("coasting_grade"),
        help=f"Grade in degrees at which rider fully coasts (default: {DEFAULTS['coasting_grade']})",
    )
    parser.add_argument(
        "--max-coast-speed",
        type=float,
        default=get_default("max_coast_speed"),
        help=f"Maximum coasting speed in km/h (default: {DEFAULTS['max_coast_speed']})",
    )
    parser.add_argument(
        "--max-coast-speed-unpaved",
        type=float,
        default=get_default("max_coast_speed_unpaved"),
        help=f"Maximum coasting speed on unpaved surfaces in km/h (default: {DEFAULTS['max_coast_speed_unpaved']})",
    )
    parser.add_argument(
        "--climb-power-factor",
        type=float,
        default=get_default("climb_power_factor"),
        help=f"Power multiplier on steep climbs, e.g. 1.5 = 50%% more power (default: {DEFAULTS['climb_power_factor']})",
    )
    parser.add_argument(
        "--climb-threshold-grade",
        type=float,
        default=get_default("climb_threshold_grade"),
        help=f"Grade in degrees at which full climb power factor is reached (default: {DEFAULTS['climb_threshold_grade']})",
    )
    parser.add_argument(
        "--steep-descent-speed",
        type=float,
        default=get_default("steep_descent_speed"),
        help=f"Max speed on steep descents in km/h (default: {DEFAULTS['steep_descent_speed']})",
    )
    parser.add_argument(
        "--steep-descent-grade",
        type=float,
        default=get_default("steep_descent_grade"),
        help=f"Grade in degrees where steep descent speed applies (default: {DEFAULTS['steep_descent_grade']})",
    )
    parser.add_argument(
        "--straight-descent-speed",
        type=float,
        default=get_default("straight_descent_speed"),
        help=f"Max speed on straight descents in km/h (default: {DEFAULTS['straight_descent_speed']})",
    )
    parser.add_argument(
        "--hairpin-speed",
        type=float,
        default=get_default("hairpin_speed"),
        help=f"Max speed through hairpin turns in km/h (default: {DEFAULTS['hairpin_speed']})",
    )
    parser.add_argument(
        "--straight-curvature",
        type=float,
        default=get_default("straight_curvature"),
        help=f"Curvature threshold for straight sections in deg/m (default: {DEFAULTS['straight_curvature']})",
    )
    parser.add_argument(
        "--hairpin-curvature",
        type=float,
        default=get_default("hairpin_curvature"),
        help=f"Curvature threshold for hairpin turns in deg/m (default: {DEFAULTS['hairpin_curvature']})",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=get_default("smoothing"),
        help=f"Elevation smoothing radius in meters (default: {DEFAULTS['smoothing']})",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable elevation smoothing",
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=get_default("elevation_scale"),
        help=f"Scale factor for elevation changes (default: {DEFAULTS['elevation_scale']}). Use <1 to reduce overestimated GPS elevation.",
    )
    parser.add_argument(
        "--headwind",
        type=float,
        default=get_default("headwind"),
        help=f"Headwind speed in km/h (default: {DEFAULTS['headwind']}). Positive = into wind, negative = tailwind.",
    )
    parser.add_argument(
        "--compare-trip",
        type=str,
        default=None,
        help="RideWithGPS trip URL to compare predictions against actual ride data.",
    )
    parser.add_argument(
        "--no-api-elevation",
        action="store_true",
        help="Disable automatic elevation scaling based on RideWithGPS API elevation data.",
    )
    return parser


def format_duration(td) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def calculate_elevation_gain(points: list[TrackPoint]) -> float:
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
HIGH_NOISE_SMOOTHING_RADIUS = 200.0  # meters, used when DEM is noisy


def is_high_noise_dem(raw_gain: float, api_gain: float) -> bool:
    """Detect if DEM data has high noise based on raw/API elevation ratio.

    When the raw GPS track elevation gain is much higher than the API's
    DEM-corrected elevation, it indicates noisy elevation data. In such cases,
    aggressive smoothing without API scaling produces better results than
    trusting the (also noisy) API elevation value.

    Args:
        raw_gain: Elevation gain from raw track points (no smoothing)
        api_gain: Elevation gain from RideWithGPS API (DEM-corrected)

    Returns:
        True if the DEM data appears to have high noise (ratio > 1.8)
    """
    if api_gain <= 0 or raw_gain <= 0:
        return False
    ratio = raw_gain / api_gain
    return ratio > HIGH_NOISE_RATIO_THRESHOLD


def calculate_surface_breakdown(points: list[TrackPoint]) -> tuple[float, float] | None:
    """Calculate distance on paved vs unpaved surfaces.

    Args:
        points: List of TrackPoints with optional unpaved flag

    Returns (paved_distance_m, unpaved_distance_m) or None if no surface data.
    """
    if not points or len(points) < 2:
        return None

    # Check if any points have surface data (crr set means we have RWGPS surface info)
    has_surface_data = any(pt.crr is not None for pt in points)
    if not has_surface_data:
        return None

    paved_dist = 0.0
    unpaved_dist = 0.0

    for i in range(1, len(points)):
        pt_a = points[i - 1]
        pt_b = points[i]
        dist = haversine_distance(pt_a.lat, pt_a.lon, pt_b.lat, pt_b.lon)

        # Use the destination point's unpaved flag to classify the segment
        if pt_b.unpaved:
            unpaved_dist += dist
        else:
            paved_dist += dist

    return paved_dist, unpaved_dist


@dataclass
class CollectionRouteResult:
    """Result of analyzing a single route in a collection."""
    name: str
    route_id: int
    distance: float  # meters
    elevation_gain: float  # meters
    elevation_scale: float
    unpaved_pct: float
    hilliness_score: float  # m/km
    steepness_score: float  # effort-weighted avg climbing grade %
    estimated_time_hours: float
    estimated_work_kj: float
    avg_speed_kmh: float


def analyze_collection(
    route_ids: list[int],
    params: RiderParams,
    smoothing_radius: float,
    elevation_scale: float,
    max_grade_window: float = DEFAULT_MAX_GRADE_WINDOW,
    max_grade_smoothing: float = DEFAULT_MAX_GRADE_SMOOTHING,
) -> list[CollectionRouteResult]:
    """Analyze all routes in a collection."""
    results = []

    for route_id in route_ids:
        route_url = f"https://ridewithgps.com/routes/{route_id}"
        print(f"Analyzing route {route_id}...", end=" ", flush=True)

        try:
            points, route_metadata = get_route_with_surface(route_url, params.crr)

            if len(points) < 2:
                print("skipped (too few points)")
                continue

            # Calculate API-based elevation scale factor
            api_elevation_scale = 1.0
            api_elevation_gain = route_metadata.get("elevation_gain") if route_metadata else None

            # Calculate raw elevation gain to detect high-noise DEM
            raw_gain = calculate_elevation_gain(points)

            # Check for high-noise DEM data
            if api_elevation_gain and is_high_noise_dem(raw_gain, api_elevation_gain):
                # High noise: use aggressive smoothing without API scaling
                unscaled_points = smooth_elevations(points, HIGH_NOISE_SMOOTHING_RADIUS, 1.0)
                points = unscaled_points
            else:
                # Normal: smooth and scale to API elevation
                unscaled_points = smooth_elevations(points, smoothing_radius, 1.0)

                if api_elevation_gain and api_elevation_gain > 0:
                    smoothed_gain = calculate_elevation_gain(unscaled_points)
                    if smoothed_gain > 0:
                        api_elevation_scale = api_elevation_gain / smoothed_gain

                # Apply smoothing with combined scale factor
                effective_scale = elevation_scale * api_elevation_scale
                if smoothing_radius > 0 or effective_scale != 1.0:
                    points = smooth_elevations(points, smoothing_radius, effective_scale)

            analysis = analyze(points, params)
            hilliness = calculate_hilliness(points, params, unscaled_points, max_grade_window, max_grade_smoothing)

            # Get unpaved percentage from API metadata
            unpaved_pct = route_metadata.get("unpaved_pct", 0) if route_metadata else 0

            route_name = route_metadata.get("name", f"Route {route_id}") if route_metadata else f"Route {route_id}"

            results.append(CollectionRouteResult(
                name=route_name,
                route_id=route_id,
                distance=analysis.total_distance,
                elevation_gain=analysis.elevation_gain,
                elevation_scale=api_elevation_scale,
                unpaved_pct=unpaved_pct,
                hilliness_score=hilliness.hilliness_score,
                steepness_score=hilliness.steepness_score,
                estimated_time_hours=analysis.estimated_moving_time_at_power.total_seconds() / 3600,
                estimated_work_kj=analysis.estimated_work / 1000,
                avg_speed_kmh=analysis.avg_speed * 3.6,
            ))
            print("done")

        except Exception as e:
            print(f"error: {e}")
            continue

    return results


def format_collection_summary(
    results: list[CollectionRouteResult],
    collection_name: str | None,
    params: RiderParams,
    imperial: bool = False,
) -> str:
    """Format collection analysis results as a human-readable report."""
    lines = []

    # Unit conversion factors
    dist_factor = 0.621371 if imperial else 1.0
    dist_unit = "mi" if imperial else "km"
    elev_factor = 3.28084 if imperial else 1.0
    elev_unit = "ft" if imperial else "m"
    speed_factor = 0.621371 if imperial else 1.0
    speed_unit = "mph" if imperial else "km/h"

    lines.append("=" * 95)
    lines.append(f"COLLECTION ANALYSIS: {collection_name or 'Unnamed'}")
    lines.append("=" * 95)
    lines.append("")

    # Config
    coast_speed = params.max_coasting_speed * 3.6 * speed_factor
    lines.append(f"Model params: mass={params.total_mass}kg cda={params.cda} crr={params.crr}")
    lines.append(f"              power={params.assumed_avg_power}W max_coast={coast_speed:.0f}{speed_unit}")
    lines.append("")

    # Totals
    total_distance = sum(r.distance for r in results) / 1000 * dist_factor
    total_elevation = sum(r.elevation_gain for r in results) * elev_factor
    total_time = sum(r.estimated_time_hours for r in results)
    total_work = sum(r.estimated_work_kj for r in results)

    lines.append(f"Routes analyzed: {len(results)}")
    lines.append(f"Total distance:  {total_distance:.0f} {dist_unit}")
    lines.append(f"Total elevation: {total_elevation:.0f} {elev_unit}")
    lines.append(f"Total time:      {total_time:.1f} hours")
    lines.append(f"Total work:      {total_work:.0f} kJ")
    lines.append("")

    # Per-route breakdown
    lines.append("PER-ROUTE BREAKDOWN:")
    lines.append("-" * 95)
    dist_hdr = "Dist" + dist_unit[0]
    elev_hdr = "Elev"
    lines.append(f"{'Route':<30} {'Time':>6} {'Work':>6} {dist_hdr:>7} {elev_hdr:>6} {'Hilly':>5} {'Steep':>5} {'Speed':>6} {'Unpvd':>5} {'EScl':>5}")
    lines.append("-" * 95)

    for r in results:
        name = r.name[:29]
        dist = r.distance / 1000 * dist_factor
        elev = r.elevation_gain * elev_factor
        speed = r.avg_speed_kmh * speed_factor
        elev_suffix = "'" if imperial else "m"
        lines.append(
            f"{name:<30} "
            f"{r.estimated_time_hours:>5.1f}h {r.estimated_work_kj:>5.0f}k "
            f"{dist:>6.0f}{dist_unit[0]} {elev:>5.0f}{elev_suffix} "
            f"{r.hilliness_score:>5.0f} {r.steepness_score:>4.1f}% {speed:>5.1f} {r.unpaved_pct:>4.0f}% {r.elevation_scale:>5.2f}"
        )

    lines.append("-" * 95)
    elev_short = "'" if imperial else "m"
    lines.append(
        f"{'TOTAL':<30} "
        f"{total_time:>5.1f}h {total_work:>5.0f}k "
        f"{total_distance:>6.0f}{dist_unit[0]} {total_elevation:>5.0f}{elev_short}"
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    config = _load_config()
    parser = build_parser(config)
    args = parser.parse_args(argv)

    params = RiderParams(
        total_mass=args.mass,
        cda=args.cda,
        crr=args.crr,
        assumed_avg_power=args.power,
        coasting_grade_threshold=args.coasting_grade,
        max_coasting_speed=args.max_coast_speed / 3.6,
        max_coasting_speed_unpaved=args.max_coast_speed_unpaved / 3.6,
        headwind=args.headwind / 3.6,
        climb_power_factor=args.climb_power_factor,
        climb_threshold_grade=args.climb_threshold_grade,
        steep_descent_speed=args.steep_descent_speed / 3.6,
        steep_descent_grade=args.steep_descent_grade,
        straight_descent_speed=args.straight_descent_speed / 3.6,
        hairpin_speed=args.hairpin_speed / 3.6,
        straight_curvature=args.straight_curvature,
        hairpin_curvature=args.hairpin_curvature,
    )

    # Optimization mode
    if args.optimize:
        optimize_path = Path(args.optimize)
        if not optimize_path.exists():
            print(f"Error: Training data file not found: {args.optimize}", file=sys.stderr)
            sys.exit(1)

        mode_desc = "physics (time & work)" if args.optimize_mode == "physics" else "grade (max grade)"
        print("=" * 60)
        print(f"PARAMETER OPTIMIZATION: {mode_desc}")
        print("=" * 60)
        print("")

        try:
            result = optimize_parameters(
                training_file=optimize_path,
                default_power=args.power,
                default_mass=args.mass,
                max_iterations=100,
                verbose=True,
                mode=args.optimize_mode,
            )

            if args.optimize_output:
                save_optimized_config(result, args.optimize_output)
                print(f"\nOptimized config saved to: {args.optimize_output}")

        except Exception as e:
            print(f"Error during optimization: {e}", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    # Training mode
    if args.training:
        training_path = Path(args.training)
        if not training_path.exists():
            print(f"Error: Training data file not found: {args.training}", file=sys.stderr)
            sys.exit(1)

        try:
            training_data = load_training_data(training_path)
        except Exception as e:
            print(f"Error loading training data: {e}", file=sys.stderr)
            sys.exit(1)

        if not training_data:
            print("Error: No routes found in training data.", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(training_data)} routes from {args.training}")
        print("")

        smoothing_radius = 0.0 if args.no_smoothing else args.smoothing
        max_grade_window_route = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])
        max_grade_window_trip = config.get("max_grade_window_trip", DEFAULTS["max_grade_window_trip"])
        max_grade_smoothing = config.get("max_grade_smoothing", DEFAULTS["max_grade_smoothing"])
        results, summary = run_training_analysis(
            training_data, params, smoothing_radius, args.elevation_scale,
            max_grade_window_route=max_grade_window_route,
            max_grade_window_trip=max_grade_window_trip,
            max_grade_smoothing=max_grade_smoothing,
        )

        print("")
        print(format_training_summary(results, summary, params, args.imperial))
        return

    # Collection mode
    if args.collection:
        if not is_ridewithgps_collection_url(args.collection):
            print(f"Error: Invalid RideWithGPS collection URL: {args.collection}", file=sys.stderr)
            sys.exit(1)

        try:
            route_ids, collection_name = get_collection_route_ids(args.collection)
        except Exception as e:
            print(f"Error fetching collection: {e}", file=sys.stderr)
            sys.exit(1)

        if not route_ids:
            print("Error: No routes found in collection.", file=sys.stderr)
            sys.exit(1)

        print(f"Analyzing {len(route_ids)} routes from collection: {collection_name or 'Unnamed'}")
        print("")

        smoothing_radius = 0.0 if args.no_smoothing else args.smoothing
        max_grade_window_route = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])
        max_grade_smoothing = config.get("max_grade_smoothing", DEFAULTS["max_grade_smoothing"])
        collection_results = analyze_collection(
            route_ids, params, smoothing_radius, args.elevation_scale, max_grade_window_route, max_grade_smoothing
        )

        print("")
        print(format_collection_summary(collection_results, collection_name, params, args.imperial))
        return

    # Single route mode - require gpx_file
    if not args.gpx_file:
        print("Error: gpx_file is required (or use --training/--collection for batch mode)", file=sys.stderr)
        sys.exit(1)

    route_metadata = None
    if is_ridewithgps_url(args.gpx_file):
        try:
            points, route_metadata = get_route_with_surface(args.gpx_file, args.crr)
        except Exception as e:
            print(f"Error downloading from RideWithGPS: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        gpx_path = args.gpx_file
        try:
            points = parse_gpx(gpx_path)
        except FileNotFoundError:
            print(f"Error: File not found: {gpx_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing GPX file: {e}", file=sys.stderr)
            sys.exit(1)

    if len(points) < 2:
        print("Error: GPX file contains fewer than 2 track points.", file=sys.stderr)
        sys.exit(1)

    smoothing_radius = 0.0 if args.no_smoothing else args.smoothing

    # Calculate API-based elevation scale factor for RideWithGPS routes
    api_elevation_scale = 1.0
    api_elevation_gain = None
    raw_elevation_gain = None
    high_noise_detected = False

    if (route_metadata
        and route_metadata.get("elevation_gain")
        and not args.no_api_elevation):
        api_elevation_gain = route_metadata["elevation_gain"]
        # Calculate raw elevation gain before any smoothing
        raw_elevation_gain = calculate_elevation_gain(points)

        # Check for high-noise DEM data
        if is_high_noise_dem(raw_elevation_gain, api_elevation_gain):
            # High noise: use aggressive smoothing without API scaling
            high_noise_detected = True
            unscaled_points = smooth_elevations(points, HIGH_NOISE_SMOOTHING_RADIUS, 1.0)
            points = unscaled_points
        else:
            # Normal: smooth and scale to API elevation
            unscaled_points = smooth_elevations(points, smoothing_radius, 1.0)
            if raw_elevation_gain > 0:
                smoothed_gain = calculate_elevation_gain(unscaled_points)
                if smoothed_gain > 0:
                    api_elevation_scale = api_elevation_gain / smoothed_gain

            # Apply smoothing with combined scale factor
            effective_scale = args.elevation_scale * api_elevation_scale
            if smoothing_radius > 0 or effective_scale != 1.0:
                points = smooth_elevations(points, smoothing_radius, effective_scale)
    else:
        # No API elevation data - just smooth without scaling
        unscaled_points = smooth_elevations(points, smoothing_radius, 1.0)
        if smoothing_radius > 0:
            points = unscaled_points

    result = analyze(points, params)
    max_grade_window_route = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])
    max_grade_smoothing = config.get("max_grade_smoothing", DEFAULTS["max_grade_smoothing"])
    hilliness = calculate_hilliness(points, params, unscaled_points, max_grade_window_route, max_grade_smoothing)

    # Unit conversion factors
    if args.imperial:
        dist_factor = 0.621371
        dist_unit = "mi"
        elev_factor = 3.28084
        elev_unit = "ft"
        speed_factor = 0.621371
        speed_unit = "mph"
    else:
        dist_factor = 1.0
        dist_unit = "km"
        elev_factor = 1.0
        elev_unit = "m"
        speed_factor = 1.0
        speed_unit = "km/h"

    print("=== GPX Route Analysis ===")
    if route_metadata and route_metadata.get("name"):
        print(f"Route: {route_metadata['name']}")
    headwind_display = args.headwind * speed_factor
    max_coast_display = args.max_coast_speed * speed_factor
    print(
        f"Config: mass={args.mass}kg cda={args.cda} crr={args.crr} power={args.power}W "
        f"coasting_grade={args.coasting_grade}° max_coast_speed={max_coast_display:.0f}{speed_unit} "
        f"smoothing={smoothing_radius}m elevation_scale={args.elevation_scale} "
        f"headwind={headwind_display:.1f}{speed_unit}"
    )
    # Primary results (most important)
    print("=" * 40)
    print(f"  Est. Time @{params.assumed_avg_power:.0f}W: {format_duration(result.estimated_moving_time_at_power)}")
    print(f"  Est. Work:       {result.estimated_work / 1000:.1f} kJ")
    print("=" * 40)

    # Route details
    dist = result.total_distance / 1000 * dist_factor
    print(f"Distance:       {dist:.2f} {dist_unit}")
    gain = result.elevation_gain * elev_factor
    # Show scaling info or high-noise detection
    if high_noise_detected:
        print(f"Elevation Gain: {gain:.0f} {elev_unit} [high-noise DEM, {HIGH_NOISE_SMOOTHING_RADIUS:.0f}m smoothing]")
    elif abs(api_elevation_scale - 1.0) > 0.05:
        print(f"Elevation Gain: {gain:.0f} {elev_unit} [scaled {api_elevation_scale:.2f}x]")
    else:
        print(f"Elevation Gain: {gain:.0f} {elev_unit}")
    loss = result.elevation_loss * elev_factor
    print(f"Elevation Loss: {loss:.0f} {elev_unit}")
    avg_speed = result.avg_speed * 3.6 * speed_factor
    print(f"Avg Speed:      {avg_speed:.1f} {speed_unit}")
    max_speed = result.max_speed * 3.6 * speed_factor
    print(f"Max Speed:      {max_speed:.1f} {speed_unit}")
    print(f"Est. Avg Power: {result.estimated_avg_power:.0f} W")

    # Surface info - prefer API's unpaved_pct if available
    api_unpaved_pct = route_metadata.get("unpaved_pct") if route_metadata else None
    if api_unpaved_pct is not None:
        total_dist_km = result.total_distance / 1000 * dist_factor
        unpaved_dist = total_dist_km * (api_unpaved_pct / 100)
        paved_dist = total_dist_km - unpaved_dist
        print(
            f"Surface:        {paved_dist:.1f} {dist_unit} paved, "
            f"{unpaved_dist:.1f} {dist_unit} unpaved ({api_unpaved_pct:.0f}%)"
        )

    # Hilliness and steepness
    hilly_factor = 5.28 if args.imperial else 1.0
    hilly_unit = "ft/mi" if args.imperial else "m/km"
    print(f"Hilliness:      {hilliness.hilliness_score * hilly_factor:.0f} {hilly_unit}")
    print(f"Steepness:      {hilliness.steepness_score:.1f}%")
    print(f"Max Grade:      {hilliness.max_grade:.1f}%")

    # Grade histogram
    print("")
    print("Time at Grade:")
    total_time = sum(hilliness.grade_time_histogram.values())
    max_seconds = max(hilliness.grade_time_histogram.values()) if total_time > 0 else 1
    for label in GRADE_LABELS:
        seconds = hilliness.grade_time_histogram.get(label, 0)
        pct = (seconds / total_time * 100) if total_time > 0 else 0
        bar_width = int((seconds / max_seconds) * 20) if max_seconds > 0 else 0
        bar = "█" * bar_width
        if pct >= 0.5:
            print(f"  {label:>6}: {bar:<20} {pct:4.0f}%")

    # Show elevation scaling note if significant API scaling was used (>5% adjustment)
    if abs(api_elevation_scale - 1.0) > 0.05 and api_elevation_gain is not None:
        print("")
        print(f"Note: Elevation scaled to match RideWithGPS API ({api_elevation_gain:.0f}m).")
        print(f"      GPS track elevation is often inaccurate; API uses corrected DEM data.")
        print(f"      Use --no-api-elevation to disable this correction.")

    # Compare with actual trip data if provided
    if args.compare_trip:
        if not is_ridewithgps_trip_url(args.compare_trip):
            print(f"Error: Invalid RideWithGPS trip URL: {args.compare_trip}", file=sys.stderr)
            sys.exit(1)

        try:
            trip_points, trip_metadata = get_trip_data(args.compare_trip)
        except Exception as e:
            print(f"Error downloading trip data: {e}", file=sys.stderr)
            sys.exit(1)

        if len(trip_points) < 2:
            print("Error: Trip contains fewer than 2 track points.", file=sys.stderr)
            sys.exit(1)

        print("")
        if trip_metadata.get("name"):
            print(f"Trip: {trip_metadata['name']}")
        comparison = compare_route_with_trip(
            points,
            trip_points,
            params,
            result.estimated_moving_time_at_power.total_seconds(),
            result.estimated_work,
            route_elevation_gain=result.elevation_gain,
            trip_elevation_gain=trip_metadata.get("elevation_gain"),
        )
        print(format_comparison_report(comparison, params))
