import argparse
import sys

from geopy.distance import geodesic

from gpx_analyzer.analyzer import analyze
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.parser import parse_gpx
from gpx_analyzer.compare import compare_route_with_trip, format_comparison_report
from gpx_analyzer.ridewithgps import (
    _load_config,
    get_gpx,
    get_route_with_surface,
    get_trip_data,
    is_ridewithgps_trip_url,
    is_ridewithgps_url,
)
from gpx_analyzer.smoothing import smooth_elevations

# Default values for CLI options
DEFAULTS = {
    "mass": 85.0,
    "cda": 0.35,
    "crr": 0.005,
    "power": 150.0,
    "coasting_grade": -5.0,
    "max_coast_speed": 48.0,
    "max_coast_speed_unpaved": 24.0,
    "smoothing": 50.0,
    "elevation_scale": 1.0,
    "headwind": 0.0,
}


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    """Build argument parser with defaults from config file."""
    if config is None:
        config = {}

    def get_default(key: str) -> float:
        return config.get(key, DEFAULTS[key])

    parser = argparse.ArgumentParser(
        description="Analyze a GPX bike route with physics-based power estimation."
    )
    parser.add_argument("gpx_file", help="Path to GPX file")
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
    return parser


def format_duration(td) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


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
        dist = geodesic((pt_a.lat, pt_a.lon), (pt_b.lat, pt_b.lon)).meters

        # Use the destination point's unpaved flag to classify the segment
        if pt_b.unpaved:
            unpaved_dist += dist
        else:
            paved_dist += dist

    return paved_dist, unpaved_dist


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
    )

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
    if smoothing_radius > 0 or args.elevation_scale != 1.0:
        points = smooth_elevations(points, smoothing_radius, args.elevation_scale)

    result = analyze(points, params)

    print("=== GPX Route Analysis ===")
    headwind_mph = args.headwind * 0.621371
    print(
        f"Config: mass={args.mass}kg cda={args.cda} crr={args.crr} power={args.power}W "
        f"coasting_grade={args.coasting_grade}° max_coast_speed={args.max_coast_speed}km/h "
        f"smoothing={smoothing_radius}m elevation_scale={args.elevation_scale} "
        f"headwind={args.headwind}km/h ({headwind_mph:.1f}mph)"
    )
    dist_km = result.total_distance / 1000
    dist_mi = dist_km * 0.621371
    print(f"Distance:       {dist_km:.2f} km ({dist_mi:.2f} mi)")
    gain_ft = result.elevation_gain * 3.28084
    print(f"Elevation Gain: {result.elevation_gain:.0f} m ({gain_ft:.0f} ft)")
    loss_ft = result.elevation_loss * 3.28084
    print(f"Elevation Loss: {result.elevation_loss:.0f} m ({loss_ft:.0f} ft)")
    print(f"Duration:       {format_duration(result.duration)}")
    print(f"Moving Time:    {format_duration(result.moving_time)}")
    avg_kmh = result.avg_speed * 3.6
    avg_mph = avg_kmh * 0.621371
    print(f"Avg Speed:      {avg_kmh:.1f} km/h ({avg_mph:.1f} mph)")
    max_kmh = result.max_speed * 3.6
    max_mph = max_kmh * 0.621371
    print(f"Max Speed:      {max_kmh:.1f} km/h ({max_mph:.1f} mph)")
    print(f"Est. Work:      {result.estimated_work / 1000:.1f} kJ")
    print(f"Est. Avg Power: {result.estimated_avg_power:.0f} W")
    print(f"Est. Time @{params.assumed_avg_power:.0f}W: {format_duration(result.estimated_moving_time_at_power)}")

    # Surface breakdown if available
    surface_breakdown = calculate_surface_breakdown(points)
    if surface_breakdown:
        paved_km, unpaved_km = surface_breakdown[0] / 1000, surface_breakdown[1] / 1000
        total_km = paved_km + unpaved_km
        if total_km > 0:
            unpaved_pct = (unpaved_km / total_km) * 100
            paved_mi = paved_km * 0.621371
            unpaved_mi = unpaved_km * 0.621371
            print(
                f"Surface:        {paved_km:.1f} km ({paved_mi:.1f} mi) paved, "
                f"{unpaved_km:.1f} km ({unpaved_mi:.1f} mi) unpaved ({unpaved_pct:.0f}%)"
            )

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
        comparison = compare_route_with_trip(
            points,
            trip_points,
            params,
            result.estimated_moving_time_at_power.total_seconds(),
        )
        print(format_comparison_report(comparison, params))
