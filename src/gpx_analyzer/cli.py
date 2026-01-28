import argparse
import sys

from gpx_analyzer.analyzer import analyze
from gpx_analyzer.models import RiderParams
from gpx_analyzer.parser import parse_gpx
from gpx_analyzer.ridewithgps import get_gpx, is_ridewithgps_url
from gpx_analyzer.smoothing import smooth_elevations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a GPX bike route with physics-based power estimation."
    )
    parser.add_argument("gpx_file", help="Path to GPX file")
    parser.add_argument(
        "--mass",
        type=float,
        default=85.0,
        help="Total mass of rider + bike in kg (default: 85)",
    )
    parser.add_argument(
        "--cda",
        type=float,
        default=0.35,
        help="Drag coefficient * frontal area in m² (default: 0.35)",
    )
    parser.add_argument(
        "--crr",
        type=float,
        default=0.005,
        help="Rolling resistance coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=150.0,
        help="Assumed average power output in watts (default: 150)",
    )
    parser.add_argument(
        "--coasting-grade",
        type=float,
        default=-5.0,
        help="Grade in degrees at which rider fully coasts (default: -5)",
    )
    parser.add_argument(
        "--max-coast-speed",
        type=float,
        default=48.0,
        help="Maximum coasting speed in km/h (default: 48)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=50.0,
        help="Elevation smoothing radius in meters (default: 50)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable elevation smoothing",
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=1.0,
        help="Scale factor for elevation changes (default: 1.0). Use <1 to reduce overestimated GPS elevation.",
    )
    return parser


def format_duration(td) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    params = RiderParams(
        total_mass=args.mass,
        cda=args.cda,
        crr=args.crr,
        assumed_avg_power=args.power,
        coasting_grade_threshold=args.coasting_grade,
        max_coasting_speed=args.max_coast_speed / 3.6,
    )

    if is_ridewithgps_url(args.gpx_file):
        try:
            gpx_path = get_gpx(args.gpx_file)
        except Exception as e:
            print(f"Error downloading GPX from RideWithGPS: {e}", file=sys.stderr)
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
