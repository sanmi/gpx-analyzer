import argparse
import sys

from gpx_analyzer.analyzer import analyze
from gpx_analyzer.models import RiderParams
from gpx_analyzer.parser import parse_gpx


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
    )

    try:
        points = parse_gpx(args.gpx_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.gpx_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing GPX file: {e}", file=sys.stderr)
        sys.exit(1)

    if len(points) < 2:
        print("Error: GPX file contains fewer than 2 track points.", file=sys.stderr)
        sys.exit(1)

    result = analyze(points, params)

    print("=== GPX Ride Analysis ===")
    print(f"Distance:       {result.total_distance / 1000:.2f} km")
    print(f"Elevation Gain: {result.elevation_gain:.0f} m")
    print(f"Elevation Loss: {result.elevation_loss:.0f} m")
    print(f"Duration:       {format_duration(result.duration)}")
    print(f"Moving Time:    {format_duration(result.moving_time)}")
    print(f"Avg Speed:      {result.avg_speed * 3.6:.1f} km/h")
    print(f"Max Speed:      {result.max_speed * 3.6:.1f} km/h")
    print(f"Est. Work:      {result.estimated_work / 1000:.1f} kJ")
    print(f"Est. Avg Power: {result.estimated_avg_power:.0f} W")
    print(f"Est. Time @{params.assumed_avg_power:.0f}W: {format_duration(result.estimated_moving_time_at_power)}")
