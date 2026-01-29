"""Compare route predictions with actual ride data."""

from dataclasses import dataclass
import math

from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.physics import estimate_speed_from_power
from gpx_analyzer.ridewithgps import TripPoint


@dataclass
class GradeBucket:
    """Statistics for a gradient bucket."""

    grade_pct: int  # Bucket center (e.g., -4 means -5% to -3%)
    actual_speeds: list[float]  # m/s
    predicted_speeds: list[float]  # m/s
    actual_powers: list[float]  # watts
    point_count: int

    @property
    def avg_actual_speed(self) -> float:
        return sum(self.actual_speeds) / len(self.actual_speeds) if self.actual_speeds else 0.0

    @property
    def avg_predicted_speed(self) -> float:
        return sum(self.predicted_speeds) / len(self.predicted_speeds) if self.predicted_speeds else 0.0

    @property
    def avg_actual_power(self) -> float:
        return sum(self.actual_powers) / len(self.actual_powers) if self.actual_powers else 0.0

    @property
    def speed_error_pct(self) -> float:
        """Percentage error: positive means predicted too fast."""
        if self.avg_actual_speed == 0:
            return 0.0
        return ((self.avg_predicted_speed - self.avg_actual_speed) / self.avg_actual_speed) * 100


@dataclass
class ComparisonResult:
    """Result of comparing route prediction with actual ride."""

    route_distance: float  # meters
    trip_distance: float  # meters
    predicted_time: float  # seconds
    actual_moving_time: float  # seconds
    time_error_pct: float  # positive means predicted too slow
    grade_buckets: list[GradeBucket]
    has_power_data: bool
    actual_avg_power: float | None  # watts, if power data available


def compare_route_with_trip(
    route_points: list[TrackPoint],
    trip_points: list[TripPoint],
    params: RiderParams,
    predicted_time_seconds: float,
) -> ComparisonResult:
    """Compare route predictions with actual trip data.

    Analyzes how well the physics model predicts actual ride performance
    by comparing predicted vs actual speeds across different gradients.

    Args:
        route_points: Route track points (for getting surface/crr data)
        trip_points: Actual ride data points
        params: Rider parameters used for prediction
        predicted_time_seconds: The predicted moving time from analyze()
    """
    # Calculate actual moving time (points where speed > 0.5 m/s)
    moving_points = [tp for tp in trip_points if tp.speed is not None and tp.speed > 0.5]
    actual_moving_time = len(moving_points)  # Each point is ~1 second

    # Check for power data
    power_points = [tp.power for tp in trip_points if tp.power is not None]
    has_power_data = len(power_points) > len(trip_points) * 0.5  # >50% coverage
    actual_avg_power = sum(power_points) / len(power_points) if power_points else None

    # Group trip points by gradient and calculate actual vs predicted speeds
    grade_buckets = _build_grade_buckets(trip_points, params)

    # Calculate distances
    route_distance = _calculate_route_distance(route_points)
    trip_distance = trip_points[-1].distance if trip_points else 0.0

    # Time error
    time_error_pct = ((predicted_time_seconds - actual_moving_time) / actual_moving_time * 100
                      if actual_moving_time > 0 else 0.0)

    return ComparisonResult(
        route_distance=route_distance,
        trip_distance=trip_distance,
        predicted_time=predicted_time_seconds,
        actual_moving_time=actual_moving_time,
        time_error_pct=time_error_pct,
        grade_buckets=grade_buckets,
        has_power_data=has_power_data,
        actual_avg_power=actual_avg_power,
    )


def _build_grade_buckets(
    trip_points: list[TripPoint], params: RiderParams
) -> list[GradeBucket]:
    """Build gradient buckets from trip data."""
    buckets: dict[int, GradeBucket] = {}

    for i in range(1, len(trip_points)):
        prev, curr = trip_points[i - 1], trip_points[i]

        # Calculate gradient
        dist_delta = curr.distance - prev.distance
        if dist_delta < 1:  # Skip very short segments
            continue

        elev_prev = prev.elevation if prev.elevation is not None else 0.0
        elev_curr = curr.elevation if curr.elevation is not None else 0.0
        elev_delta = elev_curr - elev_prev
        grade_pct = (elev_delta / dist_delta) * 100

        # Bucket by 2% increments, clamped to reasonable range
        bucket_key = round(grade_pct / 2) * 2
        bucket_key = max(-12, min(12, bucket_key))

        # Only include moving points
        if curr.speed is None or curr.speed < 0.5:
            continue

        # Initialize bucket if needed
        if bucket_key not in buckets:
            buckets[bucket_key] = GradeBucket(
                grade_pct=bucket_key,
                actual_speeds=[],
                predicted_speeds=[],
                actual_powers=[],
                point_count=0,
            )

        bucket = buckets[bucket_key]
        bucket.actual_speeds.append(curr.speed)
        bucket.point_count += 1

        # Calculate predicted speed for this gradient
        slope_angle = math.atan2(grade_pct / 100, 1)
        predicted_speed = estimate_speed_from_power(slope_angle, params)
        bucket.predicted_speeds.append(predicted_speed)

        # Record power if available
        if curr.power is not None:
            bucket.actual_powers.append(curr.power)

    # Sort by gradient and return
    return [buckets[k] for k in sorted(buckets.keys())]


def _calculate_route_distance(points: list[TrackPoint]) -> float:
    """Calculate total route distance."""
    from geopy.distance import geodesic

    total = 0.0
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        total += geodesic((p1.lat, p1.lon), (p2.lat, p2.lon)).meters
    return total


def format_comparison_report(result: ComparisonResult, params: RiderParams) -> str:
    """Format a comparison result as a human-readable report."""
    lines = []

    lines.append("=== Route vs Trip Comparison ===")
    lines.append("")

    # Distance comparison
    lines.append(f"Route distance: {result.route_distance/1000:.1f} km")
    lines.append(f"Trip distance:  {result.trip_distance/1000:.1f} km")
    lines.append("")

    # Time comparison
    pred_hours = result.predicted_time / 3600
    actual_hours = result.actual_moving_time / 3600
    diff_minutes = (result.predicted_time - result.actual_moving_time) / 60

    lines.append(f"Predicted time @{params.assumed_avg_power:.0f}W: {pred_hours:.2f} hours")
    lines.append(f"Actual moving time:        {actual_hours:.2f} hours")
    lines.append(f"Difference: {diff_minutes:+.0f} minutes ({result.time_error_pct:+.1f}%)")
    lines.append("")

    # Power data
    if result.has_power_data and result.actual_avg_power:
        lines.append(f"Actual avg power: {result.actual_avg_power:.0f}W (model assumes {params.assumed_avg_power:.0f}W)")
        lines.append("")

    # Grade breakdown
    lines.append("Speed by gradient (actual vs predicted):")
    lines.append(f"{'Grade':>6} | {'Actual':>8} | {'Pred':>8} | {'Error':>8} | {'Pwr':>6}")
    lines.append("-" * 50)

    for bucket in result.grade_buckets:
        if bucket.point_count < 10:  # Skip sparse buckets
            continue

        actual_kmh = bucket.avg_actual_speed * 3.6
        pred_kmh = bucket.avg_predicted_speed * 3.6
        pwr_str = f"{bucket.avg_actual_power:.0f}W" if bucket.actual_powers else "n/a"

        lines.append(
            f"{bucket.grade_pct:>+5}% | {actual_kmh:>7.1f} | {pred_kmh:>7.1f} | "
            f"{bucket.speed_error_pct:>+7.0f}% | {pwr_str:>6}"
        )

    return "\n".join(lines)
