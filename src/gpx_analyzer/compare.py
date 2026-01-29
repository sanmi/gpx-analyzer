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
    predicted_work: float  # joules
    actual_work: float | None  # joules, if power data available
    grade_buckets: list[GradeBucket]
    has_power_data: bool
    actual_avg_power: float | None  # watts, if power data available


def compare_route_with_trip(
    route_points: list[TrackPoint],
    trip_points: list[TripPoint],
    params: RiderParams,
    predicted_time_seconds: float,
    predicted_work_joules: float,
) -> ComparisonResult:
    """Compare route predictions with actual trip data.

    Analyzes how well the physics model predicts actual ride performance
    by comparing predicted vs actual speeds across different gradients.

    Args:
        route_points: Route track points (for getting surface/crr data)
        trip_points: Actual ride data points
        params: Rider parameters used for prediction
        predicted_time_seconds: The predicted moving time from analyze()
        predicted_work_joules: The predicted work from analyze()
    """
    # Calculate actual moving time from timestamps (when speed > 0.5 m/s)
    actual_moving_time = _calculate_moving_time(trip_points)

    # Check for power data and calculate actual work
    actual_work, actual_avg_power, has_power_data = _calculate_actual_work(trip_points)

    # Group trip points by gradient and calculate actual vs predicted speeds
    grade_buckets = _build_grade_buckets(trip_points, params)

    # Calculate distances
    route_distance = _calculate_route_distance(route_points)
    trip_distance = _calculate_trip_distance(trip_points)

    # Time error
    time_error_pct = ((predicted_time_seconds - actual_moving_time) / actual_moving_time * 100
                      if actual_moving_time > 0 else 0.0)

    return ComparisonResult(
        route_distance=route_distance,
        trip_distance=trip_distance,
        predicted_time=predicted_time_seconds,
        actual_moving_time=actual_moving_time,
        time_error_pct=time_error_pct,
        predicted_work=predicted_work_joules,
        actual_work=actual_work,
        grade_buckets=grade_buckets,
        has_power_data=has_power_data,
        actual_avg_power=actual_avg_power,
    )


def _build_grade_buckets(
    trip_points: list[TripPoint], params: RiderParams
) -> list[GradeBucket]:
    """Build gradient buckets from trip data."""
    from geopy.distance import geodesic

    buckets: dict[int, GradeBucket] = {}

    for i in range(1, len(trip_points)):
        prev, curr = trip_points[i - 1], trip_points[i]

        # Calculate distance between points using geodesic
        dist_delta = geodesic((prev.lat, prev.lon), (curr.lat, curr.lon)).meters
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


def _calculate_trip_distance(points: list[TripPoint]) -> float:
    """Calculate total trip distance from coordinates."""
    from geopy.distance import geodesic

    if not points:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        total += geodesic((p1.lat, p1.lon), (p2.lat, p2.lon)).meters
    return total


def _calculate_moving_time(points: list[TripPoint]) -> float:
    """Calculate moving time from timestamps when speed > 0.5 m/s.

    Returns total moving time in seconds.
    """
    if len(points) < 2:
        return 0.0

    moving_time = 0.0
    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]

        # Skip if timestamps missing
        if prev.timestamp is None or curr.timestamp is None:
            continue

        # Only count time when moving (speed > 0.5 m/s at current point)
        if curr.speed is not None and curr.speed > 0.5:
            time_delta = curr.timestamp - prev.timestamp
            # Cap individual gaps at 60 seconds to avoid counting long stops
            moving_time += min(time_delta, 60)

    return moving_time


def _calculate_actual_work(points: list[TripPoint]) -> tuple[float | None, float | None, bool]:
    """Calculate actual work from power data and timestamps.

    Returns (total_work_joules, avg_power_watts, has_power_data).
    Work = sum of (power * time_delta) for each segment.
    """
    if len(points) < 2:
        return None, None, False

    total_work = 0.0
    total_power_time = 0.0  # Time with power data (for averaging)
    total_power_sum = 0.0   # Sum of power * time (for weighted average)

    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]

        # Skip if timestamps missing
        if prev.timestamp is None or curr.timestamp is None:
            continue

        # Skip if no power data
        if curr.power is None:
            continue

        time_delta = curr.timestamp - prev.timestamp
        # Cap individual gaps at 60 seconds
        time_delta = min(time_delta, 60)

        # Work = power * time (watts * seconds = joules)
        total_work += curr.power * time_delta
        total_power_time += time_delta
        total_power_sum += curr.power * time_delta

    # Check if we have enough power data (>50% of points)
    points_with_power = sum(1 for p in points if p.power is not None)
    has_power_data = points_with_power > len(points) * 0.5

    if total_power_time > 0:
        avg_power = total_power_sum / total_power_time
        return total_work, avg_power, has_power_data
    else:
        return None, None, False


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

    # Work comparison
    pred_kj = result.predicted_work / 1000
    lines.append(f"Predicted work: {pred_kj:.0f} kJ")
    if result.actual_work is not None:
        actual_kj = result.actual_work / 1000
        work_diff_pct = ((result.predicted_work - result.actual_work) / result.actual_work * 100
                         if result.actual_work > 0 else 0.0)
        lines.append(f"Actual work:    {actual_kj:.0f} kJ ({work_diff_pct:+.0f}%)")
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
