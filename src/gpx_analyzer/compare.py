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
    route_elevation_gain: float | None = None  # meters
    trip_elevation_gain: float | None = None  # meters


def compare_route_with_trip(
    route_points: list[TrackPoint],
    trip_points: list[TripPoint],
    params: RiderParams,
    predicted_time_seconds: float,
    predicted_work_joules: float,
    route_elevation_gain: float | None = None,
    trip_elevation_gain: float | None = None,
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
        route_elevation_gain: Elevation gain from route analysis (meters)
        trip_elevation_gain: Elevation gain from trip metadata (meters)
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
        route_elevation_gain=route_elevation_gain,
        trip_elevation_gain=trip_elevation_gain,
    )


def _build_grade_buckets(
    trip_points: list[TripPoint], params: RiderParams
) -> list[GradeBucket]:
    """Build gradient buckets from trip data."""
    from gpx_analyzer.distance import haversine_distance

    buckets: dict[int, GradeBucket] = {}

    for i in range(1, len(trip_points)):
        prev, curr = trip_points[i - 1], trip_points[i]

        # Calculate distance between points
        dist_delta = haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)
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
    from gpx_analyzer.distance import haversine_distance

    total = 0.0
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        total += haversine_distance(p1.lat, p1.lon, p2.lat, p2.lon)
    return total


def _calculate_trip_distance(points: list[TripPoint]) -> float:
    """Calculate total trip distance from coordinates."""
    from gpx_analyzer.distance import haversine_distance

    if not points:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        total += haversine_distance(p1.lat, p1.lon, p2.lat, p2.lon)
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

        # Require BOTH points to be moving to count the time between them
        # This avoids counting stop time when transitioning from stopped to moving
        prev_speed = prev.speed if prev.speed is not None else 0
        curr_speed = curr.speed if curr.speed is not None else 0
        if prev_speed > 0.5 and curr_speed > 0.5:
            time_delta = curr.timestamp - prev.timestamp
            # Cap individual gaps at 60 seconds to avoid counting long stops
            moving_time += min(time_delta, 60)

    return moving_time


def _calculate_actual_work(points: list[TripPoint]) -> tuple[float | None, float | None, bool]:
    """Calculate actual work from power data and timestamps.

    Returns (total_work_joules, avg_power_watts, has_power_data).
    Work = sum of (power * time_delta) for each segment while moving.
    Avg power is calculated over moving time only (speed > 0.5 m/s).
    """
    if len(points) < 2:
        return None, None, False

    total_work = 0.0
    moving_power_time = 0.0  # Moving time with power data (for averaging)

    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]

        # Skip if timestamps missing
        if prev.timestamp is None or curr.timestamp is None:
            continue

        # Skip if no power data
        if curr.power is None:
            continue

        # Require BOTH points to be moving (consistent with moving time calculation)
        prev_speed = prev.speed if prev.speed is not None else 0
        curr_speed = curr.speed if curr.speed is not None else 0
        if prev_speed <= 0.5 or curr_speed <= 0.5:
            continue

        time_delta = curr.timestamp - prev.timestamp
        # Cap individual gaps at 60 seconds
        time_delta = min(time_delta, 60)

        # Work = power * time (watts * seconds = joules)
        total_work += curr.power * time_delta
        moving_power_time += time_delta

    # Check if we have enough power data (>50% of points)
    points_with_power = sum(1 for p in points if p.power is not None)
    has_power_data = points_with_power > len(points) * 0.5

    if moving_power_time > 0:
        avg_power = total_work / moving_power_time
        return total_work, avg_power, has_power_data
    else:
        return None, None, False


def format_comparison_report(result: ComparisonResult, params: RiderParams) -> str:
    """Format a comparison result as a human-readable report."""
    lines = []

    lines.append("=== Route vs Trip Comparison ===")
    lines.append("")

    # Build comparison table
    lines.append(f"{'Metric':<16} {'Estimated':>12} {'Actual':>12} {'Diff':>8}")
    lines.append("-" * 50)

    # Distance
    est_dist = f"{result.route_distance/1000:.1f} km"
    act_dist = f"{result.trip_distance/1000:.1f} km"
    dist_diff = ((result.route_distance - result.trip_distance) / result.trip_distance * 100
                 if result.trip_distance > 0 else 0.0)
    lines.append(f"{'Distance':<16} {est_dist:>12} {act_dist:>12} {dist_diff:>+7.1f}%")

    # Elevation gain
    if result.route_elevation_gain is not None and result.trip_elevation_gain is not None:
        est_elev = f"{result.route_elevation_gain:.0f} m"
        act_elev = f"{result.trip_elevation_gain:.0f} m"
        elev_diff = ((result.route_elevation_gain - result.trip_elevation_gain)
                     / result.trip_elevation_gain * 100 if result.trip_elevation_gain > 0 else 0.0)
        lines.append(f"{'Elevation gain':<16} {est_elev:>12} {act_elev:>12} {elev_diff:>+7.1f}%")

    # Moving time
    est_time = f"{result.predicted_time/3600:.2f} h"
    act_time = f"{result.actual_moving_time/3600:.2f} h"
    time_diff = result.time_error_pct
    lines.append(f"{'Moving time':<16} {est_time:>12} {act_time:>12} {time_diff:>+7.1f}%")

    # Average speed
    est_speed = result.route_distance / result.predicted_time if result.predicted_time > 0 else 0
    act_speed = result.trip_distance / result.actual_moving_time if result.actual_moving_time > 0 else 0
    est_speed_str = f"{est_speed * 3.6:.1f} km/h"
    act_speed_str = f"{act_speed * 3.6:.1f} km/h"
    speed_diff = ((est_speed - act_speed) / act_speed * 100 if act_speed > 0 else 0.0)
    lines.append(f"{'Avg speed':<16} {est_speed_str:>12} {act_speed_str:>12} {speed_diff:>+7.1f}%")

    # Average power
    est_power = f"{params.assumed_avg_power:.0f} W"
    if result.has_power_data and result.actual_avg_power:
        act_power = f"{result.actual_avg_power:.0f} W"
        power_diff = ((params.assumed_avg_power - result.actual_avg_power)
                      / result.actual_avg_power * 100 if result.actual_avg_power > 0 else 0.0)
        lines.append(f"{'Avg power':<16} {est_power:>12} {act_power:>12} {power_diff:>+7.1f}%")
    else:
        lines.append(f"{'Avg power':<16} {est_power:>12} {'n/a':>12} {'':>8}")

    # Work
    est_work = f"{result.predicted_work/1000:.0f} kJ"
    if result.actual_work is not None:
        act_work = f"{result.actual_work/1000:.0f} kJ"
        work_diff = ((result.predicted_work - result.actual_work) / result.actual_work * 100
                     if result.actual_work > 0 else 0.0)
        lines.append(f"{'Work':<16} {est_work:>12} {act_work:>12} {work_diff:>+7.1f}%")
    else:
        lines.append(f"{'Work':<16} {est_work:>12} {'n/a':>12} {'':>8}")

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
