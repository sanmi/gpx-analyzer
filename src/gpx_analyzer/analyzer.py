import math
from dataclasses import dataclass, replace
from datetime import timedelta

from geopy.distance import geodesic

from gpx_analyzer.models import RideAnalysis, RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work, estimate_speed_from_power

# Speed below this threshold (m/s) counts as stopped
MOVING_SPEED_THRESHOLD = 0.5  # ~1.8 km/h

_MAX_CALIBRATION_ITERATIONS = 5
_POWER_TOLERANCE = 0.02  # 2% convergence threshold


def _analyze_segments(
    points: list[TrackPoint], params: RiderParams
) -> tuple[float, float, float, float, float, float]:
    """Run one pass of segment analysis.

    Returns (total_distance, elevation_gain, elevation_loss,
             total_work, moving_seconds, max_speed).
    """
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0
    total_work = 0.0
    moving_seconds = 0.0
    max_speed = 0.0

    for i in range(1, len(points)):
        work, dist, elapsed = calculate_segment_work(points[i - 1], points[i], params)

        total_distance += dist
        total_work += work

        elev_prev = points[i - 1].elevation
        elev_curr = points[i].elevation
        if elev_prev is not None and elev_curr is not None:
            delta = elev_curr - elev_prev
            if delta > 0:
                elevation_gain += delta
            else:
                elevation_loss += abs(delta)

        if elapsed > 0:
            speed = dist / elapsed
            if speed > max_speed:
                max_speed = speed
            if speed >= MOVING_SPEED_THRESHOLD:
                moving_seconds += elapsed

    return total_distance, elevation_gain, elevation_loss, total_work, moving_seconds, max_speed


def analyze(points: list[TrackPoint], params: RiderParams) -> RideAnalysis:
    """Analyze a list of TrackPoints and return a RideAnalysis summary.

    When time data is missing and speed is estimated from power, the analyzer
    calibrates the internal nominal power so that the resulting time-weighted
    average power matches the user's assumed_avg_power.
    """
    if len(points) < 2:
        return RideAnalysis(
            total_distance=0.0,
            elevation_gain=0.0,
            elevation_loss=0.0,
            duration=timedelta(),
            moving_time=timedelta(),
            avg_speed=0.0,
            max_speed=0.0,
            estimated_work=0.0,
            estimated_avg_power=0.0,
            estimated_moving_time_at_power=timedelta(),
        )

    target_power = params.assumed_avg_power
    calibrated_params = params

    # Iteratively adjust nominal power so average power matches the target
    for _ in range(_MAX_CALIBRATION_ITERATIONS):
        totals = _analyze_segments(points, calibrated_params)
        total_distance, elevation_gain, elevation_loss, total_work, moving_seconds, max_speed = totals

        if target_power <= 0 or moving_seconds <= 0:
            break

        avg_power = total_work / moving_seconds
        if avg_power <= 0:
            break

        ratio = target_power / avg_power
        if abs(ratio - 1.0) < _POWER_TOLERANCE:
            break

        calibrated_params = replace(
            calibrated_params,
            assumed_avg_power=calibrated_params.assumed_avg_power * ratio,
        )

    # Duration from timestamps (if available)
    if points[0].time is not None and points[-1].time is not None:
        duration = points[-1].time - points[0].time
    else:
        duration = timedelta()

    moving_time = timedelta(seconds=moving_seconds)
    avg_speed = total_distance / moving_seconds if moving_seconds > 0 else 0.0
    avg_power = total_work / moving_seconds if moving_seconds > 0 else 0.0

    if target_power > 0 and total_work > 0:
        est_seconds = total_work / target_power
        estimated_moving_time_at_power = timedelta(seconds=est_seconds)
    else:
        estimated_moving_time_at_power = timedelta()

    return RideAnalysis(
        total_distance=total_distance,
        elevation_gain=elevation_gain,
        elevation_loss=elevation_loss,
        duration=duration,
        moving_time=moving_time,
        avg_speed=avg_speed,
        max_speed=max_speed,
        estimated_work=total_work,
        estimated_avg_power=avg_power,
        estimated_moving_time_at_power=estimated_moving_time_at_power,
    )


# Grade histogram bins (percent): ..., -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, ...
GRADE_BINS = [-float('inf'), -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, float('inf')]
GRADE_LABELS = ['<-10%', '-10%', '-8%', '-6%', '-4%', '-2%', '0%', '2%', '4%', '6%', '8%', '>10%']

# Minimum grade to count as "climbing" for steepness calculation
STEEPNESS_MIN_GRADE_PCT = 2.0


@dataclass
class HillinessAnalysis:
    """Hilliness metrics for a route."""
    hilliness_score: float  # meters of elevation gain per km
    steepness_score: float  # effort-weighted average climbing grade (%)
    grade_time_histogram: dict[str, float]  # grade bucket -> seconds spent
    total_time: float  # total seconds


def calculate_hilliness(
    points: list[TrackPoint], params: RiderParams
) -> HillinessAnalysis:
    """Calculate hilliness score, steepness score, and time-at-grade histogram.

    Hilliness score is elevation gain per km (m/km).
    Steepness score is effort-weighted average climbing grade (%) for grades >= 2%.
    Grade histogram shows time spent in each grade bucket.
    """
    if len(points) < 2:
        return HillinessAnalysis(
            hilliness_score=0.0,
            steepness_score=0.0,
            grade_time_histogram={label: 0.0 for label in GRADE_LABELS},
            total_time=0.0,
        )

    total_distance = 0.0
    elevation_gain = 0.0
    grade_times = {label: 0.0 for label in GRADE_LABELS}
    total_time = 0.0

    # For steepness calculation (effort-weighted average grade)
    weighted_grade_sum = 0.0
    climbing_work_sum = 0.0

    for i in range(1, len(points)):
        pt_a, pt_b = points[i - 1], points[i]

        # Calculate distance
        dist = geodesic((pt_a.lat, pt_a.lon), (pt_b.lat, pt_b.lon)).meters
        if dist < 0.1:
            continue

        total_distance += dist

        # Calculate elevation change and grade
        elev_a = pt_a.elevation if pt_a.elevation is not None else 0.0
        elev_b = pt_b.elevation if pt_b.elevation is not None else 0.0
        delta_elev = elev_b - elev_a

        if delta_elev > 0:
            elevation_gain += delta_elev

        # Grade in percent
        grade_pct = (delta_elev / dist) * 100 if dist > 0 else 0.0

        # Calculate work and time for this segment
        work, _, elapsed = calculate_segment_work(pt_a, pt_b, params)
        total_time += elapsed

        # Accumulate steepness data for climbing segments >= threshold
        if grade_pct >= STEEPNESS_MIN_GRADE_PCT and work > 0:
            weighted_grade_sum += grade_pct * work
            climbing_work_sum += work

        # Bin the grade
        for j in range(len(GRADE_BINS) - 1):
            if GRADE_BINS[j] <= grade_pct < GRADE_BINS[j + 1]:
                grade_times[GRADE_LABELS[j]] += elapsed
                break

    # Hilliness score: meters gained per km
    hilliness_score = (elevation_gain / (total_distance / 1000)) if total_distance > 0 else 0.0

    # Steepness score: effort-weighted average climbing grade
    steepness_score = (weighted_grade_sum / climbing_work_sum) if climbing_work_sum > 0 else 0.0

    return HillinessAnalysis(
        hilliness_score=hilliness_score,
        steepness_score=steepness_score,
        grade_time_histogram=grade_times,
        total_time=total_time,
    )
