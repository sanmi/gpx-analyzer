from dataclasses import replace
from datetime import timedelta

from gpx_analyzer.models import RideAnalysis, RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work

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
