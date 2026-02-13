import math
from dataclasses import dataclass, replace
from datetime import timedelta

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import RideAnalysis, RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work, estimate_speed_from_power
from gpx_analyzer.smoothing import smooth_elevations

# Speed below this threshold (m/s) counts as stopped
MOVING_SPEED_THRESHOLD = 0.5  # ~1.8 km/h

_MAX_CALIBRATION_ITERATIONS = 5
_POWER_TOLERANCE = 0.02  # 2% convergence threshold


def _analyze_segments(
    points: list[TrackPoint], params: RiderParams
) -> tuple[float, float, float, float, float, float, float]:
    """Run one pass of segment analysis.

    Returns (total_distance, elevation_gain, elevation_loss,
             total_work, moving_seconds, pedaling_seconds, max_speed).

    pedaling_seconds is the time spent on segments where work > 0 (rider is pedaling).
    This is used for power calibration to avoid descent time affecting the calculation.
    """
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0
    total_work = 0.0
    moving_seconds = 0.0
    pedaling_seconds = 0.0
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
                # Track time spent pedaling (work > 0) for power calibration
                if work > 0:
                    pedaling_seconds += elapsed

    return total_distance, elevation_gain, elevation_loss, total_work, moving_seconds, pedaling_seconds, max_speed


def analyze(points: list[TrackPoint], params: RiderParams) -> RideAnalysis:
    """Analyze a list of TrackPoints and return a RideAnalysis summary.

    Uses climbing_power and flat_power directly to estimate speeds and times.
    No calibration needed - power values are used as specified.
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

    # Run segment analysis with the user's power settings directly
    totals = _analyze_segments(points, params)
    total_distance, elevation_gain, elevation_loss, total_work, moving_seconds, pedaling_seconds, max_speed = totals

    # Duration from timestamps (if available)
    if points[0].time is not None and points[-1].time is not None:
        duration = points[-1].time - points[0].time
    else:
        duration = timedelta()

    moving_time = timedelta(seconds=moving_seconds)
    avg_speed = total_distance / moving_seconds if moving_seconds > 0 else 0.0
    avg_power = total_work / moving_seconds if moving_seconds > 0 else 0.0

    # Use climbing_power as reference for estimated time at power
    # (most work is done on climbs)
    if params.climbing_power > 0 and total_work > 0:
        est_seconds = total_work / params.climbing_power
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

# Steep grade histogram bins (percent): 10, 12, 14, 16, 18, 20, ...
STEEP_GRADE_BINS = [10, 12, 14, 16, 18, 20, float('inf')]
STEEP_GRADE_LABELS = ['10-12%', '12-14%', '14-16%', '16-18%', '18-20%', '>20%']

# Default rolling window for max grade calculation (filters GPS noise)
# 150m balances capturing steep sections vs filtering route GPS noise
DEFAULT_MAX_GRADE_WINDOW = 150.0  # meters

# Maximum realistic grade for paved roads (filters GPS/elevation data errors)
# Steepest paved roads are ~35% (Baldwin Street NZ), most cycling roads < 25%
MAX_REALISTIC_GRADE = 35.0  # percent

# Default smoothing radius for max grade calculation (filters GPS elevation noise)
# Separate from main smoothing to allow more aggressive noise filtering for max grade
DEFAULT_MAX_GRADE_SMOOTHING = 150.0  # meters

# Minimum grade to count as "climbing" for steepness calculation
STEEPNESS_MIN_GRADE_PCT = 2.0


def _calculate_rolling_grades(points: list[TrackPoint], window: float) -> list[float]:
    """Calculate rolling average grade over a distance window for each point.

    Returns a list of grades (%) for each segment, where each grade is calculated
    as elevation change over a forward-looking window of approximately `window` meters.
    """
    if len(points) < 2:
        return []

    # Pre-calculate cumulative distance and elevation at each point
    n = len(points)
    cum_dist = [0.0] * n
    elevations = [0.0] * n

    for i in range(n):
        elevations[i] = points[i].elevation if points[i].elevation is not None else 0.0
        if i > 0:
            d = haversine_distance(
                points[i-1].lat, points[i-1].lon,
                points[i].lat, points[i].lon
            )
            cum_dist[i] = cum_dist[i-1] + d

    # For each point, find the grade over the next `window` meters
    rolling_grades = []
    for i in range(n - 1):
        # Find the furthest point within the window
        target_dist = cum_dist[i] + window
        j = i + 1
        while j < n - 1 and cum_dist[j] < target_dist:
            j += 1

        # Calculate grade from i to j
        dist = cum_dist[j] - cum_dist[i]
        # Skip segments shorter than half the window (avoids edge effects at route ends)
        if dist >= window / 2:
            delta_elev = elevations[j] - elevations[i]
            grade = (delta_elev / dist) * 100
            # Cap at realistic max to filter GPS/elevation errors (both positive and negative)
            grade = max(-MAX_REALISTIC_GRADE, min(grade, MAX_REALISTIC_GRADE))
            rolling_grades.append(grade)
        else:
            rolling_grades.append(0.0)

    return rolling_grades


@dataclass
class HillinessAnalysis:
    """Hilliness metrics for a route."""
    hilliness_score: float  # meters of elevation gain per km
    steepness_score: float  # effort-weighted average climbing grade (%)
    grade_time_histogram: dict[str, float]  # grade bucket -> seconds spent
    grade_distance_histogram: dict[str, float]  # grade bucket -> meters
    total_time: float  # total seconds
    total_distance: float  # total meters
    # Steep climb stats (grades >= 10%)
    max_grade: float  # maximum grade encountered (%)
    steep_distance: float  # meters at >= 10% grade
    very_steep_distance: float  # meters at >= 15% grade
    steep_time: float  # seconds at >= 10% grade
    very_steep_time: float  # seconds at >= 15% grade
    steep_time_histogram: dict[str, float]  # steep grade bucket -> seconds
    steep_distance_histogram: dict[str, float]  # steep grade bucket -> meters


def calculate_hilliness(
    points: list[TrackPoint],
    params: RiderParams,
    unscaled_points: list[TrackPoint] | None = None,
    max_grade_window: float = DEFAULT_MAX_GRADE_WINDOW,
    max_grade_smoothing: float = DEFAULT_MAX_GRADE_SMOOTHING,
) -> HillinessAnalysis:
    """Calculate hilliness score, steepness score, and time-at-grade histogram.

    Hilliness score is elevation gain per km (m/km).
    Steepness score is effort-weighted average climbing grade (%) for grades >= 2%.
    Grade histogram shows time spent in each grade bucket.

    If unscaled_points is provided, max grade is calculated from those points
    (to match RWGPS methodology which uses raw GPS elevation for max grade).

    max_grade_window controls the rolling average window size for max grade calculation.
    max_grade_smoothing is deprecated and no longer used (kept for API compatibility).
    """
    if len(points) < 2:
        return HillinessAnalysis(
            hilliness_score=0.0,
            steepness_score=0.0,
            grade_time_histogram={label: 0.0 for label in GRADE_LABELS},
            grade_distance_histogram={label: 0.0 for label in GRADE_LABELS},
            total_time=0.0,
            total_distance=0.0,
            max_grade=0.0,
            steep_distance=0.0,
            very_steep_distance=0.0,
            steep_time=0.0,
            very_steep_time=0.0,
            steep_time_histogram={label: 0.0 for label in STEEP_GRADE_LABELS},
            steep_distance_histogram={label: 0.0 for label in STEEP_GRADE_LABELS},
        )

    total_distance = 0.0
    elevation_gain = 0.0
    grade_times = {label: 0.0 for label in GRADE_LABELS}
    grade_distances = {label: 0.0 for label in GRADE_LABELS}
    steep_times = {label: 0.0 for label in STEEP_GRADE_LABELS}
    steep_distances = {label: 0.0 for label in STEEP_GRADE_LABELS}
    total_time = 0.0
    steep_distance = 0.0
    very_steep_distance = 0.0
    steep_time = 0.0
    very_steep_time = 0.0

    # Calculate rolling grades for histogram binning from unscaled points (already smoothed by user setting)
    # No additional smoothing - use the same grades that the elevation profile tooltip shows
    histogram_points = unscaled_points if unscaled_points is not None else points
    rolling_grades = _calculate_rolling_grades(histogram_points, max_grade_window)

    # Max grade uses the same rolling_grades as the elevation profile for consistency
    # The rolling window (max_grade_window) already smooths out GPS noise
    max_grade = max(rolling_grades) if rolling_grades else 0.0

    # For steepness calculation (effort-weighted average grade)
    weighted_grade_sum = 0.0
    climbing_work_sum = 0.0

    for i in range(1, len(points)):
        pt_a, pt_b = points[i - 1], points[i]

        # Calculate distance
        dist = haversine_distance(pt_a.lat, pt_a.lon, pt_b.lat, pt_b.lon)
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

        # Use rolling grade for all grade metrics (300m window, matches RWGPS methodology)
        rolling_grade = rolling_grades[i - 1] if i - 1 < len(rolling_grades) else 0.0

        # Accumulate steepness data for climbing segments >= threshold
        if rolling_grade >= STEEPNESS_MIN_GRADE_PCT and work > 0:
            weighted_grade_sum += rolling_grade * work
            climbing_work_sum += work

        # Bin the grade using rolling average
        for j in range(len(GRADE_BINS) - 1):
            if GRADE_BINS[j] <= rolling_grade < GRADE_BINS[j + 1]:
                grade_times[GRADE_LABELS[j]] += elapsed
                grade_distances[GRADE_LABELS[j]] += dist
                break

        # Track steep distances and times using rolling grade
        if rolling_grade >= 10:
            steep_distance += dist
            steep_time += elapsed
        if rolling_grade >= 15:
            very_steep_distance += dist
            very_steep_time += elapsed

        # Bin steep grades (>= 10%) using rolling grade
        if rolling_grade >= 10:
            for j in range(len(STEEP_GRADE_BINS) - 1):
                if STEEP_GRADE_BINS[j] <= rolling_grade < STEEP_GRADE_BINS[j + 1]:
                    steep_times[STEEP_GRADE_LABELS[j]] += elapsed
                    steep_distances[STEEP_GRADE_LABELS[j]] += dist
                    break

    # Hilliness score: meters gained per km
    hilliness_score = (elevation_gain / (total_distance / 1000)) if total_distance > 0 else 0.0

    # Steepness score: effort-weighted average climbing grade
    steepness_score = (weighted_grade_sum / climbing_work_sum) if climbing_work_sum > 0 else 0.0

    return HillinessAnalysis(
        hilliness_score=hilliness_score,
        steepness_score=steepness_score,
        grade_time_histogram=grade_times,
        grade_distance_histogram=grade_distances,
        total_time=total_time,
        total_distance=total_distance,
        max_grade=max_grade,
        steep_distance=steep_distance,
        very_steep_distance=very_steep_distance,
        steep_time=steep_time,
        very_steep_time=very_steep_time,
        steep_time_histogram=steep_times,
        steep_distance_histogram=steep_distances,
    )
