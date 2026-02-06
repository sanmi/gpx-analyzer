"""Climb detection and analysis for route elevation data.

Automatically detects significant climbs along a route based on elevation changes,
with adjustable sensitivity for handling small dips within larger climbs.
"""

from dataclasses import dataclass

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import RiderParams, TrackPoint


@dataclass
class ClimbInfo:
    """A detected climb segment along a route."""
    climb_id: int              # 1-based sequential number
    start_idx: int             # Track point index where climb starts
    end_idx: int               # Track point index where climb ends (at peak)
    start_km: float            # Distance from route start (km)
    end_km: float              # Distance at end of climb (km)
    start_time_hours: float    # Time from route start (hours)
    end_time_hours: float      # Time at end of climb (hours)
    distance_m: float          # Climb length (meters)
    elevation_gain: float      # Total gain within climb (meters)
    elevation_loss: float      # Loss within climb (meters)
    avg_grade: float           # Average grade (percent)
    max_grade: float           # Maximum grade (percent)
    start_elevation: float     # Elevation at climb start (meters)
    peak_elevation: float      # Highest elevation reached (meters)
    duration_seconds: float    # Estimated time to complete climb
    work_kj: float             # Estimated work (kJ)
    avg_power: float           # Estimated average power (watts)
    avg_speed_kmh: float       # Average speed on climb (km/h)
    label: str                 # Display label ("Climb 1" or named)


@dataclass
class ClimbDetectionResult:
    """Result of climb detection including all climbs and metadata."""
    climbs: list[ClimbInfo]
    sensitivity_m: float       # Descent tolerance used for detection


def slider_to_sensitivity(value: int) -> float:
    """Convert slider value (0-100) to sensitivity in meters.

    Slider 0 = High sensitivity (10m tolerance) - small dips split climbs
    Slider 50 = Default (55m tolerance) - balanced
    Slider 100 = Low sensitivity (100m tolerance) - tolerates large dips

    Args:
        value: Slider value from 0 to 100

    Returns:
        Descent tolerance in meters (10-100m range)
    """
    return 10.0 + (100.0 - 10.0) * (value / 100.0)


def detect_climbs(
    points: list[TrackPoint],
    times_hours: list[float] | None = None,
    sensitivity_m: float = 30.0,
    min_climb_gain: float = 50.0,
    min_climb_distance: float = 500.0,
    grade_threshold: float = 2.0,
    params: RiderParams | None = None,
    segment_works: list[float] | None = None,
    segment_powers: list[float] | None = None,
) -> ClimbDetectionResult:
    """Detect significant climbs along a route.

    Algorithm:
    1. Walk through points tracking cumulative elevation
    2. Climb starts when grade exceeds threshold (default 2%)
    3. Track maximum elevation reached during climb
    4. Climb ends when descended > sensitivity_m below maximum
    5. The end point is set to the peak, not where descent was detected
    6. Apply minimum thresholds (gain, distance) to filter small climbs

    Args:
        points: Track points with elevation data
        times_hours: Optional list of cumulative times in hours for each point
        sensitivity_m: Descent tolerance in meters (higher = more tolerant of dips)
        min_climb_gain: Minimum elevation gain to qualify as a climb (meters)
        min_climb_distance: Minimum horizontal distance to qualify (meters)
        grade_threshold: Minimum grade % to start a climb
        params: Optional rider parameters for work/power calculations
        segment_works: Optional pre-calculated work values per segment (joules)
        segment_powers: Optional pre-calculated power values per segment (watts)

    Returns:
        ClimbDetectionResult with list of detected climbs and metadata
    """
    if len(points) < 3:
        return ClimbDetectionResult(climbs=[], sensitivity_m=sensitivity_m)

    # Calculate cumulative distance
    cum_dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon
        )
        cum_dist.append(cum_dist[-1] + d)

    # Calculate segment grades
    grades = []
    for i in range(len(points) - 1):
        dist = cum_dist[i + 1] - cum_dist[i]
        if dist > 0 and points[i].elevation is not None and points[i + 1].elevation is not None:
            grade = (points[i + 1].elevation - points[i].elevation) / dist * 100
        else:
            grade = 0.0
        grades.append(grade)

    # Use provided times or generate placeholder times based on distance
    if times_hours is None:
        # Estimate time based on average ~15 km/h (placeholder)
        times_hours = [d / 1000 / 15 for d in cum_dist]

    climbs = []
    i = 0

    while i < len(points) - 1:
        # Look for start of climb (grade exceeds threshold)
        if grades[i] < grade_threshold:
            i += 1
            continue

        # Found potential climb start
        start_idx = i
        start_elevation = points[start_idx].elevation or 0.0

        # Track the climb
        max_elevation = start_elevation
        max_elevation_idx = start_idx
        current_elevation = start_elevation

        j = i + 1
        climb_ended = False

        while j < len(points):
            pt = points[j]
            elev = pt.elevation or 0.0

            # If we find a lower elevation than start, reset the start point
            # This ensures the climb starts from the true low point, not a false start
            if elev < start_elevation:
                start_idx = j
                start_elevation = elev
                max_elevation = elev
                max_elevation_idx = j

            # Update max elevation tracking
            if elev > max_elevation:
                max_elevation = elev
                max_elevation_idx = j

            # Check if we've descended too far below the max
            descent_from_max = max_elevation - elev
            if descent_from_max > sensitivity_m:
                climb_ended = True
                break

            current_elevation = elev
            j += 1

        # Set end point at the peak, not where we detected the descent
        end_idx = max_elevation_idx

        # Calculate climb metrics
        climb_distance = cum_dist[end_idx] - cum_dist[start_idx]
        elevation_gain = 0.0
        elevation_loss = 0.0
        max_grade = 0.0

        for k in range(start_idx, end_idx):
            if k < len(grades):
                if grades[k] > max_grade:
                    max_grade = grades[k]

                if points[k].elevation is not None and points[k + 1].elevation is not None:
                    delta = points[k + 1].elevation - points[k].elevation
                    if delta > 0:
                        elevation_gain += delta
                    else:
                        elevation_loss += abs(delta)

        # Check if climb meets minimum thresholds
        if elevation_gain >= min_climb_gain and climb_distance >= min_climb_distance:
            # Calculate average grade
            avg_grade = (elevation_gain / climb_distance * 100) if climb_distance > 0 else 0.0

            # Calculate time and speed
            start_time = times_hours[start_idx]
            end_time = times_hours[end_idx]
            duration_seconds = (end_time - start_time) * 3600
            avg_speed_kmh = (climb_distance / 1000) / (duration_seconds / 3600) if duration_seconds > 0 else 0.0

            # Calculate work and power from pre-calculated segments or estimate
            work_kj = 0.0
            avg_power = 0.0

            # Prefer segment_powers if available (actual physics-calculated power)
            if segment_powers is not None:
                powers = [segment_powers[k] for k in range(start_idx, min(end_idx, len(segment_powers)))
                         if segment_powers[k] is not None and segment_powers[k] > 0]
                if powers:
                    avg_power = sum(powers) / len(powers)
                    work_kj = avg_power * duration_seconds / 1000  # Calculate work from power

            # Fall back to segment_works if available
            if work_kj == 0 and segment_works is not None:
                for k in range(start_idx, min(end_idx, len(segment_works))):
                    work_kj += segment_works[k] / 1000  # Convert J to kJ
                if duration_seconds > 0 and work_kj > 0:
                    avg_power = work_kj * 1000 / duration_seconds

            # Last resort: estimate from elevation gain
            if work_kj == 0 and params is not None:
                work_kj = params.total_mass * 9.81 * elevation_gain / 1000
                if duration_seconds > 0:
                    avg_power = work_kj * 1000 / duration_seconds

            climbs.append(ClimbInfo(
                climb_id=len(climbs) + 1,
                start_idx=start_idx,
                end_idx=end_idx,
                start_km=cum_dist[start_idx] / 1000,
                end_km=cum_dist[end_idx] / 1000,
                start_time_hours=start_time,
                end_time_hours=end_time,
                distance_m=climb_distance,
                elevation_gain=elevation_gain,
                elevation_loss=elevation_loss,
                avg_grade=avg_grade,
                max_grade=max_grade,
                start_elevation=start_elevation,
                peak_elevation=max_elevation,
                duration_seconds=duration_seconds,
                work_kj=work_kj,
                avg_power=avg_power,
                avg_speed_kmh=avg_speed_kmh,
                label=f"Climb {len(climbs) + 1}",
            ))

        # Move past this climb
        if climb_ended:
            i = j
        else:
            i = end_idx + 1 if end_idx > i else i + 1

    return ClimbDetectionResult(climbs=climbs, sensitivity_m=sensitivity_m)


def label_climbs(climbs: list[ClimbInfo], named_climbs: dict[tuple[float, float], str] | None = None) -> list[ClimbInfo]:
    """Apply labels to detected climbs.

    By default, climbs are labeled "Climb 1", "Climb 2", etc.
    Named climbs can be provided to override labels based on location.

    Args:
        climbs: List of detected climbs
        named_climbs: Optional dict mapping (start_km, end_km) ranges to names

    Returns:
        Climbs with updated labels
    """
    if named_climbs is None:
        return climbs

    for climb in climbs:
        for (start_km, end_km), name in named_climbs.items():
            # Check if climb overlaps with named range
            if climb.start_km <= end_km and climb.end_km >= start_km:
                climb.label = name
                break

    return climbs
