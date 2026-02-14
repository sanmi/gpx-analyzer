"""Elevation profile data calculation for routes and trips."""

from gpx_analyzer.analyzer import DEFAULT_MAX_GRADE_WINDOW, _calculate_rolling_grades
from gpx_analyzer.cli import calculate_elevation_gain, DEFAULTS
from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.physics import calculate_segment_work
from gpx_analyzer.ridewithgps import _load_config, get_route_with_surface, get_trip_data
from gpx_analyzer.smoothing import smooth_elevations
from gpx_analyzer.tunnel import detect_and_correct_elevation_anomalies


def smooth_speeds(speeds_ms: list, cum_dist: list, window_m: float = 300) -> list:
    """Apply distance-based running average to speed data.

    Args:
        speeds_ms: Speed in m/s for each segment (len = N).
        cum_dist: Cumulative distance at each point (len = N+1).
        window_m: Smoothing window in meters.

    Returns list of smoothed speeds in km/h (len = N).
    """
    n = len(speeds_ms)
    if n == 0:
        return []
    # Compute segment centers
    seg_centers = [(cum_dist[j] + cum_dist[j + 1]) / 2 for j in range(n)]
    half = window_m / 2
    smoothed = []
    left = 0
    running_sum = 0.0
    window_count = 0
    right = 0
    for i in range(n):
        # Expand right pointer to include segments within window
        while right < n and seg_centers[right] <= seg_centers[i] + half:
            running_sum += speeds_ms[right]
            window_count += 1
            right += 1
        # Shrink left pointer to exclude segments outside window
        while left < n and seg_centers[left] < seg_centers[i] - half:
            running_sum -= speeds_ms[left]
            window_count -= 1
            left += 1
        smoothed.append((running_sum / window_count * 3.6) if window_count > 0 else 0.0)
    return smoothed


def scale_elevation_points(points: list[TrackPoint], scale: float) -> list[TrackPoint]:
    """Apply elevation scaling to points without smoothing.

    This is used when trip_smoothing_enabled is False to scale raw elevation
    data to match the API-reported elevation gain.
    """
    if not points or scale == 1.0:
        return points
    first_elev = points[0].elevation or 0
    result = []
    for p in points:
        if p.elevation is not None:
            scaled_elev = first_elev + (p.elevation - first_elev) * scale
        else:
            scaled_elev = None
        result.append(TrackPoint(lat=p.lat, lon=p.lon, elevation=scaled_elev, time=p.time))
    return result


def calculate_route_profile_data(
    url: str,
    params: RiderParams,
    smoothing: float | None = None,
    smoothing_override: bool = False,
    process_elevation_func=None,
) -> dict:
    """Calculate elevation profile data for a route.

    Args:
        url: RideWithGPS route URL
        params: Rider parameters
        smoothing: Smoothing radius in meters (or None for default)
        smoothing_override: If True, skip auto-adjustment for high-noise data
        process_elevation_func: Function to process elevation data (noise detection, smoothing, scaling)

    Returns dict with times_hours, elevations, grades, route_name, etc.
    """
    config = _load_config() or {}
    smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Detect and correct elevation anomalies (tunnels, bridges, etc.)
    points, tunnel_corrections = detect_and_correct_elevation_anomalies(points)

    # Process elevation with noise detection, smoothing, and API scaling
    if process_elevation_func is None:
        raise ValueError("process_elevation_func is required")

    elev_result = process_elevation_func(points, route_metadata, smoothing_radius, smoothing_override)
    scaled_points = elev_result.scaled_points
    unscaled_points = elev_result.unscaled_points
    api_elevation_scale = elev_result.api_elevation_scale

    # Calculate rolling grades from UNSCALED points for accurate per-segment grades
    max_grade_window = config.get("max_grade_window_route", DEFAULT_MAX_GRADE_WINDOW)
    rolling_grades = _calculate_rolling_grades(unscaled_points, max_grade_window)

    # Calculate cumulative time, elevation, and speed at each point
    cum_time = [0.0]
    cum_dist = [0.0]
    elevations = [unscaled_points[0].elevation or 0.0]
    speeds_ms = []
    segment_distances = []
    segment_powers = []
    segment_works = []
    segment_elev_gains = []
    segment_elev_losses = []

    for i in range(1, len(scaled_points)):
        work, dist, elapsed = calculate_segment_work(scaled_points[i-1], scaled_points[i], params)
        cum_time.append(cum_time[-1] + elapsed)
        cum_dist.append(cum_dist[-1] + dist)
        elevations.append(unscaled_points[i].elevation or elevations[-1])
        speeds_ms.append(dist / elapsed if elapsed > 0 else 0.0)
        segment_distances.append(dist)
        segment_powers.append(work / elapsed if elapsed > 0 else 0.0)
        segment_works.append(work)
        unscaled_delta = (unscaled_points[i].elevation or 0) - (unscaled_points[i-1].elevation or 0)
        scaled_delta = unscaled_delta * api_elevation_scale
        segment_elev_gains.append(scaled_delta if scaled_delta > 0 else 0.0)
        segment_elev_losses.append(scaled_delta if scaled_delta < 0 else 0.0)

    # Convert to hours
    times_hours = [t / 3600 for t in cum_time]

    # Use rolling grades
    grades = rolling_grades if rolling_grades else [0.0] * (len(scaled_points) - 1)

    # Smooth speeds
    speeds_kmh = smooth_speeds(speeds_ms, cum_dist, window_m=300)

    route_name = route_metadata.get("name", "Elevation Profile") if route_metadata else "Elevation Profile"

    # Convert anomaly corrections to time ranges
    tunnel_time_ranges = []
    for tc in tunnel_corrections:
        start_time = times_hours[tc.start_idx] if tc.start_idx < len(times_hours) else 0
        end_time = times_hours[tc.end_idx] if tc.end_idx < len(times_hours) else times_hours[-1]
        tunnel_time_ranges.append((start_time, end_time))

    # Coalesce consecutive unpaved points into time ranges
    unpaved_time_ranges = []
    in_unpaved = False
    for i, pt in enumerate(scaled_points):
        if getattr(pt, 'unpaved', False):
            if not in_unpaved:
                unpaved_start = times_hours[i]
                in_unpaved = True
        else:
            if in_unpaved:
                unpaved_time_ranges.append((unpaved_start, times_hours[i]))
                in_unpaved = False
    if in_unpaved:
        unpaved_time_ranges.append((unpaved_start, times_hours[-1]))

    return {
        "times_hours": times_hours,
        "elevations": elevations,
        "grades": grades,
        "speeds_kmh": speeds_kmh,
        "distances": segment_distances,
        "powers": segment_powers,
        "works": segment_works,
        "elev_gains": segment_elev_gains,
        "elev_losses": segment_elev_losses,
        "route_name": route_name,
        "tunnel_time_ranges": tunnel_time_ranges,
        "unpaved_time_ranges": unpaved_time_ranges,
        "noise_ratio": elev_result.noise_ratio,
        "effective_smoothing": elev_result.effective_smoothing,
        "smoothing_auto_adjusted": elev_result.smoothing_auto_adjusted,
        "scaled_points": scaled_points,
    }


def calculate_trip_profile_data(
    url: str,
    collapse_stops: bool = False,
    smoothing: float | None = None,
) -> dict:
    """Calculate elevation profile data for a trip using actual timestamps.

    Args:
        url: RideWithGPS trip URL
        collapse_stops: If True, use cumulative moving time (excludes stops) for x-axis.
        smoothing: Smoothing radius in meters. If None, uses config/default.

    Returns dict with times_hours, elevations, grades, and route_name.
    """
    config = _load_config() or {}
    trip_smoothing_enabled = config.get("trip_smoothing_enabled", DEFAULTS.get("trip_smoothing_enabled", True))

    if trip_smoothing_enabled:
        smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
    else:
        smoothing_radius = 0.0

    max_grade_window = config.get("max_grade_window_route", DEFAULT_MAX_GRADE_WINDOW)

    trip_points, trip_metadata = get_trip_data(url)

    if len(trip_points) < 2:
        raise ValueError("Trip contains fewer than 2 track points")

    # Convert TripPoints to TrackPoints for smoothing
    track_points = []
    for tp in trip_points:
        track_points.append(TrackPoint(
            lat=tp.lat,
            lon=tp.lon,
            elevation=tp.elevation,
            time=tp.timestamp,
        ))

    # Detect and correct elevation anomalies
    track_points, tunnel_corrections = detect_and_correct_elevation_anomalies(track_points)

    # Calculate unscaled points (with or without smoothing)
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0

    if smoothing_radius > 0:
        unscaled_points = smooth_elevations(track_points, smoothing_radius, 1.0)
    else:
        unscaled_points = track_points

    if api_elevation_gain and api_elevation_gain > 0:
        unscaled_gain = calculate_elevation_gain(unscaled_points)
        if unscaled_gain > 0:
            api_elevation_scale = api_elevation_gain / unscaled_gain

    # Apply scaling to get final points
    if api_elevation_scale != 1.0:
        if smoothing_radius > 0:
            scaled_points = smooth_elevations(track_points, smoothing_radius, api_elevation_scale)
        else:
            scaled_points = scale_elevation_points(track_points, api_elevation_scale)
    else:
        scaled_points = unscaled_points

    # Calculate rolling grades from UNSCALED points
    rolling_grades = _calculate_rolling_grades(unscaled_points, max_grade_window)

    # Use actual timestamps for x-axis
    if trip_points[0].timestamp is None:
        raise ValueError("Trip has no timestamp data")

    # Speed threshold for detecting stops
    STOPPED_SPEED_THRESHOLD = 2.0  # km/h
    STOPPED_SPEED_MS = STOPPED_SPEED_THRESHOLD / 3.6

    # Calculate segment speeds, distances, and powers
    segment_speeds = []
    segment_distances = []
    segment_powers = []
    cum_dist = [0.0]
    for i in range(len(trip_points) - 1):
        tp0, tp1 = trip_points[i], trip_points[i + 1]
        dist = haversine_distance(tp0.lat, tp0.lon, tp1.lat, tp1.lon)
        cum_dist.append(cum_dist[-1] + dist)
        segment_distances.append(dist)
        segment_powers.append(tp1.power if tp1.power is not None else (tp0.power if tp0.power is not None else None))
        if tp0.timestamp is not None and tp1.timestamp is not None:
            time_delta = tp1.timestamp - tp0.timestamp
            if time_delta > 0:
                segment_speeds.append(dist / time_delta)
            else:
                segment_speeds.append(0.0)
        else:
            segment_speeds.append(0.0)

    # Build times array
    times_hours = []
    elevations = []

    if collapse_stops:
        moving_time_seconds = 0.0
        times_hours.append(0.0)
        elevations.append(unscaled_points[0].elevation or 0.0)

        for i in range(1, len(trip_points)):
            tp_prev, tp_curr = trip_points[i - 1], trip_points[i]
            if tp_prev.timestamp is not None and tp_curr.timestamp is not None:
                time_delta = tp_curr.timestamp - tp_prev.timestamp
                if i - 1 < len(segment_speeds) and segment_speeds[i - 1] >= STOPPED_SPEED_MS:
                    moving_time_seconds += time_delta
            times_hours.append(moving_time_seconds / 3600)
            elevations.append(unscaled_points[i].elevation or 0.0)
    else:
        base_time = trip_points[0].timestamp
        for i, (tp, up) in enumerate(zip(trip_points, unscaled_points)):
            if tp.timestamp is not None:
                hours = (tp.timestamp - base_time) / 3600
            else:
                hours = times_hours[-1] if times_hours else 0.0
            times_hours.append(hours)
            elevations.append(up.elevation or 0.0)

    # Pre-compute per-segment elevation gains and losses
    segment_elev_gains = []
    segment_elev_losses = []
    for i in range(len(elevations) - 1):
        unscaled_delta = (unscaled_points[i + 1].elevation or 0) - (unscaled_points[i].elevation or 0)
        scaled_delta = unscaled_delta * api_elevation_scale
        segment_elev_gains.append(scaled_delta if scaled_delta > 0 else 0.0)
        segment_elev_losses.append(scaled_delta if scaled_delta < 0 else 0.0)

    grades = rolling_grades if rolling_grades else [0.0] * (len(trip_points) - 1)

    # Mark stopped segments with None grade
    for i in range(len(grades)):
        if i < len(segment_speeds) and segment_speeds[i] < STOPPED_SPEED_MS:
            grades[i] = None

    trip_name = trip_metadata.get("name", "Trip Profile") if trip_metadata else "Trip Profile"

    # Smooth speeds
    speeds_kmh = smooth_speeds(segment_speeds, cum_dist, window_m=300)

    # Convert anomaly corrections to time ranges
    tunnel_time_ranges = []
    for tc in tunnel_corrections:
        start_time = times_hours[tc.start_idx] if tc.start_idx < len(times_hours) else 0
        end_time = times_hours[tc.end_idx] if tc.end_idx < len(times_hours) else times_hours[-1]
        tunnel_time_ranges.append((start_time, end_time))

    return {
        "times_hours": times_hours,
        "elevations": elevations,
        "grades": grades,
        "speeds_kmh": speeds_kmh,
        "distances": segment_distances,
        "powers": segment_powers,
        "elev_gains": segment_elev_gains,
        "elev_losses": segment_elev_losses,
        "route_name": trip_name,
        "tunnel_time_ranges": tunnel_time_ranges,
        "is_collapsed": collapse_stops,
    }
