"""Simple web interface for GPX analyzer."""

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, render_template, request, Response, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

from gpx_analyzer import __version_date__, get_git_hash
from gpx_analyzer.analyzer import analyze, calculate_hilliness, DEFAULT_MAX_GRADE_WINDOW, DEFAULT_MAX_GRADE_SMOOTHING, GRADE_BINS, GRADE_LABELS, STEEP_GRADE_BINS, STEEP_GRADE_LABELS, _calculate_rolling_grades
from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.physics import calculate_segment_work
from gpx_analyzer.cli import calculate_elevation_gain, calculate_surface_breakdown, DEFAULTS, get_gravel_grade_params
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.ridewithgps import (
    _load_config,
    clear_route_json_cache,
    extract_privacy_code,
    get_collection_route_ids,
    get_route_cache_stats,
    get_route_with_surface,
    get_trip_data,
    is_ridewithgps_collection_url,
    is_ridewithgps_url,
    is_ridewithgps_trip_url,
    TripPoint,
)
from gpx_analyzer.smoothing import smooth_elevations
from gpx_analyzer.tunnel import detect_and_correct_elevation_anomalies
from gpx_analyzer.climb import detect_climbs, slider_to_sensitivity, ClimbInfo
from gpx_analyzer.training import (
    calculate_verbose_metrics,
    VerboseMetrics,
    MAX_COASTING_SPEED_MS,
)
from gpx_analyzer.formatters import (
    format_duration,
    format_duration_long,
    format_time_diff,
    format_diff,
    format_pct_diff,
)
from gpx_analyzer.cache import (
    _get_config_hash,
    AnalysisCache,
    DictCache,
    DiskCache,
    make_profile_cache_key,
    make_ride_profile_cache_key,
    make_climb_cache_key,
    make_profile_data_cache_key,
    make_trip_profile_data_cache_key,
    make_trip_analysis_cache_key,
)
from gpx_analyzer.profile import (
    smooth_speeds,
    scale_elevation_points,
)
from gpx_analyzer.charts import (
    add_speed_overlay,
    add_grade_overlay,
    set_fixed_margins,
    generate_elevation_profile as _generate_elevation_profile_chart,
    generate_ride_profile as _generate_ride_profile_chart,
)


def _get_analytics_config() -> dict:
    """Get analytics configuration for Umami tracking.

    Configure via environment variables (preferred for production):
        UMAMI_WEBSITE_ID - your website ID from Umami
        UMAMI_SCRIPT_URL - optional, defaults to cloud.umami.is

    Or in gpx-analyzer.json:
        "umami_website_id": "your-website-id",
        "umami_script_url": "https://cloud.umami.is/script.js"

    Environment variables take precedence over config file.
    """
    import os
    config = _load_config() or {}
    return {
        "umami_website_id": os.environ.get("UMAMI_WEBSITE_ID") or config.get("umami_website_id"),
        "umami_script_url": os.environ.get("UMAMI_SCRIPT_URL") or config.get("umami_script_url", "https://cloud.umami.is/script.js"),
    }


# Global cache instances
_analysis_cache = AnalysisCache(max_size=175)
PROFILE_CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer" / "profiles"
_profile_image_cache = DiskCache(PROFILE_CACHE_DIR, max_size=525)
_climb_cache = DictCache(max_size=175, ttl_seconds=3600, entry_size_kb=2.0)  # 1 hour TTL
_profile_data_cache = DictCache(max_size=100, entry_size_kb=10.0)
_trip_analysis_cache = DictCache(max_size=50, entry_size_kb=5.0)


# Wrapper functions for backward compatibility

def _get_profile_data_cache_key(url: str, params: "RiderParams", smoothing: float, smoothing_override: bool) -> str:
    """Create a cache key for route profile data."""
    return make_profile_data_cache_key(url, params, smoothing, smoothing_override)


def _get_trip_profile_data_cache_key(url: str, collapse_stops: bool, smoothing: float,
                                      trip_smoothing_enabled: bool = True) -> str:
    """Create a cache key for trip profile data."""
    return make_trip_profile_data_cache_key(url, collapse_stops, smoothing, trip_smoothing_enabled)


def _get_trip_analysis_cache_key(url: str, smoothing: float,
                                  trip_smoothing_enabled: bool = True) -> str:
    """Create a cache key for trip analysis."""
    return make_trip_analysis_cache_key(url, smoothing, trip_smoothing_enabled)


def _get_cached_trip_analysis(cache_key: str) -> dict | None:
    """Get cached trip analysis if available."""
    return _trip_analysis_cache.get(cache_key)


def _cache_trip_analysis(cache_key: str, data: dict) -> None:
    """Cache trip analysis."""
    _trip_analysis_cache.set(cache_key, data)


def _get_cached_profile_data(cache_key: str) -> dict | None:
    """Get cached profile data if available."""
    return _profile_data_cache.get(cache_key)


def _cache_profile_data(cache_key: str, data: dict) -> None:
    """Cache profile data."""
    _profile_data_cache.set(cache_key, data)


def _get_profile_data_cache_stats() -> dict:
    """Return profile data cache statistics."""
    return _profile_data_cache.stats()


def _clear_profile_data_cache() -> int:
    """Clear the profile data cache."""
    return _profile_data_cache.clear()


def _get_trip_analysis_cache_stats() -> dict:
    """Return trip analysis cache statistics."""
    return _trip_analysis_cache.stats()


def _clear_trip_analysis_cache() -> int:
    """Clear the trip analysis cache."""
    return _trip_analysis_cache.clear()


def _make_climb_cache_key(url: str, sensitivity: int, params_hash: str) -> str:
    """Create cache key for climb detection results."""
    return make_climb_cache_key(url, sensitivity, params_hash)


def _get_cached_climbs(cache_key: str) -> tuple[list, float] | None:
    """Get cached climb detection results."""
    result = _climb_cache.get(cache_key)
    if result is not None:
        return result[0], result[1]  # (climbs_json, sensitivity_m)
    return None


def _save_climbs_to_cache(cache_key: str, climbs_json: list, sensitivity_m: float) -> None:
    """Save climb detection results to cache."""
    _climb_cache.set(cache_key, (climbs_json, sensitivity_m))


def _get_climb_cache_stats() -> dict:
    """Return climb cache statistics."""
    return _climb_cache.stats()


def _clear_climb_cache() -> int:
    """Clear the climb detection cache."""
    return _climb_cache.clear()


def _make_ride_profile_cache_key(url: str, sensitivity: int, aspect: float, params_hash: str) -> str:
    """Create cache key for ride profile images."""
    return make_ride_profile_cache_key(url, sensitivity, aspect, params_hash)


def _make_profile_cache_key(url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
                            descent_braking_factor: float = 1.0, collapse_stops: bool = False,
                            max_xlim_hours: float | None = None, descending_power: float = 20.0,
                            overlay: str = "", imperial: bool = False,
                            show_gravel: bool = False,
                            max_ylim: float | None = None,
                            max_speed_ylim: float | None = None,
                            max_grade_ylim: float | None = None,
                            gravel_grade: float = 0.0,
                            smoothing: float = 50.0,
                            min_xlim_hours: float | None = None) -> str:
    """Create a unique cache key for elevation profile parameters."""
    return make_profile_cache_key(url, climbing_power, flat_power, mass, headwind,
                                  descent_braking_factor, collapse_stops, max_xlim_hours,
                                  descending_power, overlay, imperial, show_gravel, max_ylim,
                                  max_speed_ylim, max_grade_ylim, gravel_grade, smoothing, min_xlim_hours)


def _get_cached_profile(cache_key: str) -> bytes | None:
    """Load cached profile image if available."""
    return _profile_image_cache.get(cache_key)


def _save_profile_to_cache(cache_key: str, img_bytes: bytes) -> None:
    """Save profile image to cache."""
    _profile_image_cache.set(cache_key, img_bytes)


def _scale_elevation_points(points: list[TrackPoint], scale: float) -> list[TrackPoint]:
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


app = Flask(__name__,
            template_folder='templates',
            static_folder='static')



def get_defaults():
    """Get default values from config file, falling back to DEFAULTS."""
    config = _load_config() or {}
    return {
        "climbing_power": config.get("climbing_power", DEFAULTS["climbing_power"]),
        "flat_power": config.get("flat_power", DEFAULTS["flat_power"]),
        "descending_power": config.get("descending_power", DEFAULTS["descending_power"]),
        "mass": config.get("mass", DEFAULTS["mass"]),
        "headwind": config.get("headwind", DEFAULTS["headwind"]),
        "descent_braking_factor": config.get("descent_braking_factor", DEFAULTS["descent_braking_factor"]),
        "gravel_grade": config.get("gravel_grade", DEFAULTS["gravel_grade"]),
        "smoothing": config.get("smoothing", DEFAULTS["smoothing"]),
    }


def build_params(
    climbing_power: float, flat_power: float, mass: float, headwind: float,
    descent_braking_factor: float | None = None,
    descending_power: float | None = None,
    gravel_grade: int | None = None,
) -> RiderParams:
    """Build RiderParams from user inputs and config defaults."""
    config = _load_config() or {}

    # Get gravel parameters from grade
    grade = gravel_grade if gravel_grade is not None else config.get("gravel_grade", DEFAULTS["gravel_grade"])
    gravel_params = get_gravel_grade_params(grade, config)

    return RiderParams(
        total_mass=mass,
        cda=config.get("cda", DEFAULTS["cda"]),
        crr=config.get("crr", DEFAULTS["crr"]),
        climbing_power=climbing_power,
        flat_power=flat_power,
        descending_power=descending_power if descending_power is not None else config.get("descending_power", DEFAULTS["descending_power"]),
        coasting_grade_threshold=config.get("coasting_grade", DEFAULTS["coasting_grade"]),
        max_coasting_speed=config.get("max_coast_speed", DEFAULTS["max_coast_speed"]) / 3.6,
        max_coasting_speed_unpaved=config.get("max_coast_speed_unpaved", DEFAULTS["max_coast_speed_unpaved"]) / 3.6,
        headwind=headwind / 3.6,
        climb_threshold_grade=config.get("climb_threshold_grade", DEFAULTS["climb_threshold_grade"]),
        steep_descent_speed=config.get("steep_descent_speed", DEFAULTS["steep_descent_speed"]) / 3.6,
        steep_descent_grade=config.get("steep_descent_grade", DEFAULTS["steep_descent_grade"]),
        straight_descent_speed=config.get("straight_descent_speed", DEFAULTS["straight_descent_speed"]) / 3.6,
        hairpin_speed=config.get("hairpin_speed", DEFAULTS["hairpin_speed"]) / 3.6,
        straight_curvature=config.get("straight_curvature", DEFAULTS["straight_curvature"]),
        hairpin_curvature=config.get("hairpin_curvature", DEFAULTS["hairpin_curvature"]),
        descent_braking_factor=descent_braking_factor if descent_braking_factor is not None else config.get("descent_braking_factor", DEFAULTS["descent_braking_factor"]),
        drivetrain_efficiency=config.get("drivetrain_efficiency", DEFAULTS["drivetrain_efficiency"]),
        unpaved_power_factor=gravel_params["power_factor"],
        gravel_work_multiplier=gravel_params["work_multiplier"],
        gravel_coast_speed_pct=gravel_params["coast_speed_pct"],
    )


# High-noise DEM detection constants
HIGH_NOISE_RATIO_THRESHOLD = 1.8  # raw_gain / api_gain ratio indicating noisy GPS elevation
HIGH_NOISE_SMOOTHING_RADIUS = 300.0  # meters, used when GPS elevation is noisy


@dataclass
class ElevationProcessingResult:
    """Result of processing elevation data with smoothing and scaling."""
    scaled_points: list  # Points with smoothing and API scaling applied
    unscaled_points: list  # Points with smoothing only (for display)
    api_elevation_scale: float  # Scale factor applied to match API elevation
    noise_ratio: float  # raw_gain / api_gain ratio
    effective_smoothing: float  # Actual smoothing radius used
    smoothing_auto_adjusted: bool  # True if smoothing was increased due to noise


def process_elevation_data(
    points: list,
    route_metadata: dict | None,
    user_smoothing: float,
    override_auto_adjust: bool = False,
) -> ElevationProcessingResult:
    """Process elevation data with noise detection, smoothing, and API scaling.

    This function handles:
    1. Detecting high-noise GPS elevation data
    2. Auto-adjusting smoothing radius for noisy data (unless overridden)
    3. Applying API elevation scaling to match DEM-derived elevation gain

    Args:
        points: Track points (after anomaly correction)
        route_metadata: Route metadata containing api elevation_gain
        user_smoothing: User-specified smoothing radius
        override_auto_adjust: If True, skip auto-adjustment and use user_smoothing as-is

    Returns:
        ElevationProcessingResult with processed points and metadata
    """
    api_elevation_gain = route_metadata.get("elevation_gain") if route_metadata else None
    raw_elevation_gain = calculate_elevation_gain(points)

    # Calculate noise ratio
    noise_ratio = 0.0
    if api_elevation_gain and api_elevation_gain > 0 and raw_elevation_gain > 0:
        noise_ratio = raw_elevation_gain / api_elevation_gain

    # Auto-adjust smoothing for high-noise data (unless user explicitly overrides)
    effective_smoothing = user_smoothing
    smoothing_auto_adjusted = False
    if not override_auto_adjust and noise_ratio > HIGH_NOISE_RATIO_THRESHOLD:
        if user_smoothing < HIGH_NOISE_SMOOTHING_RADIUS:
            effective_smoothing = HIGH_NOISE_SMOOTHING_RADIUS
            smoothing_auto_adjusted = True

    # Smooth without scaling first (for display and gain calculation)
    unscaled_points = smooth_elevations(points, effective_smoothing, 1.0)

    # Calculate API-based elevation scale factor
    api_elevation_scale = 1.0
    if api_elevation_gain and api_elevation_gain > 0:
        smoothed_gain = calculate_elevation_gain(unscaled_points)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    # Apply scaling
    if api_elevation_scale != 1.0:
        scaled_points = smooth_elevations(points, effective_smoothing, api_elevation_scale)
    else:
        scaled_points = unscaled_points

    return ElevationProcessingResult(
        scaled_points=scaled_points,
        unscaled_points=unscaled_points,
        api_elevation_scale=api_elevation_scale,
        noise_ratio=noise_ratio,
        effective_smoothing=effective_smoothing,
        smoothing_auto_adjusted=smoothing_auto_adjusted,
    )


def analyze_single_route(url: str, params: RiderParams, smoothing: float | None = None, smoothing_override: bool = False) -> dict:
    """Analyze a single route and return results dict.

    Results are cached based on (url, power, mass, headwind, smoothing) for faster
    repeated access when comparing routes.

    Args:
        url: RideWithGPS route URL
        params: Rider parameters
        smoothing: Smoothing radius in meters (or None for default)
        smoothing_override: If True, skip auto-adjustment for high-noise data
    """
    config = _load_config() or {}
    smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])

    # Fetch route first to get current ETag (uses ETag-based caching internally)
    points, route_metadata = get_route_with_surface(url, params.crr)
    route_etag = route_metadata.get("etag", "")

    # Check analysis cache with ETag in key (invalidates when route changes)
    # Use unpaved_power_factor in cache key as proxy for gravel grade
    cached = _analysis_cache.get(
        url + route_etag, params.climbing_power, params.flat_power, params.total_mass, params.headwind,
        params.descent_braking_factor, params.descending_power, params.unpaved_power_factor, smoothing_radius,
        smoothing_override
    )
    if cached is not None:
        return cached

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Detect and correct elevation anomalies (tunnels, bridges, etc.) in raw elevation data
    points, tunnel_corrections = detect_and_correct_elevation_anomalies(points)

    # Process elevation with noise detection, smoothing, and API scaling
    elev_result = process_elevation_data(points, route_metadata, smoothing_radius, smoothing_override)
    points = elev_result.scaled_points
    unscaled_points = elev_result.unscaled_points
    api_elevation_scale = elev_result.api_elevation_scale

    analysis = analyze(points, params)
    max_grade_window = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])
    max_grade_smoothing = config.get("max_grade_smoothing", DEFAULTS["max_grade_smoothing"])
    hilliness = calculate_hilliness(points, params, unscaled_points, max_grade_window, max_grade_smoothing)

    # Prefer API's unpaved_pct if available, otherwise calculate from track points
    unpaved_pct = route_metadata.get("unpaved_pct") if route_metadata else None
    if unpaved_pct is None:
        surface_breakdown = calculate_surface_breakdown(points)
        if surface_breakdown:
            total_dist = surface_breakdown[0] + surface_breakdown[1]
            if total_dist > 0:
                unpaved_pct = (surface_breakdown[1] / total_dist) * 100

    result = {
        "name": route_metadata.get("name") if route_metadata else None,
        "distance_km": analysis.total_distance / 1000,
        "distance_mi": analysis.total_distance / 1000 * 0.621371,
        "elevation_m": analysis.elevation_gain,
        "elevation_ft": analysis.elevation_gain * 3.28084,
        "elevation_loss_m": analysis.elevation_loss,
        "elevation_loss_ft": analysis.elevation_loss * 3.28084,
        "time_str": format_duration_long(analysis.moving_time.total_seconds()),
        "time_seconds": analysis.moving_time.total_seconds(),
        "avg_speed_kmh": analysis.avg_speed * 3.6,
        "avg_speed_mph": analysis.avg_speed * 3.6 * 0.621371,
        "work_kj": analysis.estimated_work / 1000,
        "avg_watts": (analysis.estimated_work / analysis.moving_time.total_seconds()) if analysis.moving_time.total_seconds() > 0 else None,
        "has_power": True,  # Routes have derived avg power from physics model
        # Terrain-specific power (only for trips with recorded power data)
        "avg_power_climbing": None,
        "avg_power_flat": None,
        "avg_power_descending": None,
        "braking_factor": None,
        "unpaved_pct": unpaved_pct,
        "elevation_scale": api_elevation_scale,
        "elevation_scaled": abs(api_elevation_scale - 1.0) > 0.05,
        "noise_ratio": elev_result.noise_ratio,
        "effective_smoothing": elev_result.effective_smoothing,
        "smoothing_auto_adjusted": elev_result.smoothing_auto_adjusted,
        "hilliness_score": hilliness.hilliness_score,
        "steepness_score": hilliness.steepness_score,
        "grade_histogram": hilliness.grade_time_histogram,
        "grade_distance_histogram": hilliness.grade_distance_histogram,
        "max_grade": hilliness.max_grade,
        "steep_distance": hilliness.steep_distance,
        "very_steep_distance": hilliness.very_steep_distance,
        "steep_time_histogram": hilliness.steep_time_histogram,
        "steep_distance_histogram": hilliness.steep_distance_histogram,
        "steep_time_seconds": hilliness.steep_time,
        "very_steep_time_seconds": hilliness.very_steep_time,
        "hilliness_total_time": hilliness.total_time,
        "hilliness_total_distance": hilliness.total_distance,
        # Anomaly corrections
        "tunnels_corrected": len(tunnel_corrections),
        "tunnel_corrections": [
            {
                "start_km": t.start_km,
                "end_km": t.end_km,
                "artificial_gain": t.artificial_gain,
            }
            for t in tunnel_corrections
        ],
    }

    # Store in cache (include ETag so cache invalidates when route changes)
    _analysis_cache.set(
        url + route_etag, params.climbing_power, params.flat_power, params.total_mass, params.headwind,
        params.descent_braking_factor, params.descending_power, params.unpaved_power_factor, smoothing_radius,
        smoothing_override, result
    )

    return result


def analyze_trip(url: str, smoothing: float | None = None) -> dict:
    """Analyze a recorded trip - actual values, no estimation needed.

    Args:
        url: RideWithGPS trip URL
        smoothing: Smoothing radius in meters. If None, uses config/default.

    Returns:
        Dict with trip analysis results including actual recorded time and power.
        Results are cached in memory for faster repeated access.
    """
    config = _load_config() or {}
    trip_smoothing_enabled = config.get("trip_smoothing_enabled", DEFAULTS.get("trip_smoothing_enabled", True))

    if trip_smoothing_enabled:
        smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
    else:
        smoothing_radius = 0.0  # No smoothing for trips

    # Check cache first (include trip_smoothing_enabled in key)
    cache_key = _get_trip_analysis_cache_key(url, smoothing_radius, trip_smoothing_enabled)
    cached = _get_cached_trip_analysis(cache_key)
    if cached is not None:
        return cached
    max_grade_window = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])

    trip_points, trip_metadata = get_trip_data(url)

    if len(trip_points) < 2:
        raise ValueError("Trip contains fewer than 2 track points")

    # Convert TripPoints to TrackPoints for grade calculation
    track_points = []
    for tp in trip_points:
        track_points.append(TrackPoint(
            lat=tp.lat,
            lon=tp.lon,
            elevation=tp.elevation,
            time=tp.timestamp,
        ))

    # Detect and correct elevation anomalies (tunnels, bridges, etc.) in elevation data
    track_points, tunnel_corrections = detect_and_correct_elevation_anomalies(track_points)

    # Calculate unscaled points (with or without smoothing)
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0

    if smoothing_radius > 0:
        # Apply smoothing for elevation profile and grade calculations
        unscaled = smooth_elevations(track_points, smoothing_radius, 1.0)
    else:
        # No smoothing - use raw track points
        unscaled = track_points

    if api_elevation_gain and api_elevation_gain > 0:
        unscaled_gain = calculate_elevation_gain(unscaled)
        if unscaled_gain > 0:
            api_elevation_scale = api_elevation_gain / unscaled_gain

    # Apply scaling to get final points
    if smoothing_radius > 0:
        smoothed_points = smooth_elevations(track_points, smoothing_radius, api_elevation_scale)
    else:
        # Scale raw points
        smoothed_points = _scale_elevation_points(track_points, api_elevation_scale)

    # Calculate rolling grades for max grade (filters GPS noise)
    # Use unscaled points (already computed above with user's smoothing setting)
    # No extra smoothing - matches how routes work and elevation profile display
    rolling_grades = _calculate_rolling_grades(unscaled, max_grade_window)
    max_grade = max(rolling_grades) if rolling_grades else 0.0

    # Calculate grade histograms using actual timestamps from trip
    grade_times = {label: 0.0 for label in GRADE_LABELS}
    grade_distances = {label: 0.0 for label in GRADE_LABELS}
    steep_times = {label: 0.0 for label in STEEP_GRADE_LABELS}
    steep_distances = {label: 0.0 for label in STEEP_GRADE_LABELS}

    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0
    steep_distance = 0.0
    very_steep_distance = 0.0
    steep_time = 0.0
    very_steep_time = 0.0

    # For steepness calculation (effort-weighted using time instead of work for trips)
    weighted_grade_sum = 0.0
    climbing_time_sum = 0.0

    for i in range(1, len(trip_points)):
        prev, curr = trip_points[i - 1], trip_points[i]
        prev_smooth, curr_smooth = smoothed_points[i - 1], smoothed_points[i]

        # Distance between consecutive points
        dist = curr.distance - prev.distance if curr.distance is not None and prev.distance is not None else 0.0
        if dist <= 0:
            # Fall back to haversine distance
            dist = haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)

        if dist < 0.1:
            continue

        # Time delta from actual timestamps
        if curr.timestamp is not None and prev.timestamp is not None:
            time_delta = curr.timestamp - prev.timestamp
        else:
            time_delta = 0.0

        # Skip stopped segments: if speed < 1 km/h (0.278 m/s), consider it stopped
        if time_delta > 0 and dist / time_delta < 0.278:
            continue

        total_distance += dist

        # Elevation change from smoothed data
        elev_prev = prev_smooth.elevation if prev_smooth.elevation is not None else 0.0
        elev_curr = curr_smooth.elevation if curr_smooth.elevation is not None else 0.0
        elev_change = elev_curr - elev_prev

        if elev_change > 0:
            elevation_gain += elev_change
        else:
            elevation_loss += abs(elev_change)

        # Use rolling grade for histogram binning
        rolling_grade = rolling_grades[i - 1] if i - 1 < len(rolling_grades) else 0.0

        # Bin the grade
        for j in range(len(GRADE_BINS) - 1):
            if GRADE_BINS[j] <= rolling_grade < GRADE_BINS[j + 1]:
                grade_times[GRADE_LABELS[j]] += time_delta
                grade_distances[GRADE_LABELS[j]] += dist
                break

        # Track steep distances and times
        if rolling_grade >= 10:
            steep_distance += dist
            steep_time += time_delta
        if rolling_grade >= 15:
            very_steep_distance += dist
            very_steep_time += time_delta

        # Bin steep grades (>= 10%)
        if rolling_grade >= 10:
            for j in range(len(STEEP_GRADE_BINS) - 1):
                if STEEP_GRADE_BINS[j] <= rolling_grade < STEEP_GRADE_BINS[j + 1]:
                    steep_times[STEEP_GRADE_LABELS[j]] += time_delta
                    steep_distances[STEEP_GRADE_LABELS[j]] += dist
                    break

        # Accumulate steepness data for climbing segments >= 2%
        if rolling_grade >= 2 and time_delta > 0:
            weighted_grade_sum += rolling_grade * time_delta
            climbing_time_sum += time_delta

    # Hilliness score: meters gained per km
    hilliness_score = (elevation_gain / (total_distance / 1000)) if total_distance > 0 else 0.0

    # Steepness score: time-weighted average climbing grade
    steepness_score = (weighted_grade_sum / climbing_time_sum) if climbing_time_sum > 0 else 0.0

    # Calculate total time from timestamps (sum of all time deltas in grade histogram)
    calculated_moving_time = sum(grade_times.values())

    # Calculate avg_watts from track points if not in metadata
    points_with_power = [p for p in trip_points if p.power is not None]
    if points_with_power:
        calculated_avg_watts = sum(p.power for p in points_with_power) / len(points_with_power)
    else:
        calculated_avg_watts = None

    # Get metadata values, fall back to calculated values
    moving_time = trip_metadata.get("moving_time") or calculated_moving_time
    distance = trip_metadata.get("distance") or total_distance
    avg_speed = trip_metadata.get("avg_speed")  # m/s
    avg_watts = trip_metadata.get("avg_watts") or calculated_avg_watts

    # Compute elapsed time from timestamps (includes stops)
    if trip_points[-1].timestamp is not None and trip_points[0].timestamp is not None:
        elapsed_time = trip_points[-1].timestamp - trip_points[0].timestamp
    else:
        elapsed_time = moving_time

    # Calculate avg_speed from distance/time if not in metadata
    if avg_speed is None and moving_time > 0:
        avg_speed = distance / moving_time  # m/s

    # Calculate work if we have power and time
    work_kj = None
    if avg_watts is not None and moving_time > 0:
        work_kj = (avg_watts * moving_time) / 1000  # kJ

    # Calculate terrain-specific power metrics (reuse training.py logic)
    verbose_metrics = calculate_verbose_metrics(trip_points, MAX_COASTING_SPEED_MS)

    result = {
        "name": trip_metadata.get("name"),
        "distance_km": distance / 1000,
        "distance_mi": distance / 1000 * 0.621371,
        "elevation_m": trip_metadata.get("elevation_gain") or elevation_gain,
        "elevation_ft": (trip_metadata.get("elevation_gain") or elevation_gain) * 3.28084,
        "elevation_loss_m": elevation_loss,
        "elevation_loss_ft": elevation_loss * 3.28084,
        "time_str": format_duration_long(moving_time),
        "time_seconds": moving_time,
        "elapsed_time_seconds": elapsed_time,
        "avg_speed_kmh": (avg_speed * 3.6) if avg_speed else 0,
        "avg_speed_mph": (avg_speed * 3.6 * 0.621371) if avg_speed else 0,
        "work_kj": work_kj,
        "avg_watts": avg_watts,
        "has_power": avg_watts is not None,
        "unpaved_pct": None,  # Trips don't have surface data
        "elevation_scale": api_elevation_scale,
        "elevation_scaled": abs(api_elevation_scale - 1.0) > 0.05,
        "hilliness_score": hilliness_score,
        "steepness_score": steepness_score,
        "grade_histogram": grade_times,
        "grade_distance_histogram": grade_distances,
        "max_grade": max_grade,
        "steep_distance": steep_distance,
        "very_steep_distance": very_steep_distance,
        "steep_time_histogram": steep_times,
        "steep_distance_histogram": steep_distances,
        "steep_time_seconds": steep_time,
        "very_steep_time_seconds": very_steep_time,
        "hilliness_total_time": sum(grade_times.values()),
        "hilliness_total_distance": total_distance,
        # Trip-specific flags
        "is_trip": True,
        # Terrain-specific power (calculated from trip data)
        "avg_power_climbing": verbose_metrics.avg_power_climbing,
        "avg_power_flat": verbose_metrics.avg_power_flat,
        "avg_power_descending": verbose_metrics.avg_power_descending,
        "braking_factor": verbose_metrics.braking_score,  # Renamed for UI clarity
        # Anomaly corrections
        "tunnels_corrected": len(tunnel_corrections),
        "tunnel_corrections": [
            {
                "start_km": t.start_km,
                "end_km": t.end_km,
                "artificial_gain": t.artificial_gain,
            }
            for t in tunnel_corrections
        ],
    }

    # Cache result for faster repeated access
    _cache_trip_analysis(cache_key, result)

    return result


def _get_elevation_profile_cache_stats() -> dict:
    """Get statistics for the elevation profile disk cache."""
    stats = _profile_image_cache.stats()
    # Calculate actual disk usage
    total_bytes = 0
    index = _profile_image_cache._load_index()
    for key in index:
        path = PROFILE_CACHE_DIR / f"{key}.png"
        if path.exists():
            total_bytes += path.stat().st_size
    stats["disk_kb"] = round(total_bytes / 1024, 1)
    return stats


def _clear_elevation_profile_cache() -> int:
    """Clear the elevation profile disk cache. Returns number of files removed."""
    return _profile_image_cache.clear()


# PWA routes - serve manifest and service worker from root
@app.route("/manifest.json")
def manifest():
    """Serve PWA manifest."""
    return send_file("static/manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    """Serve service worker from root path (required for scope)."""
    return send_file("static/sw.js", mimetype="application/javascript")


@app.route("/saved")
def saved_routes():
    """Page to manage saved routes for offline use."""
    analytics = _get_analytics_config()
    return render_template("saved.html",
                           umami_website_id=analytics.get("umami_website_id"),
                           umami_script_url=analytics.get("umami_script_url"))


@app.route("/cache-stats")
def cache_stats():
    """Return cache statistics as JSON for all caches."""
    analysis = _analysis_cache.stats()
    elevation_profile = _get_elevation_profile_cache_stats()
    profile_data = _get_profile_data_cache_stats()
    trip_analysis = _get_trip_analysis_cache_stats()
    climb = _get_climb_cache_stats()
    route_json = get_route_cache_stats()

    # Calculate totals (memory caches use memory_kb, disk caches use disk_kb)
    total_memory_kb = analysis["memory_kb"] + climb["memory_kb"] + profile_data["memory_kb"] + trip_analysis["memory_kb"]
    total_disk_kb = elevation_profile["disk_kb"] + route_json["disk_kb"]

    return {
        "analysis_cache": analysis,
        "elevation_profile_cache": elevation_profile,
        "profile_data_cache": profile_data,
        "trip_analysis_cache": trip_analysis,
        "climb_cache": climb,
        "route_json_cache": route_json,
        "totals": {
            "memory_kb": total_memory_kb,
            "disk_kb": total_disk_kb,
        },
    }


@app.route("/cache-clear", methods=["GET", "POST"])
def cache_clear():
    """Clear all caches: analysis, elevation profiles, profile data, trip analysis, climb detection, and route JSON."""
    analysis_size = _analysis_cache.stats()["size"]
    _analysis_cache.clear()
    profiles_cleared = _clear_elevation_profile_cache()
    profile_data_cleared = _clear_profile_data_cache()
    trip_analysis_cleared = _clear_trip_analysis_cache()
    climbs_cleared = _clear_climb_cache()
    routes_cleared = clear_route_json_cache()
    return {
        "status": "ok",
        "message": f"Caches cleared: analysis ({analysis_size}), elevation_profiles ({profiles_cleared}), profile_data ({profile_data_cleared}), trip_analysis ({trip_analysis_cleared}), climbs ({climbs_cleared}), routes ({routes_cleared})"
    }


@app.route("/analyze-collection-stream")
def analyze_collection_stream():
    """SSE endpoint for streaming collection analysis progress."""
    defaults = get_defaults()
    url = request.args.get("url", "")
    try:
        climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
        flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
        descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
        mass = float(request.args.get("mass", defaults["mass"]))
        headwind = float(request.args.get("headwind", defaults["headwind"]))
        descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
        gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
        smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    except ValueError:
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid parameters'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    if not is_ridewithgps_collection_url(url):
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid collection URL'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    def generate():
        try:
            route_ids, collection_name = get_collection_route_ids(url)

            if not route_ids:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No routes found in collection'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'start', 'name': collection_name, 'total': len(route_ids)})}\n\n"

            params = build_params(climbing_power, flat_power, mass, headwind, descending_power=descending_power, descent_braking_factor=descent_braking_factor, gravel_grade=gravel_grade)

            for i, route_id in enumerate(route_ids):
                route_url = f"https://ridewithgps.com/routes/{route_id}"

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(route_ids)})}\n\n"

                try:
                    route_result = analyze_single_route(route_url, params, smoothing)
                    route_result["time_str"] = format_duration(route_result["time_seconds"])
                    route_result["route_id"] = route_id
                    route_result["steep_time_seconds"] = sum(route_result.get("steep_time_histogram", {}).values())

                    yield f"data: {json.dumps({'type': 'route', 'route': route_result, 'total': len(route_ids)})}\n\n"
                except Exception as e:
                    # Skip failed routes but continue
                    print(f"Error analyzing route {route_id}: {e}")
                    continue

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/og-image")
def og_image():
    """Generate Open Graph preview image with time-at-grade histogram."""
    defaults = get_defaults()
    url = request.args.get("url", "")

    if not url or not is_ridewithgps_url(url):
        # Return a simple fallback image
        return generate_fallback_image()

    try:
        climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
        flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
        descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
        mass = float(request.args.get("mass", defaults["mass"]))
        headwind = float(request.args.get("headwind", defaults["headwind"]))
        smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    except ValueError:
        return generate_fallback_image()

    try:
        params = build_params(climbing_power, flat_power, mass, headwind, descending_power=descending_power)
        result = analyze_single_route(url, params, smoothing)
        return generate_histogram_image(result)
    except Exception:
        return generate_fallback_image()


def generate_fallback_image():
    """Generate a simple fallback OG image."""
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#f5f5f7')
    ax.set_facecolor('#f5f5f7')
    ax.text(0.5, 0.5, 'Reality Check\nmy Route',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='#FF6B35', transform=ax.transAxes)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#f5f5f7', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def generate_histogram_image(result: dict):
    """Generate histogram image for Open Graph preview."""
    histogram = result.get("grade_histogram", {})
    if not histogram:
        return generate_fallback_image()

    labels = ['<-10', '-10', '-8', '-6', '-4', '-2', '0', '+2', '+4', '+6', '+8', '>10']
    values = list(histogram.values())
    total = sum(values)
    percentages = [(v / total * 100) if total > 0 else 0 for v in values]

    # Create figure with route info
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white')

    # Color bars by grade (red for steep up, blue for steep down, gray for flat)
    colors = ['#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
              '#cccccc',
              '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00']

    bars = ax.bar(range(len(labels)), percentages, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Time %', fontsize=9)
    ax.set_xlabel('Grade %', fontsize=9)

    # Add route summary as title
    name = result.get("name", "Route Analysis")
    dist_km = result.get("distance_km", 0)
    elev_m = result.get("elevation_m", 0)
    time_str = result.get("time_str", "")

    title = f"{name}\n{dist_km:.0f} km • {elev_m:.0f}m climbing • {time_str}"
    ax.set_title(title, fontsize=10, fontweight='bold', color='#333', pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


# Wrapper functions that delegate to charts module
def _add_speed_overlay(ax, times_hours: list, speeds_kmh: list, imperial: bool = False,
                       max_speed_ylim: float | None = None):
    """Add speed line overlay with right Y-axis."""
    add_speed_overlay(ax, times_hours, speeds_kmh, imperial, max_speed_ylim)


def _add_grade_overlay(ax, times_hours: list, grades: list, max_grade_ylim: float | None = None):
    """Add grade line overlay with right Y-axis."""
    add_grade_overlay(ax, times_hours, grades, max_grade_ylim)


def _set_fixed_margins(fig, fig_width: float, fig_height: float) -> None:
    """Set fixed margins for consistent coordinate mapping."""
    set_fixed_margins(fig, fig_width, fig_height)


def _smooth_speeds(speeds_ms: list, cum_dist: list, window_m: float = 300) -> list:
    """Apply distance-based running average to speed data."""
    return smooth_speeds(speeds_ms, cum_dist, window_m)


def _scale_elevation_points(points: list[TrackPoint], scale: float) -> list[TrackPoint]:
    """Apply elevation scaling to points without smoothing."""
    return scale_elevation_points(points, scale)


def _calculate_elevation_profile_data(url: str, params: RiderParams, smoothing: float | None = None, smoothing_override: bool = False) -> dict:
    """Calculate elevation profile data for a route.

    Returns dict with times_hours, elevations, grades, route_name, and tunnel_corrections.
    Results are cached in memory for faster repeated access.
    """
    config = _load_config() or {}
    smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])

    # Check cache first
    cache_key = _get_profile_data_cache_key(url, params, smoothing_radius, smoothing_override)
    cached = _get_cached_profile_data(cache_key)
    if cached is not None:
        return cached

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Detect and correct elevation anomalies (tunnels, bridges, etc.)
    points, tunnel_corrections = detect_and_correct_elevation_anomalies(points)

    # Process elevation with noise detection, smoothing, and API scaling
    elev_result = process_elevation_data(points, route_metadata, smoothing_radius, smoothing_override)
    scaled_points = elev_result.scaled_points
    unscaled_points = elev_result.unscaled_points
    api_elevation_scale = elev_result.api_elevation_scale

    # Calculate rolling grades from UNSCALED points for accurate per-segment grades
    # Scaling is only used for physics (work/power) and total gain matching API
    # Using scaled points would incorrectly reduce grades on noisy routes
    # unscaled_points are already smoothed with user's smoothing setting
    max_grade_window = config.get("max_grade_window_route", DEFAULT_MAX_GRADE_WINDOW)
    rolling_grades = _calculate_rolling_grades(unscaled_points, max_grade_window)

    # Calculate cumulative time, elevation, and speed at each point
    # Use unscaled elevations for Y-axis display (accurate absolute heights)
    # Use scaled points for physics (speed/time) and gain/loss calculations
    cum_time = [0.0]
    cum_dist = [0.0]
    elevations = [unscaled_points[0].elevation or 0.0]
    speeds_ms = []
    segment_distances = []
    segment_powers = []
    segment_works = []  # Work in joules for accurate summing during downsampling
    segment_elev_gains = []
    segment_elev_losses = []

    for i in range(1, len(scaled_points)):
        work, dist, elapsed = calculate_segment_work(scaled_points[i-1], scaled_points[i], params)
        cum_time.append(cum_time[-1] + elapsed)
        cum_dist.append(cum_dist[-1] + dist)
        # Display elevation from unscaled points (accurate absolute heights)
        elevations.append(unscaled_points[i].elevation or elevations[-1])
        speeds_ms.append(dist / elapsed if elapsed > 0 else 0.0)
        segment_distances.append(dist)
        segment_powers.append(work / elapsed if elapsed > 0 else 0.0)
        segment_works.append(work)  # Store actual work for accurate totals
        # Gain/loss scaled to match API-reported totals (same as summary)
        # This ensures selection popup values match summary when selecting entire route
        unscaled_delta = (unscaled_points[i].elevation or 0) - (unscaled_points[i-1].elevation or 0)
        scaled_delta = unscaled_delta * api_elevation_scale
        segment_elev_gains.append(scaled_delta if scaled_delta > 0 else 0.0)
        segment_elev_losses.append(scaled_delta if scaled_delta < 0 else 0.0)

    # Convert to hours
    times_hours = [t / 3600 for t in cum_time]

    # Use rolling grades (one fewer than points, like segment grades)
    grades = rolling_grades if rolling_grades else [0.0] * (len(scaled_points) - 1)

    # Smooth speeds
    speeds_kmh = _smooth_speeds(speeds_ms, cum_dist, window_m=300)

    route_name = route_metadata.get("name", "Elevation Profile") if route_metadata else "Elevation Profile"

    # Convert anomaly corrections to time ranges for highlighting
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

    result = {
        "times_hours": times_hours,
        "elevations": elevations,
        "grades": grades,
        "speeds_kmh": speeds_kmh,
        "distances": segment_distances,
        "powers": segment_powers,
        "works": segment_works,  # Work in joules for accurate selection totals
        "elev_gains": segment_elev_gains,
        "elev_losses": segment_elev_losses,
        "route_name": route_name,
        "tunnel_time_ranges": tunnel_time_ranges,
        "unpaved_time_ranges": unpaved_time_ranges,
        "noise_ratio": elev_result.noise_ratio,
        "effective_smoothing": elev_result.effective_smoothing,
        "smoothing_auto_adjusted": elev_result.smoothing_auto_adjusted,
        "scaled_points": scaled_points,  # For climb detection consistency
    }

    # Cache result for faster repeated access
    _cache_profile_data(cache_key, result)

    return result


def _calculate_trip_elevation_profile_data(url: str, collapse_stops: bool = False,
                                           smoothing: float | None = None) -> dict:
    """Calculate elevation profile data for a trip using actual timestamps.

    Args:
        url: RideWithGPS trip URL
        collapse_stops: If True, use cumulative moving time (excludes stops) for x-axis.
                       This makes the profile comparable to route profiles.
        smoothing: Smoothing radius in meters. If None, uses config/default.

    Returns dict with times_hours, elevations, grades, and route_name.
    Results are cached in memory for faster repeated access.
    """
    config = _load_config() or {}
    trip_smoothing_enabled = config.get("trip_smoothing_enabled", DEFAULTS.get("trip_smoothing_enabled", True))

    if trip_smoothing_enabled:
        smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
    else:
        smoothing_radius = 0.0  # No smoothing for trips

    # Check cache first (include trip_smoothing_enabled in key)
    cache_key = _get_trip_profile_data_cache_key(url, collapse_stops, smoothing_radius, trip_smoothing_enabled)
    cached = _get_cached_profile_data(cache_key)
    if cached is not None:
        return cached
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

    # Detect and correct elevation anomalies (tunnels, bridges, etc.)
    track_points, tunnel_corrections = detect_and_correct_elevation_anomalies(track_points)

    # Calculate unscaled points (with or without smoothing)
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0

    if smoothing_radius > 0:
        unscaled_points = smooth_elevations(track_points, smoothing_radius, 1.0)
    else:
        # No smoothing - use raw track points
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
            scaled_points = _scale_elevation_points(track_points, api_elevation_scale)
    else:
        scaled_points = unscaled_points

    # Calculate rolling grades from UNSCALED points for accurate per-segment grades
    # Scaling only affects physics/total gain, not per-segment grades shown to users
    # unscaled_points are already smoothed with user's smoothing setting
    rolling_grades = _calculate_rolling_grades(unscaled_points, max_grade_window)

    # Use actual timestamps for x-axis
    if trip_points[0].timestamp is None:
        raise ValueError("Trip has no timestamp data")

    # Speed threshold for detecting stops
    STOPPED_SPEED_THRESHOLD = 2.0  # km/h
    STOPPED_SPEED_MS = STOPPED_SPEED_THRESHOLD / 3.6  # m/s

    # Calculate segment speeds, distances, and powers
    segment_speeds = []  # speed in m/s for each segment
    segment_distances = []  # distance in meters for each segment
    segment_powers = []  # power in watts for each segment
    cum_dist = [0.0]
    for i in range(len(trip_points) - 1):
        tp0, tp1 = trip_points[i], trip_points[i + 1]
        dist = haversine_distance(tp0.lat, tp0.lon, tp1.lat, tp1.lon)
        cum_dist.append(cum_dist[-1] + dist)
        segment_distances.append(dist)
        # Use recorded power from trip point if available
        segment_powers.append(tp1.power if tp1.power is not None else (tp0.power if tp0.power is not None else None))
        if tp0.timestamp is not None and tp1.timestamp is not None:
            time_delta = tp1.timestamp - tp0.timestamp
            if time_delta > 0:
                segment_speeds.append(dist / time_delta)
            else:
                segment_speeds.append(0.0)
        else:
            segment_speeds.append(0.0)

    # Build times array - use unscaled elevations for display (accurate absolute heights)
    times_hours = []
    elevations = []

    if collapse_stops:
        # Use cumulative moving time (excludes stopped segments)
        moving_time_seconds = 0.0
        times_hours.append(0.0)
        elevations.append(unscaled_points[0].elevation or 0.0)

        for i in range(1, len(trip_points)):
            tp_prev, tp_curr = trip_points[i - 1], trip_points[i]
            if tp_prev.timestamp is not None and tp_curr.timestamp is not None:
                time_delta = tp_curr.timestamp - tp_prev.timestamp
                # Only add time if moving (speed >= threshold)
                if i - 1 < len(segment_speeds) and segment_speeds[i - 1] >= STOPPED_SPEED_MS:
                    moving_time_seconds += time_delta
            times_hours.append(moving_time_seconds / 3600)
            elevations.append(unscaled_points[i].elevation or 0.0)
    else:
        # Use actual elapsed time
        base_time = trip_points[0].timestamp
        for i, (tp, up) in enumerate(zip(trip_points, unscaled_points)):
            if tp.timestamp is not None:
                hours = (tp.timestamp - base_time) / 3600
            else:
                hours = times_hours[-1] if times_hours else 0.0
            times_hours.append(hours)
            elevations.append(up.elevation or 0.0)

    # Pre-compute per-segment elevation gains and losses
    # Scale to match API-reported totals so selection popup matches summary
    segment_elev_gains = []
    segment_elev_losses = []
    for i in range(len(elevations) - 1):
        unscaled_delta = (unscaled_points[i + 1].elevation or 0) - (unscaled_points[i].elevation or 0)
        scaled_delta = unscaled_delta * api_elevation_scale
        segment_elev_gains.append(scaled_delta if scaled_delta > 0 else 0.0)
        segment_elev_losses.append(scaled_delta if scaled_delta < 0 else 0.0)

    grades = rolling_grades if rolling_grades else [0.0] * (len(trip_points) - 1)

    # Mark stopped segments with None grade (for visual indication)
    for i in range(len(grades)):
        if i < len(segment_speeds) and segment_speeds[i] < STOPPED_SPEED_MS:
            grades[i] = None  # Mark as stopped

    trip_name = trip_metadata.get("name", "Trip Profile") if trip_metadata else "Trip Profile"

    # Smooth speeds
    speeds_kmh = _smooth_speeds(segment_speeds, cum_dist, window_m=300)

    # Convert anomaly corrections to time ranges for highlighting
    tunnel_time_ranges = []
    for tc in tunnel_corrections:
        start_time = times_hours[tc.start_idx] if tc.start_idx < len(times_hours) else 0
        end_time = times_hours[tc.end_idx] if tc.end_idx < len(times_hours) else times_hours[-1]
        tunnel_time_ranges.append((start_time, end_time))

    result = {
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

    # Cache result for faster repeated access
    _cache_profile_data(cache_key, result)

    return result


def generate_elevation_profile(url: str, params: RiderParams, title_time_hours: float | None = None,
                               max_xlim_hours: float | None = None,
                               overlay: str | None = None, imperial: bool = False,
                               max_ylim: float | None = None,
                               max_speed_ylim: float | None = None,
                               max_grade_ylim: float | None = None,
                               show_gravel: bool = False,
                               smoothing: float | None = None,
                               aspect_ratio: float = 3.5,
                               min_xlim_hours: float | None = None) -> bytes:
    """Generate elevation profile image with grade-based coloring.

    Args:
        url: RideWithGPS route URL
        params: Rider parameters
        title_time_hours: Optional time to display in title (from calibrated analysis).
        max_xlim_hours: Optional max x-axis limit in hours (for synchronized comparison).
        overlay: Optional overlay type (e.g. "speed").
        imperial: If True, use imperial units for overlay axis.
        max_ylim: Optional max y-axis limit in meters (for synchronized comparison).
        max_speed_ylim: Optional max speed y-axis limit in km/h (for synchronized comparison).
        max_grade_ylim: Optional max grade y-axis limit in % (for synchronized comparison).
        show_gravel: If True, highlight unpaved/gravel sections with a brown strip.
        smoothing: Optional override for elevation smoothing radius.
        aspect_ratio: Width/height ratio (1.0 = square, 3.5 = wide default).
        min_xlim_hours: Optional min x-axis limit in hours (for zooming).

    Returns PNG image as bytes.
    """
    data = _calculate_elevation_profile_data(url, params, smoothing)
    return _generate_elevation_profile_chart(
        data,
        overlay=overlay,
        imperial=imperial,
        max_ylim=max_ylim,
        max_speed_ylim=max_speed_ylim,
        max_grade_ylim=max_grade_ylim,
        show_gravel=show_gravel,
        aspect_ratio=aspect_ratio,
        min_xlim_hours=min_xlim_hours,
        max_xlim_hours=max_xlim_hours,
    )


def generate_trip_elevation_profile(url: str, title_time_hours: float | None = None, collapse_stops: bool = False,
                                    max_xlim_hours: float | None = None,
                                    overlay: str | None = None, imperial: bool = False,
                                    max_ylim: float | None = None,
                                    max_speed_ylim: float | None = None,
                                    max_grade_ylim: float | None = None,
                                    min_xlim_hours: float | None = None,
                                    smoothing: float | None = None) -> bytes:
    """Generate elevation profile image for a trip with grade-based coloring.

    Args:
        url: RideWithGPS trip URL
        title_time_hours: Optional time to display in title.
        collapse_stops: If True, use moving time (excludes stops) for x-axis.
        max_xlim_hours: Optional max x-axis limit in hours (for synchronized comparison).
        overlay: Optional overlay type (e.g. "speed").
        imperial: If True, use imperial units for overlay axis.
        max_ylim: Optional max y-axis limit in meters (for synchronized comparison).
        max_speed_ylim: Optional max speed y-axis limit in km/h (for synchronized comparison).
        max_grade_ylim: Optional max grade y-axis limit in % (for synchronized comparison).
        min_xlim_hours: Optional min x-axis limit in hours (for zooming).
        smoothing: Smoothing radius in meters. If None, uses config/default.

    Returns PNG image as bytes.
    """
    data = _calculate_trip_elevation_profile_data(url, collapse_stops=collapse_stops, smoothing=smoothing)
    # Use aspect_ratio=3.5 for trips (14/4 = 3.5) to match previous behavior
    return _generate_elevation_profile_chart(
        data,
        overlay=overlay,
        imperial=imperial,
        max_ylim=max_ylim,
        max_speed_ylim=max_speed_ylim,
        max_grade_ylim=max_grade_ylim,
        show_gravel=False,
        aspect_ratio=3.5,
        min_xlim_hours=min_xlim_hours,
        max_xlim_hours=max_xlim_hours,
    )


@app.route("/elevation-profile")
def elevation_profile():
    """Serve elevation profile image for a route or trip."""
    defaults = get_defaults()
    url = request.args.get("url", "")
    climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
    flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
    descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
    mass = float(request.args.get("mass", defaults["mass"]))
    headwind = float(request.args.get("headwind", defaults["headwind"]))
    descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
    gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    collapse_stops = request.args.get("collapse_stops", "false").lower() == "true"
    max_xlim_str = request.args.get("max_xlim_hours", "")
    max_xlim_hours = float(max_xlim_str) if max_xlim_str else None
    min_xlim_str = request.args.get("min_xlim_hours", "")
    min_xlim_hours = float(min_xlim_str) if min_xlim_str else None
    overlay = request.args.get("overlay", "") or None
    imperial = request.args.get("imperial", "false").lower() == "true"
    max_ylim_str = request.args.get("max_ylim", "")
    max_ylim = float(max_ylim_str) if max_ylim_str else None
    max_speed_ylim_str = request.args.get("max_speed_ylim", "")
    max_speed_ylim = float(max_speed_ylim_str) if max_speed_ylim_str else None
    max_grade_ylim_str = request.args.get("max_grade_ylim", "")
    max_grade_ylim = float(max_grade_ylim_str) if max_grade_ylim_str else None
    show_gravel = request.args.get("show_gravel", "false").lower() == "true"
    square = request.args.get("square", "false").lower() == "true"
    # Dynamic aspect ratio support (overrides square if provided)
    aspect_param = request.args.get("aspect", "")
    if aspect_param:
        try:
            aspect_ratio = float(aspect_param)
            aspect_ratio = max(0.5, min(4.0, aspect_ratio))
        except ValueError:
            aspect_ratio = 1.0 if square else 3.5
    else:
        aspect_ratio = 1.0 if square else 3.5

    if not url or not (is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)):
        # Return a placeholder image
        fig_height = 4
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.text(0.5, 0.5, 'No route or trip selected', ha='center', va='center', fontsize=14, color='#999')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Skip caching for zoomed views (min_xlim_hours indicates user zoomed in)
    # These are transient interactions with low cache hit rate
    is_zoomed = min_xlim_hours is not None

    # Check disk cache first for unzoomed views (include aspect in cache key)
    if is_zoomed:
        _profile_image_cache.record_zoomed_skip()
    else:
        cache_key = _make_profile_cache_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, collapse_stops, max_xlim_hours, descending_power, overlay=(overlay or "") + f"|aspect{aspect_ratio:.1f}", imperial=imperial, show_gravel=show_gravel, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, max_grade_ylim=max_grade_ylim, gravel_grade=gravel_grade, smoothing=smoothing, min_xlim_hours=min_xlim_hours)
        cached_bytes = _get_cached_profile(cache_key)
        if cached_bytes:
            return send_file(io.BytesIO(cached_bytes), mimetype='image/png')

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps, no physics params needed
            trip_result = analyze_trip(url, smoothing=smoothing)
            title_time_hours = trip_result["time_seconds"] / 3600
            img_bytes = generate_trip_elevation_profile(url, title_time_hours, collapse_stops=collapse_stops, max_xlim_hours=max_xlim_hours, overlay=overlay, imperial=imperial, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, max_grade_ylim=max_grade_ylim, min_xlim_hours=min_xlim_hours, smoothing=smoothing)
        else:
            # Route: use physics estimation
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
            analysis = analyze_single_route(url, params, smoothing)
            title_time_hours = analysis["time_seconds"] / 3600
            img_bytes = generate_elevation_profile(url, params, title_time_hours, max_xlim_hours=max_xlim_hours, overlay=overlay, imperial=imperial, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, max_grade_ylim=max_grade_ylim, show_gravel=show_gravel, smoothing=smoothing, aspect_ratio=aspect_ratio, min_xlim_hours=min_xlim_hours)
        # Only save to disk cache for unzoomed views
        if not is_zoomed:
            _save_profile_to_cache(cache_key, img_bytes)
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        # Return error image
        fig_height = 4
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', fontsize=12, color='#cc0000')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')


@app.route("/elevation-profile-data")
def elevation_profile_data():
    """Return elevation profile data as JSON for interactive tooltip."""
    defaults = get_defaults()
    url = request.args.get("url", "")
    climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
    flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
    descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
    mass = float(request.args.get("mass", defaults["mass"]))
    headwind = float(request.args.get("headwind", defaults["headwind"]))
    descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
    gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    collapse_stops = request.args.get("collapse_stops", "false").lower() == "true"

    if not url or not (is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps (or moving time if collapse_stops)
            data = _calculate_trip_elevation_profile_data(url, collapse_stops=collapse_stops, smoothing=smoothing)
        else:
            # Route: use physics estimation
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
            data = _calculate_elevation_profile_data(url, params, smoothing)

        # Downsample data if too many points (for performance)
        max_points = 500
        times = data["times_hours"]
        elevations = data["elevations"]
        grades = data["grades"]
        speeds = data.get("speeds_kmh", [])
        distances = data.get("distances", [])
        powers = data.get("powers", [])
        works = data.get("works", [])  # Work in joules for accurate totals
        elev_gains = data.get("elev_gains", [])
        elev_losses = data.get("elev_losses", [])

        if len(times) > max_points:
            step = len(times) // max_points
            times = times[::step]
            elevations = elevations[::step]
            # Use MAX of grades in each chunk to preserve peak grades
            # This ensures hover tooltip shows the same max grade that climbs report
            if grades:
                new_grades = []
                for i in range(0, len(grades), step):
                    chunk_grades = grades[i:i+step]
                    # Filter out None grades (stopped segments in trips)
                    valid_grades = [g for g in chunk_grades if g is not None]
                    if valid_grades:
                        # Use max to preserve peak grades (consistent with climb detection)
                        new_grades.append(max(valid_grades))
                    else:
                        new_grades.append(None)
                grades = new_grades
            else:
                grades = grades[::step]
            if speeds:
                speeds = speeds[::step]
            if distances:
                distances = [sum(distances[i:i+step]) for i in range(0, len(distances), step)]
            if powers:
                def _avg_power(chunk):
                    valid = [p for p in chunk if p is not None]
                    return sum(valid) / len(valid) if valid else None
                powers = [_avg_power(powers[i:i+step]) for i in range(0, len(powers), step)]
            if works:
                # Sum work (energy is additive, preserves total work accurately)
                works = [sum(works[i:i+step]) for i in range(0, len(works), step)]
            if elev_gains:
                elev_gains = [sum(elev_gains[i:i+step]) for i in range(0, len(elev_gains), step)]
            if elev_losses:
                elev_losses = [sum(elev_losses[i:i+step]) for i in range(0, len(elev_losses), step)]

        result = {
            "times": times,
            "elevations": elevations,
            "grades": grades,
            "total_time": data["times_hours"][-1],
        }
        if speeds:
            result["speeds"] = speeds
        if distances:
            result["distances"] = distances
        if powers and any(p is not None for p in powers):
            result["powers"] = powers
        if works:
            result["works"] = works  # Work in joules for accurate selection totals
        if elev_gains:
            result["elev_gains"] = elev_gains
        if elev_losses:
            result["elev_losses"] = elev_losses
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_route_id(url: str) -> str | None:
    """Extract route ID from a RideWithGPS route URL."""
    if not url:
        return None
    import re
    match = re.search(r'/routes/(\d+)', url)
    return match.group(1) if match else None


def extract_trip_id(url: str) -> str | None:
    """Extract trip ID from a RideWithGPS trip URL."""
    if not url:
        return None
    import re
    match = re.search(r'/trips/(\d+)', url)
    return match.group(1) if match else None


def _analyze_url(url: str, params: RiderParams, smoothing: float | None = None, smoothing_override: bool = False) -> tuple[dict, bool]:
    """Analyze a URL (route or trip) and return result and is_trip flag.

    Args:
        url: RideWithGPS route or trip URL
        params: Rider parameters (only used for routes)
        smoothing: Optional override for elevation smoothing radius
        smoothing_override: If True, skip auto-adjustment for high-noise data

    Returns:
        Tuple of (analysis result dict, is_trip flag)
    """
    if is_ridewithgps_trip_url(url):
        result = analyze_trip(url, smoothing=smoothing)
        return result, True
    else:
        result = analyze_single_route(url, params, smoothing, smoothing_override)
        result["is_trip"] = False
        return result, False


def _is_valid_rwgps_url(url: str) -> bool:
    """Check if URL is a valid RideWithGPS route or trip URL."""
    return is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)


def _extract_id_from_url(url: str) -> str | None:
    """Extract route or trip ID from a RideWithGPS URL."""
    route_id = extract_route_id(url)
    if route_id:
        return route_id
    return extract_trip_id(url)


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = get_defaults()
    error = None
    result = None
    result2 = None  # Second route/trip for comparison
    url = None
    url2 = None  # Second route/trip URL for comparison
    mode = "route"
    imperial = False
    route_id = None
    route_id2 = None  # Second route/trip ID for comparison
    privacy_code = None
    privacy_code2 = None
    share_url = None
    compare_mode = False  # Flag for comparison mode
    compare_ylim = None  # Synchronized elevation Y-axis limit for comparison
    compare_speed_ylim = None  # Synchronized speed Y-axis limit for comparison
    compare_grade_ylim = None  # Synchronized grade Y-axis limit for comparison
    is_trip = False  # Flag for trip vs route
    is_trip2 = False  # Flag for second URL trip vs route

    climbing_power = defaults["climbing_power"]
    flat_power = defaults["flat_power"]
    descending_power = defaults["descending_power"]
    mass = defaults["mass"]
    headwind = defaults["headwind"]
    descent_braking_factor = defaults["descent_braking_factor"]
    gravel_grade = defaults["gravel_grade"]
    smoothing = defaults["smoothing"]

    # Check for GET parameters (shared link)
    if request.method == "GET" and request.args.get("url"):
        url = request.args.get("url", "").strip()
        url2 = request.args.get("url2", "").strip()  # Second route/trip for comparison
        imperial = request.args.get("imperial") == "1"

        try:
            climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
            flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
            descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
            mass = float(request.args.get("mass", defaults["mass"]))
            headwind = float(request.args.get("headwind", defaults["headwind"]))
            descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
            gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
            smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
        except ValueError:
            error = "Invalid number in parameters"

        # Check if user explicitly overrides smoothing auto-adjustment
        smoothing_override = request.args.get("smoothing_override") == "1"

        if not error:
            if is_ridewithgps_collection_url(url):
                # Collection - set mode and let JavaScript handle it
                mode = "collection"
            elif _is_valid_rwgps_url(url):
                # Single route or trip - analyze server-side
                try:
                    params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
                    result, is_trip = _analyze_url(url, params, smoothing, smoothing_override)
                    route_id = _extract_id_from_url(url)
                    privacy_code = extract_privacy_code(url)
                except Exception as e:
                    error = f"Error analyzing {'trip' if is_ridewithgps_trip_url(url) else 'route'}: {e}"

                # Handle second route/trip for comparison
                if url2 and not error:
                    if _is_valid_rwgps_url(url2):
                        compare_mode = True
                        try:
                            result2, is_trip2 = _analyze_url(url2, params, smoothing, smoothing_override)
                            route_id2 = _extract_id_from_url(url2)
                            privacy_code2 = extract_privacy_code(url2)
                        except Exception as e:
                            error = f"Error analyzing second {'trip' if is_ridewithgps_trip_url(url2) else 'route'}: {e}"
                    else:
                        error = "Invalid second RideWithGPS URL"

    elif request.method == "POST":
        url = request.form.get("url", "").strip()
        url2 = request.form.get("url2", "").strip()  # Second route/trip for comparison
        compare_enabled = request.form.get("compare") == "on"  # Check if compare checkbox is on
        mode = request.form.get("mode", "route")
        imperial = request.form.get("imperial") == "on"

        try:
            climbing_power = float(request.form.get("climbing_power", defaults["climbing_power"]))
            flat_power = float(request.form.get("flat_power", defaults["flat_power"]))
            descending_power = float(request.form.get("descending_power", defaults["descending_power"]))
            mass = float(request.form.get("mass", defaults["mass"]))
            headwind = float(request.form.get("headwind", defaults["headwind"]))
            descent_braking_factor = float(request.form.get("descent_braking_factor", defaults["descent_braking_factor"]))
            gravel_grade = int(request.form.get("gravel_grade", defaults["gravel_grade"]))
            smoothing = float(request.form.get("smoothing", defaults["smoothing"]))
        except ValueError:
            error = "Invalid number in parameters"

        # Check if user explicitly overrides smoothing auto-adjustment
        smoothing_override = request.form.get("smoothing_override") == "1"

        if not error:
            if not url:
                error = "Please enter a RideWithGPS URL"
            elif mode == "route":
                if not _is_valid_rwgps_url(url):
                    error = "Invalid RideWithGPS URL. Expected format: https://ridewithgps.com/routes/XXXXX or /trips/XXXXX"
                else:
                    try:
                        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
                        result, is_trip = _analyze_url(url, params, smoothing, smoothing_override)
                        route_id = _extract_id_from_url(url)
                        privacy_code = extract_privacy_code(url)
                    except Exception as e:
                        error = f"Error analyzing {'trip' if is_ridewithgps_trip_url(url) else 'route'}: {e}"

                    # Handle second route/trip for comparison (only if checkbox is checked)
                    if url2 and compare_enabled and not error:
                        if _is_valid_rwgps_url(url2):
                            compare_mode = True
                            try:
                                result2, is_trip2 = _analyze_url(url2, params, smoothing, smoothing_override)
                                route_id2 = _extract_id_from_url(url2)
                                privacy_code2 = extract_privacy_code(url2)
                            except Exception as e:
                                error = f"Error analyzing second {'trip' if is_ridewithgps_trip_url(url2) else 'route'}: {e}"
                        else:
                            error = "Invalid second RideWithGPS URL"
            # Collection mode is handled by JavaScript + SSE

    # Compute synchronized Y-axis limits for comparison mode
    if compare_mode and result and result2 and not error:
        try:
            if is_trip:
                data1 = _calculate_trip_elevation_profile_data(url, smoothing=smoothing)
            else:
                params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
                data1 = _calculate_elevation_profile_data(url, params, smoothing, smoothing_override)
            if is_trip2:
                data2 = _calculate_trip_elevation_profile_data(url2, smoothing=smoothing)
            else:
                if not is_trip:
                    # params already built above
                    pass
                else:
                    params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
                data2 = _calculate_elevation_profile_data(url2, params, smoothing, smoothing_override)

            max_elev1 = max(data1["elevations"]) * 1.1
            max_elev2 = max(data2["elevations"]) * 1.1
            compare_ylim = max(max_elev1, max_elev2)

            speeds1 = data1.get("speeds_kmh", [])
            speeds2 = data2.get("speeds_kmh", [])
            if speeds1 or speeds2:
                max_speed1 = max(speeds1) if speeds1 else 0
                max_speed2 = max(speeds2) if speeds2 else 0
                compare_speed_ylim = max(max_speed1, max_speed2)

            # Calculate synchronized grade Y-axis limit
            grades1 = [abs(g) for g in (data1.get("grades") or []) if g is not None]
            grades2 = [abs(g) for g in (data2.get("grades") or []) if g is not None]
            if grades1 or grades2:
                max_grade1 = max(grades1) if grades1 else 0
                max_grade2 = max(grades2) if grades2 else 0
                compare_grade_ylim = max(max_grade1, max_grade2)
        except Exception:
            pass  # Fall back to independent Y-axes if computation fails

    # Build share URL if we have results or a collection
    if url and (result or mode == "collection"):
        from urllib.parse import urlencode
        share_params = {
            "url": url,
            "climbing_power": climbing_power,
            "flat_power": flat_power,
            "descending_power": descending_power,
            "mass": mass,
            "headwind": headwind,
            "descent_braking_factor": descent_braking_factor,
            "gravel_grade": gravel_grade,
            "smoothing": smoothing,
        }
        if url2 and compare_mode:
            share_params["url2"] = url2
        if imperial:
            share_params["imperial"] = "1"
        base_url = request.url_root.replace('http://', 'https://')
        share_url = f"{base_url}?{urlencode(share_params)}"

    return render_template(
        'index.html',
        url=url,
        url2=url2,
        compare_mode=compare_mode,
        is_trip=is_trip,
        is_trip2=is_trip2,
        mode=mode,
        climbing_power=climbing_power,
        flat_power=flat_power,
        descending_power=descending_power,
        mass=mass,
        headwind=headwind,
        descent_braking_factor=descent_braking_factor,
        gravel_grade=gravel_grade,
        smoothing=smoothing,
        defaults=defaults,
        imperial=imperial,
        error=error,
        result=result,
        result2=result2,
        route_id=route_id,
        route_id2=route_id2,
        privacy_code=privacy_code,
        privacy_code2=privacy_code2,
        share_url=share_url,
        compare_ylim=compare_ylim,
        compare_speed_ylim=compare_speed_ylim,
        compare_grade_ylim=compare_grade_ylim,
        version_date=__version_date__,
        git_hash=get_git_hash(),
        # Helper functions for comparison formatting
        format_time_diff=format_time_diff,
        format_diff=format_diff,
        format_pct_diff=format_pct_diff,
        # Analytics
        **_get_analytics_config(),
    )




@app.route("/api/detect-climbs")
def api_detect_climbs():
    """Detect climbs for a route with configurable sensitivity.

    Returns JSON with climb data for dynamic updates. Results are cached in memory.
    """
    defaults = get_defaults()
    url = request.args.get("url", "")
    sensitivity_slider = int(request.args.get("sensitivity", 50))
    climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
    flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
    descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
    mass = float(request.args.get("mass", defaults["mass"]))
    headwind = float(request.args.get("headwind", defaults["headwind"]))
    descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
    gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))

    if not url or not is_ridewithgps_url(url):
        return jsonify({"error": "Invalid route URL"}), 400

    try:
        # Create params hash for cache key
        params_hash = hashlib.md5(f"{climbing_power}|{flat_power}|{mass}|{headwind}|{smoothing}".encode()).hexdigest()[:8]
        cache_key = _make_climb_cache_key(url, sensitivity_slider, params_hash)

        # Check cache first
        cached = _get_cached_climbs(cache_key)
        if cached:
            climbs_json, sensitivity_m = cached
            return jsonify({
                "climbs": climbs_json,
                "sensitivity_m": sensitivity_m,
            })

        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
        config = _load_config() or {}

        # Get profile data (uses cached route data)
        # This includes scaled_points for consistent climb detection
        profile_data = _calculate_elevation_profile_data(url, params, smoothing)
        times_hours = profile_data["times_hours"]
        powers = profile_data.get("powers", [])
        works = profile_data.get("works", [])
        rolling_grades = profile_data.get("grades", [])
        scaled_points = profile_data.get("scaled_points", [])

        # Convert slider to sensitivity
        sensitivity_m = slider_to_sensitivity(sensitivity_slider)

        # Get climb thresholds from config
        min_gain = config.get("climb_min_gain", DEFAULTS.get("climb_min_gain", 50.0))
        min_distance = config.get("climb_min_distance", DEFAULTS.get("climb_min_distance", 500.0))
        grade_threshold = config.get("climb_grade_threshold", DEFAULTS.get("climb_grade_threshold", 2.0))

        # Detect climbs
        result = detect_climbs(
            scaled_points,
            times_hours=times_hours,
            sensitivity_m=sensitivity_m,
            min_climb_gain=min_gain,
            min_climb_distance=min_distance,
            grade_threshold=grade_threshold,
            params=params,
            segment_powers=powers,
            segment_works=works,
            rolling_grades=rolling_grades,
        )

        # Convert to JSON-serializable format
        climbs_json = [
            {
                "climb_id": c.climb_id,
                "label": c.label,
                "start_km": c.start_km,
                "end_km": c.end_km,
                "start_time_hours": c.start_time_hours,
                "end_time_hours": c.end_time_hours,
                "distance_m": c.distance_m,
                "elevation_gain": c.elevation_gain,
                "elevation_loss": c.elevation_loss,
                "avg_grade": c.avg_grade,
                "max_grade": c.max_grade,
                "start_elevation": c.start_elevation,
                "peak_elevation": c.peak_elevation,
                "duration_seconds": c.duration_seconds,
                "work_kj": c.work_kj,
                "avg_power": c.avg_power,
                "avg_speed_kmh": c.avg_speed_kmh,
            }
            for c in result.climbs
        ]

        # Save to cache
        _save_climbs_to_cache(cache_key, climbs_json, result.sensitivity_m)

        return jsonify({
            "climbs": climbs_json,
            "sensitivity_m": result.sensitivity_m,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/elevation-profile-ride")
def elevation_profile_ride():
    """Generate elevation profile with climb highlighting for ride page.

    Supports dynamic aspect ratios via the 'aspect' parameter:
    - 'square' or '1': 1:1 aspect ratio (default, mobile portrait)
    - 'wide' or '2': 2:1 aspect ratio (tablet/desktop)
    - 'ultrawide' or '3': 3:1 aspect ratio (large desktop)
    - Custom ratio like '2.5': any numeric aspect ratio
    """
    defaults = get_defaults()
    url = request.args.get("url", "")
    aspect_param = request.args.get("aspect", "1")
    # Parse aspect ratio
    try:
        if aspect_param == "square":
            aspect_ratio = 1.0
        elif aspect_param == "wide":
            aspect_ratio = 2.0
        elif aspect_param == "ultrawide":
            aspect_ratio = 3.0
        else:
            aspect_ratio = float(aspect_param)
            aspect_ratio = max(0.5, min(4.0, aspect_ratio))  # Clamp to reasonable range
    except ValueError:
        aspect_ratio = 1.0
    sensitivity_slider = int(request.args.get("sensitivity", 50))
    climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
    flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
    descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
    mass = float(request.args.get("mass", defaults["mass"]))
    headwind = float(request.args.get("headwind", defaults["headwind"]))
    descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
    gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))

    if not url or not is_ridewithgps_url(url):
        # Return placeholder image with appropriate aspect ratio
        fig_height = 4
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.text(0.5, 0.5, 'No route selected', ha='center', va='center', fontsize=14, color='#999')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Create cache key and check disk cache
    params_hash = hashlib.md5(f"{climbing_power}|{flat_power}|{mass}|{headwind}|{smoothing}".encode()).hexdigest()[:8]
    cache_key = _make_ride_profile_cache_key(url, sensitivity_slider, aspect_ratio, params_hash)
    cached_bytes = _get_cached_profile(cache_key)
    if cached_bytes:
        return send_file(io.BytesIO(cached_bytes), mimetype='image/png')

    try:
        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
        config = _load_config() or {}

        # Get profile data (includes scaled_points for consistent climb detection)
        data = _calculate_elevation_profile_data(url, params, smoothing)
        times_hours = data["times_hours"]
        elevations = data["elevations"]
        grades = data["grades"]
        powers = data.get("powers", [])
        works = data.get("works", [])
        scaled_points = data.get("scaled_points", [])

        sensitivity_m = slider_to_sensitivity(sensitivity_slider)
        min_gain = config.get("climb_min_gain", DEFAULTS.get("climb_min_gain", 50.0))
        min_distance = config.get("climb_min_distance", DEFAULTS.get("climb_min_distance", 500.0))
        grade_threshold = config.get("climb_grade_threshold", DEFAULTS.get("climb_grade_threshold", 2.0))

        climb_result = detect_climbs(
            scaled_points,
            times_hours=times_hours,
            sensitivity_m=sensitivity_m,
            min_climb_gain=min_gain,
            min_climb_distance=min_distance,
            grade_threshold=grade_threshold,
            params=params,
            segment_powers=powers,
            segment_works=works,
            rolling_grades=grades,
        )

        # Generate profile with requested aspect ratio
        img_bytes = _generate_ride_profile(times_hours, elevations, grades, climb_result.climbs, aspect_ratio)
        # Save to disk cache
        _save_profile_to_cache(cache_key, img_bytes)
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')

    except Exception as e:
        # Return error image with appropriate aspect ratio
        fig_height = 4
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.text(0.5, 0.5, f'Error: {str(e)[:40]}', ha='center', va='center', fontsize=12, color='#cc0000')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')


def _generate_ride_profile(times_hours: list, elevations: list, grades: list, climbs: list[ClimbInfo], aspect_ratio: float = 1.0) -> bytes:
    """Generate elevation profile with climb highlighting.

    Args:
        times_hours: Cumulative time in hours for each point
        elevations: Elevation in meters for each point
        grades: Grade percentage for each segment
        climbs: List of detected climbs to highlight
        aspect_ratio: Width/height ratio (1.0 = square, 2.0 = wide, etc.)

    Returns:
        PNG image bytes
    """
    return _generate_ride_profile_chart(times_hours, elevations, grades, climbs, aspect_ratio)


@app.route("/ride")
def ride_page():
    """Mobile-focused ride details page with climb detection."""
    defaults = get_defaults()
    url = request.args.get("url", "")
    sensitivity = int(request.args.get("sensitivity", 50))
    climbing_power = float(request.args.get("climbing_power", defaults["climbing_power"]))
    flat_power = float(request.args.get("flat_power", defaults["flat_power"]))
    descending_power = float(request.args.get("descending_power", defaults["descending_power"]))
    mass = float(request.args.get("mass", defaults["mass"]))
    headwind = float(request.args.get("headwind", defaults["headwind"]))
    descent_braking_factor = float(request.args.get("descent_braking_factor", defaults["descent_braking_factor"]))
    gravel_grade = int(request.args.get("gravel_grade", defaults["gravel_grade"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    imperial = request.args.get("imperial", "").lower() in ("true", "1", "yes")

    error = None
    route_name = None
    time_str = "0h 00m"
    work_kj = 0
    distance_km = 0
    elevation_m = 0
    climbs = []

    if url and is_ridewithgps_url(url):
        try:
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_grade)
            config = _load_config() or {}

            # Get analysis data
            result = analyze_single_route(url, params, smoothing)
            route_name = result.get("name")
            time_str = result.get("time_str", "0h 00m")
            work_kj = result.get("work_kj", 0)
            distance_km = result.get("distance_km", 0)
            elevation_m = result.get("elevation_m", 0)

            # Get profile data for climb detection (includes scaled_points for consistency)
            profile_data = _calculate_elevation_profile_data(url, params, smoothing)
            times_hours = profile_data["times_hours"]
            powers = profile_data.get("powers", [])
            works = profile_data.get("works", [])
            rolling_grades = profile_data.get("grades", [])
            scaled_points = profile_data.get("scaled_points", [])

            sensitivity_m = slider_to_sensitivity(sensitivity)
            min_gain = config.get("climb_min_gain", DEFAULTS.get("climb_min_gain", 50.0))
            min_distance = config.get("climb_min_distance", DEFAULTS.get("climb_min_distance", 500.0))
            grade_threshold = config.get("climb_grade_threshold", DEFAULTS.get("climb_grade_threshold", 2.0))

            climb_result = detect_climbs(
                scaled_points,
                times_hours=times_hours,
                sensitivity_m=sensitivity_m,
                min_climb_gain=min_gain,
                min_climb_distance=min_distance,
                grade_threshold=grade_threshold,
                params=params,
                segment_powers=powers,
                segment_works=works,
                rolling_grades=rolling_grades,
            )
            climbs = climb_result.climbs

        except Exception as e:
            error = str(e)
    elif url:
        error = "Invalid RideWithGPS route URL"

    return render_template(
        'ride.html',
        url=url,
        sensitivity=sensitivity,
        climbing_power=climbing_power,
        flat_power=flat_power,
        mass=mass,
        headwind=headwind,
        smoothing=smoothing,
        route_name=route_name,
        time_str=time_str,
        work_kj=work_kj,
        distance_km=distance_km,
        elevation_m=elevation_m,
        climbs=climbs,
        error=error,
        imperial=imperial,
        version_date=__version_date__,
        git_hash=get_git_hash(),
        # Analytics
        **_get_analytics_config(),
    )


def main():
    """Run the web server."""
    import os
    port = int(os.environ.get("PORT", 5050))
    print("Starting GPX Analyzer web server...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
