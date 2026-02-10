"""Simple web interface for GPX analyzer."""

import hashlib
import io
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from flask import Flask, render_template_string, request, Response, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from gpx_analyzer import __version_date__, get_git_hash
from gpx_analyzer.analyzer import analyze, calculate_hilliness, DEFAULT_MAX_GRADE_WINDOW, DEFAULT_MAX_GRADE_SMOOTHING, GRADE_BINS, GRADE_LABELS, STEEP_GRADE_BINS, STEEP_GRADE_LABELS, _calculate_rolling_grades
from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.physics import calculate_segment_work
from gpx_analyzer.cli import calculate_elevation_gain, calculate_surface_breakdown, DEFAULTS
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
from gpx_analyzer.tunnel import detect_and_correct_tunnels
from gpx_analyzer.climb import detect_climbs, slider_to_sensitivity, ClimbInfo


def _get_config_hash() -> str:
    """Get a hash of config parameters that affect analysis but aren't in the UI."""
    config = _load_config() or {}
    # Include all config params that affect physics but aren't UI-adjustable
    # Note: smoothing is now UI-adjustable, so it's not in this list
    relevant_keys = [
        "crr", "cda", "coasting_grade", "max_coast_speed", "max_coast_speed_unpaved",
        "climb_threshold_grade", "steep_descent_speed", "steep_descent_grade",
        "straight_descent_speed", "hairpin_speed", "straight_curvature", "hairpin_curvature",
        "drivetrain_efficiency", "unpaved_power_factor", "elevation_scale",
    ]
    config_str = "|".join(f"{k}={config.get(k, '')}" for k in relevant_keys)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


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


# Simple LRU cache for route analysis results
class AnalysisCache:
    """Thread-safe LRU cache for route analysis results."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
                  descent_braking_factor: float = 1.0, descending_power: float = 20.0,
                  unpaved_power_factor: float = 0.90, smoothing: float = 50.0,
                  smoothing_override: bool = False) -> str:
        """Create a cache key from analysis parameters."""
        config_hash = _get_config_hash()
        key_str = f"{url}|{climbing_power}|{flat_power}|{descending_power}|{mass}|{headwind}|{descent_braking_factor}|{unpaved_power_factor}|{smoothing}|{smoothing_override}|{config_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
            descent_braking_factor: float = 1.0, descending_power: float = 20.0,
            unpaved_power_factor: float = 0.90, smoothing: float = 50.0,
            smoothing_override: bool = False) -> dict | None:
        """Get cached result, returns None if not found."""
        key = self._make_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor, smoothing, smoothing_override)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key][0]
            self.misses += 1
            return None

    def set(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
            descent_braking_factor: float, descending_power: float, unpaved_power_factor: float, smoothing: float,
            smoothing_override: bool, result: dict) -> None:
        """Store result in cache."""
        key = self._make_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor, smoothing, smoothing_override)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (result, time.time())
            # Evict oldest if over limit
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            # Estimate ~1.5 KB per entry based on typical result dict size
            memory_kb = round(len(self.cache) * 1.5, 1)
            return {
                "hit_rate": f"{hit_rate:.1f}%",
                "hits": self.hits,
                "max_size": self.max_size,
                "misses": self.misses,
                "size": len(self.cache),
                "memory_kb": memory_kb,
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


# Global cache instance (~0.75 MB at full capacity)
_analysis_cache = AnalysisCache(max_size=175)


# Disk cache for elevation profile images
PROFILE_CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer" / "profiles"
PROFILE_CACHE_INDEX_PATH = PROFILE_CACHE_DIR / "cache_index.json"
MAX_CACHED_PROFILES = 525  # ~525 images, each ~37KB = ~19MB max
_elevation_profile_cache_stats = {"hits": 0, "misses": 0, "zoomed_skipped": 0}

# In-memory cache for climb detection results (avoids re-processing on sensitivity changes)
# Key: (url, sensitivity, params_hash) -> Value: (climbs_json, sensitivity_m, timestamp)
_climb_cache: dict[str, tuple[list, float, float]] = {}
MAX_CLIMB_CACHE_ENTRIES = 175
CLIMB_CACHE_TTL = 3600  # 1 hour
_climb_cache_stats = {"hits": 0, "misses": 0}


def _make_climb_cache_key(url: str, sensitivity: int, params_hash: str) -> str:
    """Create cache key for climb detection results."""
    return f"{url}|{sensitivity}|{params_hash}"


def _get_cached_climbs(cache_key: str) -> tuple[list, float] | None:
    """Get cached climb detection results if available and not expired."""
    if cache_key in _climb_cache:
        climbs_json, sensitivity_m, timestamp = _climb_cache[cache_key]
        if time.time() - timestamp < CLIMB_CACHE_TTL:
            _climb_cache_stats["hits"] += 1
            return climbs_json, sensitivity_m
        else:
            del _climb_cache[cache_key]
    _climb_cache_stats["misses"] += 1
    return None


def _save_climbs_to_cache(cache_key: str, climbs_json: list, sensitivity_m: float) -> None:
    """Save climb detection results to cache with LRU eviction."""
    # Evict oldest entries if cache is full
    if len(_climb_cache) >= MAX_CLIMB_CACHE_ENTRIES:
        oldest_key = min(_climb_cache.keys(), key=lambda k: _climb_cache[k][2])
        del _climb_cache[oldest_key]
    _climb_cache[cache_key] = (climbs_json, sensitivity_m, time.time())


def _get_climb_cache_stats() -> dict:
    """Return climb cache statistics."""
    total = _climb_cache_stats["hits"] + _climb_cache_stats["misses"]
    hit_rate = (_climb_cache_stats["hits"] / total * 100) if total > 0 else 0
    # Estimate ~2 KB per entry (climbs list + metadata)
    memory_kb = round(len(_climb_cache) * 2, 1)
    return {
        "hit_rate": f"{hit_rate:.1f}%",
        "hits": _climb_cache_stats["hits"],
        "max_size": MAX_CLIMB_CACHE_ENTRIES,
        "misses": _climb_cache_stats["misses"],
        "size": len(_climb_cache),
        "memory_kb": memory_kb,
    }


def _clear_climb_cache() -> int:
    """Clear the climb detection cache. Returns number of entries removed."""
    global _climb_cache_stats
    count = len(_climb_cache)
    _climb_cache.clear()
    _climb_cache_stats = {"hits": 0, "misses": 0}
    return count


def _make_ride_profile_cache_key(url: str, sensitivity: int, aspect: float, params_hash: str) -> str:
    """Create cache key for ride profile images."""
    config_hash = _get_config_hash()
    key_str = f"ride|{url}|{sensitivity}|{aspect:.1f}|{params_hash}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _make_profile_cache_key(url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
                            descent_braking_factor: float = 1.0, collapse_stops: bool = False,
                            max_xlim_hours: float | None = None, descending_power: float = 20.0,
                            overlay: str = "", imperial: bool = False,
                            show_gravel: bool = False,
                            max_ylim: float | None = None,
                            max_speed_ylim: float | None = None,
                            unpaved_power_factor: float = 0.0,
                            smoothing: float = 50.0,
                            min_xlim_hours: float | None = None) -> str:
    """Create a unique cache key for elevation profile parameters."""
    config_hash = _get_config_hash()
    key_str = f"{url}|{climbing_power}|{flat_power}|{descending_power}|{mass}|{headwind}|{descent_braking_factor}|{collapse_stops}|{min_xlim_hours}|{max_xlim_hours}|{overlay}|{imperial}|{show_gravel}|{max_ylim}|{max_speed_ylim}|{unpaved_power_factor}|{smoothing}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_profile_cache_index() -> dict:
    """Load the profile cache index (maps cache_key -> timestamp)."""
    if PROFILE_CACHE_INDEX_PATH.exists():
        try:
            with PROFILE_CACHE_INDEX_PATH.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_profile_cache_index(index: dict) -> None:
    """Save the profile cache index."""
    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with PROFILE_CACHE_INDEX_PATH.open("w") as f:
        json.dump(index, f)


def _get_cached_profile(cache_key: str) -> bytes | None:
    """Load cached profile image if available."""
    path = PROFILE_CACHE_DIR / f"{cache_key}.png"
    if path.exists():
        # Update access time in index
        index = _load_profile_cache_index()
        index[cache_key] = time.time()
        _save_profile_cache_index(index)
        _elevation_profile_cache_stats["hits"] += 1
        return path.read_bytes()
    _elevation_profile_cache_stats["misses"] += 1
    return None


def _save_profile_to_cache(cache_key: str, img_bytes: bytes) -> None:
    """Save profile image to cache and enforce LRU limit."""
    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = PROFILE_CACHE_DIR / f"{cache_key}.png"
    path.write_bytes(img_bytes)

    # Update index
    index = _load_profile_cache_index()
    index[cache_key] = time.time()

    # Enforce LRU limit
    if len(index) > MAX_CACHED_PROFILES:
        sorted_entries = sorted(index.items(), key=lambda x: x[1])
        to_remove = sorted_entries[: len(index) - MAX_CACHED_PROFILES]
        for key, _ in to_remove:
            old_path = PROFILE_CACHE_DIR / f"{key}.png"
            if old_path.exists():
                old_path.unlink()
            del index[key]

    _save_profile_cache_index(index)


app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% if result and result.name %}{{ result.name }} | {% endif %}Reality Check my Route</title>

    <!-- Open Graph meta tags for link previews -->
    {% set base_url = request.url_root | replace('http://', 'https://') %}
    <meta property="og:site_name" content="Reality Check my Route">
    {% if result %}
    <meta property="og:title" content="Reality Check: {{ result.name or ('Trip Analysis' if is_trip else 'Route Analysis') }}">
    {% if is_trip %}
    <meta property="og:description" content="{{ result.time_str }}{% if result.work_kj is not none %} • {{ '%.0f'|format(result.work_kj) }} kJ{% endif %} | {{ '%.0f'|format(result.distance_km) }} km • {{ '%.0f'|format(result.elevation_m) }}m{% if result.avg_watts is not none %} @ {{ result.avg_watts|int }}W{% endif %}">
    {% else %}
    <meta property="og:description" content="{{ result.time_str }} • {{ '%.0f'|format(result.work_kj) }} kJ | {{ '%.0f'|format(result.distance_km) }} km • {{ '%.0f'|format(result.elevation_m) }}m @ {{ climbing_power|int }}W">
    {% endif %}
    <meta property="og:image" content="{{ base_url }}og-image?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}">
    <meta property="og:type" content="website">
    {% else %}
    <meta property="og:title" content="Reality Check my Route">
    <meta property="og:description" content="Physics-based cycling time and energy estimates from elevation, surface, and rider parameters">
    <meta property="og:image" content="{{ base_url }}og-image">
    <meta property="og:type" content="website">
    {% endif %}
    <meta property="og:url" content="{{ request.url | replace('http://', 'https://') }}">
    <style>
        :root {
            --primary: #FF6B35;
            --primary-dark: #E55A2B;
            --primary-gradient: linear-gradient(135deg, #FF6B35, #F7931E);
            --accent: #2D3047;
            --text-dark: #333;
            --text-muted: #666;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f7;
        }
        @media (min-width: 1200px) {
            body { max-width: 1100px; }
            body.collection-mode { max-width: 90%; }
        }
        @media (max-width: 480px) {
            body { padding: 12px; }
        }
        h1 { color: var(--accent); font-size: 1.5em; }
        form {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        label:not(.toggle-label) {
            display: block;
            margin-top: 15px;
            font-weight: 600;
            color: #555;
        }
        label:not(.toggle-label):first-child { margin-top: 0; }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 12px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--primary);
        }
        .param-row {
            display: flex;
            gap: 15px;
        }
        .param-row > div { flex: 1; }
        button {
            width: 100%;
            padding: 14px;
            margin-top: 20px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #999; cursor: not-allowed; }
        .results {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;  /* Allow horizontal scroll for wide tables */
        }
        .results.route-results {
            border-left: 4px solid #4CAF50;
        }
        .results.trip-results {
            border-left: 4px solid #2196F3;
        }
        .result-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .route-badge {
            background: #E8F5E9;
            color: #2E7D32;
        }
        .trip-badge {
            background: #E3F2FD;
            color: #1565C0;
        }
        .results h2 { margin-top: 0; font-size: 1.2em; color: #333; }
        .results h2 a { color: inherit; text-decoration: none; }
        .results h2 a:hover { color: var(--primary); }
        .results-header {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .results-header h2 {
            margin: 0;
        }
        .share-btn {
            padding: 4px 8px;
            background: white;
            border: 1.5px solid var(--primary);
            color: var(--primary);
            border-radius: 4px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            font-weight: 500;
            flex-shrink: 0;
            width: auto;
        }
        .share-btn svg {
            width: 14px;
            height: 14px;
            fill: currentColor;
        }
        .share-btn:hover {
            background: var(--primary);
            color: white;
        }
        .share-btn.copied {
            background: #28a745;
            border-color: #28a745;
            color: white;
        }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .result-row:last-child { border-bottom: none; }
        #singleRouteResults .result-row,
        #collectionResults .result-row {
            max-width: 350px;
        }
        .result-label { color: #666; }
        .result-value { font-weight: 600; color: #333; }
        .result-row.primary {
            background: #f0f7ff;
        }
        .result-row.primary .result-value {
            font-weight: 700;
        }
        .unit-select {
            padding: 1px 2px;
            font-size: 0.75em;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            cursor: pointer;
            max-width: 70px;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 2px center;
            background-size: 12px;
            padding-right: 16px;
        }
        .unit-select:hover {
            border-color: #4CAF50;
        }
        .result-row .unit-select {
            margin-left: 4px;
        }
        .result-row .result-label {
            flex-shrink: 0;
        }
        .result-row .result-value {
            text-align: right;
            margin-left: auto;
        }
        .primary-results {
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ccc;
        }
        #singleRouteResults .primary-results {
            max-width: 350px;
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 15px;
            border-radius: 6px;
            margin-top: 20px;
        }
        .note {
            font-size: 0.85em;
            color: #888;
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .anomaly-note {
            background: #FFF3E0;
            border-left: 3px solid #FF9800;
            color: #5D4037;
        }
        .anomaly-note strong {
            color: #E65100;
        }
        .anomaly-item {
            display: inline-block;
            background: #FFE0B2;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 2px 0;
        }
        .noise-note {
            background: #E3F2FD;
            border-left: 3px solid #2196F3;
            color: #1565C0;
            white-space: nowrap;
            padding: 4px 8px;
        }
        .noise-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-weight: 500;
            line-height: 1;
        }
        .noise-badge .info-btn {
            vertical-align: middle;
        }
        .histograms-container {
            display: flex;
            gap: 15px;
            margin-top: 15px;
        }
        @media (max-width: 600px) {
            .histograms-container {
                flex-direction: column;
            }
        }
        .grade-histogram {
            flex: 1;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
        }
        .grade-histogram h4 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            color: #555;
        }
        .steep-section {
            margin-top: 20px;
            padding: 15px;
            background: #fff5f0;
            border-radius: 6px;
            border-left: 3px solid #ff6600;
        }
        .steep-section > h4 {
            margin: 0 0 12px 0;
            font-size: 1em;
            color: #e55a00;
        }
        .steep-stats {
            display: flex;
            gap: 15px 25px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .steep-stat {
            display: flex;
            align-items: baseline;
            gap: 6px;
        }
        .steep-label {
            font-size: 0.85em;
            color: #666;
            white-space: nowrap;
        }
        .steep-value {
            font-weight: 600;
            color: #e55a00;
        }
        .steep-section .histograms-container {
            margin-top: 10px;
        }
        .steep-section .grade-histogram {
            background: white;
        }
        .histogram-bars {
            display: flex;
            align-items: flex-end;
            gap: 2px;
        }
        .histogram-bar {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            min-width: 0;
            min-height: 80px;
        }
        .histogram-bar .bar-container {
            width: 100%;
            height: 60px;
            display: flex;
            align-items: flex-end;
            cursor: pointer;
            /* Extend touch target area */
            padding: 10px 2px 0 2px;
            margin: -10px -2px 0 -2px;
        }
        .histogram-bar .bar {
            width: 100%;
            border-radius: 2px 2px 0 0;
            min-height: 4px;  /* Larger minimum for touch */
        }
        /* Make entire histogram-bar touchable on mobile */
        @media (pointer: coarse) {
            .histogram-bar {
                cursor: pointer;
            }
            .histogram-bar .bar {
                min-height: 8px;  /* Even larger on touch devices */
            }
        }
        .histogram-bar .label {
            font-size: 0.6em;
            color: #888;
            margin-top: 4px;
            white-space: nowrap;
        }
        .histogram-bar .pct {
            font-size: 0.6em;
            color: #666;
        }
        .elevation-profile {
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .elevation-profile h4 {
            margin: 0 0 10px 0;
            font-size: 0.95em;
            color: #333;
        }
        .elevation-profile-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .elevation-profile-header h4 {
            margin: 0;
        }
        .elevation-profile-header h4 a {
            color: #333;
            text-decoration: none;
        }
        .elevation-profile-header h4 a:hover {
            color: #1a73e8;
            text-decoration: underline;
        }
        .ride-link {
            white-space: nowrap;
        }
        @media (max-width: 600px) {
            .elevation-profile-header {
                flex-wrap: wrap;
                gap: 8px;
            }
            .elevation-profile-header h4 {
                width: 100%;
            }
            .ride-link {
                order: 3;
                margin-left: auto;
            }
        }
        .elevation-profile-toggles {
            display: flex;
            gap: 12px;
            align-items: center;
            font-size: 0.85em;
        }
        .collapse-stops-toggle {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            cursor: pointer;
            vertical-align: middle;
        }
        .collapse-stops-toggle input {
            margin: 0;
            cursor: pointer;
            vertical-align: middle;
        }
        .collapse-stops-toggle label {
            cursor: pointer;
            color: #666;
            font-weight: normal;
            line-height: 1;
            vertical-align: middle;
            margin-top: 0;
        }
        .elevation-profile-container {
            position: relative;
            width: 100%;
        }
        .elevation-profile img {
            width: 100%;
            height: auto;
            display: block;
            /* Prevent Safari long-press context menu */
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
        }
        .elevation-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            white-space: nowrap;
            z-index: 100;
            transform: translateX(-50%);
        }
        .elevation-tooltip.visible {
            opacity: 1;
        }
        .elevation-tooltip .grade {
            font-weight: bold;
            font-size: 14px;
        }
        .elevation-tooltip .elev {
            color: #ccc;
            margin-top: 2px;
        }
        .elevation-cursor {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 1px;
            background: rgba(0, 0, 0, 0.5);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
        }
        .elevation-cursor.visible {
            opacity: 1;
        }
        .elevation-selection {
            position: absolute;
            top: 0;
            bottom: 0;
            background: rgba(59, 130, 246, 0.2);
            border-left: 2px solid rgba(59, 130, 246, 0.6);
            border-right: 2px solid rgba(59, 130, 246, 0.6);
            pointer-events: none;
            opacity: 0;
            z-index: 50;
        }
        .elevation-selection.visible {
            opacity: 1;
        }
        .elevation-selection-popup {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 200;
            white-space: nowrap;
            transform: translateX(-50%);
            pointer-events: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .elevation-selection-popup .selection-close {
            position: absolute;
            top: 2px;
            right: 6px;
            cursor: pointer;
            color: #888;
            font-size: 14px;
            line-height: 1;
        }
        .elevation-selection-popup .selection-close:hover {
            color: white;
        }
        .elevation-selection-popup .selection-stat {
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        .elevation-selection-popup .stat-label {
            color: #aaa;
        }
        .elevation-selection-popup .stat-value {
            font-weight: bold;
        }
        .selection-zoom-btn {
            margin-top: 8px;
            padding: 6px 12px;
            background: #3b82f6;
            color: white;
            border-radius: 4px;
            text-align: center;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }
        .selection-zoom-btn:hover {
            background: #2563eb;
        }
        .zoom-out-link {
            position: absolute;
            bottom: 2px;
            right: 10%;
            z-index: 100;
        }
        .zoom-out-link a {
            color: #3b82f6;
            font-size: 12px;
            text-decoration: none;
        }
        .zoom-out-link a:hover {
            text-decoration: underline;
        }

        /* Long-press indicator for touch selection */
        .long-press-indicator {
            position: absolute;
            width: 60px;
            height: 60px;
            margin-left: -30px;
            margin-top: -30px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s ease-out;
            z-index: 100;
        }
        .long-press-indicator.active {
            opacity: 1;
        }
        .long-press-ring {
            width: 100%;
            height: 100%;
            border: 3px solid #2196F3;
            border-radius: 50%;
            animation: long-press-pulse 0.4s ease-out forwards;
            box-sizing: border-box;
        }
        @keyframes long-press-pulse {
            0% {
                transform: scale(0.3);
                opacity: 0.8;
            }
            100% {
                transform: scale(1);
                opacity: 0;
                border-width: 2px;
            }
        }

        .elevation-loading {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 120px;
            background: #f8f8f8;
            border-radius: 4px;
        }
        .elevation-loading.hidden {
            display: none;
        }
        .elevation-spinner {
            width: 32px;
            height: 32px;
            border: 3px solid #e0e0e0;
            border-top-color: #666;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .elevation-profile img.loading {
            display: none;
        }
        .route-map {
            margin-top: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            width: 100%;
            /* Tall ratio on mobile to show both map and elevation */
            padding-bottom: 120%;
        }
        .route-map iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }
        @media (min-width: 500px) {
            .route-map {
                /* Square-ish on medium screens */
                padding-bottom: 90%;
            }
        }
        @media (min-width: 768px) {
            .route-map {
                /* 16:10 aspect ratio on larger screens */
                padding-bottom: 62.5%;
            }
        }
        /* Progress bar styles */
        .progress-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .progress-bar {
            width: 100%;
            height: 24px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-fill {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 12px;
            transition: width 0.3s ease;
            width: 0%;
        }
        .progress-text {
            font-size: 0.9em;
            color: #666;
        }
        .progress-route {
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Collection table styles */
        .collection-table {
            width: 100%;
            min-width: 700px;  /* Prevent excessive squishing on mobile */
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }
        .collection-table th {
            background: #f5f5f5;
            padding: 10px 8px;
            text-align: left;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #ddd;
        }
        .collection-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
        }
        .collection-table tr:last-child td {
            border-bottom: none;
        }
        .collection-table .num {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        .totals-row {
            font-weight: 600;
            background: #f9f9f9;
        }
        .totals-row td {
            border-top: 2px solid #ddd;
        }
        .collection-table .primary {
            background: #f0f7ff;
            font-weight: 600;
        }
        .collection-table th.primary {
            background: #e3effa;
        }
        .collection-table .separator {
            border-right: 2px solid #ccc;
        }
        .route-name {
            max-width: 280px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .route-name a:first-child {
            color: var(--primary);
            text-decoration: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            min-width: 0;
        }
        .route-name a:first-child:hover {
            text-decoration: underline;
        }
        .route-name .rwgps-link {
            flex-shrink: 0;
            display: inline-block;
            background: #FA6400;
            color: white;
            padding: 1px 4px;
            font-size: 0.65em;
            font-weight: 600;
            text-decoration: none;
            border-radius: 3px;
        }
        .route-name .rwgps-link:hover {
            background: #e55a00;
        }
        @media (min-width: 1200px) {
            .route-name { max-width: 400px; }
            .collection-table { font-size: 0.95em; }
        }
        @media (max-width: 768px) {
            .route-name { max-width: 200px; }
        }
        @media (max-width: 600px) {
            .collection-table { font-size: 0.8em; }
            .collection-table th, .collection-table td { padding: 8px 4px; }
            .route-name { max-width: 140px; }
            /* Compact mobile form layout */
            form { padding: 12px; }
            .param-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }
            .param-row > div {
                display: flex;
                flex-direction: row;
                align-items: center;
                gap: 6px;
            }
            .param-row .label-row {
                flex-shrink: 0;
                min-width: 0;
            }
            .param-row .label-row label {
                font-size: 0.75em;
                margin-top: 0;
                white-space: nowrap;
            }
            .param-row .label-row .info-btn {
                display: none;
            }
            .param-row input[type="number"] {
                width: 70px;
                min-width: 60px;
                padding: 6px 4px;
                font-size: 14px;
                margin-top: 0;
                flex-shrink: 0;
            }
            /* Shorter labels on mobile */
            .param-row label[for="climbing_power"] { font-size: 0; }
            .param-row label[for="climbing_power"]::before { content: "Climb (W)"; font-size: 11px; }
            .param-row label[for="flat_power"] { font-size: 0; }
            .param-row label[for="flat_power"]::before { content: "Flat (W)"; font-size: 11px; }
            .param-row label[for="mass"] { font-size: 0; }
            .param-row label[for="mass"]::before { content: "Mass (kg)"; font-size: 11px; }
            .param-row label[for="headwind"] { font-size: 0; }
            .param-row label[for="headwind"]::before { content: "Wind (km/h)"; font-size: 11px; }
            /* Reduce button margin */
            button { margin-top: 12px; padding: 10px; }
            /* Compact results */
            .results { padding: 12px; margin-top: 12px; }
            /* Compact URL input */
            input[type="text"] { padding: 10px; font-size: 14px; }
            /* Compact compare and advanced rows */
            .compare-toggle { margin-top: 6px; margin-bottom: 4px; }
            .advanced-row { margin-top: 8px; }
            .advanced-toggle { font-size: 0.85em; }
            /* Compact labels */
            label:not(.toggle-label) { margin-top: 8px; font-size: 0.85em; }
            .label-row label { font-size: 0.85em; }
            /* Compact units row */
            .units-row { margin-top: 6px; }
            .units-row span { font-size: 0.85em; }
            /* Compact advanced options */
            .advanced-options { padding: 10px; margin-top: 8px; }
            .advanced-options .param-row label[for="descending_power"] { font-size: 0; }
            .advanced-options .param-row label[for="descending_power"]::before { content: "Desc (W)"; font-size: 11px; }
            .advanced-options .param-row label[for="descent_braking_factor"] { font-size: 0; }
            .advanced-options .param-row label[for="descent_braking_factor"]::before { content: "Braking"; font-size: 11px; }
            .advanced-options .param-row label[for="unpaved_power_factor"] { font-size: 0; }
            .advanced-options .param-row label[for="unpaved_power_factor"]::before { content: "Gravel"; font-size: 11px; }
            .advanced-options .param-row label[for="smoothing"] { font-size: 0; }
            .advanced-options .param-row label[for="smoothing"]::before { content: "Smooth (m)"; font-size: 11px; }
        }
        .hidden { display: none; }
        /* Header styles */
        .header-section {
            margin-bottom: 20px;
            text-align: center;
        }
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .logo {
            width: 48px;
            height: 48px;
        }
        .header-section h1 {
            margin: 0;
            font-size: 1.4em;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        @media (max-width: 480px) {
            .logo { width: 36px; height: 36px; }
            .header-section { margin-bottom: 10px; }
            .header-section h1 { font-size: 1.1em; }
            .logo-container { gap: 8px; margin-bottom: 4px; }
            .tagline { font-size: 0.75em; }
            .how-link { margin-top: 4px; font-size: 0.8em; }
        }
        .tagline {
            color: var(--text-muted);
            font-size: 0.9em;
            margin: 0;
            line-height: 1.4;
        }
        .how-link {
            display: inline-block;
            margin-top: 8px;
            color: var(--primary);
            font-size: 0.9em;
            text-decoration: none;
        }
        .how-link:hover {
            text-decoration: underline;
        }
        /* Info button styles */
        .label-row {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .info-btn {
            width: 16px;
            height: 16px;
            min-width: 16px;
            min-height: 16px;
            border-radius: 50%;
            border: 1.5px solid var(--primary);
            background: white;
            color: var(--primary);
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            line-height: 1;
            flex-shrink: 0;
            vertical-align: middle;
        }
        .info-btn:hover {
            background: var(--primary);
            color: white;
        }
        .th-with-info, .label-with-info {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            vertical-align: middle;
        }
        .th-with-info .info-btn, .label-with-info .info-btn {
            margin: 0;
            top: 0;
        }
        @media (max-width: 600px) {
            .info-btn {
                width: 18px;
                height: 18px;
                min-width: 18px;
                min-height: 18px;
                font-size: 11px;
            }
        }
        /* Inline tooltip for input fields */
        .input-with-tooltip {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .input-with-tooltip input {
            flex: 1;
            min-width: 0;
        }
        .noise-warning-btn {
            background: none;
            border: none;
            color: #F57C00;
            font-size: 16px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
            flex-shrink: 0;
            width: 20px;
        }
        .noise-warning-btn:hover {
            color: #E65100;
        }
        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal {
            background: white;
            border-radius: 12px;
            max-width: 400px;
            width: 100%;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .modal.modal-large {
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal h4 {
            margin: 16px 0 8px 0;
            font-size: 1em;
            color: #444;
        }
        .modal h4:first-of-type {
            margin-top: 0;
        }
        .modal .param-list {
            margin: 0 0 12px 0;
            padding-left: 0;
            list-style: none;
        }
        .modal .param-list li {
            margin-bottom: 8px;
            padding-left: 16px;
            position: relative;
        }
        .modal .param-list li::before {
            content: "•";
            position: absolute;
            left: 0;
            color: var(--primary);
        }
        .modal .param-name {
            font-weight: 600;
            color: #333;
        }
        .modal h3 {
            margin: 0 0 12px 0;
            font-size: 1.1em;
            color: #333;
        }
        .modal p {
            margin: 0 0 16px 0;
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .modal-close {
            width: 100%;
            padding: 12px;
            background: #f0f0f0;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            color: #333;
        }
        .modal-close:hover {
            background: #e0e0e0;
        }
        /* Units toggle */
        .units-row {
            margin-top: 15px;
        }
        .toggle-label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: normal;
            color: #555;
            margin: 0;
        }
        .toggle-label input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
            accent-color: var(--primary);
        }
        /* Advanced options */
        .advanced-row {
            margin-top: 15px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .advanced-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            color: #666;
            font-size: 0.9em;
            user-select: none;
        }
        .advanced-toggle:hover {
            color: var(--primary);
        }
        .advanced-toggle .chevron {
            transition: transform 0.2s ease;
            font-size: 0.8em;
        }
        .advanced-toggle.expanded .chevron {
            transform: rotate(90deg);
        }
        .advanced-options {
            display: none;
            margin-top: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .advanced-options.visible {
            display: block;
        }
        .advanced-options .param-row {
            margin-top: 0;
        }
        .advanced-options label {
            font-size: 0.85em;
            margin-top: 10px;
        }
        .advanced-options label:first-child {
            margin-top: 0;
        }
        .advanced-options input[type="number"] {
            padding: 8px 10px;
            font-size: 14px;
        }
        .advanced-reset {
            margin-top: 12px;
            text-align: right;
        }
        .reset-btn {
            background: none;
            border: 1px solid #ccc;
            color: #666;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            cursor: pointer;
        }
        .reset-btn:hover {
            border-color: var(--primary);
            color: var(--primary);
        }
        .custom-settings {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-left: 12px;
            padding-left: 12px;
            border-left: 1px solid #ccc;
            font-size: 0.85em;
            color: #555;
        }
        .custom-settings .setting {
            white-space: nowrap;
        }
        .custom-settings .setting-label {
            color: #888;
        }
        .custom-settings .setting-value {
            font-weight: 500;
        }
        .custom-settings .setting.modified .setting-value {
            color: var(--primary);
            font-weight: 600;
        }
        @media (max-width: 600px) {
            .custom-settings {
                margin-left: 0;
                margin-top: 4px;
                padding-left: 0;
                border-left: none;
                font-size: 0.8em;
                flex-basis: 100%;
                flex-wrap: wrap;
                row-gap: 2px;
            }
            .units-row {
                flex-wrap: wrap;
            }
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.85em;
            color: #888;
        }
        .footer a {
            color: var(--primary);
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .footer-links {
            display: flex;
            gap: 20px;
        }
        .footer-version {
            color: #aaa;
            font-size: 0.9em;
        }
        .footer-copyright {
            color: #aaa;
            font-size: 0.9em;
        }
        .url-input-wrapper {
            position: relative;
        }
        .url-name-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 12px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            color: #333;
            cursor: text;
            display: none;
            align-items: center;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        .url-name-overlay.visible {
            display: flex;
        }
        .url-name-overlay .route-name {
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .url-name-overlay .route-type {
            color: #888;
            font-size: 13px;
            margin-left: 8px;
            flex-shrink: 0;
        }
        .recent-urls-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 6px 6px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 100;
            max-height: 300px;
            overflow-y: auto;
        }
        .recent-urls-dropdown.hidden {
            display: none;
        }
        .recent-url-item {
            padding: 10px 12px;
            cursor: pointer;
            font-size: 14px;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .recent-url-item:last-child {
            border-bottom: none;
        }
        .recent-url-item:hover,
        .recent-url-item.highlighted {
            background: #f0f7ff;
        }
        .recent-urls-header {
            padding: 8px 12px;
            font-size: 12px;
            color: #888;
            background: #f9f9f9;
            border-bottom: 1px solid #eee;
        }
        .recent-url-name {
            font-weight: 500;
            color: #333;
        }
        .recent-url-path {
            font-size: 12px;
            color: #888;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Compare mode styles */
        .compare-toggle {
            margin-top: 8px;
            margin-bottom: 5px;
        }
        .compare-toggle label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: normal;
            color: #555;
        }
        .compare-toggle input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
            accent-color: var(--primary);
        }
        #compareUrlWrapper {
            margin-top: 10px;
        }
        #compareUrlWrapper.hidden {
            display: none;
        }
        .compare-label {
            display: block;
            font-size: 0.9em;
            color: #555;
            margin-bottom: 4px;
        }
        /* Comparison table styles */
        .comparison-table-wrapper {
            margin: 20px 0;
            overflow-x: auto;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .comparison-table th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .comparison-table td:first-child {
            white-space: nowrap;
            padding-right: 3%;
        }
        @media (max-width: 700px) {
            .comparison-table td:first-child {
                white-space: normal;
            }
        }
        .comparison-table .route-col {
            text-align: right;
            min-width: 100px;
        }
        .comparison-table .diff-col {
            text-align: right;
            color: #666;
            font-size: 0.9em;
            min-width: 80px;
        }
        .comparison-table tr.primary {
            background: #f0f7ff;
        }
        .comparison-table tr.primary td {
            font-weight: 600;
        }
        .est-marker {
            color: #999;
            font-size: 0.85em;
            font-weight: normal;
            cursor: help;
        }
        .comparison-footnote {
            font-size: 0.8em;
            color: #888;
            margin-top: 8px;
            font-style: italic;
        }
        /* Steep comparison table - wider columns for route names on desktop */
        .steep-comparison-table th:not(:first-child) {
            min-width: 120px;
        }
        .steep-comparison-table th:first-child,
        .steep-comparison-table td:first-child {
            white-space: nowrap;
        }
        @media (max-width: 600px) {
            .steep-comparison-table th:not(:first-child) {
                min-width: auto;
            }
            .steep-comparison-table th:first-child {
                width: auto;
            }
            .steep-comparison-table th .result-badge {
                display: block;
                margin-top: 4px;
            }
        }
        /* Comparison histograms */
        .histogram-bars.comparison-mode {
            gap: 12px;  /* spacing between grade bins */
        }
        .histogram-bars.comparison-mode .histogram-bar {
            flex: 1;
            min-width: 0;
        }
        .histogram-bars.comparison-mode .bar-container {
            display: flex;
            gap: 2px;
            align-items: flex-end;
        }
        .histogram-bars.comparison-mode .bar {
            flex: 1;
        }
        .histogram-bars.comparison-mode .bar.route2 {
            background-image: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 2px,
                rgba(255,255,255,0.4) 2px,
                rgba(255,255,255,0.4) 4px
            ) !important;
        }
        /* Histogram tooltip */
        .bar-tooltip {
            position: fixed;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85em;
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .bar-tooltip .tooltip-title {
            font-weight: 600;
            margin-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            padding-bottom: 4px;
        }
        .bar-tooltip .tooltip-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        .bar-tooltip .tooltip-label {
            color: #aaa;
        }
        .histogram-legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.8em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        .legend-color.striped {
            background-image: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 2px,
                rgba(255,255,255,0.4) 2px,
                rgba(255,255,255,0.4) 4px
            );
        }
        /* Stacked elevation profiles */
        .elevation-profiles-stacked .elevation-profile {
            margin-bottom: 15px;
        }
        .elevation-profiles-stacked .elevation-profile:last-child {
            margin-bottom: 0;
        }
        /* Side-by-side RWGPS embeds */
        .route-maps-comparison {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        .route-maps-comparison .route-map {
            flex: 1;
        }
        .route-maps-comparison .route-map h4 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            color: #555;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        @media (max-width: 768px) {
            .route-maps-comparison {
                flex-direction: column;
            }
        }
    </style>
    {% if umami_website_id %}
    <script defer src="{{ umami_script_url }}" data-website-id="{{ umami_website_id }}"></script>
    {% endif %}
</head>
<body>
    <div class="header-section">
        <div class="logo-container">
            <svg class="logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="mountainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#FF6B35"/>
                        <stop offset="100%" style="stop-color:#F7931E"/>
                    </linearGradient>
                </defs>
                <!-- Mountain range -->
                <path d="M0 85 L25 45 L40 60 L60 30 L80 50 L100 85 Z" fill="url(#mountainGrad)"/>
                <!-- Snow caps -->
                <path d="M60 30 L67 42 L53 42 Z" fill="white" opacity="0.85"/>
                <path d="M25 45 L30 52 L20 52 Z" fill="white" opacity="0.7"/>
                <!-- Cyclist climbing - positioned on left slope -->
                <g transform="translate(18, 50) rotate(-20) scale(1.5)">
                    <!-- Wheels -->
                    <circle cx="0" cy="14" r="7" fill="none" stroke="#2D3047" stroke-width="1.5"/>
                    <circle cx="22" cy="14" r="7" fill="none" stroke="#2D3047" stroke-width="1.5"/>
                    <!-- Diamond frame -->
                    <path d="M0 14 L8 6 L18 6 L22 14 M8 6 L11 14 L18 6 M11 14 L0 14"
                          fill="none" stroke="#2D3047" stroke-width="1.5" stroke-linejoin="round"/>
                    <!-- Seat post -->
                    <line x1="8" y1="6" x2="7" y2="3" stroke="#2D3047" stroke-width="1.5"/>
                    <!-- Rider - aggressive climbing posture, bent forward -->
                    <line x1="7" y1="3" x2="14" y2="-1" stroke="#2D3047" stroke-width="2" stroke-linecap="round"/>
                    <!-- Head - tucked forward -->
                    <circle cx="16" cy="-2" r="2.5" fill="#2D3047"/>
                    <!-- Arms down to drops -->
                    <line x1="12" y1="-1" x2="18" y2="5" stroke="#2D3047" stroke-width="1.5" stroke-linecap="round"/>
                </g>
            </svg>
            <h1>Reality Check my Route</h1>
        </div>
        <p class="tagline">Physics-based cycling time and energy estimates from elevation, surface, and rider parameters</p>
        <a href="#" class="how-link" onclick="showModal('physicsModal'); return false;">How does it work?</a>
    </div>

    <form method="POST" id="analyzeForm">
        <div class="label-row">
            <label for="url">RideWithGPS URL (route, trip, or collection)</label>
            <button type="button" class="info-btn" onclick="showModal('urlModal')">?</button>
        </div>
        <div class="url-input-wrapper">
            <input type="text" id="url" name="url"
                   placeholder="https://ridewithgps.com/routes/... or .../collections/..."
                   value="{{ url or '' }}" required
                   autocomplete="off">
            <div class="url-name-overlay" id="urlNameOverlay" onclick="document.getElementById('url').focus(); event.stopPropagation();">
                <span class="route-name" id="urlNameText"></span>
                <span class="route-type" id="urlTypeText"></span>
            </div>
            <div id="recentUrlsDropdown" class="recent-urls-dropdown hidden"></div>
        </div>
        <input type="hidden" id="mode" name="mode" value="route">

        <div class="compare-toggle">
            <label>
                <input type="checkbox" id="compareCheckbox" {{ 'checked' if compare_mode else '' }} onchange="toggleCompareMode()">
                Compare this
            </label>
            <input type="hidden" id="compareMode" name="compare" value="{{ 'on' if compare_mode else '' }}">
        </div>
        <div id="compareUrlWrapper" class="{{ '' if compare_mode else 'hidden' }}">
            <div class="label-row">
                <label for="url2" class="compare-label">Compare URL (route or trip)</label>
                <button type="button" class="info-btn" onclick="showModal('urlModal')">?</button>
            </div>
            <div class="url-input-wrapper">
                <input type="text" id="url2" name="url2"
                       placeholder="https://ridewithgps.com/routes/... or .../trips/..."
                       value="{{ url2 or '' }}"
                       autocomplete="off">
                <div class="url-name-overlay" id="url2NameOverlay" onclick="document.getElementById('url2').focus(); event.stopPropagation();">
                    <span class="route-name" id="url2NameText"></span>
                    <span class="route-type" id="url2TypeText"></span>
                </div>
                <div id="recentUrlsDropdown2" class="recent-urls-dropdown hidden"></div>
            </div>
        </div>

        <div class="param-row">
            <div>
                <div class="label-row">
                    <label for="climbing_power">Climbing Power (W)</label>
                    <button type="button" class="info-btn" onclick="showModal('powerModal')">?</button>
                </div>
                <input type="number" id="climbing_power" name="climbing_power" value="{{ climbing_power }}" step="1">
            </div>
            <div>
                <div class="label-row">
                    <label for="flat_power">Flat Power (W)</label>
                    <button type="button" class="info-btn" onclick="showModal('flatPowerModal')">?</button>
                </div>
                <input type="number" id="flat_power" name="flat_power" value="{{ flat_power }}" step="1">
            </div>
            <div>
                <div class="label-row">
                    <label for="mass">Mass (kg)</label>
                    <button type="button" class="info-btn" onclick="showModal('massModal')">?</button>
                </div>
                <input type="number" id="mass" name="mass" value="{{ mass }}" step="0.1">
            </div>
            <div>
                <div class="label-row">
                    <label for="headwind">Headwind (km/h)</label>
                    <button type="button" class="info-btn" onclick="showModal('headwindModal')">?</button>
                </div>
                <input type="number" id="headwind" name="headwind" value="{{ headwind }}" step="0.1">
            </div>
        </div>

        <div class="units-row" id="unitsRow">
            <label class="toggle-label">
                <input type="checkbox" id="imperial" name="imperial" {{ 'checked' if imperial else '' }}>
                <span>Imperial units (mi, ft)</span>
            </label>
        </div>

        <div class="advanced-row">
            <div class="advanced-toggle" id="advancedToggle" onclick="toggleAdvanced()">
                <span class="chevron">▶</span>
                <span>Advanced Options</span>
            </div>
            <span class="custom-settings" title="Advanced physics model settings">
                <span class="setting" id="summaryDescPwr"><span class="setting-label">Desc Pwr:</span> <span class="setting-value">{{ descending_power|int }}W</span></span>
                <span class="setting" id="summaryBraking"><span class="setting-label">Braking:</span> <span class="setting-value">{{ "%.2f"|format(descent_braking_factor) }}</span></span>
                <span class="setting" id="summaryGravelPwr"><span class="setting-label">Gravel Pwr:</span> <span class="setting-value">{{ "%.2f"|format(unpaved_power_factor) }}</span></span>
                <span class="setting" id="summarySmoothing"><span class="setting-label">Smoothing:</span> <span class="setting-value">{% if result and result.effective_smoothing %}{{ result.effective_smoothing|int }}{% else %}{{ smoothing|int }}{% endif %}m</span></span>
            </span>
        </div>

        <div class="advanced-options" id="advancedOptions">
            <div class="param-row">
                <div>
                    <div class="label-row">
                        <label for="descending_power">Descent Power (W)</label>
                        <button type="button" class="info-btn" onclick="showModal('descentPowerModal')">?</button>
                    </div>
                    <input type="number" id="descending_power" name="descending_power" value="{{ descending_power }}" step="1">
                </div>
                <div>
                    <div class="label-row">
                        <label for="descent_braking_factor">Descent Braking Factor</label>
                        <button type="button" class="info-btn" onclick="showModal('descentBrakingModal')">?</button>
                    </div>
                    <input type="number" id="descent_braking_factor" name="descent_braking_factor" value="{{ descent_braking_factor }}" step="0.01" min="0.2" max="1.5">
                </div>
                <div>
                    <div class="label-row">
                        <label for="unpaved_power_factor">Gravel Power Factor</label>
                        <button type="button" class="info-btn" onclick="showModal('gravelPowerModal')">?</button>
                    </div>
                    <input type="number" id="unpaved_power_factor" name="unpaved_power_factor" value="{{ unpaved_power_factor }}" step="0.01" min="0.5" max="1.0">
                </div>
                <div>
                    <div class="label-row">
                        <label for="smoothing">Smoothing Radius (m)</label>
                        <button type="button" class="info-btn" onclick="showModal('smoothingModal')">?</button>
                    </div>
                    <div class="input-with-tooltip">
                        <input type="number" id="smoothing" name="smoothing" value="{% if result and result.effective_smoothing %}{{ result.effective_smoothing|int }}{% else %}{{ smoothing|int }}{% endif %}" step="10" min="{% if result and result.noise_ratio and result.noise_ratio > 1.8 %}300{% else %}10{% endif %}" max="400">
                        {% if result and result.noise_ratio and result.noise_ratio > 1.8 %}
                        <button type="button" class="noise-warning-btn" onclick="showModal('noiseWarningModal')">⚠</button>
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="advanced-reset">
                <button type="button" class="reset-btn" onclick="resetAdvancedOptions()">Reset to defaults</button>
            </div>
        </div>

        <button type="submit" id="submitBtn">Analyze</button>
    </form>

    <div id="progressContainer" class="progress-container hidden">
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="progress-text" id="progressText">Analyzing routes...</div>
        <div class="progress-route" id="progressRoute"></div>
    </div>

    <div id="errorContainer" class="error hidden"></div>

    <div id="collectionResults" class="results hidden">
        <input type="hidden" id="collectionShareUrl" value="">
        <div class="results-header">
            <h2 id="collectionName">Collection Analysis</h2>
            <button type="button" class="share-btn" onclick="copyShareLink('collectionShareUrl', this)" title="Copy link to share">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z"/></svg>
                <span>Share</span>
            </button>
        </div>
        <div class="result-row">
            <span class="result-label">Routes</span>
            <span class="result-value" id="totalRoutes">-</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Distance</span>
            <span class="result-value" id="totalDistance">-</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Elevation</span>
            <span class="result-value" id="totalElevation">-</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Time</span>
            <span class="result-value" id="totalTime">-</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Work</span>
            <span class="result-value" id="totalWork">-</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Energy</span>
            <span class="result-value" id="totalEnergy">-</span>
        </div>
        <table class="collection-table">
            <thead>
                <tr>
                    <th>Route</th>
                    <th class="num primary"><span class="th-with-info">Time <button type="button" class="info-btn" onclick="showModal('timeModal')">?</button></span></th>
                    <th class="num primary"><span class="th-with-info">Work <button type="button" class="info-btn" onclick="showModal('workModal')">?</button></span></th>
                    <th class="num primary separator"><span class="th-with-info">kcal <button type="button" class="info-btn" onclick="showModal('energyModal')">?</button></span></th>
                    <th class="num">Dist</th>
                    <th class="num">Elev</th>
                    <th class="num"><span class="th-with-info">Hilly <button type="button" class="info-btn" onclick="showModal('hillyModal')">?</button></span></th>
                    <th class="num"><span class="th-with-info">&gt;10% <button type="button" class="info-btn" onclick="showModal('steepTimeModal')">?</button></span></th>
                    <th class="num"><span class="th-with-info">Steep <button type="button" class="info-btn" onclick="showModal('steepModal')">?</button></span></th>
                    <th class="num">Speed</th>
                    <th class="num">Unpvd</th>
                    <th class="num"><span class="th-with-info">EScl <button type="button" class="info-btn" onclick="showModal('esclModal')">?</button></span></th>
                </tr>
            </thead>
            <tbody id="routesTableBody">
            </tbody>
        </table>
    </div>

    <!-- Info Modals -->
    <div id="urlModal" class="modal-overlay" onclick="hideModal('urlModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>RideWithGPS URL</h3>
            <p>Paste the URL of a route or collection from <a href="https://ridewithgps.com" target="_blank">ridewithgps.com</a>.</p>
            <p><strong>Route URL format:</strong><br>
            <code>https://ridewithgps.com/routes/12345678</code></p>
            <p><strong>Collection URL format:</strong><br>
            <code>https://ridewithgps.com/collections/12345</code></p>
            <p>The route must be public, or you must be logged into RideWithGPS with access to private routes.</p>
            <button class="modal-close" onclick="hideModal('urlModal')">Got it</button>
        </div>
    </div>

    <div id="powerModal" class="modal-overlay" onclick="hideModal('powerModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Climbing Power</h3>
            <p>Your sustained power output on steep climbs (grades above ~7%). Riders typically push harder uphill. For reference:</p>
            <p>• Casual climbing: 80-120W<br>• Moderate effort: 120-180W<br>• Strong rider: 180-250W</p>
            <p>For grades between 0% and 7%, power is interpolated between flat power and climbing power. Full climbing power is used only on steeper grades.</p>
            <button class="modal-close" onclick="hideModal('powerModal')">Got it</button>
        </div>
    </div>

    <div id="massModal" class="modal-overlay" onclick="hideModal('massModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Total Mass</h3>
            <p>Combined weight of rider, bike, gear, and any cargo in kilograms. This significantly affects climbing speed and energy requirements.</p>
            <p>• Rider weight + bike (~8-12kg) + gear/bags</p>
            <button class="modal-close" onclick="hideModal('massModal')">Got it</button>
        </div>
    </div>

    <div id="headwindModal" class="modal-overlay" onclick="hideModal('headwindModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Headwind</h3>
            <p>Average wind speed you expect to ride against, in km/h. Use negative values for tailwind.</p>
            <p>• Headwind (into wind): positive values<br>• Tailwind (wind behind): negative values<br>• No wind: 0</p>
            <p>Even a modest 10-15 km/h headwind significantly increases effort on flat terrain.</p>
            <button class="modal-close" onclick="hideModal('headwindModal')">Got it</button>
        </div>
    </div>

    <div id="physicsModal" class="modal-overlay" onclick="hideModal('physicsModal')">
        <div class="modal modal-large" onclick="event.stopPropagation()">
            <h3>Physics Model</h3>
            <p>Speed is calculated by solving the power balance equation: your power output equals the sum of resistive forces times velocity. All parameters below are tunable and have been calibrated against a training set of planned routes compared with actual ride data.</p>

            <h4>Primary Parameters (Biggest Impact)</h4>
            <ul class="param-list">
                <li><span class="param-name">Climbing Power (W)</span> — Power output on steep climbs. Most important input for hilly routes.</li>
                <li><span class="param-name">Flat Power (W)</span> — Power output on flat terrain. Power ramps linearly from flat to climbing power as grade increases.</li>
                <li><span class="param-name">Mass (kg)</span> — Total weight of rider + bike + gear. Dominates climbing speed since you're lifting this weight against gravity.</li>
                <li><span class="param-name">CdA (m²)</span> — Aerodynamic drag coefficient × frontal area. Controls air resistance, which grows with the cube of speed. Typical values: 0.25 (racing tuck) to 0.45 (upright touring).</li>
                <li><span class="param-name">Crr</span> — Rolling resistance coefficient. Energy lost to tire deformation and surface friction. Road tires ~0.004, gravel ~0.008-0.012.</li>
            </ul>

            <h4>Environmental Factors</h4>
            <ul class="param-list">
                <li><span class="param-name">Headwind (km/h)</span> — Wind adds to or subtracts from your effective air speed. A 15 km/h headwind at 25 km/h means you experience drag as if riding 40 km/h.</li>
                <li><span class="param-name">Air density (kg/m³)</span> — Affects aerodynamic drag. Lower at altitude (1.225 at sea level, ~1.0 at 2000m).</li>
            </ul>

            <h4>Descent Model</h4>
            <p style="margin-bottom: 0.5em; font-size: 0.9em;">Descent speed is limited by gradient steepness AND road curvature (the more restrictive wins).</p>
            <p style="margin: 0.5em 0; font-size: 0.85em;"><strong>Gradient-based:</strong></p>
            <ul class="param-list">
                <li><span class="param-name">Max coasting speed</span> — Speed limit when coasting downhill on paved roads.</li>
                <li><span class="param-name">Max coasting speed unpaved</span> — Lower speed limit for gravel/dirt descents.</li>
                <li><span class="param-name">Steep descent speed</span> — Even slower limit for very steep descents.</li>
                <li><span class="param-name">Steep descent grade</span> — Grade threshold where steep descent speed applies.</li>
                <li><span class="param-name">Coasting grade threshold</span> — Grade where you stop pedaling entirely and coast.</li>
                <li><span class="param-name">Descent braking factor</span> — Multiplier for descent speeds (1.0 = full physics speed, 0.5 = cautious braking).</li>
                <li><span class="param-name">Descent power</span> — Light pedaling power on gentle descents (drops to zero on steep grades).</li>
            </ul>
            <p style="margin: 0.5em 0; font-size: 0.85em;"><strong>Curvature-based:</strong></p>
            <ul class="param-list">
                <li><span class="param-name">Straight descent speed</span> — Max speed on straight sections (low curvature).</li>
                <li><span class="param-name">Hairpin speed</span> — Max speed through tight switchbacks (high curvature).</li>
            </ul>

            <h4>Gravel/Unpaved Model</h4>
            <ul class="param-list">
                <li><span class="param-name">Surface Crr deltas</span> — Per-surface-type rolling resistance increases based on RideWithGPS surface data. Rougher surfaces get higher Crr.</li>
                <li><span class="param-name">Gravel power factor</span> — Multiplier on power for unpaved surfaces (default 0.90 = 10% reduction). Models traction limits, vibration fatigue, and seated climbing on rough terrain.</li>
                <li><span class="param-name">Max coasting speed unpaved</span> — Lower descent speed limit on gravel/dirt roads.</li>
            </ul>
            <p style="margin: 0.3em 0; font-size: 0.85em;">Gravel sections are shown as brown strips on the elevation profile when "Show gravel" is toggled.</p>

            <h4>Data Processing</h4>
            <ul class="param-list">
                <li><span class="param-name">Smoothing radius (m)</span> — Gaussian smoothing applied to elevation data. Reduces GPS noise and unrealistic grade spikes while preserving overall climb profile.</li>
                <li><span class="param-name">Elevation scale</span> — Multiplier applied after smoothing. Auto-calculated from RideWithGPS API (DEM-corrected) elevation when available.</li>
                <li><span class="param-name">Anomaly detection</span> — Automatic detection of elevation anomalies in DEM (Digital Elevation Model) data, such as tunnels or bridges where DEM shows the surface above rather than the actual path. Anomalies appear as artificial elevation spikes. Detected anomalies are corrected by linear interpolation and highlighted with yellow bands in the elevation profile.</li>
            </ul>

            <button class="modal-close" onclick="hideModal('physicsModal')">Got it</button>
        </div>
    </div>

    <div id="flatPowerModal" class="modal-overlay" onclick="hideModal('flatPowerModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Flat Power</h3>
            <p>Your sustained power output on flat terrain. For reference:</p>
            <p>• Casual riding: 60-100W<br>• Moderate effort: 100-150W<br>• Strong rider: 150-200W</p>
            <p>Power ramps linearly from this value up to climbing power as grade increases toward ~7%.</p>
            <button class="modal-close" onclick="hideModal('flatPowerModal')">Got it</button>
        </div>
    </div>

    <div id="descentPowerModal" class="modal-overlay" onclick="hideModal('descentPowerModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Descent Power</h3>
            <p>Power output when descending (grade steeper than -2%). Models light pedaling on gentle descents.</p>
            <p><strong>0 W</strong> = pure coasting (no pedaling)<br>
            <strong>20 W</strong> = light pedaling (typical)<br>
            <strong>50+ W</strong> = active pedaling on descents</p>
            <p>On steep descents (beyond coasting threshold), power drops to zero regardless of this setting.</p>
            <button class="modal-close" onclick="hideModal('descentPowerModal')">Got it</button>
        </div>
    </div>

    <div id="descentBrakingModal" class="modal-overlay" onclick="hideModal('descentBrakingModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Descent Braking Factor</h3>
            <p>Multiplier for descent speeds. Models how cautiously you brake on descents relative to pure physics.</p>
            <p><strong>1.0</strong> = physics-based speed (aggressive)<br>
            <strong>0.5</strong> = 50% of physics speed (typical/cautious)<br>
            <strong>0.3</strong> = very cautious braking</p>
            <p>Lower values model riders who brake more on descents due to comfort, experience, or road conditions.</p>
            <button class="modal-close" onclick="hideModal('descentBrakingModal')">Got it</button>
        </div>
    </div>

    <div id="gravelPowerModal" class="modal-overlay" onclick="hideModal('gravelPowerModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Gravel Power Factor</h3>
            <p>Multiplier for power output on unpaved/gravel surfaces. Models reduced power due to traction limits, vibration fatigue, and seated climbing.</p>
            <p><strong>1.0</strong> = same power as paved<br>
            <strong>0.90</strong> = 10% power reduction (default)<br>
            <strong>0.80</strong> = 20% power reduction (rough gravel)</p>
            <p>Works alongside rolling resistance (crr) increase. Lower values produce slower gravel speed estimates.</p>
            <button class="modal-close" onclick="hideModal('gravelPowerModal')">Got it</button>
        </div>
    </div>

    <div id="smoothingModal" class="modal-overlay" onclick="hideModal('smoothingModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Smoothing Radius</h3>
            <p>Window size (in meters) for elevation smoothing. Reduces GPS noise in elevation data before calculating grades.</p>
            <p><strong>20m</strong> = default, preserves detail for clean GPS data<br>
            <strong>50-100m</strong> = moderate smoothing for typical routes<br>
            <strong>200m+</strong> = aggressive smoothing for very noisy data</p>
            <p>Higher values reduce noise but may underestimate short steep sections.</p>
            <button class="modal-close" onclick="hideModal('smoothingModal')">Got it</button>
        </div>
    </div>

    <div id="noiseWarningModal" class="modal-overlay" onclick="hideModal('noiseWarningModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>⚠ High Elevation Noise Detected</h3>
            <p>This route has noisy GPS elevation data. The <strong>elevation noise ratio</strong> ({{ "%.1f"|format(result.noise_ratio) if result and result.noise_ratio else "?" }}×) measures raw GPS elevation gain divided by the DEM-corrected gain from RideWithGPS.</p>
            <p>A ratio above 1.8× indicates the GPS recorded significantly more elevation change than the satellite-derived terrain model, typically due to GPS signal errors or poor reception.</p>
            <p><strong>Automatic adjustment:</strong> Smoothing radius is set to 300m minimum to filter this noise. You can increase it further, but not below 300m for this route.</p>
            <button class="modal-close" onclick="hideModal('noiseWarningModal')">Got it</button>
        </div>
    </div>

    <div id="esclModal" class="modal-overlay" onclick="hideModal('esclModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Elevation Scale (EScl)</h3>
            <p>A correction factor applied to the route's elevation data. GPS elevation is often inaccurate, so this scales it to match RideWithGPS's corrected elevation from their API.</p>
            <p><strong>1.00</strong> = no correction needed<br>
            <strong>&gt;1.00</strong> = route elevation was understated<br>
            <strong>&lt;1.00</strong> = route elevation was overstated</p>
            <p>Values far from 1.0 may indicate poor GPS data quality for that route.</p>
            <button class="modal-close" onclick="hideModal('esclModal')">Got it</button>
        </div>
    </div>

    <div id="timeModal" class="modal-overlay" onclick="hideModal('timeModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Est. Moving Time</h3>
            <p>Moving time estimate based on your power output and the route's terrain. This is the time you'd spend actually riding, excluding stops.</p>
            <p>The estimate accounts for:</p>
            <p>• Slower speeds on climbs (more power needed to fight gravity)<br>
            • Faster speeds on descents (limited by safety/comfort)<br>
            • Surface type (gravel/dirt is slower than pavement)</p>
            <button class="modal-close" onclick="hideModal('timeModal')">Got it</button>
        </div>
    </div>

    <div id="workModal" class="modal-overlay" onclick="hideModal('workModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Est. Work</h3>
            <p>Total mechanical energy expenditure in kilojoules (kJ). This is the energy your legs put into the pedals.</p>
            <p>Useful for estimating food/fuel needs:</p>
            <p>• Human efficiency is ~20-25%, so multiply by 4-5 for calories burned<br>
            • Example: 1000 kJ of work ≈ 4000-5000 kJ (950-1200 kcal) of food energy</p>
            <button class="modal-close" onclick="hideModal('workModal')">Got it</button>
        </div>
    </div>

    <div id="energyModal" class="modal-overlay" onclick="hideModal('energyModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Est. Energy</h3>
            <p>Estimated food energy needed to fuel this ride, accounting for ~22% human efficiency.</p>
            <p><strong>Unit equivalents:</strong></p>
            <p>• 1 medium banana ≈ 100 kcal<br>
            • 1 baguette (250g) ≈ 680 kcal</p>
            <button class="modal-close" onclick="hideModal('energyModal')">Got it</button>
        </div>
    </div>

    <div id="hillyModal" class="modal-overlay" onclick="hideModal('hillyModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Hilliness</h3>
            <p>Total elevation gain per unit distance (m/km or ft/mi). Measures <em>how much</em> climbing a route has, normalized by length.</p>
            <p>Typical values:</p>
            <p>• Flat: 0-5 m/km (0-26 ft/mi)<br>
            • Rolling: 5-15 m/km (26-79 ft/mi)<br>
            • Hilly: 15-25 m/km (79-132 ft/mi)<br>
            • Mountainous: 25+ m/km (132+ ft/mi)</p>
            <button class="modal-close" onclick="hideModal('hillyModal')">Got it</button>
        </div>
    </div>

    <div id="steepModal" class="modal-overlay" onclick="hideModal('steepModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Steepness</h3>
            <p>Effort-weighted average grade of climbs 2% and steeper. Measures <em>how steep</em> the climbs are, not just how much climbing.</p>
            <p>Steeper sections count more because they require disproportionately more power. A route with punchy 10% grades will score higher than one with gentle 4% grades, even if total climbing is similar.</p>
            <p>Typical values:</p>
            <p>• Gentle climbs: 3-5%<br>
            • Moderate climbs: 5-7%<br>
            • Steep climbs: 7-10%<br>
            • Very steep: 10%+</p>
            <button class="modal-close" onclick="hideModal('steepModal')">Got it</button>
        </div>
    </div>

    <div id="steepTimeModal" class="modal-overlay" onclick="hideModal('steepTimeModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Time at &gt;10% Grade</h3>
            <p>Total time spent climbing at grades steeper than 10%, based on the physics simulation using your input parameters (power, mass, etc.).</p>
            <p>This measures how long you'll be grinding up very steep sections. Even short steep pitches add up and can significantly affect overall effort and pacing.</p>
            <button class="modal-close" onclick="hideModal('steepTimeModal')">Got it</button>
        </div>
    </div>

    <div id="steepClimbsModal" class="modal-overlay" onclick="hideModal('steepClimbsModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Steep Climbs Methodology</h3>
            <p><strong>Max Grade</strong> is calculated using a 150m rolling average to filter GPS noise. This gives the maximum <em>sustained</em> grade over a meaningful distance.</p>
            <p><strong>Grade Histogram</strong> uses the same 150m rolling average, so grades shown will never exceed the max grade. This ensures consistency between the reported maximum and the histogram distribution.</p>
            <p><strong>Why 150m?</strong> Point-to-point GPS measurements can show unrealistic spikes (50%+ grades) due to elevation noise. Averaging over 150m filters these artifacts while still capturing steep sections that riders actually experience.</p>
            <p>Elevation data is smoothed (150m Gaussian) before grade calculation to reduce GPS noise.</p>
            <button class="modal-close" onclick="hideModal('steepClimbsModal')">Got it</button>
        </div>
    </div>

    <div id="noiseModal" class="modal-overlay" onclick="hideModal('noiseModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Elevation Noise Ratio</h3>
            <p>The <strong>noise ratio</strong> compares raw GPS elevation gain to the DEM (Digital Elevation Model) elevation gain from RideWithGPS.</p>
            <p><strong>What it means:</strong></p>
            <p>• <strong>1.0x</strong>: GPS and DEM match perfectly (rare)<br>
            • <strong>1.0-1.5x</strong>: Normal GPS noise<br>
            • <strong>1.5-1.8x</strong>: Elevated noise (tree cover, canyons)<br>
            • <strong>&gt;1.8x</strong>: High noise - smoothing auto-increased to 300m</p>
            <p><strong>Why this matters:</strong> Noisy GPS elevation creates artificial ups and downs that inflate grade calculations. Higher smoothing reduces this noise but may slightly soften real grade changes.</p>
            <p><strong>Auto-adjustment:</strong> When noise exceeds 1.8x, smoothing is automatically increased to 300m minimum. You can set a higher value in advanced options if needed.</p>
            <button class="modal-close" onclick="hideModal('noiseModal')">Got it</button>
        </div>
    </div>

    <script>
        // Recent URLs management
        var MAX_RECENT_URLS = 20;

        function getRecentUrls() {
            try {
                var data = JSON.parse(localStorage.getItem('recentUrls') || '[]');
                // Handle migration from old format (array of strings) to new format (array of objects)
                if (data.length > 0 && typeof data[0] === 'string') {
                    data = data.map(function(url) { return {url: url, name: null}; });
                    localStorage.setItem('recentUrls', JSON.stringify(data));
                }
                return data;
            } catch (e) {
                return [];
            }
        }

        function saveRecentUrl(url, name) {
            if (!url || !url.includes('ridewithgps.com')) return;
            var urls = getRecentUrls();
            // Remove if already exists (will re-add at top)
            urls = urls.filter(function(u) { return u.url !== url; });
            // Add to front
            urls.unshift({url: url, name: name || null});
            // Keep only last N
            urls = urls.slice(0, MAX_RECENT_URLS);
            localStorage.setItem('recentUrls', JSON.stringify(urls));
            populateRecentUrls();
            // Update name overlays (delayed to avoid interfering with current focus)
            setTimeout(function() {
                updateUrlNameOverlay('url', 'urlNameOverlay', 'urlNameText', 'urlTypeText');
                updateUrlNameOverlay('url2', 'url2NameOverlay', 'url2NameText', 'url2TypeText');
            }, 100);
        }

        function getNameForUrl(url) {
            if (!url) return null;
            var urls = getRecentUrls();
            for (var i = 0; i < urls.length; i++) {
                if (urls[i].url === url) {
                    return urls[i].name;
                }
            }
            return null;
        }

        function getUrlType(url) {
            if (!url) return '';
            if (url.includes('/collections/')) return 'collection';
            if (url.includes('/trips/')) return 'trip';
            if (url.includes('/routes/')) return 'route';
            return '';
        }

        function updateUrlNameOverlay(inputId, overlayId, textId, typeId) {
            var input = document.getElementById(inputId);
            var overlay = document.getElementById(overlayId);
            var text = document.getElementById(textId);
            var typeEl = document.getElementById(typeId);
            if (!input || !overlay || !text) return;

            var url = input.value.trim();
            var name = getNameForUrl(url);
            var type = getUrlType(url);

            // Only show overlay if input is not focused and we have a name
            if (name && document.activeElement !== input) {
                text.textContent = name;
                if (typeEl) typeEl.textContent = type ? '(' + type + ')' : '';
                overlay.classList.add('visible');
            } else {
                overlay.classList.remove('visible');
            }
        }

        function hideUrlNameOverlay(overlayId) {
            var overlay = document.getElementById(overlayId);
            if (overlay) overlay.classList.remove('visible');
        }

        function showUrlNameOverlay(inputId, overlayId, textId, typeId, dropdownId) {
            // Small delay to let dropdown click complete
            setTimeout(function() {
                // Don't show overlay if dropdown is still visible (user might be clicking it)
                var dropdown = document.getElementById(dropdownId);
                if (dropdown && !dropdown.classList.contains('hidden')) {
                    return;
                }
                updateUrlNameOverlay(inputId, overlayId, textId, typeId);
            }, 200);
        }

        function populateRecentUrls(dropdownId, inputId, excludeCurrentUrl) {
            var dropdown = document.getElementById(dropdownId);
            if (!dropdown) return;  // Guard against missing element
            var urls = getRecentUrls();
            // For compare dropdown, exclude the URL already in the primary input
            if (excludeCurrentUrl) {
                var primaryUrl = document.getElementById('url').value;
                if (primaryUrl) {
                    urls = urls.filter(function(item) { return item.url !== primaryUrl; });
                }
            }
            if (urls.length === 0) {
                dropdown.innerHTML = '';
                return;
            }
            var html = '<div class="recent-urls-header">Recent</div>';
            urls.forEach(function(item) {
                var urlPreview = item.url.replace('https://ridewithgps.com/', '');
                if (item.name) {
                    html += '<div class="recent-url-item" data-url="' + item.url + '">' +
                            '<div class="recent-url-name">' + item.name + '</div>' +
                            '<div class="recent-url-path">' + urlPreview + '</div>' +
                            '</div>';
                } else {
                    html += '<div class="recent-url-item" data-url="' + item.url + '">' +
                            '<div class="recent-url-path">' + urlPreview + '</div>' +
                            '</div>';
                }
            });
            dropdown.innerHTML = html;

            // Add click handlers
            dropdown.querySelectorAll('.recent-url-item').forEach(function(item) {
                item.addEventListener('click', function() {
                    document.getElementById(inputId).value = this.getAttribute('data-url');
                    dropdown.classList.add('hidden');
                    if (inputId === 'url') {
                        updateModeIndicator();
                        updateUrlNameOverlay('url', 'urlNameOverlay', 'urlNameText', 'urlTypeText');
                    } else if (inputId === 'url2') {
                        updateUrlNameOverlay('url2', 'url2NameOverlay', 'url2NameText', 'url2TypeText');
                    }
                });
            });
        }

        function setupUrlDropdown() {
            // Setup primary URL input
            var urlInput = document.getElementById('url');
            var dropdown = document.getElementById('recentUrlsDropdown');

            urlInput.addEventListener('focus', function() {
                hideUrlNameOverlay('urlNameOverlay');
                if (getRecentUrls().length > 0) {
                    populateRecentUrls('recentUrlsDropdown', 'url', false);
                    dropdown.classList.remove('hidden');
                }
            });

            urlInput.addEventListener('input', updateModeIndicator);

            // Show name overlay on blur (after paste or manual entry)
            urlInput.addEventListener('blur', function() {
                showUrlNameOverlay('url', 'urlNameOverlay', 'urlNameText', 'urlTypeText', 'recentUrlsDropdown');
            });

            // Setup secondary URL input (compare mode)
            var url2Input = document.getElementById('url2');
            var dropdown2 = document.getElementById('recentUrlsDropdown2');

            url2Input.addEventListener('focus', function() {
                hideUrlNameOverlay('url2NameOverlay');
                var urls = getRecentUrls().filter(function(item) { return item.url.includes('/routes/'); });
                if (urls.length > 0) {
                    populateRecentUrls('recentUrlsDropdown2', 'url2', true);
                    dropdown2.classList.remove('hidden');
                }
            });

            // Show name overlay on blur for url2
            url2Input.addEventListener('blur', function() {
                showUrlNameOverlay('url2', 'url2NameOverlay', 'url2NameText', 'url2TypeText', 'recentUrlsDropdown2');
            });

            // Initialize name overlays on page load
            updateUrlNameOverlay('url', 'urlNameOverlay', 'urlNameText', 'urlTypeText');
            updateUrlNameOverlay('url2', 'url2NameOverlay', 'url2NameText', 'url2TypeText');

            // Keyboard navigation helper
            function handleDropdownKeyboard(e, inputId, dropdownId) {
                var dropdown = document.getElementById(dropdownId);

                // If dropdown is hidden and ArrowDown pressed, show it
                if (dropdown.classList.contains('hidden')) {
                    if (e.key === 'ArrowDown' && getRecentUrls().length > 0) {
                        e.preventDefault();
                        var excludeCurrent = (inputId === 'url2');
                        populateRecentUrls(dropdownId, inputId, excludeCurrent);
                        dropdown.classList.remove('hidden');
                        return true;
                    }
                    return false;
                }

                var items = dropdown.querySelectorAll('.recent-url-item');
                if (items.length === 0) return false;

                // Find current highlighted index
                var currentIndex = -1;
                for (var i = 0; i < items.length; i++) {
                    if (items[i].classList.contains('highlighted')) {
                        currentIndex = i;
                        break;
                    }
                }

                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    // Clear all highlights
                    items.forEach(function(item) { item.classList.remove('highlighted'); });
                    var nextIndex = (currentIndex + 1) % items.length;
                    items[nextIndex].classList.add('highlighted');
                    items[nextIndex].scrollIntoView({ block: 'nearest' });
                    return true;
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    // Clear all highlights
                    items.forEach(function(item) { item.classList.remove('highlighted'); });
                    var prevIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1;
                    items[prevIndex].classList.add('highlighted');
                    items[prevIndex].scrollIntoView({ block: 'nearest' });
                    return true;
                } else if (e.key === 'Enter' && currentIndex >= 0) {
                    e.preventDefault();
                    items[currentIndex].click();
                    return true;
                }
                return false;
            }

            // Hide dropdowns when clicking outside
            document.addEventListener('click', function(e) {
                if (!urlInput.contains(e.target) && !dropdown.contains(e.target)) {
                    dropdown.classList.add('hidden');
                }
                if (!url2Input.contains(e.target) && !dropdown2.contains(e.target)) {
                    dropdown2.classList.add('hidden');
                }
            });

            // Keyboard navigation for dropdowns
            urlInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    dropdown.classList.add('hidden');
                } else {
                    handleDropdownKeyboard(e, 'url', 'recentUrlsDropdown');
                }
            });
            url2Input.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    dropdown2.classList.add('hidden');
                } else {
                    handleDropdownKeyboard(e, 'url2', 'recentUrlsDropdown2');
                }
            });

            // Initialize mode indicator
            updateModeIndicator();

            // Prepopulate with example route on first visit
            if (getRecentUrls().length === 0 && !urlInput.value) {
                urlInput.value = 'https://ridewithgps.com/routes/48889111';
                updateModeIndicator();
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            setupUrlDropdown();
            initAdvancedOptions();
            initCollapseStops();
        });
        if (document.readyState !== 'loading') {
            setupUrlDropdown();
            initAdvancedOptions();
            initCollapseStops();
        }

        function showModal(id) {
            document.getElementById(id).classList.add('active');
        }

        function hideModal(id) {
            document.getElementById(id).classList.remove('active');
        }

        function toggleAdvanced() {
            var toggle = document.getElementById('advancedToggle');
            var options = document.getElementById('advancedOptions');
            toggle.classList.toggle('expanded');
            options.classList.toggle('visible');
            // Save state to localStorage
            localStorage.setItem('advancedOptionsExpanded', options.classList.contains('visible'));
        }

        function initAdvancedOptions() {
            // Restore advanced options visibility from localStorage
            var expanded = localStorage.getItem('advancedOptionsExpanded') === 'true';
            if (expanded) {
                var toggle = document.getElementById('advancedToggle');
                var options = document.getElementById('advancedOptions');
                if (toggle && options) {
                    toggle.classList.add('expanded');
                    options.classList.add('visible');
                }
            }
            // Sync summary with actual input values
            updateAdvancedSummary();
            // Add listeners to keep summary in sync
            ['descending_power', 'descent_braking_factor', 'unpaved_power_factor', 'smoothing'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.addEventListener('input', updateAdvancedSummary);
            });
        }

        function resetAdvancedOptions() {
            document.getElementById('descent_braking_factor').value = {{ defaults.descent_braking_factor }};
            document.getElementById('descending_power').value = {{ defaults.descending_power|int }};
            document.getElementById('unpaved_power_factor').value = {{ defaults.unpaved_power_factor }};
            var smoothingInput = document.getElementById('smoothing');
            var minSmoothing = parseInt(smoothingInput.min) || 10;
            smoothingInput.value = Math.max({{ defaults.smoothing|int }}, minSmoothing);
            updateAdvancedSummary();
        }

        function updateAdvancedSummary() {
            var defaults = {
                descending_power: {{ defaults.descending_power|int }},
                descent_braking_factor: {{ defaults.descent_braking_factor }},
                unpaved_power_factor: {{ defaults.unpaved_power_factor }},
                smoothing: {{ defaults.smoothing|int }}
            };

            var descPwr = parseInt(document.getElementById('descending_power').value) || 0;
            var braking = parseFloat(document.getElementById('descent_braking_factor').value) || 0;
            var gravelPwr = parseFloat(document.getElementById('unpaved_power_factor').value) || 0;
            var smoothingInput = document.getElementById('smoothing');
            var smoothing = parseInt(smoothingInput.value) || 0;
            var minSmoothing = parseInt(smoothingInput.min) || 10;

            var summaryDescPwr = document.getElementById('summaryDescPwr');
            var summaryBraking = document.getElementById('summaryBraking');
            var summaryGravelPwr = document.getElementById('summaryGravelPwr');
            var summarySmoothing = document.getElementById('summarySmoothing');

            if (summaryDescPwr) {
                summaryDescPwr.querySelector('.setting-value').textContent = descPwr + 'W';
                summaryDescPwr.classList.toggle('modified', descPwr !== defaults.descending_power);
            }
            if (summaryBraking) {
                summaryBraking.querySelector('.setting-value').textContent = braking.toFixed(2);
                summaryBraking.classList.toggle('modified', Math.abs(braking - defaults.descent_braking_factor) > 0.001);
            }
            if (summaryGravelPwr) {
                summaryGravelPwr.querySelector('.setting-value').textContent = gravelPwr.toFixed(2);
                summaryGravelPwr.classList.toggle('modified', Math.abs(gravelPwr - defaults.unpaved_power_factor) > 0.001);
            }
            if (summarySmoothing) {
                summarySmoothing.querySelector('.setting-value').textContent = smoothing + 'm';
                // Modified if different from default (considering auto-adjusted minimum)
                var isModified = (minSmoothing > defaults.smoothing) ? (smoothing > minSmoothing) : (smoothing !== defaults.smoothing);
                summarySmoothing.classList.toggle('modified', isModified);
            }
        }

        function _buildOverlayParams() {
            var params = '';
            var speedCheckbox = document.getElementById('overlay_speed');
            if (speedCheckbox && speedCheckbox.checked) params += '&overlay=speed';
            var gravelCheckbox = document.getElementById('overlay_gravel');
            if (gravelCheckbox && gravelCheckbox.checked) params += '&show_gravel=true';
            if (isImperial()) params += '&imperial=true';
            return params;
        }

        function _getEffectiveTimeHours(container, collapseCheckbox) {
            // Return the effective time (moving or elapsed) based on checkbox state
            if (collapseCheckbox && collapseCheckbox.checked) {
                return parseFloat(container.getAttribute('data-moving-time-hours') || 0);
            } else {
                return parseFloat(container.getAttribute('data-elapsed-time-hours') ||
                                  container.getAttribute('data-moving-time-hours') || 0);
            }
        }

        function _recalcMaxXlimHours() {
            // Recalculate synchronized max_xlim_hours based on current checkbox states
            var container1 = document.getElementById('elevationContainer1');
            var container2 = document.getElementById('elevationContainer2');
            if (!container1 || !container2) return null;  // Not in comparison mode

            var cb1 = document.getElementById('collapseStops1');
            var cb2 = document.getElementById('collapseStops2');
            var t1 = _getEffectiveTimeHours(container1, cb1);
            var t2 = _getEffectiveTimeHours(container2, cb2);
            return Math.max(t1, t2);
        }

        function _refreshComparisonProfiles() {
            // Refresh both profiles with synchronized max_xlim_hours
            var maxXlim = _recalcMaxXlimHours();
            if (maxXlim === null) return;  // Not in comparison mode

            ['1', '2'].forEach(function(suffix) {
                var container = document.getElementById('elevationContainer' + suffix);
                var img = document.getElementById('elevationImg' + suffix);
                var loading = document.getElementById('elevationLoading' + suffix);
                if (!container || !img) return;

                var baseProfileUrl = container.getAttribute('data-base-profile-url');
                var baseDataUrl = container.getAttribute('data-base-data-url');
                if (!baseProfileUrl) return;

                var collapseCheckbox = document.getElementById('collapseStops' + suffix);
                var collapseParam = (collapseCheckbox && collapseCheckbox.checked) ? '&collapse_stops=true' : '';
                var overlayParams = _buildOverlayParams();

                // Update stored max_xlim_hours
                container.setAttribute('data-max-xlim-hours', maxXlim.toFixed(4));

                // Show loading spinner
                if (loading) {
                    loading.classList.remove('hidden');
                    img.classList.add('loading');
                }

                // Update image source with new max_xlim_hours
                img.src = baseProfileUrl + '&max_xlim_hours=' + maxXlim.toFixed(4) + collapseParam + overlayParams;

                // Re-setup the elevation profile tooltip
                if (typeof window.setupElevationProfile === 'function' && baseDataUrl) {
                    window.setupElevationProfile(
                        'elevationContainer' + suffix,
                        'elevationImg' + suffix,
                        'elevationTooltip' + suffix,
                        'elevationCursor' + suffix,
                        baseDataUrl + collapseParam,
                        maxXlim
                    );
                }
            });
        }

        function toggleCollapseStops(profileNum) {
            // Handle both single profile (no num) and comparison mode (1 or 2)
            var suffix = profileNum ? profileNum : '';
            var checkbox = document.getElementById('collapseStops' + suffix);
            var container = document.getElementById('elevationContainer' + suffix);
            var img = document.getElementById('elevationImg' + suffix);
            var loading = document.getElementById('elevationLoading' + suffix);

            if (!checkbox || !container || !img) return;

            // Save preference to localStorage
            localStorage.setItem('collapseStops' + suffix, checkbox.checked);

            // In comparison mode, refresh both profiles with synchronized x-axis
            var container1 = document.getElementById('elevationContainer1');
            var container2 = document.getElementById('elevationContainer2');
            if (container1 && container2) {
                _refreshComparisonProfiles();
                return;
            }

            // Single profile mode - original behavior
            var baseProfileUrl = container.getAttribute('data-base-profile-url');
            var baseDataUrl = container.getAttribute('data-base-data-url');
            var maxXlimHours = container.getAttribute('data-max-xlim-hours');
            var maxXlim = maxXlimHours ? parseFloat(maxXlimHours) : null;

            if (!baseProfileUrl) return;

            var collapseParam = checkbox.checked ? '&collapse_stops=true' : '';
            var overlayParams = _buildOverlayParams();

            // Show loading spinner
            loading.classList.remove('hidden');
            img.classList.add('loading');

            // Update image source
            img.src = baseProfileUrl + collapseParam + overlayParams;

            // Re-setup the elevation profile tooltip with new data URL
            if (typeof window.setupElevationProfile === 'function' && baseDataUrl) {
                window.setupElevationProfile(
                    'elevationContainer' + suffix,
                    'elevationImg' + suffix,
                    'elevationTooltip' + suffix,
                    'elevationCursor' + suffix,
                    baseDataUrl + collapseParam,
                    maxXlim
                );
            }
        }

        function initCollapseStops() {
            // Restore collapse stops preferences from localStorage
            ['', '1', '2'].forEach(function(suffix) {
                var checkbox = document.getElementById('collapseStops' + suffix);
                if (checkbox) {
                    var saved = localStorage.getItem('collapseStops' + suffix) === 'true';
                    if (saved) {
                        checkbox.checked = true;
                        toggleCollapseStops(suffix === '' ? null : parseInt(suffix));
                    }
                }
            });
        }

        function goToRidePage() {
            var urlInput = document.getElementById('url');
            if (!urlInput || !urlInput.value) return;
            var url = '/ride?url=' + encodeURIComponent(urlInput.value);
            url += '&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&smoothing={{ smoothing }}';
            if (document.getElementById('imperial')?.checked) url += '&imperial=1';
            // Save current overlay states to localStorage for ride page to read
            try {
                localStorage.setItem('ride_show_speed', document.getElementById('overlay_speed')?.checked || false);
                localStorage.setItem('ride_show_gravel', document.getElementById('overlay_gravel')?.checked || false);
                localStorage.setItem('ride_imperial', document.getElementById('imperial')?.checked || false);
            } catch (e) {}
            window.location.href = url;
        }

        function toggleOverlay(type) {
            var checkbox = document.getElementById('overlay_' + type);
            if (!checkbox) return;

            // Refresh all visible elevation profiles
            ['', '1', '2'].forEach(function(suffix) {
                var container = document.getElementById('elevationContainer' + suffix);
                var img = document.getElementById('elevationImg' + suffix);
                var loading = document.getElementById('elevationLoading' + suffix);
                if (!container || !img) return;

                var baseProfileUrl = container.getAttribute('data-base-profile-url');
                var baseDataUrl = container.getAttribute('data-base-data-url');
                var maxXlimHours = container.getAttribute('data-max-xlim-hours');
                var maxXlim = maxXlimHours ? parseFloat(maxXlimHours) : null;
                if (!baseProfileUrl) return;

                // Build collapse_stops param for this specific profile
                var collapseCheckbox = document.getElementById('collapseStops' + suffix);
                var collapseParam = (collapseCheckbox && collapseCheckbox.checked) ? '&collapse_stops=true' : '';
                var overlayParams = _buildOverlayParams();

                // Preserve zoom state from container data attributes
                var zoomParams = '';
                var zoomMin = container.getAttribute('data-zoom-min');
                var zoomMax = container.getAttribute('data-zoom-max');
                if (zoomMin && zoomMax) {
                    zoomParams = '&min_xlim_hours=' + zoomMin + '&max_xlim_hours=' + zoomMax;
                }

                loading.classList.remove('hidden');
                img.classList.add('loading');
                img.src = baseProfileUrl + collapseParam + overlayParams + zoomParams;

                if (typeof window.setupElevationProfile === 'function' && baseDataUrl) {
                    window.setupElevationProfile(
                        'elevationContainer' + suffix,
                        'elevationImg' + suffix,
                        'elevationTooltip' + suffix,
                        'elevationCursor' + suffix,
                        baseDataUrl + collapseParam,
                        maxXlim
                    );
                }
            });

            localStorage.setItem('overlay_' + type, checkbox.checked);
        }

        function initOverlays() {
            ['speed', 'gravel'].forEach(function(type) {
                var checkbox = document.getElementById('overlay_' + type);
                if (checkbox) {
                    var saved = localStorage.getItem('overlay_' + type) === 'true';
                    if (saved) {
                        checkbox.checked = true;
                        toggleOverlay(type);
                    }
                }
            });
        }

        function toggleCompareMode() {
            var wrapper = document.getElementById('compareUrlWrapper');
            var checkbox = document.getElementById('compareCheckbox');
            var hiddenInput = document.getElementById('compareMode');

            if (checkbox.checked) {
                wrapper.classList.remove('hidden');
                hiddenInput.value = 'on';
            } else {
                wrapper.classList.add('hidden');
                hiddenInput.value = '';
            }
        }

        function copyShareLink(inputId, btn) {
            var shareUrl = document.getElementById(inputId).value;
            var originalHtml = btn.innerHTML;
            navigator.clipboard.writeText(shareUrl).then(function() {
                btn.innerHTML = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg><span>Copied</span>';
                btn.classList.add('copied');
                setTimeout(function() {
                    btn.innerHTML = originalHtml;
                    btn.classList.remove('copied');
                }, 2000);
            }).catch(function() {
                // Fallback for older browsers
                prompt('Copy this link:', shareUrl);
            });
        }

        function detectModeFromUrl(url) {
            if (url.includes('/collections/')) {
                return 'collection';
            } else if (url.includes('/routes/')) {
                return 'route';
            }
            return null;
        }

        function updateModeIndicator() {
            var url = document.getElementById('url').value;
            var mode = detectModeFromUrl(url);
            var modeInput = document.getElementById('mode');
            var compareToggle = document.querySelector('.compare-toggle');
            var compareCheckbox = document.getElementById('compareCheckbox');

            if (mode === 'collection') {
                modeInput.value = 'collection';
                // Hide compare mode for collections
                if (compareCheckbox.checked) {
                    compareCheckbox.checked = false;
                    toggleCompareMode();
                }
                compareToggle.classList.add('hidden');
            } else {
                modeInput.value = 'route';
                compareToggle.classList.remove('hidden');
            }
        }

        function formatDuration(seconds) {
            var hours = Math.floor(seconds / 3600);
            var minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return hours + 'h ' + String(minutes).padStart(2, '0') + 'm';
            }
            return minutes + 'm';
        }

        function hideAllResults() {
            document.getElementById('progressContainer').classList.add('hidden');
            document.getElementById('errorContainer').classList.add('hidden');
            document.getElementById('collectionResults').classList.add('hidden');
            document.body.classList.remove('collection-mode');
            // Hide server-rendered single route results
            var singleResults = document.getElementById('singleRouteResults');
            if (singleResults) singleResults.style.display = 'none';
            // Hide server-rendered errors
            var serverErrors = document.querySelectorAll('.server-error');
            serverErrors.forEach(function(el) { el.style.display = 'none'; });
        }

        function showError(message) {
            hideAllResults();
            var errorEl = document.getElementById('errorContainer');
            errorEl.textContent = message;
            errorEl.classList.remove('hidden');
        }

        function isImperial() {
            return document.getElementById('imperial').checked;
        }

        function formatDist(km) {
            if (isImperial()) {
                return Math.round(km * 0.621371) + 'mi';
            }
            return Math.round(km) + 'km';
        }

        function formatElev(m) {
            if (isImperial()) {
                return Math.round(m * 3.28084) + "'";
            }
            return Math.round(m) + 'm';
        }

        function formatSpeed(kmh) {
            if (isImperial()) {
                return (kmh * 0.621371).toFixed(1);
            }
            return kmh.toFixed(1);
        }

        function formatHilliness(mkm) {
            if (isImperial()) {
                // m/km to ft/mi: (3.28084 ft/m) / (0.621371 mi/km) ≈ 5.28
                return Math.round(mkm * 5.28);
            }
            return Math.round(mkm);
        }

        function formatSteepTime(seconds) {
            if (!seconds || seconds < 60) return '-';
            var mins = Math.round(seconds / 60);
            if (mins < 60) return mins + 'm';
            var hours = Math.floor(mins / 60);
            var remainMins = mins % 60;
            return hours + 'h ' + (remainMins < 10 ? '0' : '') + remainMins + 'm';
        }

        function formatDistFull(km) {
            if (isImperial()) {
                return Math.round(km * 0.621371) + ' mi';
            }
            return Math.round(km) + ' km';
        }

        function formatElevFull(m) {
            if (isImperial()) {
                return Math.round(m * 3.28084) + ' ft';
            }
            return Math.round(m) + ' m';
        }

        function buildAnalyzeUrl(routeUrl) {
            var params = new URLSearchParams({
                url: routeUrl,
                climbing_power: document.getElementById('climbing_power').value,
                flat_power: document.getElementById('flat_power').value,
                mass: document.getElementById('mass').value,
                headwind: document.getElementById('headwind').value
            });
            if (document.getElementById('imperial').checked) {
                params.set('imperial', '1');
            }
            return window.location.origin + window.location.pathname + '?' + params.toString();
        }

        function updateTotals(routes) {
            var totalDist = 0, totalElev = 0, totalTime = 0, totalWork = 0, totalSteepTime = 0;
            routes.forEach(function(r) {
                totalDist += r.distance_km;
                totalElev += r.elevation_m;
                totalTime += r.time_seconds;
                totalWork += r.work_kj;
                totalSteepTime += r.steep_time_seconds || 0;
            });
            document.getElementById('totalRoutes').textContent = routes.length;
            document.getElementById('totalDistance').textContent = formatDistFull(totalDist);
            document.getElementById('totalElevation').textContent = formatElevFull(totalElev);
            document.getElementById('totalTime').textContent = formatDuration(totalTime);
            document.getElementById('totalWork').textContent = Math.round(totalWork) + ' kJ';
            document.getElementById('totalEnergy').textContent = Math.round(totalWork * 1.075) + ' kcal';

            // Update totals row
            var tbody = document.getElementById('routesTableBody');
            var existingTotals = tbody.querySelector('.totals-row');
            if (existingTotals) {
                existingTotals.remove();
            }
            var totalsRow = document.createElement('tr');
            totalsRow.className = 'totals-row';
            totalsRow.innerHTML = '<td>Total</td>' +
                '<td class="num primary">' + formatDuration(totalTime) + '</td>' +
                '<td class="num primary">' + Math.round(totalWork) + 'kJ</td>' +
                '<td class="num primary separator">' + Math.round(totalWork * 1.075) + '</td>' +
                '<td class="num">' + formatDist(totalDist) + '</td>' +
                '<td class="num">' + formatElev(totalElev) + '</td>' +
                '<td class="num"></td><td class="num">' + formatSteepTime(totalSteepTime) + '</td><td class="num"></td><td class="num"></td><td class="num"></td><td class="num"></td>';
            tbody.appendChild(totalsRow);
        }

        document.getElementById('analyzeForm').addEventListener('submit', function(e) {
            var mode = document.getElementById('mode').value;
            if (mode !== 'collection') {
                // For single routes, URL will be saved after results load (with name)
                return;
            }

            e.preventDefault();
            hideAllResults();

            var url = document.getElementById('url').value;
            var climbing_power = document.getElementById('climbing_power').value;
            var flat_power = document.getElementById('flat_power').value;
            var mass = document.getElementById('mass').value;
            var headwind = document.getElementById('headwind').value;

            var submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';

            document.getElementById('progressContainer').classList.remove('hidden');
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = 'Connecting...';
            document.getElementById('progressRoute').textContent = '';

            // Clear previous results
            document.getElementById('routesTableBody').innerHTML = '';
            collectionRoutes = [];

            var params = new URLSearchParams({
                url: url,
                climbing_power: climbing_power,
                flat_power: flat_power,
                descending_power: document.getElementById('descending_power').value,
                mass: mass,
                headwind: headwind,
                descent_braking_factor: document.getElementById('descent_braking_factor').value,
                unpaved_power_factor: document.getElementById('unpaved_power_factor').value,
                smoothing: document.getElementById('smoothing').value
            });

            var eventSource = new EventSource('/analyze-collection-stream?' + params.toString());

            eventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);

                if (data.type === 'start') {
                    var collectionNameEl = document.getElementById('collectionName');
                    var nameText = data.name || 'Collection Analysis';
                    collectionNameEl.innerHTML = '<a href="' + url + '" target="_blank">' + nameText + '</a>';
                    document.getElementById('progressText').textContent =
                        'Analyzing route 0 of ' + data.total + '...';
                    // Update page title
                    document.title = nameText + ' | Reality Check my Route';
                    // Track collection analysis with Umami
                    if (typeof umami !== 'undefined') {
                        umami.track('analyze', {
                            type: 'collection',
                            url: url,
                            name: nameText
                        });
                    }
                    // Save URL with collection name
                    saveRecentUrl(url, data.name);
                    // Build share URL with all parameters
                    var shareParams = new URLSearchParams({
                        url: url,
                        climbing_power: document.getElementById('climbing_power').value,
                        flat_power: document.getElementById('flat_power').value,
                        descending_power: document.getElementById('descending_power').value,
                        mass: document.getElementById('mass').value,
                        headwind: document.getElementById('headwind').value,
                        descent_braking_factor: document.getElementById('descent_braking_factor').value,
                        unpaved_power_factor: document.getElementById('unpaved_power_factor').value,
                        smoothing: document.getElementById('smoothing').value
                    });
                    if (document.getElementById('imperial').checked) {
                        shareParams.set('imperial', '1');
                    }
                    var shareUrl = window.location.origin + window.location.pathname + '?' + shareParams.toString();
                    document.getElementById('collectionShareUrl').value = shareUrl;
                    // Update browser URL and title
                    history.pushState({}, nameText + ' | Reality Check my Route', shareUrl);
                } else if (data.type === 'progress') {
                    document.getElementById('progressText').textContent =
                        'Analyzing route ' + data.current + ' of ' + data.total + '...';
                    document.getElementById('progressRoute').textContent = '';
                } else if (data.type === 'route') {
                    collectionRoutes.push(data.route);
                    var r = data.route;

                    // Update progress bar to show completed route
                    var pct = (collectionRoutes.length / data.total * 100).toFixed(0);
                    document.getElementById('progressFill').style.width = pct + '%';
                    document.getElementById('progressRoute').textContent = r.name || '';

                    var row = document.createElement('tr');
                    var rwgpsUrl = 'https://ridewithgps.com/routes/' + r.route_id;
                    var analyzeUrl = buildAnalyzeUrl(rwgpsUrl);
                    row.innerHTML = '<td class="route-name" title="' + r.name + '"><a href="' + analyzeUrl + '">' + r.name + '</a><a href="' + rwgpsUrl + '" target="_blank" class="rwgps-link" title="View on RideWithGPS">↗</a></td>' +
                        '<td class="num primary">' + r.time_str + '</td>' +
                        '<td class="num primary">' + Math.round(r.work_kj) + 'kJ</td>' +
                        '<td class="num primary separator">' + Math.round(r.work_kj * 1.075) + '</td>' +
                        '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                        '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
                        '<td class="num">' + formatHilliness(r.hilliness_score || 0) + '</td>' +
                        '<td class="num">' + formatSteepTime(r.steep_time_seconds) + '</td>' +
                        '<td class="num">' + (r.steepness_score || 0).toFixed(1) + '%</td>' +
                        '<td class="num">' + formatSpeed(r.avg_speed_kmh) + '</td>' +
                        '<td class="num">' + Math.round(r.unpaved_pct || 0) + '%</td>' +
                        '<td class="num">' + r.elevation_scale.toFixed(2) + '</td>';
                    document.getElementById('routesTableBody').appendChild(row);

                    // Show results container and update totals
                    document.getElementById('collectionResults').classList.remove('hidden');
                    document.body.classList.add('collection-mode');
                    updateTotals(collectionRoutes);
                } else if (data.type === 'complete') {
                    eventSource.close();
                    document.getElementById('progressContainer').classList.add('hidden');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Analyze';
                } else if (data.type === 'error') {
                    eventSource.close();
                    showError(data.message);
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Analyze';
                }
            };

            eventSource.onerror = function() {
                eventSource.close();
                showError('Connection lost. Please try again.');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Analyze';
            };
        });

        // Store routes globally so we can re-render when units change
        var collectionRoutes = [];

        function rerenderCollectionTable() {
            if (collectionRoutes.length === 0) return;

            var tbody = document.getElementById('routesTableBody');
            tbody.innerHTML = '';

            collectionRoutes.forEach(function(r) {
                var row = document.createElement('tr');
                var rwgpsUrl = 'https://ridewithgps.com/routes/' + r.route_id;
                var analyzeUrl = buildAnalyzeUrl(rwgpsUrl);
                row.innerHTML = '<td class="route-name" title="' + r.name + '"><a href="' + analyzeUrl + '">' + r.name + '</a><a href="' + rwgpsUrl + '" target="_blank" class="rwgps-link" title="View on RideWithGPS">↗</a></td>' +
                    '<td class="num primary">' + r.time_str + '</td>' +
                    '<td class="num primary">' + Math.round(r.work_kj) + 'kJ</td>' +
                    '<td class="num primary separator">' + Math.round(r.work_kj * 1.075) + '</td>' +
                    '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                    '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
                    '<td class="num">' + formatHilliness(r.hilliness_score || 0) + '</td>' +
                    '<td class="num">' + formatSteepTime(r.steep_time_seconds) + '</td>' +
                    '<td class="num">' + (r.steepness_score || 0).toFixed(1) + '%</td>' +
                    '<td class="num">' + formatSpeed(r.avg_speed_kmh) + '</td>' +
                    '<td class="num">' + Math.round(r.unpaved_pct || 0) + '%</td>' +
                    '<td class="num">' + r.elevation_scale.toFixed(2) + '</td>';
                tbody.appendChild(row);
            });

            updateTotals(collectionRoutes);
        }

        function updateSingleRouteUnits() {
            var imperial = isImperial();
            var distEl = document.getElementById('singleDistance');
            var gainEl = document.getElementById('singleElevGain');
            var lossEl = document.getElementById('singleElevLoss');
            var speedEl = document.getElementById('singleSpeed');

            if (distEl) {
                var km = parseFloat(distEl.dataset.km);
                if (imperial) {
                    distEl.textContent = (km * 0.621371).toFixed(1) + ' mi';
                } else {
                    distEl.textContent = km.toFixed(1) + ' km';
                }
            }
            if (gainEl) {
                var m = parseFloat(gainEl.dataset.m);
                if (imperial) {
                    gainEl.textContent = Math.round(m * 3.28084) + ' ft';
                } else {
                    gainEl.textContent = Math.round(m) + ' m';
                }
            }
            if (lossEl) {
                var m = parseFloat(lossEl.dataset.m);
                if (imperial) {
                    lossEl.textContent = Math.round(m * 3.28084) + ' ft';
                } else {
                    lossEl.textContent = Math.round(m) + ' m';
                }
            }
            if (speedEl) {
                var kmh = parseFloat(speedEl.dataset.kmh);
                if (imperial) {
                    speedEl.textContent = (kmh * 0.621371).toFixed(1) + ' mph';
                } else {
                    speedEl.textContent = kmh.toFixed(1) + ' km/h';
                }
            }
            var hillyEl = document.getElementById('singleHilliness');
            if (hillyEl) {
                var mkm = parseFloat(hillyEl.dataset.mkm);
                if (imperial) {
                    // m/km to ft/mi: (3.28084 ft/m) / (0.621371 mi/km) ≈ 5.28
                    hillyEl.textContent = Math.round(mkm * 5.28) + ' ft/mi';
                } else {
                    hillyEl.textContent = Math.round(mkm) + ' m/km';
                }
            }
            // Steep distance stats (including comparison mode _2 variants)
            ['steepDist10', 'steepDist15', 'steepDist10_2', 'steepDist15_2'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el && el.dataset.m) {
                    var m = parseFloat(el.dataset.m);
                    if (imperial) {
                        el.textContent = (m * 0.000621371).toFixed(2) + ' mi';
                    } else {
                        el.textContent = (m / 1000).toFixed(2) + ' km';
                    }
                }
            });
        }

        function getEnergyUnit() {
            return document.getElementById('energyUnitSelect')?.value || 'kcal';
        }

        function formatEnergy(kj) {
            if (kj === null || kj === undefined || isNaN(kj)) return '-';
            var kcal = kj * 1.075;  // Convert work to food calories
            var unit = getEnergyUnit();
            if (unit === 'bananas') {
                return (kcal / 100).toFixed(1) + ' bananas';
            } else if (unit === 'baguettes') {
                return (kcal / 680).toFixed(2) + ' baguettes';
            }
            return Math.round(kcal) + ' kcal';
        }

        function formatEnergyDiff(kj1, kj2) {
            if (kj1 === null || kj2 === null || isNaN(kj1) || isNaN(kj2)) return '-';
            var kcal1 = kj1 * 1.075;
            var kcal2 = kj2 * 1.075;
            var diff = kcal1 - kcal2;
            var unit = getEnergyUnit();
            var sign = diff > 0 ? '+' : '';
            if (unit === 'bananas') {
                return sign + (diff / 100).toFixed(1) + ' bananas';
            } else if (unit === 'baguettes') {
                return sign + (diff / 680).toFixed(2) + ' baguettes';
            }
            return sign + Math.round(diff) + ' kcal';
        }

        function updateEnergyUnits() {
            // Single route
            var el = document.getElementById('singleEnergy');
            if (el && el.dataset.kj) {
                el.textContent = formatEnergy(parseFloat(el.dataset.kj));
            }
            // Comparison mode
            ['1', '2'].forEach(function(n) {
                var el = document.getElementById('energy' + n);
                if (el && el.dataset.kj) {
                    el.textContent = formatEnergy(parseFloat(el.dataset.kj));
                }
            });
            // Comparison diff
            var diffEl = document.getElementById('energyDiff');
            if (diffEl && diffEl.dataset.kj1 && diffEl.dataset.kj2) {
                diffEl.textContent = formatEnergyDiff(parseFloat(diffEl.dataset.kj1), parseFloat(diffEl.dataset.kj2));
            }
            // Save preference
            try { localStorage.setItem('energyUnit', getEnergyUnit()); } catch (e) {}
        }

        function initEnergyUnit() {
            try {
                var saved = localStorage.getItem('energyUnit');
                if (saved) {
                    var select = document.getElementById('energyUnitSelect');
                    if (select) {
                        select.value = saved;
                        updateEnergyUnits();
                    }
                }
            } catch (e) {}
        }

        function updateComparisonTableUnits() {
            var imperial = isImperial();
            // Distance columns
            ['cmpDist1', 'cmpDist2', 'cmpDistDiff'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el && el.dataset.km) {
                    var km = parseFloat(el.dataset.km);
                    var sign = (id.includes('Diff') && km > 0) ? '+' : '';
                    if (imperial) {
                        el.textContent = sign + (km * 0.621371).toFixed(1) + ' mi';
                    } else {
                        el.textContent = sign + km.toFixed(1) + ' km';
                    }
                }
            });
            // Elevation columns
            ['cmpElev1', 'cmpElev2', 'cmpElevDiff'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el && el.dataset.m) {
                    var m = parseFloat(el.dataset.m);
                    var sign = (id.includes('Diff') && m > 0) ? '+' : '';
                    if (imperial) {
                        el.textContent = sign + Math.round(m * 3.28084) + ' ft';
                    } else {
                        el.textContent = sign + Math.round(m) + ' m';
                    }
                }
            });
            // Speed columns
            ['cmpSpeed1', 'cmpSpeed2', 'cmpSpeedDiff'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el && el.dataset.kmh) {
                    var kmh = parseFloat(el.dataset.kmh);
                    var sign = (id.includes('Diff') && kmh > 0) ? '+' : '';
                    if (imperial) {
                        el.textContent = sign + (kmh * 0.621371).toFixed(1) + ' mph';
                    } else {
                        el.textContent = sign + kmh.toFixed(1) + ' km/h';
                    }
                }
            });
            // Hilliness columns
            ['cmpHilly1', 'cmpHilly2', 'cmpHillyDiff'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el && el.dataset.mkm) {
                    var mkm = parseFloat(el.dataset.mkm);
                    var sign = (id.includes('Diff') && mkm > 0) ? '+' : '';
                    if (imperial) {
                        el.textContent = sign + Math.round(mkm * 5.28) + ' ft/mi';
                    } else {
                        el.textContent = sign + Math.round(mkm) + ' m/km';
                    }
                }
            });
        }

        document.getElementById('imperial').addEventListener('change', function() {
            rerenderCollectionTable();
            updateSingleRouteUnits();
            updateComparisonTableUnits();
            // Refresh elevation profiles if speed overlay is active (axis labels change units)
            var speedCheckbox = document.getElementById('overlay_speed');
            if (speedCheckbox && speedCheckbox.checked) {
                toggleOverlay('speed');
            }
        });

        // Auto-trigger collection analysis when loaded via shared link
        (function() {
            var mode = '{{ mode }}';
            var url = document.getElementById('url').value;
            if (mode === 'collection' && url && url.includes('/collections/')) {
                // Small delay to ensure page is fully loaded
                setTimeout(function() {
                    document.getElementById('analyzeForm').dispatchEvent(new Event('submit'));
                }, 100);
            }
        })();

        // Histogram bar tooltips (touch-friendly)
        (function() {
            var tooltip = null;

            function formatTime(seconds) {
                var hours = Math.floor(seconds / 3600);
                var mins = Math.floor((seconds % 3600) / 60);
                if (hours > 0) {
                    return hours + 'h ' + mins + 'm';
                }
                return mins + 'm';
            }

            function formatDistance(meters) {
                var imperial = isImperial();
                if (imperial) {
                    var miles = meters * 0.000621371;
                    return miles.toFixed(2) + ' mi';
                } else {
                    var km = meters / 1000;
                    return km.toFixed(2) + ' km';
                }
            }

            function showTooltip(bar, x, y) {
                if (!tooltip) {
                    tooltip = document.createElement('div');
                    tooltip.className = 'bar-tooltip';
                    document.body.appendChild(tooltip);
                }

                var name = bar.dataset.name || 'Route';
                var pct = parseFloat(bar.dataset.pct) || 0;
                var type = bar.dataset.type;
                var absValue = '';

                if (type === 'time') {
                    var seconds = parseFloat(bar.dataset.seconds) || 0;
                    absValue = formatTime(seconds);
                } else if (type === 'distance') {
                    var meters = parseFloat(bar.dataset.meters) || 0;
                    absValue = formatDistance(meters);
                }

                tooltip.innerHTML = '<div class="tooltip-title">' + name + '</div>' +
                    '<div class="tooltip-row"><span class="tooltip-label">Percent:</span><span>' + pct.toFixed(1) + '%</span></div>' +
                    '<div class="tooltip-row"><span class="tooltip-label">' + (type === 'time' ? 'Time:' : 'Distance:') + '</span><span>' + absValue + '</span></div>';

                tooltip.style.display = 'block';

                // Position tooltip - use bar's position for pinch-zoom compatibility
                var barRect = bar.getBoundingClientRect();
                var tooltipRect = tooltip.getBoundingClientRect();

                // Account for pinch-zoom using visualViewport if available
                var vv = window.visualViewport;
                var offsetX = vv ? vv.offsetLeft : 0;
                var offsetY = vv ? vv.offsetTop : 0;
                var scale = vv ? vv.scale : 1;

                // Calculate position relative to bar, adjusted for zoom
                var left = barRect.left + barRect.width / 2 - tooltipRect.width / 2 + offsetX;
                var top = barRect.top - tooltipRect.height - 10 + offsetY;

                // Get visible viewport bounds
                var viewWidth = vv ? vv.width : window.innerWidth;
                var viewHeight = vv ? vv.height : window.innerHeight;
                var viewLeft = offsetX;
                var viewTop = offsetY;

                // Keep tooltip within visible viewport
                if (left < viewLeft + 5) left = viewLeft + 5;
                if (left + tooltipRect.width > viewLeft + viewWidth - 5) {
                    left = viewLeft + viewWidth - tooltipRect.width - 5;
                }
                if (top < viewTop + 5) {
                    top = barRect.bottom + 10 + offsetY;  // Show below bar instead
                }

                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }

            function hideTooltip() {
                if (tooltip) {
                    tooltip.style.display = 'none';
                }
            }

            // Find nearest bar in a container given touch X position
            function findNearestBar(container, touchX) {
                var bars = container.querySelectorAll('.bar-hover');
                if (bars.length === 0) return null;
                if (bars.length === 1) return bars[0];

                // Find which bar is closest to touch point
                var containerRect = container.getBoundingClientRect();
                var relativeX = touchX - containerRect.left;
                var containerWidth = containerRect.width;

                // In comparison mode, left half = first bar, right half = second bar
                if (relativeX < containerWidth / 2) {
                    return bars[0];
                } else {
                    return bars[1] || bars[0];
                }
            }

            // Event delegation for all bar-hover elements
            document.addEventListener('mouseover', function(e) {
                if (e.target.classList.contains('bar-hover')) {
                    var rect = e.target.getBoundingClientRect();
                    showTooltip(e.target, rect.left + rect.width / 2, rect.top);
                }
            });

            document.addEventListener('mouseout', function(e) {
                if (e.target.classList.contains('bar-hover')) {
                    hideTooltip();
                }
            });

            // Touch support - enhanced for better mobile UX
            document.addEventListener('touchstart', function(e) {
                var target = e.target;
                var touch = e.touches[0];

                // Direct touch on bar
                if (target.classList.contains('bar-hover')) {
                    e.preventDefault();
                    showTooltip(target, touch.clientX, touch.clientY - 50);
                    return;
                }

                // Touch on bar-container - find nearest bar
                var container = target.closest('.bar-container');
                if (container) {
                    var bar = findNearestBar(container, touch.clientX);
                    if (bar) {
                        e.preventDefault();
                        showTooltip(bar, touch.clientX, touch.clientY - 50);
                        return;
                    }
                }

                // Touch on histogram-bar (includes label area) - find bar in that group
                var histogramBar = target.closest('.histogram-bar');
                if (histogramBar) {
                    var barContainer = histogramBar.querySelector('.bar-container');
                    if (barContainer) {
                        var bar = findNearestBar(barContainer, touch.clientX);
                        if (bar) {
                            e.preventDefault();
                            showTooltip(bar, touch.clientX, touch.clientY - 50);
                            return;
                        }
                    }
                }

                hideTooltip();
            }, { passive: false });

            document.addEventListener('touchend', function(e) {
                // Hide tooltip immediately when finger is lifted
                hideTooltip();
            }, { passive: true });
        })();

    </script>

    {% if error %}
    <div class="error server-error">{{ error }}</div>
    {% endif %}

    {% if result %}
    <div class="results {% if is_trip %}trip-results{% else %}route-results{% endif %}" id="singleRouteResults">
        {% if share_url %}
        <input type="hidden" id="shareUrl" value="{{ share_url }}">
        {% endif %}
        <div class="results-header">
            {% if compare_mode and result2 %}
            <h2>Comparison</h2>
            {% else %}
            <h2>{{ result.name or ('Trip Analysis' if is_trip else 'Route Analysis') }}</h2>
            {% endif %}
            {% if not compare_mode %}
            <span class="result-badge {% if is_trip %}trip-badge{% else %}route-badge{% endif %}">
                {% if is_trip %}Recorded Ride{% else %}Planned Route{% endif %}
            </span>
            {% endif %}
            {% if share_url %}
            <button type="button" class="share-btn" onclick="copyShareLink('shareUrl', this)" title="Copy link to share">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z"/></svg>
                <span>Share</span>
            </button>
            {% endif %}
        </div>

        {% if compare_mode and result2 %}
        <!-- Comparison table -->
        <div class="comparison-table-wrapper">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th class="route-col">{{ (result.name or ('Trip 1' if is_trip else 'Route 1'))|truncate(25) }} <span class="result-badge {% if is_trip %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip %}Ride{% else %}Route{% endif %}</span></th>
                        <th class="route-col">{{ (result2.name or ('Trip 2' if is_trip2 else 'Route 2'))|truncate(25) }} <span class="result-badge {% if is_trip2 %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip2 %}Ride{% else %}Route{% endif %}</span></th>
                        <th class="diff-col">Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="primary">
                        <td>{% if is_trip or is_trip2 %}Moving Time{% else %}Est. Moving Time{% endif %}</td>
                        <td class="route-col">{{ result.time_str }}</td>
                        <td class="route-col">{{ result2.time_str }}</td>
                        <td class="diff-col">{{ format_time_diff(result.time_seconds, result2.time_seconds) }}</td>
                    </tr>
                    {% if (not is_trip or result.work_kj is not none) and (not is_trip2 or result2.work_kj is not none) %}
                    <tr class="primary">
                        <td>{% if is_trip or is_trip2 %}Work{% else %}Est. Work{% endif %}</td>
                        <td class="route-col">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj) }} kJ{% else %}-{% endif %}</td>
                        <td class="route-col">{% if result2.work_kj is not none %}{{ "%.0f"|format(result2.work_kj) }} kJ{% else %}-{% endif %}</td>
                        <td class="diff-col">{% if result.work_kj is not none and result2.work_kj is not none %}{{ format_diff(result.work_kj, result2.work_kj, 'kJ') }}{% else %}-{% endif %}</td>
                    </tr>
                    {% endif %}
                    {% if result.work_kj is not none or result2.work_kj is not none %}
                    <tr class="primary">
                        <td>{% if is_trip or is_trip2 %}Energy{% else %}Est. Energy{% endif %} <select id="energyUnitSelect" class="unit-select" onchange="updateEnergyUnits()"><option value="kcal">kcal</option><option value="bananas">Bananas</option><option value="baguettes">Baguettes</option></select></td>
                        <td class="route-col"><span id="energy1" data-kj="{{ result.work_kj if result.work_kj is not none else '' }}">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj * 1.075) }} kcal{% else %}-{% endif %}</span></td>
                        <td class="route-col"><span id="energy2" data-kj="{{ result2.work_kj if result2.work_kj is not none else '' }}">{% if result2.work_kj is not none %}{{ "%.0f"|format(result2.work_kj * 1.075) }} kcal{% else %}-{% endif %}</span></td>
                        <td class="diff-col"><span id="energyDiff" data-kj1="{{ result.work_kj if result.work_kj is not none else '' }}" data-kj2="{{ result2.work_kj if result2.work_kj is not none else '' }}">{% if result.work_kj is not none and result2.work_kj is not none %}{{ format_diff(result.work_kj * 1.075, result2.work_kj * 1.075, 'kcal') }}{% else %}-{% endif %}</span></td>
                    </tr>
                    {% endif %}
                    <tr class="primary">
                        <td>{% if is_trip or is_trip2 %}Avg Power{% else %}Est. Avg Power{% endif %}</td>
                        <td class="route-col">{% if result.avg_watts is not none %}{{ result.avg_watts|int }} W{% else %}-{% endif %}</td>
                        <td class="route-col">{% if result2.avg_watts is not none %}{{ result2.avg_watts|int }} W{% else %}-{% endif %}</td>
                        <td class="diff-col">{% if result.avg_watts is not none and result2.avg_watts is not none %}{{ format_diff(result.avg_watts, result2.avg_watts, 'W') }}{% else %}-{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Distance</td>
                        <td class="route-col" id="cmpDist1" data-km="{{ result.distance_km }}">{{ "%.1f"|format(result.distance_km) }} km</td>
                        <td class="route-col" id="cmpDist2" data-km="{{ result2.distance_km }}">{{ "%.1f"|format(result2.distance_km) }} km</td>
                        <td class="diff-col" id="cmpDistDiff" data-km="{{ result.distance_km - result2.distance_km }}">{{ "%+.1f"|format(result.distance_km - result2.distance_km) }} km</td>
                    </tr>
                    <tr>
                        <td>Elevation Gain</td>
                        <td class="route-col" id="cmpElev1" data-m="{{ result.elevation_m }}">{{ "%.0f"|format(result.elevation_m) }} m</td>
                        <td class="route-col" id="cmpElev2" data-m="{{ result2.elevation_m }}">{{ "%.0f"|format(result2.elevation_m) }} m</td>
                        <td class="diff-col" id="cmpElevDiff" data-m="{{ result.elevation_m - result2.elevation_m }}">{{ "%+.0f"|format(result.elevation_m - result2.elevation_m) }} m</td>
                    </tr>
                    <tr>
                        <td>Avg Speed</td>
                        <td class="route-col" id="cmpSpeed1" data-kmh="{{ result.avg_speed_kmh }}">{{ "%.1f"|format(result.avg_speed_kmh) }} km/h</td>
                        <td class="route-col" id="cmpSpeed2" data-kmh="{{ result2.avg_speed_kmh }}">{{ "%.1f"|format(result2.avg_speed_kmh) }} km/h</td>
                        <td class="diff-col" id="cmpSpeedDiff" data-kmh="{{ result.avg_speed_kmh - result2.avg_speed_kmh }}">{{ "%+.1f"|format(result.avg_speed_kmh - result2.avg_speed_kmh) }} km/h</td>
                    </tr>
                    <tr>
                        <td><span class="label-with-info">Hilliness <button type="button" class="info-btn" onclick="showModal('hillyModal')">?</button></span></td>
                        <td class="route-col" id="cmpHilly1" data-mkm="{{ result.hilliness_score }}">{{ "%.0f"|format(result.hilliness_score) }} m/km</td>
                        <td class="route-col" id="cmpHilly2" data-mkm="{{ result2.hilliness_score }}">{{ "%.0f"|format(result2.hilliness_score) }} m/km</td>
                        <td class="diff-col" id="cmpHillyDiff" data-mkm="{{ result.hilliness_score - result2.hilliness_score }}">{{ "%+.0f"|format(result.hilliness_score - result2.hilliness_score) }} m/km</td>
                    </tr>
                    <tr>
                        <td><span class="label-with-info">Steepness <button type="button" class="info-btn" onclick="showModal('steepModal')">?</button></span></td>
                        <td class="route-col">{{ "%.1f"|format(result.steepness_score) }}%</td>
                        <td class="route-col">{{ "%.1f"|format(result2.steepness_score) }}%</td>
                        <td class="diff-col">{{ format_pct_diff(result.steepness_score, result2.steepness_score) }}</td>
                    </tr>
                    <tr>
                        <td><span class="label-with-info">&gt;10% <button type="button" class="info-btn" onclick="showModal('steepTimeModal')">?</button></span></td>
                        <td class="route-col">{% if result.steep_time_seconds and result.steep_time_seconds >= 60 %}{% set mins = (result.steep_time_seconds / 60)|round|int %}{% if mins < 60 %}{{ mins }}m{% else %}{{ mins // 60 }}h {{ "%02d"|format(mins % 60) }}m{% endif %}{% else %}-{% endif %}</td>
                        <td class="route-col">{% if result2.steep_time_seconds and result2.steep_time_seconds >= 60 %}{% set mins2 = (result2.steep_time_seconds / 60)|round|int %}{% if mins2 < 60 %}{{ mins2 }}m{% else %}{{ mins2 // 60 }}h {{ "%02d"|format(mins2 % 60) }}m{% endif %}{% else %}-{% endif %}</td>
                        <td class="diff-col">{% if result.steep_time_seconds and result2.steep_time_seconds %}{{ format_pct_diff(result.steep_time_seconds, result2.steep_time_seconds) }}{% else %}-{% endif %}</td>
                    </tr>
                    {% if result.unpaved_pct is not none and result2.unpaved_pct is not none %}
                    <tr>
                        <td>Unpaved</td>
                        <td class="route-col">{{ "%.0f"|format(result.unpaved_pct) }}%</td>
                        <td class="route-col">{{ "%.0f"|format(result2.unpaved_pct) }}%</td>
                        <td class="diff-col">{{ format_pct_diff(result.unpaved_pct, result2.unpaved_pct) }}</td>
                    </tr>
                    {% endif %}
                </tbody>
            </table>
        </div>
        {% if not is_trip or not is_trip2 %}
        <div class="comparison-footnote">* Estimated from physics model</div>
        {% endif %}
        {% else %}
        <!-- Single route/trip results -->
        <div class="primary-results">
            <div class="result-row primary">
                <span class="result-label label-with-info">{% if is_trip %}Moving Time{% else %}Est. Moving Time{% endif %} <button type="button" class="info-btn" onclick="showModal('timeModal')">?</button></span>
                <span class="result-value">{{ result.time_str }}</span>
            </div>
            {% if not is_trip or result.work_kj is not none %}
            <div class="result-row primary">
                <span class="result-label label-with-info">{% if is_trip %}Work{% else %}Est. Work{% endif %} <button type="button" class="info-btn" onclick="showModal('workModal')">?</button></span>
                <span class="result-value">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj) }} kJ{% else %}-{% endif %}</span>
            </div>
            <div class="result-row primary">
                <span class="result-label label-with-info">{% if is_trip %}Energy{% else %}Est. Energy{% endif %} <button type="button" class="info-btn" onclick="showModal('energyModal')">?</button>
                    <select id="energyUnitSelect" class="unit-select" onchange="updateEnergyUnits()">
                        <option value="kcal">kcal</option>
                        <option value="bananas">Bananas</option>
                        <option value="baguettes">Baguettes</option>
                    </select>
                </span>
                <span class="result-value" id="singleEnergy" data-kj="{{ result.work_kj if result.work_kj is not none else '' }}">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj * 1.075) }} kcal{% else %}-{% endif %}</span>
            </div>
            {% endif %}
            {% if result.has_power and result.avg_watts is not none %}
            <div class="result-row primary">
                <span class="result-label">{% if is_trip %}Avg Power{% else %}Est. Avg Power{% endif %}</span>
                <span class="result-value">{{ result.avg_watts|int }} W</span>
            </div>
            {% endif %}
        </div>

        <div class="result-row">
            <span class="result-label">Distance</span>
            <span class="result-value" id="singleDistance" data-km="{{ result.distance_km }}">{{ "%.1f"|format(result.distance_km) }} km</span>
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Gain</span>
            <span class="result-value" id="singleElevGain" data-m="{{ result.elevation_m }}">{{ "%.0f"|format(result.elevation_m) }} m</span>
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Loss</span>
            <span class="result-value" id="singleElevLoss" data-m="{{ result.elevation_loss_m }}">{{ "%.0f"|format(result.elevation_loss_m) }} m</span>
        </div>
        <div class="result-row">
            <span class="result-label">Avg Speed</span>
            <span class="result-value" id="singleSpeed" data-kmh="{{ result.avg_speed_kmh }}">{{ "%.1f"|format(result.avg_speed_kmh) }} km/h</span>
        </div>
        {% if result.unpaved_pct is not none %}
        <div class="result-row">
            <span class="result-label">Surface</span>
            <span class="result-value">{{ "%.0f"|format(result.unpaved_pct) }}% unpaved</span>
        </div>
        {% endif %}
        <div class="result-row">
            <span class="result-label label-with-info">Hilliness <button type="button" class="info-btn" onclick="showModal('hillyModal')">?</button></span>
            <span class="result-value" id="singleHilliness" data-mkm="{{ result.hilliness_score }}">{{ "%.0f"|format(result.hilliness_score) }} m/km</span>
        </div>
        <div class="result-row">
            <span class="result-label label-with-info">Steepness <button type="button" class="info-btn" onclick="showModal('steepModal')">?</button></span>
            <span class="result-value">{{ "%.1f"|format(result.steepness_score) }}%</span>
        </div>
        <div class="result-row">
            <span class="result-label label-with-info">&gt;10% <button type="button" class="info-btn" onclick="showModal('steepTimeModal')">?</button></span>
            <span class="result-value">{% if result.steep_time_seconds and result.steep_time_seconds >= 60 %}{% set mins = (result.steep_time_seconds / 60)|round|int %}{% if mins < 60 %}{{ mins }}m{% else %}{{ mins // 60 }}h {{ "%02d"|format(mins % 60) }}m{% endif %}{% else %}-{% endif %}</span>
        </div>
        {% endif %}

        {% if result.grade_histogram %}
        {% set labels = ['<-10', '-10', '-8', '-6', '-4', '-2', '0', '+2', '+4', '+6', '+8', '>10'] %}
        {% set bar_colors = ['#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6', '#cccccc', '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'] %}
        <div class="histograms-container">
            <div class="grade-histogram">
                <h4>Time at Grade{% if compare_mode and result2 %} (Comparison){% endif %}</h4>
                <div class="histogram-bars{% if compare_mode and result2 %} comparison-mode{% endif %}">
                    {% set total_time = result.grade_histogram.values() | sum %}
                    {% set max_seconds = result.grade_histogram.values() | max %}
                    {% if compare_mode and result2 %}
                        {% set total_time_2 = result2.grade_histogram.values() | sum %}
                        {% set max_seconds_2 = result2.grade_histogram.values() | max %}
                        {% set max_seconds_both = [max_seconds, max_seconds_2] | max %}
                    {% endif %}
                    {% for label in result.grade_histogram.keys() %}
                    {% set seconds = result.grade_histogram[label] %}
                    {% set pct = (seconds / total_time * 100) if total_time > 0 else 0 %}
                    {% if compare_mode and result2 %}
                        {% set bar_height = (seconds / max_seconds_both * 100) if max_seconds_both > 0 else 0 %}
                    {% else %}
                        {% set bar_height = (seconds / max_seconds * 100) if max_seconds > 0 else 0 %}
                    {% endif %}
                    <div class="histogram-bar">
                        <div class="bar-container">
                            <div class="bar bar-hover" style="height: {{ bar_height }}%; background: {{ bar_colors[loop.index0] }};" data-name="{{ (result.name or 'Route 1')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct) }}" data-seconds="{{ seconds }}" data-type="time"></div>
                            {% if compare_mode and result2 %}
                            {% set seconds_2 = result2.grade_histogram[label] %}
                            {% set pct_2 = (seconds_2 / total_time_2 * 100) if total_time_2 > 0 else 0 %}
                            {% set bar_height_2 = (seconds_2 / max_seconds_both * 100) if max_seconds_both > 0 else 0 %}
                            <div class="bar route2 bar-hover" style="height: {{ bar_height_2 }}%; background: {{ bar_colors[loop.index0] }};" data-name="{{ (result2.name or 'Route 2')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct_2) }}" data-seconds="{{ seconds_2 }}" data-type="time"></div>
                            {% endif %}
                        </div>
                        <span class="label">{{ labels[loop.index0] }}</span>
                        {% if not (compare_mode and result2) %}
                        <span class="pct">{% if pct >= 1 %}{{ "%.0f"|format(pct) }}%{% else %}&nbsp;{% endif %}</span>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% if compare_mode and result2 %}
                <div class="histogram-legend">
                    <span class="legend-item"><span class="legend-color" style="background-color: #ff6600;"></span>{{ (result.name or 'Route 1')|truncate(20) }}</span>
                    <span class="legend-item"><span class="legend-color striped" style="background-color: #ff6600;"></span>{{ (result2.name or 'Route 2')|truncate(20) }}</span>
                </div>
                {% endif %}
            </div>
            <div class="grade-histogram">
                <h4>Distance at Grade{% if compare_mode and result2 %} (Comparison){% endif %}</h4>
                <div class="histogram-bars{% if compare_mode and result2 %} comparison-mode{% endif %}">
                    {% set total_dist = result.grade_distance_histogram.values() | sum %}
                    {% set max_dist = result.grade_distance_histogram.values() | max %}
                    {% if compare_mode and result2 %}
                        {% set total_dist_2 = result2.grade_distance_histogram.values() | sum %}
                        {% set max_dist_2 = result2.grade_distance_histogram.values() | max %}
                        {% set max_dist_both = [max_dist, max_dist_2] | max %}
                    {% endif %}
                    {% for label in result.grade_distance_histogram.keys() %}
                    {% set meters = result.grade_distance_histogram[label] %}
                    {% set pct = (meters / total_dist * 100) if total_dist > 0 else 0 %}
                    {% if compare_mode and result2 %}
                        {% set bar_height = (meters / max_dist_both * 100) if max_dist_both > 0 else 0 %}
                    {% else %}
                        {% set bar_height = (meters / max_dist * 100) if max_dist > 0 else 0 %}
                    {% endif %}
                    <div class="histogram-bar">
                        <div class="bar-container">
                            <div class="bar bar-hover" style="height: {{ bar_height }}%; background: {{ bar_colors[loop.index0] }};" data-name="{{ (result.name or 'Route 1')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct) }}" data-meters="{{ meters }}" data-type="distance"></div>
                            {% if compare_mode and result2 %}
                            {% set meters_2 = result2.grade_distance_histogram[label] %}
                            {% set pct_2 = (meters_2 / total_dist_2 * 100) if total_dist_2 > 0 else 0 %}
                            {% set bar_height_2 = (meters_2 / max_dist_both * 100) if max_dist_both > 0 else 0 %}
                            <div class="bar route2 bar-hover" style="height: {{ bar_height_2 }}%; background: {{ bar_colors[loop.index0] }};" data-name="{{ (result2.name or 'Route 2')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct_2) }}" data-meters="{{ meters_2 }}" data-type="distance"></div>
                            {% endif %}
                        </div>
                        <span class="label">{{ labels[loop.index0] }}</span>
                        {% if not (compare_mode and result2) %}
                        <span class="pct">{% if pct >= 1 %}{{ "%.0f"|format(pct) }}%{% else %}&nbsp;{% endif %}</span>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% if compare_mode and result2 %}
                <div class="histogram-legend">
                    <span class="legend-item"><span class="legend-color" style="background-color: #ff6600;"></span>{{ (result.name or 'Route 1')|truncate(20) }}</span>
                    <span class="legend-item"><span class="legend-color striped" style="background-color: #ff6600;"></span>{{ (result2.name or 'Route 2')|truncate(20) }}</span>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        {% if result.max_grade >= 10 or (compare_mode and result2 and result2.max_grade >= 10) %}
        <div class="steep-section">
            <h4><span class="th-with-info">Steep Climbs <button type="button" class="info-btn" onclick="showModal('steepClimbsModal')">?</button></span></h4>
            {% if compare_mode and result2 %}
            <table class="comparison-table steep-comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>{{ (result.name or ('Trip 1' if is_trip else 'Route 1')) }} <span class="result-badge {% if is_trip %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip %}Ride{% else %}Route{% endif %}</span></th>
                        <th>{{ (result2.name or ('Trip 2' if is_trip2 else 'Route 2')) }} <span class="result-badge {% if is_trip2 %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip2 %}Ride{% else %}Route{% endif %}</span></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Max Grade</td>
                        <td>{{ "%.1f"|format(result.max_grade) }}%</td>
                        <td>{{ "%.1f"|format(result2.max_grade) }}%</td>
                    </tr>
                    <tr>
                        <td>Distance ≥10%</td>
                        <td id="steepDist10" data-m="{{ result.steep_distance }}">{{ "%.2f"|format(result.steep_distance / 1000) }} km</td>
                        <td id="steepDist10_2" data-m="{{ result2.steep_distance }}">{{ "%.2f"|format(result2.steep_distance / 1000) }} km</td>
                    </tr>
                    <tr>
                        <td>Distance ≥15%</td>
                        <td id="steepDist15" data-m="{{ result.very_steep_distance }}">{{ "%.2f"|format(result.very_steep_distance / 1000) }} km</td>
                        <td id="steepDist15_2" data-m="{{ result2.very_steep_distance }}">{{ "%.2f"|format(result2.very_steep_distance / 1000) }} km</td>
                    </tr>
                </tbody>
            </table>
            {% else %}
            <div class="steep-stats">
                <div class="steep-stat">
                    <span class="steep-label">Max Grade</span>
                    <span class="steep-value">{{ "%.1f"|format(result.max_grade) }}%</span>
                </div>
                <div class="steep-stat">
                    <span class="steep-label">Distance ≥10%</span>
                    <span class="steep-value" id="steepDist10" data-m="{{ result.steep_distance }}">{{ "%.2f"|format(result.steep_distance / 1000) }} km</span>
                </div>
                <div class="steep-stat">
                    <span class="steep-label">Distance ≥15%</span>
                    <span class="steep-value" id="steepDist15" data-m="{{ result.very_steep_distance }}">{{ "%.2f"|format(result.very_steep_distance / 1000) }} km</span>
                </div>
            </div>
            {% endif %}
            {% set steep_labels = ['10-12', '12-14', '14-16', '16-18', '18-20', '>20'] %}
            {% set steep_colors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000'] %}
            <div class="histograms-container">
                <div class="grade-histogram">
                    <h4>Time at Steep Grade{% if compare_mode and result2 %} (Comparison){% endif %}</h4>
                    <div class="histogram-bars{% if compare_mode and result2 %} comparison-mode{% endif %}">
                        {% set max_steep_time = result.steep_time_histogram.values() | max %}
                        {% if compare_mode and result2 %}
                            {% set max_steep_time_2 = result2.steep_time_histogram.values() | max %}
                            {% set max_steep_time_both = [max_steep_time, max_steep_time_2] | max %}
                        {% endif %}
                        {% for label in result.steep_time_histogram.keys() %}
                        {% set seconds = result.steep_time_histogram[label] %}
                        {% set pct = (seconds / result.hilliness_total_time * 100) if result.hilliness_total_time > 0 else 0 %}
                        {% if compare_mode and result2 %}
                            {% set bar_height = (seconds / max_steep_time_both * 100) if max_steep_time_both > 0 else 0 %}
                        {% else %}
                            {% set bar_height = (seconds / max_steep_time * 100) if max_steep_time > 0 else 0 %}
                        {% endif %}
                        <div class="histogram-bar">
                            <div class="bar-container">
                                <div class="bar bar-hover" style="height: {{ bar_height }}%; background: {{ steep_colors[loop.index0] }};" data-name="{{ (result.name or 'Route 1')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct) }}" data-seconds="{{ seconds }}" data-type="time"></div>
                                {% if compare_mode and result2 %}
                                {% set seconds_2 = result2.steep_time_histogram[label] %}
                                {% set pct_2 = (seconds_2 / result2.hilliness_total_time * 100) if result2.hilliness_total_time > 0 else 0 %}
                                {% set bar_height_2 = (seconds_2 / max_steep_time_both * 100) if max_steep_time_both > 0 else 0 %}
                                <div class="bar route2 bar-hover" style="height: {{ bar_height_2 }}%; background: {{ steep_colors[loop.index0] }};" data-name="{{ (result2.name or 'Route 2')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct_2) }}" data-seconds="{{ seconds_2 }}" data-type="time"></div>
                                {% endif %}
                            </div>
                            <span class="label">{{ steep_labels[loop.index0] }}</span>
                            {% if not (compare_mode and result2) %}
                            <span class="pct">{% if pct >= 0.5 %}{{ "%.0f"|format(pct) }}%{% elif seconds > 0 %}<1%{% else %}&nbsp;{% endif %}</span>
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                    {% if compare_mode and result2 %}
                    <div class="histogram-legend">
                        <span class="legend-item"><span class="legend-color" style="background-color: #cc4400;"></span>{{ (result.name or 'Route 1')|truncate(20) }}</span>
                        <span class="legend-item"><span class="legend-color striped" style="background-color: #cc4400;"></span>{{ (result2.name or 'Route 2')|truncate(20) }}</span>
                    </div>
                    {% endif %}
                </div>
                <div class="grade-histogram">
                    <h4>Distance at Steep Grade{% if compare_mode and result2 %} (Comparison){% endif %}</h4>
                    <div class="histogram-bars{% if compare_mode and result2 %} comparison-mode{% endif %}">
                        {% set max_steep_dist = result.steep_distance_histogram.values() | max %}
                        {% if compare_mode and result2 %}
                            {% set max_steep_dist_2 = result2.steep_distance_histogram.values() | max %}
                            {% set max_steep_dist_both = [max_steep_dist, max_steep_dist_2] | max %}
                        {% endif %}
                        {% for label in result.steep_distance_histogram.keys() %}
                        {% set meters = result.steep_distance_histogram[label] %}
                        {% set pct = (meters / result.hilliness_total_distance * 100) if result.hilliness_total_distance > 0 else 0 %}
                        {% if compare_mode and result2 %}
                            {% set bar_height = (meters / max_steep_dist_both * 100) if max_steep_dist_both > 0 else 0 %}
                        {% else %}
                            {% set bar_height = (meters / max_steep_dist * 100) if max_steep_dist > 0 else 0 %}
                        {% endif %}
                        <div class="histogram-bar">
                            <div class="bar-container">
                                <div class="bar bar-hover" style="height: {{ bar_height }}%; background: {{ steep_colors[loop.index0] }};" data-name="{{ (result.name or 'Route 1')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct) }}" data-meters="{{ meters }}" data-type="distance"></div>
                                {% if compare_mode and result2 %}
                                {% set meters_2 = result2.steep_distance_histogram[label] %}
                                {% set pct_2 = (meters_2 / result2.hilliness_total_distance * 100) if result2.hilliness_total_distance > 0 else 0 %}
                                {% set bar_height_2 = (meters_2 / max_steep_dist_both * 100) if max_steep_dist_both > 0 else 0 %}
                                <div class="bar route2 bar-hover" style="height: {{ bar_height_2 }}%; background: {{ steep_colors[loop.index0] }};" data-name="{{ (result2.name or 'Route 2')|truncate(25) }}" data-pct="{{ '%.1f'|format(pct_2) }}" data-meters="{{ meters_2 }}" data-type="distance"></div>
                                {% endif %}
                            </div>
                            <span class="label">{{ steep_labels[loop.index0] }}</span>
                            {% if not (compare_mode and result2) %}
                            <span class="pct">{% if pct >= 0.5 %}{{ "%.0f"|format(pct) }}%{% elif meters > 0 %}<1%{% else %}&nbsp;{% endif %}</span>
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                    {% if compare_mode and result2 %}
                    <div class="histogram-legend">
                        <span class="legend-item"><span class="legend-color" style="background-color: #cc4400;"></span>{{ (result.name or 'Route 1')|truncate(20) }}</span>
                        <span class="legend-item"><span class="legend-color striped" style="background-color: #cc4400;"></span>{{ (result2.name or 'Route 2')|truncate(20) }}</span>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endif %}

        {% if result.elevation_scaled %}
        <div class="note">
            Elevation scaled {{ "%.2f"|format(result.elevation_scale) }}x to match RideWithGPS API data.
        </div>
        {% endif %}

        {% if result.noise_ratio and result.noise_ratio > 1.2 %}
        <div class="note noise-note">
            <span class="noise-badge" title="Ratio of raw GPS elevation gain to DEM elevation gain. Higher values indicate noisier GPS data.">
                Elevation noise: {{ "%.1f"|format(result.noise_ratio) }}x
                <button type="button" class="info-btn" onclick="showModal('noiseModal')" aria-label="Info about elevation noise">?</button>
            </span>
        </div>
        {% endif %}

        {% if result.tunnels_corrected > 0 %}
        <div class="note anomaly-note">
            <strong>{% if compare_mode %}{{ (result.name or 'Route 1')|truncate(20) }}: {% endif %}{{ result.tunnels_corrected }} anomal{{ 'ies' if result.tunnels_corrected > 1 else 'y' }} detected and corrected:</strong>
            {% for t in result.tunnel_corrections %}
            <span class="anomaly-item">{{ "%.1f"|format(t.start_km) }}-{{ "%.1f"|format(t.end_km) }} km ({{ "%.0f"|format(t.artificial_gain) }}m artificial gain removed)</span>{% if not loop.last %}, {% endif %}
            {% endfor %}
        </div>
        {% endif %}

        {% if compare_mode and result2 %}
        <!-- Stacked elevation profiles for comparison -->
        {% set t1_moving = result.time_seconds / 3600 %}
        {% set t1_elapsed = (result.elapsed_time_seconds | default(result.time_seconds)) / 3600 %}
        {% set t2_moving = result2.time_seconds / 3600 %}
        {% set t2_elapsed = (result2.elapsed_time_seconds | default(result2.time_seconds)) / 3600 %}
        {% set max_time_hours = [t1_elapsed, t2_elapsed] | max %}
        <div class="elevation-profiles-stacked">
            <div class="elevation-profile-header" style="margin-bottom: 8px;">
                <div class="elevation-profile-toggles">
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="overlay_speed" onchange="toggleOverlay('speed')">
                        <label for="overlay_speed">Speed</label>
                    </div>
                    {% if not is_trip or not is_trip2 %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="overlay_gravel" onchange="toggleOverlay('gravel')">
                        <label for="overlay_gravel">Unpaved</label>
                    </div>
                    {% endif %}
                </div>
            </div>
            <div class="elevation-profile">
                <div class="elevation-profile-header">
                    <h4><a href="{{ url }}" target="_blank">{{ (result.name or ('Trip 1' if is_trip else 'Route 1'))|truncate(40) }}</a> <span class="result-badge {% if is_trip %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip %}Ride{% else %}Route{% endif %}</span> - {{ "%.1f"|format(result.time_seconds / 3600) }}h</h4>
                    {% if is_trip %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="collapseStops1" onchange="toggleCollapseStops(1)">
                        <label for="collapseStops1">Moving time</label>
                    </div>
                    {% endif %}
                </div>
                <div class="elevation-profile-container" id="elevationContainer1"
                     data-url="{{ url|urlencode }}"
                     data-moving-time-hours="{{ '%.4f'|format(t1_moving) }}"
                     data-elapsed-time-hours="{{ '%.4f'|format(t1_elapsed) }}"
                     data-max-xlim-hours="{{ '%.4f'|format(max_time_hours) }}"
                     {% if compare_ylim %}data-max-ylim="{{ '%.2f'|format(compare_ylim) }}"{% endif %}
                     {% if compare_speed_ylim %}data-max-speed-ylim="{{ '%.2f'|format(compare_speed_ylim) }}"{% endif %}
                     data-base-profile-url="/elevation-profile?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}{% if compare_ylim %}&max_ylim={{ '%.2f'|format(compare_ylim) }}{% endif %}{% if compare_speed_ylim %}&max_speed_ylim={{ '%.2f'|format(compare_speed_ylim) }}{% endif %}"
                     data-base-data-url="/elevation-profile-data?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}">
                    <div class="elevation-loading" id="elevationLoading1">
                        <div class="elevation-spinner"></div>
                    </div>
                    <img src="/elevation-profile?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}&max_xlim_hours={{ '%.4f'|format(max_time_hours) }}{% if compare_ylim %}&max_ylim={{ '%.2f'|format(compare_ylim) }}{% endif %}{% if compare_speed_ylim %}&max_speed_ylim={{ '%.2f'|format(compare_speed_ylim) }}{% endif %}"
                         alt="Elevation profile - Route 1" id="elevationImg1" class="loading"
                         onload="document.getElementById('elevationLoading1').classList.add('hidden'); this.classList.remove('loading');">
                    <div class="elevation-cursor" id="elevationCursor1"></div>
                    <div class="elevation-tooltip" id="elevationTooltip1">
                        <div class="grade">--</div>
                        <div class="elev">--</div>
                    </div>
                    <div class="elevation-selection" id="elevationSelection1"></div>
                    <div class="elevation-selection-popup" id="elevationSelectionPopup1" style="display: none;"></div>
                    <div class="zoom-out-link" id="zoomOutLink1" style="display: none;"><a href="#" onclick="return false;">Zoom Out</a></div>
                </div>
            </div>
            {% if result2.tunnels_corrected > 0 %}
            <div class="note anomaly-note" style="margin: 8px 0;">
                <strong>{{ (result2.name or 'Route 2')|truncate(20) }}: {{ result2.tunnels_corrected }} anomal{{ 'ies' if result2.tunnels_corrected > 1 else 'y' }} detected and corrected:</strong>
                {% for t in result2.tunnel_corrections %}
                <span class="anomaly-item">{{ "%.1f"|format(t.start_km) }}-{{ "%.1f"|format(t.end_km) }} km ({{ "%.0f"|format(t.artificial_gain) }}m artificial gain removed)</span>{% if not loop.last %}, {% endif %}
                {% endfor %}
            </div>
            {% endif %}
            <div class="elevation-profile">
                <div class="elevation-profile-header">
                    <h4><a href="{{ url2 }}" target="_blank">{{ (result2.name or ('Trip 2' if is_trip2 else 'Route 2'))|truncate(40) }}</a> <span class="result-badge {% if is_trip2 %}trip-badge{% else %}route-badge{% endif %}">{% if is_trip2 %}Ride{% else %}Route{% endif %}</span> - {{ "%.1f"|format(result2.time_seconds / 3600) }}h</h4>
                    {% if is_trip2 %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="collapseStops2" onchange="toggleCollapseStops(2)">
                        <label for="collapseStops2">Moving time</label>
                    </div>
                    {% endif %}
                </div>
                <div class="elevation-profile-container" id="elevationContainer2"
                     data-url="{{ url2|urlencode }}"
                     data-moving-time-hours="{{ '%.4f'|format(t2_moving) }}"
                     data-elapsed-time-hours="{{ '%.4f'|format(t2_elapsed) }}"
                     data-max-xlim-hours="{{ '%.4f'|format(max_time_hours) }}"
                     {% if compare_ylim %}data-max-ylim="{{ '%.2f'|format(compare_ylim) }}"{% endif %}
                     {% if compare_speed_ylim %}data-max-speed-ylim="{{ '%.2f'|format(compare_speed_ylim) }}"{% endif %}
                     data-base-profile-url="/elevation-profile?url={{ url2|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}{% if compare_ylim %}&max_ylim={{ '%.2f'|format(compare_ylim) }}{% endif %}{% if compare_speed_ylim %}&max_speed_ylim={{ '%.2f'|format(compare_speed_ylim) }}{% endif %}"
                     data-base-data-url="/elevation-profile-data?url={{ url2|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}">
                    <div class="elevation-loading" id="elevationLoading2">
                        <div class="elevation-spinner"></div>
                    </div>
                    <img src="/elevation-profile?url={{ url2|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}&max_xlim_hours={{ '%.4f'|format(max_time_hours) }}{% if compare_ylim %}&max_ylim={{ '%.2f'|format(compare_ylim) }}{% endif %}{% if compare_speed_ylim %}&max_speed_ylim={{ '%.2f'|format(compare_speed_ylim) }}{% endif %}"
                         alt="Elevation profile - Route 2" id="elevationImg2" class="loading"
                         onload="document.getElementById('elevationLoading2').classList.add('hidden'); this.classList.remove('loading');">
                    <div class="elevation-cursor" id="elevationCursor2"></div>
                    <div class="elevation-tooltip" id="elevationTooltip2">
                        <div class="grade">--</div>
                        <div class="elev">--</div>
                    </div>
                    <div class="elevation-selection" id="elevationSelection2"></div>
                    <div class="elevation-selection-popup" id="elevationSelectionPopup2" style="display: none;"></div>
                    <div class="zoom-out-link" id="zoomOutLink2" style="display: none;"><a href="#" onclick="return false;">Zoom Out</a></div>
                </div>
            </div>
        </div>
        {% else %}
        <!-- Single elevation profile -->
        <div class="elevation-profile">
            <div class="elevation-profile-header">
                <h4><a href="{{ url }}" target="_blank">{{ result.name or ('Trip' if is_trip else 'Route') }}</a> - {{ "%.1f"|format(result.time_seconds / 3600) }}h</h4>
                <div class="elevation-profile-toggles">
                    {% if is_trip %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="collapseStops" onchange="toggleCollapseStops()">
                        <label for="collapseStops">Show moving time only</label>
                    </div>
                    {% endif %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="overlay_speed" onchange="toggleOverlay('speed')">
                        <label for="overlay_speed">Speed</label>
                    </div>
                    {% if not is_trip %}
                    <div class="collapse-stops-toggle">
                        <input type="checkbox" id="overlay_gravel" onchange="toggleOverlay('gravel')">
                        <label for="overlay_gravel">Unpaved</label>
                    </div>
                    {% endif %}
                </div>
                {% if not is_trip %}
                <a href="#" class="ride-link" id="viewClimbsLink" onclick="goToRidePage(); return false;" style="font-size: 0.85em; color: #4CAF50;">View Climbs &rarr;</a>
                {% endif %}
            </div>
            <div class="elevation-profile-container" id="elevationContainer"
                 data-url="{{ url|urlencode }}"
                 data-base-profile-url="/elevation-profile?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}"
                 data-base-data-url="/elevation-profile-data?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}">
                <div class="elevation-loading" id="elevationLoading">
                    <div class="elevation-spinner"></div>
                </div>
                <img src="/elevation-profile?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}"
                     alt="Elevation profile" id="elevationImg" class="loading"
                     onload="document.getElementById('elevationLoading').classList.add('hidden'); this.classList.remove('loading');">
                <div class="elevation-cursor" id="elevationCursor"></div>
                <div class="elevation-tooltip" id="elevationTooltip">
                    <div class="grade">--</div>
                    <div class="elev">--</div>
                </div>
                <div class="elevation-selection" id="elevationSelection"></div>
                <div class="elevation-selection-popup" id="elevationSelectionPopup" style="display: none;"></div>
                <div class="zoom-out-link" id="zoomOutLink" style="display: none;"><a href="#" onclick="return false;">Zoom Out</a></div>
            </div>
        </div>
        {% endif %}

        {% if compare_mode and route_id and route_id2 %}
        <!-- Side-by-side RWGPS embeds for comparison (only for routes, trips don't support embeds) -->
        {% if not is_trip and not is_trip2 %}
        <div class="route-maps-comparison">
            <div class="route-map">
                <h4>{{ (result.name or 'Route 1')|truncate(30) }}</h4>
                <iframe src="https://ridewithgps.com/embeds?type=route&id={{ route_id }}&sampleGraph=true{% if privacy_code %}&privacy_code={{ privacy_code }}{% endif %}"
                        scrolling="no"></iframe>
            </div>
            <div class="route-map">
                <h4>{{ (result2.name or 'Route 2')|truncate(30) }}</h4>
                <iframe src="https://ridewithgps.com/embeds?type=route&id={{ route_id2 }}&sampleGraph=true{% if privacy_code2 %}&privacy_code={{ privacy_code2 }}{% endif %}"
                        scrolling="no"></iframe>
            </div>
        </div>
        {% elif not is_trip and is_trip2 %}
        <!-- Only first is a route -->
        <div class="route-map">
            <h4>{{ (result.name or 'Route')|truncate(30) }}</h4>
            <iframe src="https://ridewithgps.com/embeds?type=route&id={{ route_id }}&sampleGraph=true{% if privacy_code %}&privacy_code={{ privacy_code }}{% endif %}"
                    scrolling="no"></iframe>
        </div>
        {% elif is_trip and not is_trip2 %}
        <!-- Only second is a route -->
        <div class="route-map">
            <h4>{{ (result2.name or 'Route')|truncate(30) }}</h4>
            <iframe src="https://ridewithgps.com/embeds?type=route&id={{ route_id2 }}&sampleGraph=true{% if privacy_code2 %}&privacy_code={{ privacy_code2 }}{% endif %}"
                    scrolling="no"></iframe>
        </div>
        {% endif %}
        {% elif route_id and not is_trip %}
        <!-- Single RWGPS embed (only for routes) -->
        <div class="route-map">
            <iframe src="https://ridewithgps.com/embeds?type=route&id={{ route_id }}&sampleGraph=true{% if privacy_code %}&privacy_code={{ privacy_code }}{% endif %}"
                    scrolling="no"></iframe>
        </div>
        {% endif %}
    </div>
    <script>
        // Save URL with route/trip name for recent URLs dropdown
        saveRecentUrl('{{ url }}', '{{ result.name|e if result.name else '' }}');
        {% if compare_mode and url2 and result2 %}
        // Also save the second URL when comparing
        saveRecentUrl('{{ url2 }}', '{{ result2.name|e if result2.name else '' }}');
        {% endif %}
        // Initialize units display
        updateSingleRouteUnits();
        updateComparisonTableUnits();
        initEnergyUnit();

        // Elevation profile hover interaction
        // Setup function for a single profile (global so toggleCollapseStops/toggleOverlay can call it)
        // maxXlimHours: optional max x-axis time (for synchronized comparison profiles)
        window.setupElevationProfile = function(containerId, imgId, tooltipId, cursorId, dataUrl, maxXlimHours) {
            const container = document.getElementById(containerId);
            const img = document.getElementById(imgId);
            const tooltip = document.getElementById(tooltipId);
            const cursor = document.getElementById(cursorId);
            if (!container || !img || !tooltip || !cursor) return;

            // Derive selection element IDs from container suffix
            const suffix = containerId.replace('elevationContainer', '');
            const selection = document.getElementById('elevationSelection' + suffix);
            const selectionPopup = document.getElementById('elevationSelectionPopup' + suffix);

            let profileData = null;
            let xlimHours = maxXlimHours || null;

            // Selection state
            let selectionStart = null;  // xPct where drag started
            let isSelecting = false;
            let selectionActive = false;

            // Zoom state - restore from container data attributes if present
            const savedZoomMin = container.getAttribute('data-zoom-min');
            const savedZoomMax = container.getAttribute('data-zoom-max');
            let isZoomed = !!(savedZoomMin && savedZoomMax);
            let zoomMinHours = savedZoomMin ? parseFloat(savedZoomMin) : null;
            let zoomMaxHours = savedZoomMax ? parseFloat(savedZoomMax) : null;
            let selectionStartTime = null;
            let selectionEndTime = null;
            const zoomOutLink = document.getElementById('zoomOutLink' + suffix);

            // Clear any stale popup and selection from previous initialization
            if (selectionPopup) { selectionPopup.style.display = 'none'; selectionPopup.innerHTML = ''; }
            if (selection) selection.classList.remove('visible');

            // Show/hide zoom out link based on restored state
            if (zoomOutLink) {
                zoomOutLink.style.display = isZoomed ? 'block' : 'none';
                if (isZoomed) {
                    zoomOutLink.querySelector('a').onclick = function(e) {
                        e.preventDefault();
                        zoomOut();
                    };
                }
            }

            // Fetch profile data
            fetch(dataUrl)
                .then(r => {
                    if (!r.ok) {
                        console.error('Profile data fetch failed:', r.status, r.statusText, dataUrl);
                        return { error: 'HTTP ' + r.status };
                    }
                    return r.json();
                })
                .then(data => {
                    if (data && !data.error) {
                        profileData = data;
                        if (!xlimHours) xlimHours = profileData.total_time;
                    } else if (data && data.error) {
                        console.error('Profile data error:', data.error, dataUrl);
                    }
                })
                .catch(err => console.error('Profile data fetch exception:', err, dataUrl));

            // The plot area margins match _set_fixed_margins() for figsize=(14,4)
            // Left: 0.77in/14in = 0.055, Right: 1 - 0.18in/14in = 0.987
            // These are constant regardless of speed overlay (overlay shares same plot area)
            const plotLeftPct = 0.055;
            const plotRightPct = 0.987;

            function getDataAtPosition(xPct) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return null;

                const plotXPct = (xPct - plotLeftPct) / (plotRightPct - plotLeftPct);
                if (plotXPct < 0 || plotXPct > 1) return null;

                // Use zoom bounds if zoomed, otherwise full range
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : (xlimHours || profileData.total_time);
                const time = minTime + plotXPct * (maxTime - minTime);
                if (time > profileData.total_time || time < 0) return null;

                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) {
                        return {
                            grade: profileData.grades[i],
                            elevation: profileData.elevations[i] || 0,
                            speed: profileData.speeds ? profileData.speeds[i] : null,
                            time: time
                        };
                    }
                }
                const lastIdx = profileData.grades.length - 1;
                return {
                    grade: profileData.grades[lastIdx],
                    elevation: profileData.elevations[lastIdx + 1] || 0,
                    speed: profileData.speeds ? profileData.speeds[lastIdx] : null,
                    time: time
                };
            }

            function getIndexAtPosition(xPct) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return -1;
                // Clamp to plot boundaries so extremities still work
                const plotXPct = Math.max(0, Math.min(1, (xPct - plotLeftPct) / (plotRightPct - plotLeftPct)));
                // Use zoom bounds if zoomed
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : (xlimHours || profileData.total_time);
                const time = Math.max(0, Math.min(minTime + plotXPct * (maxTime - minTime), profileData.total_time));
                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) return i;
                }
                return profileData.times.length - 2;
            }

            // Convert time to index (for selection repositioning after zoom)
            function getIndexAtTime(time) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return -1;
                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) return i;
                }
                return profileData.times.length - 2;
            }

            function formatGrade(g) {
                if (g === null || g === undefined) return 'Stopped';
                const sign = g >= 0 ? '+' : '';
                return sign + g.toFixed(1) + '%';
            }

            function formatTime(hours) {
                const h = Math.floor(hours);
                const m = Math.floor((hours - h) * 60);
                return h + 'h ' + m.toString().padStart(2, '0') + 'm';
            }

            function formatDuration(hours) {
                const totalMin = Math.round(hours * 60);
                if (totalMin < 60) return totalMin + 'min';
                const h = Math.floor(totalMin / 60);
                const m = totalMin % 60;
                return h + 'h ' + m.toString().padStart(2, '0') + 'm';
            }

            // Tooltip helpers
            function updateTooltip(xPct, clientX) {
                const data = getDataAtPosition(xPct);
                if (data) {
                    tooltip.querySelector('.grade').textContent = formatGrade(data.grade);
                    const elevUnit = isImperial() ? 'ft' : 'm';
                    const elevVal = isImperial() ? data.elevation * 3.28084 : data.elevation;
                    var speedText = '';
                    if (data.speed !== null && data.speed !== undefined) {
                        var speedUnit = isImperial() ? 'mph' : 'km/h';
                        var speedVal = isImperial() ? data.speed * 0.621371 : data.speed;
                        speedText = ' | ' + speedVal.toFixed(1) + ' ' + speedUnit;
                    }
                    tooltip.querySelector('.elev').textContent = Math.round(elevVal) + ' ' + elevUnit + speedText + ' | ' + formatTime(data.time);

                    const rect = img.getBoundingClientRect();
                    const xPos = clientX - rect.left;
                    tooltip.style.left = xPos + 'px';
                    tooltip.style.bottom = '60px';
                    tooltip.classList.add('visible');
                    cursor.style.left = xPos + 'px';
                    cursor.classList.add('visible');
                } else {
                    hideTooltip();
                }
            }

            function hideTooltip() {
                tooltip.classList.remove('visible');
                cursor.classList.remove('visible');
            }

            // Selection helpers
            function updateSelectionHighlight(startXPct, endXPct) {
                if (!selection) return;
                // Clamp to plot area so highlight doesn't extend into axis margins
                const clampedStart = Math.max(plotLeftPct, Math.min(plotRightPct, startXPct));
                const clampedEnd = Math.max(plotLeftPct, Math.min(plotRightPct, endXPct));
                const left = Math.min(clampedStart, clampedEnd) * 100;
                const right = Math.max(clampedStart, clampedEnd) * 100;
                selection.style.left = left + '%';
                selection.style.width = (right - left) + '%';
                selection.classList.add('visible');
            }

            function computeSelectionStats(startIdx, endIdx) {
                if (!profileData) return null;
                const d = profileData;
                const i = Math.min(startIdx, endIdx);
                const j = Math.max(startIdx, endIdx);
                if (i < 0 || j >= d.times.length) return null;

                // Duration (hours)
                const duration = d.times[j + 1 < d.times.length ? j + 1 : j] - d.times[i];

                // Distance (meters -> km)
                let distM = 0;
                if (d.distances) {
                    for (let k = i; k <= j; k++) distM += (d.distances[k] || 0);
                }
                const distKm = distM / 1000;

                // Elevation gain/loss - use pre-computed arrays to survive downsampling
                let elevGain = 0, elevLoss = 0;
                if (d.elev_gains && d.elev_losses) {
                    for (let k = i; k <= j; k++) {
                        elevGain += (d.elev_gains[k] || 0);
                        elevLoss += (d.elev_losses[k] || 0);
                    }
                } else {
                    for (let k = i; k <= j; k++) {
                        const diff = (d.elevations[k + 1] !== undefined ? d.elevations[k + 1] : d.elevations[k]) - d.elevations[k];
                        if (diff > 0) elevGain += diff;
                        else elevLoss += diff;
                    }
                }

                // Avg grade
                let gradeSum = 0, gradeCount = 0;
                for (let k = i; k <= j; k++) {
                    if (d.grades[k] !== null && d.grades[k] !== undefined) {
                        gradeSum += d.grades[k];
                        gradeCount++;
                    }
                }
                const avgGrade = gradeCount > 0 ? gradeSum / gradeCount : null;

                // Avg speed (km/h) = total distance / total time
                const avgSpeed = (duration > 0 && distKm > 0) ? distKm / duration : null;

                // Total work (using pre-computed works array for accuracy)
                let workJ = 0;
                if (d.works) {
                    for (let k = i; k <= j; k++) {
                        workJ += (d.works[k] || 0);
                    }
                } else if (d.powers) {
                    // Fall back to power*time if works not available
                    for (let k = i; k <= j; k++) {
                        if (d.powers[k] !== null && d.powers[k] !== undefined) {
                            const dt = ((d.times[k + 1 < d.times.length ? k + 1 : k] - d.times[k]) * 3600);
                            workJ += d.powers[k] * dt;
                        }
                    }
                }
                // Avg power = work / time (matches summary calculation)
                const durationSec = duration * 3600;
                const avgPower = (workJ > 0 && durationSec > 0) ? workJ / durationSec : null;
                const workKJ = workJ / 1000;

                return { duration, distKm, elevGain, elevLoss, avgGrade, avgSpeed, avgPower, workKJ };
            }

            function showSelectionPopup(stats, xPctCenter) {
                if (!selectionPopup || !stats) return;
                const imp = isImperial();
                const distVal = imp ? (stats.distKm * 0.621371) : stats.distKm;
                const distUnit = imp ? 'mi' : 'km';
                const elevGainVal = imp ? (stats.elevGain * 3.28084) : stats.elevGain;
                const elevLossVal = imp ? (stats.elevLoss * 3.28084) : stats.elevLoss;
                const elevUnit = imp ? 'ft' : 'm';
                const speedVal = stats.avgSpeed !== null ? (imp ? stats.avgSpeed * 0.621371 : stats.avgSpeed) : null;
                const speedUnit = imp ? 'mph' : 'km/h';

                let html = '<span class="selection-close">&times;</span>';
                html += '<div class="selection-stat"><span class="stat-label">Duration</span><span class="stat-value">' + formatDuration(stats.duration) + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Distance</span><span class="stat-value">' + distVal.toFixed(1) + ' ' + distUnit + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Elev Gain</span><span class="stat-value">+' + Math.round(elevGainVal) + ' ' + elevUnit + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Elev Loss</span><span class="stat-value">' + Math.round(elevLossVal) + ' ' + elevUnit + '</span></div>';
                if (stats.avgGrade !== null) {
                    html += '<div class="selection-stat"><span class="stat-label">Avg Grade</span><span class="stat-value">' + formatGrade(stats.avgGrade) + '</span></div>';
                }
                if (speedVal !== null) {
                    html += '<div class="selection-stat"><span class="stat-label">Avg Speed</span><span class="stat-value">' + speedVal.toFixed(1) + ' ' + speedUnit + '</span></div>';
                }
                if (stats.avgPower !== null) {
                    html += '<div class="selection-stat"><span class="stat-label">Avg Power</span><span class="stat-value">' + Math.round(stats.avgPower) + ' W</span></div>';
                    html += '<div class="selection-stat"><span class="stat-label">Work</span><span class="stat-value">' + stats.workKJ.toFixed(1) + ' kJ</span></div>';
                }

                // Add zoom button
                html += '<div class="selection-zoom-btn">' + (isZoomed ? 'Zoom Out' : 'Zoom In') + '</div>';

                selectionPopup.innerHTML = html;
                selectionPopup.style.display = 'block';
                selectionPopup.style.left = (xPctCenter * 100) + '%';
                selectionPopup.style.bottom = '70px';
                selectionActive = true;

                // Close button handler
                var closeBtn = selectionPopup.querySelector('.selection-close');
                if (closeBtn) {
                    closeBtn.addEventListener('click', function(ev) {
                        ev.stopPropagation();
                        clearSelection();
                    });
                }

                // Zoom button handler
                var zoomBtn = selectionPopup.querySelector('.selection-zoom-btn');
                if (zoomBtn) {
                    zoomBtn.addEventListener('click', function(ev) {
                        ev.stopPropagation();
                        if (isZoomed) {
                            zoomOut();
                        } else {
                            zoomIn();
                        }
                    });
                }
            }

            function clearSelection() {
                if (selection) selection.classList.remove('visible');
                if (selectionPopup) { selectionPopup.style.display = 'none'; selectionPopup.innerHTML = ''; }
                selectionStart = null;
                isSelecting = false;
                selectionActive = false;
            }

            function zoomIn() {
                if (!profileData || !selection) return;

                // Get selection bounds in pixel percentages
                const selLeft = parseFloat(selection.style.left) / 100;
                const selWidth = parseFloat(selection.style.width) / 100;

                // Convert to time using current zoom state
                const plotRange = plotRightPct - plotLeftPct;
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : (xlimHours || profileData.total_time);
                const viewRange = maxTime - minTime;

                const startPct = (selLeft - plotLeftPct) / plotRange;
                const endPct = (selLeft + selWidth - plotLeftPct) / plotRange;

                selectionStartTime = minTime + startPct * viewRange;
                selectionEndTime = minTime + endPct * viewRange;

                // Calculate zoom bounds with padding so selection takes 90% of view
                const selectionDuration = selectionEndTime - selectionStartTime;
                const totalZoomRange = selectionDuration / 0.9;
                const padding = (totalZoomRange - selectionDuration) / 2;

                zoomMinHours = Math.max(0, selectionStartTime - padding);
                zoomMaxHours = Math.min(profileData.total_time, selectionEndTime + padding);

                // Store zoom state on container for persistence across overlay toggles
                container.setAttribute('data-zoom-min', zoomMinHours.toFixed(4));
                container.setAttribute('data-zoom-max', zoomMaxHours.toFixed(4));

                isZoomed = true;
                refreshZoomedProfile();
            }

            function zoomOut() {
                isZoomed = false;
                zoomMinHours = null;
                zoomMaxHours = null;
                selectionStartTime = null;
                selectionEndTime = null;

                // Clear zoom state from container
                container.removeAttribute('data-zoom-min');
                container.removeAttribute('data-zoom-max');

                clearSelection();
                refreshZoomedProfile();
            }

            function refreshZoomedProfile() {
                var baseProfileUrl = container.getAttribute('data-base-profile-url');
                var baseDataUrl = container.getAttribute('data-base-data-url');
                var loading = document.getElementById('elevationLoading' + suffix);

                if (!baseProfileUrl) return;

                // Build URL parameters
                var params = '';
                var collapseCheckbox = document.getElementById('collapseStops' + suffix);
                if (collapseCheckbox && collapseCheckbox.checked) params += '&collapse_stops=true';
                if (typeof _buildOverlayParams === 'function') params += _buildOverlayParams();

                // Add zoom parameters
                if (isZoomed && zoomMinHours !== null && zoomMaxHours !== null) {
                    params += '&min_xlim_hours=' + zoomMinHours.toFixed(4);
                    params += '&max_xlim_hours=' + zoomMaxHours.toFixed(4);
                }

                // Show loading spinner
                if (loading) loading.classList.remove('hidden');
                img.classList.add('loading');

                // Update image
                img.src = baseProfileUrl + params;

                img.onload = function() {
                    if (loading) loading.classList.add('hidden');
                    img.classList.remove('loading');

                    // Show/hide zoom out link
                    if (zoomOutLink) {
                        zoomOutLink.style.display = isZoomed ? 'block' : 'none';
                        if (isZoomed) {
                            zoomOutLink.querySelector('a').onclick = function(e) {
                                e.preventDefault();
                                zoomOut();
                            };
                        }
                    }

                    // Reposition selection if zoomed
                    if (isZoomed && selectionStartTime !== null && selectionEndTime !== null) {
                        repositionSelectionAfterZoom();
                    }
                };
            }

            function repositionSelectionAfterZoom() {
                if (!selection || !isZoomed || selectionStartTime === null || selectionEndTime === null) return;

                // Calculate the new position of selection in zoomed view
                const zoomRange = zoomMaxHours - zoomMinHours;
                const plotRange = plotRightPct - plotLeftPct;

                // Convert time to plot percentage
                const startPct = plotLeftPct + ((selectionStartTime - zoomMinHours) / zoomRange) * plotRange;
                const endPct = plotLeftPct + ((selectionEndTime - zoomMinHours) / zoomRange) * plotRange;

                // Clamp to plot boundaries
                const clampedStart = Math.max(plotLeftPct, Math.min(plotRightPct, startPct));
                const clampedEnd = Math.max(plotLeftPct, Math.min(plotRightPct, endPct));

                selection.style.left = (Math.min(clampedStart, clampedEnd) * 100) + '%';
                selection.style.width = (Math.abs(clampedEnd - clampedStart) * 100) + '%';
                selection.classList.add('visible');

                // Re-show popup with Zoom Out button
                const centerXPct = (clampedStart + clampedEnd) / 2;
                const startIdx = getIndexAtTime(selectionStartTime);
                const endIdx = getIndexAtTime(selectionEndTime);
                const stats = computeSelectionStats(startIdx, endIdx);
                if (stats) {
                    showSelectionPopup(stats, centerXPct);
                }
            }

            // Event handlers
            function onMouseMove(e) {
                if (!profileData) {
                    // Uncomment for debugging: console.log('No profile data yet for', containerId);
                    return;
                }
                const rect = img.getBoundingClientRect();
                const xPct = (e.clientX - rect.left) / rect.width;

                if (isSelecting) {
                    updateSelectionHighlight(selectionStart, xPct);
                    updateTooltip(xPct, e.clientX);
                } else if (!selectionActive) {
                    updateTooltip(xPct, e.clientX);
                }
            }

            function onMouseDown(e) {
                if (!profileData) return;
                // If popup is showing and click is outside popup, dismiss popup but keep selection if zoomed
                if (selectionActive) {
                    // Don't dismiss if clicking inside the popup
                    if (selectionPopup && selectionPopup.contains(e.target)) {
                        return;
                    }
                    if (selectionPopup) {
                        selectionPopup.style.display = 'none';
                        selectionPopup.innerHTML = '';
                    }
                    selectionActive = false;
                    // Keep selection visible if zoomed
                    if (!isZoomed && selection) {
                        selection.classList.remove('visible');
                    }
                    return;
                }
                const rect = img.getBoundingClientRect();
                selectionStart = (e.clientX - rect.left) / rect.width;
                isSelecting = true;
                e.preventDefault();
            }

            function onMouseUp(e) {
                if (!isSelecting) return;
                isSelecting = false;
                const rect = img.getBoundingClientRect();
                const xPctEnd = (e.clientX - rect.left) / rect.width;
                const dragPx = Math.abs(e.clientX - rect.left - selectionStart * rect.width);

                if (dragPx > 5) {
                    const startIdx = getIndexAtPosition(selectionStart);
                    const endIdx = getIndexAtPosition(xPctEnd);
                    if (startIdx >= 0 && endIdx >= 0 && startIdx !== endIdx) {
                        const stats = computeSelectionStats(startIdx, endIdx);
                        const centerXPct = (selectionStart + xPctEnd) / 2;
                        hideTooltip();
                        showSelectionPopup(stats, centerXPct);
                    } else {
                        clearSelection();
                    }
                } else {
                    clearSelection();
                }
            }

            function onMouseLeave(e) {
                if (isSelecting) {
                    // Finish selection on leave
                    onMouseUp(e);
                }
                hideTooltip();
            }

            // Long-press touch selection
            const LONG_PRESS_DURATION = 400;  // ms to trigger long-press
            const LONG_PRESS_MOVE_THRESHOLD = 15;  // px movement allowed during long-press wait
            let longPressTimer = null;
            let longPressStartX = 0;
            let longPressStartY = 0;
            let longPressPending = false;

            // Create long-press indicator element
            let longPressIndicator = container.querySelector('.long-press-indicator');
            if (!longPressIndicator) {
                longPressIndicator = document.createElement('div');
                longPressIndicator.className = 'long-press-indicator';
                longPressIndicator.innerHTML = '<div class="long-press-ring"></div>';
                container.appendChild(longPressIndicator);
            }

            function showLongPressIndicator(clientX, clientY) {
                const rect = container.getBoundingClientRect();
                longPressIndicator.style.left = (clientX - rect.left) + 'px';
                longPressIndicator.style.top = (clientY - rect.top) + 'px';
                longPressIndicator.classList.add('active');
            }

            function hideLongPressIndicator() {
                longPressIndicator.classList.remove('active');
            }

            function triggerHapticFeedback() {
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
            }

            function cancelLongPress() {
                if (longPressTimer) {
                    clearTimeout(longPressTimer);
                    longPressTimer = null;
                }
                longPressPending = false;
                hideLongPressIndicator();
            }

            function onTouchStart(e) {
                if (!profileData) return;
                // If popup is showing, check if tap is inside popup
                if (selectionActive) {
                    // Don't dismiss if tapping inside the popup
                    if (selectionPopup && e.touches[0] && selectionPopup.contains(document.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY))) {
                        return;
                    }
                    if (selectionPopup) {
                        selectionPopup.style.display = 'none';
                        selectionPopup.innerHTML = '';
                    }
                    selectionActive = false;
                    // Keep selection visible if zoomed
                    if (!isZoomed && selection) {
                        selection.classList.remove('visible');
                    }
                    return;
                }

                const touch = e.touches[0];
                longPressStartX = touch.clientX;
                longPressStartY = touch.clientY;
                longPressPending = true;

                // Show visual indicator immediately
                showLongPressIndicator(touch.clientX, touch.clientY);

                // Start long-press timer
                longPressTimer = setTimeout(() => {
                    if (!longPressPending) return;
                    longPressPending = false;
                    hideLongPressIndicator();

                    // Trigger haptic feedback
                    triggerHapticFeedback();

                    // Enter selection mode
                    const rect = img.getBoundingClientRect();
                    selectionStart = (touch.clientX - rect.left) / rect.width;
                    isSelecting = true;

                    // Show initial selection highlight at touch point
                    updateSelectionHighlight(selectionStart, selectionStart);
                }, LONG_PRESS_DURATION);
            }

            function onTouchMove(e) {
                if (!profileData) return;

                const touch = e.touches[0];

                // If waiting for long-press, check if moved too much
                if (longPressPending) {
                    const dx = touch.clientX - longPressStartX;
                    const dy = touch.clientY - longPressStartY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance > LONG_PRESS_MOVE_THRESHOLD) {
                        // User is scrolling, cancel long-press
                        cancelLongPress();
                        return;
                    }
                    // Still waiting for long-press, don't prevent default (allow scroll)
                    return;
                }

                // In selection mode - prevent scrolling and update selection
                if (isSelecting) {
                    e.preventDefault();
                    const rect = img.getBoundingClientRect();
                    const xPct = (touch.clientX - rect.left) / rect.width;
                    updateSelectionHighlight(selectionStart, xPct);
                    updateTooltip(xPct, touch.clientX);
                }
            }

            function onTouchEnd(e) {
                // Cancel any pending long-press
                if (longPressPending) {
                    cancelLongPress();
                    hideTooltip();
                    return;
                }

                if (!isSelecting) { hideTooltip(); return; }
                isSelecting = false;
                if (!selection) { hideTooltip(); return; }

                const rect = img.getBoundingClientRect();
                const selLeft = parseFloat(selection.style.left) / 100;
                const selWidth = parseFloat(selection.style.width) / 100;
                if (selWidth * rect.width > 30) {
                    const startIdx = getIndexAtPosition(selLeft);
                    const endIdx = getIndexAtPosition(selLeft + selWidth);
                    if (startIdx >= 0 && endIdx >= 0 && startIdx !== endIdx) {
                        const stats = computeSelectionStats(startIdx, endIdx);
                        hideTooltip();
                        showSelectionPopup(stats, selLeft + selWidth / 2);
                    } else clearSelection();
                } else clearSelection();
                hideTooltip();
            }

            function onTouchCancel(e) {
                cancelLongPress();
                if (isSelecting) {
                    isSelecting = false;
                    clearSelection();
                }
                hideTooltip();
            }

            function onKeyDown(e) {
                if (e.key === 'Escape') clearSelection();
            }

            // Clean up old listeners if this profile was previously set up
            if (container._profileCleanup) {
                container._profileCleanup();
            }

            var abortController = new AbortController();
            container.addEventListener('mousemove', onMouseMove, { signal: abortController.signal });
            container.addEventListener('mousedown', onMouseDown, { signal: abortController.signal });
            container.addEventListener('mouseup', onMouseUp, { signal: abortController.signal });
            container.addEventListener('mouseleave', onMouseLeave, { signal: abortController.signal });
            container.addEventListener('touchstart', onTouchStart, { passive: true, signal: abortController.signal });
            container.addEventListener('touchmove', onTouchMove, { passive: false, signal: abortController.signal });
            container.addEventListener('touchend', onTouchEnd, { passive: true, signal: abortController.signal });
            container.addEventListener('touchcancel', onTouchCancel, { passive: true, signal: abortController.signal });
            document.addEventListener('keydown', onKeyDown, { signal: abortController.signal });

            container._profileCleanup = function() {
                abortController.abort();
            };
        };

        (function() {
            // Check for comparison mode (two profiles)
            var container1 = document.getElementById('elevationContainer1');
            var container2 = document.getElementById('elevationContainer2');
            if (container1 && container2) {
                // Comparison mode - setup both profiles with synchronized x-axis
                var maxXlimHours = parseFloat(container1.getAttribute('data-max-xlim-hours')) || null;
                window.setupElevationProfile(
                    'elevationContainer1', 'elevationImg1', 'elevationTooltip1', 'elevationCursor1',
                    '/elevation-profile-data?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}',
                    maxXlimHours
                );
                window.setupElevationProfile(
                    'elevationContainer2', 'elevationImg2', 'elevationTooltip2', 'elevationCursor2',
                    '/elevation-profile-data?url={{ url2|urlencode if url2 else "" }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}',
                    maxXlimHours
                );
            } else {
                // Single route mode
                window.setupElevationProfile(
                    'elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor',
                    '/elevation-profile-data?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&descending_power={{ descending_power }}&descent_braking_factor={{ descent_braking_factor }}&unpaved_power_factor={{ unpaved_power_factor }}&smoothing={{ smoothing|int }}'
                );
            }

            // Initialize overlays from localStorage (after profiles are set up)
            initOverlays();
        })();
    </script>
    {% endif %}

    <div class="footer">
        <div class="footer-content">
            <div class="footer-links">
                <a href="https://github.com/sanmi/gpx-analyzer" target="_blank">Source Code</a>
                <a href="https://github.com/sanmi/gpx-analyzer/issues" target="_blank">Report a Bug</a>
            </div>
            <div class="footer-version">{{ version_date }} ({{ git_hash }})</div>
            <div class="footer-copyright">© 2025 Frank San Miguel</div>
        </div>
    </div>
    {% if umami_website_id and result %}
    <script>
        // Track analysis with Umami (wait for script to load)
        window.addEventListener('load', function() {
            if (typeof umami !== 'undefined') {
                {% if compare_mode and result2 %}
                umami.track('analyze', {
                    type: '{{ "trip" if is_trip else "route" }}',
                    compare: true,
                    url: '{{ url }}',
                    distance_km: {{ "%.1f"|format(result.distance_km) }},
                    elevation_m: {{ "%.0f"|format(result.elevation_m) }},
                    name: '{{ result.name|replace("'", "\\'") if result.name else "" }}'
                });
                umami.track('analyze', {
                    type: '{{ "trip" if is_trip2 else "route" }}',
                    compare: true,
                    url: '{{ url2 }}',
                    distance_km: {{ "%.1f"|format(result2.distance_km) }},
                    elevation_m: {{ "%.0f"|format(result2.elevation_m) }},
                    name: '{{ result2.name|replace("'", "\\'") if result2.name else "" }}'
                });
                {% else %}
                umami.track('analyze', {
                    type: '{{ "trip" if is_trip else "route" }}',
                    url: '{{ url }}',
                    distance_km: {{ "%.1f"|format(result.distance_km) }},
                    elevation_m: {{ "%.0f"|format(result.elevation_m) }},
                    name: '{{ result.name|replace("'", "\\'") if result.name else "" }}'
                });
                {% endif %}
            }
        });
    </script>
    {% endif %}
</body>
</html>
"""


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
        "unpaved_power_factor": config.get("unpaved_power_factor", DEFAULTS["unpaved_power_factor"]),
        "smoothing": config.get("smoothing", DEFAULTS["smoothing"]),
    }


def build_params(
    climbing_power: float, flat_power: float, mass: float, headwind: float,
    descent_braking_factor: float | None = None,
    descending_power: float | None = None,
    unpaved_power_factor: float | None = None,
) -> RiderParams:
    """Build RiderParams from user inputs and config defaults."""
    config = _load_config() or {}
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
        unpaved_power_factor=unpaved_power_factor if unpaved_power_factor is not None else config.get("unpaved_power_factor", DEFAULTS["unpaved_power_factor"]),
    )


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_duration_long(seconds: float) -> str:
    """Format seconds as Xh Ym Zs string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def format_time_diff(seconds1: float, seconds2: float) -> str:
    """Format time difference as +/- Xh Ym."""
    diff = seconds1 - seconds2
    sign = "+" if diff > 0 else ""
    return sign + format_duration(abs(diff))


def format_diff(val1: float, val2: float, unit: str, decimals: int = 0) -> str:
    """Format numeric difference with sign and unit."""
    diff = val1 - val2
    sign = "+" if diff > 0 else ""
    if decimals == 0:
        return f"{sign}{diff:.0f} {unit}"
    return f"{sign}{diff:.{decimals}f} {unit}"


def format_pct_diff(val1: float, val2: float) -> str:
    """Format percentage point difference."""
    diff = val1 - val2
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"


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
    points, tunnel_corrections = detect_and_correct_tunnels(points)

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
        "steep_time_seconds": sum(hilliness.steep_time_histogram.values()),
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


def analyze_trip(url: str) -> dict:
    """Analyze a recorded trip - actual values, no estimation needed.

    Args:
        url: RideWithGPS trip URL

    Returns:
        Dict with trip analysis results including actual recorded time and power.
    """
    config = _load_config() or {}
    smoothing_radius = config.get("smoothing", DEFAULTS["smoothing"])
    max_grade_window = config.get("max_grade_window_route", DEFAULTS["max_grade_window_route"])
    max_grade_smoothing = config.get("max_grade_smoothing", DEFAULTS["max_grade_smoothing"])

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
    track_points, tunnel_corrections = detect_and_correct_tunnels(track_points)

    # Apply smoothing for elevation profile and grade calculations
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0
    if api_elevation_gain and api_elevation_gain > 0:
        unscaled = smooth_elevations(track_points, smoothing_radius, 1.0)
        smoothed_gain = calculate_elevation_gain(unscaled)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    smoothed_points = smooth_elevations(track_points, smoothing_radius, api_elevation_scale)

    # Calculate rolling grades for max grade (filters GPS noise)
    unscaled_points = smooth_elevations(track_points, smoothing_radius, 1.0)
    if max_grade_smoothing > 0:
        grade_points = smooth_elevations(unscaled_points, max_grade_smoothing, 1.0)
    else:
        grade_points = unscaled_points
    rolling_grades = _calculate_rolling_grades(grade_points, max_grade_window)
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

        # Track steep distances
        if rolling_grade >= 10:
            steep_distance += dist
        if rolling_grade >= 15:
            very_steep_distance += dist

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
        "steep_time_seconds": sum(steep_times.values()),
        "hilliness_total_time": sum(grade_times.values()),
        "hilliness_total_distance": total_distance,
        # Trip-specific flags
        "is_trip": True,
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

    return result


def _get_elevation_profile_cache_stats() -> dict:
    """Get statistics for the elevation profile disk cache."""
    index = _load_profile_cache_index()
    total_bytes = 0
    for key in index:
        path = PROFILE_CACHE_DIR / f"{key}.png"
        if path.exists():
            total_bytes += path.stat().st_size
    total = _elevation_profile_cache_stats["hits"] + _elevation_profile_cache_stats["misses"]
    hit_rate = (_elevation_profile_cache_stats["hits"] / total * 100) if total > 0 else 0
    return {
        "hit_rate": f"{hit_rate:.1f}%",
        "hits": _elevation_profile_cache_stats["hits"],
        "max_size": MAX_CACHED_PROFILES,
        "misses": _elevation_profile_cache_stats["misses"],
        "zoomed_skipped": _elevation_profile_cache_stats["zoomed_skipped"],
        "size": len(index),
        "disk_kb": round(total_bytes / 1024, 1),
    }


def _clear_elevation_profile_cache() -> int:
    """Clear the elevation profile disk cache. Returns number of files removed."""
    global _elevation_profile_cache_stats
    index = _load_profile_cache_index()
    count = 0
    for key in list(index.keys()):
        path = PROFILE_CACHE_DIR / f"{key}.png"
        if path.exists():
            path.unlink()
            count += 1
    # Clear the index and reset stats
    _save_profile_cache_index({})
    _elevation_profile_cache_stats = {"hits": 0, "misses": 0, "zoomed_skipped": 0}
    return count


@app.route("/cache-stats")
def cache_stats():
    """Return cache statistics as JSON for all four caches."""
    analysis = _analysis_cache.stats()
    elevation_profile = _get_elevation_profile_cache_stats()
    climb = _get_climb_cache_stats()
    route_json = get_route_cache_stats()

    # Calculate totals (memory caches use memory_kb, disk caches use disk_kb)
    total_memory_kb = analysis["memory_kb"] + climb["memory_kb"]
    total_disk_kb = elevation_profile["disk_kb"] + route_json["disk_kb"]

    return {
        "analysis_cache": analysis,
        "elevation_profile_cache": elevation_profile,
        "climb_cache": climb,
        "route_json_cache": route_json,
        "totals": {
            "memory_kb": total_memory_kb,
            "disk_kb": total_disk_kb,
        },
    }


@app.route("/cache-clear", methods=["GET", "POST"])
def cache_clear():
    """Clear all caches: analysis, elevation profiles, climb detection, and route JSON."""
    analysis_size = _analysis_cache.stats()["size"]
    _analysis_cache.clear()
    profiles_cleared = _clear_elevation_profile_cache()
    climbs_cleared = _clear_climb_cache()
    routes_cleared = clear_route_json_cache()
    return {
        "status": "ok",
        "message": f"Caches cleared: analysis ({analysis_size}), elevation_profiles ({profiles_cleared}), climbs ({climbs_cleared}), routes ({routes_cleared})"
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
        unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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

            params = build_params(climbing_power, flat_power, mass, headwind, descending_power=descending_power, descent_braking_factor=descent_braking_factor, unpaved_power_factor=unpaved_power_factor)

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


def _smooth_speeds(speeds_ms: list, cum_dist: list, window_m: float = 300) -> list:
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


def _add_speed_overlay(ax, times_hours: list, speeds_kmh: list, imperial: bool = False,
                       max_speed_ylim: float | None = None):
    """Add speed line overlay with right Y-axis to an elevation profile plot.

    Args:
        max_speed_ylim: Optional max speed in km/h for synchronized Y-axis.
                        Converted to display units internally.
    """
    if imperial:
        speeds = [s * 0.621371 for s in speeds_kmh]
        label = 'Speed (mph)'
        tick_interval = 5
        max_display = max_speed_ylim * 0.621371 if max_speed_ylim is not None else None
    else:
        speeds = list(speeds_kmh)
        label = 'Speed (km/h)'
        tick_interval = 10
        max_display = max_speed_ylim if max_speed_ylim is not None else None

    # Segment midpoint times
    speed_times = [(times_hours[i] + times_hours[i + 1]) / 2
                   for i in range(len(speeds))]

    ax2 = ax.twinx()
    ax2.plot(speed_times, speeds, color='#2196F3', linewidth=1.2, alpha=0.7)
    ax2.set_ylabel(label, fontsize=10, color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    max_speed = max_display if max_display is not None else (max(speeds) if speeds else 50)
    max_tick = int(max_speed / tick_interval + 1) * tick_interval
    ax2.set_yticks(range(0, max_tick + 1, tick_interval))
    ax2.set_ylim(0, max_tick)
    ax2.spines['top'].set_visible(False)


def _calculate_elevation_profile_data(url: str, params: RiderParams, smoothing: float | None = None, smoothing_override: bool = False) -> dict:
    """Calculate elevation profile data for a route.

    Returns dict with times_hours, elevations, grades, route_name, and tunnel_corrections.
    """
    config = _load_config() or {}
    smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Detect and correct elevation anomalies (tunnels, bridges, etc.)
    points, tunnel_corrections = detect_and_correct_tunnels(points)

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

    return {
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
    }


def _calculate_trip_elevation_profile_data(url: str, collapse_stops: bool = False) -> dict:
    """Calculate elevation profile data for a trip using actual timestamps.

    Args:
        url: RideWithGPS trip URL
        collapse_stops: If True, use cumulative moving time (excludes stops) for x-axis.
                       This makes the profile comparable to route profiles.

    Returns dict with times_hours, elevations, grades, and route_name.
    """
    config = _load_config() or {}
    smoothing_radius = config.get("smoothing", DEFAULTS["smoothing"])
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
    track_points, tunnel_corrections = detect_and_correct_tunnels(track_points)

    # Apply smoothing with API elevation scaling
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0
    unscaled_points = smooth_elevations(track_points, smoothing_radius, 1.0)
    if api_elevation_gain and api_elevation_gain > 0:
        smoothed_gain = calculate_elevation_gain(unscaled_points)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    if api_elevation_scale != 1.0:
        scaled_points = smooth_elevations(track_points, smoothing_radius, api_elevation_scale)
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


def _set_fixed_margins(fig, fig_width: float, fig_height: float) -> None:
    """Set fixed margins in inches for consistent JavaScript coordinate mapping.

    Uses fixed inch-based margins so that the plot area is predictable
    regardless of content. The JavaScript uses the formula:
        left_pct = 0.77 / fig_width
        right_pct = 1 - 0.18 / fig_width
    which corresponds to 0.77 inch left margin and 0.18 inch right margin.
    """
    # Fixed margins in inches - these match the JavaScript formula
    left_margin_in = 0.77
    right_margin_in = 0.18
    bottom_margin_in = 0.55
    top_margin_in = 0.35

    # Convert to figure fractions
    left = left_margin_in / fig_width
    right = 1 - right_margin_in / fig_width
    bottom = bottom_margin_in / fig_height
    top = 1 - top_margin_in / fig_height

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def generate_elevation_profile(url: str, params: RiderParams, title_time_hours: float | None = None,
                               max_xlim_hours: float | None = None,
                               overlay: str | None = None, imperial: bool = False,
                               max_ylim: float | None = None,
                               max_speed_ylim: float | None = None,
                               show_gravel: bool = False,
                               smoothing: float | None = None,
                               aspect_ratio: float = 3.5,
                               min_xlim_hours: float | None = None) -> bytes:
    """Generate elevation profile image with grade-based coloring.

    Args:
        url: RideWithGPS route URL
        params: Rider parameters
        title_time_hours: Optional time to display in title (from calibrated analysis).
                          If None, uses uncalibrated time from profile data.
        max_xlim_hours: Optional max x-axis limit in hours (for synchronized comparison).
                        If None, uses the profile's own max time.
        overlay: Optional overlay type (e.g. "speed").
        imperial: If True, use imperial units for overlay axis.
        max_ylim: Optional max y-axis limit in meters (for synchronized comparison).
        max_speed_ylim: Optional max speed y-axis limit in km/h (for synchronized comparison).
        show_gravel: If True, highlight unpaved/gravel sections with a brown strip.
        smoothing: Optional override for elevation smoothing radius.
        aspect_ratio: Width/height ratio (1.0 = square, 3.5 = wide default).
        min_xlim_hours: Optional min x-axis limit in hours (for zooming).

    Returns PNG image as bytes.
    """
    data = _calculate_elevation_profile_data(url, params, smoothing)
    times_hours = data["times_hours"]
    elevations = data["elevations"]
    grades = data["grades"]
    tunnel_time_ranges = data.get("tunnel_time_ranges", [])

    # Grade to color mapping - matches histogram colors exactly
    # Main histogram colors: <-10, -10, -8, -6, -4, -2, 0, +2, +4, +6, +8, >10
    main_colors = [
        '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
        '#cccccc',
        '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
    ]
    # Steep histogram colors: 10-12, 12-14, 14-16, 16-18, 18-20, >20
    steep_colors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000']

    def grade_to_color(g):
        """Map grade to histogram color."""
        # For grades >= 10%, use steep histogram colors
        if g >= 10:
            if g < 12:
                return steep_colors[0]
            elif g < 14:
                return steep_colors[1]
            elif g < 16:
                return steep_colors[2]
            elif g < 18:
                return steep_colors[3]
            elif g < 20:
                return steep_colors[4]
            else:
                return steep_colors[5]
        # For grades < 10%, use main histogram colors
        for i, threshold in enumerate(GRADE_BINS[1:]):
            if g < threshold:
                return main_colors[i]
        return main_colors[-1]

    # Create figure with dynamic aspect ratio
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

    # Build all polygons and colors at once for efficient rendering
    # Using PolyCollection is ~30x faster than calling fill_between in a loop
    polygons = []
    colors = []
    for i in range(len(grades)):
        t0, t1 = times_hours[i], times_hours[i+1]
        e0, e1 = elevations[i], elevations[i+1]
        # Each polygon: bottom-left, bottom-right, top-right, top-left
        polygons.append([(t0, 0), (t1, 0), (t1, e1), (t0, e0)])
        colors.append(grade_to_color(grades[i]))

    # Draw all filled segments at once
    coll = PolyCollection(polygons, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(coll)

    # Add outline on top
    ax.plot(times_hours, elevations, color='#333333', linewidth=0.5)

    # Highlight anomaly-corrected regions with vertical bands and markers
    max_elev = max_ylim if max_ylim is not None else max(elevations) * 1.1
    for start_time, end_time in tunnel_time_ranges:
        ax.axvspan(start_time, end_time, alpha=0.25, color='#FFC107', zorder=0.5,
                   label='Anomaly corrected' if start_time == tunnel_time_ranges[0][0] else None)
        # Add "A" marker at top center of band
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, max_elev * 0.92, 'A', fontsize=9, fontweight='bold',
                color='#E65100', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=1))

    # Highlight unpaved/gravel sections with a thin brown strip at the bottom
    if show_gravel:
        strip_height = max_elev * 0.03
        for start_time, end_time in data.get("unpaved_time_ranges", []):
            ax.fill_between([start_time, end_time], 0, strip_height,
                            color='#8B6914', alpha=0.5, zorder=1)

    # Style the plot - use xlim parameters for zooming/comparison
    x_min = min_xlim_hours if min_xlim_hours is not None else 0
    x_max = max_xlim_hours if max_xlim_hours is not None else times_hours[-1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_elev)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)

    ax.spines['top'].set_visible(False)

    if overlay == "speed" and data.get("speeds_kmh"):
        _add_speed_overlay(ax, times_hours, data["speeds_kmh"], imperial, max_speed_ylim=max_speed_ylim)
    else:
        ax.spines['right'].set_visible(False)

    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    _set_fixed_margins(fig, fig_width, fig_height)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_trip_elevation_profile(url: str, title_time_hours: float | None = None, collapse_stops: bool = False,
                                    max_xlim_hours: float | None = None,
                                    overlay: str | None = None, imperial: bool = False,
                                    max_ylim: float | None = None,
                                    max_speed_ylim: float | None = None,
                                    min_xlim_hours: float | None = None) -> bytes:
    """Generate elevation profile image for a trip with grade-based coloring.

    Args:
        url: RideWithGPS trip URL
        title_time_hours: Optional time to display in title.
        collapse_stops: If True, use moving time (excludes stops) for x-axis.
        max_xlim_hours: Optional max x-axis limit in hours (for synchronized comparison).
                        If None, uses the profile's own max time.
        overlay: Optional overlay type (e.g. "speed").
        imperial: If True, use imperial units for overlay axis.
        max_ylim: Optional max y-axis limit in meters (for synchronized comparison).
        max_speed_ylim: Optional max speed y-axis limit in km/h (for synchronized comparison).
        min_xlim_hours: Optional min x-axis limit in hours (for zooming).

    Returns PNG image as bytes.
    """
    data = _calculate_trip_elevation_profile_data(url, collapse_stops=collapse_stops)
    times_hours = data["times_hours"]
    elevations = data["elevations"]
    grades = data["grades"]
    tunnel_time_ranges = data.get("tunnel_time_ranges", [])

    # Reuse the same grade-to-color mapping from route profile
    main_colors = [
        '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
        '#cccccc',
        '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
    ]
    steep_colors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000']

    def grade_to_color(g):
        if g is None:
            return '#ffffff'  # White for stopped segments
        if g >= 10:
            if g < 12:
                return steep_colors[0]
            elif g < 14:
                return steep_colors[1]
            elif g < 16:
                return steep_colors[2]
            elif g < 18:
                return steep_colors[3]
            elif g < 20:
                return steep_colors[4]
            else:
                return steep_colors[5]
        for i, threshold in enumerate(GRADE_BINS[1:]):
            if g < threshold:
                return main_colors[i]
        return main_colors[-1]

    fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')

    polygons = []
    colors = []
    for i in range(len(grades)):
        t0, t1 = times_hours[i], times_hours[i+1]
        e0, e1 = elevations[i], elevations[i+1]
        polygons.append([(t0, 0), (t1, 0), (t1, e1), (t0, e0)])
        colors.append(grade_to_color(grades[i]))

    coll = PolyCollection(polygons, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(coll)

    ax.plot(times_hours, elevations, color='#333333', linewidth=0.5)

    # Highlight anomaly-corrected regions with vertical bands and markers
    max_elev = max_ylim if max_ylim is not None else max(elevations) * 1.1
    for start_time, end_time in tunnel_time_ranges:
        ax.axvspan(start_time, end_time, alpha=0.25, color='#FFC107', zorder=0.5,
                   label='Anomaly corrected' if start_time == tunnel_time_ranges[0][0] else None)
        # Add "A" marker at top center of band
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, max_elev * 0.92, 'A', fontsize=9, fontweight='bold',
                color='#E65100', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=1))

    # Use xlim parameters for zooming/comparison
    x_min = min_xlim_hours if min_xlim_hours is not None else 0
    x_max = max_xlim_hours if max_xlim_hours is not None else times_hours[-1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_elev)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)

    ax.spines['top'].set_visible(False)

    if overlay == "speed" and data.get("speeds_kmh"):
        _add_speed_overlay(ax, times_hours, data["speeds_kmh"], imperial, max_speed_ylim=max_speed_ylim)
    else:
        ax.spines['right'].set_visible(False)

    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    _set_fixed_margins(fig, 14, 4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
    unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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
        _elevation_profile_cache_stats["zoomed_skipped"] += 1
    else:
        cache_key = _make_profile_cache_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, collapse_stops, max_xlim_hours, descending_power, overlay=(overlay or "") + f"|aspect{aspect_ratio:.1f}", imperial=imperial, show_gravel=show_gravel, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, unpaved_power_factor=unpaved_power_factor, smoothing=smoothing, min_xlim_hours=min_xlim_hours)
        cached_bytes = _get_cached_profile(cache_key)
        if cached_bytes:
            return send_file(io.BytesIO(cached_bytes), mimetype='image/png')

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps, no physics params needed
            trip_result = analyze_trip(url)
            title_time_hours = trip_result["time_seconds"] / 3600
            img_bytes = generate_trip_elevation_profile(url, title_time_hours, collapse_stops=collapse_stops, max_xlim_hours=max_xlim_hours, overlay=overlay, imperial=imperial, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, min_xlim_hours=min_xlim_hours)
        else:
            # Route: use physics estimation
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
            analysis = analyze_single_route(url, params, smoothing)
            title_time_hours = analysis["time_seconds"] / 3600
            img_bytes = generate_elevation_profile(url, params, title_time_hours, max_xlim_hours=max_xlim_hours, overlay=overlay, imperial=imperial, max_ylim=max_ylim, max_speed_ylim=max_speed_ylim, show_gravel=show_gravel, smoothing=smoothing, aspect_ratio=aspect_ratio, min_xlim_hours=min_xlim_hours)
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
    unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
    smoothing = float(request.args.get("smoothing", defaults["smoothing"]))
    collapse_stops = request.args.get("collapse_stops", "false").lower() == "true"

    if not url or not (is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps (or moving time if collapse_stops)
            data = _calculate_trip_elevation_profile_data(url, collapse_stops=collapse_stops)
        else:
            # Route: use physics estimation
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
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
            # Use weighted average of rolling grades to match histogram methodology
            # The grades array already contains rolling grades calculated the same way as the histogram
            # Don't recalculate from raw elevation data as that bypasses the rolling window
            if distances and grades:
                new_grades = []
                for i in range(0, len(grades), step):
                    chunk_grades = grades[i:i+step]
                    chunk_dists = distances[i:i+step]
                    # Filter out None grades (stopped segments in trips)
                    valid_pairs = [(g, d) for g, d in zip(chunk_grades, chunk_dists) if g is not None]
                    if valid_pairs:
                        total_valid_dist = sum(d for _, d in valid_pairs)
                        if total_valid_dist > 0:
                            weighted_grade = sum(g * d for g, d in valid_pairs) / total_valid_dist
                            new_grades.append(weighted_grade)
                        else:
                            new_grades.append(valid_pairs[0][0])
                    else:
                        # All grades in chunk are None (stopped)
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
        result = analyze_trip(url)
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
    is_trip = False  # Flag for trip vs route
    is_trip2 = False  # Flag for second URL trip vs route

    climbing_power = defaults["climbing_power"]
    flat_power = defaults["flat_power"]
    descending_power = defaults["descending_power"]
    mass = defaults["mass"]
    headwind = defaults["headwind"]
    descent_braking_factor = defaults["descent_braking_factor"]
    unpaved_power_factor = defaults["unpaved_power_factor"]
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
            unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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
                    params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
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
            unpaved_power_factor = float(request.form.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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
                        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
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
                data1 = _calculate_trip_elevation_profile_data(url)
            else:
                params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
                data1 = _calculate_elevation_profile_data(url, params, smoothing, smoothing_override)
            if is_trip2:
                data2 = _calculate_trip_elevation_profile_data(url2)
            else:
                if not is_trip:
                    # params already built above
                    pass
                else:
                    params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
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
            "unpaved_power_factor": unpaved_power_factor,
            "smoothing": smoothing,
        }
        if url2 and compare_mode:
            share_params["url2"] = url2
        if imperial:
            share_params["imperial"] = "1"
        base_url = request.url_root.replace('http://', 'https://')
        share_url = f"{base_url}?{urlencode(share_params)}"

    return render_template_string(
        HTML_TEMPLATE,
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
        unpaved_power_factor=unpaved_power_factor,
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
        version_date=__version_date__,
        git_hash=get_git_hash(),
        # Helper functions for comparison formatting
        format_time_diff=format_time_diff,
        format_diff=format_diff,
        format_pct_diff=format_pct_diff,
        # Analytics
        **_get_analytics_config(),
    )


RIDE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>{% if route_name %}{{ route_name }} | {% endif %}Ride Details</title>
    <style>
        :root {
            --primary: #FF6B35;
            --climb-green: #4CAF50;
            --text-dark: #333;
            --text-muted: #666;
            --bg-light: #f5f5f7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 12px;
            background: var(--bg-light);
            color: var(--text-dark);
        }
        /* Responsive container for main content */
        @media (min-width: 768px) {
            body { padding: 24px 32px; }
        }
        .back-link {
            display: inline-block;
            margin-bottom: 12px;
            color: var(--primary);
            text-decoration: none;
            font-size: 14px;
        }
        .back-link:hover { text-decoration: underline; }
        .summary-card {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        @media (min-width: 768px) {
            .summary-card {
                padding: 24px;
                margin-bottom: 24px;
            }
        }
        .route-name {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-dark);
        }
        @media (min-width: 768px) {
            .route-name {
                font-size: 1.5em;
                margin-bottom: 16px;
            }
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        @media (min-width: 600px) {
            .summary-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        .summary-item {
            text-align: center;
            padding: 8px;
            background: var(--bg-light);
            border-radius: 8px;
        }
        .summary-value {
            font-size: 1.4em;
            font-weight: 700;
            color: var(--text-dark);
        }
        @media (min-width: 768px) {
            .summary-value {
                font-size: 1.6em;
            }
        }
        .summary-label {
            font-size: 0.75em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .elevation-section {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        @media (min-width: 768px) {
            .elevation-section {
                padding: 24px;
                margin-bottom: 24px;
            }
        }
        .section-title {
            font-size: 0.9em;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .profile-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .profile-toggles {
            display: flex;
            gap: 12px;
            font-size: 0.85em;
        }
        .profile-toggle {
            display: flex;
            align-items: center;
            gap: 4px;
            cursor: pointer;
        }
        .profile-toggle input { cursor: pointer; }
        .profile-toggle label { cursor: pointer; color: var(--text-muted); }
        .main-profile-container {
            position: relative;
            width: 100%;
            min-height: 120px;
            background: #fafafa;
            border-radius: 8px;
            overflow: hidden;
        }
        .main-profile-container img {
            width: 100%;
            height: auto;
            display: block;
            /* Prevent Safari long-press context menu */
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
        }
        .main-profile-container img.loading {
            display: none;
        }
        .elevation-loading {
            display: flex;
            align-items: center;
            justify-content: center;
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: #f8f8f8;
        }
        .elevation-loading.hidden { display: none; }
        .elevation-spinner {
            width: 32px;
            height: 32px;
            border: 3px solid #e0e0e0;
            border-top-color: #666;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .elevation-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            white-space: nowrap;
            z-index: 100;
            transform: translateX(-50%);
        }
        .elevation-tooltip.visible { opacity: 1; }
        .elevation-tooltip .grade { font-weight: bold; font-size: 14px; }
        .elevation-tooltip .elev { color: #ccc; margin-top: 2px; }
        .elevation-cursor {
            position: absolute;
            top: 0; bottom: 0;
            width: 1px;
            background: rgba(0, 0, 0, 0.5);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
        }
        .elevation-cursor.visible { opacity: 1; }
        .elevation-selection {
            position: absolute;
            top: 0; bottom: 0;
            background: rgba(59, 130, 246, 0.2);
            border-left: 2px solid rgba(59, 130, 246, 0.6);
            border-right: 2px solid rgba(59, 130, 246, 0.6);
            pointer-events: none;
            opacity: 0;
            z-index: 50;
        }
        .elevation-selection.visible { opacity: 1; }
        .elevation-selection-popup {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 200;
            white-space: nowrap;
            transform: translateX(-50%);
            pointer-events: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .elevation-selection-popup .selection-close {
            position: absolute;
            top: 2px; right: 6px;
            cursor: pointer;
            color: #888;
            font-size: 14px;
            line-height: 1;
        }
        .elevation-selection-popup .selection-close:hover { color: white; }
        .elevation-selection-popup .selection-stat {
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        .elevation-selection-popup .stat-label { color: #aaa; }
        .elevation-selection-popup .stat-value { font-weight: bold; }
        .selection-zoom-btn {
            margin-top: 8px;
            padding: 6px 12px;
            background: #3b82f6;
            color: white;
            border-radius: 4px;
            text-align: center;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }
        .selection-zoom-btn:hover { background: #2563eb; }
        .zoom-out-link {
            position: absolute;
            bottom: 2px;
            right: 10%;
            z-index: 100;
        }
        .zoom-out-link a {
            color: #3b82f6;
            font-size: 12px;
            text-decoration: none;
        }
        .zoom-out-link a:hover { text-decoration: underline; }

        /* Long-press indicator for touch selection */
        .long-press-indicator {
            position: absolute;
            width: 60px;
            height: 60px;
            margin-left: -30px;
            margin-top: -30px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s ease-out;
            z-index: 100;
        }
        .long-press-indicator.active {
            opacity: 1;
        }
        .long-press-ring {
            width: 100%;
            height: 100%;
            border: 3px solid var(--primary-color, #2196F3);
            border-radius: 50%;
            animation: long-press-pulse 0.4s ease-out forwards;
            box-sizing: border-box;
        }
        @keyframes long-press-pulse {
            0% {
                transform: scale(0.3);
                opacity: 0.8;
            }
            100% {
                transform: scale(1);
                opacity: 0;
                border-width: 2px;
            }
        }

        .climb-profile-container {
            position: relative;
            width: 100%;
            min-height: 120px;
            background: #fafafa;
            border-radius: 8px;
            overflow: hidden;
        }
        .climb-profile-container img {
            width: 100%;
            height: auto;
            display: block;
            /* Prevent Safari long-press context menu */
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
        }
        .climb-profile-container img.loading {
            display: none;
        }
        .sensitivity-control { margin-top: 16px; }
        .sensitivity-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .sensitivity-slider {
            width: 100%;
            height: 32px;
            -webkit-appearance: none;
            appearance: none;
            background: #e0e0e0;
            border-radius: 16px;
            outline: none;
            cursor: pointer;
        }
        .sensitivity-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 28px; height: 28px;
            background: var(--climb-green);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .sensitivity-slider::-moz-range-thumb {
            width: 28px; height: 28px;
            background: var(--climb-green);
            border-radius: 50%;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .climb-section {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        @media (min-width: 768px) {
            .climb-section {
                padding: 24px;
                margin-bottom: 24px;
            }
        }
        .climb-count { font-size: 0.85em; color: var(--text-muted); margin-left: 8px; }
        .climb-list { margin-top: 12px; }
        .climb-row {
            display: flex;
            align-items: flex-start;
            padding: 12px;
            margin-bottom: 8px;
            background: var(--bg-light);
            border-radius: 8px;
            border-left: 4px solid var(--climb-green);
        }
        .climb-number {
            width: 32px; height: 32px;
            background: var(--climb-green);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9em;
            flex-shrink: 0;
            margin-right: 12px;
        }
        .climb-details { flex: 1; }
        .climb-name { font-weight: 600; margin-bottom: 6px; }
        .climb-metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px 12px;
            font-size: 0.85em;
        }
        @media (min-width: 600px) {
            .climb-metrics {
                grid-template-columns: repeat(4, 1fr);
                gap: 8px 16px;
            }
        }
        .climb-metric { display: flex; gap: 8px; }
        .metric-label { color: var(--text-muted); }
        .metric-label::after { content: ':'; }
        .metric-value { font-weight: 500; }
        .no-climbs {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-style: italic;
        }
        .loading { opacity: 0.6; pointer-events: none; }
        /* Desktop layout: side-by-side sections */
        @media (min-width: 900px) {
            .desktop-grid {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 24px;
                align-items: start;
            }
            .desktop-sidebar {
                position: sticky;
                top: 24px;
            }
        }
        /* Header styles */
        .header-section {
            margin-bottom: 16px;
            text-align: center;
        }
        @media (min-width: 768px) {
            .header-section {
                margin-bottom: 24px;
            }
            .header-section h1 {
                font-size: 1.5em;
            }
        }
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 6px;
        }
        .logo {
            width: 40px;
            height: 40px;
        }
        .header-section h1 {
            margin: 0;
            font-size: 1.3em;
            background: linear-gradient(135deg, #FF6B35, #F7931E);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        @media (max-width: 480px) {
            .logo { width: 32px; height: 32px; }
            .header-section h1 { font-size: 1.1em; }
        }
        .tagline {
            color: var(--text-muted);
            font-size: 0.8em;
            margin: 0;
            line-height: 1.3;
        }
        .back-link {
            display: inline-block;
            margin-top: 8px;
            color: var(--primary);
            text-decoration: none;
            font-size: 13px;
        }
        .back-link:hover { text-decoration: underline; }
        /* Footer styles */
        .footer {
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.8em;
            color: #888;
        }
        .footer a {
            color: var(--primary);
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .footer-links {
            display: flex;
            gap: 16px;
        }
        .footer-version {
            color: #aaa;
            font-size: 0.9em;
        }
        .footer-copyright {
            color: #aaa;
            font-size: 0.9em;
        }
        @media (max-width: 480px) {
            .footer-content {
                flex-direction: column;
                gap: 6px;
            }
        }
    </style>
    {% if umami_website_id %}
    <script defer src="{{ umami_script_url }}" data-website-id="{{ umami_website_id }}"></script>
    {% endif %}
</head>
<body>
    <div class="header-section">
        <div class="logo-container">
            <svg class="logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="mountainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#FF6B35"/>
                        <stop offset="100%" style="stop-color:#F7931E"/>
                    </linearGradient>
                </defs>
                <path d="M0 85 L25 45 L40 60 L60 30 L80 50 L100 85 Z" fill="url(#mountainGrad)"/>
                <path d="M60 30 L67 42 L53 42 Z" fill="white" opacity="0.85"/>
                <path d="M25 45 L30 52 L20 52 Z" fill="white" opacity="0.7"/>
                <g transform="translate(18, 50) rotate(-20) scale(1.5)">
                    <circle cx="0" cy="14" r="7" fill="none" stroke="#2D3047" stroke-width="1.5"/>
                    <circle cx="22" cy="14" r="7" fill="none" stroke="#2D3047" stroke-width="1.5"/>
                    <path d="M0 14 L8 6 L18 6 L22 14 M8 6 L11 14 L18 6 M11 14 L0 14"
                          fill="none" stroke="#2D3047" stroke-width="1.5" stroke-linejoin="round"/>
                    <line x1="8" y1="6" x2="7" y2="3" stroke="#2D3047" stroke-width="1.5"/>
                    <line x1="7" y1="3" x2="14" y2="-1" stroke="#2D3047" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="16" cy="-2" r="2.5" fill="#2D3047"/>
                    <line x1="12" y1="-1" x2="18" y2="5" stroke="#2D3047" stroke-width="1.5" stroke-linecap="round"/>
                </g>
            </svg>
            <h1>Reality Check my Route</h1>
        </div>
        <p class="tagline">Climb Details</p>
        <a href="#" class="back-link" id="backLink" onclick="goBack(); return false;">&larr; Back to Analysis</a>
    </div>

    <div class="summary-card">
        <div class="route-name">{{ route_name or "Route" }}</div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{{ time_str }}</div>
                <div class="summary-label">Est. Time</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{{ "%.0f"|format(work_kj) }} kJ</div>
                <div class="summary-label">Work</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" id="summaryDistance" data-km="{{ distance_km }}">{{ "%.1f"|format(distance_km) }} km</div>
                <div class="summary-label">Distance</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" id="summaryElevation" data-m="{{ elevation_m }}">{{ "%.0f"|format(elevation_m) }} m</div>
                <div class="summary-label">Elevation</div>
            </div>
        </div>
    </div>

    <div class="elevation-section">
        <div class="profile-header">
            <div class="section-title">Elevation Profile</div>
            <div class="profile-toggles">
                <div class="profile-toggle">
                    <input type="checkbox" id="overlay_speed" onchange="toggleOverlay('speed')">
                    <label for="overlay_speed">Speed</label>
                </div>
                <div class="profile-toggle">
                    <input type="checkbox" id="overlay_gravel" onchange="toggleOverlay('gravel')">
                    <label for="overlay_gravel">Unpaved</label>
                </div>
                <div class="profile-toggle">
                    <input type="checkbox" id="imperial" {{ 'checked' if imperial else '' }} onchange="toggleImperial()">
                    <label for="imperial">Imperial</label>
                </div>
            </div>
        </div>
        <div class="main-profile-container" id="elevationContainer"
             data-base-profile-url="/elevation-profile?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&smoothing={{ smoothing }}"
             data-base-data-url="/elevation-profile-data?url={{ url|urlencode }}&climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&smoothing={{ smoothing }}">
            <div class="elevation-loading" id="elevationLoading">
                <div class="elevation-spinner"></div>
            </div>
            <img src="" alt="Elevation Profile" id="elevationImg" class="loading">
            <div class="elevation-cursor" id="elevationCursor"></div>
            <div class="elevation-tooltip" id="elevationTooltip">
                <div class="grade">--</div>
                <div class="elev">--</div>
            </div>
            <div class="elevation-selection" id="elevationSelection"></div>
            <div class="elevation-selection-popup" id="elevationSelectionPopup" style="display: none;"></div>
            <div class="zoom-out-link" id="zoomOutLink" style="display: none;"><a href="#" onclick="return false;">Zoom Out</a></div>
        </div>
    </div>

    <div class="elevation-section">
        <div class="section-title">Climb Detection</div>
        <div class="climb-profile-container" id="climbProfileContainer">
            <div class="elevation-loading" id="climbLoading">
                <div class="elevation-spinner"></div>
            </div>
            <img id="climbProfileImage" src="" alt="Climb Profile" class="loading">
        </div>
        <div class="sensitivity-control">
            <div class="sensitivity-label">
                <span>High Sensitivity</span>
                <span>Low Sensitivity</span>
            </div>
            <input type="range" class="sensitivity-slider" id="sensitivitySlider"
                   min="0" max="100" step="10" value="{{ sensitivity }}"
                   aria-label="Climb detection sensitivity">
        </div>
    </div>

    <div class="climb-section">
        <div class="section-title">
            Detected Climbs
            <span class="climb-count" id="climbCount">({{ climbs|length }})</span>
        </div>
        <div class="climb-list" id="climbList">
            {% if climbs %}
                {% for climb in climbs %}
                <div class="climb-row">
                    <div class="climb-number">{{ climb.climb_id }}</div>
                    <div class="climb-details">
                        <div class="climb-name">{{ climb.label }}</div>
                        <div class="climb-metrics">
                            <div class="climb-metric">
                                <span class="metric-label">Distance</span>
                                <span class="metric-value">{% if imperial %}{{ "%.1f"|format(climb.distance_m / 1000 * 0.621371) }} mi{% else %}{{ "%.1f"|format(climb.distance_m / 1000) }} km{% endif %}</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Gain</span>
                                <span class="metric-value">{% if imperial %}{{ "%.0f"|format(climb.elevation_gain * 3.28084) }} ft{% else %}{{ "%.0f"|format(climb.elevation_gain) }} m{% endif %}</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Avg Grade</span>
                                <span class="metric-value">{{ "%.1f"|format(climb.avg_grade) }}%</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Max Grade</span>
                                <span class="metric-value">{{ "%.1f"|format(climb.max_grade) }}%</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Duration</span>
                                <span class="metric-value">{% if climb.duration_seconds >= 3600 %}{{ (climb.duration_seconds // 3600)|int }}h {{ ((climb.duration_seconds % 3600) // 60)|int }}m{% else %}{{ "%.0f"|format(climb.duration_seconds / 60) }} min{% endif %}</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Work</span>
                                <span class="metric-value">{{ "%.0f"|format(climb.work_kj) }} kJ</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Avg Power</span>
                                <span class="metric-value">{{ "%.0f"|format(climb.avg_power) }} W</span>
                            </div>
                            <div class="climb-metric">
                                <span class="metric-label">Avg Speed</span>
                                <span class="metric-value">{% if imperial %}{{ "%.1f"|format(climb.avg_speed_kmh * 0.621371) }} mph{% else %}{{ "%.1f"|format(climb.avg_speed_kmh) }} km/h{% endif %}</span>
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-climbs">No significant climbs detected</div>
            {% endif %}
        </div>
    </div>

    <script>
        const routeUrl = "{{ url }}";
        const baseParams = "climbing_power={{ climbing_power }}&flat_power={{ flat_power }}&mass={{ mass }}&headwind={{ headwind }}&smoothing={{ smoothing }}";

        // Climb detection
        const slider = document.getElementById('sensitivitySlider');
        const climbProfileImage = document.getElementById('climbProfileImage');
        const climbProfileContainer = document.getElementById('climbProfileContainer');
        const climbList = document.getElementById('climbList');
        const climbCount = document.getElementById('climbCount');
        let debounceTimer = null;
        let resizeTimer = null;
        let lastAspectRatio = null;

        // Calculate aspect ratio based on container width
        function getAspectRatio() {
            const container = climbProfileContainer || document.getElementById('elevationContainer');
            if (!container) return 1;
            const width = container.offsetWidth;
            // Mobile portrait: square (1:1)
            if (width < 500) return 1;
            // Mobile landscape / tablet: 2:1
            if (width < 800) return 2;
            // Desktop: wider ratio based on actual width
            // Height stays constant, so aspect = width / fixed_height
            // Use width / 300 to get a good ratio (300px is a reasonable chart height)
            return Math.min(3.5, width / 250);
        }

        function formatClimbDuration(seconds) {
            const totalMins = Math.floor(seconds / 60);
            if (totalMins >= 60) {
                const hours = Math.floor(totalMins / 60);
                const mins = totalMins % 60;
                return hours + 'h ' + mins + 'm';
            }
            return totalMins + ' min';
        }

        function isImperial() {
            return document.getElementById('imperial')?.checked || false;
        }

        function formatDistance(km) {
            if (isImperial()) return (km * 0.621371).toFixed(1) + ' mi';
            return km.toFixed(1) + ' km';
        }

        function formatElevation(m) {
            if (isImperial()) return Math.round(m * 3.28084) + ' ft';
            return Math.round(m) + ' m';
        }

        function formatSpeed(kmh) {
            if (isImperial()) return (kmh * 0.621371).toFixed(1) + ' mph';
            return kmh.toFixed(1) + ' km/h';
        }

        // Overlay params helper (defined early for use in profile updates)
        function _buildOverlayParams() {
            let params = '';
            if (document.getElementById('overlay_speed')?.checked) params += '&overlay=speed';
            if (document.getElementById('overlay_gravel')?.checked) params += '&show_gravel=true';
            if (isImperial()) params += '&imperial=true';
            return params;
        }

        function renderClimbTable(climbs) {
            if (climbs.length === 0) {
                climbList.innerHTML = '<div class="no-climbs">No significant climbs detected</div>';
                climbCount.textContent = '(0)';
                return;
            }
            climbCount.textContent = '(' + climbs.length + ')';
            let html = '';
            climbs.forEach(climb => {
                html += `<div class="climb-row">
                    <div class="climb-number">${climb.climb_id}</div>
                    <div class="climb-details">
                        <div class="climb-name">${climb.label}</div>
                        <div class="climb-metrics">
                            <div class="climb-metric"><span class="metric-label">Distance</span><span class="metric-value">${formatDistance(climb.distance_m / 1000)}</span></div>
                            <div class="climb-metric"><span class="metric-label">Gain</span><span class="metric-value">${formatElevation(climb.elevation_gain)}</span></div>
                            <div class="climb-metric"><span class="metric-label">Avg Grade</span><span class="metric-value">${climb.avg_grade.toFixed(1)}%</span></div>
                            <div class="climb-metric"><span class="metric-label">Max Grade</span><span class="metric-value">${climb.max_grade.toFixed(1)}%</span></div>
                            <div class="climb-metric"><span class="metric-label">Duration</span><span class="metric-value">${formatClimbDuration(climb.duration_seconds)}</span></div>
                            <div class="climb-metric"><span class="metric-label">Work</span><span class="metric-value">${climb.work_kj.toFixed(0)} kJ</span></div>
                            <div class="climb-metric"><span class="metric-label">Avg Power</span><span class="metric-value">${climb.avg_power.toFixed(0)} W</span></div>
                            <div class="climb-metric"><span class="metric-label">Avg Speed</span><span class="metric-value">${formatSpeed(climb.avg_speed_kmh)}</span></div>
                        </div>
                    </div>
                </div>`;
            });
            climbList.innerHTML = html;
        }

        function updateClimbs(forceRefresh = false) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const sensitivity = slider.value;
                const aspect = getAspectRatio();
                // Only refresh image if aspect ratio changed significantly or forced
                if (forceRefresh || lastAspectRatio === null || Math.abs(aspect - lastAspectRatio) > 0.1) {
                    lastAspectRatio = aspect;
                    const climbLoading = document.getElementById('climbLoading');
                    if (climbLoading) climbLoading.classList.remove('hidden');
                    climbProfileImage.classList.add('loading');
                    climbProfileImage.src = `/elevation-profile-ride?url=${encodeURIComponent(routeUrl)}&${baseParams}&sensitivity=${sensitivity}&aspect=${aspect.toFixed(2)}`;
                    climbProfileImage.onload = () => {
                        climbProfileImage.classList.remove('loading');
                        if (climbLoading) climbLoading.classList.add('hidden');
                    };
                }
                fetch(`/api/detect-climbs?url=${encodeURIComponent(routeUrl)}&${baseParams}&sensitivity=${sensitivity}`)
                    .then(r => r.json())
                    .then(data => renderClimbTable(data.climbs))
                    .catch(err => console.error('Error fetching climbs:', err));
            }, 300);
        }
        slider.addEventListener('input', () => updateClimbs(true));

        // Update main elevation profile with current aspect ratio
        // Calculate aspect ratio for a specific container
        function getAspectForContainer(container) {
            if (!container) return 1;
            const width = container.offsetWidth;
            if (width < 500) return 1;
            if (width < 800) return 2;
            return Math.min(3.5, width / 250);
        }

        function updateMainProfile() {
            const container = document.getElementById('elevationContainer');
            const img = document.getElementById('elevationImg');
            const loading = document.getElementById('elevationLoading');
            if (!container || !img) return;
            const baseProfileUrl = container.getAttribute('data-base-profile-url');
            const baseDataUrl = container.getAttribute('data-base-data-url');
            if (!baseProfileUrl) return;
            const aspect = getAspectForContainer(container);  // Use this container's width
            const overlayParams = _buildOverlayParams();
            if (loading) loading.classList.remove('hidden');
            img.classList.add('loading');
            img.src = baseProfileUrl + `&aspect=${aspect.toFixed(2)}` + overlayParams;
            img.onload = () => {
                if (loading) loading.classList.add('hidden');
                img.classList.remove('loading');
                if (typeof setupElevationProfile === 'function' && baseDataUrl) {
                    setupElevationProfile('elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor', baseDataUrl);
                }
            };
        }

        // Handle window resize and orientation change
        function handleResize() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                const newAspect = getAspectRatio();
                if (lastAspectRatio !== null && Math.abs(newAspect - lastAspectRatio) > 0.2) {
                    updateClimbs(true);
                    updateMainProfile();
                }
            }, 250);
        }
        window.addEventListener('resize', handleResize);
        window.addEventListener('orientationchange', () => setTimeout(handleResize, 100));

        // Initial load with correct aspect ratio
        document.addEventListener('DOMContentLoaded', () => {
            updateMainProfile();
            updateClimbs(true);
        });

        // Save sensitivity to localStorage when changed
        slider.addEventListener('change', () => {
            try { localStorage.setItem('ride_sensitivity', slider.value); } catch (e) {}
        });

        // Overlay toggles
        function toggleOverlay(type) {
            const container = document.getElementById('elevationContainer');
            const img = document.getElementById('elevationImg');
            const loading = document.getElementById('elevationLoading');
            if (!container || !img) return;
            const baseProfileUrl = container.getAttribute('data-base-profile-url');
            const baseDataUrl = container.getAttribute('data-base-data-url');
            if (!baseProfileUrl) return;
            const aspect = getAspectForContainer(container);  // Use this container's width
            const overlayParams = _buildOverlayParams();

            // Preserve zoom state from container data attributes
            let zoomParams = '';
            const zoomMin = container.getAttribute('data-zoom-min');
            const zoomMax = container.getAttribute('data-zoom-max');
            if (zoomMin && zoomMax) {
                zoomParams = `&min_xlim_hours=${zoomMin}&max_xlim_hours=${zoomMax}`;
            }

            if (loading) loading.classList.remove('hidden');
            img.classList.add('loading');
            img.src = baseProfileUrl + `&aspect=${aspect.toFixed(2)}` + overlayParams + zoomParams;
            img.onload = () => {
                if (loading) loading.classList.add('hidden');
                img.classList.remove('loading');
                if (typeof setupElevationProfile === 'function' && baseDataUrl) {
                    setupElevationProfile('elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor', baseDataUrl);
                }
            };
            // Save toggle states to localStorage
            try {
                localStorage.setItem('ride_show_speed', document.getElementById('overlay_speed')?.checked || false);
                localStorage.setItem('ride_show_gravel', document.getElementById('overlay_gravel')?.checked || false);
            } catch (e) {}
        }

        function updateSummaryUnits() {
            const distEl = document.getElementById('summaryDistance');
            const elevEl = document.getElementById('summaryElevation');
            if (distEl) {
                const km = parseFloat(distEl.dataset.km);
                distEl.textContent = formatDistance(km);
            }
            if (elevEl) {
                const m = parseFloat(elevEl.dataset.m);
                elevEl.textContent = formatElevation(m);
            }
        }

        function toggleImperial() {
            const imperial = isImperial();
            try { localStorage.setItem('ride_imperial', imperial); } catch (e) {}
            updateSummaryUnits();
            // Refresh elevation profile with new y-axis units
            refreshMainProfile();
            // Re-render climb table with new units
            updateClimbs();
        }

        function refreshMainProfile() {
            const container = document.getElementById('elevationContainer');
            const img = document.getElementById('elevationImg');
            const loading = document.getElementById('elevationLoading');
            if (!container || !img) return;
            const baseProfileUrl = container.getAttribute('data-base-profile-url');
            const baseDataUrl = container.getAttribute('data-base-data-url');
            if (!baseProfileUrl) return;
            const aspect = getAspectForContainer(container);  // Use this container's width
            const overlayParams = _buildOverlayParams();
            loading.classList.remove('hidden');
            img.classList.add('loading');
            img.src = baseProfileUrl + `&aspect=${aspect.toFixed(2)}` + overlayParams;
            img.onload = () => {
                loading.classList.add('hidden');
                img.classList.remove('loading');
                if (typeof setupElevationProfile === 'function' && baseDataUrl) {
                    setupElevationProfile('elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor', baseDataUrl);
                }
            };
        }

        function goBack() {
            let url = '/?url=' + encodeURIComponent(routeUrl) + '&' + baseParams;
            // Add settings from localStorage
            try {
                if (localStorage.getItem('overlay_speed') === 'true') url += '&overlay=speed';
                if (localStorage.getItem('overlay_gravel') === 'true') url += '&show_gravel=true';
                if (localStorage.getItem('ride_imperial') === 'true') url += '&imperial=1';
            } catch (e) {}
            window.location.href = url;
        }

        // Elevation profile interaction
        function setupElevationProfile(containerId, imgId, tooltipId, cursorId, dataUrl) {
            const container = document.getElementById(containerId);
            const img = document.getElementById(imgId);
            const tooltip = document.getElementById(tooltipId);
            const cursor = document.getElementById(cursorId);
            if (!container || !img || !tooltip || !cursor) return;

            const selection = document.getElementById('elevationSelection');
            const selectionPopup = document.getElementById('elevationSelectionPopup');

            let profileData = null;
            let selectionStart = null;
            let isSelecting = false;
            let selectionActive = false;

            // Zoom state - restore from container data attributes if present
            const savedZoomMin = container.getAttribute('data-zoom-min');
            const savedZoomMax = container.getAttribute('data-zoom-max');
            let isZoomed = !!(savedZoomMin && savedZoomMax);
            let zoomMinHours = savedZoomMin ? parseFloat(savedZoomMin) : null;
            let zoomMaxHours = savedZoomMax ? parseFloat(savedZoomMax) : null;
            let selectionStartTime = null;
            let selectionEndTime = null;
            const zoomOutLink = document.getElementById('zoomOutLink');

            // Clear any stale popup and selection from previous initialization
            if (selectionPopup) { selectionPopup.style.display = 'none'; selectionPopup.innerHTML = ''; }
            if (selection) selection.classList.remove('visible');

            // Show/hide zoom out link based on restored state
            if (zoomOutLink) {
                zoomOutLink.style.display = isZoomed ? 'block' : 'none';
                if (isZoomed) {
                    zoomOutLink.querySelector('a').onclick = function(e) {
                        e.preventDefault();
                        zoomOut();
                    };
                }
            }

            fetch(dataUrl)
                .then(r => {
                    if (!r.ok) { console.error('Profile data fetch failed:', r.status, dataUrl); return { error: 'HTTP ' + r.status }; }
                    return r.json();
                })
                .then(data => { if (data && !data.error) profileData = data; else if (data?.error) console.error('Profile data error:', data.error, dataUrl); })
                .catch(err => console.error('Profile data fetch exception:', err, dataUrl));

            // Calculate plot margins based on aspect ratio
            // Margins in inches are roughly constant, so as percentage they scale with 1/aspectRatio
            // Calibrated from figsize=(14,4) where leftMargin=0.77in, rightMargin=0.18in
            function getContainerAspect() {
                // Use THIS container's width, not the global getAspectRatio which uses climbProfileContainer
                const width = container.offsetWidth;
                if (width < 500) return 1;
                if (width < 800) return 2;
                return Math.min(3.5, width / 250);
            }
            function getPlotMargins() {
                const aspect = getContainerAspect();
                return {
                    left: 0.1925 / aspect,   // 0.77in / (4in * aspect) = 0.1925/aspect
                    right: 1 - 0.045 / aspect  // 1 - 0.18in / (4in * aspect) = 1 - 0.045/aspect
                };
            }
            let plotMargins = getPlotMargins();
            function getPlotLeftPct() { return plotMargins.left; }
            function getPlotRightPct() { return plotMargins.right; }
            function updatePlotMargins() { plotMargins = getPlotMargins(); }

            function getDataAtPosition(xPct) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return null;
                const leftPct = getPlotLeftPct(), rightPct = getPlotRightPct();
                const plotXPct = (xPct - leftPct) / (rightPct - leftPct);
                if (plotXPct < 0 || plotXPct > 1) return null;
                // Use zoom bounds if zoomed, otherwise full range
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : profileData.total_time;
                const time = minTime + plotXPct * (maxTime - minTime);
                if (time > profileData.total_time || time < 0) return null;
                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) {
                        return { grade: profileData.grades[i], elevation: profileData.elevations[i] || 0, speed: profileData.speeds ? profileData.speeds[i] : null, time: time };
                    }
                }
                const lastIdx = profileData.grades.length - 1;
                return { grade: profileData.grades[lastIdx], elevation: profileData.elevations[lastIdx + 1] || 0, speed: profileData.speeds ? profileData.speeds[lastIdx] : null, time: time };
            }

            function getIndexAtPosition(xPct) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return -1;
                const leftPct = getPlotLeftPct(), rightPct = getPlotRightPct();
                const plotXPct = Math.max(0, Math.min(1, (xPct - leftPct) / (rightPct - leftPct)));
                // Use zoom bounds if zoomed
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : profileData.total_time;
                const time = Math.max(0, Math.min(minTime + plotXPct * (maxTime - minTime), profileData.total_time));
                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) return i;
                }
                return profileData.times.length - 2;
            }

            // Convert time to index (for selection repositioning after zoom)
            function getIndexAtTime(time) {
                if (!profileData || !profileData.times || profileData.times.length < 2) return -1;
                for (let i = 0; i < profileData.times.length - 1; i++) {
                    if (time >= profileData.times[i] && time < profileData.times[i + 1]) return i;
                }
                return profileData.times.length - 2;
            }

            function formatGrade(g) {
                if (g === null || g === undefined) return 'Stopped';
                return (g >= 0 ? '+' : '') + g.toFixed(1) + '%';
            }

            function formatTime(hours) {
                const h = Math.floor(hours);
                const m = Math.floor((hours - h) * 60);
                return h + 'h ' + m.toString().padStart(2, '0') + 'm';
            }

            function formatDuration(hours) {
                const totalMin = Math.round(hours * 60);
                if (totalMin < 60) return totalMin + 'min';
                return Math.floor(totalMin / 60) + 'h ' + (totalMin % 60).toString().padStart(2, '0') + 'm';
            }

            function updateTooltip(xPct, clientX) {
                const data = getDataAtPosition(xPct);
                if (data) {
                    tooltip.querySelector('.grade').textContent = formatGrade(data.grade);
                    const imp = isImperial();
                    const elevUnit = imp ? 'ft' : 'm';
                    const elevVal = imp ? data.elevation * 3.28084 : data.elevation;
                    let speedText = '';
                    if (data.speed !== null && data.speed !== undefined) {
                        const speedUnit = imp ? 'mph' : 'km/h';
                        const speedVal = imp ? data.speed * 0.621371 : data.speed;
                        speedText = ' | ' + speedVal.toFixed(1) + ' ' + speedUnit;
                    }
                    tooltip.querySelector('.elev').textContent = Math.round(elevVal) + ' ' + elevUnit + speedText + ' | ' + formatTime(data.time);
                    const rect = img.getBoundingClientRect();
                    const xPos = clientX - rect.left;
                    tooltip.style.left = xPos + 'px';
                    tooltip.style.bottom = '60px';
                    tooltip.classList.add('visible');
                    cursor.style.left = xPos + 'px';
                    cursor.classList.add('visible');
                } else {
                    hideTooltip();
                }
            }

            function hideTooltip() {
                tooltip.classList.remove('visible');
                cursor.classList.remove('visible');
            }

            function updateSelectionHighlight(startXPct, endXPct) {
                if (!selection) return;
                const leftPct = getPlotLeftPct(), rightPct = getPlotRightPct();
                const clampedStart = Math.max(leftPct, Math.min(rightPct, startXPct));
                const clampedEnd = Math.max(leftPct, Math.min(rightPct, endXPct));
                selection.style.left = (Math.min(clampedStart, clampedEnd) * 100) + '%';
                selection.style.width = (Math.abs(clampedEnd - clampedStart) * 100) + '%';
                selection.classList.add('visible');
            }

            function computeSelectionStats(startIdx, endIdx) {
                if (!profileData) return null;
                const d = profileData;
                const i = Math.min(startIdx, endIdx), j = Math.max(startIdx, endIdx);
                if (i < 0 || j >= d.times.length) return null;
                const duration = d.times[j + 1 < d.times.length ? j + 1 : j] - d.times[i];
                let distM = 0;
                if (d.distances) for (let k = i; k <= j; k++) distM += (d.distances[k] || 0);
                let elevGain = 0, elevLoss = 0;
                if (d.elev_gains && d.elev_losses) {
                    for (let k = i; k <= j; k++) { elevGain += (d.elev_gains[k] || 0); elevLoss += (d.elev_losses[k] || 0); }
                } else {
                    for (let k = i; k <= j; k++) {
                        const diff = (d.elevations[k + 1] !== undefined ? d.elevations[k + 1] : d.elevations[k]) - d.elevations[k];
                        if (diff > 0) elevGain += diff; else elevLoss += diff;
                    }
                }
                let gradeSum = 0, gradeCount = 0;
                for (let k = i; k <= j; k++) if (d.grades[k] !== null && d.grades[k] !== undefined) { gradeSum += d.grades[k]; gradeCount++; }
                const avgGrade = gradeCount > 0 ? gradeSum / gradeCount : null;
                const avgSpeed = (duration > 0 && distM > 0) ? (distM / 1000) / duration : null;
                let workJ = 0;
                // Use pre-computed works array (accurate through downsampling) if available
                if (d.works) {
                    for (let k = i; k <= j; k++) { workJ += (d.works[k] || 0); }
                } else if (d.powers) {
                    for (let k = i; k <= j; k++) {
                        if (d.powers[k] !== null && d.powers[k] !== undefined) {
                            workJ += d.powers[k] * ((d.times[k + 1 < d.times.length ? k + 1 : k] - d.times[k]) * 3600);
                        }
                    }
                }
                // Avg power = work / time (matches summary calculation)
                const durationSec = duration * 3600;
                const avgPower = (workJ > 0 && durationSec > 0) ? workJ / durationSec : null;
                return { duration, distKm: distM / 1000, elevGain, elevLoss, avgGrade, avgSpeed, avgPower, workKJ: workJ / 1000 };
            }

            function showSelectionPopup(stats, xPctCenter) {
                if (!selectionPopup || !stats) return;
                const imp = isImperial();
                const distVal = imp ? (stats.distKm * 0.621371) : stats.distKm;
                const distUnit = imp ? 'mi' : 'km';
                const elevGainVal = imp ? (stats.elevGain * 3.28084) : stats.elevGain;
                const elevLossVal = imp ? (stats.elevLoss * 3.28084) : stats.elevLoss;
                const elevUnit = imp ? 'ft' : 'm';
                const speedVal = stats.avgSpeed !== null ? (imp ? stats.avgSpeed * 0.621371 : stats.avgSpeed) : null;
                const speedUnit = imp ? 'mph' : 'km/h';

                let html = '<span class="selection-close">&times;</span>';
                html += '<div class="selection-stat"><span class="stat-label">Duration</span><span class="stat-value">' + formatDuration(stats.duration) + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Distance</span><span class="stat-value">' + distVal.toFixed(1) + ' ' + distUnit + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Elev Gain</span><span class="stat-value">+' + Math.round(elevGainVal) + ' ' + elevUnit + '</span></div>';
                html += '<div class="selection-stat"><span class="stat-label">Elev Loss</span><span class="stat-value">' + Math.round(elevLossVal) + ' ' + elevUnit + '</span></div>';
                if (stats.avgGrade !== null) html += '<div class="selection-stat"><span class="stat-label">Avg Grade</span><span class="stat-value">' + formatGrade(stats.avgGrade) + '</span></div>';
                if (speedVal !== null) html += '<div class="selection-stat"><span class="stat-label">Avg Speed</span><span class="stat-value">' + speedVal.toFixed(1) + ' ' + speedUnit + '</span></div>';
                if (stats.avgPower !== null) {
                    html += '<div class="selection-stat"><span class="stat-label">Avg Power</span><span class="stat-value">' + Math.round(stats.avgPower) + ' W</span></div>';
                    html += '<div class="selection-stat"><span class="stat-label">Work</span><span class="stat-value">' + stats.workKJ.toFixed(1) + ' kJ</span></div>';
                }
                // Add zoom button
                html += '<div class="selection-zoom-btn">' + (isZoomed ? 'Zoom Out' : 'Zoom In') + '</div>';
                selectionPopup.innerHTML = html;
                selectionPopup.style.display = 'block';
                selectionPopup.style.left = (xPctCenter * 100) + '%';
                selectionPopup.style.bottom = '70px';
                selectionActive = true;
                selectionPopup.querySelector('.selection-close')?.addEventListener('click', (ev) => { ev.stopPropagation(); clearSelection(); });
                // Zoom button handler
                var zoomBtn = selectionPopup.querySelector('.selection-zoom-btn');
                if (zoomBtn) {
                    zoomBtn.addEventListener('click', function(ev) {
                        ev.stopPropagation();
                        if (isZoomed) {
                            zoomOut();
                        } else {
                            zoomIn();
                        }
                    });
                }
            }

            function clearSelection() {
                if (selection) selection.classList.remove('visible');
                if (selectionPopup) { selectionPopup.style.display = 'none'; selectionPopup.innerHTML = ''; }
                selectionStart = null; isSelecting = false; selectionActive = false;
            }

            function zoomIn() {
                if (!profileData || !selection) return;

                // Get selection bounds in pixel percentages
                const selLeft = parseFloat(selection.style.left) / 100;
                const selWidth = parseFloat(selection.style.width) / 100;

                // Convert to time using current zoom state
                const leftPct = getPlotLeftPct(), rightPct = getPlotRightPct();
                const plotRange = rightPct - leftPct;
                const minTime = isZoomed ? zoomMinHours : 0;
                const maxTime = isZoomed ? zoomMaxHours : profileData.total_time;
                const viewRange = maxTime - minTime;

                const startPct = (selLeft - leftPct) / plotRange;
                const endPct = (selLeft + selWidth - leftPct) / plotRange;

                selectionStartTime = minTime + startPct * viewRange;
                selectionEndTime = minTime + endPct * viewRange;

                // Calculate zoom bounds with padding so selection takes 90% of view
                const selectionDuration = selectionEndTime - selectionStartTime;
                const totalZoomRange = selectionDuration / 0.9;
                const padding = (totalZoomRange - selectionDuration) / 2;

                zoomMinHours = Math.max(0, selectionStartTime - padding);
                zoomMaxHours = Math.min(profileData.total_time, selectionEndTime + padding);

                // Store zoom state on container for persistence across overlay toggles
                container.setAttribute('data-zoom-min', zoomMinHours.toFixed(4));
                container.setAttribute('data-zoom-max', zoomMaxHours.toFixed(4));

                isZoomed = true;
                refreshZoomedProfile();
            }

            function zoomOut() {
                isZoomed = false;
                zoomMinHours = null;
                zoomMaxHours = null;
                selectionStartTime = null;
                selectionEndTime = null;

                // Clear zoom state from container
                container.removeAttribute('data-zoom-min');
                container.removeAttribute('data-zoom-max');

                clearSelection();
                refreshZoomedProfile();
            }

            function refreshZoomedProfile() {
                var baseProfileUrl = container.getAttribute('data-base-profile-url');
                var loading = document.getElementById('elevationLoading');

                if (!baseProfileUrl) return;

                // Build URL parameters
                var params = '';
                var aspect = getContainerAspect();
                params += '&aspect=' + aspect.toFixed(2);

                // Add overlay parameters (must match _buildOverlayParams format)
                if (document.getElementById('overlay_speed')?.checked) params += '&overlay=speed';
                if (document.getElementById('overlay_gravel')?.checked) params += '&show_gravel=true';
                if (isImperial()) params += '&imperial=true';

                // Add zoom parameters
                if (isZoomed && zoomMinHours !== null && zoomMaxHours !== null) {
                    params += '&min_xlim_hours=' + zoomMinHours.toFixed(4);
                    params += '&max_xlim_hours=' + zoomMaxHours.toFixed(4);
                }

                // Show loading spinner
                if (loading) loading.classList.remove('hidden');
                img.classList.add('loading');

                // Update image
                img.src = baseProfileUrl + params;

                img.onload = function() {
                    if (loading) loading.classList.add('hidden');
                    img.classList.remove('loading');
                    updatePlotMargins();

                    // Show/hide zoom out link
                    if (zoomOutLink) {
                        zoomOutLink.style.display = isZoomed ? 'block' : 'none';
                        if (isZoomed) {
                            zoomOutLink.querySelector('a').onclick = function(e) {
                                e.preventDefault();
                                zoomOut();
                            };
                        }
                    }

                    // Reposition selection if zoomed
                    if (isZoomed && selectionStartTime !== null && selectionEndTime !== null) {
                        repositionSelectionAfterZoom();
                    }
                };
            }

            function repositionSelectionAfterZoom() {
                if (!selection || !isZoomed || selectionStartTime === null || selectionEndTime === null) return;

                // Calculate the new position of selection in zoomed view
                const zoomRange = zoomMaxHours - zoomMinHours;
                const leftPct = getPlotLeftPct(), rightPct = getPlotRightPct();
                const plotRange = rightPct - leftPct;

                // Convert time to plot percentage
                const startPct = leftPct + ((selectionStartTime - zoomMinHours) / zoomRange) * plotRange;
                const endPct = leftPct + ((selectionEndTime - zoomMinHours) / zoomRange) * plotRange;

                // Clamp to plot boundaries
                const clampedStart = Math.max(leftPct, Math.min(rightPct, startPct));
                const clampedEnd = Math.max(leftPct, Math.min(rightPct, endPct));

                selection.style.left = (Math.min(clampedStart, clampedEnd) * 100) + '%';
                selection.style.width = (Math.abs(clampedEnd - clampedStart) * 100) + '%';
                selection.classList.add('visible');

                // Re-show popup with Zoom Out button
                const centerXPct = (clampedStart + clampedEnd) / 2;
                const startIdx = getIndexAtTime(selectionStartTime);
                const endIdx = getIndexAtTime(selectionEndTime);
                const stats = computeSelectionStats(startIdx, endIdx);
                if (stats) {
                    showSelectionPopup(stats, centerXPct);
                }
            }

            function onMouseMove(e) {
                if (!profileData) return;
                const rect = img.getBoundingClientRect();
                const xPct = (e.clientX - rect.left) / rect.width;
                if (isSelecting) { updateSelectionHighlight(selectionStart, xPct); updateTooltip(xPct, e.clientX); }
                else if (!selectionActive) updateTooltip(xPct, e.clientX);
            }

            function onMouseDown(e) {
                if (!profileData) return;
                // If popup is showing and click is outside popup, dismiss popup but keep selection if zoomed
                if (selectionActive) {
                    // Don't dismiss if clicking inside the popup
                    if (selectionPopup && selectionPopup.contains(e.target)) {
                        return;
                    }
                    if (selectionPopup) {
                        selectionPopup.style.display = 'none';
                        selectionPopup.innerHTML = '';
                    }
                    selectionActive = false;
                    // Keep selection visible if zoomed
                    if (!isZoomed && selection) {
                        selection.classList.remove('visible');
                    }
                    return;
                }
                const rect = img.getBoundingClientRect();
                selectionStart = (e.clientX - rect.left) / rect.width;
                isSelecting = true;
                e.preventDefault();
            }

            function onMouseUp(e) {
                if (!isSelecting) return;
                isSelecting = false;
                const rect = img.getBoundingClientRect();
                const xPctEnd = (e.clientX - rect.left) / rect.width;
                const dragPx = Math.abs(e.clientX - rect.left - selectionStart * rect.width);
                if (dragPx > 5) {
                    const startIdx = getIndexAtPosition(selectionStart);
                    const endIdx = getIndexAtPosition(xPctEnd);
                    if (startIdx >= 0 && endIdx >= 0 && startIdx !== endIdx) {
                        const stats = computeSelectionStats(startIdx, endIdx);
                        hideTooltip();
                        showSelectionPopup(stats, (selectionStart + xPctEnd) / 2);
                    } else clearSelection();
                } else clearSelection();
            }

            function onMouseLeave(e) { if (isSelecting) onMouseUp(e); hideTooltip(); }

            // Long-press touch selection
            const LONG_PRESS_DURATION = 400;  // ms to trigger long-press
            const LONG_PRESS_MOVE_THRESHOLD = 15;  // px movement allowed during long-press wait
            let longPressTimer = null;
            let longPressStartX = 0;
            let longPressStartY = 0;
            let longPressPending = false;

            // Create long-press indicator element
            let longPressIndicator = container.querySelector('.long-press-indicator');
            if (!longPressIndicator) {
                longPressIndicator = document.createElement('div');
                longPressIndicator.className = 'long-press-indicator';
                longPressIndicator.innerHTML = '<div class="long-press-ring"></div>';
                container.appendChild(longPressIndicator);
            }

            function showLongPressIndicator(clientX, clientY) {
                const rect = container.getBoundingClientRect();
                longPressIndicator.style.left = (clientX - rect.left) + 'px';
                longPressIndicator.style.top = (clientY - rect.top) + 'px';
                longPressIndicator.classList.add('active');
            }

            function hideLongPressIndicator() {
                longPressIndicator.classList.remove('active');
            }

            function triggerHapticFeedback() {
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
            }

            function cancelLongPress() {
                if (longPressTimer) {
                    clearTimeout(longPressTimer);
                    longPressTimer = null;
                }
                longPressPending = false;
                hideLongPressIndicator();
            }

            function onTouchStart(e) {
                if (!profileData) return;
                // If popup is showing, check if tap is inside popup
                if (selectionActive) {
                    // Don't dismiss if tapping inside the popup
                    if (selectionPopup && e.touches[0] && selectionPopup.contains(document.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY))) {
                        return;
                    }
                    if (selectionPopup) {
                        selectionPopup.style.display = 'none';
                        selectionPopup.innerHTML = '';
                    }
                    selectionActive = false;
                    // Keep selection visible if zoomed
                    if (!isZoomed && selection) {
                        selection.classList.remove('visible');
                    }
                    return;
                }

                const touch = e.touches[0];
                longPressStartX = touch.clientX;
                longPressStartY = touch.clientY;
                longPressPending = true;

                // Show visual indicator immediately
                showLongPressIndicator(touch.clientX, touch.clientY);

                // Start long-press timer
                longPressTimer = setTimeout(() => {
                    if (!longPressPending) return;
                    longPressPending = false;
                    hideLongPressIndicator();

                    // Trigger haptic feedback
                    triggerHapticFeedback();

                    // Enter selection mode
                    const rect = img.getBoundingClientRect();
                    selectionStart = (touch.clientX - rect.left) / rect.width;
                    isSelecting = true;

                    // Show initial selection highlight at touch point
                    updateSelectionHighlight(selectionStart, selectionStart);
                }, LONG_PRESS_DURATION);
            }

            function onTouchMove(e) {
                if (!profileData) return;

                const touch = e.touches[0];

                // If waiting for long-press, check if moved too much
                if (longPressPending) {
                    const dx = touch.clientX - longPressStartX;
                    const dy = touch.clientY - longPressStartY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance > LONG_PRESS_MOVE_THRESHOLD) {
                        // User is scrolling, cancel long-press
                        cancelLongPress();
                        return;
                    }
                    // Still waiting for long-press, don't prevent default (allow scroll)
                    return;
                }

                // In selection mode - prevent scrolling and update selection
                if (isSelecting) {
                    e.preventDefault();
                    const rect = img.getBoundingClientRect();
                    const xPct = (touch.clientX - rect.left) / rect.width;
                    updateSelectionHighlight(selectionStart, xPct);
                    updateTooltip(xPct, touch.clientX);
                }
            }

            function onTouchEnd(e) {
                // Cancel any pending long-press
                if (longPressPending) {
                    cancelLongPress();
                    hideTooltip();
                    return;
                }

                if (!isSelecting) { hideTooltip(); return; }
                isSelecting = false;
                if (!selection) { hideTooltip(); return; }

                const rect = img.getBoundingClientRect();
                const selLeft = parseFloat(selection.style.left) / 100;
                const selWidth = parseFloat(selection.style.width) / 100;
                if (selWidth * rect.width > 30) {
                    const startIdx = getIndexAtPosition(selLeft);
                    const endIdx = getIndexAtPosition(selLeft + selWidth);
                    if (startIdx >= 0 && endIdx >= 0 && startIdx !== endIdx) {
                        const stats = computeSelectionStats(startIdx, endIdx);
                        hideTooltip();
                        showSelectionPopup(stats, selLeft + selWidth / 2);
                    } else clearSelection();
                } else clearSelection();
                hideTooltip();
            }

            function onTouchCancel(e) {
                cancelLongPress();
                if (isSelecting) {
                    isSelecting = false;
                    clearSelection();
                }
                hideTooltip();
            }

            if (container._profileCleanup) container._profileCleanup();
            const ac = new AbortController();
            container.addEventListener('mousemove', onMouseMove, { signal: ac.signal });
            container.addEventListener('mousedown', onMouseDown, { signal: ac.signal });
            container.addEventListener('mouseup', onMouseUp, { signal: ac.signal });
            container.addEventListener('mouseleave', onMouseLeave, { signal: ac.signal });
            container.addEventListener('touchstart', onTouchStart, { passive: true, signal: ac.signal });
            container.addEventListener('touchmove', onTouchMove, { passive: false, signal: ac.signal });
            container.addEventListener('touchend', onTouchEnd, { passive: true, signal: ac.signal });
            container.addEventListener('touchcancel', onTouchCancel, { passive: true, signal: ac.signal });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape') clearSelection(); }, { signal: ac.signal });
            container._profileCleanup = () => ac.abort();
        }

        // Initialize elevation profile
        const container = document.getElementById('elevationContainer');
        const dataUrl = container?.getAttribute('data-base-data-url');
        if (dataUrl) setupElevationProfile('elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor', dataUrl);

        // Load saved settings from localStorage
        function initSavedSettings() {
            try {
                // Load speed toggle
                const savedSpeed = localStorage.getItem('ride_show_speed');
                if (savedSpeed === 'true') {
                    const speedCb = document.getElementById('overlay_speed');
                    if (speedCb) speedCb.checked = true;
                }
                // Load gravel toggle
                const savedGravel = localStorage.getItem('ride_show_gravel');
                if (savedGravel === 'true') {
                    const gravelCb = document.getElementById('overlay_gravel');
                    if (gravelCb) gravelCb.checked = true;
                }
                // Load imperial setting (check ride_imperial first, fall back to main page's setting)
                const savedImperial = localStorage.getItem('ride_imperial');
                const imperialCb = document.getElementById('imperial');
                if (savedImperial !== null && imperialCb) {
                    imperialCb.checked = savedImperial === 'true';
                }
                // Load sensitivity slider
                const savedSensitivity = localStorage.getItem('ride_sensitivity');
                if (savedSensitivity !== null) {
                    const sensitivityVal = parseInt(savedSensitivity, 10);
                    if (!isNaN(sensitivityVal) && sensitivityVal >= 0 && sensitivityVal <= 100) {
                        slider.value = sensitivityVal;
                    }
                }
                // If any toggles were restored, update the profile image
                const speedCb = document.getElementById('overlay_speed');
                const gravelCb = document.getElementById('overlay_gravel');
                if ((speedCb?.checked) || (gravelCb?.checked)) {
                    toggleOverlay(null);
                }
                // If sensitivity was restored and differs from default, update climbs
                const savedSens = localStorage.getItem('ride_sensitivity');
                if (savedSens !== null && parseInt(savedSens, 10) !== {{ sensitivity }}) {
                    updateClimbs();
                }
                // Update units based on imperial setting
                updateSummaryUnits();
            } catch (e) {}
        }
        initSavedSettings();
    </script>

    <div class="footer">
        <div class="footer-content">
            <div class="footer-links">
                <a href="https://github.com/sanmi/gpx-analyzer" target="_blank">Source Code</a>
                <a href="https://github.com/sanmi/gpx-analyzer/issues" target="_blank">Report a Bug</a>
            </div>
            <div class="footer-version">{{ version_date }} ({{ git_hash }})</div>
            <div class="footer-copyright">© 2025 Frank San Miguel</div>
        </div>
    </div>
    {% if umami_website_id and route_name %}
    <script>
        // Track ride page view with Umami (wait for script to load)
        window.addEventListener('load', function() {
            if (typeof umami !== 'undefined') {
                umami.track('ride-view', {
                    distance_km: {{ "%.1f"|format(distance_km) if distance_km else 0 }},
                    elevation_m: {{ "%.0f"|format(elevation_m) if elevation_m else 0 }},
                    climbs: {{ climbs|length if climbs else 0 }},
                    name: '{{ route_name|replace("'", "\\'") if route_name else "" }}'
                });
            }
        });
    </script>
    {% endif %}
</body>
</html>
"""


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
    unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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

        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
        config = _load_config() or {}

        # Get profile data (uses cached route data)
        profile_data = _calculate_elevation_profile_data(url, params, smoothing)
        times_hours = profile_data["times_hours"]
        powers = profile_data.get("powers", [])

        # Get route data for climb detection
        points, route_metadata = get_route_with_surface(url, params.crr)
        points, _ = detect_and_correct_tunnels(points)

        # Process elevation
        smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
        elev_result = process_elevation_data(points, route_metadata, smoothing_radius)
        scaled_points = elev_result.scaled_points

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
    unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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
        params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
        config = _load_config() or {}

        # Get profile data
        data = _calculate_elevation_profile_data(url, params, smoothing)
        times_hours = data["times_hours"]
        elevations = data["elevations"]
        grades = data["grades"]
        powers = data.get("powers", [])

        # Get climb data
        points, route_metadata = get_route_with_surface(url, params.crr)
        points, _ = detect_and_correct_tunnels(points)
        smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
        elev_result = process_elevation_data(points, route_metadata, smoothing_radius)
        scaled_points = elev_result.scaled_points

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
    # Grade to color mapping
    main_colors = [
        '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
        '#cccccc',
        '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
    ]
    steep_colors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000']

    def grade_to_color(g):
        if g >= 10:
            if g < 12: return steep_colors[0]
            elif g < 14: return steep_colors[1]
            elif g < 16: return steep_colors[2]
            elif g < 18: return steep_colors[3]
            elif g < 20: return steep_colors[4]
            else: return steep_colors[5]
        for i, threshold in enumerate(GRADE_BINS[1:]):
            if g < threshold:
                return main_colors[i]
        return main_colors[-1]

    # Figure with dynamic aspect ratio (height fixed, width varies)
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

    # Build polygons for grade coloring
    polygons = []
    colors = []
    for i in range(len(grades)):
        t0, t1 = times_hours[i], times_hours[i+1]
        e0, e1 = elevations[i], elevations[i+1]
        polygons.append([(t0, 0), (t1, 0), (t1, e1), (t0, e0)])
        colors.append(grade_to_color(grades[i]))

    coll = PolyCollection(polygons, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(coll)

    # Add outline
    ax.plot(times_hours, elevations, color='#333333', linewidth=0.5)

    max_elev = max(elevations) * 1.15

    # Highlight climbs with green bands and numbered markers
    for climb in climbs:
        # Green band
        ax.axvspan(climb.start_time_hours, climb.end_time_hours,
                   alpha=0.2, color='#4CAF50', zorder=0.5)

        # Numbered marker at top center of band
        mid_time = (climb.start_time_hours + climb.end_time_hours) / 2
        ax.text(mid_time, max_elev * 0.92, str(climb.climb_id),
                fontsize=10, fontweight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#4CAF50', edgecolor='#388E3C', linewidth=1.5))

    # Style
    ax.set_xlim(0, times_hours[-1])
    ax.set_ylim(0, max_elev)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    _set_fixed_margins(fig, fig_width, fig_height)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
    unpaved_power_factor = float(request.args.get("unpaved_power_factor", defaults["unpaved_power_factor"]))
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
            params = build_params(climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, unpaved_power_factor)
            config = _load_config() or {}

            # Get analysis data
            result = analyze_single_route(url, params, smoothing)
            route_name = result.get("name")
            time_str = result.get("time_str", "0h 00m")
            work_kj = result.get("work_kj", 0)
            distance_km = result.get("distance_km", 0)
            elevation_m = result.get("elevation_m", 0)

            # Get profile data for climb detection
            profile_data = _calculate_elevation_profile_data(url, params, smoothing)
            times_hours = profile_data["times_hours"]
            powers = profile_data.get("powers", [])

            # Get route data for climb detection
            points, route_metadata = get_route_with_surface(url, params.crr)
            points, _ = detect_and_correct_tunnels(points)
            smoothing_radius = smoothing if smoothing is not None else config.get("smoothing", DEFAULTS["smoothing"])
            elev_result = process_elevation_data(points, route_metadata, smoothing_radius)
            scaled_points = elev_result.scaled_points

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
            )
            climbs = climb_result.climbs

        except Exception as e:
            error = str(e)
    elif url:
        error = "Invalid RideWithGPS route URL"

    return render_template_string(
        RIDE_TEMPLATE,
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
