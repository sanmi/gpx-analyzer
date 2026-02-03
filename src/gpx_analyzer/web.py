"""Simple web interface for GPX analyzer."""

import hashlib
import io
import json
import time
from collections import OrderedDict
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
    extract_privacy_code,
    get_collection_route_ids,
    get_route_with_surface,
    get_trip_data,
    is_ridewithgps_collection_url,
    is_ridewithgps_url,
    is_ridewithgps_trip_url,
    TripPoint,
)
from gpx_analyzer.smoothing import smooth_elevations


# Simple LRU cache for route analysis results
class AnalysisCache:
    """Thread-safe LRU cache for route analysis results."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, url: str, power: float, mass: float, headwind: float) -> str:
        """Create a cache key from analysis parameters."""
        key_str = f"{url}|{power}|{mass}|{headwind}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, url: str, power: float, mass: float, headwind: float) -> dict | None:
        """Get cached result, returns None if not found."""
        key = self._make_key(url, power, mass, headwind)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key][0]
            self.misses += 1
            return None

    def set(self, url: str, power: float, mass: float, headwind: float, result: dict) -> None:
        """Store result in cache."""
        key = self._make_key(url, power, mass, headwind)
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
            # Estimate ~1.2 KB per entry based on typical result dict size
            memory_kb = int(len(self.cache) * 1.2)
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_kb": memory_kb,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


# Global cache instance (~0.75 MB at full capacity)
_analysis_cache = AnalysisCache(max_size=500)


# Disk cache for elevation profile images
PROFILE_CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer" / "profiles"
PROFILE_CACHE_INDEX_PATH = PROFILE_CACHE_DIR / "cache_index.json"
MAX_CACHED_PROFILES = 150  # ~150 images, each ~60KB = ~9MB max


def _make_profile_cache_key(url: str, power: float, mass: float, headwind: float) -> str:
    """Create a unique cache key for elevation profile parameters."""
    key_str = f"{url}|{power}|{mass}|{headwind}"
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
        return path.read_bytes()
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
    <meta property="og:description" content="{{ result.time_str }} • {{ '%.0f'|format(result.work_kj) }} kJ | {{ '%.0f'|format(result.distance_km) }} km • {{ '%.0f'|format(result.elevation_m) }}m @ {{ power|int }}W">
    {% endif %}
    <meta property="og:image" content="{{ base_url }}og-image?url={{ url|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}">
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
            gap: 20px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .steep-stat {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .steep-label {
            font-size: 0.8em;
            color: #888;
        }
        .steep-value {
            font-size: 1.1em;
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
        .elevation-profile-container {
            position: relative;
            width: 100%;
        }
        .elevation-profile img {
            width: 100%;
            height: auto;
            display: block;
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
            .param-row { flex-direction: column; gap: 0; }
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
            .logo { width: 40px; height: 40px; }
            .header-section h1 { font-size: 1.2em; }
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
        .url-input-wrapper {
            position: relative;
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
        .recent-url-item:hover {
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
        <div id="compareUrlWrapper" class="url-input-wrapper{{ '' if compare_mode else ' hidden' }}">
            <label for="url2" class="compare-label">Second Route URL</label>
            <input type="text" id="url2" name="url2"
                   placeholder="https://ridewithgps.com/routes/..."
                   value="{{ url2 or '' }}"
                   autocomplete="off">
            <div id="recentUrlsDropdown2" class="recent-urls-dropdown hidden"></div>
        </div>

        <div class="param-row">
            <div>
                <div class="label-row">
                    <label for="power">Average Power (W)</label>
                    <button type="button" class="info-btn" onclick="showModal('powerModal')">?</button>
                </div>
                <input type="number" id="power" name="power" value="{{ power }}" step="1">
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

        <div class="units-row">
            <label class="toggle-label">
                <input type="checkbox" id="imperial" name="imperial" {{ 'checked' if imperial else '' }}>
                <span>Imperial units (mi, ft)</span>
            </label>
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
        <table class="collection-table">
            <thead>
                <tr>
                    <th>Route</th>
                    <th class="num primary"><span class="th-with-info">Time <button type="button" class="info-btn" onclick="showModal('timeModal')">?</button></span></th>
                    <th class="num primary separator"><span class="th-with-info">Work <button type="button" class="info-btn" onclick="showModal('workModal')">?</button></span></th>
                    <th class="num">Dist</th>
                    <th class="num">Elev</th>
                    <th class="num"><span class="th-with-info">Hilly <button type="button" class="info-btn" onclick="showModal('hillyModal')">?</button></span></th>
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
            <h3>Average Power</h3>
            <p>Your expected average power output in watts. This is the sustained power you can maintain over the ride duration. For reference:</p>
            <p>• Casual riding: 80-120W<br>• Moderate effort: 120-180W<br>• Strong rider: 180-250W</p>
            <p>If you have a power meter, use your typical average from similar rides.</p>
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
                <li><span class="param-name">Average Power (W)</span> — Your sustained power output. This is the most important input; doubling power roughly doubles your speed on flat ground.</li>
                <li><span class="param-name">Mass (kg)</span> — Total weight of rider + bike + gear. Dominates climbing speed since you're lifting this weight against gravity.</li>
                <li><span class="param-name">CdA (m²)</span> — Aerodynamic drag coefficient × frontal area. Controls air resistance, which grows with the cube of speed. Typical values: 0.25 (racing tuck) to 0.45 (upright touring).</li>
                <li><span class="param-name">Crr</span> — Rolling resistance coefficient. Energy lost to tire deformation and surface friction. Road tires ~0.004, gravel ~0.008-0.012.</li>
            </ul>

            <h4>Environmental Factors</h4>
            <ul class="param-list">
                <li><span class="param-name">Headwind (km/h)</span> — Wind adds to or subtracts from your effective air speed. A 15 km/h headwind at 25 km/h means you experience drag as if riding 40 km/h.</li>
                <li><span class="param-name">Air density (kg/m³)</span> — Affects aerodynamic drag. Lower at altitude (1.225 at sea level, ~1.0 at 2000m).</li>
            </ul>

            <h4>Climbing Model</h4>
            <ul class="param-list">
                <li><span class="param-name">Climb power factor</span> — Multiplier for power on steep climbs (e.g., 1.5 = 50% more power when climbing hard). Models the tendency to push harder uphill.</li>
                <li><span class="param-name">Climb threshold grade</span> — Grade (in degrees) where full climb power factor kicks in. Below this, power scales linearly.</li>
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
            </ul>
            <p style="margin: 0.5em 0; font-size: 0.85em;"><strong>Curvature-based:</strong></p>
            <ul class="param-list">
                <li><span class="param-name">Straight descent speed</span> — Max speed on straight sections (low curvature).</li>
                <li><span class="param-name">Hairpin speed</span> — Max speed through tight switchbacks (high curvature).</li>
            </ul>

            <h4>Data Processing</h4>
            <ul class="param-list">
                <li><span class="param-name">Smoothing radius (m)</span> — Gaussian smoothing applied to elevation data. Reduces GPS noise and unrealistic grade spikes while preserving overall climb profile.</li>
                <li><span class="param-name">Elevation scale</span> — Multiplier applied after smoothing. Auto-calculated from RideWithGPS API (DEM-corrected) elevation when available.</li>
                <li><span class="param-name">Surface Crr deltas</span> — Per-surface-type rolling resistance adjustments based on RideWithGPS surface data.</li>
            </ul>

            <button class="modal-close" onclick="hideModal('physicsModal')">Got it</button>
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
            <h3>Estimated Moving Time</h3>
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
            <h3>Estimated Work</h3>
            <p>Total mechanical energy expenditure in kilojoules (kJ). This is the energy your legs put into the pedals.</p>
            <p>Useful for estimating food/fuel needs:</p>
            <p>• Human efficiency is ~20-25%, so multiply by 4-5 for calories burned<br>
            • Example: 1000 kJ of work ≈ 4000-5000 kJ (950-1200 kcal) of food energy</p>
            <button class="modal-close" onclick="hideModal('workModal')">Got it</button>
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

    <div id="steepClimbsModal" class="modal-overlay" onclick="hideModal('steepClimbsModal')">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Steep Climbs Methodology</h3>
            <p><strong>Max Grade</strong> is calculated using a 300m rolling average to filter GPS noise and match RideWithGPS methodology. This gives the maximum <em>sustained</em> grade over a meaningful distance.</p>
            <p><strong>Grade Histogram</strong> uses the same 300m rolling average, so grades shown will never exceed the max grade. This ensures consistency between the reported maximum and the histogram distribution.</p>
            <p><strong>Why 300m?</strong> Point-to-point GPS measurements can show unrealistic spikes (50%+ grades) due to elevation noise. Averaging over 300m filters these artifacts while still capturing steep sections that riders actually experience.</p>
            <p>Elevation data is smoothed (50m Gaussian) before grade calculation to reduce GPS noise.</p>
            <button class="modal-close" onclick="hideModal('steepClimbsModal')">Got it</button>
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
                    }
                });
            });
        }

        function setupUrlDropdown() {
            // Setup primary URL input
            var urlInput = document.getElementById('url');
            var dropdown = document.getElementById('recentUrlsDropdown');

            urlInput.addEventListener('focus', function() {
                if (getRecentUrls().length > 0) {
                    populateRecentUrls('recentUrlsDropdown', 'url', false);
                    dropdown.classList.remove('hidden');
                }
            });

            urlInput.addEventListener('input', updateModeIndicator);

            // Setup secondary URL input (compare mode)
            var url2Input = document.getElementById('url2');
            var dropdown2 = document.getElementById('recentUrlsDropdown2');

            url2Input.addEventListener('focus', function() {
                var urls = getRecentUrls().filter(function(item) { return item.url.includes('/routes/'); });
                if (urls.length > 0) {
                    populateRecentUrls('recentUrlsDropdown2', 'url2', true);
                    dropdown2.classList.remove('hidden');
                }
            });

            // Hide dropdowns when clicking outside
            document.addEventListener('click', function(e) {
                if (!urlInput.contains(e.target) && !dropdown.contains(e.target)) {
                    dropdown.classList.add('hidden');
                }
                if (!url2Input.contains(e.target) && !dropdown2.contains(e.target)) {
                    dropdown2.classList.add('hidden');
                }
            });

            // Hide dropdowns on escape
            urlInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') dropdown.classList.add('hidden');
            });
            url2Input.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') dropdown2.classList.add('hidden');
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
        });
        if (document.readyState !== 'loading') {
            setupUrlDropdown();
        }

        function showModal(id) {
            document.getElementById(id).classList.add('active');
        }

        function hideModal(id) {
            document.getElementById(id).classList.remove('active');
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

            if (mode === 'collection') {
                modeInput.value = 'collection';
            } else {
                modeInput.value = 'route';
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
                power: document.getElementById('power').value,
                mass: document.getElementById('mass').value,
                headwind: document.getElementById('headwind').value
            });
            if (document.getElementById('imperial').checked) {
                params.set('imperial', '1');
            }
            return window.location.origin + window.location.pathname + '?' + params.toString();
        }

        function updateTotals(routes) {
            var totalDist = 0, totalElev = 0, totalTime = 0, totalWork = 0;
            routes.forEach(function(r) {
                totalDist += r.distance_km;
                totalElev += r.elevation_m;
                totalTime += r.time_seconds;
                totalWork += r.work_kj;
            });
            document.getElementById('totalRoutes').textContent = routes.length;
            document.getElementById('totalDistance').textContent = formatDistFull(totalDist);
            document.getElementById('totalElevation').textContent = formatElevFull(totalElev);
            document.getElementById('totalTime').textContent = formatDuration(totalTime);
            document.getElementById('totalWork').textContent = Math.round(totalWork) + ' kJ';

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
                '<td class="num primary separator">' + Math.round(totalWork) + 'kJ</td>' +
                '<td class="num">' + formatDist(totalDist) + '</td>' +
                '<td class="num">' + formatElev(totalElev) + '</td>' +
                '<td class="num"></td><td class="num"></td><td class="num"></td><td class="num"></td><td class="num"></td>';
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
            var power = document.getElementById('power').value;
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
                power: power,
                mass: mass,
                headwind: headwind
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
                    // Save URL with collection name
                    saveRecentUrl(url, data.name);
                    // Build share URL
                    var shareParams = new URLSearchParams({
                        url: url,
                        power: power,
                        mass: mass,
                        headwind: headwind
                    });
                    if (document.getElementById('imperial').checked) {
                        shareParams.set('imperial', '1');
                    }
                    document.getElementById('collectionShareUrl').value =
                        window.location.origin + window.location.pathname + '?' + shareParams.toString();
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
                        '<td class="num primary separator">' + Math.round(r.work_kj) + 'kJ</td>' +
                        '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                        '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
                        '<td class="num">' + formatHilliness(r.hilliness_score || 0) + '</td>' +
                        '<td class="num">' + (r.steepness_score || 0).toFixed(1) + '%</td>' +
                        '<td class="num">' + formatSpeed(r.avg_speed_kmh) + '</td>' +
                        '<td class="num">' + Math.round(r.unpaved_pct || 0) + '%</td>' +
                        '<td class="num">' + r.elevation_scale.toFixed(2) + '</td>';
                    document.getElementById('routesTableBody').appendChild(row);

                    // Show results container and update totals
                    document.getElementById('collectionResults').classList.remove('hidden');
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
                    '<td class="num primary separator">' + Math.round(r.work_kj) + 'kJ</td>' +
                    '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                    '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
                    '<td class="num">' + formatHilliness(r.hilliness_score || 0) + '</td>' +
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
            });
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
                        <td><span class="label-with-info">{% if is_trip and is_trip2 %}Moving Time{% else %}Est. Moving Time{% endif %} <button type="button" class="info-btn" onclick="showModal('timeModal')">?</button></span></td>
                        <td class="route-col">{{ result.time_str }}</td>
                        <td class="route-col">{{ result2.time_str }}</td>
                        <td class="diff-col">{{ format_time_diff(result.time_seconds, result2.time_seconds) }}</td>
                    </tr>
                    {% if (not is_trip or result.work_kj is not none) and (not is_trip2 or result2.work_kj is not none) %}
                    <tr class="primary">
                        <td><span class="label-with-info">{% if is_trip and is_trip2 %}Work{% else %}Est. Work{% endif %} <button type="button" class="info-btn" onclick="showModal('workModal')">?</button></span></td>
                        <td class="route-col">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj) }} kJ{% else %}-{% endif %}</td>
                        <td class="route-col">{% if result2.work_kj is not none %}{{ "%.0f"|format(result2.work_kj) }} kJ{% else %}-{% endif %}</td>
                        <td class="diff-col">{% if result.work_kj is not none and result2.work_kj is not none %}{{ format_diff(result.work_kj, result2.work_kj, 'kJ') }}{% else %}-{% endif %}</td>
                    </tr>
                    {% endif %}
                    {% if (is_trip and result.has_power) or (is_trip2 and result2.has_power) %}
                    <tr class="primary">
                        <td>Avg Power</td>
                        <td class="route-col">{% if result.avg_watts is not none %}{{ result.avg_watts|int }} W{% else %}-{% endif %}</td>
                        <td class="route-col">{% if result2.avg_watts is not none %}{{ result2.avg_watts|int }} W{% else %}-{% endif %}</td>
                        <td class="diff-col">{% if result.avg_watts is not none and result2.avg_watts is not none %}{{ format_diff(result.avg_watts, result2.avg_watts, 'W') }}{% else %}-{% endif %}</td>
                    </tr>
                    {% endif %}
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
        {% else %}
        <!-- Single route/trip results -->
        <div class="primary-results">
            <div class="result-row primary">
                <span class="result-label label-with-info">{% if is_trip %}Moving Time{% else %}Estimated Moving Time{% endif %} <button type="button" class="info-btn" onclick="showModal('timeModal')">?</button></span>
                <span class="result-value">{{ result.time_str }}</span>
            </div>
            {% if not is_trip or result.work_kj is not none %}
            <div class="result-row primary">
                <span class="result-label label-with-info">{% if is_trip %}Work{% else %}Estimated Work{% endif %} <button type="button" class="info-btn" onclick="showModal('workModal')">?</button></span>
                <span class="result-value">{% if result.work_kj is not none %}{{ "%.0f"|format(result.work_kj) }} kJ{% else %}-{% endif %}</span>
            </div>
            {% endif %}
            {% if is_trip and result.has_power %}
            <div class="result-row primary">
                <span class="result-label">Avg Power</span>
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
                        <th>{{ (result.name or 'Route 1')|truncate(15) }}</th>
                        <th>{{ (result2.name or 'Route 2')|truncate(15) }}</th>
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

        {% if compare_mode and result2 %}
        <!-- Stacked elevation profiles for comparison -->
        <div class="elevation-profiles-stacked">
            <div class="elevation-profile">
                <h4>{{ (result.name or 'Route 1')|truncate(40) }}</h4>
                <div class="elevation-profile-container" id="elevationContainer1" data-url="{{ url|urlencode }}">
                    <div class="elevation-loading" id="elevationLoading1">
                        <div class="elevation-spinner"></div>
                    </div>
                    <img src="/elevation-profile?url={{ url|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}"
                         alt="Elevation profile - Route 1" id="elevationImg1" class="loading"
                         onload="document.getElementById('elevationLoading1').classList.add('hidden'); this.classList.remove('loading');">
                    <div class="elevation-cursor" id="elevationCursor1"></div>
                    <div class="elevation-tooltip" id="elevationTooltip1">
                        <div class="grade">--</div>
                        <div class="elev">--</div>
                    </div>
                </div>
            </div>
            <div class="elevation-profile">
                <h4>{{ (result2.name or 'Route 2')|truncate(40) }}</h4>
                <div class="elevation-profile-container" id="elevationContainer2" data-url="{{ url2|urlencode }}">
                    <div class="elevation-loading" id="elevationLoading2">
                        <div class="elevation-spinner"></div>
                    </div>
                    <img src="/elevation-profile?url={{ url2|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}"
                         alt="Elevation profile - Route 2" id="elevationImg2" class="loading"
                         onload="document.getElementById('elevationLoading2').classList.add('hidden'); this.classList.remove('loading');">
                    <div class="elevation-cursor" id="elevationCursor2"></div>
                    <div class="elevation-tooltip" id="elevationTooltip2">
                        <div class="grade">--</div>
                        <div class="elev">--</div>
                    </div>
                </div>
            </div>
        </div>
        {% else %}
        <!-- Single elevation profile -->
        <div class="elevation-profile">
            <h4>Time-Based Elevation Profile</h4>
            <div class="elevation-profile-container" id="elevationContainer">
                <div class="elevation-loading" id="elevationLoading">
                    <div class="elevation-spinner"></div>
                </div>
                <img src="/elevation-profile?url={{ url|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}"
                     alt="Elevation profile" id="elevationImg" class="loading"
                     onload="document.getElementById('elevationLoading').classList.add('hidden'); this.classList.remove('loading');">
                <div class="elevation-cursor" id="elevationCursor"></div>
                <div class="elevation-tooltip" id="elevationTooltip">
                    <div class="grade">--</div>
                    <div class="elev">--</div>
                </div>
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

        // Elevation profile hover interaction
        (function() {
            // Setup function for a single profile
            function setupElevationProfile(containerId, imgId, tooltipId, cursorId, dataUrl) {
                const container = document.getElementById(containerId);
                const img = document.getElementById(imgId);
                const tooltip = document.getElementById(tooltipId);
                const cursor = document.getElementById(cursorId);
                if (!container || !img || !tooltip || !cursor) return;

                let profileData = null;

                // Fetch profile data
                fetch(dataUrl)
                    .then(r => r.json())
                    .then(data => {
                        if (!data.error) profileData = data;
                    })
                    .catch(() => {});

                // The plot area is approximately 85% of image width (matplotlib default margins)
                const plotLeftPct = 0.10;
                const plotRightPct = 0.98;

                function getGradeAtPosition(xPct) {
                    if (!profileData || !profileData.times || profileData.times.length < 2) return null;

                    const plotXPct = (xPct - plotLeftPct) / (plotRightPct - plotLeftPct);
                    if (plotXPct < 0 || plotXPct > 1) return null;

                    const time = plotXPct * profileData.total_time;

                    for (let i = 0; i < profileData.times.length - 1; i++) {
                        if (time >= profileData.times[i] && time < profileData.times[i + 1]) {
                            return {
                                grade: profileData.grades[i] || 0,
                                elevation: profileData.elevations[i] || 0,
                                time: time
                            };
                        }
                    }
                    const lastIdx = profileData.grades.length - 1;
                    return {
                        grade: profileData.grades[lastIdx] || 0,
                        elevation: profileData.elevations[lastIdx + 1] || 0,
                        time: time
                    };
                }

                function formatGrade(g) {
                    const sign = g >= 0 ? '+' : '';
                    return sign + g.toFixed(1) + '%';
                }

                function formatTime(hours) {
                    const h = Math.floor(hours);
                    const m = Math.floor((hours - h) * 60);
                    return h + 'h ' + m.toString().padStart(2, '0') + 'm';
                }

                function handleMove(e) {
                    if (!profileData) return;

                    const rect = img.getBoundingClientRect();
                    let clientX;

                    if (e.touches) {
                        clientX = e.touches[0].clientX;
                    } else {
                        clientX = e.clientX;
                    }

                    const xPct = (clientX - rect.left) / rect.width;
                    const data = getGradeAtPosition(xPct);

                    if (data) {
                        tooltip.querySelector('.grade').textContent = formatGrade(data.grade);
                        const elevUnit = isImperial() ? 'ft' : 'm';
                        const elevVal = isImperial() ? data.elevation * 3.28084 : data.elevation;
                        tooltip.querySelector('.elev').textContent = Math.round(elevVal) + ' ' + elevUnit + ' | ' + formatTime(data.time);

                        const xPos = clientX - rect.left;
                        tooltip.style.left = xPos + 'px';
                        tooltip.style.bottom = '60px';
                        tooltip.classList.add('visible');

                        cursor.style.left = xPos + 'px';
                        cursor.classList.add('visible');
                    } else {
                        tooltip.classList.remove('visible');
                        cursor.classList.remove('visible');
                    }
                }

                function handleLeave() {
                    tooltip.classList.remove('visible');
                    cursor.classList.remove('visible');
                }

                container.addEventListener('mousemove', handleMove);
                container.addEventListener('mouseleave', handleLeave);
                container.addEventListener('touchmove', function(e) {
                    e.preventDefault();
                    handleMove(e);
                }, { passive: false });
                container.addEventListener('touchend', handleLeave);
            }

            // Check for comparison mode (two profiles)
            var container1 = document.getElementById('elevationContainer1');
            var container2 = document.getElementById('elevationContainer2');
            if (container1 && container2) {
                // Comparison mode - setup both profiles
                setupElevationProfile(
                    'elevationContainer1', 'elevationImg1', 'elevationTooltip1', 'elevationCursor1',
                    '/elevation-profile-data?url={{ url|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}'
                );
                setupElevationProfile(
                    'elevationContainer2', 'elevationImg2', 'elevationTooltip2', 'elevationCursor2',
                    '/elevation-profile-data?url={{ url2|urlencode if url2 else "" }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}'
                );
            } else {
                // Single route mode
                setupElevationProfile(
                    'elevationContainer', 'elevationImg', 'elevationTooltip', 'elevationCursor',
                    '/elevation-profile-data?url={{ url|urlencode }}&power={{ power }}&mass={{ mass }}&headwind={{ headwind }}'
                );
            }
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
        </div>
    </div>
</body>
</html>
"""


def get_defaults():
    """Get default values from config file, falling back to DEFAULTS."""
    config = _load_config() or {}
    return {
        "power": config.get("power", DEFAULTS["power"]),
        "mass": config.get("mass", DEFAULTS["mass"]),
        "headwind": config.get("headwind", DEFAULTS["headwind"]),
    }


def build_params(power: float, mass: float, headwind: float) -> RiderParams:
    """Build RiderParams from user inputs and config defaults."""
    config = _load_config() or {}
    return RiderParams(
        total_mass=mass,
        cda=config.get("cda", DEFAULTS["cda"]),
        crr=config.get("crr", DEFAULTS["crr"]),
        assumed_avg_power=power,
        coasting_grade_threshold=config.get("coasting_grade", DEFAULTS["coasting_grade"]),
        max_coasting_speed=config.get("max_coast_speed", DEFAULTS["max_coast_speed"]) / 3.6,
        max_coasting_speed_unpaved=config.get("max_coast_speed_unpaved", DEFAULTS["max_coast_speed_unpaved"]) / 3.6,
        headwind=headwind / 3.6,
        climb_power_factor=config.get("climb_power_factor", DEFAULTS["climb_power_factor"]),
        climb_threshold_grade=config.get("climb_threshold_grade", DEFAULTS["climb_threshold_grade"]),
        steep_descent_speed=config.get("steep_descent_speed", DEFAULTS["steep_descent_speed"]) / 3.6,
        steep_descent_grade=config.get("steep_descent_grade", DEFAULTS["steep_descent_grade"]),
        drivetrain_efficiency=config.get("drivetrain_efficiency", DEFAULTS["drivetrain_efficiency"]),
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


def analyze_single_route(url: str, params: RiderParams) -> dict:
    """Analyze a single route and return results dict.

    Results are cached based on (url, power, mass, headwind) for faster
    repeated access when comparing routes.
    """
    # Check cache first
    cached = _analysis_cache.get(
        url, params.assumed_avg_power, params.total_mass, params.headwind
    )
    if cached is not None:
        return cached

    config = _load_config() or {}
    smoothing_radius = config.get("smoothing", DEFAULTS["smoothing"])

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Smooth without scaling first (for max grade calculation)
    unscaled_points = smooth_elevations(points, smoothing_radius, 1.0)

    # Calculate API-based elevation scale factor
    api_elevation_scale = 1.0
    api_elevation_gain = route_metadata.get("elevation_gain") if route_metadata else None
    if api_elevation_gain and api_elevation_gain > 0:
        smoothed_gain = calculate_elevation_gain(unscaled_points)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    if smoothing_radius > 0 or api_elevation_scale != 1.0:
        points = smooth_elevations(points, smoothing_radius, api_elevation_scale)

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
        "time_str": format_duration_long(analysis.estimated_moving_time_at_power.total_seconds()),
        "time_seconds": analysis.estimated_moving_time_at_power.total_seconds(),
        "avg_speed_kmh": analysis.avg_speed * 3.6,
        "avg_speed_mph": analysis.avg_speed * 3.6 * 0.621371,
        "work_kj": analysis.estimated_work / 1000,
        "avg_watts": None,  # Routes don't have recorded power
        "has_power": False,
        "unpaved_pct": unpaved_pct,
        "elevation_scale": api_elevation_scale,
        "elevation_scaled": abs(api_elevation_scale - 1.0) > 0.05,
        "hilliness_score": hilliness.hilliness_score,
        "steepness_score": hilliness.steepness_score,
        "grade_histogram": hilliness.grade_time_histogram,
        "grade_distance_histogram": hilliness.grade_distance_histogram,
        "max_grade": hilliness.max_grade,
        "steep_distance": hilliness.steep_distance,
        "very_steep_distance": hilliness.very_steep_distance,
        "steep_time_histogram": hilliness.steep_time_histogram,
        "steep_distance_histogram": hilliness.steep_distance_histogram,
        "hilliness_total_time": hilliness.total_time,
        "hilliness_total_distance": hilliness.total_distance,
    }

    # Store in cache
    _analysis_cache.set(
        url, params.assumed_avg_power, params.total_mass, params.headwind, result
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

        total_distance += dist

        # Time delta from actual timestamps
        if curr.timestamp is not None and prev.timestamp is not None:
            time_delta = curr.timestamp - prev.timestamp
        else:
            time_delta = 0.0

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
        "hilliness_total_time": sum(grade_times.values()),
        "hilliness_total_distance": total_distance,
        # Trip-specific flags
        "is_trip": True,
    }

    return result


def _get_profile_cache_stats() -> dict:
    """Get statistics for the elevation profile disk cache."""
    index = _load_profile_cache_index()
    total_bytes = 0
    for key in index:
        path = PROFILE_CACHE_DIR / f"{key}.png"
        if path.exists():
            total_bytes += path.stat().st_size
    return {
        "size": len(index),
        "max_size": MAX_CACHED_PROFILES,
        "disk_mb": round(total_bytes / (1024 * 1024), 2),
    }


def _clear_profile_cache() -> int:
    """Clear the elevation profile disk cache. Returns number of files removed."""
    index = _load_profile_cache_index()
    count = 0
    for key in list(index.keys()):
        path = PROFILE_CACHE_DIR / f"{key}.png"
        if path.exists():
            path.unlink()
            count += 1
    # Clear the index
    _save_profile_cache_index({})
    return count


@app.route("/cache-stats")
def cache_stats():
    """Return cache statistics as JSON."""
    return {
        "analysis_cache": _analysis_cache.stats(),
        "profile_cache": _get_profile_cache_stats(),
    }


@app.route("/cache-clear", methods=["GET", "POST"])
def cache_clear():
    """Clear both analysis and profile image caches."""
    _analysis_cache.clear()
    profiles_cleared = _clear_profile_cache()
    return {
        "status": "ok",
        "message": f"Caches cleared (analysis cache + {profiles_cleared} profile images)"
    }


@app.route("/analyze-collection-stream")
def analyze_collection_stream():
    """SSE endpoint for streaming collection analysis progress."""
    url = request.args.get("url", "")
    try:
        power = float(request.args.get("power", 100))
        mass = float(request.args.get("mass", 85))
        headwind = float(request.args.get("headwind", 0))
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

            params = build_params(power, mass, headwind)

            for i, route_id in enumerate(route_ids):
                route_url = f"https://ridewithgps.com/routes/{route_id}"

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(route_ids)})}\n\n"

                try:
                    route_result = analyze_single_route(route_url, params)
                    route_result["time_str"] = format_duration(route_result["time_seconds"])
                    route_result["route_id"] = route_id

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
    url = request.args.get("url", "")

    if not url or not is_ridewithgps_url(url):
        # Return a simple fallback image
        return generate_fallback_image()

    try:
        power = float(request.args.get("power", DEFAULTS["power"]))
        mass = float(request.args.get("mass", DEFAULTS["mass"]))
        headwind = float(request.args.get("headwind", DEFAULTS["headwind"]))
    except ValueError:
        return generate_fallback_image()

    try:
        params = build_params(power, mass, headwind)
        result = analyze_single_route(url, params)
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


def _calculate_elevation_profile_data(url: str, params: RiderParams) -> dict:
    """Calculate elevation profile data for a route.

    Returns dict with times_hours, elevations, grades, and route_name.
    """
    config = _load_config() or {}
    smoothing_radius = config.get("smoothing", DEFAULTS["smoothing"])

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Apply smoothing
    api_elevation_scale = 1.0
    api_elevation_gain = route_metadata.get("elevation_gain") if route_metadata else None
    if api_elevation_gain and api_elevation_gain > 0:
        unscaled = smooth_elevations(points, smoothing_radius, 1.0)
        smoothed_gain = calculate_elevation_gain(unscaled)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    points = smooth_elevations(points, smoothing_radius, api_elevation_scale)

    # Calculate rolling grades (smoothed over a window, same as histograms)
    max_grade_window = config.get("max_grade_window_route", DEFAULT_MAX_GRADE_WINDOW)
    rolling_grades = _calculate_rolling_grades(points, max_grade_window)

    # Calculate cumulative time and elevation at each point
    cum_time = [0.0]
    elevations = [points[0].elevation or 0.0]

    for i in range(1, len(points)):
        _, dist, elapsed = calculate_segment_work(points[i-1], points[i], params)
        cum_time.append(cum_time[-1] + elapsed)
        elevations.append(points[i].elevation or elevations[-1])

    # Convert to hours
    times_hours = [t / 3600 for t in cum_time]

    # Use rolling grades (one fewer than points, like segment grades)
    grades = rolling_grades if rolling_grades else [0.0] * (len(points) - 1)

    route_name = route_metadata.get("name", "Elevation Profile") if route_metadata else "Elevation Profile"

    return {
        "times_hours": times_hours,
        "elevations": elevations,
        "grades": grades,
        "route_name": route_name,
    }


def _calculate_trip_elevation_profile_data(url: str) -> dict:
    """Calculate elevation profile data for a trip using actual timestamps.

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

    # Apply smoothing with elevation scaling
    api_elevation_gain = trip_metadata.get("elevation_gain")
    api_elevation_scale = 1.0
    if api_elevation_gain and api_elevation_gain > 0:
        unscaled = smooth_elevations(track_points, smoothing_radius, 1.0)
        smoothed_gain = calculate_elevation_gain(unscaled)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    smoothed_points = smooth_elevations(track_points, smoothing_radius, api_elevation_scale)

    # Calculate rolling grades
    rolling_grades = _calculate_rolling_grades(smoothed_points, max_grade_window)

    # Use actual timestamps for x-axis
    if trip_points[0].timestamp is None:
        raise ValueError("Trip has no timestamp data")

    base_time = trip_points[0].timestamp
    times_hours = []
    elevations = []

    for i, (tp, sp) in enumerate(zip(trip_points, smoothed_points)):
        if tp.timestamp is not None:
            hours = (tp.timestamp - base_time) / 3600
        else:
            hours = times_hours[-1] if times_hours else 0.0
        times_hours.append(hours)
        elevations.append(sp.elevation or 0.0)

    grades = rolling_grades if rolling_grades else [0.0] * (len(trip_points) - 1)

    trip_name = trip_metadata.get("name", "Trip Profile") if trip_metadata else "Trip Profile"

    return {
        "times_hours": times_hours,
        "elevations": elevations,
        "grades": grades,
        "route_name": trip_name,
    }


def generate_elevation_profile(url: str, params: RiderParams, title_time_hours: float | None = None) -> bytes:
    """Generate elevation profile image with grade-based coloring.

    Args:
        url: RideWithGPS route URL
        params: Rider parameters
        title_time_hours: Optional time to display in title (from calibrated analysis).
                          If None, uses uncalibrated time from profile data.

    Returns PNG image as bytes.
    """
    data = _calculate_elevation_profile_data(url, params)
    times_hours = data["times_hours"]
    elevations = data["elevations"]
    grades = data["grades"]
    route_name = data["route_name"]

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

    # Create figure - wide aspect ratio for full width
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')

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

    # Style the plot
    ax.set_xlim(0, times_hours[-1])
    ax.set_ylim(0, max(elevations) * 1.1)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)

    # Add title with route info (use calibrated time if provided)
    display_time = title_time_hours if title_time_hours is not None else times_hours[-1]
    ax.set_title(f"{route_name} - {display_time:.1f}h", fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_trip_elevation_profile(url: str, title_time_hours: float | None = None) -> bytes:
    """Generate elevation profile image for a trip with grade-based coloring.

    Args:
        url: RideWithGPS trip URL
        title_time_hours: Optional time to display in title.

    Returns PNG image as bytes.
    """
    data = _calculate_trip_elevation_profile_data(url)
    times_hours = data["times_hours"]
    elevations = data["elevations"]
    grades = data["grades"]
    route_name = data["route_name"]

    # Reuse the same grade-to-color mapping from route profile
    main_colors = [
        '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
        '#cccccc',
        '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
    ]
    steep_colors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000']

    def grade_to_color(g):
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

    ax.set_xlim(0, times_hours[-1])
    ax.set_ylim(0, max(elevations) * 1.1)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)

    display_time = title_time_hours if title_time_hours is not None else times_hours[-1]
    ax.set_title(f"{route_name} - {display_time:.1f}h", fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.route("/elevation-profile")
def elevation_profile():
    """Serve elevation profile image for a route or trip."""
    url = request.args.get("url", "")
    power = float(request.args.get("power", DEFAULTS["power"]))
    mass = float(request.args.get("mass", DEFAULTS["mass"]))
    headwind = float(request.args.get("headwind", DEFAULTS["headwind"]))

    if not url or not (is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)):
        # Return a placeholder image
        fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')
        ax.text(0.5, 0.5, 'No route or trip selected', ha='center', va='center', fontsize=14, color='#999')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Check disk cache first
    cache_key = _make_profile_cache_key(url, power, mass, headwind)
    cached_bytes = _get_cached_profile(cache_key)
    if cached_bytes:
        return send_file(io.BytesIO(cached_bytes), mimetype='image/png')

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps, no physics params needed
            trip_result = analyze_trip(url)
            title_time_hours = trip_result["time_seconds"] / 3600
            img_bytes = generate_trip_elevation_profile(url, title_time_hours)
        else:
            # Route: use physics estimation
            params = build_params(power, mass, headwind)
            analysis = analyze_single_route(url, params)
            title_time_hours = analysis["time_seconds"] / 3600
            img_bytes = generate_elevation_profile(url, params, title_time_hours)
        # Save to disk cache
        _save_profile_to_cache(cache_key, img_bytes)
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        # Return error image
        fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')
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
    url = request.args.get("url", "")
    power = float(request.args.get("power", DEFAULTS["power"]))
    mass = float(request.args.get("mass", DEFAULTS["mass"]))
    headwind = float(request.args.get("headwind", DEFAULTS["headwind"]))

    if not url or not (is_ridewithgps_url(url) or is_ridewithgps_trip_url(url)):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        if is_ridewithgps_trip_url(url):
            # Trip: use actual timestamps
            data = _calculate_trip_elevation_profile_data(url)
        else:
            # Route: use physics estimation
            params = build_params(power, mass, headwind)
            data = _calculate_elevation_profile_data(url, params)

        # Downsample data if too many points (for performance)
        max_points = 500
        times = data["times_hours"]
        elevations = data["elevations"]
        grades = data["grades"]

        if len(times) > max_points:
            step = len(times) // max_points
            times = times[::step]
            elevations = elevations[::step]
            grades = grades[::step]

        return jsonify({
            "times": times,
            "elevations": elevations,
            "grades": grades,
            "total_time": data["times_hours"][-1],
        })
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


def _analyze_url(url: str, params: RiderParams) -> tuple[dict, bool]:
    """Analyze a URL (route or trip) and return result and is_trip flag.

    Args:
        url: RideWithGPS route or trip URL
        params: Rider parameters (only used for routes)

    Returns:
        Tuple of (analysis result dict, is_trip flag)
    """
    if is_ridewithgps_trip_url(url):
        result = analyze_trip(url)
        return result, True
    else:
        result = analyze_single_route(url, params)
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
    is_trip = False  # Flag for trip vs route
    is_trip2 = False  # Flag for second URL trip vs route

    power = defaults["power"]
    mass = defaults["mass"]
    headwind = defaults["headwind"]

    # Check for GET parameters (shared link)
    if request.method == "GET" and request.args.get("url"):
        url = request.args.get("url", "").strip()
        url2 = request.args.get("url2", "").strip()  # Second route/trip for comparison
        imperial = request.args.get("imperial") == "1"

        try:
            power = float(request.args.get("power", defaults["power"]))
            mass = float(request.args.get("mass", defaults["mass"]))
            headwind = float(request.args.get("headwind", defaults["headwind"]))
        except ValueError:
            error = "Invalid number in parameters"

        if not error:
            if is_ridewithgps_collection_url(url):
                # Collection - set mode and let JavaScript handle it
                mode = "collection"
            elif _is_valid_rwgps_url(url):
                # Single route or trip - analyze server-side
                try:
                    params = build_params(power, mass, headwind)
                    result, is_trip = _analyze_url(url, params)
                    route_id = _extract_id_from_url(url)
                    privacy_code = extract_privacy_code(url)
                except Exception as e:
                    error = f"Error analyzing {'trip' if is_ridewithgps_trip_url(url) else 'route'}: {e}"

                # Handle second route/trip for comparison
                if url2 and not error:
                    if _is_valid_rwgps_url(url2):
                        compare_mode = True
                        try:
                            result2, is_trip2 = _analyze_url(url2, params)
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
            power = float(request.form.get("power", defaults["power"]))
            mass = float(request.form.get("mass", defaults["mass"]))
            headwind = float(request.form.get("headwind", defaults["headwind"]))
        except ValueError:
            error = "Invalid number in parameters"

        if not error:
            if not url:
                error = "Please enter a RideWithGPS URL"
            elif mode == "route":
                if not _is_valid_rwgps_url(url):
                    error = "Invalid RideWithGPS URL. Expected format: https://ridewithgps.com/routes/XXXXX or /trips/XXXXX"
                else:
                    try:
                        params = build_params(power, mass, headwind)
                        result, is_trip = _analyze_url(url, params)
                        route_id = _extract_id_from_url(url)
                        privacy_code = extract_privacy_code(url)
                    except Exception as e:
                        error = f"Error analyzing {'trip' if is_ridewithgps_trip_url(url) else 'route'}: {e}"

                    # Handle second route/trip for comparison (only if checkbox is checked)
                    if url2 and compare_enabled and not error:
                        if _is_valid_rwgps_url(url2):
                            compare_mode = True
                            try:
                                result2, is_trip2 = _analyze_url(url2, params)
                                route_id2 = _extract_id_from_url(url2)
                                privacy_code2 = extract_privacy_code(url2)
                            except Exception as e:
                                error = f"Error analyzing second {'trip' if is_ridewithgps_trip_url(url2) else 'route'}: {e}"
                        else:
                            error = "Invalid second RideWithGPS URL"
            # Collection mode is handled by JavaScript + SSE

    # Build share URL if we have results
    if result and url:
        from urllib.parse import urlencode
        share_params = {
            "url": url,
            "power": power,
            "mass": mass,
            "headwind": headwind,
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
        power=power,
        mass=mass,
        headwind=headwind,
        imperial=imperial,
        error=error,
        result=result,
        result2=result2,
        route_id=route_id,
        route_id2=route_id2,
        privacy_code=privacy_code,
        privacy_code2=privacy_code2,
        share_url=share_url,
        version_date=__version_date__,
        git_hash=get_git_hash(),
        # Helper functions for comparison formatting
        format_time_diff=format_time_diff,
        format_diff=format_diff,
        format_pct_diff=format_pct_diff,
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
