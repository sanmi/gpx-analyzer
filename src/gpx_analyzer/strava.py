"""Strava URL support with local caching."""

from dataclasses import dataclass
import json
import logging
import os
import re
import time
from pathlib import Path
from io import BytesIO

import requests

from gpx_analyzer.models import TrackPoint
from gpx_analyzer.parser import parse_gpx

logger = logging.getLogger(__name__)

# Config and cache directories (shared with ridewithgps)
CONFIG_DIR = Path.home() / ".config" / "gpx-analyzer"
CONFIG_PATH = CONFIG_DIR / "gpx-analyzer.json"
LOCAL_CONFIG_PATH = Path("gpx-analyzer.json")
CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer"

# Strava-specific cache directories
STRAVA_ROUTES_DIR = CACHE_DIR / "strava_routes"
STRAVA_ACTIVITIES_DIR = CACHE_DIR / "strava_activities"
STRAVA_ROUTES_CACHE_INDEX = CACHE_DIR / "strava_routes_cache_index.json"
STRAVA_ACTIVITIES_CACHE_INDEX = CACHE_DIR / "strava_activities_cache_index.json"

# Cache limits
MAX_CACHED_STRAVA_ROUTES = 50
MAX_CACHED_STRAVA_ACTIVITIES = 50

# Strava API
STRAVA_API_BASE = "https://www.strava.com/api/v3"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"

# URL patterns
STRAVA_ROUTE_PATTERN = re.compile(r"^https?://(?:www\.)?strava\.com/routes/(\d+)")
STRAVA_ACTIVITY_PATTERN = re.compile(r"^https?://(?:www\.)?strava\.com/activities/(\d+)")

# Token cache (in-memory)
_token_cache = {
    "access_token": None,
    "expires_at": 0,
}


@dataclass
class TripPoint:
    """A point from an actual recorded ride (Strava activity)."""
    lat: float
    lon: float
    elevation: float | None
    distance: float  # cumulative distance in meters
    speed: float | None  # m/s
    timestamp: float | None  # unix timestamp
    power: float | None  # watts
    heart_rate: int | None
    cadence: int | None


def is_strava_route_url(url: str) -> bool:
    """Check if the given URL is a Strava route URL."""
    return bool(STRAVA_ROUTE_PATTERN.match(url))


def is_strava_activity_url(url: str) -> bool:
    """Check if the given URL is a Strava activity URL."""
    return bool(STRAVA_ACTIVITY_PATTERN.match(url))


def is_strava_url(url: str) -> bool:
    """Check if the given URL is any Strava URL (route or activity)."""
    return is_strava_route_url(url) or is_strava_activity_url(url)


def extract_strava_route_id(url: str) -> int:
    """Extract the route ID from a Strava route URL.

    Raises:
        ValueError: If the URL is not a valid Strava route URL.
    """
    match = STRAVA_ROUTE_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid Strava route URL: {url}")
    return int(match.group(1))


def extract_strava_activity_id(url: str) -> int:
    """Extract the activity ID from a Strava activity URL.

    Raises:
        ValueError: If the URL is not a valid Strava activity URL.
    """
    match = STRAVA_ACTIVITY_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid Strava activity URL: {url}")
    return int(match.group(1))


# ============================================================================
# Configuration
# ============================================================================

def _load_config() -> dict:
    """Load configuration from config files.

    Merges config from global and local files:
    1. ~/.config/gpx-analyzer/gpx-analyzer.json (global, loaded first)
    2. ./gpx-analyzer.json (local, overrides global)

    Returns:
        Dict with merged config values, empty dict if no files exist.
    """
    config = {}
    for config_path in [CONFIG_PATH, LOCAL_CONFIG_PATH]:
        if config_path.exists():
            try:
                with config_path.open() as f:
                    config.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return config


def _get_strava_credentials() -> tuple[str, str, str] | None:
    """Get Strava API credentials from config or environment.

    Returns:
        Tuple of (client_id, client_secret, refresh_token) or None if not configured.
    """
    config = _load_config()

    client_id = config.get("strava_client_id") or os.environ.get("STRAVA_CLIENT_ID")
    client_secret = config.get("strava_client_secret") or os.environ.get("STRAVA_CLIENT_SECRET")
    refresh_token = config.get("strava_refresh_token") or os.environ.get("STRAVA_REFRESH_TOKEN")

    if client_id and client_secret and refresh_token:
        return (client_id, client_secret, refresh_token)
    return None


def _get_access_token() -> str:
    """Get a valid Strava access token, refreshing if necessary.

    Returns:
        Valid access token string.

    Raises:
        ValueError: If Strava credentials are not configured.
        requests.RequestException: If token refresh fails.
    """
    global _token_cache

    # Check if we have a valid cached token
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    # Need to refresh
    credentials = _get_strava_credentials()
    if not credentials:
        raise ValueError(
            "Strava credentials not configured. Add strava_client_id, "
            "strava_client_secret, and strava_refresh_token to your config file."
        )

    client_id, client_secret, refresh_token = credentials

    response = requests.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = data["expires_at"]

    # Log rate limit info if available
    if "X-RateLimit-Limit" in response.headers:
        logger.debug(
            "Strava rate limit: %s used of %s",
            response.headers.get("X-RateLimit-Usage"),
            response.headers.get("X-RateLimit-Limit"),
        )

    return _token_cache["access_token"]


def _get_auth_headers() -> dict[str, str]:
    """Get authorization headers for Strava API requests."""
    token = _get_access_token()
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Caching (similar to RideWithGPS)
# ============================================================================

def _ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    STRAVA_ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    STRAVA_ACTIVITIES_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache_index(index_path: Path) -> dict:
    """Load cache index from disk."""
    if index_path.exists():
        try:
            with index_path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache_index(index_path: Path, index: dict):
    """Save cache index to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w") as f:
        json.dump(index, f)


def _update_lru(index_path: Path, item_id: str):
    """Update LRU timestamp for a cached item."""
    index = _load_cache_index(index_path)
    index[item_id] = time.time()
    _save_cache_index(index_path, index)


def _enforce_lru_limit(index_path: Path, cache_dir: Path, max_items: int, extension: str):
    """Remove oldest items if cache exceeds limit."""
    index = _load_cache_index(index_path)
    if len(index) <= max_items:
        return

    # Sort by access time, oldest first
    sorted_items = sorted(index.items(), key=lambda x: x[1])
    items_to_remove = len(index) - max_items

    for item_id, _ in sorted_items[:items_to_remove]:
        cache_file = cache_dir / f"{item_id}.{extension}"
        if cache_file.exists():
            cache_file.unlink()
        del index[item_id]

    _save_cache_index(index_path, index)


# ============================================================================
# Route fetching
# ============================================================================

def _get_cached_route_path(route_id: int) -> Path | None:
    """Get cached route GPX path if it exists."""
    cache_path = STRAVA_ROUTES_DIR / f"{route_id}.gpx"
    if cache_path.exists():
        return cache_path
    return None


def _download_route_gpx(route_id: int) -> bytes:
    """Download GPX data for a Strava route.

    Raises:
        requests.RequestException: If the download fails.
    """
    url = f"{STRAVA_API_BASE}/routes/{route_id}/export_gpx"
    headers = _get_auth_headers()

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return response.content


def _save_route_to_cache(route_id: int, gpx_data: bytes) -> Path:
    """Save route GPX to cache."""
    _ensure_cache_dirs()
    cache_path = STRAVA_ROUTES_DIR / f"{route_id}.gpx"
    cache_path.write_bytes(gpx_data)
    _update_lru(STRAVA_ROUTES_CACHE_INDEX, str(route_id))
    _enforce_lru_limit(
        STRAVA_ROUTES_CACHE_INDEX,
        STRAVA_ROUTES_DIR,
        MAX_CACHED_STRAVA_ROUTES,
        "gpx",
    )
    return cache_path


def get_strava_route(url: str, baseline_crr: float = 0.005) -> tuple[list[TrackPoint], dict]:
    """Fetch and parse a Strava route.

    Args:
        url: Strava route URL
        baseline_crr: Baseline rolling resistance coefficient (not used for Strava,
            but kept for API compatibility with RideWithGPS)

    Returns:
        Tuple of (list of TrackPoints, metadata dict)

    Raises:
        ValueError: If the URL is not a valid Strava route URL.
        requests.RequestException: If the download fails.
    """
    route_id = extract_strava_route_id(url)

    # Check cache first
    cached_path = _get_cached_route_path(route_id)
    if cached_path:
        _update_lru(STRAVA_ROUTES_CACHE_INDEX, str(route_id))
        points, name = parse_gpx(str(cached_path))
    else:
        gpx_data = _download_route_gpx(route_id)
        cache_path = _save_route_to_cache(route_id, gpx_data)
        points, name = parse_gpx(str(cache_path))

    # Get route metadata from API
    metadata = _get_route_metadata(route_id)
    if name and not metadata.get("name"):
        metadata["name"] = name

    # Strava doesn't provide surface data, so all points have default CRR
    for point in points:
        point.crr = baseline_crr
        point.unpaved = False

    return points, metadata


def _get_route_metadata(route_id: int) -> dict:
    """Get route metadata from Strava API."""
    url = f"{STRAVA_API_BASE}/routes/{route_id}"
    headers = _get_auth_headers()

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        return {
            "name": data.get("name", ""),
            "distance": data.get("distance", 0),  # meters
            "elevation_gain": data.get("elevation_gain", 0),  # meters
            "unpaved_pct": 0,  # Strava doesn't provide this
            "source": "strava",
        }
    except requests.RequestException as e:
        logger.warning("Failed to get Strava route metadata: %s", e)
        return {"source": "strava"}


# ============================================================================
# Activity fetching
# ============================================================================

def _get_cached_activity_path(activity_id: int) -> Path | None:
    """Get cached activity JSON path if it exists."""
    cache_path = STRAVA_ACTIVITIES_DIR / f"{activity_id}.json"
    if cache_path.exists():
        return cache_path
    return None


def _download_activity_data(activity_id: int) -> dict:
    """Download activity metadata and streams from Strava.

    Returns:
        Dict with 'metadata' and 'streams' keys.

    Raises:
        requests.RequestException: If the download fails.
    """
    headers = _get_auth_headers()

    # Get activity metadata
    metadata_url = f"{STRAVA_API_BASE}/activities/{activity_id}"
    metadata_response = requests.get(metadata_url, headers=headers, timeout=30)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()

    # Get activity streams
    streams_url = f"{STRAVA_API_BASE}/activities/{activity_id}/streams"
    streams_params = {
        "keys": "latlng,altitude,time,distance,velocity_smooth,watts,heartrate,cadence",
        "key_type": "distance",
    }
    streams_response = requests.get(
        streams_url, headers=headers, params=streams_params, timeout=30
    )
    streams_response.raise_for_status()
    streams = streams_response.json()

    return {
        "metadata": metadata,
        "streams": streams,
    }


def _save_activity_to_cache(activity_id: int, data: dict) -> Path:
    """Save activity data to cache."""
    _ensure_cache_dirs()
    cache_path = STRAVA_ACTIVITIES_DIR / f"{activity_id}.json"
    with cache_path.open("w") as f:
        json.dump(data, f)
    _update_lru(STRAVA_ACTIVITIES_CACHE_INDEX, str(activity_id))
    _enforce_lru_limit(
        STRAVA_ACTIVITIES_CACHE_INDEX,
        STRAVA_ACTIVITIES_DIR,
        MAX_CACHED_STRAVA_ACTIVITIES,
        "json",
    )
    return cache_path


def _parse_activity_streams(data: dict) -> tuple[list[TripPoint], dict]:
    """Parse activity streams into TripPoints.

    Args:
        data: Dict with 'metadata' and 'streams' keys.

    Returns:
        Tuple of (list of TripPoints, metadata dict)
    """
    metadata = data["metadata"]
    streams = data["streams"]

    # Build lookup for streams by type
    stream_data = {}
    for stream in streams:
        stream_data[stream["type"]] = stream["data"]

    # Get arrays (some may be missing)
    latlng = stream_data.get("latlng", [])
    altitude = stream_data.get("altitude", [])
    time_arr = stream_data.get("time", [])
    distance = stream_data.get("distance", [])
    velocity = stream_data.get("velocity_smooth", [])
    watts = stream_data.get("watts", [])
    heartrate = stream_data.get("heartrate", [])
    cadence = stream_data.get("cadence", [])

    # Get activity start time for timestamp calculation
    start_time = metadata.get("start_date_local")
    if start_time:
        from datetime import datetime
        try:
            # Parse ISO format: "2024-01-15T10:30:00Z"
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            start_timestamp = start_dt.timestamp()
        except ValueError:
            start_timestamp = None
    else:
        start_timestamp = None

    # Build TripPoints
    points = []
    num_points = len(latlng) if latlng else len(distance)

    for i in range(num_points):
        lat = latlng[i][0] if i < len(latlng) else 0
        lon = latlng[i][1] if i < len(latlng) else 0
        elev = altitude[i] if i < len(altitude) else None
        dist = distance[i] if i < len(distance) else 0
        speed = velocity[i] if i < len(velocity) else None
        power = watts[i] if i < len(watts) else None
        hr = heartrate[i] if i < len(heartrate) else None
        cad = cadence[i] if i < len(cadence) else None

        # Calculate timestamp from start time + elapsed seconds
        if start_timestamp and i < len(time_arr):
            timestamp = start_timestamp + time_arr[i]
        else:
            timestamp = None

        points.append(TripPoint(
            lat=lat,
            lon=lon,
            elevation=elev,
            distance=dist,
            speed=speed,
            timestamp=timestamp,
            power=power,
            heart_rate=hr,
            cadence=cad,
        ))

    # Build metadata
    result_metadata = {
        "name": metadata.get("name", ""),
        "distance": metadata.get("distance", 0),
        "elevation_gain": metadata.get("total_elevation_gain", 0),
        "moving_time": metadata.get("moving_time", 0),
        "duration": metadata.get("elapsed_time", 0),
        "avg_speed": metadata.get("average_speed", 0),
        "avg_watts": metadata.get("average_watts"),
        "source": "strava",
    }

    return points, result_metadata


def get_strava_activity(url: str) -> tuple[list[TripPoint], dict]:
    """Fetch and parse a Strava activity.

    Args:
        url: Strava activity URL

    Returns:
        Tuple of (list of TripPoints, metadata dict)

    Raises:
        ValueError: If the URL is not a valid Strava activity URL.
        requests.RequestException: If the download fails.
    """
    activity_id = extract_strava_activity_id(url)

    # Check cache first
    cached_path = _get_cached_activity_path(activity_id)
    if cached_path:
        _update_lru(STRAVA_ACTIVITIES_CACHE_INDEX, str(activity_id))
        with cached_path.open() as f:
            data = json.load(f)
    else:
        data = _download_activity_data(activity_id)
        _save_activity_to_cache(activity_id, data)

    return _parse_activity_streams(data)
