"""RideWithGPS URL support with local GPX caching."""

from dataclasses import dataclass
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from gpx_analyzer.models import TrackPoint

CONFIG_DIR = Path.home() / ".config" / "gpx-analyzer"
CONFIG_PATH = CONFIG_DIR / "gpx-analyzer.json"
CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer"
ROUTES_DIR = CACHE_DIR / "routes"
ROUTES_JSON_DIR = CACHE_DIR / "routes_json"
TRIPS_DIR = CACHE_DIR / "trips"
CACHE_INDEX_PATH = CACHE_DIR / "cache_index.json"
ROUTE_JSON_CACHE_INDEX_PATH = CACHE_DIR / "route_json_cache_index.json"
MAX_CACHED_ROUTES = 10
MAX_CACHED_ROUTE_JSON = 50
MAX_CACHED_TRIPS = 20

RIDEWITHGPS_PATTERN = re.compile(r"^https?://(?:www\.)?ridewithgps\.com/routes/(\d+)")
RIDEWITHGPS_TRIP_PATTERN = re.compile(r"^https?://(?:www\.)?ridewithgps\.com/trips/(\d+)")
RIDEWITHGPS_COLLECTION_PATTERN = re.compile(r"^https?://(?:www\.)?ridewithgps\.com/collections/(\d+)")


def is_ridewithgps_url(path: str) -> bool:
    """Check if the given path is a RideWithGPS URL."""
    return bool(RIDEWITHGPS_PATTERN.match(path))


def extract_route_id(url: str) -> int:
    """Extract the route ID from a RideWithGPS URL.

    Raises:
        ValueError: If the URL is not a valid RideWithGPS route URL.
    """
    match = RIDEWITHGPS_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid RideWithGPS URL: {url}")
    return int(match.group(1))


def extract_privacy_code(url: str) -> str | None:
    """Extract the privacy_code query parameter from a URL, if present."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    codes = params.get("privacy_code", [])
    return codes[0] if codes else None


def get_gpx(url: str) -> str:
    """Get the local path to a GPX file for a RideWithGPS URL.

    Downloads and caches the GPX file if not already cached.
    Updates LRU access time on cache hit.

    Returns:
        Path to the local GPX file.

    Raises:
        ValueError: If the URL is not a valid RideWithGPS URL.
        requests.RequestException: If the download fails.
    """
    route_id = extract_route_id(url)
    privacy_code = extract_privacy_code(url)

    cached_path = _get_cached_path(route_id)
    if cached_path is not None:
        _update_lru(route_id)
        return str(cached_path)

    gpx_data = _download_gpx(route_id, privacy_code)
    path = _save_to_cache(route_id, gpx_data)
    return str(path)


LOCAL_CONFIG_PATH = Path("gpx-analyzer.json")


def _load_config() -> dict:
    """Load configuration from config files.

    Merges config from global and local files:
    1. ~/.config/gpx-analyzer/gpx-analyzer.json (global, loaded first)
    2. ./gpx-analyzer.json (local, overrides global)

    This allows credentials in global config with project-specific settings in local.

    Returns:
        Dict with merged config values, empty dict if no files exist.
    """
    config = {}
    # Load global first, then local overrides
    for config_path in [CONFIG_PATH, LOCAL_CONFIG_PATH]:
        if config_path.exists():
            try:
                with config_path.open() as f:
                    config.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return config


def _get_auth_headers() -> dict[str, str]:
    """Get authentication headers from config file or environment variables.

    Checks config file first (~/.config/gpx-analyzer/gpx-analyzer.json), then
    falls back to environment variables.

    Config file format:
        {
            "ridewithgps_api_key": "your-api-key",
            "ridewithgps_auth_token": "your-auth-token"
        }

    Environment variables:
        RIDEWITHGPS_API_KEY, RIDEWITHGPS_AUTH_TOKEN

    Returns:
        Dict with auth headers if credentials are set, empty dict otherwise.
    """
    config = _load_config()

    api_key = config.get("ridewithgps_api_key") or os.environ.get("RIDEWITHGPS_API_KEY")
    auth_token = config.get("ridewithgps_auth_token") or os.environ.get(
        "RIDEWITHGPS_AUTH_TOKEN"
    )

    if api_key and auth_token:
        return {
            "x-rwgps-api-key": api_key,
            "x-rwgps-auth-token": auth_token,
        }
    return {}


def _download_gpx(route_id: int, privacy_code: str | None = None) -> bytes:
    """Download GPX data from RideWithGPS.

    Uses authentication headers from environment variables if available:
    - RIDEWITHGPS_API_KEY: Your API client key
    - RIDEWITHGPS_AUTH_TOKEN: Your authentication token

    Raises:
        requests.RequestException: If the download fails.
    """
    url = f"https://ridewithgps.com/routes/{route_id}.gpx?sub_format=track"
    if privacy_code:
        url += f"&privacy_code={privacy_code}"

    headers = _get_auth_headers()
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


def _get_cached_path(route_id: int) -> Path | None:
    """Get the cached file path if it exists, None otherwise."""
    path = ROUTES_DIR / f"{route_id}.gpx"
    if path.exists():
        return path
    return None


def _save_to_cache(route_id: int, gpx_data: bytes) -> Path:
    """Save GPX data to cache and update the index."""
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)

    path = ROUTES_DIR / f"{route_id}.gpx"
    path.write_bytes(gpx_data)

    _update_lru(route_id)
    _enforce_lru_limit()

    return path


def _load_cache_index() -> dict[str, float]:
    """Load the cache index from disk."""
    if not CACHE_INDEX_PATH.exists():
        return {}
    try:
        with CACHE_INDEX_PATH.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache_index(index: dict[str, float]) -> None:
    """Save the cache index to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_INDEX_PATH.open("w") as f:
        json.dump(index, f)


def _update_lru(route_id: int) -> None:
    """Update the LRU access time for a route."""
    index = _load_cache_index()
    index[str(route_id)] = time.time()
    _save_cache_index(index)


def _enforce_lru_limit() -> None:
    """Remove oldest cached files if over the limit."""
    index = _load_cache_index()

    if len(index) <= MAX_CACHED_ROUTES:
        return

    sorted_entries = sorted(index.items(), key=lambda x: x[1])
    to_remove = sorted_entries[: len(index) - MAX_CACHED_ROUTES]

    for route_id_str, _ in to_remove:
        path = ROUTES_DIR / f"{route_id_str}.gpx"
        if path.exists():
            path.unlink()
        del index[route_id_str]

    _save_cache_index(index)


# Trip caching functions (similar to route caching but for JSON trip data)
TRIP_CACHE_INDEX_PATH = CACHE_DIR / "trip_cache_index.json"


def _get_cached_trip_path(trip_id: int) -> Path | None:
    """Get the cached trip file path if it exists, None otherwise."""
    path = TRIPS_DIR / f"{trip_id}.json"
    if path.exists():
        return path
    return None


def _save_trip_to_cache(trip_id: int, trip_data: dict) -> Path:
    """Save trip JSON data to cache and update the index."""
    TRIPS_DIR.mkdir(parents=True, exist_ok=True)

    path = TRIPS_DIR / f"{trip_id}.json"
    with path.open("w") as f:
        json.dump(trip_data, f)

    _update_trip_lru(trip_id)
    _enforce_trip_lru_limit()

    return path


def _load_trip_cache_index() -> dict[str, float]:
    """Load the trip cache index from disk."""
    if not TRIP_CACHE_INDEX_PATH.exists():
        return {}
    try:
        with TRIP_CACHE_INDEX_PATH.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_trip_cache_index(index: dict[str, float]) -> None:
    """Save the trip cache index to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with TRIP_CACHE_INDEX_PATH.open("w") as f:
        json.dump(index, f)


def _update_trip_lru(trip_id: int) -> None:
    """Update the LRU access time for a trip."""
    index = _load_trip_cache_index()
    index[str(trip_id)] = time.time()
    _save_trip_cache_index(index)


def _enforce_trip_lru_limit() -> None:
    """Remove oldest cached trip files if over the limit."""
    index = _load_trip_cache_index()

    if len(index) <= MAX_CACHED_TRIPS:
        return

    sorted_entries = sorted(index.items(), key=lambda x: x[1])
    to_remove = sorted_entries[: len(index) - MAX_CACHED_TRIPS]

    for trip_id_str, _ in to_remove:
        path = TRIPS_DIR / f"{trip_id_str}.json"
        if path.exists():
            path.unlink()
        del index[trip_id_str]

    _save_trip_cache_index(index)


def _load_cached_trip(trip_id: int) -> dict | None:
    """Load trip data from cache if available."""
    path = _get_cached_trip_path(trip_id)
    if path is None:
        return None
    try:
        with path.open() as f:
            _update_trip_lru(trip_id)
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# Route JSON caching functions
def _get_cached_route_json_path(route_id: int) -> Path | None:
    """Get the cached route JSON file path if it exists, None otherwise."""
    path = ROUTES_JSON_DIR / f"{route_id}.json"
    if path.exists():
        return path
    return None


def _save_route_json_to_cache(route_id: int, route_data: dict) -> Path:
    """Save route JSON data to cache and update the index."""
    ROUTES_JSON_DIR.mkdir(parents=True, exist_ok=True)

    path = ROUTES_JSON_DIR / f"{route_id}.json"
    with path.open("w") as f:
        json.dump(route_data, f)

    _update_route_json_lru(route_id)
    _enforce_route_json_lru_limit()

    return path


def _load_route_json_cache_index() -> dict[str, float]:
    """Load the route JSON cache index from disk."""
    if not ROUTE_JSON_CACHE_INDEX_PATH.exists():
        return {}
    try:
        with ROUTE_JSON_CACHE_INDEX_PATH.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_route_json_cache_index(index: dict[str, float]) -> None:
    """Save the route JSON cache index to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ROUTE_JSON_CACHE_INDEX_PATH.open("w") as f:
        json.dump(index, f)


def _update_route_json_lru(route_id: int) -> None:
    """Update the LRU access time for a route JSON."""
    index = _load_route_json_cache_index()
    index[str(route_id)] = time.time()
    _save_route_json_cache_index(index)


def _enforce_route_json_lru_limit() -> None:
    """Remove oldest cached route JSON files if over the limit."""
    index = _load_route_json_cache_index()

    if len(index) <= MAX_CACHED_ROUTE_JSON:
        return

    sorted_entries = sorted(index.items(), key=lambda x: x[1])
    to_remove = sorted_entries[: len(index) - MAX_CACHED_ROUTE_JSON]

    for route_id_str, _ in to_remove:
        path = ROUTES_JSON_DIR / f"{route_id_str}.json"
        if path.exists():
            path.unlink()
        del index[route_id_str]

    _save_route_json_cache_index(index)


def _load_cached_route_json(route_id: int) -> dict | None:
    """Load route JSON data from cache if available."""
    path = _get_cached_route_json_path(route_id)
    if path is None:
        return None
    try:
        with path.open() as f:
            _update_route_json_lru(route_id)
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# Surface type crr deltas from baseline (R=3 quality paved is the baseline)
# R values from RideWithGPS JSON route data
# Actual crr = baseline_crr + R_delta + (unpaved_delta if S >= 50)
SURFACE_CRR_DELTAS: dict[int, float] = {
    3: 0.0,     # Paved (quality) - baseline
    4: 0.001,   # Paved (standard)
    5: 0.001,   # Paved
    6: 0.001,   # Paved
    15: 0.006,  # Gravel/unpaved
    25: 0.008,  # Rough gravel
}

# S values >= this threshold indicate unpaved surfaces
UNPAVED_S_THRESHOLD = 50

# Additional crr delta for unpaved surfaces (S >= threshold)
UNPAVED_CRR_DELTA = 0.005


def _get_surface_crr_deltas() -> dict[int, float]:
    """Get surface crr deltas, allowing config file override.

    Config file can specify 'surface_crr_deltas' as a dict mapping
    R values (as strings) to delta values.
    """
    config = _load_config()
    config_deltas = config.get("surface_crr_deltas")

    if config_deltas and isinstance(config_deltas, dict):
        # Convert string keys to int (JSON keys are strings)
        return {int(k): float(v) for k, v in config_deltas.items()}

    return SURFACE_CRR_DELTAS


def _get_unpaved_crr_delta() -> float:
    """Get the additional crr delta for unpaved surfaces.

    Config file can specify 'unpaved_crr_delta' to override the default.
    """
    config = _load_config()
    return config.get("unpaved_crr_delta", UNPAVED_CRR_DELTA)


def _surface_type_to_crr(
    r_value: int | None, s_value: int | None, baseline_crr: float
) -> float:
    """Convert RideWithGPS surface values to rolling resistance coefficient.

    Args:
        r_value: The R surface type value from RideWithGPS (road quality)
        s_value: The S surface value from RideWithGPS (S >= 50 indicates unpaved)
        baseline_crr: The baseline crr (used for R=3 quality paved)

    Returns:
        The effective crr = baseline_crr + R_delta + unpaved_delta.
        Unknown R values use delta=0 (baseline).
        S values >= 50 add an additional unpaved penalty.
    """
    # Start with baseline
    crr = baseline_crr

    # Add R-based delta for road quality
    if r_value is not None:
        deltas = _get_surface_crr_deltas()
        crr += deltas.get(r_value, 0.0)

    # Add unpaved penalty if S >= threshold
    if s_value is not None and s_value >= UNPAVED_S_THRESHOLD:
        crr += _get_unpaved_crr_delta()

    return crr


def is_unpaved(s_value: int | None) -> bool:
    """Check if an S value indicates an unpaved surface."""
    return s_value is not None and s_value >= UNPAVED_S_THRESHOLD


def _download_json(route_id: int, privacy_code: str | None = None) -> dict:
    """Download route JSON data from RideWithGPS.

    Raises:
        requests.RequestException: If the download fails.
    """
    url = f"https://ridewithgps.com/routes/{route_id}.json"
    if privacy_code:
        url += f"?privacy_code={privacy_code}"

    headers = _get_auth_headers()
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_json_track_points(route_data: dict, baseline_crr: float) -> list[TrackPoint]:
    """Parse RideWithGPS JSON route data into TrackPoints with surface crr.

    Extracts track points from the 'track_points' array. Surface data is read
    from the 'R' field (road quality) and 'S' field (S >= 50 indicates unpaved).

    The route_data can have track_points at the top level (API format) or
    nested under 'route' key.

    Args:
        route_data: The JSON route data from RideWithGPS API
        baseline_crr: The baseline crr value (from config/CLI) used for R=3 quality paved
    """
    # Handle both top-level and nested route data
    if "route" in route_data and "track_points" in route_data.get("route", {}):
        track_points_data = route_data["route"]["track_points"]
    else:
        track_points_data = route_data.get("track_points", [])

    if not track_points_data:
        return []

    points: list[TrackPoint] = []
    for tp in track_points_data:
        lat = tp.get("y")
        lon = tp.get("x")
        elevation = tp.get("e")

        if lat is None or lon is None:
            continue

        # Get surface values from track point
        r_value = tp.get("R")
        s_value = tp.get("S")

        # Calculate crr if we have surface data
        if r_value is not None or s_value is not None:
            crr = _surface_type_to_crr(r_value, s_value, baseline_crr)
            unpaved = is_unpaved(s_value)
        else:
            crr = None
            unpaved = False

        points.append(
            TrackPoint(
                lat=lat,
                lon=lon,
                elevation=elevation,
                time=None,
                crr=crr,
                unpaved=unpaved,
            )
        )

    return points


def get_route_with_surface(url: str, baseline_crr: float) -> tuple[list[TrackPoint], dict]:
    """Get route track points with surface data from RideWithGPS JSON API.

    Downloads and caches the route JSON if not already cached.
    Updates LRU access time on cache hit.

    Args:
        url: The RideWithGPS route URL
        baseline_crr: The baseline crr value (from config/CLI) used for R=3 quality paved

    Returns:
        Tuple of (list of TrackPoints with crr, route metadata dict).

    Raises:
        ValueError: If the URL is not a valid RideWithGPS URL.
        requests.RequestException: If the download fails.
    """
    route_id = extract_route_id(url)
    privacy_code = extract_privacy_code(url)

    # Check cache first
    route_data = _load_cached_route_json(route_id)
    if route_data is None:
        route_data = _download_json(route_id, privacy_code)
        _save_route_json_to_cache(route_id, route_data)

    points = parse_json_track_points(route_data, baseline_crr)

    # Extract useful metadata - handle both top-level and nested formats
    if "route" in route_data:
        route_info = route_data["route"]
    else:
        route_info = route_data

    metadata = {
        "name": route_info.get("name"),
        "distance": route_info.get("distance"),  # meters
        "elevation_gain": route_info.get("elevation_gain"),
        "unpaved_pct": route_info.get("unpaved_pct"),
        "surface": route_info.get("surface"),
    }

    return points, metadata


def is_ridewithgps_trip_url(path: str) -> bool:
    """Check if the given path is a RideWithGPS trip URL."""
    return bool(RIDEWITHGPS_TRIP_PATTERN.match(path))


def extract_trip_id(url: str) -> int:
    """Extract the trip ID from a RideWithGPS trip URL.

    Raises:
        ValueError: If the URL is not a valid RideWithGPS trip URL.
    """
    match = RIDEWITHGPS_TRIP_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid RideWithGPS trip URL: {url}")
    return int(match.group(1))


def _download_trip_json(trip_id: int, privacy_code: str | None = None) -> dict:
    """Download trip JSON data from RideWithGPS.

    Raises:
        requests.RequestException: If the download fails.
    """
    url = f"https://ridewithgps.com/trips/{trip_id}.json"
    if privacy_code:
        url += f"?privacy_code={privacy_code}"

    headers = _get_auth_headers()
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


@dataclass
class TripPoint:
    """A point from an actual ride with recorded data."""

    lat: float
    lon: float
    elevation: float | None
    distance: float  # cumulative distance in meters
    speed: float | None  # m/s
    timestamp: float | None  # unix timestamp
    power: float | None  # watts
    heart_rate: int | None
    cadence: int | None


def parse_trip_track_points(trip_data: dict) -> list[TripPoint]:
    """Parse RideWithGPS trip JSON data into TripPoints.

    Args:
        trip_data: The JSON trip data from RideWithGPS API
    """
    # Handle both top-level and nested trip data
    if "trip" in trip_data and "track_points" in trip_data.get("trip", {}):
        track_points_data = trip_data["trip"]["track_points"]
    else:
        track_points_data = trip_data.get("track_points", [])

    if not track_points_data:
        return []

    points: list[TripPoint] = []
    for tp in track_points_data:
        lat = tp.get("y")
        lon = tp.get("x")

        if lat is None or lon is None:
            continue

        points.append(
            TripPoint(
                lat=lat,
                lon=lon,
                elevation=tp.get("e"),
                distance=tp.get("d", 0.0),
                speed=tp.get("s"),
                timestamp=tp.get("t"),
                power=tp.get("p"),
                heart_rate=tp.get("h"),
                cadence=tp.get("c"),
            )
        )

    return points


def get_trip_data(url: str) -> tuple[list[TripPoint], dict]:
    """Get trip track points from RideWithGPS JSON API.

    Downloads and caches the trip JSON if not already cached.
    Updates LRU access time on cache hit.

    Args:
        url: The RideWithGPS trip URL

    Returns:
        Tuple of (list of TripPoints, trip metadata dict).

    Raises:
        ValueError: If the URL is not a valid RideWithGPS trip URL.
        requests.RequestException: If the download fails.
    """
    trip_id = extract_trip_id(url)
    privacy_code = extract_privacy_code(url)

    # Check cache first
    trip_data = _load_cached_trip(trip_id)
    if trip_data is None:
        trip_data = _download_trip_json(trip_id, privacy_code)
        _save_trip_to_cache(trip_id, trip_data)

    points = parse_trip_track_points(trip_data)

    # Extract useful metadata
    if "trip" in trip_data:
        trip_info = trip_data["trip"]
    else:
        trip_info = trip_data

    metadata = {
        "name": trip_info.get("name"),
        "distance": trip_info.get("distance"),
        "elevation_gain": trip_info.get("elevation_gain"),
        "moving_time": trip_info.get("moving_time"),
        "duration": trip_info.get("duration"),
        "avg_speed": trip_info.get("avg_speed"),
        "avg_watts": trip_info.get("avg_watts"),
    }

    return points, metadata


def is_ridewithgps_collection_url(path: str) -> bool:
    """Check if the given path is a RideWithGPS collection URL."""
    return bool(RIDEWITHGPS_COLLECTION_PATTERN.match(path))


def extract_collection_id(url: str) -> int:
    """Extract the collection ID from a RideWithGPS collection URL.

    Raises:
        ValueError: If the URL is not a valid RideWithGPS collection URL.
    """
    match = RIDEWITHGPS_COLLECTION_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid RideWithGPS collection URL: {url}")
    return int(match.group(1))


def get_collection_route_ids(url: str) -> tuple[list[int], str | None]:
    """Get route IDs from a RideWithGPS collection via the API.

    Args:
        url: The RideWithGPS collection URL

    Returns:
        Tuple of (list of route IDs, collection name or None).

    Raises:
        ValueError: If the URL is not a valid RideWithGPS collection URL.
        requests.RequestException: If the download fails.
    """
    collection_id = extract_collection_id(url)
    privacy_code = extract_privacy_code(url)

    headers = _get_auth_headers()

    params = {}
    if privacy_code:
        params["privacy_code"] = privacy_code

    resp = requests.get(
        f"https://ridewithgps.com/api/v1/collections/{collection_id}.json",
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    collection = data.get("collection", {})

    collection_name = collection.get("name")
    routes = collection.get("routes", [])
    route_ids = [r["id"] for r in routes if "id" in r]

    return route_ids, collection_name
