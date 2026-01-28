"""RideWithGPS URL support with local GPX caching."""

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

CONFIG_DIR = Path.home() / ".config" / "gpx-analyzer"
CONFIG_PATH = CONFIG_DIR / "gpx-analyzer.json"
CACHE_DIR = Path.home() / ".cache" / "gpx-analyzer"
ROUTES_DIR = CACHE_DIR / "routes"
CACHE_INDEX_PATH = CACHE_DIR / "cache_index.json"
MAX_CACHED_ROUTES = 10

RIDEWITHGPS_PATTERN = re.compile(r"^https?://(?:www\.)?ridewithgps\.com/routes/(\d+)")


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
    """Load configuration from config file.

    Checks for config files in order:
    1. ./gpx-analyzer.json (project root)
    2. ~/.config/gpx-analyzer/gpx-analyzer.json (global)

    Returns:
        Dict with config values, empty dict if no file exists or is invalid.
    """
    for config_path in [LOCAL_CONFIG_PATH, CONFIG_PATH]:
        if config_path.exists():
            try:
                with config_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return {}


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
