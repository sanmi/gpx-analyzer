"""Fetch elevation data from DEM APIs."""

import time
from dataclasses import dataclass

import requests

from gpx_analyzer.models import TrackPoint


@dataclass
class ElevationResult:
    """Result of fetching DEM elevation for a route."""

    points: list[TrackPoint]  # Points with DEM elevation
    dem_elevation_gain: float
    dem_elevation_loss: float
    original_elevation_gain: float
    original_elevation_loss: float


# API endpoints
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_TOPO_DATA_URL = "https://api.opentopodata.org/v1/srtm30m"

# Batch size limits
OPEN_ELEVATION_BATCH_SIZE = 100
OPEN_TOPO_DATA_BATCH_SIZE = 100


def fetch_dem_elevation(
    points: list[TrackPoint],
    api: str = "open-elevation",
    batch_size: int = 100,
) -> list[float]:
    """Fetch DEM elevation for a list of points.

    Args:
        points: List of TrackPoints with lat/lon
        api: Which API to use ("open-elevation" or "opentopodata")
        batch_size: Number of points per API request

    Returns:
        List of elevations in meters, same length as points.

    Raises:
        requests.RequestException: If API request fails.
    """
    if api == "open-elevation":
        return _fetch_open_elevation(points, batch_size)
    elif api == "opentopodata":
        return _fetch_opentopodata(points, batch_size)
    else:
        raise ValueError(f"Unknown elevation API: {api}")


def _fetch_open_elevation(points: list[TrackPoint], batch_size: int) -> list[float]:
    """Fetch elevation from Open-Elevation API."""
    elevations = []

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]

        # Build locations string
        locations = "|".join(f"{p.lat},{p.lon}" for p in batch)

        response = requests.get(
            OPEN_ELEVATION_URL,
            params={"locations": locations},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        for result in data.get("results", []):
            elevations.append(result.get("elevation", 0.0))

        # Rate limiting - be nice to free APIs
        if i + batch_size < len(points):
            time.sleep(0.5)

    return elevations


def _fetch_opentopodata(points: list[TrackPoint], batch_size: int) -> list[float]:
    """Fetch elevation from OpenTopoData API."""
    elevations = []

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]

        # Build locations string
        locations = "|".join(f"{p.lat},{p.lon}" for p in batch)

        response = requests.get(
            OPEN_TOPO_DATA_URL,
            params={"locations": locations},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        for result in data.get("results", []):
            elev = result.get("elevation")
            elevations.append(elev if elev is not None else 0.0)

        # Rate limiting - OpenTopoData asks for 1 req/sec
        if i + batch_size < len(points):
            time.sleep(1.0)

    return elevations


def apply_dem_elevation(
    points: list[TrackPoint],
    api: str = "open-elevation",
) -> ElevationResult:
    """Fetch DEM elevation and apply to points.

    Args:
        points: Original track points
        api: Which API to use

    Returns:
        ElevationResult with updated points and elevation stats.
    """
    # Calculate original elevation gain/loss
    original_gain, original_loss = _calculate_elevation_change(points)

    # Fetch DEM elevations
    dem_elevations = fetch_dem_elevation(points, api)

    # Create new points with DEM elevation
    new_points = []
    for pt, dem_elev in zip(points, dem_elevations):
        new_points.append(
            TrackPoint(
                lat=pt.lat,
                lon=pt.lon,
                elevation=dem_elev,
                time=pt.time,
                crr=pt.crr,
                unpaved=pt.unpaved,
            )
        )

    # Calculate DEM elevation gain/loss
    dem_gain, dem_loss = _calculate_elevation_change(new_points)

    return ElevationResult(
        points=new_points,
        dem_elevation_gain=dem_gain,
        dem_elevation_loss=dem_loss,
        original_elevation_gain=original_gain,
        original_elevation_loss=original_loss,
    )


def _calculate_elevation_change(points: list[TrackPoint]) -> tuple[float, float]:
    """Calculate total elevation gain and loss."""
    if len(points) < 2:
        return 0.0, 0.0

    gain = 0.0
    loss = 0.0

    for i in range(1, len(points)):
        elev_a = points[i - 1].elevation or 0.0
        elev_b = points[i].elevation or 0.0
        delta = elev_b - elev_a

        if delta > 0:
            gain += delta
        else:
            loss += abs(delta)

    return gain, loss


def compare_elevation_sources(
    points: list[TrackPoint],
    api: str = "open-elevation",
) -> dict:
    """Compare route elevation with DEM elevation.

    Returns dict with comparison statistics.
    """
    result = apply_dem_elevation(points, api)

    original_gain = result.original_elevation_gain
    dem_gain = result.dem_elevation_gain

    if dem_gain > 0:
        gain_diff_pct = ((original_gain - dem_gain) / dem_gain) * 100
    else:
        gain_diff_pct = 0.0

    return {
        "original_gain": original_gain,
        "dem_gain": dem_gain,
        "gain_difference": original_gain - dem_gain,
        "gain_difference_pct": gain_diff_pct,
        "original_loss": result.original_elevation_loss,
        "dem_loss": result.dem_elevation_loss,
        "points_count": len(points),
    }
