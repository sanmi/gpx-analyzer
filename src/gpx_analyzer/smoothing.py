from bisect import bisect_left, bisect_right

from gpx_analyzer.distance import haversine_distance
from gpx_analyzer.models import TrackPoint


def _local_linear_regression(distances: list[float], elevations: list[float], target_dist: float) -> float:
    """Fit a local linear regression and return the fitted value at target_dist.

    Uses simple least squares: y = slope * x + intercept
    Returns the fitted elevation at target_dist.
    """
    n = len(distances)
    if n == 0:
        return 0.0
    if n == 1:
        return elevations[0]

    # Calculate means
    x_mean = sum(distances) / n
    y_mean = sum(elevations) / n

    # Calculate slope: Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(distances, elevations))
    denominator = sum((x - x_mean) ** 2 for x in distances)

    if denominator == 0:
        # All x values are the same, return mean elevation
        return y_mean

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope * target_dist + intercept


def smooth_elevations(
    points: list[TrackPoint], radius_m: float = 50.0, elevation_scale: float = 1.0
) -> list[TrackPoint]:
    """Smooth elevation values using local linear regression.

    For each point, fits a line to all points within
    [cumulative_dist - radius_m, cumulative_dist + radius_m] and uses the
    fitted value at the center point. This preserves local trends/slopes
    while smoothing out noise and outliers.
    Points with elevation=None are left unchanged.

    If elevation_scale != 1.0, elevation changes are scaled relative to the
    starting elevation. For example, scale=0.81 reduces all elevation deltas
    by 19%, useful when GPS elevation data overstates the actual gain.

    Returns new TrackPoint instances with smoothed elevations;
    lat, lon, and time are preserved.
    """
    if len(points) < 2:
        return list(points)

    if radius_m <= 0 and elevation_scale == 1.0:
        return list(points)

    # Build cumulative distance array
    cum_dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon,
        )
        cum_dist.append(cum_dist[-1] + d)

    # Collect indices of points that have elevation data
    has_elev = [i for i in range(len(points)) if points[i].elevation is not None]
    elev_dists = [cum_dist[i] for i in has_elev]
    elev_values = [points[i].elevation for i in has_elev]

    smoothed = []
    for i, pt in enumerate(points):
        if pt.elevation is None:
            smoothed.append(pt)
            continue

        d = cum_dist[i]
        lo = bisect_left(elev_dists, d - radius_m)
        hi = bisect_right(elev_dists, d + radius_m)

        if hi > lo:
            # Fit local linear regression and get fitted value at this distance
            window_dists = elev_dists[lo:hi]
            window_elevs = elev_values[lo:hi]
            smoothed_elev = _local_linear_regression(window_dists, window_elevs, d)
        else:
            smoothed_elev = pt.elevation

        smoothed.append(
            TrackPoint(lat=pt.lat, lon=pt.lon, elevation=smoothed_elev, time=pt.time, crr=pt.crr, unpaved=pt.unpaved, curvature=pt.curvature)
        )

    # Apply elevation scaling if needed
    if elevation_scale != 1.0:
        # Find reference elevation (first point with elevation data)
        ref_elev = None
        for pt in smoothed:
            if pt.elevation is not None:
                ref_elev = pt.elevation
                break

        if ref_elev is not None:
            scaled = []
            for pt in smoothed:
                if pt.elevation is None:
                    scaled.append(pt)
                else:
                    new_elev = ref_elev + (pt.elevation - ref_elev) * elevation_scale
                    scaled.append(
                        TrackPoint(lat=pt.lat, lon=pt.lon, elevation=new_elev, time=pt.time, crr=pt.crr, unpaved=pt.unpaved, curvature=pt.curvature)
                    )
            return scaled

    return smoothed
