import gpxpy

from gpx_analyzer.models import TrackPoint


def parse_gpx(filepath: str) -> list[TrackPoint]:
    """Parse a GPX file and return a list of TrackPoints."""
    with open(filepath, "r") as f:
        gpx = gpxpy.parse(f)

    points: list[TrackPoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append(
                    TrackPoint(
                        lat=pt.latitude,
                        lon=pt.longitude,
                        elevation=pt.elevation,
                        time=pt.time,
                    )
                )
    return points
