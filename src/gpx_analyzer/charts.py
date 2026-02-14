"""Elevation profile chart generation."""

import io

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from gpx_analyzer.analyzer import GRADE_BINS
from gpx_analyzer.climb import ClimbInfo


# Grade color mappings - matches histogram colors exactly
MAIN_GRADE_COLORS = [
    '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
    '#cccccc',
    '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
]

STEEP_GRADE_COLORS = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000']


def grade_to_color(g: float | None) -> str:
    """Map grade to histogram color.

    Args:
        g: Grade percentage. None indicates a stopped segment.

    Returns:
        Hex color string.
    """
    if g is None:
        return '#ffffff'  # White for stopped segments

    # For grades >= 10%, use steep histogram colors
    if g >= 10:
        if g < 12:
            return STEEP_GRADE_COLORS[0]
        elif g < 14:
            return STEEP_GRADE_COLORS[1]
        elif g < 16:
            return STEEP_GRADE_COLORS[2]
        elif g < 18:
            return STEEP_GRADE_COLORS[3]
        elif g < 20:
            return STEEP_GRADE_COLORS[4]
        else:
            return STEEP_GRADE_COLORS[5]

    # For grades < 10%, use main histogram colors
    for i, threshold in enumerate(GRADE_BINS[1:]):
        if g < threshold:
            return MAIN_GRADE_COLORS[i]
    return MAIN_GRADE_COLORS[-1]


def add_speed_overlay(ax, times_hours: list, speeds_kmh: list, imperial: bool = False,
                      max_speed_ylim: float | None = None):
    """Add speed line overlay with right Y-axis to an elevation profile plot.

    Args:
        ax: Primary matplotlib axis
        times_hours: X-axis time values
        speeds_kmh: Speed values in km/h
        imperial: If True, use imperial units (mph)
        max_speed_ylim: Optional max speed in km/h for synchronized Y-axis.
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


def add_grade_overlay(ax, times_hours: list, grades: list, max_grade_ylim: float | None = None):
    """Add grade line overlay with right Y-axis to an elevation profile plot.

    Args:
        ax: Primary matplotlib axis
        times_hours: X-axis time values
        grades: Grade values in percent
        max_grade_ylim: Optional max grade for synchronized Y-axis in comparison mode
    """
    # Filter out None grades (stopped segments in trips)
    grade_times = []
    grade_values = []
    for i, g in enumerate(grades):
        if g is not None and i < len(times_hours) - 1:
            grade_times.append((times_hours[i] + times_hours[i + 1]) / 2)
            grade_values.append(g)

    if not grade_values:
        ax.spines['right'].set_visible(False)
        return

    ax2 = ax.twinx()
    ax2.plot(grade_times, grade_values, color='#333333', linewidth=1.2, alpha=0.7)
    ax2.set_ylabel('Grade (%)', fontsize=10, color='#333333')
    ax2.tick_params(axis='y', labelcolor='#333333')

    # Calculate Y-axis limits
    actual_max = max(grade_values)
    actual_min = min(grade_values)
    tick_interval = 5

    if max_grade_ylim is not None:
        max_tick = int(max_grade_ylim / tick_interval + 1) * tick_interval
        min_tick = -max_tick  # Symmetric for comparison mode
    elif actual_min >= 0:
        max_tick = int(max(actual_max, 15) / tick_interval + 1) * tick_interval
        min_tick = 0
    else:
        max_tick = int(max(actual_max, 10) / tick_interval + 1) * tick_interval
        min_tick = int(min(actual_min, -10) / tick_interval - 1) * tick_interval

    ax2.set_yticks(range(min_tick, max_tick + 1, tick_interval))
    ax2.set_ylim(min_tick, max_tick)

    if min_tick < 0:
        ax2.axhline(y=0, color='#333333', linewidth=0.5, alpha=0.3, linestyle='--')

    ax2.spines['top'].set_visible(False)


def set_fixed_margins(fig, fig_width: float, fig_height: float) -> None:
    """Set fixed margins in inches for consistent JavaScript coordinate mapping.

    Uses fixed inch-based margins so that the plot area is predictable
    regardless of content.
    """
    left_margin_in = 0.7
    right_margin_in = 0.7
    bottom_margin_in = 0.55
    top_margin_in = 0.35

    left = left_margin_in / fig_width
    right = 1 - right_margin_in / fig_width
    bottom = bottom_margin_in / fig_height
    top = 1 - top_margin_in / fig_height

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def generate_elevation_profile(
    data: dict,
    overlay: str | None = None,
    imperial: bool = False,
    max_ylim: float | None = None,
    max_speed_ylim: float | None = None,
    max_grade_ylim: float | None = None,
    show_gravel: bool = False,
    aspect_ratio: float = 3.5,
    min_xlim_hours: float | None = None,
    max_xlim_hours: float | None = None,
) -> bytes:
    """Generate elevation profile image with grade-based coloring.

    Args:
        data: Profile data dict from calculate_route_profile_data or calculate_trip_profile_data
        overlay: Optional overlay type ("speed" or "grade")
        imperial: If True, use imperial units for overlay axis
        max_ylim: Optional max y-axis limit in meters
        max_speed_ylim: Optional max speed y-axis limit in km/h
        max_grade_ylim: Optional max grade y-axis limit in %
        show_gravel: If True, highlight unpaved sections
        aspect_ratio: Width/height ratio (1.0 = square, 3.5 = wide default)
        min_xlim_hours: Optional min x-axis limit in hours (for zooming)
        max_xlim_hours: Optional max x-axis limit in hours

    Returns PNG image as bytes.
    """
    times_hours = data["times_hours"]
    elevations = data["elevations"]
    grades = data["grades"]
    tunnel_time_ranges = data.get("tunnel_time_ranges", [])

    # Create figure with dynamic aspect ratio
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

    # Build all polygons and colors at once for efficient rendering
    polygons = []
    colors = []
    for i in range(len(grades)):
        t0, t1 = times_hours[i], times_hours[i+1]
        e0, e1 = elevations[i], elevations[i+1]
        polygons.append([(t0, 0), (t1, 0), (t1, e1), (t0, e0)])
        colors.append(grade_to_color(grades[i]))

    coll = PolyCollection(polygons, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(coll)

    # Add outline on top
    ax.plot(times_hours, elevations, color='#333333', linewidth=0.5)

    # Highlight anomaly-corrected regions
    max_elev = max_ylim if max_ylim is not None else max(elevations) * 1.1
    for start_time, end_time in tunnel_time_ranges:
        ax.axvspan(start_time, end_time, alpha=0.25, color='#FFC107', zorder=0.5,
                   label='Anomaly corrected' if start_time == tunnel_time_ranges[0][0] else None)
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, max_elev * 0.92, 'A', fontsize=9, fontweight='bold',
                color='#E65100', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=1))

    # Highlight unpaved sections
    if show_gravel:
        strip_height = max_elev * 0.03
        for start_time, end_time in data.get("unpaved_time_ranges", []):
            ax.fill_between([start_time, end_time], 0, strip_height,
                            color='#8B6914', alpha=0.5, zorder=1)

    # Style the plot
    x_min = min_xlim_hours if min_xlim_hours is not None else 0
    x_max = max_xlim_hours if max_xlim_hours is not None else times_hours[-1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_elev)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)

    ax.spines['top'].set_visible(False)

    if overlay == "speed" and data.get("speeds_kmh"):
        add_speed_overlay(ax, times_hours, data["speeds_kmh"], imperial, max_speed_ylim=max_speed_ylim)
    elif overlay == "grade" and data.get("grades"):
        add_grade_overlay(ax, times_hours, data["grades"], max_grade_ylim=max_grade_ylim)
    else:
        ax.spines['right'].set_visible(False)

    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    set_fixed_margins(fig, fig_width, fig_height)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_ride_profile(
    times_hours: list,
    elevations: list,
    grades: list,
    climbs: list[ClimbInfo],
    aspect_ratio: float = 1.0,
) -> bytes:
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
        ax.axvspan(climb.start_time_hours, climb.end_time_hours,
                   alpha=0.2, color='#4CAF50', zorder=0.5)

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

    set_fixed_margins(fig, fig_width, fig_height)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
