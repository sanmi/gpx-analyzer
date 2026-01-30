"""Simple web interface for GPX analyzer."""

from flask import Flask, render_template_string, request

from gpx_analyzer.analyzer import analyze
from gpx_analyzer.cli import calculate_elevation_gain, calculate_surface_breakdown, DEFAULTS
from gpx_analyzer.models import RiderParams
from gpx_analyzer.ridewithgps import (
    _load_config,
    get_collection_route_ids,
    get_route_with_surface,
    is_ridewithgps_collection_url,
    is_ridewithgps_url,
)
from gpx_analyzer.smoothing import smooth_elevations

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPX Analyzer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 700px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; font-size: 1.5em; }
        form {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: 600;
            color: #555;
        }
        label:first-child { margin-top: 0; }
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
            border-color: #007aff;
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
            background: #007aff;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; }
        .results {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results h2 { margin-top: 0; font-size: 1.2em; color: #333; }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .result-row:last-child { border-bottom: none; }
        .result-label { color: #666; }
        .result-value { font-weight: 600; color: #333; }
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
        .route-name {
            max-width: 180px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        @media (max-width: 600px) {
            .collection-table { font-size: 0.8em; }
            .collection-table th, .collection-table td { padding: 8px 4px; }
            .route-name { max-width: 120px; }
        }
    </style>
</head>
<body>
    <h1>GPX Route Analyzer</h1>

    <form method="POST">
        <label for="mode">Mode</label>
        <select id="mode" name="mode" onchange="toggleUrlPlaceholder()">
            <option value="route" {{ 'selected' if mode == 'route' else '' }}>Single Route</option>
            <option value="collection" {{ 'selected' if mode == 'collection' else '' }}>Collection</option>
        </select>

        <label for="url" id="url-label">RideWithGPS URL</label>
        <input type="text" id="url" name="url"
               placeholder="{{ 'https://ridewithgps.com/routes/...' if mode == 'route' else 'https://ridewithgps.com/collections/...' }}"
               value="{{ url or '' }}" required>

        <div class="param-row">
            <div>
                <label for="power">Power (W)</label>
                <input type="number" id="power" name="power" value="{{ power }}" step="1">
            </div>
            <div>
                <label for="mass">Mass (kg)</label>
                <input type="number" id="mass" name="mass" value="{{ mass }}" step="0.1">
            </div>
            <div>
                <label for="headwind">Headwind (km/h)</label>
                <input type="number" id="headwind" name="headwind" value="{{ headwind }}" step="0.1">
            </div>
        </div>

        <button type="submit">Analyze</button>
    </form>

    <script>
        function toggleUrlPlaceholder() {
            var mode = document.getElementById('mode').value;
            var urlInput = document.getElementById('url');
            if (mode === 'route') {
                urlInput.placeholder = 'https://ridewithgps.com/routes/...';
            } else {
                urlInput.placeholder = 'https://ridewithgps.com/collections/...';
            }
        }
    </script>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if result %}
    <div class="results">
        <h2>{{ result.name or 'Route Analysis' }}</h2>

        <div class="result-row">
            <span class="result-label">Distance</span>
            <span class="result-value">{{ "%.1f"|format(result.distance_km) }} km ({{ "%.1f"|format(result.distance_mi) }} mi)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Gain</span>
            <span class="result-value">{{ "%.0f"|format(result.elevation_m) }} m ({{ "%.0f"|format(result.elevation_ft) }} ft)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Loss</span>
            <span class="result-value">{{ "%.0f"|format(result.elevation_loss_m) }} m ({{ "%.0f"|format(result.elevation_loss_ft) }} ft)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Estimated Time</span>
            <span class="result-value">{{ result.time_str }}</span>
        </div>
        <div class="result-row">
            <span class="result-label">Avg Speed</span>
            <span class="result-value">{{ "%.1f"|format(result.avg_speed_kmh) }} km/h ({{ "%.1f"|format(result.avg_speed_mph) }} mph)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Estimated Work</span>
            <span class="result-value">{{ "%.0f"|format(result.work_kj) }} kJ</span>
        </div>
        {% if result.unpaved_pct is not none %}
        <div class="result-row">
            <span class="result-label">Surface</span>
            <span class="result-value">{{ "%.0f"|format(result.unpaved_pct) }}% unpaved</span>
        </div>
        {% endif %}

        {% if result.elevation_scaled %}
        <div class="note">
            Elevation scaled {{ "%.2f"|format(result.elevation_scale) }}x to match RideWithGPS API data.
        </div>
        {% endif %}
    </div>
    {% endif %}

    {% if collection_result %}
    <div class="results">
        <h2>{{ collection_result.name or 'Collection Analysis' }}</h2>

        <div class="result-row">
            <span class="result-label">Routes</span>
            <span class="result-value">{{ collection_result.routes | length }}</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Distance</span>
            <span class="result-value">{{ "%.0f"|format(collection_result.total_distance_km) }} km ({{ "%.0f"|format(collection_result.total_distance_mi) }} mi)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Elevation</span>
            <span class="result-value">{{ "%.0f"|format(collection_result.total_elevation_m) }} m ({{ "%.0f"|format(collection_result.total_elevation_ft) }} ft)</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Time</span>
            <span class="result-value">{{ collection_result.total_time_str }}</span>
        </div>
        <div class="result-row">
            <span class="result-label">Total Work</span>
            <span class="result-value">{{ "%.0f"|format(collection_result.total_work_kj) }} kJ</span>
        </div>

        <table class="collection-table">
            <thead>
                <tr>
                    <th>Route</th>
                    <th class="num">Dist</th>
                    <th class="num">Elev</th>
                    <th class="num">Time</th>
                    <th class="num">Unpvd</th>
                </tr>
            </thead>
            <tbody>
                {% for route in collection_result.routes %}
                <tr>
                    <td class="route-name" title="{{ route.name }}">{{ route.name }}</td>
                    <td class="num">{{ "%.0f"|format(route.distance_km) }}km</td>
                    <td class="num">{{ "%.0f"|format(route.elevation_m) }}m</td>
                    <td class="num">{{ route.time_str }}</td>
                    <td class="num">{{ "%.0f"|format(route.unpaved_pct or 0) }}%</td>
                </tr>
                {% endfor %}
                <tr class="totals-row">
                    <td>Total</td>
                    <td class="num">{{ "%.0f"|format(collection_result.total_distance_km) }}km</td>
                    <td class="num">{{ "%.0f"|format(collection_result.total_elevation_m) }}m</td>
                    <td class="num">{{ collection_result.total_time_str }}</td>
                    <td class="num"></td>
                </tr>
            </tbody>
        </table>
    </div>
    {% endif %}
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


def analyze_single_route(url: str, params: RiderParams) -> dict:
    """Analyze a single route and return results dict."""
    config = _load_config() or {}
    smoothing_radius = config.get("smoothing", DEFAULTS["smoothing"])

    points, route_metadata = get_route_with_surface(url, params.crr)

    if len(points) < 2:
        raise ValueError("Route contains fewer than 2 track points")

    # Calculate API-based elevation scale factor
    api_elevation_scale = 1.0
    api_elevation_gain = route_metadata.get("elevation_gain") if route_metadata else None
    if api_elevation_gain and api_elevation_gain > 0:
        smoothed_test = smooth_elevations(points, smoothing_radius, 1.0)
        smoothed_gain = calculate_elevation_gain(smoothed_test)
        if smoothed_gain > 0:
            api_elevation_scale = api_elevation_gain / smoothed_gain

    if smoothing_radius > 0 or api_elevation_scale != 1.0:
        points = smooth_elevations(points, smoothing_radius, api_elevation_scale)

    analysis = analyze(points, params)

    unpaved_pct = None
    surface_breakdown = calculate_surface_breakdown(points)
    if surface_breakdown:
        total_dist = surface_breakdown[0] + surface_breakdown[1]
        if total_dist > 0:
            unpaved_pct = (surface_breakdown[1] / total_dist) * 100

    return {
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
        "unpaved_pct": unpaved_pct,
        "elevation_scale": api_elevation_scale,
        "elevation_scaled": abs(api_elevation_scale - 1.0) > 0.05,
    }


def analyze_collection(url: str, params: RiderParams) -> dict:
    """Analyze all routes in a collection and return results dict."""
    route_ids, collection_name = get_collection_route_ids(url)

    if not route_ids:
        raise ValueError("No routes found in collection")

    routes = []
    for route_id in route_ids:
        route_url = f"https://ridewithgps.com/routes/{route_id}"
        try:
            route_result = analyze_single_route(route_url, params)
            route_result["time_str"] = format_duration(route_result["time_seconds"])
            routes.append(route_result)
        except Exception as e:
            # Skip failed routes but continue with others
            print(f"Error analyzing route {route_id}: {e}")
            continue

    if not routes:
        raise ValueError("Failed to analyze any routes in the collection")

    total_distance_km = sum(r["distance_km"] for r in routes)
    total_elevation_m = sum(r["elevation_m"] for r in routes)
    total_time_seconds = sum(r["time_seconds"] for r in routes)
    total_work_kj = sum(r["work_kj"] for r in routes)

    return {
        "name": collection_name,
        "routes": routes,
        "total_distance_km": total_distance_km,
        "total_distance_mi": total_distance_km * 0.621371,
        "total_elevation_m": total_elevation_m,
        "total_elevation_ft": total_elevation_m * 3.28084,
        "total_time_str": format_duration(total_time_seconds),
        "total_work_kj": total_work_kj,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = get_defaults()
    error = None
    result = None
    collection_result = None
    url = None
    mode = "route"

    power = defaults["power"]
    mass = defaults["mass"]
    headwind = defaults["headwind"]

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        mode = request.form.get("mode", "route")

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
                if not is_ridewithgps_url(url):
                    error = "Invalid RideWithGPS route URL. Expected format: https://ridewithgps.com/routes/XXXXX"
                else:
                    try:
                        params = build_params(power, mass, headwind)
                        result = analyze_single_route(url, params)
                    except Exception as e:
                        error = f"Error analyzing route: {e}"
            elif mode == "collection":
                if not is_ridewithgps_collection_url(url):
                    error = "Invalid RideWithGPS collection URL. Expected format: https://ridewithgps.com/collections/XXXXX"
                else:
                    try:
                        params = build_params(power, mass, headwind)
                        collection_result = analyze_collection(url, params)
                    except Exception as e:
                        error = f"Error analyzing collection: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        url=url,
        mode=mode,
        power=power,
        mass=mass,
        headwind=headwind,
        error=error,
        result=result,
        collection_result=collection_result,
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
