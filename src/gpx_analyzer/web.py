"""Simple web interface for GPX analyzer."""

import json

from flask import Flask, render_template_string, request, Response

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
    <title>Route Estimator</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        @media (min-width: 1200px) {
            body { max-width: 1100px; }
        }
        @media (max-width: 480px) {
            body { padding: 12px; }
        }
        h1 { color: #333; font-size: 1.5em; }
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
        button:disabled { background: #999; cursor: not-allowed; }
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
            background: linear-gradient(90deg, #007aff, #00c6ff);
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
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        @media (min-width: 1200px) {
            .route-name { max-width: 400px; }
            .collection-table { font-size: 0.95em; }
        }
        @media (max-width: 768px) {
            .route-name { max-width: 180px; }
        }
        @media (max-width: 600px) {
            .collection-table { font-size: 0.8em; }
            .collection-table th, .collection-table td { padding: 8px 4px; }
            .route-name { max-width: 120px; }
            .param-row { flex-direction: column; gap: 0; }
        }
        .hidden { display: none; }
        /* Header styles */
        .header-section {
            margin-bottom: 20px;
        }
        .header-section h1 {
            margin-bottom: 8px;
        }
        .tagline {
            color: #666;
            font-size: 0.95em;
            margin: 0;
            line-height: 1.4;
        }
        /* Info button styles */
        .label-row {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .info-btn {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1.5px solid #007aff;
            background: white;
            color: #007aff;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            line-height: 1;
            flex-shrink: 0;
        }
        .info-btn:hover {
            background: #007aff;
            color: white;
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
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }
        .toggle-label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-weight: normal;
            color: #555;
            margin: 0;
        }
        .toggle-label input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header-section">
        <h1>Route Estimator</h1>
        <p class="tagline">Uses a physics model to estimate cycling time and energy expenditure based on elevation, surface type, and rider parameters.</p>
    </div>

    <form method="POST" id="analyzeForm">
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
                <div class="label-row">
                    <label for="power">Power (W)</label>
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
        <h2 id="collectionName">Collection Analysis</h2>
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
                    <th class="num primary">Time</th>
                    <th class="num primary separator">Work</th>
                    <th class="num">Dist</th>
                    <th class="num">Elev</th>
                    <th class="num">Speed</th>
                    <th class="num">Unpvd</th>
                    <th class="num">EScl</th>
                </tr>
            </thead>
            <tbody id="routesTableBody">
            </tbody>
        </table>
    </div>

    <!-- Info Modals -->
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

    <script>
        function showModal(id) {
            document.getElementById(id).classList.add('active');
        }

        function hideModal(id) {
            document.getElementById(id).classList.remove('active');
        }

        function toggleUrlPlaceholder() {
            var mode = document.getElementById('mode').value;
            var urlInput = document.getElementById('url');
            if (mode === 'route') {
                urlInput.placeholder = 'https://ridewithgps.com/routes/...';
            } else {
                urlInput.placeholder = 'https://ridewithgps.com/collections/...';
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
                '<td class="num"></td><td class="num"></td><td class="num"></td>';
            tbody.appendChild(totalsRow);
        }

        document.getElementById('analyzeForm').addEventListener('submit', function(e) {
            var mode = document.getElementById('mode').value;
            if (mode !== 'collection') {
                return; // Let form submit normally for single route
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
                    document.getElementById('collectionName').textContent = data.name || 'Collection Analysis';
                    document.getElementById('progressText').textContent =
                        'Analyzing route 0 of ' + data.total + '...';
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
                    row.innerHTML = '<td class="route-name" title="' + r.name + '">' + r.name + '</td>' +
                        '<td class="num primary">' + r.time_str + '</td>' +
                        '<td class="num primary separator">' + Math.round(r.work_kj) + 'kJ</td>' +
                        '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                        '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
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
                row.innerHTML = '<td class="route-name" title="' + r.name + '">' + r.name + '</td>' +
                    '<td class="num primary">' + r.time_str + '</td>' +
                    '<td class="num primary separator">' + Math.round(r.work_kj) + 'kJ</td>' +
                    '<td class="num">' + formatDist(r.distance_km) + '</td>' +
                    '<td class="num">' + formatElev(r.elevation_m) + '</td>' +
                    '<td class="num">' + formatSpeed(r.avg_speed_kmh) + '</td>' +
                    '<td class="num">' + Math.round(r.unpaved_pct || 0) + '%</td>' +
                    '<td class="num">' + r.elevation_scale.toFixed(2) + '</td>';
                tbody.appendChild(row);
            });

            updateTotals(collectionRoutes);
        }

        document.getElementById('imperial').addEventListener('change', function() {
            rerenderCollectionTable();
        });
    </script>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if result %}
    <div class="results">
        <h2>{{ result.name or 'Route Analysis' }}</h2>

        <div class="result-row">
            <span class="result-label">Distance</span>
            {% if imperial %}
            <span class="result-value">{{ "%.1f"|format(result.distance_mi) }} mi</span>
            {% else %}
            <span class="result-value">{{ "%.1f"|format(result.distance_km) }} km</span>
            {% endif %}
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Gain</span>
            {% if imperial %}
            <span class="result-value">{{ "%.0f"|format(result.elevation_ft) }} ft</span>
            {% else %}
            <span class="result-value">{{ "%.0f"|format(result.elevation_m) }} m</span>
            {% endif %}
        </div>
        <div class="result-row">
            <span class="result-label">Elevation Loss</span>
            {% if imperial %}
            <span class="result-value">{{ "%.0f"|format(result.elevation_loss_ft) }} ft</span>
            {% else %}
            <span class="result-value">{{ "%.0f"|format(result.elevation_loss_m) }} m</span>
            {% endif %}
        </div>
        <div class="result-row">
            <span class="result-label">Estimated Time</span>
            <span class="result-value">{{ result.time_str }}</span>
        </div>
        <div class="result-row">
            <span class="result-label">Avg Speed</span>
            {% if imperial %}
            <span class="result-value">{{ "%.1f"|format(result.avg_speed_mph) }} mph</span>
            {% else %}
            <span class="result-value">{{ "%.1f"|format(result.avg_speed_kmh) }} km/h</span>
            {% endif %}
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

                    yield f"data: {json.dumps({'type': 'route', 'route': route_result, 'total': len(route_ids)})}\n\n"
                except Exception as e:
                    # Skip failed routes but continue
                    print(f"Error analyzing route {route_id}: {e}")
                    continue

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = get_defaults()
    error = None
    result = None
    url = None
    mode = "route"
    imperial = False

    power = defaults["power"]
    mass = defaults["mass"]
    headwind = defaults["headwind"]

    if request.method == "POST":
        url = request.form.get("url", "").strip()
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
                if not is_ridewithgps_url(url):
                    error = "Invalid RideWithGPS route URL. Expected format: https://ridewithgps.com/routes/XXXXX"
                else:
                    try:
                        params = build_params(power, mass, headwind)
                        result = analyze_single_route(url, params)
                    except Exception as e:
                        error = f"Error analyzing route: {e}"
            # Collection mode is handled by JavaScript + SSE

    return render_template_string(
        HTML_TEMPLATE,
        url=url,
        mode=mode,
        power=power,
        mass=mass,
        headwind=headwind,
        imperial=imperial,
        error=error,
        result=result,
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
