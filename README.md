# GPX Bike Route Analyzer

Analyze GPX bike routes with physics-based power estimation. Calculates distance, elevation gain/loss, speed, and estimates work and average power using gravitational, rolling resistance, and aerodynamic drag models.

Supports RideWithGPS route URLs with automatic surface type detection for mixed road/gravel routes.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
gpx-analyzer <path-to-gpx-file-or-url>
```

Or run as a module:

```bash
python -m gpx_analyzer <path-to-gpx-file-or-url>
```

### RideWithGPS Integration

Analyze routes directly from RideWithGPS URLs:

```bash
gpx-analyzer https://ridewithgps.com/routes/48889111
```

This fetches route data including surface type information (road vs gravel) for more accurate rolling resistance estimation.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mass` | Total mass of rider + bike (kg) | 85 |
| `--cda` | Drag coefficient × frontal area (m²) | 0.35 |
| `--crr` | Rolling resistance coefficient | 0.005 |
| `--power` | Assumed average power output (W) | 150 |
| `--coasting-grade` | Grade in degrees at which rider fully coasts | -5.0 |
| `--max-coast-speed` | Maximum coasting speed on paved (km/h) | 48 |
| `--max-coast-speed-unpaved` | Maximum coasting speed on gravel (km/h) | 24 |
| `--smoothing` | Elevation smoothing radius (m) | 50 |
| `--elevation-scale` | Scale factor for elevation changes | 1.0 |
| `--headwind` | Headwind speed (km/h, negative = tailwind) | 0 |
| `--compare-trip` | RideWithGPS trip URL to compare against | - |
| `--training` | Training data JSON file for batch analysis | - |

### Example

```bash
gpx-analyzer ride.gpx --mass 75 --cda 0.30 --power 120
```

Output:

```
=== GPX Route Analysis ===
Distance:       72.23 km (44.88 mi)
Elevation Gain: 1802 m (5913 ft)
Elevation Loss: 1799 m (5901 ft)
Moving Time:    6h 23m 44s
Est. Work:      2281.4 kJ
Est. Avg Power: 99 W
Est. Time @100W: 6h 20m 14s
Surface:        58.0 km paved, 14.2 km unpaved (20%)
```

## Comparing Predictions with Actual Rides

Compare route predictions against actual ride data from RideWithGPS trips:

```bash
gpx-analyzer https://ridewithgps.com/routes/48889111 \
  --compare-trip https://ridewithgps.com/trips/233763291
```

This shows predicted vs actual speed by gradient, time/work prediction errors, and actual power data if available from a power meter.

## Training Data for Parameter Tuning

Run batch analysis on multiple route/trip pairs to tune model parameters:

```bash
gpx-analyzer --training training-data.json
```

Training data JSON format:

```json
{
  "routes": [
    {
      "name": "Santa Cruz Mixed",
      "route_url": "https://ridewithgps.com/routes/48889111",
      "trip_url": "https://ridewithgps.com/trips/233763291",
      "tags": ["road", "gravel", "hilly"],
      "notes": "Mixed surface route with significant climbing"
    }
  ]
}
```

Output shows aggregate statistics across all routes, broken down by terrain type.

## Configuration File

Create `gpx-analyzer.json` in the project directory or `~/.config/gpx-analyzer/gpx-analyzer.json` for persistent settings:

```json
{
  "ridewithgps_api_key": "your-api-key",
  "ridewithgps_auth_token": "your-auth-token",
  "mass": 84.0,
  "power": 100.0,
  "cda": 0.32,
  "crr": 0.012,
  "max_coast_speed": 52.0,
  "max_coast_speed_unpaved": 24.0
}
```

## Running Tests

Run all tests:

```bash
pytest
```

Run unit tests only:

```bash
pytest tests/unit
```

Run functional (end-to-end CLI) tests only:

```bash
pytest tests/functional
```

Run with verbose output:

```bash
pytest -v
```
