# GPX Bike Route Analyzer

Analyze GPX bike routes with physics-based power estimation. Calculates distance, elevation gain/loss, speed, and estimates work and average power using gravitational, rolling resistance, and aerodynamic drag models.

Supports RideWithGPS route URLs with automatic surface type detection for mixed road/gravel routes.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
gpx-analyzer <path-to-gpx-file-or-url>
```

Or run as a module:

```bash
python -m gpx_analyzer <path-to-gpx-file-or-url>
```

### Web Interface

Start the web server for mobile-friendly access:

```bash
gpx-analyzer-web
```

Then open http://localhost:5050 in your browser. The web interface supports:
- Single route analysis
- Collection analysis with real-time progress
- Imperial/metric unit toggle
- Customizable power, mass, and headwind parameters

### RideWithGPS Integration

Analyze routes directly from RideWithGPS URLs:

```bash
gpx-analyzer https://ridewithgps.com/routes/48889111
```

This fetches route data including surface type information (road vs gravel) for more accurate rolling resistance estimation.

### Analyzing Collections

Analyze all routes in a RideWithGPS collection:

```bash
gpx-analyzer --collection https://ridewithgps.com/collections/12345
```

Output shows per-route breakdown with totals:

```
============================================================
Collection: Summer Tour 2024
============================================================
Route                         Time    Work   Dist  Elev  Speed Unpvd  EScl
--------------------------------------------------------------------------------
Day 1: Coast to Mountains   4h 32m  1250kJ   85km  950m  18.7    5%  1.02
Day 2: Mountain Pass        6h 15m  1820kJ   72km 1800m  11.5    0%  0.98
Day 3: Valley Route         3h 45m   980kJ   92km  450m  24.5   12%  1.05
--------------------------------------------------------------------------------
Total                      14h 32m  4050kJ  249km 3200m
```

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
| `--climb-power-factor` | Power multiplier on steep climbs (1.5 = 50% more) | 1.5 |
| `--climb-threshold-grade` | Grade (degrees) at which full climb power is reached | 4.0 |
| `--steep-descent-speed` | Max speed on steep descents (km/h) | 18 |
| `--steep-descent-grade` | Grade (degrees) where steep descent speed applies | -8.0 |
| `--smoothing` | Elevation smoothing radius (m) | 50 |
| `--elevation-scale` | Scale factor for elevation changes | 1.0 |
| `--headwind` | Headwind speed (km/h, negative = tailwind) | 0 |
| `--compare-trip` | RideWithGPS trip URL to compare against | - |
| `--training` | Training data JSON file for batch analysis | - |
| `--collection` | RideWithGPS collection URL to analyze all routes | - |
| `--imperial` | Display output in imperial units (mi, ft, mph) | false |

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

Output shows predicted vs actual metrics:

```
=== Route vs Trip Comparison ===

Route distance: 72.2 km
Trip distance:  73.2 km

Predicted time @100W: 6.22 hours
Actual moving time:        5.68 hours
Difference: +33 minutes (+9.5%)

Predicted work: 2240 kJ
Actual work:    1889 kJ (+19%)

Actual avg power: 101W (model assumes 100W)

Speed by gradient (actual vs predicted):
 Grade |   Actual |     Pred |    Error |    Pwr
--------------------------------------------------
  -12% |    16.5 |    52.0 |    +214% |     1W
   -6% |    20.6 |    50.7 |    +146% |    13W
   +0% |    13.5 |    21.4 |     +59% |    98W
   +6% |    10.9 |     8.5 |     -22% |   145W
  +12% |     6.5 |     4.2 |     -35% |   155W
```

The gradient breakdown shows how well the model predicts speed at different grades, with actual power data from the ride.

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
      "avg_watts": 101,
      "tags": ["road", "gravel", "hilly"],
      "notes": "Mixed surface route with significant climbing"
    }
  ]
}
```

The `avg_watts` field specifies the average power for that specific ride (from power meter data). This allows comparing rides at different intensities. If omitted, the default `--power` value is used.

Output shows aggregate statistics and per-route breakdown:

```
============================================================
TRAINING DATA ANALYSIS SUMMARY
============================================================

Model params: mass=84.0kg cda=0.32 crr=0.012
              power=100.0W max_coast=52km/h

Routes analyzed: 5
Total distance:  334 km
Total elevation: 7001 m

PREDICTION ERRORS (positive = predicted too slow/high)
--------------------------------------------------
  Avg time error:  +4.3%
  Avg work error:  +12.8%

BY TERRAIN TYPE:
  Road (5 routes):   +4.3% time error
  Gravel (4 routes): +8.6% time error
  Hilly (5 routes):  +4.3% time error

PER-ROUTE BREAKDOWN:
----------------------------------------------------------------------
Route                    Pwr   Dist   Elev  Elev%   Time%   Work%
----------------------------------------------------------------------
Loma Prieta Rd, Eurek   101W    72k  1802m   +12%   +8.6%    +19%
Col de La Croix de Fe   118W    63k  1801m    -0%  -13.1%     +2%
Lexington, OSC, Loma    101W    86k  1794m   +24%  +10.5%    +20%
```

The `Elev%` column shows the difference between route and actual trip elevation gain, helping identify routes with inaccurate elevation data.

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
  "max_coast_speed_unpaved": 24.0,
  "climb_power_factor": 1.5,
  "climb_threshold_grade": 4.0,
  "steep_descent_speed": 20.0,
  "steep_descent_grade": -4.0
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
