# Reality Check my Route

Analyze GPX bike routes with physics-based power estimation. Calculates distance, elevation gain/loss, speed, and estimates work and average power using gravitational, rolling resistance, and aerodynamic drag models.

Supports RideWithGPS route URLs with automatic surface type detection for mixed road/gravel routes.

## How It Works

Speed is calculated by solving the power balance equation: your power output equals the sum of resistive forces times velocity. All parameters are tunable and have been calibrated against a training set of planned routes compared with actual ride data.

### Primary Parameters (Biggest Impact)

- **Power (W)** — Your sustained power output. This is the most important input; doubling power roughly doubles your speed on flat ground.
- **Mass (kg)** — Total weight of rider + bike + gear. Dominates climbing speed since you're lifting this weight against gravity.
- **CdA (m²)** — Aerodynamic drag coefficient × frontal area. Controls air resistance, which grows with the cube of speed. Typical values: 0.25 (racing tuck) to 0.45 (upright touring).
- **Crr** — Rolling resistance coefficient. Energy lost to tire deformation and surface friction. Road tires ~0.004, gravel ~0.008-0.012.

### Environmental Factors

- **Headwind (km/h)** — Wind adds to or subtracts from your effective air speed. A 15 km/h headwind at 25 km/h means you experience drag as if riding 40 km/h.
- **Air density (kg/m³)** — Affects aerodynamic drag. Lower at altitude (1.225 at sea level, ~1.0 at 2000m).

### Climbing Model

- **Climb power factor** — Multiplier for power on steep climbs (e.g., 1.5 = 50% more power when climbing hard). Models the tendency to push harder uphill.
- **Climb threshold grade** — Grade (in degrees) where full climb power factor kicks in. Below this, power scales linearly.

### Descent Model

- **Max coasting speed** — Speed limit when coasting downhill on paved roads. Models braking for safety/comfort.
- **Max coasting speed unpaved** — Lower speed limit for gravel/dirt descents.
- **Steep descent speed** — Even slower limit for very steep descents (technical terrain).
- **Steep descent grade** — Grade threshold where steep descent speed applies.
- **Coasting grade threshold** — Grade where you stop pedaling entirely and coast.

### Data Processing

- **Smoothing radius (m)** — Gaussian smoothing applied to elevation data to reduce GPS noise.
- **Elevation scale** — Multiplier for elevation changes. Auto-calculated from RideWithGPS API data when available.
- **Surface Crr deltas** — Per-surface-type rolling resistance adjustments based on RideWithGPS surface data.

### Terrain Metrics

- **Hilliness** — Total elevation gain per unit distance (m/km or ft/mi). Measures *how much* climbing a route has, normalized by length. Typical values: flat (0-5), rolling (5-15), hilly (15-25), mountainous (25+).

- **Steepness** — Effort-weighted average grade of climbs ≥2%. Measures *how steep* the climbs are, not just total climbing. Steeper sections contribute more because they require disproportionately more power. A route with punchy 10% grades scores higher than one with gentle 4% grades, even if total climbing is similar.

- **Grade histogram** — Time spent at each grade bucket, showing the character of a route. Displayed as a bar chart in the web UI and ASCII art in the CLI.

- **Steep climbs section** — Detailed breakdown of grades ≥10%, including:
  - **Max grade** — Maximum sustained grade, calculated using a 300m rolling average to filter GPS noise and match RideWithGPS methodology.
  - **Distance at steep grades** — How much of the route is at ≥10% and ≥15% grades.
  - **Steep grade histogram** — Time and distance distribution across steep grade buckets (10-12%, 12-14%, etc.), using the same 300m rolling average for consistency with max grade.

## Installation

```bash
pip install -e ".[dev]"
```

### RideWithGPS API Credentials (Optional)

To access private routes or get surface type data, add your RideWithGPS credentials to `~/.config/gpx-analyzer/gpx-analyzer.json`:

```json
{
    "ridewithgps_api_key": "your-api-key",
    "ridewithgps_auth_token": "your-auth-token"
}
```

You can find your API credentials in your [RideWithGPS account settings](https://ridewithgps.com/api).

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

### Caching

The application uses two levels of caching for performance:

1. **GPX Download Cache** - RideWithGPS route data is cached to disk (`~/.cache/gpx-analyzer/`) with LRU eviction. This reduces API calls when re-analyzing routes.

2. **Analysis Result Cache** - Computed analysis results are cached in-memory (500 entries, ~0.75 MB). This speeds up back-and-forth route comparisons and collection re-analysis when using the same parameters.

Cache management endpoints (web interface):

```bash
# View cache statistics (size, hits, misses, hit rate)
curl https://your-server/cache-stats

# Clear the analysis cache (for testing)
curl -X POST https://your-server/cache-clear
```

The analysis cache is keyed by `(url, power, mass, headwind)` - changing any parameter triggers a fresh computation. The cache resets on server restart.

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
===============================================================================================
COLLECTION ANALYSIS: Summer Tour 2024
===============================================================================================

Model params: mass=85kg cda=0.35 crr=0.005
              power=150W max_coast=48km/h

Routes analyzed: 3
Total distance:  249 km
Total elevation: 3200 m
Total time:      14.5 hours
Total work:      4050 kJ

PER-ROUTE BREAKDOWN:
-----------------------------------------------------------------------------------------------
Route                          Time   Work   Distk   Elev Hilly Steep  Speed Unpvd  EScl
-----------------------------------------------------------------------------------------------
Day 1: Coast to Mountains      4.5h  1250k     85k   950m    11  5.2%  18.7    5%  1.02
Day 2: Mountain Pass           6.3h  1820k     72k  1800m    25  8.5%  11.5    0%  0.98
Day 3: Valley Route            3.8h   980k     92k   450m     5  4.1%  24.5   12%  1.05
-----------------------------------------------------------------------------------------------
TOTAL                         14.5h  4050k    249k  3200m
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
Config: mass=75kg cda=0.30 crr=0.005 power=120W ...
========================================
  Est. Time @120W: 5h 18m 22s
  Est. Work:       1901.4 kJ
========================================
Distance:       72.23 km
Elevation Gain: 1802 m
Elevation Loss: 1799 m
Avg Speed:      13.6 km/h
Max Speed:      52.0 km/h
Est. Avg Power: 99 W
Surface:        58.0 km paved, 14.2 km unpaved (20%)
Hilliness:      21 m/km
Steepness:      8.3%

Time at Grade:
   <-10%: █████                   5%
    -10%: ███                     3%
     -8%: ████                    4%
     -6%: ████                    4%
     -4%: █████                   5%
     -2%: ██████████              9%
      0%: ████████████████████   18%
      2%: ████████████           11%
      4%: ████████████           11%
      6%: █████████               9%
      8%: ███████                 6%
    >10%: █████████████████      15%
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

The physics model parameters (CdA, Crr, coasting speeds, climb power factors, etc.) have been calibrated against a training set of planned routes compared with actual ride data. You can run the same analysis on your own routes to tune parameters for your riding style:


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
