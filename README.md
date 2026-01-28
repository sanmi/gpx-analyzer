# GPX Bike Route Analyzer

Analyze GPX bike routes with physics-based power estimation. Calculates distance, elevation gain/loss, speed, and estimates work and average power using gravitational, rolling resistance, and aerodynamic drag models.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
gpx-analyzer <path-to-gpx-file>
```

Or run as a module:

```bash
python -m gpx_analyzer <path-to-gpx-file>
```

### Options

| Flag     | Description                          | Default |
|----------|--------------------------------------|---------|
| `--mass` | Total mass of rider + bike (kg)      | 85      |
| `--cda`  | Drag coefficient × frontal area (m²) | 0.35    |
| `--crr`  | Rolling resistance coefficient       | 0.005   |

### Example

```bash
gpx-analyzer ride.gpx --mass 75 --cda 0.30
```

Output:

```
=== GPX Route Analysis ===
Distance:       2.21 km
Elevation Gain: 62 m
Elevation Loss: 27 m
Duration:       0h 06m 55s
Moving Time:    0h 06m 55s
Avg Speed:      19.2 km/h
Max Speed:      22.5 km/h
Est. Work:      63.3 kJ
Est. Avg Power: 152 W
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
