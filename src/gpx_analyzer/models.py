from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TrackPoint:
    lat: float
    lon: float
    elevation: float | None  # meters
    time: datetime | None
    crr: float | None = None  # per-segment rolling resistance coefficient
    unpaved: bool = False  # whether segment is on unpaved surface


@dataclass
class RiderParams:
    total_mass: float = 85.0  # kg (rider + bike)
    cda: float = 0.35  # m² (drag coefficient * frontal area)
    crr: float = 0.005  # rolling resistance coefficient
    air_density: float = 1.225  # kg/m³
    assumed_avg_power: float = 150.0  # watts (base power on flat terrain)
    coasting_grade_threshold: float = -5.0  # degrees; fully coasting at this grade
    max_coasting_speed: float = 48.0 / 3.6  # m/s (default 48 km/h)
    max_coasting_speed_unpaved: float = 32.0 / 3.6  # m/s (default 32 km/h for gravel)
    headwind: float = 0.0  # m/s (positive = into the wind, negative = tailwind)
    # Gradient-dependent power model
    climb_power_factor: float = 1.5  # power multiplier on steep climbs (1.5 = 50% more power)
    climb_threshold_grade: float = 4.0  # degrees; full climb factor reached at this grade


@dataclass
class RideAnalysis:
    total_distance: float  # meters
    elevation_gain: float  # meters
    elevation_loss: float  # meters
    duration: timedelta
    moving_time: timedelta
    avg_speed: float  # m/s (based on moving time)
    max_speed: float  # m/s
    estimated_work: float  # joules
    estimated_avg_power: float  # watts
    estimated_moving_time_at_power: timedelta  # time at assumed power
