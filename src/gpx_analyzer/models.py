from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TrackPoint:
    lat: float
    lon: float
    elevation: float | None  # meters
    time: datetime | None


@dataclass
class RiderParams:
    total_mass: float = 85.0  # kg (rider + bike)
    cda: float = 0.35  # m² (drag coefficient * frontal area)
    crr: float = 0.005  # rolling resistance coefficient
    air_density: float = 1.225  # kg/m³


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
