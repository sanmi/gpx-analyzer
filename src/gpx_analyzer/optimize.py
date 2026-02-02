"""Parameter optimization for GPX analyzer using differential evolution.

Minimizes prediction error for estimated work, estimated time, and max grade
across a set of training routes.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from gpx_analyzer.analyzer import analyze, calculate_hilliness, DEFAULT_MAX_GRADE_WINDOW, DEFAULT_MAX_GRADE_SMOOTHING
from gpx_analyzer.models import RiderParams, TrackPoint
from gpx_analyzer.ridewithgps import get_route_with_surface, get_trip_data, is_ridewithgps_url, is_ridewithgps_trip_url, TripPoint
from gpx_analyzer.smoothing import smooth_elevations
from gpx_analyzer.compare import _calculate_actual_work, _calculate_moving_time
from gpx_analyzer.training import load_training_data, TrainingRoute, _calculate_trip_max_grade


@dataclass
class PreloadedRoute:
    """Pre-loaded route data for fast optimization."""
    name: str
    route_points: list[TrackPoint]
    trip_points: list[TripPoint]
    trip_metadata: dict
    power: float
    headwind: float
    actual_work: float | None
    actual_time: float
    actual_max_grade: float
    api_elevation_gain: float | None  # DEM-derived elevation from RWGPS API
    elevation_scale: float  # Precomputed scale factor to match API elevation


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    params: dict[str, float]
    final_error: float
    iterations: int
    route_errors: list[dict]


# Parameters to optimize with their bounds (min, max)
# Note: elevation_scale is not optimized - we auto-scale using RWGPS API elevation data
PARAM_BOUNDS = {
    "crr": (0.004, 0.015),           # Rolling resistance coefficient
    "cda": (0.25, 0.40),             # Aerodynamic drag area (m²)
    "coasting_grade": (-6.0, -2.0),  # Grade threshold for coasting (%)
    "max_coast_speed": (45.0, 60.0), # Max coasting speed on paved (km/h)
    "climb_power_factor": (1.2, 1.8), # Power multiplier on climbs
    "climb_threshold_grade": (3.0, 5.0), # Grade to start applying climb power (%)
    "smoothing": (30.0, 80.0),       # Elevation smoothing radius (m)
}

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    "air_density": 1.2,
    "max_coasting_speed_unpaved": 24.0,
    "steep_descent_speed": 20.0,
    "steep_descent_grade": -8.0,
}


def _calculate_elevation_gain(points: list) -> float:
    """Calculate total elevation gain from a list of points."""
    if len(points) < 2:
        return 0.0
    gain = 0.0
    for i in range(1, len(points)):
        elev_a = points[i - 1].elevation or 0.0
        elev_b = points[i].elevation or 0.0
        delta = elev_b - elev_a
        if delta > 0:
            gain += delta
    return gain


def _preload_routes(routes: list[TrainingRoute], default_power: float, default_crr: float = 0.005) -> list[PreloadedRoute]:
    """Pre-load all route and trip data once."""
    preloaded = []

    for route in routes:
        if not is_ridewithgps_url(route.route_url) or not is_ridewithgps_trip_url(route.trip_url):
            print(f"  Skipping {route.name}: invalid URL")
            continue

        try:
            # Load route data
            route_points, route_metadata = get_route_with_surface(route.route_url, default_crr)
            if len(route_points) < 2:
                print(f"  Skipping {route.name}: insufficient route points")
                continue

            # Load trip data
            trip_points, trip_metadata = get_trip_data(route.trip_url)
            if len(trip_points) < 2:
                print(f"  Skipping {route.name}: insufficient trip points")
                continue

            # Get actual values from trip
            actual_work, _, _ = _calculate_actual_work(trip_points)
            actual_time = _calculate_moving_time(trip_points)
            actual_max_grade = _calculate_trip_max_grade(trip_points, 50.0)

            if actual_time <= 0:
                print(f"  Skipping {route.name}: no moving time")
                continue

            power = route.avg_watts if route.avg_watts else default_power
            headwind = route.headwind if route.headwind else 0.0

            # Get API elevation gain (DEM-derived) for auto-scaling
            api_elevation_gain = route_metadata.get("elevation_gain")

            # Precompute elevation scale factor using raw elevation gain
            # This avoids expensive double-smoothing during optimization
            raw_elevation_gain = _calculate_elevation_gain(route_points)
            if api_elevation_gain and api_elevation_gain > 0 and raw_elevation_gain > 0:
                elevation_scale = api_elevation_gain / raw_elevation_gain
            else:
                elevation_scale = 1.0

            preloaded.append(PreloadedRoute(
                name=route.name,
                route_points=route_points,
                trip_points=trip_points,
                trip_metadata=trip_metadata,
                power=power,
                headwind=headwind,
                actual_work=actual_work / 1000 if actual_work else None,  # Convert to kJ
                actual_time=actual_time,
                actual_max_grade=actual_max_grade,
                api_elevation_gain=api_elevation_gain,
                elevation_scale=elevation_scale,
            ))
            print(f"  Loaded: {route.name}")

        except Exception as e:
            print(f"  Error loading {route.name}: {e}")
            continue

    return preloaded


def _build_rider_params(opt_values: np.ndarray, param_names: list[str],
                        power: float, mass: float, headwind: float = 0.0) -> RiderParams:
    """Build RiderParams from optimization values."""
    params = {name: val for name, val in zip(param_names, opt_values)}

    max_coast_speed_ms = params.get("max_coast_speed", 52.0) / 3.6
    coasting_grade_frac = params.get("coasting_grade", -4.0) / 100
    climb_threshold_frac = params.get("climb_threshold_grade", 4.0) / 100
    steep_descent_frac = FIXED_PARAMS["steep_descent_grade"] / 100

    return RiderParams(
        total_mass=mass,
        cda=params.get("cda", 0.3),
        crr=params.get("crr", 0.005),
        air_density=FIXED_PARAMS["air_density"],
        assumed_avg_power=power,
        coasting_grade_threshold=coasting_grade_frac,
        max_coasting_speed=max_coast_speed_ms,
        max_coasting_speed_unpaved=FIXED_PARAMS["max_coasting_speed_unpaved"] / 3.6,
        headwind=headwind / 3.6,
        climb_power_factor=params.get("climb_power_factor", 1.5),
        climb_threshold_grade=climb_threshold_frac,
        steep_descent_speed=FIXED_PARAMS["steep_descent_speed"] / 3.6,
        steep_descent_grade=steep_descent_frac,
    )


def _analyze_preloaded_route(route: PreloadedRoute, opt_values: np.ndarray,
                              param_names: list[str], default_mass: float) -> dict | None:
    """Analyze a pre-loaded route with given parameters."""
    try:
        params_dict = {name: val for name, val in zip(param_names, opt_values)}
        smoothing = params_dict.get("smoothing", 50.0)

        rider_params = _build_rider_params(opt_values, param_names, route.power, default_mass, route.headwind)

        # Apply smoothing with precomputed elevation scale (from API data)
        smoothed = smooth_elevations(route.route_points, smoothing, route.elevation_scale)

        # Run analysis
        analysis = analyze(smoothed, rider_params)

        pred_time = analysis.estimated_moving_time_at_power.total_seconds()
        pred_work = analysis.estimated_work / 1000  # Convert to kJ

        # Calculate max grade
        hilliness = calculate_hilliness(smoothed, rider_params, route.route_points,
                                        DEFAULT_MAX_GRADE_WINDOW, DEFAULT_MAX_GRADE_SMOOTHING)
        pred_max_grade = hilliness.max_grade

        return {
            "name": route.name,
            "pred_time": pred_time,
            "actual_time": route.actual_time,
            "pred_work": pred_work,
            "actual_work": route.actual_work,
            "pred_max_grade": pred_max_grade,
            "actual_max_grade": route.actual_max_grade,
        }

    except Exception as e:
        return None


def _objective_function(opt_values: np.ndarray, param_names: list[str],
                        routes: list[PreloadedRoute], default_mass: float,
                        weights: tuple[float, float, float]) -> float:
    """Compute weighted error across all pre-loaded routes."""
    w_work, w_time, w_grade = weights
    total_error = 0.0
    valid_routes = 0

    for route in routes:
        result = _analyze_preloaded_route(route, opt_values, param_names, default_mass)
        if result is None:
            continue

        time_err = abs(result["pred_time"] - result["actual_time"]) / result["actual_time"]

        work_err = 0.0
        if result["actual_work"] and result["actual_work"] > 0:
            work_err = abs(result["pred_work"] - result["actual_work"]) / result["actual_work"]

        grade_err = 0.0
        if result["actual_max_grade"] and result["actual_max_grade"] > 0:
            grade_err = abs(result["pred_max_grade"] - result["actual_max_grade"]) / result["actual_max_grade"]

        route_error = w_work * work_err + w_time * time_err + w_grade * grade_err
        total_error += route_error
        valid_routes += 1

    if valid_routes == 0:
        return float('inf')

    return total_error / valid_routes


def optimize_parameters(
    training_file: Path | str,
    default_power: float = 100.0,
    default_mass: float = 84.0,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_iterations: int = 50,
    population_size: int = 10,
    param_subset: list[str] | None = None,
    verbose: bool = True,
) -> OptimizationResult:
    """Run parameter optimization on training data.

    Args:
        training_file: Path to training data JSON
        default_power: Default rider power (W) when not specified per-route
        default_mass: Rider + bike mass (kg)
        weights: (work_weight, time_weight, grade_weight) for objective function
        max_iterations: Maximum optimization iterations
        population_size: Population size for differential evolution (smaller = faster)
        param_subset: List of parameter names to optimize (None = all)
        verbose: Print progress

    Returns:
        OptimizationResult with optimized parameters and final error
    """
    # Load training data
    routes = load_training_data(Path(training_file))
    if not routes:
        raise ValueError("No valid training routes found")

    if verbose:
        print(f"Loading {len(routes)} training routes...")

    # Pre-load all route/trip data
    preloaded = _preload_routes(routes, default_power)
    if not preloaded:
        raise ValueError("No routes could be loaded")

    if verbose:
        print(f"\nSuccessfully loaded {len(preloaded)} routes")

    # Determine which parameters to optimize
    if param_subset:
        param_names = [p for p in param_subset if p in PARAM_BOUNDS]
    else:
        param_names = list(PARAM_BOUNDS.keys())

    bounds = [PARAM_BOUNDS[name] for name in param_names]

    if verbose:
        print(f"Optimizing {len(param_names)} parameters: {param_names}")
        print(f"Weights: work={weights[0]}, time={weights[1]}, grade={weights[2]}")
        print(f"Population size: {population_size}, Max iterations: {max_iterations}")

    # Track iteration progress with timing
    iteration_count = [0]
    best_error = [float('inf')]
    best_params = [None]
    start_time = [None]
    last_time = [None]
    eval_count = [0]

    def format_time(seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def callback(xk, convergence=None):
        iteration_count[0] += 1
        now = time.time()

        if start_time[0] is None:
            start_time[0] = now
            last_time[0] = now

        iter_time = now - last_time[0]
        elapsed = now - start_time[0]
        last_time[0] = now

        error = _objective_function(xk, param_names, preloaded, default_mass, weights)
        improved = error < best_error[0]
        if improved:
            best_error[0] = error
            best_params[0] = {name: val for name, val in zip(param_names, xk)}

        if verbose:
            # Estimate remaining time based on average iteration time
            avg_iter_time = elapsed / iteration_count[0]
            remaining_iters = max_iterations - iteration_count[0]
            eta = avg_iter_time * remaining_iters

            # Build status line
            marker = "*" if improved else " "
            status = (f"  Gen {iteration_count[0]:3d}/{max_iterations} {marker} "
                     f"error={best_error[0]:.4f}  "
                     f"elapsed={format_time(elapsed)}  "
                     f"ETA={format_time(eta)}")
            print(status)

            # Show best params every 10 iterations or on improvement
            if improved and iteration_count[0] > 1:
                params_str = "  Best: " + ", ".join(f"{k}={v:.3f}" for k, v in best_params[0].items())
                print(params_str)

    # Run optimization
    if verbose:
        print(f"\nStarting optimization (max {max_iterations} generations)...")
        print(f"Each generation evaluates ~{population_size * len(param_names)} parameter sets")
        print()

    result: OptimizeResult = differential_evolution(
        func=lambda x: _objective_function(x, param_names, preloaded, default_mass, weights),
        bounds=bounds,
        maxiter=max_iterations,
        popsize=population_size,
        callback=callback,
        seed=42,
        polish=True,
        workers=1,
        disp=False,
        tol=0.01,
        atol=0.001,
    )

    # Extract optimized parameters
    opt_params = {name: val for name, val in zip(param_names, result.x)}

    if verbose:
        total_time = time.time() - start_time[0] if start_time[0] else 0
        print(f"\nOptimization complete!")
        print(f"Total time: {format_time(total_time)}")
        print(f"Final error: {result.fun:.4f}")
        print(f"Generations: {iteration_count[0]}")
        print("\nOptimized parameters:")
        for name, val in opt_params.items():
            print(f"  {name}: {val:.4f}")

    # Calculate per-route errors with final parameters
    route_errors = []
    if verbose:
        print("\nPer-route results:")
    for route in preloaded:
        res = _analyze_preloaded_route(route, result.x, param_names, default_mass)
        if res:
            time_err = (res["pred_time"] - res["actual_time"]) / res["actual_time"]
            work_err = 0.0
            if res["actual_work"] and res["actual_work"] > 0:
                work_err = (res["pred_work"] - res["actual_work"]) / res["actual_work"]
            grade_err = 0.0
            if res["actual_max_grade"] and res["actual_max_grade"] > 0:
                grade_err = (res["pred_max_grade"] - res["actual_max_grade"]) / res["actual_max_grade"]

            route_errors.append({
                "name": res["name"],
                "time_error_pct": time_err * 100,
                "work_error_pct": work_err * 100,
                "grade_error_pct": grade_err * 100,
            })

            if verbose:
                print(f"  {res['name']}: time={time_err:+.1%}, work={work_err:+.1%}, grade={grade_err:+.1%}")

    return OptimizationResult(
        params=opt_params,
        final_error=result.fun,
        iterations=iteration_count[0],
        route_errors=route_errors,
    )


def save_optimized_config(result: OptimizationResult, output_path: Path | str) -> None:
    """Save optimized parameters to a config JSON file."""
    config = {
        "mass": 84.0,
        "power": 100.0,
    }

    param_mapping = {
        "crr": "crr",
        "cda": "cda",
        "coasting_grade": "coasting_grade",
        "max_coast_speed": "max_coast_speed",
        "climb_power_factor": "climb_power_factor",
        "climb_threshold_grade": "climb_threshold_grade",
        "smoothing": "smoothing",
        "elevation_scale": "elevation_scale",
    }

    for opt_name, config_name in param_mapping.items():
        if opt_name in result.params:
            config[config_name] = round(result.params[opt_name], 4)

    config["max_coast_speed_unpaved"] = FIXED_PARAMS["max_coasting_speed_unpaved"]
    config["steep_descent_speed"] = FIXED_PARAMS["steep_descent_speed"]
    config["steep_descent_grade"] = FIXED_PARAMS["steep_descent_grade"]

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
