"""Caching utilities for gpx-analyzer.

Provides memory and disk-based caching with LRU eviction.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from gpx_analyzer.ridewithgps import _load_config


def _get_config_hash() -> str:
    """Get a hash of config parameters that affect analysis but aren't in the UI."""
    config = _load_config() or {}
    # Include all config params that affect physics but aren't UI-adjustable
    # Note: smoothing is now UI-adjustable, so it's not in this list
    relevant_keys = [
        "crr", "cda", "coasting_grade", "max_coast_speed", "max_coast_speed_unpaved",
        "climb_threshold_grade", "steep_descent_speed", "steep_descent_grade",
        "straight_descent_speed", "hairpin_speed", "straight_curvature", "hairpin_curvature",
        "drivetrain_efficiency", "gravel_grade", "elevation_scale",
        "trip_smoothing_enabled",
    ]
    config_str = "|".join(f"{k}={config.get(k, '')}" for k in relevant_keys)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class CacheStats:
    """Statistics for a cache instance."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    memory_kb: float = 0.0

    @property
    def hit_rate(self) -> str:
        """Return hit rate as percentage string."""
        total = self.hits + self.misses
        if total == 0:
            return "0.0%"
        return f"{(self.hits / total * 100):.1f}%"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hit_rate": self.hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "max_size": self.max_size,
            "memory_kb": self.memory_kb,
        }


class AnalysisCache:
    """Thread-safe LRU cache for route analysis results."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
                  descent_braking_factor: float = 1.0, descending_power: float = 20.0,
                  gravel_power_factor: float = 0.90, smoothing: float = 50.0,
                  smoothing_override: bool = False) -> str:
        """Create a cache key from analysis parameters."""
        config_hash = _get_config_hash()
        key_str = f"{url}|{climbing_power}|{flat_power}|{descending_power}|{mass}|{headwind}|{descent_braking_factor}|{gravel_power_factor}|{smoothing}|{smoothing_override}|{config_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
            descent_braking_factor: float = 1.0, descending_power: float = 20.0,
            gravel_power_factor: float = 0.90, smoothing: float = 50.0,
            smoothing_override: bool = False) -> dict | None:
        """Get cached result, returns None if not found."""
        key = self._make_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_power_factor, smoothing, smoothing_override)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key][0]
            self.misses += 1
            return None

    def set(self, url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
            descent_braking_factor: float, descending_power: float, gravel_power_factor: float, smoothing: float,
            smoothing_override: bool, result: dict) -> None:
        """Store result in cache."""
        key = self._make_key(url, climbing_power, flat_power, mass, headwind, descent_braking_factor, descending_power, gravel_power_factor, smoothing, smoothing_override)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (result, time.time())
            # Evict oldest if over limit
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            # Estimate ~1.5 KB per entry based on typical result dict size
            memory_kb = round(len(self.cache) * 1.5, 1)
            return {
                "hit_rate": f"{hit_rate:.1f}%",
                "hits": self.hits,
                "max_size": self.max_size,
                "misses": self.misses,
                "size": len(self.cache),
                "memory_kb": memory_kb,
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class DictCache:
    """Simple dictionary-based LRU cache with optional TTL.

    Used for climb detection, profile data, and trip analysis caches.
    """

    def __init__(self, max_size: int, ttl_seconds: float | None = None, entry_size_kb: float = 2.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.entry_size_kb = entry_size_kb
        self._cache: dict[str, tuple[Any, float]] = {}
        self._stats = {"hits": 0, "misses": 0}

    def get(self, key: str) -> Any | None:
        """Get cached value if available and not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if self.ttl is None or (time.time() - timestamp < self.ttl):
                self._stats["hits"] += 1
                return value
            else:
                # Expired, remove it
                del self._cache[key]
        self._stats["misses"] += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with LRU eviction."""
        self._cache[key] = (value, time.time())
        # Evict oldest entries if over limit
        if len(self._cache) > self.max_size:
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])
            for k in sorted_keys[:len(self._cache) - self.max_size]:
                del self._cache[k]

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        memory_kb = round(len(self._cache) * self.entry_size_kb, 1)
        return {
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self._stats["hits"],
            "max_size": self.max_size,
            "misses": self._stats["misses"],
            "size": len(self._cache),
            "memory_kb": memory_kb,
        }


class DiskCache:
    """Disk-backed cache with LRU eviction via index file.

    Used for elevation profile images.
    """

    def __init__(self, cache_dir: Path, max_size: int = 525):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.index_path = cache_dir / "cache_index.json"
        self._stats = {"hits": 0, "misses": 0, "zoomed_skipped": 0}

    def _load_index(self) -> dict:
        """Load the cache index (maps cache_key -> timestamp)."""
        if self.index_path.exists():
            try:
                with self.index_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self, index: dict) -> None:
        """Save the cache index."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w") as f:
            json.dump(index, f)

    def get(self, key: str) -> bytes | None:
        """Load cached image if available."""
        path = self.cache_dir / f"{key}.png"
        if path.exists():
            # Update access time in index
            index = self._load_index()
            index[key] = time.time()
            self._save_index(index)
            self._stats["hits"] += 1
            return path.read_bytes()
        self._stats["misses"] += 1
        return None

    def set(self, key: str, data: bytes) -> None:
        """Save image to cache and enforce LRU limit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        path = self.cache_dir / f"{key}.png"
        path.write_bytes(data)

        # Update index
        index = self._load_index()
        index[key] = time.time()

        # Enforce LRU limit
        if len(index) > self.max_size:
            sorted_entries = sorted(index.items(), key=lambda x: x[1])
            to_remove = sorted_entries[: len(index) - self.max_size]
            for k, _ in to_remove:
                old_path = self.cache_dir / f"{k}.png"
                if old_path.exists():
                    old_path.unlink()
                del index[k]

        self._save_index(index)

    def clear(self) -> int:
        """Clear all cached images. Returns number of entries cleared."""
        index = self._load_index()
        count = len(index)
        for key in index:
            path = self.cache_dir / f"{key}.png"
            if path.exists():
                path.unlink()
        if self.index_path.exists():
            self.index_path.unlink()
        self._stats = {"hits": 0, "misses": 0, "zoomed_skipped": 0}
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        index = self._load_index()
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        # Estimate ~37 KB per profile image
        memory_kb = round(len(index) * 37, 1)
        return {
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self._stats["hits"],
            "max_size": self.max_size,
            "misses": self._stats["misses"],
            "size": len(index),
            "memory_kb": memory_kb,
            "zoomed_skipped": self._stats.get("zoomed_skipped", 0),
        }

    def record_zoomed_skip(self) -> None:
        """Record that a zoomed profile request was skipped for caching."""
        self._stats["zoomed_skipped"] = self._stats.get("zoomed_skipped", 0) + 1


# Cache key generation helpers

def make_profile_cache_key(url: str, climbing_power: float, flat_power: float, mass: float, headwind: float,
                           descent_braking_factor: float = 1.0, collapse_stops: bool = False,
                           max_xlim_hours: float | None = None, descending_power: float = 20.0,
                           overlay: str = "", imperial: bool = False,
                           show_gravel: bool = False,
                           max_ylim: float | None = None,
                           max_speed_ylim: float | None = None,
                           max_grade_ylim: float | None = None,
                           gravel_grade: float = 0.0,
                           smoothing: float = 50.0,
                           min_xlim_hours: float | None = None) -> str:
    """Create a unique cache key for elevation profile parameters."""
    config_hash = _get_config_hash()
    key_str = f"{url}|{climbing_power}|{flat_power}|{descending_power}|{mass}|{headwind}|{descent_braking_factor}|{collapse_stops}|{min_xlim_hours}|{max_xlim_hours}|{overlay}|{imperial}|{show_gravel}|{max_ylim}|{max_speed_ylim}|{max_grade_ylim}|{gravel_grade}|{smoothing}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def make_ride_profile_cache_key(url: str, sensitivity: int, aspect: float, params_hash: str) -> str:
    """Create cache key for ride profile images."""
    config_hash = _get_config_hash()
    key_str = f"ride|{url}|{sensitivity}|{aspect:.1f}|{params_hash}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def make_climb_cache_key(url: str, sensitivity: int, params_hash: str) -> str:
    """Create cache key for climb detection results."""
    return f"{url}|{sensitivity}|{params_hash}"


def make_profile_data_cache_key(url: str, params: Any, smoothing: float, smoothing_override: bool) -> str:
    """Create a cache key for route profile data."""
    key_str = f"{url}|{params.climbing_power}|{params.flat_power}|{params.descending_power}|{params.total_mass}|{params.headwind}|{params.unpaved_power_factor}|{params.gravel_work_multiplier}|{smoothing}|{smoothing_override}"
    return hashlib.md5(key_str.encode()).hexdigest()


def make_trip_profile_data_cache_key(url: str, collapse_stops: bool, smoothing: float,
                                     trip_smoothing_enabled: bool = True) -> str:
    """Create a cache key for trip profile data."""
    config_hash = _get_config_hash()
    key_str = f"trip|{url}|{collapse_stops}|{smoothing}|{trip_smoothing_enabled}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def make_trip_analysis_cache_key(url: str, smoothing: float,
                                 trip_smoothing_enabled: bool = True) -> str:
    """Create a cache key for trip analysis."""
    config_hash = _get_config_hash()
    key_str = f"trip_analysis|{url}|{smoothing}|{trip_smoothing_enabled}|{config_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()
