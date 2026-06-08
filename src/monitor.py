"""
System telemetry for the booth dashboard.

Reads thermal + power consumption from sysfs when available, falls back
to plausible simulated values otherwise. Exposes a single `snapshot()`
that returns a dict for the UI widget.
"""
from __future__ import annotations

import glob
import os
import random
import time
from dataclasses import dataclass, asdict

import config


@dataclass
class Telemetry:
    temp_c:    float = 0.0
    power_w:   float = 0.0
    throttled: bool  = False
    cpu_pct:   float = 0.0


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _read_temperature_c() -> float:
    v = _read_int(config.THERMAL_ZONE_PATH)
    if v is not None:
        # Kernel exposes milli-degrees C
        return v / 1000.0
    # Try any thermal zone
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        v = _read_int(p)
        if v is not None:
            return v / 1000.0
    # Simulated value with light random walk
    return 55.0 + random.uniform(-3, 3)


def _read_power_w() -> float:
    """
    Best-effort board power. QCS8550 SBCs expose voltage_now / current_now
    via /sys/class/power_supply or /sys/bus/iio. Otherwise simulate.
    """
    base = os.path.join(config.POWER_SUPPLY_PATH)
    if os.path.isdir(base):
        for entry in os.listdir(base):
            v_path = os.path.join(base, entry, "voltage_now")
            i_path = os.path.join(base, entry, "current_now")
            v = _read_int(v_path)
            i = _read_int(i_path)
            if v is not None and i is not None:
                # uV * uA -> pW; divide by 1e12 for watts (then * 1e6 = 1e6 -> W is 1e-12 already)
                return (v / 1e6) * (i / 1e6)
    # Fallback - synthetic but bounded
    return 4.8 + random.uniform(-0.4, 0.6)


def _read_cpu_pct() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return random.uniform(15, 45)


class HealthMonitor:
    def __init__(self) -> None:
        self._last = Telemetry()
        self._last_t = 0.0

    def snapshot(self) -> Telemetry:
        # Cache for 250 ms to avoid hammering sysfs at UI refresh rate
        now = time.time()
        if now - self._last_t < 0.25:
            return self._last
        t = Telemetry(
            temp_c=_read_temperature_c(),
            power_w=_read_power_w(),
            cpu_pct=_read_cpu_pct(),
        )
        t.throttled = t.temp_c >= config.THERMAL_THROTTLE_TEMP
        self._last = t
        self._last_t = now
        return t

    def as_dict(self) -> dict:
        return asdict(self.snapshot())
