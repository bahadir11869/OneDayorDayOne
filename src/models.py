#!/usr/bin/env python3
"""Veri Modelleri - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class EVState(Enum):
    WAITING = "waiting"
    CHARGING = "charging"
    DONE = "done"


class StationType(Enum):
    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    STANDARD = "standard"


@dataclass(frozen=True)
class EVModel:
    model_name: str
    battery_capacity_kwh: float
    max_dc_power_kw: float
    probability: float


@dataclass
class GridLimitPolicy:
    # 1600 kVA trafo × 0.90 güç faktörü × 0.88 safety margin
    trafo_max_kw: float = round(1600 * 0.90 * 0.88, 1)   # → 1267.2 kW
    evening_peak_kw: float = 950.0   # 17-22 pik: baz yük ~700 kW, EV'ye dar bütçe
    peak_start_min: int = 1020  # 17:00
    peak_end_min: int = 1320    # 22:00


@dataclass
class MetricsSummary:
    peak_power_kw: float
    overload_minutes: int
    total_overload_kwh: float
    avg_delay_minutes: float
    evs_completed: int
    protected_capacity_percent: float
    avg_grid_limit_kw: float


@dataclass
class VehicleSession:
    session_id: str
    model_name: str
    arrival_minute: int
    initial_soc: float
    final_soc: float
    wait_time_minutes: int
    charge_time_minutes: int
    energy_delivered_kwh: float


@dataclass
class SimulationResult:
    scenario_name: str
    power_timeseries: np.ndarray
    metrics_summary: MetricsSummary
    vehicle_sessions: List[VehicleSession]
    grid_limit_timeseries: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class EV:
    session_id: str
    model_name: str
    battery_capacity_kwh: float
    max_dc_power_kw: float
    arrival_minute: int
    initial_soc: float
    target_soc: float = 0.8
    current_soc: float = field(init=False)
    state: EVState = field(default=EVState.WAITING)
    charge_start_minute: Optional[int] = None
    departure_minute: Optional[int] = None
    charge_minutes: int = 0
    energy_delivered_kwh: float = 0.0

    def __post_init__(self):
        self.current_soc = self.initial_soc

    @property
    def energy_needed_kwh(self) -> float:
        return max(0.0, (self.target_soc - self.current_soc) * self.battery_capacity_kwh)

    @property
    def is_satisfied(self) -> bool:
        return self.current_soc >= self.target_soc

    def apply_power(self, power_kw: float, minute: int) -> float:
        if self.current_soc >= self.target_soc: return 0.0
        energy_to_absorb = min(power_kw * (1.0 / 60.0), self.energy_needed_kwh)
        self.current_soc = min(self.current_soc + energy_to_absorb / self.battery_capacity_kwh, self.target_soc)
        self.energy_delivered_kwh += energy_to_absorb
        if power_kw > 0.0: self.charge_minutes += 1
        return energy_to_absorb

    @property
    def wait_time_minutes(self) -> int:
        return self.charge_start_minute - self.arrival_minute if self.charge_start_minute is not None else 0


@dataclass
class ChargingStation:
    station_id: str
    station_type: StationType
    max_power_kw: float
    current_ev: Optional[EV] = None

    def is_available(self) -> bool:
        return self.current_ev is None

    def effective_max_power_kw(self) -> float:
        return min(self.max_power_kw, self.current_ev.max_dc_power_kw) if self.current_ev else 0.0


@dataclass
class EnvironmentProfile:
    name: str
    base_min_kw: float
    base_max_kw: float
    operation_start_hour: float
    operation_duration_hours: float
    morning_peak_hour: float
    morning_peak_kw: float
    morning_peak_width: float
    evening_peak_hour: float
    evening_peak_kw: float
    evening_peak_width: float
    noise_kw: float
    load_min_kw: float
    load_max_kw: float


@dataclass
class GridConfig:
    trafo_kva: float
    power_factor: float = 0.90
    safety_margin: float = 0.88
    evening_peak_kw: float = 950.0
    peak_start_hour: float = 17.0
    peak_end_hour: float = 22.0

    @property
    def trafo_max_kw(self) -> float:
        return round(self.trafo_kva * self.power_factor * self.safety_margin, 1)

    @property
    def peak_start_min(self) -> int:
        return int(self.peak_start_hour * 60)

    @property
    def peak_end_min(self) -> int:
        return int(self.peak_end_hour * 60)


@dataclass
class ArrivalPattern:
    mean_hour: float
    std_minutes: float
    fraction: float


@dataclass
class FleetProfile:
    daily_ev_count: int
    ev_models: List[EVModel]
    arrival_patterns: List[ArrivalPattern]
    initial_soc_min: float = 0.10
    initial_soc_max: float = 0.40
    target_soc: float = 0.80


@dataclass
class StationLayout:
    stations: List[ChargingStation]


@dataclass
class ScenarioConfig:
    name: str
    environment: EnvironmentProfile
    grid: GridConfig
    fleet: FleetProfile
    layout: StationLayout

    def to_grid_limit_policy(self) -> GridLimitPolicy:
        return GridLimitPolicy(
            trafo_max_kw=self.grid.trafo_max_kw,
            evening_peak_kw=self.grid.evening_peak_kw,
            peak_start_min=self.grid.peak_start_min,
            peak_end_min=self.grid.peak_end_min,
        )

    def to_dict(self) -> dict:
        import dataclasses
        def convert(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        return convert(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        env = EnvironmentProfile(**d['environment'])
        grid = GridConfig(**d['grid'])
        patterns = [ArrivalPattern(**p) for p in d['fleet']['arrival_patterns']]
        models = [EVModel(**m) for m in d['fleet']['ev_models']]
        fleet_d = {k: v for k, v in d['fleet'].items() if k not in ('arrival_patterns', 'ev_models')}
        fleet = FleetProfile(ev_models=models, arrival_patterns=patterns, **fleet_d)
        stations = [ChargingStation(s['station_id'], StationType(s['station_type']), s['max_power_kw'])
                    for s in d['layout']['stations']]
        return cls(name=d['name'], environment=env, grid=grid, fleet=fleet,
                   layout=StationLayout(stations))
