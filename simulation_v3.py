#!/usr/bin/env python3
"""
EV Charging Load Balancing Simulation v4.5 (STATION MATRIX EXCEL)
- Her dakika sistemin (Kuyruk + Prizler) fotoğrafı çekilir.
- Excel'e İstasyon (S1-S5) x Zaman (0-1439 dk) matrisi olarak basılır.
"""

from __future__ import annotations
import copy
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

# ==============================================================================
# Veri Modelleri
# ==============================================================================

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

# ==============================================================================
# Senaryo Konfigürasyonu
# ==============================================================================

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
    def from_dict(cls, d: dict) -> ScenarioConfig:
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

# ==============================================================================
# Veri Üreticiler
# ==============================================================================

_DEFAULT_EV_MODELS = [
    EVModel("Togg T10X",         88.5, 150.0, 0.25),
    EVModel("Tesla Model Y",     75.0, 250.0, 0.20),
    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
    EVModel("BYD Atto 3",        60.4,  88.0, 0.15),
    EVModel("MG4 Standard",      51.0, 117.0, 0.10),
    EVModel("Renault Megane",    60.0, 130.0, 0.10),
    EVModel("Porsche Taycan",    93.4, 270.0, 0.05),
]

class ArrivalGenerator:
    def __init__(self, fleet: FleetProfile):
        self.fleet = fleet

    def generate_arrivals(self, rng: np.random.Generator) -> Dict[int, List[EV]]:
        patterns = self.fleet.arrival_patterns
        remaining = self.fleet.daily_ev_count
        counts = []
        for i, p in enumerate(patterns):
            n = int(self.fleet.daily_ev_count * p.fraction) if i < len(patterns) - 1 else remaining
            remaining -= n
            counts.append(n)

        arrs = np.concatenate([
            rng.normal(p.mean_hour * 60, p.std_minutes, counts[i])
            for i, p in enumerate(patterns)
        ])
        rng.shuffle(arrs)

        schedule: Dict[int, List[EV]] = {}
        probs = [m.probability for m in self.fleet.ev_models]
        for i, m_val in enumerate(np.clip(arrs, 0, 1439)):
            minute_idx = int(m_val)
            model = rng.choice(self.fleet.ev_models, p=probs)
            if minute_idx not in schedule:
                schedule[minute_idx] = []
            schedule[minute_idx].append(EV(
                f"EV_{i+1:02d}", model.model_name, model.battery_capacity_kwh,
                model.max_dc_power_kw, minute_idx,
                rng.uniform(self.fleet.initial_soc_min, self.fleet.initial_soc_max),
                target_soc=self.fleet.target_soc,
            ))
        return schedule

class BackgroundLoadGenerator:
    @staticmethod
    def generate(rng: np.random.Generator, profile: EnvironmentProfile) -> np.ndarray:
        hrs = np.arange(1440) / 60.0
        base = profile.base_min_kw + (profile.base_max_kw - profile.base_min_kw) * np.clip(
            0.5 * (1 - np.cos(np.pi * np.clip(
                hrs - profile.operation_start_hour, 0, profile.operation_duration_hours
            ) / profile.operation_duration_hours)), 0, 1)
        peak_m = profile.morning_peak_kw * np.exp(
            -0.5 * ((hrs - profile.morning_peak_hour) / profile.morning_peak_width) ** 2)
        peak_e = profile.evening_peak_kw * np.exp(
            -0.5 * ((hrs - profile.evening_peak_hour) / profile.evening_peak_width) ** 2)
        return np.clip(
            base + peak_m + peak_e + rng.normal(0, profile.noise_kw, 1440),
            profile.load_min_kw, profile.load_max_kw
        )

class Scenarios:
    @staticmethod
    def avm_medium() -> ScenarioConfig:
        """Orta ölçek AVM — varsayılan senaryo"""
        return ScenarioConfig(
            name="AVM_Orta",
            environment=EnvironmentProfile(
                name="AVM_Orta",
                base_min_kw=180.0, base_max_kw=500.0,
                operation_start_hour=7.0, operation_duration_hours=15.0,
                morning_peak_hour=10.0, morning_peak_kw=80.0, morning_peak_width=1.2,
                evening_peak_hour=18.5, evening_peak_kw=140.0, evening_peak_width=1.5,
                noise_kw=25.0, load_min_kw=150.0, load_max_kw=800.0,
            ),
            grid=GridConfig(trafo_kva=1600, evening_peak_kw=950.0,
                            peak_start_hour=17.0, peak_end_hour=22.0),
            fleet=FleetProfile(
                daily_ev_count=35,
                ev_models=_DEFAULT_EV_MODELS,
                arrival_patterns=[
                    ArrivalPattern(mean_hour=9.5,  std_minutes=40, fraction=0.25),
                    ArrivalPattern(mean_hour=13.0, std_minutes=35, fraction=0.25),
                    ArrivalPattern(mean_hour=18.5, std_minutes=75, fraction=0.50),
                ],
            ),
            layout=StationLayout([
                ChargingStation("S1", StationType.ULTRA_FAST, 200),
                ChargingStation("S2", StationType.ULTRA_FAST, 200),
                ChargingStation("S3", StationType.FAST, 180),
                ChargingStation("S4", StationType.FAST, 180),
                ChargingStation("S5", StationType.STANDARD, 120),
            ]),
        )

    @staticmethod
    def office_large() -> ScenarioConfig:
        """Büyük ofis binası — 08-19 operasyon, sabah ağırlıklı geliş"""
        return ScenarioConfig(
            name="Ofis_Buyuk",
            environment=EnvironmentProfile(
                name="Ofis_Buyuk",
                base_min_kw=50.0, base_max_kw=350.0,
                operation_start_hour=8.0, operation_duration_hours=11.0,
                morning_peak_hour=9.0, morning_peak_kw=120.0, morning_peak_width=1.0,
                evening_peak_hour=17.5, evening_peak_kw=60.0, evening_peak_width=1.0,
                noise_kw=15.0, load_min_kw=40.0, load_max_kw=500.0,
            ),
            grid=GridConfig(trafo_kva=1000, evening_peak_kw=550.0,
                            peak_start_hour=8.0, peak_end_hour=19.0),
            fleet=FleetProfile(
                daily_ev_count=40,
                ev_models=_DEFAULT_EV_MODELS,
                arrival_patterns=[
                    ArrivalPattern(mean_hour=8.5,  std_minutes=30, fraction=0.70),
                    ArrivalPattern(mean_hour=13.0, std_minutes=45, fraction=0.30),
                ],
            ),
            layout=StationLayout([
                ChargingStation("S1", StationType.FAST, 150),
                ChargingStation("S2", StationType.FAST, 150),
                ChargingStation("S3", StationType.STANDARD, 22),
                ChargingStation("S4", StationType.STANDARD, 22),
                ChargingStation("S5", StationType.STANDARD, 22),
            ]),
        )

    @staticmethod
    def hotel() -> ScenarioConfig:
        """Otel — 7/24 baz yük, araçlar akşam gelir"""
        return ScenarioConfig(
            name="Otel",
            environment=EnvironmentProfile(
                name="Otel",
                base_min_kw=120.0, base_max_kw=400.0,
                operation_start_hour=0.0, operation_duration_hours=24.0,
                morning_peak_hour=8.0,  morning_peak_kw=100.0, morning_peak_width=1.5,
                evening_peak_hour=20.0, evening_peak_kw=130.0, evening_peak_width=2.0,
                noise_kw=20.0, load_min_kw=100.0, load_max_kw=600.0,
            ),
            grid=GridConfig(trafo_kva=1200, evening_peak_kw=700.0,
                            peak_start_hour=18.0, peak_end_hour=23.0),
            fleet=FleetProfile(
                daily_ev_count=30,
                ev_models=_DEFAULT_EV_MODELS,
                arrival_patterns=[
                    ArrivalPattern(mean_hour=15.0, std_minutes=90, fraction=0.40),
                    ArrivalPattern(mean_hour=20.0, std_minutes=60, fraction=0.60),
                ],
            ),
            layout=StationLayout([
                ChargingStation("S1", StationType.FAST, 150),
                ChargingStation("S2", StationType.FAST, 150),
                ChargingStation("S3", StationType.STANDARD, 22),
                ChargingStation("S4", StationType.STANDARD, 22),
                ChargingStation("S5", StationType.STANDARD, 22),
            ]),
        )

    @staticmethod
    def hospital() -> ScenarioConfig:
        """Hastane — 7/24 düz yük, vardiya değişimi geliş dalgaları"""
        return ScenarioConfig(
            name="Hastane",
            environment=EnvironmentProfile(
                name="Hastane",
                base_min_kw=400.0, base_max_kw=600.0,
                operation_start_hour=0.0, operation_duration_hours=24.0,
                morning_peak_hour=8.0,  morning_peak_kw=80.0, morning_peak_width=0.8,
                evening_peak_hour=16.0, evening_peak_kw=70.0, evening_peak_width=0.8,
                noise_kw=20.0, load_min_kw=380.0, load_max_kw=750.0,
            ),
            grid=GridConfig(trafo_kva=2000, evening_peak_kw=1100.0,
                            peak_start_hour=7.0, peak_end_hour=20.0),
            fleet=FleetProfile(
                daily_ev_count=25,
                ev_models=_DEFAULT_EV_MODELS,
                arrival_patterns=[
                    ArrivalPattern(mean_hour=7.5,  std_minutes=20, fraction=0.34),
                    ArrivalPattern(mean_hour=15.5, std_minutes=20, fraction=0.33),
                    ArrivalPattern(mean_hour=23.5, std_minutes=20, fraction=0.33),
                ],
            ),
            layout=StationLayout([
                ChargingStation("S1", StationType.FAST, 150),
                ChargingStation("S2", StationType.FAST, 150),
                ChargingStation("S3", StationType.STANDARD, 22),
                ChargingStation("S4", StationType.STANDARD, 22),
                ChargingStation("S5", StationType.STANDARD, 22),
            ]),
        )

    @staticmethod
    def airport() -> ScenarioConfig:
        """Havalimanı — yüksek baz yük, sabah+akşam uçuş dalgaları"""
        return ScenarioConfig(
            name="Havalimani",
            environment=EnvironmentProfile(
                name="Havalimani",
                base_min_kw=500.0, base_max_kw=1200.0,
                operation_start_hour=4.0, operation_duration_hours=20.0,
                morning_peak_hour=7.0,  morning_peak_kw=300.0, morning_peak_width=1.5,
                evening_peak_hour=18.0, evening_peak_kw=350.0, evening_peak_width=2.0,
                noise_kw=50.0, load_min_kw=450.0, load_max_kw=1800.0,
            ),
            grid=GridConfig(trafo_kva=4000, evening_peak_kw=2200.0,
                            peak_start_hour=6.0, peak_end_hour=22.0),
            fleet=FleetProfile(
                daily_ev_count=80,
                ev_models=_DEFAULT_EV_MODELS,
                arrival_patterns=[
                    ArrivalPattern(mean_hour=6.5,  std_minutes=60, fraction=0.50),
                    ArrivalPattern(mean_hour=17.5, std_minutes=60, fraction=0.50),
                ],
            ),
            layout=StationLayout([
                ChargingStation("S1", StationType.ULTRA_FAST, 200),
                ChargingStation("S2", StationType.ULTRA_FAST, 200),
                ChargingStation("S3", StationType.FAST, 180),
                ChargingStation("S4", StationType.FAST, 180),
                ChargingStation("S5", StationType.STANDARD, 120),
            ]),
        )

# ==============================================================================
# Kontrolcüler (Algoritmasız ve Algoritmalı)
# ==============================================================================
class UnmanagedController:
    def __init__(self, stations, limit_policy, bg_load):
        self.stations = stations
        self.policy = limit_policy
        self.bg_load = bg_load
        self.queue = []
        self.power_log = []
        self.limit_log = []
        self.completed = []
        self.timeline_log = []

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        tod = minute % 1440
        limit = self.policy.evening_peak_kw if self.policy.peak_start_min <= tod < self.policy.peak_end_min else self.policy.trafo_max_kw
        base = self.bg_load[minute]
        allocs = {s.station_id: s.effective_max_power_kw() for s in self.stations}

        # MASTER LOG (Durum Fotoğrafı Çek)
        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-"})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        for s in self.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    self.completed.append(s.current_ev)
                    s.current_ev = None

        self.power_log.append(base + sum(allocs.values()))
        self.limit_log.append(limit)

class ManagedController:
    def __init__(self, stations, limit_policy, bg_load):
        self.stations = stations
        self.policy = limit_policy
        self.bg_load = bg_load
        self.queue = []
        self.power_log = []
        self.limit_log = []
        self.completed = []
        self.timeline_log = []

    def allocate_power(self, minute: int) -> Dict[str, float]:
        tod = minute % 1440
        is_peak = self.policy.peak_start_min <= tod < self.policy.peak_end_min
        limit = self.policy.evening_peak_kw if is_peak else self.policy.trafo_max_kw
        base = self.bg_load[minute]

        occ = [s for s in self.stations if s.current_ev]
        active = [s for s in occ if not s.current_ev.is_satisfied]
        allocs = {s.station_id: 0.0 for s in self.stations}
        if not active: return allocs

        budget = max(0.0, limit - base - 0.01)
        vehicles = []
        for s in active:
            ev = s.current_ev
            max_p = s.effective_max_power_kw()
            ideal = ((ev.target_soc - ev.initial_soc) * ev.battery_capacity_kwh / max_p) * 60.0 if max_p > 0 else 999
            starve = max(0.0, ev.charge_minutes - (ideal * 1.2))
            t_fin = ev.energy_needed_kwh / max_p * 60.0 if max_p > 0 else 9999
            # Yüksek skor = daha acil: starvation baskın, sonra bitiş yakınlığı
            priority = starve * 10.0 + (1.0 / max(t_fin, 1.0))
            vehicles.append({'id': s.station_id, 'max_p': max_p, 'priority': priority})

        # Aciliyet sırasına diz (en acil başta)
        vehicles.sort(key=lambda x: -x['priority'])

        # 1. Geçiş: herkese "ölmeyecek kadar" minimum ayır (11 kW veya max_p hangisi küçükse)
        keep_alive = 11.0
        for v in vehicles:
            give = min(keep_alive, v['max_p'], budget)
            allocs[v['id']] = give
            budget -= give

        # 2. Geçiş: kalan bütçeyi aciliyet sırasıyla dağıt (en acil önce dolar)
        for v in vehicles:
            if budget <= 0.1: break
            extra = min(v['max_p'] - allocs[v['id']], budget)
            allocs[v['id']] += extra
            budget -= extra
        return allocs

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        tod = minute % 1440
        limit = self.policy.evening_peak_kw if self.policy.peak_start_min <= tod < self.policy.peak_end_min else self.policy.trafo_max_kw
        allocs = self.allocate_power(minute)

        # MASTER LOG (Durum Fotoğrafı Çek)
        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-", "BazGüç (kW)": round(self.bg_load[minute], 1)})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        for s in self.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    self.completed.append(s.current_ev)
                    s.current_ev = None

        self.power_log.append(self.bg_load[minute] + sum(allocs.values()))
        self.limit_log.append(limit)

# ==============================================================================
# Yeni Kontrolcüler
# ==============================================================================

class SRPTController:
    """Shortest Remaining Processing Time — en az enerjisi kalan araca öncelik verir.
    15 dk üzeri bekleyen araçların skoru ağırlık katsayısıyla düşürülür (= öncelik artar)."""

    def __init__(self, stations, limit_policy, bg_load):
        self.stations = stations
        self.policy = limit_policy
        self.bg_load = bg_load
        self.queue = []
        self.power_log = []
        self.limit_log = []
        self.completed = []
        self.timeline_log = []

    def allocate_power(self, minute: int) -> Dict[str, float]:
        tod = minute % 1440
        is_peak = self.policy.peak_start_min <= tod < self.policy.peak_end_min
        limit = self.policy.evening_peak_kw if is_peak else self.policy.trafo_max_kw
        base = self.bg_load[minute]

        active = [s for s in self.stations if s.current_ev and not s.current_ev.is_satisfied]
        allocs = {s.station_id: 0.0 for s in self.stations}
        if not active:
            return allocs

        budget = max(0.0, limit - base - 0.01)

        def srpt_key(s):
            ev = s.current_ev
            energy = ev.energy_needed_kwh
            wait = minute - ev.arrival_minute
            if wait > 15:
                # Uzun bekleyenlerin efektif energy_needed değeri düşer → öncelik artar
                penalty = 1.0 / (1.0 + 0.05 * (wait - 15))
                energy = energy * penalty
            return energy

        # En düşük (ağırlıklı) energy_needed = en yüksek öncelik
        sorted_active = sorted(active, key=srpt_key)

        for s in sorted_active:
            if budget <= 0.01:
                break
            give = min(s.effective_max_power_kw(), budget)
            allocs[s.station_id] = give
            budget -= give

        return allocs

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        tod = minute % 1440
        limit = self.policy.evening_peak_kw if self.policy.peak_start_min <= tod < self.policy.peak_end_min else self.policy.trafo_max_kw
        allocs = self.allocate_power(minute)

        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-", "BazGüç (kW)": round(self.bg_load[minute], 1)})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        for s in self.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    self.completed.append(s.current_ev)
                    s.current_ev = None

        self.power_log.append(self.bg_load[minute] + sum(allocs.values()))
        self.limit_log.append(limit)


class WaterFillingController:
    """Water-Filling — bütçeyi araçlara eşit böler; limite takılan araçın artığını
    kalan araçlara yeniden eşit dağıtır. Döngü bütçe bitene kadar sürer."""

    def __init__(self, stations, limit_policy, bg_load):
        self.stations = stations
        self.policy = limit_policy
        self.bg_load = bg_load
        self.queue = []
        self.power_log = []
        self.limit_log = []
        self.completed = []
        self.timeline_log = []

    def allocate_power(self, minute: int) -> Dict[str, float]:
        tod = minute % 1440
        is_peak = self.policy.peak_start_min <= tod < self.policy.peak_end_min
        limit = self.policy.evening_peak_kw if is_peak else self.policy.trafo_max_kw
        base = self.bg_load[minute]

        active = [s for s in self.stations if s.current_ev and not s.current_ev.is_satisfied]
        allocs = {s.station_id: 0.0 for s in self.stations}
        if not active:
            return allocs

        budget = max(0.0, limit - base - 0.01)

        # Su doldurma: cap değeri küçük olanlar önce kesilebileceğinden artan sıraya diz
        sorted_active = sorted(active, key=lambda s: s.effective_max_power_kw())
        n = len(sorted_active)

        for i, s in enumerate(sorted_active):
            if budget <= 0.01:
                break
            remaining_count = n - i
            share = budget / remaining_count          # kalan bütçeyi kalan araçlara eşit böl
            give = min(s.effective_max_power_kw(), share)
            allocs[s.station_id] = give
            budget -= give                            # artan miktar sonraki araçlara geçer

        return allocs

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        tod = minute % 1440
        limit = self.policy.evening_peak_kw if self.policy.peak_start_min <= tod < self.policy.peak_end_min else self.policy.trafo_max_kw
        allocs = self.allocate_power(minute)

        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-", "BazGüç (kW)": round(self.bg_load[minute], 1)})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        for s in self.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    self.completed.append(s.current_ev)
                    s.current_ev = None

        self.power_log.append(self.bg_load[minute] + sum(allocs.values()))
        self.limit_log.append(limit)


class DynamicFairController:
    """Dinamik Ağırlıklı Aciliyet — bekleme süresi / kalan enerji skoruna göre
    orantılı dağıtım yapar; SoC > %80 ise skoru %80 düşürür; taşan gücü yeniden dağıtır."""

    def __init__(self, stations, limit_policy, bg_load):
        self.stations = stations
        self.policy = limit_policy
        self.bg_load = bg_load
        self.queue = []
        self.power_log = []
        self.limit_log = []
        self.completed = []
        self.timeline_log = []

    def allocate_power(self, minute: int) -> Dict[str, float]:
        tod = minute % 1440
        is_peak = self.policy.peak_start_min <= tod < self.policy.peak_end_min
        limit = self.policy.evening_peak_kw if is_peak else self.policy.trafo_max_kw
        base = self.bg_load[minute]

        active = [s for s in self.stations if s.current_ev and not s.current_ev.is_satisfied]
        allocs = {s.station_id: 0.0 for s in self.stations}
        if not active:
            return allocs

        budget = max(0.0, limit - base - 0.01)
        caps = {s.station_id: s.effective_max_power_kw() for s in active}

        def compute_score(s):
            ev = s.current_ev
            wait = minute - ev.arrival_minute + 1        # +1: 0'a bölünmeyi önle
            energy = max(ev.energy_needed_kwh, 0.01)
            soc_penalty = 0.2 if ev.current_soc > 0.80 else 1.0
            return (wait / energy) * soc_penalty

        pending = list(active)

        while budget > 0.01 and pending:
            scores = {s.station_id: compute_score(s) for s in pending}
            total_score = sum(scores.values())
            if total_score <= 0:
                break

            overflow = 0.0
            new_pending = []
            for s in pending:
                proportion = scores[s.station_id] / total_score
                want = budget * proportion
                remaining_cap = caps[s.station_id] - allocs[s.station_id]
                if want >= remaining_cap:
                    allocs[s.station_id] += remaining_cap
                    overflow += want - remaining_cap      # artığı havuza geri at
                else:
                    allocs[s.station_id] += want
                    new_pending.append(s)

            if not new_pending or overflow < 0.01:
                break

            budget = overflow
            pending = new_pending

        return allocs

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        tod = minute % 1440
        limit = self.policy.evening_peak_kw if self.policy.peak_start_min <= tod < self.policy.peak_end_min else self.policy.trafo_max_kw
        allocs = self.allocate_power(minute)

        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-", "BazGüç (kW)": round(self.bg_load[minute], 1)})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        for s in self.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    self.completed.append(s.current_ev)
                    s.current_ev = None

        self.power_log.append(self.bg_load[minute] + sum(allocs.values()))
        self.limit_log.append(limit)


class Simulation:
    def __init__(self, ctrl, schedule):
        self.ctrl = ctrl
        self.sched = schedule

    def run(self) -> SimulationResult:
        for m in range(1440):
            for ev in self.sched.get(m, []): self.ctrl.queue.append(ev)
            self.ctrl.step(m)

        p = np.array(self.ctrl.power_log)
        l = np.array(self.ctrl.limit_log)
        over = p > l
        m = MetricsSummary(float(p.max()), int(over.sum()), float(np.where(over, p-l, 0).sum()/60), float(np.mean([e.wait_time_minutes for e in self.ctrl.completed])) if self.ctrl.completed else 0, len(self.ctrl.completed), 100.0 if not over.any() else float((l/np.maximum(p, 1e-9)).clip(max=1.0).mean()*100), float(l.mean()))
        s = [VehicleSession(e.session_id, e.model_name, e.arrival_minute, e.initial_soc, e.current_soc, e.wait_time_minutes, e.charge_minutes, e.energy_delivered_kwh) for e in self.ctrl.completed]
        return SimulationResult(self.ctrl.__class__.__name__, p, m, s, l)


# ==============================================================================
# Excel İhracat (MATRİS FORMATI)
# ==============================================================================
def build_station_matrix(timeline_log):
    """Zaman çizelgesini 1440 satırlık Matris formuna dönüştürür"""
    rows = {m: {"Dakika": m, "Saat": f"{m//60:02d}:{m%60:02d}", "S1": "-", "S2": "-", "S3": "-", "S4": "-", "S5": "-", "Kuyruk": "-"} for m in range(1440)}
    queues = {m: [] for m in range(1440)}
    
    for t in timeline_log:
        m = t["Dakika"]
        if t["Durum"] == "Şarjda":
            rows[m][t["İstasyon"]] = f"{t['Araç ID']} | {t['Güç (kW)']}kW | %{t['SoC (%)']}  | {t.get('BazGüç (kW)', '-')}"
        elif t["Durum"] == "Kuyrukta":
            queues[m].append(t['Araç ID'])
            
    for m in range(1440):
        if queues[m]:
            rows[m]["Kuyruk"] = ", ".join(queues[m])
            
    return list(rows.values())

def export_comparative_excel(ctrl_u: UnmanagedController, ctrl_m: ManagedController, fn="ev_sarj_karsilastirma_raporu.xlsx"):
    ud = {e.session_id: e for e in ctrl_u.completed}
    
    # 1. ÖZET SAYFASI
    sum_data = []
    for em in ctrl_m.completed:
        eu = ud.get(em.session_id)
        if not eu: continue
        sum_data.append({
            "Araç ID": em.session_id,
            "Model": em.model_name,
            "Geliş Dakikası": em.arrival_minute,
            "Geliş Saati": f"{em.arrival_minute//60:02d}:{em.arrival_minute%60:02d}",
            "Algoritmasız Şarj Süresi (dk)": eu.charge_minutes,
            "Algoritmalı Şarj Süresi (dk)": em.charge_minutes,
            "Şarj Süresi Farkı (dk)": em.charge_minutes - eu.charge_minutes,
            "Algoritmasız Bekleme (dk)": eu.wait_time_minutes,
            "Algoritmalı Bekleme (dk)": em.wait_time_minutes,
            "Bitiş SoC (%)": round(em.current_soc * 100, 1)
        })

    try:
        with pd.ExcelWriter(fn, engine='openpyxl') as w:
            pd.DataFrame(sum_data).to_excel(w, sheet_name='Özet Karşılaştırma', index=False)
            
            # MATRİSLERİ YAZ
            if ctrl_u.timeline_log:
                pd.DataFrame(build_station_matrix(ctrl_u.timeline_log)).to_excel(w, sheet_name='Algoritmasiz_Istasyon_Logu', index=False)
            if ctrl_m.timeline_log:
                pd.DataFrame(build_station_matrix(ctrl_m.timeline_log)).to_excel(w, sheet_name='Algoritmali_Istasyon_Logu', index=False)
                
        print(f"\n✓ ✓ ✓ İSTASYON BAZLI (MATRİS) EXCEL RAPORU OLUŞTURULDU: {fn} ✓ ✓ ✓\n")
    except Exception as e:
        print(f"Excel aktarımı başarısız: {e}")

def export_multi_controller_excel(all_controllers: list, fn="ev_coklu_kontrolcu_raporu.xlsx"):
    """Tüm kontrolcülerin metriklerini ve istasyon loglarını tek Excel dosyasına yazar."""
    metrics_data = []
    for name, ctrl in all_controllers:
        p = np.array(ctrl.power_log)
        l = np.array(ctrl.limit_log)
        over = p > l
        completed = ctrl.completed
        metrics_data.append({
            "Kontrolcü": name,
            "Tamamlanan Araç": len(completed),
            "Maks Güç (kW)": round(float(p.max()), 1),
            "Aşım Dakikası": int(over.sum()),
            "Toplam Aşım (kWh)": round(float(np.where(over, p - l, 0).sum() / 60), 2),
            "Ort. Bekleme (dk)": round(float(np.mean([e.wait_time_minutes for e in completed])) if completed else 0.0, 1),
            "Ort. Şarj Süresi (dk)": round(float(np.mean([e.charge_minutes for e in completed])) if completed else 0.0, 1),
        })

    try:
        with pd.ExcelWriter(fn, engine='openpyxl') as w:
            pd.DataFrame(metrics_data).to_excel(w, sheet_name='Metrik_Karsilastirma', index=False)
            for name, ctrl in all_controllers:
                if ctrl.timeline_log:
                    sheet_name = (name + "_Log")[:31]
                    pd.DataFrame(build_station_matrix(ctrl.timeline_log)).to_excel(w, sheet_name=sheet_name, index=False)
        print(f"✓ Çoklu kontrolcü Excel raporu: {fn}")
    except Exception as e:
        print(f"Çoklu Excel aktarımı başarısız: {e}")

# ==============================================================================
# Grafik ve Ana Akış
# ==============================================================================
class ExecutiveDashboard:
    @staticmethod
    def create(r_u: SimulationResult, r_m: SimulationResult,
               ctrl_label: str = "Kontrol Sonrası",
               filename: str = "executive_dashboard_v4.png",
               bg_load: Optional[np.ndarray] = None):
        fig = plt.figure(figsize=(16, 30))
        fig.suptitle(f"EV Şarj Yük Dengeleme — {ctrl_label}", fontsize=18, fontweight="bold", y=0.995)
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.25)
        axs = [fig.add_subplot(gs[i]) for i in [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3, slice(None)), (4, slice(None))]]

                        
        mods = sorted(list({s.model_name.split()[0] for s in r_u.vehicle_sessions}))
        x = np.arange(len(mods)); wd = 0.35
        ud = {s.session_id: s for s in r_u.vehicle_sessions}
        d_c, d_w = {m: [] for m in mods}, {m: [] for m in mods}
        for s in r_m.vehicle_sessions:
            if s.session_id in ud:
                m = s.model_name.split()[0]
                d_c[m].append(s.charge_time_minutes - ud[s.session_id].charge_time_minutes)
                d_w[m].append(max(0, s.wait_time_minutes - ud[s.session_id].wait_time_minutes))


        w7 = 0.2
        axs[0].bar(x-1.5*w7, [max(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#e74c3c", ec="black", label="Maks Şarj Artışı")
        axs[0].bar(x-0.5*w7, [min(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#f1948a", ec="black", label="Min Şarj Artışı")
        axs[0].bar(x+0.5*w7, [max(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#2980b9", ec="black", label="Maks Bekleme Artışı")
        axs[0].bar(x+1.5*w7, [min(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#7fb3d5", ec="black", label="Min Bekleme Artışı")
        axs[0].set(xticks=x, xticklabels=mods, ylabel="Delta (dk)", title="Panel 1: Maks/Min Gecikme")
        axs[0].legend(loc="upper left"); axs[0].grid(True, alpha=0.3, axis="y")


        hrs = np.arange(1440) / 60
        axs[6].fill_between(hrs, 0, r_u.power_timeseries, alpha=0.3, color="red", label="Kontrol Öncesi")
        axs[6].plot(hrs, r_m.power_timeseries, lw=2.5, color="darkgreen", label=ctrl_label)
        axs[6].plot(hrs, r_m.grid_limit_timeseries, color="red", ls="--", lw=2, label="Trafo Limiti")
        axs[6].axvspan(17, 22, alpha=0.08, color="orange", label="Şebeke Pik (17-22)")
        axs[6].set(xlabel="Gün Saati", ylabel="Yük (kW)", title="Panel 7: Toplam Yük Profili", xlim=(0,24))
        axs[6].legend(loc="upper left"); axs[6].grid(True, alpha=0.3)

        w = [s.wait_time_minutes for s in r_m.vehicle_sessions]
        bars = axs[1].bar(["0 dk", "1-15 dk", "15+ dk"], [sum(1 for x in w if x==0), sum(1 for x in w if 0<x<=15), sum(1 for x in w if x>15)], color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black")
        axs[1].set(ylabel="Araç", title="Panel 2: Bekleme Dağılımı")
        for b in bars: axs[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", va="bottom", fontweight="bold")


        g = lambda r: {m: [] for m in mods}
        c_u, c_m, w_u, w_m = g(r_u), g(r_m), g(r_u), g(r_m)
        for s in r_u.vehicle_sessions: m=s.model_name.split()[0]; c_u[m].append(s.charge_time_minutes); w_u[m].append(s.wait_time_minutes)
        for s in r_m.vehicle_sessions: m=s.model_name.split()[0]; c_m[m].append(s.charge_time_minutes); w_m[m].append(s.wait_time_minutes)
        

        axs[2].bar(x-wd/2, [np.mean(c_u[m]) for m in mods], wd, color="lightcoral", ec="black", label="Öncesi")
        axs[2].bar(x+wd/2, [np.mean(c_m[m]) for m in mods], wd, color="lightgreen", ec="black", label=ctrl_label)
        axs[2].set(ylabel="Şarj (dk)", title="Panel 3: Şarj Süresi", xticks=x, xticklabels=mods); axs[2].legend()

        axs[3].bar(x-wd/2, [np.mean(w_u[m]) for m in mods], wd, color="#f9c784", ec="black", label="Öncesi")
        axs[3].bar(x+wd/2, [np.mean(w_m[m]) for m in mods], wd, color="#74b9ff", ec="black", label=ctrl_label)
        axs[3].set(ylabel="Bekleme (dk)", title="Panel 4: Bekleme Süresi", xticks=x, xticklabels=mods); axs[3].legend()

        axs[4].axis("off")
        axs[4].text(0.05, 0.95,
                    f"Algoritma: {ctrl_label}\n"
                    f"Kapasite: {r_m.metrics_summary.protected_capacity_percent:.1f}%\n"
                    f"Servis: {r_m.metrics_summary.evs_completed} araç\n"
                    f"Ort. Bekleme: {r_m.metrics_summary.avg_delay_minutes:.1f} dk\n"
                    f"Aşım: {r_m.metrics_summary.overload_minutes} dk  |  {r_m.metrics_summary.total_overload_kwh:.1f} kWh",
                    transform=axs[4].transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.4))

        axs[5].bar(x-wd/2, [np.mean(c_u[m])+np.mean(w_u[m]) for m in mods], wd, color="lightcoral", ec="black", label="Öncesi")
        axs[5].bar(x+wd/2, [np.mean(c_m[m])+np.mean(w_m[m]) for m in mods], wd, color="lightgreen", ec="black", label=ctrl_label)
        axs[5].set(ylabel="Sistem (dk)", title="Panel 6: Toplam Süre", xticks=x, xticklabels=mods); axs[5].legend()





        # ── Panel 8: Baz Yük Profili (İstasyon Yükü Hariç) ───────────────────
        if bg_load is not None:
            axs[7].fill_between(hrs, 0, bg_load, alpha=0.35, color="steelblue", label="Baz Yük")
            axs[7].plot(hrs, bg_load, lw=2, color="steelblue")
            axs[7].plot(hrs, r_m.grid_limit_timeseries, color="red", ls="--", lw=2, label="Trafo Limiti")
            axs[7].axvspan(17, 22, alpha=0.08, color="orange", label="Şebeke Pik (17-22)")
            axs[7].set(xlabel="Gün Saati", ylabel="Yük (kW)",
                       title="Panel 8: Baz Yük Profili (İstasyon Yükü Hariç)", xlim=(0, 24))
            axs[7].legend(loc="upper left"); axs[7].grid(True, alpha=0.3)
        else:
            axs[7].axis("off")
            axs[7].text(0.5, 0.5, "Baz yük verisi mevcut değil",
                        ha="center", va="center", transform=axs[7].transAxes, fontsize=12)

        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Grafik: {filename}")
        plt.show()

def main(generate_new: bool = False, config: Optional[ScenarioConfig] = None):
    if config is None:
        config = Scenarios.avm_medium()
    print(f"EV Yük Dengeleme Simülasyonu başlatılıyor... Senaryo: {config.name}")
    dataset_file = "dataset.json"
    policy = config.to_grid_limit_policy()

    if not generate_new:
        if not os.path.exists(dataset_file):
            print(f"\nHATA: '{dataset_file}' dosyası yok! '--generate-new' ile oluşturun.\n")
            sys.exit(1)
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        schedule = {}
        for v in data['vehicles']:
            m = v['arrival_minute']
            if m not in schedule: schedule[m] = []
            schedule[m].append(EV(v['session_id'], v['model_name'], v['battery_capacity_kwh'],
                                  v['max_dc_power_kw'], m, v['initial_soc'],
                                  target_soc=config.fleet.target_soc))
        bg_load = np.array(data['background_load_profile'])
    else:
        rng = np.random.default_rng(42)
        schedule = ArrivalGenerator(config.fleet).generate_arrivals(rng)
        bg_load = BackgroundLoadGenerator.generate(np.random.default_rng(101), config.environment)
        vehicles_data = [{"session_id": ev.session_id, "model_name": ev.model_name,
                          "battery_capacity_kwh": ev.battery_capacity_kwh,
                          "max_dc_power_kw": ev.max_dc_power_kw,
                          "arrival_minute": ev.arrival_minute, "initial_soc": ev.initial_soc}
                         for m in sorted(schedule.keys()) for ev in schedule[m]]
        with open(dataset_file, 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "scenario": config.name,
                       "vehicles": vehicles_data, "background_load_profile": bg_load.tolist()}, f, indent=2)

    st_a = config.layout.stations
    st_b = copy.deepcopy(st_a)
    st_c = copy.deepcopy(st_a)
    st_d = copy.deepcopy(st_a)
    st_e = copy.deepcopy(st_a)

    ctrl_a = UnmanagedController(st_a, policy, bg_load)
    res_a = Simulation(ctrl_a, copy.deepcopy(schedule)).run()

    ctrl_b = ManagedController(st_b, policy, bg_load)
    res_b = Simulation(ctrl_b, copy.deepcopy(schedule)).run()

    ctrl_c = SRPTController(st_c, policy, bg_load)
    res_c = Simulation(ctrl_c, copy.deepcopy(schedule)).run()

    ctrl_d = WaterFillingController(st_d, policy, bg_load)
    res_d = Simulation(ctrl_d, copy.deepcopy(schedule)).run()

    ctrl_e = DynamicFairController(st_e, policy, bg_load)
    res_e = Simulation(ctrl_e, copy.deepcopy(schedule)).run()

    # Orijinal ikili karşılaştırma raporunu koru
    export_comparative_excel(ctrl_a, ctrl_b)

    # Tüm 5 kontrolcünün karşılaştırmalı raporu
    all_ctrls = [
        ("Algoritmasiz", ctrl_a),
        ("Yonetimli",    ctrl_b),
        ("SRPT",         ctrl_c),
        ("Su_Doldurma",  ctrl_d),
        ("Dinamik_Adil", ctrl_e),
    ]
    export_multi_controller_excel(all_ctrls)

    # Konsol özeti
    print(f"\n{'Kontrolcü':<20} {'Araç':>6} {'MaksGüç(kW)':>12} {'AşımDk':>8} {'AşımkWh':>9} {'OrtBkl(dk)':>11} {'OrtŞarj(dk)':>12}")
    print("-" * 82)
    for name, ctrl in all_ctrls:
        p = np.array(ctrl.power_log); l = np.array(ctrl.limit_log); over = p > l
        c = ctrl.completed
        avw = np.mean([e.wait_time_minutes for e in c]) if c else 0.0
        avc = np.mean([e.charge_minutes   for e in c]) if c else 0.0
        print(f"{name:<20} {len(c):>6} {p.max():>12.1f} {int(over.sum()):>8} {float(np.where(over,p-l,0).sum()/60):>9.2f} {avw:>11.1f} {avc:>12.1f}")

    ExecutiveDashboard.create(res_a, res_b, ctrl_label="Yönetimli",    filename="dashboard_yonetimli.png",    bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_c, ctrl_label="SRPT",         filename="dashboard_srpt.png",         bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_d, ctrl_label="Su Doldurma",  filename="dashboard_su_doldurma.png",  bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_e, ctrl_label="Dinamik Adil", filename="dashboard_dinamik_adil.png", bg_load=bg_load)

if __name__ == "__main__":
    _SCENARIO_MAP = {
        "avm_medium":   Scenarios.avm_medium,
        "office_large": Scenarios.office_large,
        "hotel":        Scenarios.hotel,
        "hospital":     Scenarios.hospital,
        "airport":      Scenarios.airport,
    }
    parser = argparse.ArgumentParser(description="EV Yük Dengeleme")
    parser.add_argument("--generate-new", action="store_true")
    parser.add_argument("--scenario", default="avm_medium", choices=list(_SCENARIO_MAP.keys()))
    parser.add_argument("--scenario-file", help="JSON senaryo dosyası")
    args = parser.parse_args()

    if args.scenario_file:
        with open(args.scenario_file, encoding="utf-8") as f:
            cfg = ScenarioConfig.from_dict(json.load(f))
    else:
        cfg = _SCENARIO_MAP[args.scenario]()

    main(generate_new=args.generate_new, config=cfg)