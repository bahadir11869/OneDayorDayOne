#!/usr/bin/env python3
"""
EV Charging Load Balancing Simulation v3
Smart Queue & SoC Priority Algorithm (Optimized) with Turkish Executive Dashboard
"""

from __future__ import annotations
import copy
import json
import argparse
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Set matplotlib to use Turkish locale if available
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']


# ==============================================================================
# Enums
# ==============================================================================

class EVState(Enum):
    WAITING = "waiting"
    CHARGING = "charging"
    DONE = "done"


class StationType(Enum):
    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    STANDARD = "standard"


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass(frozen=True)
class EVModel:
    """Vehicle specification with model name."""
    model_name: str
    battery_capacity_kwh: float
    max_dc_power_kw: float
    probability: float


@dataclass
class GridLimitPolicy:
    """Time-of-use dynamic grid limit configuration."""
    base_limit_kw: float = 400.0
    peak_boost_limit_kw: float = 500.0
    morning_peak_start: int = 420   # 07:00
    morning_peak_end:   int = 600   # 10:00
    evening_peak_start: int = 1020  # 17:00
    evening_peak_end:   int = 1200  # 20:00


@dataclass
class MetricsSummary:
    """Post-simulation metrics."""
    peak_power_kw: float
    overload_minutes: int
    total_overload_kwh: float
    avg_delay_minutes: float
    evs_completed: int
    protected_capacity_percent: float
    avg_grid_limit_kw: float = 400.0
    peak_boost_minutes: int = 0


@dataclass
class VehicleSession:
    """Per-vehicle session details."""
    session_id: str
    model_name: str
    arrival_minute: int
    initial_soc: float
    final_soc: float
    wait_time_minutes: int
    charge_time_minutes: int
    energy_delivered_kwh: float
    avg_power_kw: float


@dataclass
class SimulationResult:
    """Complete simulation outcome."""
    scenario_name: str
    power_timeseries: np.ndarray
    metrics_summary: MetricsSummary
    vehicle_sessions: List[VehicleSession]
    grid_limit_timeseries: np.ndarray = field(default_factory=lambda: np.array([]))


# ==============================================================================
# Stateful Entities
# ==============================================================================

@dataclass
class EV:
    """Active EV charging session."""
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
        needed = (self.target_soc - self.current_soc) * self.battery_capacity_kwh
        return max(0.0, needed)

    @property
    def is_satisfied(self) -> bool:
        return self.current_soc >= self.target_soc

    @property
    def soc_category(self) -> str:
        if self.current_soc >= 0.70:
            return "high_soc"
        else:
            return "low_soc"

    def apply_power(self, power_kw: float, duration_minutes: float) -> float:
        """Apply power, return energy absorbed."""
        if self.current_soc >= self.target_soc:
            return 0.0

        energy_available = power_kw * (duration_minutes / 60.0)
        energy_to_absorb = min(energy_available, self.energy_needed_kwh)

        self.current_soc += energy_to_absorb / self.battery_capacity_kwh
        self.current_soc = min(self.current_soc, self.target_soc)
        self.energy_delivered_kwh += energy_to_absorb

        if power_kw > 0.0:
            self.charge_minutes += 1

        return energy_to_absorb

    @property
    def wait_time_minutes(self) -> int:
        if self.charge_start_minute is None:
            return 0
        return self.charge_start_minute - self.arrival_minute


@dataclass
class ChargingStation:
    """Physical charging point."""
    station_id: str
    station_type: StationType
    max_power_kw: float
    current_ev: Optional[EV] = None
    allocated_power_kw: float = 0.0

    def is_available(self) -> bool:
        return self.current_ev is None

    def effective_max_power_kw(self) -> float:
        if self.current_ev is None:
            return 0.0
        return min(self.max_power_kw, self.current_ev.max_dc_power_kw)

    def plug_in(self, ev: EV) -> None:
        if not self.is_available():
            raise RuntimeError(f"Station {self.station_id} occupied")
        self.current_ev = ev
        ev.state = EVState.CHARGING

    def unplug(self) -> Optional[EV]:
        ev = self.current_ev
        self.current_ev = None
        self.allocated_power_kw = 0.0
        return ev

    def step(self, duration_minutes: float) -> float:
        if self.current_ev is None:
            return 0.0
        energy = self.current_ev.apply_power(self.allocated_power_kw, duration_minutes)
        if self.current_ev.is_satisfied:
            self.current_ev.state = EVState.DONE
        return energy


class WaitingQueue:
    """FIFO queue for waiting EVs."""
    def __init__(self):
        self.queue: deque = deque()

    def enqueue(self, ev: EV) -> None:
        self.queue.append(ev)

    def dequeue(self) -> Optional[EV]:
        return self.queue.popleft() if self.queue else None

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)


# ==============================================================================
# Grid Control
# ==============================================================================

class GridController(ABC):
    """Abstract edge controller for power allocation."""

    def __init__(self, stations: List[ChargingStation], grid_limit_kw: float,
                 limit_policy: Optional['GridLimitPolicy'] = None,
                 background_load: Optional[np.ndarray] = None,
                 vehicle_time_budgets: Optional[Dict[str, float]] = None):
        self.stations = stations
        self.queue = WaitingQueue()
        self.grid_limit_kw = grid_limit_kw
        self.limit_policy = limit_policy or GridLimitPolicy(base_limit_kw=grid_limit_kw)
        self.background_load = background_load if background_load is not None else np.zeros(1440)
        self.vehicle_time_budgets: Dict[str, float] = vehicle_time_budgets or {}
        self.power_log: List[float] = []
        self.grid_limit_log: List[float] = []
        self.completed_sessions: List[EV] = []
        self.queued_count: int = 0

    def get_baseline_at_minute(self, minute: int) -> float:
        """Non-EV background load from the pre-generated daily profile."""
        return float(self.background_load[int(np.clip(minute, 0, 1439))])

    def get_grid_limit_at_minute(self, minute: int) -> float:
        """TOU dynamic limit: boost during peak arrival windows."""
        p = self.limit_policy
        tod = minute % 1440
        if p.morning_peak_start <= tod < p.morning_peak_end:
            return p.peak_boost_limit_kw
        if p.evening_peak_start <= tod < p.evening_peak_end:
            return p.peak_boost_limit_kw
        return p.base_limit_kw

    @abstractmethod
    def allocate_power(self, baseline_kw: float, minute: int, grid_limit_kw: float) -> Dict[str, float]:
        """Return allocation {station_id: kw}."""
        pass

    def dispatch_waiting_evs(self, minute: int) -> None:
        """Assign queued EVs to free stations using SPT (Shortest Processing Time)."""
        for station in self.stations:
            if station.is_available() and not self.queue.is_empty():
                ev = self.queue.dequeue()
                station.plug_in(ev)
                ev.charge_start_minute = minute

    def collect_finished_evs(self) -> List[EV]:
        """Remove satisfied EVs."""
        finished = []
        for station in self.stations:
            if station.current_ev and station.current_ev.state == EVState.DONE:
                ev = station.unplug()
                finished.append(ev)
        return finished

    def step(self, minute: int) -> None:
        """Execute one minute."""
        self.dispatch_waiting_evs(minute)

        baseline_now = self.get_baseline_at_minute(minute)
        current_limit = self.get_grid_limit_at_minute(minute)
        allocations = self.allocate_power(baseline_now, minute, current_limit)

        for station in self.stations:
            station.allocated_power_kw = allocations.get(station.station_id, 0.0)
            station.step(duration_minutes=1.0)

        finished_evs = self.collect_finished_evs()
        for ev in finished_evs:
            ev.departure_minute = minute
            self.completed_sessions.append(ev)

        total_power = baseline_now + sum(allocations.values())
        self.power_log.append(total_power)
        self.grid_limit_log.append(current_limit)


class UnmanagedController(GridController):
    """Uncontrolled: no budgeting."""

    def allocate_power(self, baseline_kw: float, minute: int, grid_limit_kw: float) -> Dict[str, float]:
        allocations = {}
        for station in self.stations:
            allocations[station.station_id] = station.effective_max_power_kw()
        return allocations


class ManagedController(GridController):
    """Controlled: Dynamic Priority Allocation with HRRN & Greedy Waterfall."""

    EXTENSION_FACTOR = 1.2  # Legacy for compatibility if needed elsewhere

    def _dispatch_score(self, ev: EV, station: 'ChargingStation', minute: int) -> float:
        """
        HRRN (Highest Response Ratio Next) Mantığı:
        Dışarıda bekleyen isyan etmesin diye bekleme süresi skoru artırır.
        """
        eff_power = min(ev.max_dc_power_kw, station.max_power_kw)
        if eff_power <= 0:
            return -9999.0
            
        min_charge_min = (ev.energy_needed_kwh / eff_power) * 60.0
        wait_time = minute - ev.arrival_minute
        
        # 1 dakika bekleme, işin 2 dakika kısaymış gibi öncelik verir
        return (wait_time * 2.0) - min_charge_min

    def dispatch_waiting_evs(self, minute: int) -> None:
        """Dispatch based on the new _dispatch_score (HRRN)."""
        for station in self.stations:
            if not station.is_available() or self.queue.is_empty():
                continue
            best_ev = max(
                self.queue.queue,
                key=lambda ev: self._dispatch_score(ev, station, minute),
            )
            self.queue.queue.remove(best_ev)
            station.plug_in(best_ev)
            best_ev.charge_start_minute = minute

    def allocate_power(self, baseline_kw: float, minute: int, grid_limit_kw: float) -> Dict[str, float]:
        occupied_stations = [s for s in self.stations if not s.is_available()]
        allocations = {s.station_id: 0.0 for s in self.stations}

        if not occupied_stations:
            return allocations

        available_budget = max(0.0, grid_limit_kw - baseline_kw - 0.01)

        vehicles = []
        for station in occupied_stations:
            ev = station.current_ev
            if ev.is_satisfied:
                continue
            max_power = station.effective_max_power_kw()
            if ev.energy_needed_kwh > 0.0 and max_power > 0.0:
                time_to_finish = ev.energy_needed_kwh / max_power
                vehicles.append({
                    'station_id': station.station_id,
                    'max_power': max_power,
                    'time_to_finish': time_to_finish
                })

        if not vehicles:
            return allocations

        # ADIM 1: Herkese Minimum "Can Suyu" (22 kW) - Soket Kilitlenmesini Önler
        MIN_POWER = 22.0
        for v in vehicles:
            give = min(MIN_POWER, v['max_power'], available_budget)
            allocations[v['station_id']] = give
            available_budget -= give

        # ADIM 2: Kalan bütçeyi işi en çabuk bitecek olanlara bas (Şelale / Greedy)
        vehicles.sort(key=lambda x: x['time_to_finish'])

        for v in vehicles:
            if available_budget <= 0.1:
                break
            
            headroom = v['max_power'] - allocations[v['station_id']]
            if headroom > 0:
                give = min(headroom, available_budget)
                allocations[v['station_id']] += give
                available_budget -= give

        return allocations


# ==============================================================================
# Simulation
# ==============================================================================

class ArrivalGenerator:
    """Generate EV arrival schedule."""

    EV_MODELS = [
        EVModel("Togg T10X", 88.5, 150.0, 0.25),
        EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
        EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
        EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
        EVModel("MG4 Standard", 51.0, 117.0, 0.10),
        EVModel("Renault Megane", 60.0, 130.0, 0.10),
        EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
    ]

    def __init__(self, daily_ev_count: int = 50):
        self.daily_ev_count = daily_ev_count

    def generate_arrivals(self, rng: np.random.Generator) -> Dict[int, List[EV]]:
        """Generate 24-hour arrival schedule."""
        arrival_minutes = self._trimodal_arrival_minutes(rng, self.daily_ev_count)

        schedule: Dict[int, List[EV]] = {}
        for i, minute in enumerate(arrival_minutes):
            minute_idx = int(np.clip(minute, 0, 1439))
            if minute_idx not in schedule:
                schedule[minute_idx] = []

            model = self._sample_ev_model(rng)
            initial_soc = rng.uniform(0.10, 0.50)

            ev = EV(
                session_id=f"EV_{i:02d}",
                model_name=model.model_name,
                battery_capacity_kwh=model.battery_capacity_kwh,
                max_dc_power_kw=model.max_dc_power_kw,
                arrival_minute=minute_idx,
                initial_soc=initial_soc,
            )
            schedule[minute_idx].append(ev)

        return schedule

    def _trimodal_arrival_minutes(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """Trimodal distribution for urban Turkey: morning (09:30), lunch (13:00), evening (18:30)."""
        n_morning = count // 5          # ~20% — esnaf / kuaför / sabah üst yük
        n_lunch   = count // 4          # ~25% — öğle arası hızlı şarj
        n_evening = count - n_morning - n_lunch  # ~55% — iş sonrası / AVM

        morning = rng.normal(570, 40, n_morning)
        lunch   = rng.normal(780, 35, n_lunch)
        evening = rng.normal(1110, 75, n_evening)

        arrivals = np.concatenate([morning, lunch, evening])
        rng.shuffle(arrivals)
        return np.clip(arrivals, 0, 1439)

    def _sample_ev_model(self, rng: np.random.Generator) -> EVModel:
        weights = [m.probability for m in self.EV_MODELS]
        idx = rng.choice(len(self.EV_MODELS), p=weights)
        return self.EV_MODELS[idx]


class BackgroundLoadGenerator:
    """1440-dakikalık şebeke arka plan yük profili üretir."""

    @staticmethod
    def generate(rng: np.random.Generator) -> np.ndarray:
        hours = np.arange(1440) / 60.0

        # Düzgün gündüz-gece eğrisi
        base = 45.0 + 90.0 * np.clip(
            0.5 * (1 - np.cos(np.pi * np.clip(hours - 6, 0, 14) / 14)), 0, 1
        )

        # Sabah ofis tepe ~09:00
        morning_peak = 70.0 * np.exp(-0.5 * ((hours - 9.0) / 1.0) ** 2)

        # Akşam konut tepe ~19:00
        evening_peak = 100.0 * np.exp(-0.5 * ((hours - 19.0) / 1.2) ** 2)

        noise = rng.normal(0.0, 15.0, 1440)

        profile = base + morning_peak + evening_peak + noise
        return np.clip(profile, 25.0, 280.0)


class Simulation:
    """Run simulation with controller."""

    def __init__(self, controller: GridController, arrival_schedule: Dict[int, List[EV]]):
        self.controller = controller
        self.arrival_schedule = arrival_schedule

    def run(self) -> SimulationResult:
        """Execute 24-hour simulation."""
        for minute in range(1440):
            for ev in self.arrival_schedule.get(minute, []):
                self.controller.queue.enqueue(ev)

            self.controller.step(minute)


        # Build result
        power_arr = np.array(self.controller.power_log)
        limit_arr = np.array(self.controller.grid_limit_log)

        overload_mask = power_arr > limit_arr
        overload_excess = np.where(overload_mask, power_arr - limit_arr, 0.0)

        metrics = MetricsSummary(
            peak_power_kw=float(power_arr.max()),
            overload_minutes=int(overload_mask.sum()),
            total_overload_kwh=float(overload_excess.sum() / 60.0),
            avg_delay_minutes=float(np.mean([ev.wait_time_minutes for ev in self.controller.completed_sessions]))
                if self.controller.completed_sessions else 0.0,
            evs_completed=len(self.controller.completed_sessions),
            protected_capacity_percent=100.0 if not overload_mask.any() else
                float((limit_arr / np.maximum(power_arr, 1e-9)).clip(max=1.0).mean() * 100.0),
            avg_grid_limit_kw=float(limit_arr.mean()),
            peak_boost_minutes=int((limit_arr > self.controller.limit_policy.base_limit_kw).sum()),
        )

        vehicle_sessions = [
            VehicleSession(
                session_id=ev.session_id,
                model_name=ev.model_name,
                arrival_minute=ev.arrival_minute,
                initial_soc=ev.initial_soc,
                final_soc=ev.current_soc,
                wait_time_minutes=ev.wait_time_minutes,
                charge_time_minutes=ev.charge_minutes,
                energy_delivered_kwh=ev.energy_delivered_kwh,
                avg_power_kw=ev.energy_delivered_kwh / ev.charge_minutes * 60.0 if ev.charge_minutes > 0 else 0.0,
            )
            for ev in self.controller.completed_sessions
        ]

        return SimulationResult(
            scenario_name=self.controller.__class__.__name__,
            power_timeseries=power_arr,
            metrics_summary=metrics,
            vehicle_sessions=vehicle_sessions,
            grid_limit_timeseries=limit_arr,
        )


# ==============================================================================
# Visualization (Turkish)
# ==============================================================================

class ExecutiveDashboard:
    """Executive-level dashboard with Turkish labels."""

    @staticmethod
    def create_dashboard(result_unmanaged: SimulationResult, result_managed: SimulationResult) -> None:
        """Create 7-panel executive dashboard in Turkish."""
        fig = plt.figure(figsize=(16, 22))
        fig.suptitle("EV Şarj Yük Dengeleme — Yönetici Özeti", fontsize=18, fontweight="bold", y=0.995)

        gs = GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[3, :])   # Panel 7 tam genişlik

        unmanaged_sessions = result_unmanaged.vehicle_sessions
        managed_sessions   = result_managed.vehicle_sessions
        hours = np.arange(1440) / 60

        # ------------------------------------------------------------------
        # Panel 1: Grid Load
        # ------------------------------------------------------------------
        ax1.fill_between(hours, 0, result_unmanaged.power_timeseries,
                         alpha=0.3, color="red", label="Kontrol Öncesi")
        ax1.plot(hours, result_managed.power_timeseries,
                 linewidth=2.5, color="darkgreen", label="Kontrol Sonrası")
        limit_series = result_managed.grid_limit_timeseries
        if len(limit_series) == 1440:
            ax1.plot(hours, limit_series, color="red", linestyle="--", linewidth=2.0,
                     label=f"Dinamik Limit ({int(limit_series.min())}-{int(limit_series.max())} kW)")
            for sh, eh in [(7, 10), (17, 20)]:
                ax1.axvspan(sh, eh, alpha=0.08, color="orange",
                            label="TOU Pik Penceresi" if sh == 7 else "")
        else:
            ax1.axhline(400, color="red", linestyle="--", linewidth=2.0, label="Şebeke Limiti (400 kW)")
        ax1.set_xlabel("Gün Saati (saat)", fontsize=10)
        ax1.set_ylabel("Şebeke Yükü (kW)", fontsize=10)
        ax1.set_title("Panel 1: 24 Saatlik Şebeke Yükü Profili", fontsize=11, fontweight="bold")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 2))
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # Panel 2: Wait Time Categories
        # ------------------------------------------------------------------
        no_wait    = sum(1 for s in managed_sessions if s.wait_time_minutes == 0)
        short_wait = sum(1 for s in managed_sessions if 0 < s.wait_time_minutes <= 15)
        long_wait  = sum(1 for s in managed_sessions if s.wait_time_minutes > 15)

        categories = ["Bekleme Yok\n(0 dk)", "Kabul Edilebilir\n(1-15 dk)", "Uzun Bekleme\n(15+ dk)"]
        values     = [no_wait, short_wait, long_wait]
        colors_cat = ["#2ecc71", "#f39c12", "#e74c3c"]

        bars = ax2.bar(categories, values, color=colors_cat, edgecolor="black", linewidth=1.5, width=0.6)
        ax2.set_ylabel("Araç Sayısı", fontsize=10)
        ax2.set_title("Panel 2: Kuyruk Bekleme Dağılımı (Kontrol Sonrası)", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{int(val)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # ------------------------------------------------------------------
        # Panel 3: Charge Time by Model
        # ------------------------------------------------------------------
        model_charge_um: Dict[str, list] = {}
        model_charge_mg: Dict[str, list] = {}
        for s in unmanaged_sessions:
            model_charge_um.setdefault(s.model_name, []).append(s.charge_time_minutes)
        for s in managed_sessions:
            model_charge_mg.setdefault(s.model_name, []).append(s.charge_time_minutes)

        models3 = sorted(set(list(model_charge_um) + list(model_charge_mg)))
        avg_um3  = [np.mean(model_charge_um.get(m, [0])) for m in models3]
        avg_mg3  = [np.mean(model_charge_mg.get(m, [0])) for m in models3]
        labels3  = [f"{m}\n(n={len(model_charge_um.get(m,[]))})" for m in models3]
        x3 = np.arange(len(models3)); w3 = 0.35

        b3a = ax3.bar(x3 - w3/2, avg_um3, w3, label="Kontrol Öncesi", color="lightcoral", edgecolor="black")
        b3b = ax3.bar(x3 + w3/2, avg_mg3, w3, label="Kontrol Sonrası", color="lightgreen", edgecolor="black")
        ax3.set_ylabel("Ort. Şarj Süresi (dk)", fontsize=10)
        ax3.set_title("Panel 3: Şarj Süresi Kıyası", fontsize=11, fontweight="bold")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(labels3, fontsize=8, rotation=30, ha="right")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")
        for b in [b3a, b3b]:
            for bar in b:
                h = bar.get_height()
                if h > 0.5:
                    ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                             f"{h:.0f}", ha="center", va="bottom", fontsize=7)

        # ------------------------------------------------------------------
        # Panel 4: Wait Time by Model
        # ------------------------------------------------------------------
        model_wait_um: Dict[str, list] = {}
        model_wait_mg: Dict[str, list] = {}
        for s in unmanaged_sessions:
            model_wait_um.setdefault(s.model_name, []).append(s.wait_time_minutes)
        for s in managed_sessions:
            model_wait_mg.setdefault(s.model_name, []).append(s.wait_time_minutes)

        models4    = sorted(set(list(model_wait_um) + list(model_wait_mg)))
        avg_wait_um = [np.mean(model_wait_um.get(m, [0])) for m in models4]
        avg_wait_mg = [np.mean(model_wait_mg.get(m, [0])) for m in models4]
        labels4    = [f"{m.split()[0]}\n(n={len(model_wait_um.get(m,[]))})" for m in models4]
        x4 = np.arange(len(models4)); w4 = 0.35

        b4a = ax4.bar(x4 - w4/2, avg_wait_um, w4, label="Kontrol Öncesi",
                      color="#f9c784", edgecolor="black")
        b4b = ax4.bar(x4 + w4/2, avg_wait_mg, w4, label="Kontrol Sonrası",
                      color="#74b9ff", edgecolor="black")
        ax4.set_ylabel("Ort. Bekleme Süresi (dk)", fontsize=10)
        ax4.set_title("Panel 4: Bekleme Süresi Kıyası", fontsize=11, fontweight="bold")
        ax4.set_xticks(x4)
        ax4.set_xticklabels(labels4, fontsize=9)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis="y")
        y4_max = max(max(avg_wait_um, default=0), max(avg_wait_mg, default=0))
        ax4.set_ylim(0, max(y4_max * 1.35, 1.0))
        offset4 = max(y4_max * 0.04, 0.08)
        for b in [b4a, b4b]:
            for bar in b:
                h = bar.get_height()
                if h >= 0.05:
                    ax4.text(bar.get_x() + bar.get_width() / 2, h + offset4,
                             f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # ------------------------------------------------------------------
        # Panel 5: Metrics Summary (Text)
        # ------------------------------------------------------------------
        ax5.axis("off")
        peak_red_pct = ((result_unmanaged.metrics_summary.peak_power_kw -
                         result_managed.metrics_summary.peak_power_kw) /
                        result_unmanaged.metrics_summary.peak_power_kw * 100)
        overload_red_pct = ((result_unmanaged.metrics_summary.overload_minutes -
                             result_managed.metrics_summary.overload_minutes) /
                            max(result_unmanaged.metrics_summary.overload_minutes, 1) * 100)
        kwh_diff     = result_unmanaged.metrics_summary.total_overload_kwh - result_managed.metrics_summary.total_overload_kwh
        delay_inc    = result_managed.metrics_summary.avg_delay_minutes - result_unmanaged.metrics_summary.avg_delay_minutes

        summary_text = (
            f"ŞEBEKE KORUMASI & PERFORMANS\n\n"
            f"Korunan Kapasite : {result_managed.metrics_summary.protected_capacity_percent:.1f}%\n"
            f"Pik Güç          : {result_unmanaged.metrics_summary.peak_power_kw:.0f} → "
            f"{result_managed.metrics_summary.peak_power_kw:.0f} kW  (↓{peak_red_pct:.1f}%)\n"
            f"Aşım Süresi      : {result_unmanaged.metrics_summary.overload_minutes} → "
            f"{result_managed.metrics_summary.overload_minutes} dk  (↓{overload_red_pct:.1f}%)\n"
            f"Aşım Enerjisi    : ↓ {kwh_diff:.1f} kWh\n\n"
            f"Servis Verilen   : {result_managed.metrics_summary.evs_completed} araç\n"
            f"Ort. Bekleme     : {result_unmanaged.metrics_summary.avg_delay_minutes:.1f} → "
            f"{result_managed.metrics_summary.avg_delay_minutes:.1f} dk  (+{delay_inc:.1f} dk)\n\n"
            f"Algoritma        : Dinamik Priority (HRRN) + TOU\n"
            f"Baz Limit        : {result_managed.metrics_summary.avg_grid_limit_kw:.0f} kW (ort.)\n"
            f"Pik Boost        : {result_managed.grid_limit_timeseries.max():.0f} kW\n"
            f"Boost Süresi     : {result_managed.metrics_summary.peak_boost_minutes} dk"
        )
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                 fontsize=9.5, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4))

        # ------------------------------------------------------------------
        # Panel 6: Total System Time
        # ------------------------------------------------------------------
        model_sys_um: Dict[str, list] = {}
        model_sys_mg: Dict[str, list] = {}
        for s in unmanaged_sessions:
            model_sys_um.setdefault(s.model_name.split()[0], []).append(
                s.wait_time_minutes + s.charge_time_minutes)
        for s in managed_sessions:
            model_sys_mg.setdefault(s.model_name.split()[0], []).append(
                s.wait_time_minutes + s.charge_time_minutes)

        models6 = sorted(set(list(model_sys_um) + list(model_sys_mg)))
        avg6_um = [np.mean(model_sys_um.get(m, [0])) for m in models6]
        avg6_mg = [np.mean(model_sys_mg.get(m, [0])) for m in models6]
        labels6 = [f"{m}\n(n={len(model_sys_mg.get(m,[]))})" for m in models6]
        x6 = np.arange(len(models6)); w6 = 0.35

        b6a = ax6.bar(x6 - w6/2, avg6_um, w6, label="Kontrol Öncesi", color="lightcoral", edgecolor="black")
        b6b = ax6.bar(x6 + w6/2, avg6_mg, w6, label="Kontrol Sonrası", color="lightgreen", edgecolor="black")
        ax6.set_ylabel("Ort. Sistem Süresi (dk)", fontsize=10)
        ax6.set_title("Panel 6: Toplam Sistemde Kalma (Bekleme + Şarj)", fontsize=11, fontweight="bold")
        ax6.set_xticks(x6)
        ax6.set_xticklabels(labels6, fontsize=9)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis="y")
        for b in [b6a, b6b]:
            for bar in b:
                h = bar.get_height()
                if h > 0.5:
                    ax6.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                             f"{h:.0f}", ha="center", va="bottom", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 7: Şarj Süresi Uzatım Aralığı (Min / Ort. / Maks)
        # ------------------------------------------------------------------
        model_ext: Dict[str, list] = {}
        for s_um, s_mg in zip(
            sorted(unmanaged_sessions, key=lambda s: s.session_id),
            sorted(managed_sessions,   key=lambda s: s.session_id),
        ):
            key = s_mg.model_name.split()[0]
            model_ext.setdefault(key, []).append(s_mg.charge_time_minutes - s_um.charge_time_minutes)

        ext_models = sorted(model_ext.keys())
        ext_min  = [min(model_ext[m])  for m in ext_models]
        ext_avg  = [np.mean(model_ext[m]) for m in ext_models]
        ext_max  = [max(model_ext[m])  for m in ext_models]
        ext_labels = [
            f"{m}\n(n={len(model_ext[m])})" for m in ext_models
        ]

        x7 = np.arange(len(ext_models))
        # Arka plan: min'den max'a uzanan gri bant
        for i, (mn, mx) in enumerate(zip(ext_min, ext_max)):
            ax7.bar(i, mx - mn, bottom=mn, width=0.45,
                    color="steelblue", alpha=0.25, edgecolor="none")
        # Ortalama çizgisi
        ax7.plot(x7, ext_avg, "o-", color="navy", linewidth=2, markersize=7,
                 label="Ortalama Artış", zorder=3)
        # Min / Max işaret
        ax7.scatter(x7, ext_min, marker="^", color="green",  s=60, zorder=4, label="En Az Artış")
        ax7.scatter(x7, ext_max, marker="v", color="crimson", s=60, zorder=4, label="En Çok Artış")

        # Değer etiketleri
        for i, (mn, av, mx) in enumerate(zip(ext_min, ext_avg, ext_max)):
            ax7.text(i - 0.25, mn - 0.6,  f"{mn:+.0f}", fontsize=8, color="green",   ha="center")
            ax7.text(i,        av + 0.5,   f"{av:+.1f}", fontsize=8, color="navy",    ha="center", fontweight="bold")
            ax7.text(i + 0.25, mx + 0.5,   f"{mx:+.0f}", fontsize=8, color="crimson", ha="center")

        ax7.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax7.set_xticks(x7)
        ax7.set_xticklabels(ext_labels, fontsize=10)
        ax7.set_ylabel("Şarj Süresi Değişimi (dakika)", fontsize=11)
        ax7.set_title(
            "Panel 7: Algoritmanın Model Başına Şarj Süresi Uzatım Aralığı "
            "(yeşil▲ = en az, mavi● = ortalama, kırmızı▼ = en çok)",
            fontsize=11, fontweight="bold",
        )
        ax7.legend(fontsize=9, loc="upper right")
        ax7.grid(True, alpha=0.3, axis="y")

        plt.savefig("executive_dashboard_tr.png", dpi=150, bbox_inches="tight")
        print("✓ Türkçe yönetici dashboard kaydedildi: executive_dashboard_tr.png")
        plt.show()

    @staticmethod
    def print_executive_summary(result_unmanaged: SimulationResult, result_managed: SimulationResult) -> None:
        """Print executive summary to console in Turkish."""
        print("\n" + "="*80)
        print("YÖNETİCİ ÖZETİ — EV YÜK DENGELEMESİ OPTİMİZASYONU")
        print("="*80)

        print(f"\n📊 ŞEBEKE KORUMASI METRİKLERİ")
        print(f"  • Korunan Kapasite:              {result_managed.metrics_summary.protected_capacity_percent:>6.1f}% (Hedef: 100%)")
        print(f"  • Önlenen Pik Aşım:              {result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw:>6.0f} kW azalma")
        print(f"  • Aşım Zaman Eleme:              {result_unmanaged.metrics_summary.overload_minutes:>6d} → {result_managed.metrics_summary.overload_minutes:>3d} dakika")

        print(f"\n🚗 HİZMET SUNUMU")
        print(f"  • Toplam Servis Verilen:         {result_managed.metrics_summary.evs_completed:>6d} araç")
        print(f"  • Algoritma Kaynaklı Gecikme:    {result_managed.metrics_summary.avg_delay_minutes:>6.1f} dk (kabul edilebilir trade-off)")

        print(f"\n⚡ ALGORİTMA VERİMLİLİĞİ")
        peak_reduction = ((result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw)
                         / result_unmanaged.metrics_summary.peak_power_kw * 100)
        print(f"  • Pik Yük Azaltma:               {peak_reduction:>6.1f}%")
        print(f"  • Aşım Enerjisi Azalma:          {result_unmanaged.metrics_summary.total_overload_kwh - result_managed.metrics_summary.total_overload_kwh:>6.1f} kWh")

        # Model-specific comparison
        print(f"\n📈 MODEL BAŞINA PERFORMANS KARŞILAŞTIRMASI (KONTROL ÖNCESİ vs SONRASI)")

        model_sessions_um = {}
        model_sessions_mg = {}

        for session in result_unmanaged.vehicle_sessions:
            model = session.model_name.split()[0]
            if model not in model_sessions_um:
                model_sessions_um[model] = []
            model_sessions_um[model].append(session)

        for session in result_managed.vehicle_sessions:
            model = session.model_name.split()[0]
            if model not in model_sessions_mg:
                model_sessions_mg[model] = []
            model_sessions_mg[model].append(session)

        all_models = sorted(set(list(model_sessions_um.keys()) + list(model_sessions_mg.keys())))

        print(f"\n{'Model':<10} {'Kontrol Öncesi':<50} {'Kontrol Sonrası':<50} {'Gelişim':<15}")
        print(f"{'':10} {'Bekleme(dk)':<12} {'Şarj(dk)':<12} {'Max(dk)':<12} {'Bekleme(dk)':<12} {'Şarj(dk)':<12} {'Max(dk)':<12} {'Şarj %Δ':<10}")
        print("-" * 152)

        for model in all_models:
            um_sessions = model_sessions_um.get(model, [])
            mg_sessions = model_sessions_mg.get(model, [])

            if um_sessions:
                um_wait = np.mean([s.wait_time_minutes for s in um_sessions])
                um_charge = np.mean([s.charge_time_minutes for s in um_sessions])
                um_charge_max = max([s.charge_time_minutes for s in um_sessions])
            else:
                um_wait = um_charge = um_charge_max = 0

            if mg_sessions:
                mg_wait = np.mean([s.wait_time_minutes for s in mg_sessions])
                mg_charge = np.mean([s.charge_time_minutes for s in mg_sessions])
                mg_charge_max = max([s.charge_time_minutes for s in mg_sessions])
            else:
                mg_wait = mg_charge = mg_charge_max = 0

            # Calculate average energy delivered to verify physics
            um_energy = [s.energy_delivered_kwh for s in um_sessions] if um_sessions else []
            mg_energy = [s.energy_delivered_kwh for s in mg_sessions] if mg_sessions else []

            charge_improvement = ((um_charge - mg_charge) / um_charge * 100) if um_charge > 0 else 0

            print(f"{model:<10} {um_wait:>11.1f} {um_charge:>11.1f} {um_charge_max:>11.1f} {mg_wait:>11.1f} {mg_charge:>11.1f} {mg_charge_max:>11.1f} {charge_improvement:>9.1f}%")

            # Debug: Check energy parity
            if um_energy and mg_energy and len(um_energy) == len(mg_energy):
                avg_um_energy = np.mean(um_energy)
                avg_mg_energy = np.mean(mg_energy)
                if abs(avg_um_energy - avg_mg_energy) > 0.1:
                    print(f"  ⚠️  ENERJI FARKLI! Unmanaged: {avg_um_energy:.1f} kWh, Managed: {avg_mg_energy:.1f} kWh")

        print("\n" + "="*80)
        print("⚠️  HER MODEL İÇİN ŞARJ SÜRESİ DEĞİŞİM ARALIKLARI")
        print("="*80)

        for model in all_models:
            um_sessions = model_sessions_um.get(model, [])
            mg_sessions = model_sessions_mg.get(model, [])

            if um_sessions and mg_sessions:
                um_charges = [s.charge_time_minutes for s in um_sessions]
                mg_charges = [s.charge_time_minutes for s in mg_sessions]

                # Calculate per-vehicle increases
                charge_increases = []
                for i, mg_charge in enumerate(mg_charges):
                    if i < len(um_charges):
                        um_charge = um_charges[i]
                        increase = mg_charge - um_charge
                        charge_increases.append(increase)

                if charge_increases:
                    min_increase = min(charge_increases)
                    max_increase = max(charge_increases)
                    avg_increase = np.mean(charge_increases)
                    count = len(mg_sessions)

                    print(f"\n{model}: ({count} araç)")
                    print(f"  En az artış: +{min_increase:.1f} min (en iyi senaryo)")
                    print(f"  En çok artış: +{max_increase:.1f} min (en kötü senaryo)")
                    print(f"  Ortalama artış: +{avg_increase:.1f} min")

        print("\n" + "="*80 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main(generate_new: bool = False):
    """Run full simulation."""
    print("EV Yük Dengeleme Simülasyonu v3 başlatılıyor...")

    # Dataset management
    dataset_file = "dataset.json" # Dizin yolunu kendi ortamına göre ayarlarsın
    rng_bg = np.random.default_rng(seed=99)  # separate RNG for background load

    if not generate_new and __import__('os').path.exists(dataset_file):
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Veri seti yüklendi: {len(data['vehicles'])} araç")
        arrival_schedule_dict = {}
        for v in data['vehicles']:
            minute = v['arrival_minute']
            if minute not in arrival_schedule_dict:
                arrival_schedule_dict[minute] = []
            arrival_schedule_dict[minute].append(
                EV(
                    session_id=v['session_id'],
                    model_name=v['model_name'],
                    battery_capacity_kwh=v['battery_capacity_kwh'],
                    max_dc_power_kw=v['max_dc_power_kw'],
                    arrival_minute=minute,
                    initial_soc=v['initial_soc'],
                )
            )
        arrival_schedule = arrival_schedule_dict

        if 'background_load_profile' in data:
            background_load = np.array(data['background_load_profile'])
            print(f"✓ Arka plan yük profili yüklendi (ort. {background_load.mean():.0f} kW, maks. {background_load.max():.0f} kW)")
        else:
            background_load = BackgroundLoadGenerator.generate(rng_bg)
            data['background_load_profile'] = background_load.tolist()
            with open(dataset_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Arka plan yük profili oluşturuldu ve kaydedildi")
    else:
        print("✓ Yeni veri seti oluşturuluyor...")
        rng = np.random.default_rng(seed=42)
        gen = ArrivalGenerator(50)
        arrival_schedule = gen.generate_arrivals(rng)
        background_load = BackgroundLoadGenerator.generate(rng_bg)

        # Save dataset
        vehicles_data = []
        for minute in sorted(arrival_schedule.keys()):
            for ev in arrival_schedule[minute]:
                vehicles_data.append({
                    "session_id": ev.session_id,
                    "model_name": ev.model_name,
                    "battery_capacity_kwh": ev.battery_capacity_kwh,
                    "max_dc_power_kw": ev.max_dc_power_kw,
                    "arrival_minute": ev.arrival_minute,
                    "initial_soc": ev.initial_soc,
                })
        with open(dataset_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "vehicles": vehicles_data,
                "background_load_profile": background_load.tolist(),
            }, f, indent=2)
        print(f"✓ Veri seti kaydedildi: {len(vehicles_data)} araç, arka plan yük profili dahil")

    stations_a = [
        ChargingStation("S1", StationType.ULTRA_FAST, 200.0),
        ChargingStation("S2", StationType.ULTRA_FAST, 200.0),
        ChargingStation("S3", StationType.FAST,       180.0),
        ChargingStation("S4", StationType.FAST,       180.0),
        ChargingStation("S5", StationType.STANDARD,   120.0),
    ]
    stations_b = [copy.deepcopy(s) for s in stations_a]

    print("\n⚙️  Senaryo A (Kontrol Öncesi) çalıştırılıyor...")
    schedule_a = copy.deepcopy(arrival_schedule)
    schedule_b = copy.deepcopy(arrival_schedule)

    for minute_evs_a in schedule_a.values():
        for ev in minute_evs_a:
            ev.current_soc = ev.initial_soc
            ev.state = EVState.WAITING
            ev.charge_minutes = 0
            ev.charge_start_minute = None
            ev.departure_minute = None
            ev.energy_delivered_kwh = 0.0

    for minute_evs_b in schedule_b.values():
        for ev in minute_evs_b:
            ev.current_soc = ev.initial_soc
            ev.state = EVState.WAITING
            ev.charge_minutes = 0
            ev.charge_start_minute = None
            ev.departure_minute = None
            ev.energy_delivered_kwh = 0.0

    controller_a = UnmanagedController(stations_a, 400.0, background_load=background_load)
    result_a = Simulation(controller_a, schedule_a).run()
    print(f"   ✓ Tamamlandı: Pik={result_a.metrics_summary.peak_power_kw:.0f}kW, Aşım={result_a.metrics_summary.overload_minutes}dk")

    vehicle_time_budgets = {
        s.session_id: (s.wait_time_minutes + s.charge_time_minutes) * ManagedController.EXTENSION_FACTOR
        for s in result_a.vehicle_sessions
    }

    print("⚙️  Senaryo B (Kontrol Sonrası — TOU Dinamik Limit) çalıştırılıyor...")
    tou_policy = GridLimitPolicy(
        base_limit_kw=400.0,
        peak_boost_limit_kw=500.0,
        morning_peak_start=420,   # 07:00
        morning_peak_end=600,     # 10:00
        evening_peak_start=1020,  # 17:00
        evening_peak_end=1200,    # 20:00
    )
    controller_b = ManagedController(
        stations_b, 400.0,
        limit_policy=tou_policy,
        background_load=background_load,
        vehicle_time_budgets=vehicle_time_budgets,
    )
    result_b = Simulation(controller_b, schedule_b).run()
    print(f"   ✓ Tamamlandı: Pik={result_b.metrics_summary.peak_power_kw:.0f}kW, Aşım={result_b.metrics_summary.overload_minutes}dk")

    # Output
    ExecutiveDashboard.print_executive_summary(result_a, result_b)
    ExecutiveDashboard.create_dashboard(result_a, result_b)

    # Validation
    max_limit = controller_b.limit_policy.peak_boost_limit_kw
    assert result_b.metrics_summary.peak_power_kw <= max_limit + 0.5, \
        f"Kontrol sonrası pik {max_limit} kW'ı geçmemelidir (gerçek: {result_b.metrics_summary.peak_power_kw:.1f} kW)"
    print("✓ Tüm doğrulamalar başarılı")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Yük Dengeleme Simülasyonu v3")
    parser.add_argument("--generate-new", action="store_true", help="Yeni veri seti oluştur")
    args = parser.parse_args()

    main(generate_new=args.generate_new)