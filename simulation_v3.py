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
                 limit_policy: Optional['GridLimitPolicy'] = None):
        self.stations = stations
        self.queue = WaitingQueue()
        self.grid_limit_kw = grid_limit_kw
        self.limit_policy = limit_policy or GridLimitPolicy(base_limit_kw=grid_limit_kw)
        self.power_log: List[float] = []
        self.grid_limit_log: List[float] = []
        self.completed_sessions: List[EV] = []
        self.queued_count: int = 0

    def get_baseline_at_minute(self, minute: int) -> float:
        """Non-EV background load. Zero for a pure EV charging hub."""
        return 0.0

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
        """Assign queued EVs to free stations using SPT (Shortest Processing Time).
        Instead of FIFO, pick the car that will complete fastest — reduces average wait.
        Only ManagedController overrides this; UnmanagedController uses base FIFO."""
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
    """Controlled: Dynamic Priority Allocation with adaptive extension ratio.

    Each tick:
      1. Compute system-state signals (congestion, grid stress, urgency)
      2. Derive dynamic max_extension_ratio — limits how much any vehicle's
         charge time can grow relative to its unmanaged baseline
      3. Floor = max_power / ratio  (bounded extension guarantee)
      4. Remaining budget distributed proportional to per-vehicle priority score
         score = 0.40 * soc_urgency + 0.35 * completion_proximity + 0.25 * wait_factor
    """

    def _compute_priority_score(self, ev: EV, current_minute: int) -> float:
        soc_urgency = 1.0 - ev.current_soc

        total_energy = max((ev.target_soc - ev.initial_soc) * ev.battery_capacity_kwh, 1e-6)
        completion_proximity = float(np.clip(1.0 - ev.energy_needed_kwh / total_energy, 0.0, 1.0))

        if ev.charge_start_minute is None:
            wait_minutes = current_minute - ev.arrival_minute
        else:
            wait_minutes = ev.wait_time_minutes
        wait_factor = min(wait_minutes / 30.0, 1.0)

        return 0.40 * soc_urgency + 0.35 * completion_proximity + 0.25 * wait_factor

    def _compute_dynamic_ratio(self, grid_limit_kw: float, baseline_kw: float) -> float:
        """Adaptive ratio driven by live system signals.

        Signals:
          congestion  — queue pressure relative to station count
          grid_stress — how much of the grid budget baseline already consumes
          urgency     — mean SoC deficit of vehicles currently charging
        """
        congestion = min(len(self.queue) / max(len(self.stations), 1), 1.0)
        grid_stress = min(baseline_kw / max(grid_limit_kw, 1e-6), 1.0)

        charging_evs = [s.current_ev for s in self.stations
                        if s.current_ev and not s.current_ev.is_satisfied]
        urgency = float(np.mean([1.0 - ev.current_soc for ev in charging_evs])) \
                  if charging_evs else 0.0

        ratio = 1.5 + congestion * 2.0 + grid_stress * 1.0 + urgency * 0.5
        return float(np.clip(ratio, 1.2, 5.0))

    def dispatch_waiting_evs(self, minute: int) -> None:
        """Priority-based dispatch: highest-score EV in queue enters next free station."""
        for station in self.stations:
            if station.is_available() and not self.queue.is_empty():
                best_ev = max(
                    self.queue.queue,
                    key=lambda ev: self._compute_priority_score(ev, minute),
                )
                self.queue.queue.remove(best_ev)
                station.plug_in(best_ev)
                best_ev.charge_start_minute = minute

    def allocate_power(self, baseline_kw: float, minute: int, grid_limit_kw: float) -> Dict[str, float]:
        occupied_stations = [s for s in self.stations if not s.is_available()]

        if not occupied_stations:
            return {s.station_id: 0.0 for s in self.stations}

        available_budget = max(0.0, grid_limit_kw - baseline_kw - 0.01)

        vehicles = []
        for station in occupied_stations:
            ev = station.current_ev
            if ev.is_satisfied:
                continue
            max_power = station.effective_max_power_kw()
            if ev.energy_needed_kwh > 0.0 and max_power > 0.0:
                vehicles.append({
                    'station_id': station.station_id,
                    'ev': ev,
                    'max_power': max_power,
                    'priority': self._compute_priority_score(ev, minute),
                })

        if not vehicles:
            return {s.station_id: 0.0 for s in self.stations}

        # Dynamic floor: max_power / ratio guarantees bounded charge-time extension.
        # ratio adapts every tick — low congestion → ratio ~1.5 (barely throttle),
        # high congestion + stressed grid → ratio up to 5.0 (spread power more).
        ratio = self._compute_dynamic_ratio(grid_limit_kw, baseline_kw)

        allocations: Dict[str, float] = {}
        floor_total = 0.0
        for v in vehicles:
            floor = v['max_power'] / ratio
            allocations[v['station_id']] = floor
            floor_total += floor

        if floor_total > available_budget:
            scale = available_budget / floor_total
            allocations = {sid: p * scale for sid, p in allocations.items()}
            greedy_budget = 0.0
        else:
            greedy_budget = available_budget - floor_total

        # Priority-weighted distribution of remaining budget
        if greedy_budget > 0.0:
            total_priority = sum(v['priority'] for v in vehicles)
            if total_priority > 0.0:
                for v in vehicles:
                    share = (v['priority'] / total_priority) * greedy_budget
                    headroom = v['max_power'] - allocations[v['station_id']]
                    allocations[v['station_id']] += min(share, max(headroom, 0.0))

        for station in occupied_stations:
            if station.current_ev.is_satisfied:
                allocations[station.station_id] = 0.0

        for station in self.stations:
            if station.station_id not in allocations:
                allocations[station.station_id] = 0.0

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
        arrival_minutes = self._bimodal_arrival_minutes(rng, self.daily_ev_count)

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

    def _bimodal_arrival_minutes(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """Bimodal distribution: morning (08:00) and evening (18:00)."""
        n_morning = count // 2
        n_evening = count - n_morning

        morning = rng.normal(480, 60, n_morning)
        evening = rng.normal(1080, 60, n_evening)

        arrivals = np.concatenate([morning, evening])
        rng.shuffle(arrivals)
        return np.clip(arrivals, 0, 1439)

    def _sample_ev_model(self, rng: np.random.Generator) -> EVModel:
        weights = [m.probability for m in self.EV_MODELS]
        idx = rng.choice(len(self.EV_MODELS), p=weights)
        return self.EV_MODELS[idx]


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
        """Create 6-panel executive dashboard in Turkish."""
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle("EV Şarj Yük Dengeleme — Yönetici Özeti", fontsize=18, fontweight="bold", y=0.985)

        # Panel 1: Grid Load (Filled Area)
        ax1 = plt.subplot(3, 2, 1)
        minutes = np.arange(1440)
        hours = minutes / 60

        ax1.fill_between(hours, 0, result_unmanaged.power_timeseries, alpha=0.3, color="red", label="Kontrol Öncesi")
        ax1.plot(hours, result_managed.power_timeseries, linewidth=2.5, color="darkgreen", label="Kontrol Sonrası")
        # Dynamic TOU limit line
        limit_series = result_managed.grid_limit_timeseries
        if len(limit_series) == 1440:
            ax1.plot(hours, limit_series, color="red", linestyle="--", linewidth=2.5,
                     label=f"Dinamik Şebeke Limiti ({int(limit_series.min())}-{int(limit_series.max())} kW)")
            for sh, eh in [(7, 10), (17, 20)]:
                ax1.axvspan(sh, eh, alpha=0.08, color="orange",
                            label="TOU Pik Penceresi" if sh == 7 else "")
        else:
            ax1.axhline(400, color="red", linestyle="--", linewidth=2.5, label="Şebeke Limiti (400 kW)")
        ax1.set_xlabel("Gün Saati (saat)", fontsize=11)
        ax1.set_ylabel("Şebeke Yükü (kW)", fontsize=11)
        ax1.set_title("Panel 1: 24 Saatlik Şebeke Yükü Profili", fontsize=12, fontweight="bold")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 2))
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Wait Time Categories (Managed)
        ax2 = plt.subplot(3, 2, 2)

        managed_sessions = result_managed.vehicle_sessions
        no_wait = sum(1 for s in managed_sessions if s.wait_time_minutes == 0)
        short_wait = sum(1 for s in managed_sessions if 0 < s.wait_time_minutes <= 15)
        long_wait = sum(1 for s in managed_sessions if s.wait_time_minutes > 15)

        categories = ["Bekleme Yok\n(0 dk)", "Kabul Edilebilir\n(1-15 dk)", "Uzun Bekleme\n(15+ dk)"]
        values = [no_wait, short_wait, long_wait]
        colors_cat = ["#2ecc71", "#f39c12", "#e74c3c"]

        bars = ax2.bar(categories, values, color=colors_cat, edgecolor="black", linewidth=1.5, width=0.6)
        ax2.set_ylabel("Araç Sayısı", fontsize=11)
        ax2.set_title("Panel 2: Kuyruk Bekleme Dağılımı (Kontrol Sonrası)", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, max(values) * 1.15)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{int(val)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Model Comparison - CHARGE TIME (Unmanaged vs Managed)
        ax3 = plt.subplot(3, 2, 3)

        unmanaged_sessions = result_unmanaged.vehicle_sessions
        managed_sessions = result_managed.vehicle_sessions

        # Group by model for charge times
        model_charge_um = {}
        model_charge_mg = {}

        for session in unmanaged_sessions:
            model = session.model_name
            if model not in model_charge_um:
                model_charge_um[model] = []
            model_charge_um[model].append(session.charge_time_minutes)

        for session in managed_sessions:
            model = session.model_name
            if model not in model_charge_mg:
                model_charge_mg[model] = []
            model_charge_mg[model].append(session.charge_time_minutes)

        models = sorted(set(list(model_charge_um.keys()) + list(model_charge_mg.keys())))
        avg_um = [np.mean(model_charge_um.get(m, [0])) for m in models]
        avg_mg = [np.mean(model_charge_mg.get(m, [0])) for m in models]

        # Create labels with model names and counts
        model_labels = []
        for m in models:
            count = len(model_charge_um.get(m, []))
            model_labels.append(f"{m}\n(n={count})")

        x_pos = np.arange(len(models))
        width = 0.35

        bars1 = ax3.bar(x_pos - width/2, avg_um, width, label="Kontrol Öncesi", color="lightcoral", edgecolor="black")
        bars2 = ax3.bar(x_pos + width/2, avg_mg, width, label="Kontrol Sonrası", color="lightgreen", edgecolor="black")

        ax3.set_ylabel("Ortalama Şarj Süresi (dakika)", fontsize=11)
        ax3.set_xlabel("Araç Modeli (n=araç sayısı)", fontsize=11)
        ax3.set_title("Panel 3: Şarj Süresi Kıyası (Öncesi vs Sonrası)", fontsize=12, fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_labels, fontsize=9, rotation=45, ha="right")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f"{height:.0f}", ha="center", va="bottom", fontsize=8)

        # Panel 4: Model Comparison - WAIT TIME (Unmanaged vs Managed)
        ax4 = plt.subplot(3, 2, 4)

        # Group by model for wait times
        model_wait_um = {}
        model_wait_mg = {}

        for session in unmanaged_sessions:
            model = session.model_name
            if model not in model_wait_um:
                model_wait_um[model] = []
            model_wait_um[model].append(session.wait_time_minutes)

        for session in managed_sessions:
            model = session.model_name
            if model not in model_wait_mg:
                model_wait_mg[model] = []
            model_wait_mg[model].append(session.wait_time_minutes)

        models = sorted(set(list(model_wait_um.keys()) + list(model_wait_mg.keys())))
        avg_wait_um = [np.mean(model_wait_um.get(m, [0])) for m in models]
        avg_wait_mg = [np.mean(model_wait_mg.get(m, [0])) for m in models]

        # Create labels with model names and counts
        model_labels = []
        for m in models:
            count = len(model_wait_um.get(m, []))
            model_labels.append(f"{m}\n(n={count})")

        x_pos = np.arange(len(models))
        width = 0.35

        bars1 = ax4.bar(x_pos - width/2, avg_wait_um, width, label="Kontrol Öncesi", color="lightyellow", edgecolor="black")
        bars2 = ax4.bar(x_pos + width/2, avg_wait_mg, width, label="Kontrol Sonrası", color="lightblue", edgecolor="black")

        ax4.set_ylabel("Ortalama Bekleme Süresi (dakika)", fontsize=11)
        ax4.set_xlabel("Araç Modeli (n=araç sayısı)", fontsize=11)
        ax4.set_title("Panel 4: Bekleme Süresi Kıyası (Öncesi vs Sonrası)", fontsize=12, fontweight="bold")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_labels, fontsize=9, rotation=45, ha="right")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f"{height:.0f}", ha="center", va="bottom", fontsize=8)

        # Panel 5: Metrics Summary (Text)
        ax5 = plt.subplot(3, 2, 5)
        ax5.axis("off")

        peak_reduction_pct = ((result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw)
                             / result_unmanaged.metrics_summary.peak_power_kw * 100)
        overload_reduction_pct = ((result_unmanaged.metrics_summary.overload_minutes - result_managed.metrics_summary.overload_minutes)
                                 / max(result_unmanaged.metrics_summary.overload_minutes, 1) * 100)

        # Calculate additional metrics
        total_overload_kwh_diff = result_unmanaged.metrics_summary.total_overload_kwh - result_managed.metrics_summary.total_overload_kwh
        avg_delay_um = result_unmanaged.metrics_summary.avg_delay_minutes
        avg_delay_mg = result_managed.metrics_summary.avg_delay_minutes
        delay_increase = avg_delay_mg - avg_delay_um

        summary_text = f"""
ŞEBEKE KORUMASI & PERFORMANS ANALİZİ

ŞEBEKE KORUMASI:
  Korunan Kapasite: {result_managed.metrics_summary.protected_capacity_percent:.1f}% ✓
  Pik Güç: {result_unmanaged.metrics_summary.peak_power_kw:.0f}→{result_managed.metrics_summary.peak_power_kw:.0f}kW (↓{peak_reduction_pct:.1f}%)
  Aşım: {result_unmanaged.metrics_summary.overload_minutes}→{result_managed.metrics_summary.overload_minutes}dk (↓{overload_reduction_pct:.1f}%)
  Aşım Enerjisi: ↓ {total_overload_kwh_diff:.1f} kWh

HİZMET KALİTESİ:
  Servis Verilen: {result_managed.metrics_summary.evs_completed} araç (100%)
  Ort. Bekleme: {result_unmanaged.metrics_summary.avg_delay_minutes:.1f}→{result_managed.metrics_summary.avg_delay_minutes:.1f}dk (+{delay_increase:.1f}dk)

ALGORİTMA:
  Tür: SPT Dispatch + SREF + TOU Dinamik Limit
  Baz Limit: {result_managed.metrics_summary.avg_grid_limit_kw:.0f} kW (ort.)
  Pik Limit: {result_managed.grid_limit_timeseries.max():.0f} kW (07-10 & 17-20)
  Pik Boost Süresi: {result_managed.metrics_summary.peak_boost_minutes} dk
"""

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4))

        # Panel 6: Total System Time (Wait + Charge) by Model
        ax6 = plt.subplot(3, 2, 6)

        unmanaged_sessions = result_unmanaged.vehicle_sessions
        managed_sessions = result_managed.vehicle_sessions

        model_sessions_um = {}
        model_sessions_mg = {}

        for session in unmanaged_sessions:
            model = session.model_name.split()[0]
            if model not in model_sessions_um:
                model_sessions_um[model] = []
            model_sessions_um[model].append(session)

        for session in managed_sessions:
            model = session.model_name.split()[0]
            if model not in model_sessions_mg:
                model_sessions_mg[model] = []
            model_sessions_mg[model].append(session)

        all_models = sorted(set(list(model_sessions_um.keys()) + list(model_sessions_mg.keys())))

        avg_system_time_um = []
        avg_system_time_mg = []
        model_labels_short = []

        for model in all_models:
            um_sessions = model_sessions_um.get(model, [])
            mg_sessions = model_sessions_mg.get(model, [])

            if um_sessions:
                um_total = [s.wait_time_minutes + s.charge_time_minutes for s in um_sessions]
                avg_system_time_um.append(np.mean(um_total))
            else:
                avg_system_time_um.append(0)

            if mg_sessions:
                mg_total = [s.wait_time_minutes + s.charge_time_minutes for s in mg_sessions]
                avg_system_time_mg.append(np.mean(mg_total))
            else:
                avg_system_time_mg.append(0)

            model_labels_short.append(f"{model}\n(n={len(mg_sessions) if mg_sessions else 0})")

        x_pos = np.arange(len(model_labels_short))
        width = 0.35

        bars1 = ax6.bar(x_pos - width/2, avg_system_time_um, width, label="Kontrol Öncesi", color="lightcoral", edgecolor="black")
        bars2 = ax6.bar(x_pos + width/2, avg_system_time_mg, width, label="Kontrol Sonrası", color="lightgreen", edgecolor="black")

        ax6.set_ylabel("Ortalama Sistem Süresi (dakika)", fontsize=11)
        ax6.set_xlabel("Araç Modeli (n=araç sayısı)", fontsize=11)
        ax6.set_title("Panel 6: Toplam Sistemde Kalma Süresi (Bekleme + Şarj)", fontsize=12, fontweight="bold")
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(model_labels_short, fontsize=9)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax6.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f"{height:.0f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig("F:/LoadBalancing/executive_dashboard_tr.png", dpi=150, bbox_inches="tight")
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
    dataset_file = "F:/LoadBalancing/dataset.json"
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
    else:
        print("✓ Yeni veri seti oluşturuluyor...")
        rng = np.random.default_rng(seed=42)
        gen = ArrivalGenerator(50)
        arrival_schedule = gen.generate_arrivals(rng)

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
            json.dump({"timestamp": datetime.now().isoformat(), "vehicles": vehicles_data}, f, indent=2)
        print(f"✓ Veri seti kaydedildi: {len(vehicles_data)} araç")

    # Stations: Tüm istasyonlar aynı capacity → soket assignment fair
    # (aksi halde, scenario A/B'de farklı istasyonlara giden araçlar farklı güç alır)
    STATION_MAX_KW = 300.0  # All stations identical capacity
    stations_a = [
        ChargingStation("S1", StationType.ULTRA_FAST, STATION_MAX_KW),
        ChargingStation("S2", StationType.FAST, STATION_MAX_KW),
        ChargingStation("S3", StationType.FAST, STATION_MAX_KW),
        ChargingStation("S4", StationType.STANDARD, STATION_MAX_KW),
        ChargingStation("S5", StationType.STANDARD, STATION_MAX_KW),
    ]
    stations_b = [copy.deepcopy(s) for s in stations_a]

    # CRITICAL: Deepcopy BEFORE running any scenario to ensure identical vehicles
    # AND reset all EV states to initial conditions
    print("\n⚙️  Senaryo A (Kontrol Öncesi) çalıştırılıyor...")
    schedule_a = copy.deepcopy(arrival_schedule)
    schedule_b = copy.deepcopy(arrival_schedule)

    # Reset all EVs to initial state (critical for fair comparison)
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

    # SANITY CHECK: Verify schedule_a and schedule_b have identical vehicles
    vehicles_a = [(ev.session_id, ev.initial_soc, ev.battery_capacity_kwh) for minute_evs in schedule_a.values() for ev in minute_evs]
    vehicles_b = [(ev.session_id, ev.initial_soc, ev.battery_capacity_kwh) for minute_evs in schedule_b.values() for ev in minute_evs]
    vehicles_a.sort()
    vehicles_b.sort()
    if vehicles_a != vehicles_b:
        print("⚠️  HATA: schedule_a ve schedule_b farklı araçlar içeriyor!")
        print(f"  A: {len(vehicles_a)}, B: {len(vehicles_b)}")
        if len(vehicles_a) > 0 and len(vehicles_b) > 0:
            print(f"  İlk araç A: {vehicles_a[0]}, B: {vehicles_b[0]}")
    else:
        print("✓ Schedule kontrol: A ve B özdeş araçlar")

    controller_a = UnmanagedController(stations_a, 400.0)
    result_a = Simulation(controller_a, schedule_a).run()
    print(f"   ✓ Tamamlandı: Pik={result_a.metrics_summary.peak_power_kw:.0f}kW, Aşım={result_a.metrics_summary.overload_minutes}dk")

    print("⚙️  Senaryo B (Kontrol Sonrası — TOU Dinamik Limit) çalıştırılıyor...")
    tou_policy = GridLimitPolicy(
        base_limit_kw=400.0,
        peak_boost_limit_kw=500.0,
        morning_peak_start=420,   # 07:00
        morning_peak_end=600,     # 10:00
        evening_peak_start=1020,  # 17:00
        evening_peak_end=1200,    # 20:00
    )
    controller_b = ManagedController(stations_b, 400.0, limit_policy=tou_policy)
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
    import os

    parser = argparse.ArgumentParser(description="EV Yük Dengeleme Simülasyonu v3")
    parser.add_argument("--generate-new", action="store_true", help="Yeni veri seti oluştur")
    args = parser.parse_args()

    main(generate_new=args.generate_new)
