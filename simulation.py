#!/usr/bin/env python3
"""
EV Charging Load Balancing Simulation
Controlled (Managed) vs Uncontrolled (Unmanaged) scenarios
"""

from __future__ import annotations
import copy
import json
import sys
import argparse
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


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
# Data Models (Immutable)
# ==============================================================================

@dataclass(frozen=True)
class EVModel:
    """Vehicle specification template."""
    name: str
    battery_capacity_kwh: float
    max_dc_power_kw: float
    probability: float


@dataclass
class StepResult:
    """One-minute simulation step outcome."""
    minute: int
    total_power_kw: float
    overloaded: bool
    evs_completed: List[EV] = field(default_factory=list)


@dataclass
class MetricsSummary:
    """Post-simulation aggregate metrics."""
    peak_power_kw: float
    overload_minutes: int
    total_overload_kwh: float
    avg_wait_time_minutes: float
    avg_charge_time_minutes: float
    avg_energy_delivered_kwh: float
    evs_completed: int


@dataclass
class VehicleMetrics:
    """Per-vehicle detailed metrics."""
    session_id: str
    model_name: str
    arrival_minute: int
    initial_soc: float
    target_soc: float
    final_soc: float
    entry_minute: int
    departure_minute: int
    wait_time_minutes: int
    charge_time_minutes: int
    energy_delivered_kwh: float
    avg_power_kw: float
    station_id: str
    station_max_power_kw: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SimulationResult:
    """Complete simulation outcome."""
    scenario_name: str
    power_timeseries: np.ndarray  # shape (1440,)
    completed_evs: List[EV]
    metrics_summary: MetricsSummary
    vehicle_metrics: List[VehicleMetrics] = field(default_factory=list)


# ==============================================================================
# Stateful Entities
# ==============================================================================

@dataclass
class EV:
    """Active vehicle charging session."""
    session_id: str
    model: EVModel
    arrival_minute: int
    initial_soc: float  # 0.0 to 1.0
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
        """Energy required to reach target SoC."""
        needed = (self.target_soc - self.current_soc) * self.model.battery_capacity_kwh
        return max(0.0, needed)

    @property
    def is_satisfied(self) -> bool:
        """True if vehicle reached target SoC."""
        return self.current_soc >= self.target_soc

    def apply_power(self, power_kw: float, duration_minutes: float) -> float:
        """
        Apply charging power for duration.
        Returns actual energy absorbed (kWh).
        Clamps at target SoC.
        """
        if self.current_soc >= self.target_soc:
            return 0.0

        energy_available_kwh = power_kw * (duration_minutes / 60.0)
        energy_to_absorb = min(energy_available_kwh, self.energy_needed_kwh)

        self.current_soc += energy_to_absorb / self.model.battery_capacity_kwh
        self.current_soc = min(self.current_soc, self.target_soc)
        self.energy_delivered_kwh += energy_to_absorb

        if power_kw > 0.0:
            self.charge_minutes += 1

        return energy_to_absorb

    @property
    def wait_time_minutes(self) -> int:
        """Minutes from arrival to charge start."""
        if self.charge_start_minute is None:
            return 0
        return self.charge_start_minute - self.arrival_minute


@dataclass
class ChargingStation:
    """Physical EVSE at a location."""
    station_id: str
    station_type: StationType
    max_power_kw: float
    current_ev: Optional[EV] = None
    allocated_power_kw: float = 0.0

    def is_available(self) -> bool:
        return self.current_ev is None

    def effective_max_power_kw(self) -> float:
        """Max power limited by station and EV."""
        if self.current_ev is None:
            return 0.0
        return min(self.max_power_kw, self.current_ev.model.max_dc_power_kw)

    def plug_in(self, ev: EV) -> None:
        if not self.is_available():
            raise RuntimeError(f"Station {self.station_id} is occupied")
        self.current_ev = ev
        ev.state = EVState.CHARGING

    def unplug(self) -> Optional[EV]:
        ev = self.current_ev
        self.current_ev = None
        self.allocated_power_kw = 0.0
        return ev

    def step(self, duration_minutes: float) -> float:
        """Execute one time step. Returns energy delivered (kWh)."""
        if self.current_ev is None:
            return 0.0
        energy = self.current_ev.apply_power(self.allocated_power_kw, duration_minutes)
        if self.current_ev.is_satisfied:
            self.current_ev.state = EVState.DONE
        return energy


# ==============================================================================
# Queue
# ==============================================================================

class WaitingQueue:
    """FIFO queue for waiting EVs."""

    def __init__(self):
        self.queue: deque = deque()

    def enqueue(self, ev: EV) -> None:
        self.queue.append(ev)

    def dequeue(self) -> Optional[EV]:
        return self.queue.popleft() if self.queue else None

    def peek(self) -> Optional[EV]:
        return self.queue[0] if self.queue else None

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)


# ==============================================================================
# Grid Control (Abstract Base)
# ==============================================================================

class GridController(ABC):
    """Abstract edge controller for power allocation."""

    def __init__(self, stations: List[ChargingStation], grid_limit_kw: float):
        self.stations = stations
        self.queue = WaitingQueue()
        self.grid_limit_kw = grid_limit_kw
        self.metrics = MetricsCollector(grid_limit_kw, stations)

    def get_baseline_at_minute(self, minute: int) -> float:
        """Time-dependent baseline consumption (business hours vs night)."""
        hour = (minute // 60) % 24
        # 06:00-22:00: high business load (200 kW), 22:00-06:00: low load (50 kW)
        if 6 <= hour < 22:
            return 200.0
        else:
            return 50.0

    @abstractmethod
    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        """Return {station_id: kw_allocated} mapping. Respects grid_limit - baseline."""
        pass

    def dispatch_waiting_evs(self, minute: int) -> None:
        """Assign queued EVs to free stations (greedy first-fit)."""
        for station in self.stations:
            if station.is_available() and not self.queue.is_empty():
                ev = self.queue.dequeue()
                station.plug_in(ev)
                ev.charge_start_minute = minute

    def collect_finished_evs(self) -> List[tuple]:
        """Remove satisfied EVs, return them with station IDs."""
        finished = []
        for station in self.stations:
            if station.current_ev and station.current_ev.state == EVState.DONE:
                ev = station.unplug()
                finished.append((ev, station.station_id))
        return finished

    def step(self, minute: int) -> StepResult:
        """Execute one minute: dispatch → allocate → charge → collect."""
        self.dispatch_waiting_evs(minute)

        baseline_now = self.get_baseline_at_minute(minute)
        allocations = self.allocate_power(baseline_now)
        for station in self.stations:
            station.allocated_power_kw = allocations.get(station.station_id, 0.0)
            station.step(duration_minutes=1.0)

        finished_evs = self.collect_finished_evs()
        total_power = baseline_now + sum(allocations.values())
        overloaded = total_power > self.grid_limit_kw

        self.metrics.record_step(minute, total_power)
        for ev, station_id in finished_evs:
            ev.departure_minute = minute
            self.metrics.record_completed_ev(ev, station_id)

        return StepResult(
            minute=minute,
            total_power_kw=total_power,
            overloaded=overloaded,
            evs_completed=[ev for ev, _ in finished_evs]
        )


class UnmanagedController(GridController):
    """Uncontrolled scenario: no power budgeting."""

    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        """Each station gets its effective max unconstrained."""
        allocations = {}
        for station in self.stations:
            allocations[station.station_id] = station.effective_max_power_kw()
        return allocations


class ManagedController(GridController):
    """Controlled scenario: max-min fair-share algorithm."""

    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        """
        Max-min fair allocation: stations with lower capacity get
        their full demand first, remaining budget split fairly among rest.
        Budget = grid_limit - baseline_consumption (dynamic).
        """
        occupied_stations = [s for s in self.stations if not s.is_available()]

        if not occupied_stations:
            return {s.station_id: 0.0 for s in self.stations}

        allocations = {}
        available_for_ev = self.grid_limit_kw - baseline_kw
        remaining_budget = available_for_ev
        unresolved = occupied_stations[:]

        while unresolved:
            fair_share = remaining_budget / len(unresolved)
            resolved_this_round = []

            for station in unresolved:
                cap = station.effective_max_power_kw()
                if cap <= fair_share:
                    allocations[station.station_id] = cap
                    remaining_budget -= cap
                    resolved_this_round.append(station)

            if not resolved_this_round:
                # All remaining want more than fair_share → cap all equally
                for station in unresolved:
                    allocations[station.station_id] = fair_share
                break

            unresolved = [s for s in unresolved if s not in resolved_this_round]

        # Fill in unallocated stations with zero
        for station in self.stations:
            if station.station_id not in allocations:
                allocations[station.station_id] = 0.0

        return allocations


# ==============================================================================
# Arrival Generation
# ==============================================================================

class ArrivalGenerator:
    """Synthesize EV arrival schedule."""

    # Real Turkish market EV models
    EV_MODELS = [
        EVModel("Togg T10X Long Range", 88.5, 150.0, 0.25),
        EVModel("Tesla Model Y Long Range", 75.0, 250.0, 0.20),
        EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
        EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
        EVModel("MG4 Standard", 51.0, 117.0, 0.10),
        EVModel("Renault Megane E-Tech", 60.0, 130.0, 0.10),
        EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
    ]

    def __init__(self, daily_ev_count: int = 50):
        self.daily_ev_count = daily_ev_count
        self.rng = None

    def generate_arrivals(self, rng: np.random.Generator) -> Dict[int, List[EV]]:
        """Generate 24-hour arrival schedule. Returns {minute: [EV, ...]}."""
        self.rng = rng

        # Arrival minutes: bimodal distribution
        arrival_minutes = self._bimodal_arrival_minutes(self.daily_ev_count)

        schedule: Dict[int, List[EV]] = {}
        for i, minute in enumerate(arrival_minutes):
            minute_idx = int(np.clip(minute, 0, 1439))
            if minute_idx not in schedule:
                schedule[minute_idx] = []

            ev = self._create_ev(session_id=f"EV_{i:03d}", arrival_minute=minute_idx)
            schedule[minute_idx].append(ev)

        return schedule

    def _bimodal_arrival_minutes(self, count: int) -> np.ndarray:
        """Sample from bimodal distribution: morning (08:00) and evening (18:00) peaks."""
        # 08:00 = 480 minutes, 18:00 = 1080 minutes
        morning_std = 60.0  # ~1 hour spread
        evening_std = 60.0

        n_morning = count // 2
        n_evening = count - n_morning

        morning = self.rng.normal(480, morning_std, n_morning)
        evening = self.rng.normal(1080, evening_std, n_evening)

        arrivals = np.concatenate([morning, evening])
        self.rng.shuffle(arrivals)

        return np.clip(arrivals, 0, 1439)

    def _create_ev(self, session_id: str, arrival_minute: int) -> EV:
        """Create an EV instance with random model and SoC."""
        model = self._sample_ev_model()
        initial_soc = self.rng.uniform(0.10, 0.50)
        return EV(
            session_id=session_id,
            model=model,
            arrival_minute=arrival_minute,
            initial_soc=initial_soc,
            target_soc=0.80
        )

    def _sample_ev_model(self) -> EVModel:
        """Weighted random model selection."""
        models = self.EV_MODELS
        weights = [m.probability for m in models]
        idx = self.rng.choice(len(models), p=weights)
        return models[idx]


# ==============================================================================
# Metrics Collection
# ==============================================================================

class MetricsCollector:
    """Accumulate simulation metrics."""

    def __init__(self, grid_limit_kw: float, stations: List[ChargingStation]):
        self.grid_limit_kw = grid_limit_kw
        self.stations = stations
        self.power_log: List[float] = []
        self.completed_evs: List[EV] = []

    def record_step(self, minute: int, power_kw: float) -> None:
        self.power_log.append(power_kw)

    def record_completed_ev(self, ev: EV, station_id: str) -> None:
        self.completed_evs.append((ev, station_id))

    def get_vehicle_metrics(self) -> List[VehicleMetrics]:
        """Convert completed EVs to detailed vehicle metrics."""
        metrics = []
        station_map = {s.station_id: s for s in self.stations}

        for ev, station_id in self.completed_evs:
            station = station_map.get(station_id)
            station_max = station.max_power_kw if station else 0.0

            avg_power = (ev.energy_delivered_kwh / ev.charge_minutes * 60.0
                         if ev.charge_minutes > 0 else 0.0)

            metrics.append(VehicleMetrics(
                session_id=ev.session_id,
                model_name=ev.model.name,
                arrival_minute=ev.arrival_minute,
                initial_soc=ev.initial_soc,
                target_soc=ev.target_soc,
                final_soc=ev.current_soc,
                entry_minute=ev.charge_start_minute or 0,
                departure_minute=ev.departure_minute or 0,
                wait_time_minutes=ev.wait_time_minutes,
                charge_time_minutes=ev.charge_minutes,
                energy_delivered_kwh=ev.energy_delivered_kwh,
                avg_power_kw=avg_power,
                station_id=station_id,
                station_max_power_kw=station_max,
            ))
        return metrics

    def summarize(self) -> MetricsSummary:
        """Compute final metrics."""
        power_arr = np.array(self.power_log)

        peak_power = float(power_arr.max())
        overload_mask = power_arr > self.grid_limit_kw
        overload_minutes = int(overload_mask.sum())
        total_overload_kwh = float(
            (power_arr[overload_mask] - self.grid_limit_kw).sum() / 60.0
        ) if overload_minutes > 0 else 0.0

        if self.completed_evs:
            evs_only = [ev for ev, _ in self.completed_evs]
            wait_times = [
                ev.wait_time_minutes for ev in evs_only
                if ev.charge_start_minute is not None
            ]
            charge_times = [ev.charge_minutes for ev in evs_only]
            energy_delivered = [ev.energy_delivered_kwh for ev in evs_only]

            avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
            avg_charge = float(np.mean(charge_times)) if charge_times else 0.0
            avg_energy = float(np.mean(energy_delivered)) if energy_delivered else 0.0
        else:
            avg_wait = avg_charge = avg_energy = 0.0

        return MetricsSummary(
            peak_power_kw=peak_power,
            overload_minutes=overload_minutes,
            total_overload_kwh=total_overload_kwh,
            avg_wait_time_minutes=avg_wait,
            avg_charge_time_minutes=avg_charge,
            avg_energy_delivered_kwh=avg_energy,
            evs_completed=len(self.completed_evs)
        )


# ==============================================================================
# Simulation Orchestration
# ==============================================================================

class Simulation:
    """Top-level simulation runner."""

    def __init__(self, controller: GridController, arrival_schedule: Dict[int, List[EV]]):
        self.controller = controller
        self.arrival_schedule = arrival_schedule

    def run(self) -> SimulationResult:
        """Execute 24-hour simulation."""
        for minute in range(1440):
            # Inject arrivals
            for ev in self.arrival_schedule.get(minute, []):
                self.controller.queue.enqueue(ev)

            # Execute one minute
            self.controller.step(minute)

        metrics = self.controller.metrics.summarize()
        power_arr = np.array(self.controller.metrics.power_log)
        vehicle_metrics = self.controller.metrics.get_vehicle_metrics()
        completed_evs = [ev for ev, _ in self.controller.metrics.completed_evs]

        return SimulationResult(
            scenario_name=self.controller.__class__.__name__,
            power_timeseries=power_arr,
            completed_evs=completed_evs,
            metrics_summary=metrics,
            vehicle_metrics=vehicle_metrics
        )


# ==============================================================================
# Reporting
# ==============================================================================

class DatasetManager:
    """Save and load vehicle datasets for reproducibility."""

    @staticmethod
    def save_dataset(arrival_schedule: Dict[int, List[EV]], seed: int,
                    baseline_consumption_kw: float = 50.0,
                    filename: str = "F:/LoadBalancing/dataset.json") -> None:
        """Save arrival schedule and grid baseline to JSON."""
        vehicles_data = []
        # Deduplicate vehicles (in case same schedule used multiple times)
        seen_sessions = set()
        for minute in sorted(arrival_schedule.keys()):
            for ev in arrival_schedule[minute]:
                if ev.session_id not in seen_sessions:
                    vehicles_data.append({
                        "session_id": ev.session_id,
                        "model_name": ev.model.name,
                        "battery_capacity_kwh": ev.model.battery_capacity_kwh,
                        "max_dc_power_kw": ev.model.max_dc_power_kw,
                        "arrival_minute": ev.arrival_minute,
                        "initial_soc": ev.initial_soc,
                        "target_soc": ev.target_soc,
                    })
                    seen_sessions.add(ev.session_id)

        dataset = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "grid_baseline_consumption_kw": baseline_consumption_kw,
            "total_vehicles": len(vehicles_data),
            "vehicles": vehicles_data
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Dataset saved: {filename} ({len(vehicles_data)} vehicles, {baseline_consumption_kw} kW baseline)")

    @staticmethod
    def load_dataset(filename: str = "F:/LoadBalancing/dataset.json") -> tuple:
        """Load arrival schedule and baseline from JSON. Returns (schedule, baseline_kw)."""
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        baseline = dataset.get("grid_baseline_consumption_kw", 50.0)

        arrival_schedule: Dict[int, List[EV]] = {}
        for vehicle_data in dataset["vehicles"]:
            ev = EV(
                session_id=vehicle_data["session_id"],
                model=EVModel(
                    name=vehicle_data["model_name"],
                    battery_capacity_kwh=vehicle_data["battery_capacity_kwh"],
                    max_dc_power_kw=vehicle_data["max_dc_power_kw"],
                    probability=0.0
                ),
                arrival_minute=vehicle_data["arrival_minute"],
                initial_soc=vehicle_data["initial_soc"],
                target_soc=vehicle_data["target_soc"],
            )
            minute = vehicle_data["arrival_minute"]
            if minute not in arrival_schedule:
                arrival_schedule[minute] = []
            arrival_schedule[minute].append(ev)

        print(f"✓ Dataset loaded: {filename} ({len(dataset['vehicles'])} vehicles, {baseline} kW baseline)")
        return arrival_schedule, baseline


class Reporter:
    """Visualization and console output."""

    @staticmethod
    def print_vehicle_comparison_table(result_a: SimulationResult, result_b: SimulationResult) -> None:
        """Print detailed per-vehicle comparison."""
        print("\n" + "=" * 200)
        print("DETAILED VEHICLE-BY-VEHICLE COMPARISON (Unmanaged vs Managed)")
        print("=" * 200)

        # Create a mapping for easier lookup
        metrics_a_map = {m.session_id: m for m in result_a.vehicle_metrics}
        metrics_b_map = {m.session_id: m for m in result_b.vehicle_metrics}

        # Get all session IDs
        all_sessions = sorted(set(list(metrics_a_map.keys()) + list(metrics_b_map.keys())))

        # Header
        print(f"{'Session':<10} {'Model':<25} {'Arr(min)':<8} "
              f"{'Init.SoC':<8} {'Unm.Entry':<10} {'Unm.Wait':<9} {'Unm.Dur':<9} {'Unm.AvgPow':<11} "
              f"{'Mgd.Entry':<10} {'Mgd.Wait':<9} {'Mgd.Dur':<9} {'Mgd.AvgPow':<11} "
              f"{'WaitΔ':<8} {'DurΔ':<8} {'PowΔ':<8}")
        print("-" * 200)

        for session_id in all_sessions:
            m_a = metrics_a_map.get(session_id)
            m_b = metrics_b_map.get(session_id)

            if m_a and m_b:
                wait_delta = m_b.wait_time_minutes - m_a.wait_time_minutes
                dur_delta = m_b.charge_time_minutes - m_a.charge_time_minutes
                pow_delta = m_b.avg_power_kw - m_a.avg_power_kw

                print(f"{session_id:<10} {m_a.model_name[:24]:<25} {m_a.arrival_minute:<8} "
                      f"{m_a.initial_soc:<8.2f} {m_a.entry_minute:<10} {m_a.wait_time_minutes:<9} "
                      f"{m_a.charge_time_minutes:<9} {m_a.avg_power_kw:<11.1f} "
                      f"{m_b.entry_minute:<10} {m_b.wait_time_minutes:<9} "
                      f"{m_b.charge_time_minutes:<9} {m_b.avg_power_kw:<11.1f} "
                      f"{wait_delta:<8} {dur_delta:<8} {pow_delta:<8.1f}")

        print("=" * 200 + "\n")

    @staticmethod
    def save_vehicle_comparison_csv(result_a: SimulationResult, result_b: SimulationResult,
                                     filename: str = "F:/LoadBalancing/vehicle_comparison.csv") -> None:
        """Save detailed comparison to CSV."""
        metrics_a_map = {m.session_id: m for m in result_a.vehicle_metrics}
        metrics_b_map = {m.session_id: m for m in result_b.vehicle_metrics}
        all_sessions = sorted(set(list(metrics_a_map.keys()) + list(metrics_b_map.keys())))

        rows = ["Session ID,Model,Arrival(min),Initial SoC,Unm.Entry(min),Unm.Wait(min),Unm.Charge(min),Unm.AvgPower(kW),"
                "Mgd.Entry(min),Mgd.Wait(min),Mgd.Charge(min),Mgd.AvgPower(kW),Wait Change(min),Duration Change(min),Power Change(kW)"]

        for session_id in all_sessions:
            m_a = metrics_a_map.get(session_id)
            m_b = metrics_b_map.get(session_id)

            if m_a and m_b:
                wait_delta = m_b.wait_time_minutes - m_a.wait_time_minutes
                dur_delta = m_b.charge_time_minutes - m_a.charge_time_minutes
                pow_delta = m_b.avg_power_kw - m_a.avg_power_kw

                row = (f"{session_id},{m_a.model_name},{m_a.arrival_minute},"
                       f"{m_a.initial_soc:.2f},{m_a.entry_minute},{m_a.wait_time_minutes},"
                       f"{m_a.charge_time_minutes},{m_a.avg_power_kw:.1f},"
                       f"{m_b.entry_minute},{m_b.wait_time_minutes},"
                       f"{m_b.charge_time_minutes},{m_b.avg_power_kw:.1f},"
                       f"{wait_delta},{dur_delta},{pow_delta:.1f}")
                rows.append(row)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows))
        print(f"✓ Vehicle comparison saved: {filename}")

    @staticmethod
    def print_console_comparison(result_a: SimulationResult, result_b: SimulationResult) -> None:
        """Print side-by-side metrics table."""
        print("\n" + "=" * 90)
        print("EV CHARGING LOAD BALANCING SIMULATION — RESULTS COMPARISON")
        print("=" * 90)

        metrics_a = result_a.metrics_summary
        metrics_b = result_b.metrics_summary

        rows = [
            ("Metric", "Unmanaged (A)", "Managed (B)"),
            ("-" * 40, "-" * 20, "-" * 20),
            ("Peak Power (kW)", f"{metrics_a.peak_power_kw:.1f}", f"{metrics_b.peak_power_kw:.1f}"),
            ("Overload Minutes", f"{metrics_a.overload_minutes}", f"{metrics_b.overload_minutes}"),
            ("Total Overload Energy (kWh)", f"{metrics_a.total_overload_kwh:.1f}", f"{metrics_b.total_overload_kwh:.1f}"),
            ("Avg Wait Time (min)", f"{metrics_a.avg_wait_time_minutes:.1f}", f"{metrics_b.avg_wait_time_minutes:.1f}"),
            ("Avg Charge Time (min)", f"{metrics_a.avg_charge_time_minutes:.1f}", f"{metrics_b.avg_charge_time_minutes:.1f}"),
            ("Avg Energy Delivered (kWh)", f"{metrics_a.avg_energy_delivered_kwh:.1f}", f"{metrics_b.avg_energy_delivered_kwh:.1f}"),
            ("EVs Completed", f"{metrics_a.evs_completed}", f"{metrics_b.evs_completed}"),
        ]

        for row in rows:
            print(f"{row[0]:<40} {row[1]:>20} {row[2]:>20}")

        print("=" * 90 + "\n")

    @staticmethod
    def plot_comparison(result_a: SimulationResult, result_b: SimulationResult) -> None:
        """Create 8-panel comparison figure (grid + per-vehicle)."""
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        fig.suptitle("EV Charging Load Balancing Simulation — Detailed Scenario Comparison", fontsize=14, fontweight="bold")

        # ===== GRID-LEVEL PANELS (Rows 0-1) =====

        # Panel 1: Power time series
        ax = axes[0, 0]
        minutes = np.arange(1440)
        ax.plot(minutes, result_a.power_timeseries, label="Unmanaged (A)", linewidth=1.5, color="red", alpha=0.7)
        ax.plot(minutes, result_b.power_timeseries, label="Managed (B)", linewidth=1.5, color="green", alpha=0.7)
        ax.axhline(400, color="red", linestyle="--", linewidth=2, label="Grid Limit (400 kW)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Total Power Draw (kW)")
        ax.set_title("Grid Power Consumption Over 24 Hours")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Overload shading
        ax = axes[0, 1]
        ax.fill_between(minutes, 0, result_a.power_timeseries, where=(result_a.power_timeseries > 400),
                         label="Unmanaged Overload", color="red", alpha=0.4)
        ax.plot(minutes, result_b.power_timeseries, label="Managed (B)", linewidth=1.5, color="green")
        ax.axhline(400, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Overload Identification (Unmanaged)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 600])

        # Panel 3: Impact severity heatmap (wait + charge impact)
        ax = axes[1, 0]
        metrics_a_map = {m.session_id: m for m in result_a.vehicle_metrics}
        metrics_b_map = {m.session_id: m for m in result_b.vehicle_metrics}
        all_sessions = sorted(set(list(metrics_a_map.keys()) + list(metrics_b_map.keys())))

        total_impact = []
        model_names = []
        for sid in all_sessions:
            if sid in metrics_a_map and sid in metrics_b_map:
                wait_impact = metrics_b_map[sid].wait_time_minutes - metrics_a_map[sid].wait_time_minutes
                charge_impact = metrics_b_map[sid].charge_time_minutes - metrics_a_map[sid].charge_time_minutes
                total_impact.append(wait_impact + charge_impact)
                model_names.append(metrics_a_map[sid].model_name.split()[0])  # First word of model

        sorted_indices = np.argsort(total_impact)
        sorted_impact = np.array(total_impact)[sorted_indices]
        sorted_models = np.array(model_names)[sorted_indices]

        colors = ['red' if x > 0 else 'green' for x in sorted_impact]
        ax.barh(range(len(sorted_impact)), sorted_impact, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(0, len(sorted_impact), 5))
        ax.set_yticklabels([f"{all_sessions[sorted_indices[i]]}" for i in range(0, len(sorted_impact), 5)])
        ax.set_xlabel("Total Impact: Wait + Charge Duration (min)")
        ax.set_title("Impact Severity by Vehicle (Red=Delayed, Green=Improved)")
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 4: Cost benefit analysis (power savings vs time cost)
        ax = axes[1, 1]
        power_savings = []
        time_costs = []
        for sid in all_sessions:
            if sid in metrics_a_map and sid in metrics_b_map:
                power_saved = metrics_a_map[sid].avg_power_kw - metrics_b_map[sid].avg_power_kw
                time_cost = (metrics_b_map[sid].wait_time_minutes + metrics_b_map[sid].charge_time_minutes) - \
                           (metrics_a_map[sid].wait_time_minutes + metrics_a_map[sid].charge_time_minutes)
                if power_saved > 0:  # Only include vehicles with power reduction
                    power_savings.append(power_saved)
                    time_costs.append(time_cost)

        if power_savings:
            scatter = ax.scatter(power_savings, time_costs, s=100, alpha=0.6, c=time_costs,
                                cmap='RdYlGn_r', edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel("Power Reduction (kW)")
            ax.set_ylabel("Time Cost (minutes)")
            ax.set_title("Trade-off: Power Savings vs User Wait Time")
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Time Cost (min)", rotation=270, labelpad=15)

        # ===== PER-VEHICLE PANELS (Rows 2-3) =====

        # Panel 5: Impact vs Initial SoC
        ax = axes[2, 0]
        metrics_a_map = {m.session_id: m for m in result_a.vehicle_metrics}
        metrics_b_map = {m.session_id: m for m in result_b.vehicle_metrics}
        all_sessions = sorted(set(list(metrics_a_map.keys()) + list(metrics_b_map.keys())))

        initial_soc_list = []
        wait_impact_list = []
        for sid in all_sessions:
            if sid in metrics_a_map and sid in metrics_b_map:
                initial_soc = metrics_a_map[sid].initial_soc
                wait_impact = (metrics_b_map[sid].wait_time_minutes -
                              metrics_a_map[sid].wait_time_minutes)
                initial_soc_list.append(initial_soc * 100)
                wait_impact_list.append(wait_impact)

        scatter = ax.scatter(initial_soc_list, wait_impact_list, s=100, alpha=0.6, c=wait_impact_list,
                            cmap='RdYlGn_r', edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Initial SoC (%)")
        ax.set_ylabel("Wait Time Impact (min)")
        ax.set_title("Wait Time Impact by Battery Level")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Impact (min)", rotation=270, labelpad=15)

        # Panel 6: Impact vs Arrival Time
        ax = axes[2, 1]
        arrival_hour_list = []
        charge_impact_list = []
        for sid in all_sessions:
            if sid in metrics_a_map and sid in metrics_b_map:
                arrival_minute = metrics_a_map[sid].arrival_minute
                arrival_hour = (arrival_minute / 60) % 24
                charge_impact = (metrics_b_map[sid].charge_time_minutes -
                                metrics_a_map[sid].charge_time_minutes)
                arrival_hour_list.append(arrival_hour)
                charge_impact_list.append(charge_impact)

        scatter = ax.scatter(arrival_hour_list, charge_impact_list, s=100, alpha=0.6, c=charge_impact_list,
                            cmap='RdYlGn_r', edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvspan(6, 22, alpha=0.1, color='orange', label='Business Hours')
        ax.set_xlabel("Arrival Time (hour of day)")
        ax.set_ylabel("Charge Duration Impact (min)")
        ax.set_title("Charge Time Impact by Arrival Time")
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Impact (min)", rotation=270, labelpad=15)

        # Panel 7: Total session time (wait + charge)
        ax = axes[3, 0]
        total_a = [metrics_a_map[sid].wait_time_minutes + metrics_a_map[sid].charge_time_minutes
                   for sid in all_sessions if sid in metrics_a_map]
        total_b = [metrics_b_map[sid].wait_time_minutes + metrics_b_map[sid].charge_time_minutes
                   for sid in all_sessions if sid in metrics_b_map]

        x_pos = np.arange(len(all_sessions))
        width = 0.35
        ax.bar(x_pos - width/2, total_a, width, label="Unmanaged", alpha=0.7, color="red")
        ax.bar(x_pos + width/2, total_b, width, label="Managed", alpha=0.7, color="green")
        ax.set_xlabel("Vehicle")
        ax.set_ylabel("Total Time (minutes)")
        ax.set_title("Per-Vehicle Total Session Time (Wait + Charge)")
        ax.set_xticks(x_pos[::5])
        ax.set_xticklabels([all_sessions[i] for i in range(0, len(all_sessions), 5)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 8: Average power comparison
        ax = axes[3, 1]
        power_a_list = [metrics_a_map[sid].avg_power_kw for sid in all_sessions if sid in metrics_a_map]
        power_b_list = [metrics_b_map[sid].avg_power_kw for sid in all_sessions if sid in metrics_b_map]

        x_pos = np.arange(len(all_sessions))
        width = 0.35
        ax.bar(x_pos - width/2, power_a_list, width, label="Unmanaged", alpha=0.7, color="red")
        ax.bar(x_pos + width/2, power_b_list, width, label="Managed", alpha=0.7, color="green")
        ax.set_xlabel("Vehicle")
        ax.set_ylabel("Average Power (kW)")
        ax.set_title("Per-Vehicle Average Charging Power")
        ax.set_xticks(x_pos[::5])
        ax.set_xticklabels([all_sessions[i] for i in range(0, len(all_sessions), 5)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("F:/LoadBalancing/simulation_results.png", dpi=150, bbox_inches="tight")
        plt.show()


# ==============================================================================
# Main
# ==============================================================================

def main(generate_new: bool = False):
    """Run full simulation with both scenarios."""
    print("Initializing EV Charging Load Balancing Simulation...")

    # Load or generate arrival schedule
    if not generate_new:
        try:
            arrival_schedule, _ = DatasetManager.load_dataset()
            print("ℹ️  Loaded saved dataset (use --generate-new to create fresh vehicles)")
        except FileNotFoundError:
            print("⚠️  No saved dataset found. Generating new one...")
            generate_new = True

    if generate_new:
        rng = np.random.default_rng(seed=42)
        generator = ArrivalGenerator(daily_ev_count=50)
        arrival_schedule = generator.generate_arrivals(rng)
        DatasetManager.save_dataset(arrival_schedule, seed=42, baseline_consumption_kw=50.0)
        print("✓ New dataset generated and saved")

    print(f"Using arrivals: {sum(len(evs) for evs in arrival_schedule.values())} EVs across 24 hours")
    print(f"Grid baseline consumption: dynamic (06:00-22:00: 200 kW, 22:00-06:00: 50 kW)")

    # Setup stations
    stations_a = [
        ChargingStation("S1", StationType.ULTRA_FAST, 350.0),
        ChargingStation("S2", StationType.FAST, 150.0),
        ChargingStation("S3", StationType.FAST, 150.0),
        ChargingStation("S4", StationType.STANDARD, 50.0),
        ChargingStation("S5", StationType.STANDARD, 50.0),
    ]

    stations_b = [
        ChargingStation("S1", StationType.ULTRA_FAST, 350.0),
        ChargingStation("S2", StationType.FAST, 150.0),
        ChargingStation("S3", StationType.FAST, 150.0),
        ChargingStation("S4", StationType.STANDARD, 50.0),
        ChargingStation("S5", StationType.STANDARD, 50.0),
    ]

    # Deep copy arrival schedule to ensure independence
    schedule_a = copy.deepcopy(arrival_schedule)
    schedule_b = copy.deepcopy(arrival_schedule)

    # Scenario A: Unmanaged
    print("\nRunning Scenario A (Unmanaged)...")
    controller_a = UnmanagedController(stations_a, grid_limit_kw=400.0)
    sim_a = Simulation(controller_a, schedule_a)
    result_a = sim_a.run()
    print("✓ Scenario A complete")

    # Scenario B: Managed
    print("Running Scenario B (Managed)...")
    controller_b = ManagedController(stations_b, grid_limit_kw=400.0)
    sim_b = Simulation(controller_b, schedule_b)
    result_b = sim_b.run()
    print("✓ Scenario B complete")

    # Output
    Reporter.print_console_comparison(result_a, result_b)
    Reporter.print_vehicle_comparison_table(result_a, result_b)
    Reporter.save_vehicle_comparison_csv(result_a, result_b)
    Reporter.plot_comparison(result_a, result_b)

    # Assertions
    assert result_b.metrics_summary.overload_minutes == 0, "Managed scenario should have zero overload minutes"
    assert result_b.metrics_summary.peak_power_kw <= 400.0, "Managed scenario peak should not exceed 400 kW"
    print("✓ All validation assertions passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Charging Load Balancing Simulation")
    parser.add_argument("--generate-new", action="store_true",
                       help="Generate new vehicle dataset (default: use saved dataset)")
    args = parser.parse_args()

    main(generate_new=args.generate_new)
