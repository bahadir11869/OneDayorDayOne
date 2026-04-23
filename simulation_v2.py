#!/usr/bin/env python3
"""
EV Charging Load Balancing Simulation v2
Smart Queue & SoC Priority Algorithm with Executive Dashboard
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
class MetricsSummary:
    """Post-simulation metrics for reporting."""
    peak_power_kw: float
    overload_minutes: int
    total_overload_kwh: float
    avg_delay_minutes: float
    evs_completed: int
    evs_queued: int
    protected_capacity_percent: float


@dataclass
class VehicleSession:
    """Per-vehicle charging session details."""
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
        """SoC-based category for prioritization."""
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

    def __init__(self, stations: List[ChargingStation], grid_limit_kw: float):
        self.stations = stations
        self.queue = WaitingQueue()
        self.grid_limit_kw = grid_limit_kw
        self.power_log: List[float] = []
        self.completed_sessions: List[EV] = []
        self.queued_count: int = 0

    def get_baseline_at_minute(self, minute: int) -> float:
        """Time-dependent baseline (06:00-22:00: 200 kW, else: 50 kW)."""
        hour = (minute // 60) % 24
        return 200.0 if 6 <= hour < 22 else 50.0

    @abstractmethod
    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        """Return allocation {station_id: kw}."""
        pass

    def dispatch_waiting_evs(self, minute: int) -> None:
        """Assign queued EVs to free stations."""
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
        allocations = self.allocate_power(baseline_now)

        for station in self.stations:
            station.allocated_power_kw = allocations.get(station.station_id, 0.0)
            station.step(duration_minutes=1.0)

        finished_evs = self.collect_finished_evs()
        for ev in finished_evs:
            ev.departure_minute = minute
            self.completed_sessions.append(ev)

        total_power = baseline_now + sum(allocations.values())
        self.power_log.append(total_power)


class UnmanagedController(GridController):
    """Uncontrolled: no budgeting."""

    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        allocations = {}
        for station in self.stations:
            allocations[station.station_id] = station.effective_max_power_kw()
        return allocations


class ManagedController(GridController):
    """Controlled: Smart Queue & SoC Priority."""

    MIN_POWER_GUARANTEE_KW = 50.0

    def allocate_power(self, baseline_kw: float) -> Dict[str, float]:
        """
        Smart allocation algorithm:
        1. Low SoC (< 70%): Guarantee minimum 50 kW, else queue
        2. High SoC (>= 70%): Get remaining budget (fair-share)
        """
        occupied_stations = [s for s in self.stations if not s.is_available()]

        if not occupied_stations:
            return {s.station_id: 0.0 for s in self.stations}

        available_budget = max(0, self.grid_limit_kw - baseline_kw)

        # Separate by SoC
        low_soc = [s for s in occupied_stations if s.current_ev.soc_category == "low_soc"]
        high_soc = [s for s in occupied_stations if s.current_ev.soc_category == "high_soc"]

        allocations = {}
        remaining = available_budget

        # Phase 1: Allocate to low_soc with minimum guarantee
        unmet = []
        for station in low_soc:
            max_power = station.effective_max_power_kw()
            min_power = min(self.MIN_POWER_GUARANTEE_KW, max_power)

            if remaining >= min_power:
                alloc = min(max_power, remaining)
                allocations[station.station_id] = alloc
                remaining -= alloc
            else:
                unmet.append(station)
                allocations[station.station_id] = 0.0

        # Unplug stations that can't meet minimum
        for station in unmet:
            ev = station.unplug()
            ev.state = EVState.WAITING
            self.queue.enqueue(ev)
            self.queued_count += 1

        # Phase 2: Fair-share for high_soc
        if remaining > 0 and high_soc:
            for station in high_soc:
                max_power = station.effective_max_power_kw()
                fair_share = remaining / len(high_soc)
                alloc = min(max_power, fair_share)
                allocations[station.station_id] = alloc
                remaining -= alloc

        # Fill remaining
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
        baseline_avg = 125  # Average of 50 and 200

        metrics = MetricsSummary(
            peak_power_kw=float(power_arr.max()),
            overload_minutes=int((power_arr > self.controller.grid_limit_kw).sum()),
            total_overload_kwh=float(
                (power_arr[power_arr > self.controller.grid_limit_kw] - self.controller.grid_limit_kw).sum() / 60.0
                if any(power_arr > self.controller.grid_limit_kw) else 0.0
            ),
            avg_delay_minutes=float(np.mean([ev.wait_time_minutes for ev in self.controller.completed_sessions]))
                if self.controller.completed_sessions else 0.0,
            evs_completed=len(self.controller.completed_sessions),
            evs_queued=self.controller.queued_count,
            protected_capacity_percent=100.0 if max(power_arr) <= self.controller.grid_limit_kw else
                                       (self.controller.grid_limit_kw / max(power_arr)) * 100.0
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
        )


# ==============================================================================
# Visualization
# ==============================================================================

class ExecutiveDashboard:
    """Executive-level dashboard with 3 key charts."""

    @staticmethod
    def create_dashboard(result_unmanaged: SimulationResult, result_managed: SimulationResult) -> None:
        """Create 3-panel executive dashboard."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("EV Charging Load Balancing — Executive Summary", fontsize=16, fontweight="bold", y=0.98)

        # Panel 1: Grid Load (Filled Area)
        ax1 = plt.subplot(2, 2, 1)
        minutes = np.arange(1440)
        hours = minutes / 60

        ax1.fill_between(hours, 0, result_unmanaged.power_timeseries, alpha=0.3, color="red", label="Unmanaged Load")
        ax1.plot(hours, result_managed.power_timeseries, linewidth=2.5, color="darkgreen", label="Managed Load")
        ax1.axhline(400, color="red", linestyle="--", linewidth=2.5, label="Grid Limit (400 kW)")
        ax1.set_xlabel("Time of Day (hours)", fontsize=11)
        ax1.set_ylabel("Grid Load (kW)", fontsize=11)
        ax1.set_title("Panel 1: Grid Load Profile (24-Hour)", fontsize=12, fontweight="bold")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 2))
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Wait Time Categories
        ax2 = plt.subplot(2, 2, 2)

        # Categorize wait times
        managed_sessions = result_managed.vehicle_sessions
        no_wait = sum(1 for s in managed_sessions if s.wait_time_minutes == 0)
        short_wait = sum(1 for s in managed_sessions if 0 < s.wait_time_minutes <= 15)
        long_wait = sum(1 for s in managed_sessions if s.wait_time_minutes > 15)

        categories = ["No Wait\n(0 min)", "Acceptable\n(1-15 min)", "Long Wait\n(15+ min)"]
        values = [no_wait, short_wait, long_wait]
        colors_cat = ["#2ecc71", "#f39c12", "#e74c3c"]

        bars = ax2.bar(categories, values, color=colors_cat, edgecolor="black", linewidth=1.5, width=0.6)
        ax2.set_ylabel("Number of Vehicles", fontsize=11)
        ax2.set_title("Panel 2: Vehicle Queue Wait Distribution", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, max(values) * 1.15)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{int(val)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Model Performance
        ax3 = plt.subplot(2, 2, 3)

        # Group by model and get average charge time
        model_times = {}
        for session in managed_sessions:
            model = session.model_name
            if model not in model_times:
                model_times[model] = []
            model_times[model].append(session.charge_time_minutes)

        models = sorted(model_times.keys())
        avg_times = [np.mean(model_times[m]) for m in models]

        # Short labels
        model_labels = [m.split()[0] for m in models]  # First word only

        y_pos = np.arange(len(models))
        bars = ax3.barh(y_pos, avg_times, color="steelblue", edgecolor="black", linewidth=1)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(model_labels, fontsize=10)
        ax3.set_xlabel("Average Charge Time (minutes)", fontsize=11)
        ax3.set_title("Panel 3: Charging Performance by Model", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_times)):
            ax3.text(val + 1, i, f"{val:.0f} min", va="center", fontsize=9)

        # Panel 4: Metrics Summary (Text)
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis("off")

        summary_text = f"""
MANAGED SCENARIO PERFORMANCE

┌────────────────────────────────────┐
│ GRID PROTECTION                     │
├────────────────────────────────────┤
│ Protected Capacity: {result_managed.metrics_summary.protected_capacity_percent:.1f}%
│ Peak Load: {result_managed.metrics_summary.peak_power_kw:.0f} kW (vs {result_unmanaged.metrics_summary.peak_power_kw:.0f} kW unmanaged)
│ Overload Events: {result_managed.metrics_summary.overload_minutes} minutes (vs {result_unmanaged.metrics_summary.overload_minutes} min)
│
├────────────────────────────────────┤
│ ALGORITHM IMPACT                    │
├────────────────────────────────────┤
│ EVs Served: {result_managed.metrics_summary.evs_completed} vehicles
│ Avg Delay: {result_managed.metrics_summary.avg_delay_minutes:.1f} minutes
│ Peak Avoidance: {result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw:.0f} kW
│ Overload Prevented: {result_unmanaged.metrics_summary.overload_minutes - result_managed.metrics_summary.overload_minutes} minutes
└────────────────────────────────────┘
"""

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.savefig("F:/LoadBalancing/executive_dashboard.png", dpi=150, bbox_inches="tight")
        print("✓ Executive dashboard saved: executive_dashboard.png")
        plt.show()

    @staticmethod
    def print_executive_summary(result_unmanaged: SimulationResult, result_managed: SimulationResult) -> None:
        """Print executive summary to console."""
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY — EV LOAD BALANCING OPTIMIZATION")
        print("="*70)

        print(f"\n📊 GRID PROTECTION METRICS")
        print(f"  • Protected Capacity:        {result_managed.metrics_summary.protected_capacity_percent:>6.1f}% (Target: 100%)")
        print(f"  • Prevented Peak Overload:   {result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw:>6.0f} kW reduction")
        print(f"  • Eliminated Overload Time:  {result_unmanaged.metrics_summary.overload_minutes:>6d} → {result_managed.metrics_summary.overload_minutes:>3d} minutes")

        print(f"\n🚗 SERVICE DELIVERY")
        print(f"  • Total EVs Served:          {result_managed.metrics_summary.evs_completed:>6d} vehicles")
        print(f"  • Algorithm-Induced Delay:   {result_managed.metrics_summary.avg_delay_minutes:>6.1f} min (acceptable trade-off)")

        print(f"\n⚡ ALGORITHM EFFICIENCY")
        peak_reduction = ((result_unmanaged.metrics_summary.peak_power_kw - result_managed.metrics_summary.peak_power_kw)
                         / result_unmanaged.metrics_summary.peak_power_kw * 100)
        print(f"  • Peak Load Reduction:       {peak_reduction:>6.1f}%")
        print(f"  • Overload Prevention:       {result_unmanaged.metrics_summary.total_overload_kwh - result_managed.metrics_summary.total_overload_kwh:>6.1f} kWh avoided")

        print("\n" + "="*70 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main(generate_new: bool = False):
    """Run full simulation."""
    print("Initializing EV Load Balancing Simulation v2...")

    # Dataset management
    dataset_file = "F:/LoadBalancing/dataset.json"
    if not generate_new and os.path.exists(dataset_file):
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded dataset: {len(data['vehicles'])} vehicles")
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
        print("✓ Generating new dataset...")
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
        print(f"✓ Dataset saved: {len(vehicles_data)} vehicles")

    # Stations
    stations_a = [
        ChargingStation("S1", StationType.ULTRA_FAST, 350.0),
        ChargingStation("S2", StationType.FAST, 150.0),
        ChargingStation("S3", StationType.FAST, 150.0),
        ChargingStation("S4", StationType.STANDARD, 50.0),
        ChargingStation("S5", StationType.STANDARD, 50.0),
    ]
    stations_b = [copy.deepcopy(s) for s in stations_a]

    # Run scenarios
    print("\n⚙️  Running Scenario A (Unmanaged)...")
    schedule_a = copy.deepcopy(arrival_schedule)
    controller_a = UnmanagedController(stations_a, 400.0)
    result_a = Simulation(controller_a, schedule_a).run()
    print(f"   ✓ Complete: Peak={result_a.metrics_summary.peak_power_kw:.0f}kW, Overload={result_a.metrics_summary.overload_minutes}min")

    print("⚙️  Running Scenario B (Managed)...")
    schedule_b = copy.deepcopy(arrival_schedule)
    controller_b = ManagedController(stations_b, 400.0)
    result_b = Simulation(controller_b, schedule_b).run()
    print(f"   ✓ Complete: Peak={result_b.metrics_summary.peak_power_kw:.0f}kW, Overload={result_b.metrics_summary.overload_minutes}min")

    # Output
    ExecutiveDashboard.print_executive_summary(result_a, result_b)
    ExecutiveDashboard.create_dashboard(result_a, result_b)

    # Validation
    assert result_b.metrics_summary.peak_power_kw <= 400.0, "Managed peak should not exceed 400 kW"
    print("✓ All validations passed")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="EV Load Balancing Simulation v2")
    parser.add_argument("--generate-new", action="store_true", help="Generate new dataset")
    args = parser.parse_args()

    main(generate_new=args.generate_new)
