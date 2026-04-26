#!/usr/bin/env python3
"""Veri Üreticiler - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from models import (
    FleetProfile, EnvironmentProfile, ScenarioConfig,
    ArrivalPattern, EV
)


class ArrivalGenerator:
    def __init__(self, fleet: FleetProfile):
        self.fleet = fleet

    def generate_arrivals(self, rng: np.random.Generator) -> Dict[int, List]:
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

        schedule: Dict[int, List] = {}
        probs = [m.probability for m in self.fleet.ev_models]
        for i, m_val in enumerate(np.clip(arrs, 0, 1439)):
            minute_idx = int(m_val)
            model = rng.choice(self.fleet.ev_models, p=probs)
            if minute_idx not in schedule:
                schedule[minute_idx] = []
            schedule[minute_idx].append(EV(
                session_id=f"EV_{i+1:02d}",
                model_name=model.model_name,
                battery_capacity_kwh=model.battery_capacity_kwh,
                max_dc_power_kw=model.max_dc_power_kw,
                arrival_minute=minute_idx,
                initial_soc=rng.uniform(self.fleet.initial_soc_min, self.fleet.initial_soc_max),
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
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
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
                ev_models=[
                    EVModel("Togg T10X", 88.5, 150.0, 0.25),
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
                    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                    EVModel("Renault Megane", 60.0, 130.0, 0.10),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
                ],
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
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
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
                ev_models=[
                    EVModel("Togg T10X", 88.5, 150.0, 0.25),
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
                    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                    EVModel("Renault Megane", 60.0, 130.0, 0.10),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
                ],
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
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
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
                ev_models=[
                    EVModel("Togg T10X", 88.5, 150.0, 0.25),
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
                    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                    EVModel("Renault Megane", 60.0, 130.0, 0.10),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
                ],
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
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
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
                ev_models=[
                    EVModel("Togg T10X", 88.5, 150.0, 0.25),
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
                    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                    EVModel("Renault Megane", 60.0, 130.0, 0.10),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
                ],
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
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
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
                ev_models=[
                    EVModel("Togg T10X", 88.5, 150.0, 0.25),
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
                    EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                    EVModel("Renault Megane", 60.0, 130.0, 0.10),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
                ],
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
