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
                    ArrivalPattern(mean_hour=9.5,  std_minutes=25, fraction=0.25),
                    ArrivalPattern(mean_hour=13.0, std_minutes=20, fraction=0.25),
                    ArrivalPattern(mean_hour=18.5, std_minutes=40, fraction=0.50),
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
                    ArrivalPattern(mean_hour=8.5,  std_minutes=20, fraction=0.70),
                    ArrivalPattern(mean_hour=13.0, std_minutes=20, fraction=0.30),
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
                    ArrivalPattern(mean_hour=15.0, std_minutes=45, fraction=0.40),
                    ArrivalPattern(mean_hour=20.0, std_minutes=30, fraction=0.60),
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
                    ArrivalPattern(mean_hour=7.5,  std_minutes=12, fraction=0.34),
                    ArrivalPattern(mean_hour=15.5, std_minutes=12, fraction=0.33),
                    ArrivalPattern(mean_hour=23.5, std_minutes=12, fraction=0.33),
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
                    ArrivalPattern(mean_hour=6.5,  std_minutes=35, fraction=0.50),
                    ArrivalPattern(mean_hour=17.5, std_minutes=35, fraction=0.50),
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
    def stress_test() -> ScenarioConfig:
        """Stres testi — yaz sıcağı + yoğun EV + küçük trafo.

        Amaç: Sıcaklık etkisini, termal baskıyı ve batarya yaşlanmasını
        gözlemlemek için maksimum zorlayıcı senaryo.
          - 60 araç / gün (normal AVM'nin ~1.7 katı)
          - ambient_temp_max = 40°C (yaz öğlesi)
          - Trafo küçük tutulmuş (800 kVA) → yük faktörü yüksek
          - Araçlar düşük SoC'la gelir (0.05–0.25) → uzun şarj süresi
          - target_soc = 0.90 → daha fazla enerji talebi
          - Ani geliş dalgaları → ramp stress
        """
        from models import (
            ScenarioConfig, EnvironmentProfile, GridConfig, FleetProfile,
            ArrivalPattern, ChargingStation, StationType, StationLayout, EVModel
        )
        return ScenarioConfig(
            name="Stres_Testi",
            environment=EnvironmentProfile(
                name="Stres_Testi",
                base_min_kw=200.0, base_max_kw=550.0,
                operation_start_hour=6.0, operation_duration_hours=16.0,
                morning_peak_hour=9.0,  morning_peak_kw=120.0, morning_peak_width=0.8,
                evening_peak_hour=18.0, evening_peak_kw=160.0, evening_peak_width=1.0,
                noise_kw=30.0, load_min_kw=180.0, load_max_kw=850.0,
                ambient_temp_min=26.0,
                ambient_temp_max=40.0,
                ambient_temp_peak_hour=14.0,
            ),
            grid=GridConfig(
                trafo_kva=1000,
                evening_peak_kw=620.0,
                peak_start_hour=17.0,
                peak_end_hour=22.0,
                delta_theta_oil_rated=55.0,
                delta_theta_hs_rated=25.0,
                tau_oil_min=120.0,
            ),
            fleet=FleetProfile(
                daily_ev_count=60,
                initial_soc_min=0.05,
                initial_soc_max=0.25,
                target_soc=0.90,
                ev_models=[
                    EVModel("Tesla Model Y", 75.0, 250.0, 0.30),
                    EVModel("Porsche Taycan", 93.4, 270.0, 0.25),
                    EVModel("Togg T10X", 88.5, 150.0, 0.20),
                    EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
                    EVModel("MG4 Standard", 51.0, 117.0, 0.10),
                ],
                arrival_patterns=[
                    ArrivalPattern(mean_hour=8.0,  std_minutes=12, fraction=0.15),
                    ArrivalPattern(mean_hour=12.5, std_minutes=12, fraction=0.15),
                    ArrivalPattern(mean_hour=17.5, std_minutes=12, fraction=0.70),
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


class MonthlyDataGenerator:
    """30 günlük senaryo verisi üretir — haftaiçi/haftasonu farklı profil.

    Haziran 2026 takvimi baz alınır: 1 Haziran = Pazartesi.
    Her gün bağımsız rastgele tohumla üretilir; günlük araç sayısı,
    geliş zamanları, model dağılımı ve baz yük profili değişir.
    """

    WEEKDAY_NAMES = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    _START_WEEKDAY = 0  # 1 Haziran 2026 = Pazartesi

    def __init__(self, config, n_days: int = 30):
        self.config = config
        self.n_days = n_days

    # ── Takvim yardımcıları ──────────────────────────────────────

    def _weekday(self, day_idx: int) -> int:
        return (self._START_WEEKDAY + day_idx) % 7

    def _is_weekend(self, day_idx: int) -> bool:
        return self._weekday(day_idx) >= 5  # Cumartesi=5, Pazar=6

    # ── Günlük araç sayısı ───────────────────────────────────────

    def _ev_count(self, day_idx: int, rng) -> int:
        base = self.config.fleet.daily_ev_count
        wday = self._weekday(day_idx)
        name = self.config.name.lower()

        if "avm" in name or "stres" in name:
            w = [0.88, 0.84, 0.90, 0.94, 1.14, 1.50, 1.36]
        elif "ofis" in name:
            w = [0.92, 1.12, 1.14, 1.08, 0.86, 0.07, 0.03]
        elif "hastane" in name:
            w = [1.00, 1.03, 1.03, 1.00, 0.97, 0.88, 0.78]
        elif "otel" in name:
            w = [0.84, 0.80, 0.83, 0.88, 1.18, 1.42, 1.32]
        elif "havaliman" in name:
            w = [0.90, 0.88, 0.92, 0.96, 1.18, 1.32, 1.22]
        else:
            w = [0.90, 0.90, 0.92, 0.90, 1.00, 1.12, 1.06]

        return max(3, int(base * w[wday] * rng.uniform(0.92, 1.08)))

    # ── Günlük geliş desenleri ───────────────────────────────────

    def _arrival_patterns(self, day_idx: int):
        from models import ArrivalPattern
        name    = self.config.name.lower()
        is_wknd = self._is_weekend(day_idx)
        base_pt = self.config.fleet.arrival_patterns

        if not is_wknd:
            return base_pt

        if "avm" in name or "stres" in name:
            return [
                ArrivalPattern(10.5, 48, 0.14),
                ArrivalPattern(14.0, 52, 0.46),
                ArrivalPattern(17.5, 42, 0.40),
            ]
        elif "ofis" in name:
            return [ArrivalPattern(11.0, 90, 1.0)]
        elif "otel" in name:
            return [
                ArrivalPattern(13.0, 65, 0.25),
                ArrivalPattern(19.5, 52, 0.75),
            ]
        elif "havaliman" in name:
            return [
                ArrivalPattern(8.0, 55, 0.45),
                ArrivalPattern(18.0, 55, 0.55),
            ]
        else:
            return [ArrivalPattern(p.mean_hour + 1.0, p.std_minutes + 15, p.fraction)
                    for p in base_pt]

    # ── Günlük baz yük profili ───────────────────────────────────

    def _bg_load(self, day_idx: int, rng) -> np.ndarray:
        env     = self.config.environment
        name    = self.config.name.lower()
        wday    = self._weekday(day_idx)
        is_wknd = self._is_weekend(day_idx)
        hrs     = np.arange(1440) / 60.0

        if is_wknd:
            if "ofis" in name:
                base_scale = rng.uniform(0.11, 0.22)
                op_start   = env.operation_start_hour
                op_dur     = env.operation_duration_hours * 0.14
                m_peak_kw  = env.morning_peak_kw * rng.uniform(0.06, 0.12)
                e_peak_kw  = env.evening_peak_kw * rng.uniform(0.04, 0.09)
                m_peak_h   = env.morning_peak_hour
                e_peak_h   = env.evening_peak_hour
            elif "avm" in name or "stres" in name:
                base_scale = rng.uniform(0.82, 0.96)
                op_start   = env.operation_start_hour + rng.uniform(1.2, 2.2)
                op_dur     = max(0.5, env.operation_duration_hours - rng.uniform(0.5, 1.5))
                m_peak_kw  = env.morning_peak_kw * rng.uniform(0.66, 0.86)
                e_peak_kw  = env.evening_peak_kw * rng.uniform(0.88, 1.06)
                m_peak_h   = env.morning_peak_hour + rng.uniform(1.5, 2.8)
                e_peak_h   = env.evening_peak_hour - rng.uniform(0.5, 1.5)
            else:
                base_scale = rng.uniform(0.78, 0.93)
                op_start   = env.operation_start_hour + rng.uniform(0.5, 1.5)
                op_dur     = max(0.5, env.operation_duration_hours - rng.uniform(1.0, 2.0))
                m_peak_kw  = env.morning_peak_kw * rng.uniform(0.70, 0.92)
                e_peak_kw  = env.evening_peak_kw * rng.uniform(0.75, 0.96)
                m_peak_h   = env.morning_peak_hour + rng.uniform(0.5, 1.5)
                e_peak_h   = env.evening_peak_hour - rng.uniform(0.5, 1.0)
        else:
            if "ofis" in name:
                dw = [0.88, 1.07, 1.10, 1.06, 0.90, 0.12, 0.05][wday]
            elif "avm" in name:
                dw = [0.88, 0.84, 0.90, 0.94, 1.10, 1.0, 1.0][wday]
            else:
                dw = 1.0
            base_scale = dw * rng.uniform(0.92, 1.08)
            op_start   = env.operation_start_hour + rng.uniform(-0.5, 0.5)
            op_dur     = max(0.5, env.operation_duration_hours + rng.uniform(-0.8, 0.8))
            m_peak_kw  = env.morning_peak_kw * dw * rng.uniform(0.88, 1.12)
            e_peak_kw  = env.evening_peak_kw * dw * rng.uniform(0.88, 1.12)
            m_peak_h   = env.morning_peak_hour + rng.uniform(-0.4, 0.4)
            e_peak_h   = env.evening_peak_hour + rng.uniform(-0.4, 0.4)

        base = (env.base_min_kw * base_scale
                + (env.base_max_kw - env.base_min_kw) * base_scale
                * np.clip(0.5 * (1 - np.cos(
                    np.pi * np.clip(hrs - op_start, 0, op_dur) / op_dur
                )), 0, 1))
        peak_m = m_peak_kw * np.exp(-0.5 * ((hrs - m_peak_h) / env.morning_peak_width) ** 2)
        peak_e = e_peak_kw * np.exp(-0.5 * ((hrs - e_peak_h) / env.evening_peak_width) ** 2)
        return np.clip(base + peak_m + peak_e + rng.normal(0, env.noise_kw, 1440),
                       env.load_min_kw, env.load_max_kw)

    # ── Ana üretici ──────────────────────────────────────────────

    def generate(self, rng) -> list:
        """30 günlük veri listesi döner.

        Her eleman: day_index, day_number, date, day_name, is_weekend,
                    vehicle_count, vehicles (list), bg_load (list[float]).
        """
        result = []
        fleet  = self.config.fleet
        probs  = [m.probability for m in fleet.ev_models]

        for day_idx in range(self.n_days):
            day_rng  = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
            ev_count = self._ev_count(day_idx, day_rng)
            patterns = self._arrival_patterns(day_idx)

            remaining = ev_count
            counts    = []
            for i, p in enumerate(patterns):
                n = int(ev_count * p.fraction) if i < len(patterns) - 1 else remaining
                remaining -= n
                counts.append(max(0, n))

            arrs = np.concatenate([
                day_rng.normal(p.mean_hour * 60, p.std_minutes, counts[i])
                for i, p in enumerate(patterns)
            ])
            day_rng.shuffle(arrs)

            vehicles = []
            for i, m_val in enumerate(np.clip(arrs, 1, 1438).astype(int)):
                model  = day_rng.choice(fleet.ev_models, p=probs)
                soc_lo = max(0.05, fleet.initial_soc_min + day_rng.uniform(-0.04, 0.04))
                soc_hi = min(0.55, fleet.initial_soc_max + day_rng.uniform(-0.04, 0.04))
                soc_hi = max(soc_lo + 0.05, soc_hi)
                vehicles.append({
                    "session_id":           f"G{day_idx+1:02d}_{i+1:03d}",
                    "model_name":           model.model_name,
                    "battery_capacity_kwh": float(model.battery_capacity_kwh),
                    "max_dc_power_kw":      float(model.max_dc_power_kw),
                    "arrival_minute":       int(m_val),
                    "initial_soc":          float(day_rng.uniform(soc_lo, soc_hi)),
                })

            bg = self._bg_load(day_idx, day_rng)
            result.append({
                "day_index":     day_idx,
                "day_number":    day_idx + 1,
                "date":          f"2026-06-{day_idx+1:02d}",
                "day_name":      self.WEEKDAY_NAMES[self._weekday(day_idx)],
                "is_weekend":    bool(self._is_weekend(day_idx)),
                "vehicle_count": ev_count,
                "vehicles":      vehicles,
                "bg_load":       [round(float(v), 2) for v in bg],
            })
        return result
