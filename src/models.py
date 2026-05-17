#!/usr/bin/env python3
"""Veri Modelleri - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
import math
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
    soc_taper_start: float = 0.70       # CC-CV geçiş SoC eşiği
    soc_taper_min_frac: float = 0.05    # SoC=1.0'de max gücün minimum fraksiyonu


@dataclass
class GridLimitPolicy:
    # 1600 kVA trafo × 0.90 güç faktörü × 0.88 safety margin
    trafo_max_kw: float = round(1600 * 0.90 * 0.88, 1)   # → 1267.2 kW
    evening_peak_kw: float = 950.0   # 17-22 pik: baz yük ~700 kW, EV'ye dar bütçe
    peak_start_min: int = 1020  # 17:00
    peak_end_min: int = 1320    # 22:00


class DynamicGridLimitPolicy(GridLimitPolicy):
    """IEC 60076-7 sadeleştirilmiş termal model ile dinamik trafo güç tavanı.

    Fizik:
        θ_oil(t+Δt) = θ_oil(t) + (Δt/τ_oil) · [Δθ_oil_rated·K² − (θ_oil(t)−θ_amb)]
        θ_hs(t)     = θ_oil(t) + Δθ_hs_rated · K^1.6
        V(t)        = 2^((θ_hs(t) − 98) / 6)   — yaşlanma hızı

    Kaynak: IEC 60076-7 Ed.2 (2018), Annex E — ONAN dağıtım trafoları.
    """

    def __init__(self, trafo_max_kw: float, evening_peak_kw: float,
                 peak_start_min: int, peak_end_min: int,
                 grid_config: GridConfig, environment: EnvironmentProfile,
                 theta_hs_target: float = 98.0):
        super().__init__(trafo_max_kw, evening_peak_kw, peak_start_min, peak_end_min)
        self.grid_config   = grid_config
        self.environment   = environment
        self.theta_hs_target = theta_hs_target

        # Termal parametreler (GridConfig'den)
        self._dtheta_oil = grid_config.delta_theta_oil_rated
        self._dtheta_hs  = grid_config.delta_theta_hs_rated
        self._tau_oil    = grid_config.tau_oil_min
        self._s_rated    = grid_config.s_rated_kw

        # Hard limits
        self._hard_ceiling = grid_config.trafo_kva * grid_config.power_factor * 1.5  # IEC kısa süreli
        self._hard_floor_base = 80.0  # en az bir hızlı soket açık

        # Durum değişkenleri
        amb0 = self._ambient_temp(0)
        self.theta_oil = amb0 + 5.0   # soğuk başlangıç
        self.theta_hs  = self.theta_oil + self._dtheta_hs * 0.0  # K=0 başlangıçta
        self.aging_rate = 0.0
        self.aging_integral = 0.0
        self.peak_hotspot = 0.0

        # Geçmiş kayıtları (dashboard için)
        self.theta_oil_log: List[float] = []
        self.theta_hs_log: List[float] = []
        self.aging_rate_log: List[float] = []
        self.dynamic_limit_log: List[float] = []
        self.dielectric_stress_log: List[float] = []  # |ΔP/S_rated| ani yük geçiş stresi
        self._prev_total_load: float = 0.0

    def _ambient_temp(self, minute: int) -> float:
        """Sentetik günlük ortam sıcaklığı profili (sinüzoidal)."""
        hour = (minute % 1440) / 60.0
        env = self.environment
        avg = (env.ambient_temp_min + env.ambient_temp_max) / 2.0
        amp = (env.ambient_temp_max - env.ambient_temp_min) / 2.0
        return avg - amp * math.cos(2 * math.pi * (hour - env.ambient_temp_peak_hour) / 24.0)

    def update(self, minute: int, total_load_kw: float):
        """Her dakika çağrılır — termal durumu günceller.
        total_load_kw: bg_load + EV şarj toplam gücü.
        """
        # Yük faktörü K = S(t) / S_rated
        K = max(0.0, total_load_kw / max(self._s_rated, 1.0))
        theta_amb = self._ambient_temp(minute)
        dt = 1.0  # Δt = 1 dk

        # Yağ sıcaklığı ODE (Euler)
        dtheta = (dt / self._tau_oil) * (
            self._dtheta_oil * K ** 2 - (self.theta_oil - theta_amb)
        )
        self.theta_oil += dtheta

        # Hot-spot sıcaklığı (cebirsel)
        self.theta_hs = self.theta_oil + self._dtheta_hs * (K ** 1.6)
        self.peak_hotspot = max(self.peak_hotspot, self.theta_hs)

        # IEC 60076-7 yaşlanma hızı — non-thermally upgraded Kraft paper
        self.aging_rate = 2.0 ** ((self.theta_hs - 98.0) / 6.0)
        self.aging_integral += self.aging_rate / 1440.0  # günlük normalize

        # Dielektrik stres — ani yük geçişi (|ΔP / S_rated|)
        # Hızlı akım değişimleri → elektromanyetik kuvvet + geçici gerilim dalgalanması
        dielectric_stress = abs(total_load_kw - self._prev_total_load) / max(self._s_rated, 1.0)
        self.dielectric_stress_log.append(dielectric_stress)
        self._prev_total_load = total_load_kw

        # Log
        self.theta_oil_log.append(self.theta_oil)
        self.theta_hs_log.append(self.theta_hs)
        self.aging_rate_log.append(self.aging_rate)

    def current_limit_kw(self, minute: int, bg_load_kw: float) -> float:
        """Dinamik güç tavanı — θ_hs_target'i aşmamak için üç K_max kontrolü:
        1) Anlık:       θ_oil(t) + Δθ_hs·K^1.6 ≤ target        (yağ sıcaksa koru)
        2) SS-peak:     θ_amb_max + (Δθ_oil+Δθ_hs)·K² ≤ target  (gün boyunca koruma)
        3) SS-current:  θ_amb(t) + (Δθ_oil+Δθ_hs)·K² ≤ target   (anlık SS)
        En kısıtlayıcı K_max seçilir.
        """
        theta_amb = self._ambient_temp(minute)
        theta_amb_max = self.environment.ambient_temp_max

        # ── K_max_inst: mevcut yağ sıcaklığına göre ─────────────────────────
        headroom_inst = self.theta_hs_target - self.theta_oil
        if headroom_inst <= 0:
            K_max_inst = 0.0
        else:
            K_max_inst = (headroom_inst / max(self._dtheta_hs, 0.1)) ** (1.0 / 1.6)

        # ── K_max_ss_peak: günün MAX ortam sıcaklığına göre (en kötü durum) ─
        # Bu tavan tüm gün sabit → sabah yüksek K'nın öğleden sonra aşım
        # yapmasını önler.
        headroom_ss_peak = self.theta_hs_target - theta_amb_max
        if headroom_ss_peak <= 0:
            K_max_ss_peak = 0.0
        else:
            K_max_ss_peak = (headroom_ss_peak / (self._dtheta_oil + self._dtheta_hs)) ** 0.5

        # ── K_max_ss_now: anlık ortam sıcaklığına göre ──────────────────────
        headroom_ss_now = self.theta_hs_target - theta_amb
        if headroom_ss_now <= 0:
            K_max_ss_now = 0.0
        else:
            K_max_ss_now = (headroom_ss_now / (self._dtheta_oil + self._dtheta_hs)) ** 0.5

        # En kısıtlayıcı
        K_max = min(K_max_inst, K_max_ss_peak, K_max_ss_now)

        if K_max <= 0:
            p_max = bg_load_kw + self._hard_floor_base
        else:
            p_max = K_max * self._s_rated
            p_max = min(p_max, self._hard_ceiling)
            p_max = max(p_max, bg_load_kw + self._hard_floor_base)

        return p_max


@dataclass
class MetricsSummary:
    peak_power_kw: float
    overload_minutes: int
    total_overload_kwh: float
    avg_delay_minutes: float
    evs_completed: int
    protected_capacity_percent: float
    avg_grid_limit_kw: float
    # IEC 60076-7 termal metrikler (termal mod kapalıysa 0.0)
    peak_hotspot_temp_c: float = 0.0
    daily_aging_integral: float = 0.0
    avg_marginal_aging_cost: float = 0.0


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
    # CC-CV taper parametreleri
    soc_taper_start: float = 0.70
    soc_taper_min_frac: float = 0.05
    # Batarya termal parametreleri
    battery_temp_c: float = 25.0      # başlangıç batarya sıcaklığı (°C)
    tau_battery_min: float = 30.0     # termal zaman sabiti (dk) — tipik hava soğutmalı paket
    # Kümülatif yaşlanma integralleri (şarj süresi boyunca izlenir)
    elec_deg_integral: float = 0.0    # elektrolit bozulma kümülatifi (SoC+sıcaklık bazlı)
    mech_stress_integral: float = 0.0 # mekanik stres kümülatifi (C-rate bazlı)

    def __post_init__(self):
        self.current_soc = self.initial_soc

    @property
    def energy_needed_kwh(self) -> float:
        return max(0.0, (self.target_soc - self.current_soc) * self.battery_capacity_kwh)

    @property
    def is_satisfied(self) -> bool:
        return self.current_soc >= self.target_soc

    @property
    def max_acceptable_power_kw(self) -> float:
        """CC-CV fiziksel güç sınırı — SoC ve batarya sıcaklığına göre.

        İki segmentli SoC taper:
          SoC < soc_taper_start (0.70)  → tam güç
          0.70 ≤ SoC < 0.80            → 1.0 → 0.40 (CC-CV geçiş bölgesi)
          0.80 ≤ SoC ≤ 1.00            → 0.40 → soc_taper_min_frac (CV bölgesi)
        Kaynak: NMC CC-CV karakteristiği (Ecker et al., 2014).

        BMS termal derating:
          T_bat > 40°C  → lineer azalma, T_bat=50°C'de %50 güç (aşırı ısı koruması)
          T_bat < 5°C   → lineer azalma, T_bat=-10°C'de %20 güç (lityum kaplama riski)
        Kaynak: Plett, "Battery Management Systems Vol.2" (2015), Bölüm 5.
        """
        soc = self.current_soc

        # SoC taper
        if soc < self.soc_taper_start:
            soc_frac = 1.0
        else:
            mid_soc = 0.80
            if soc < mid_soc:
                t = (soc - self.soc_taper_start) / (mid_soc - self.soc_taper_start)
                soc_frac = 1.0 - 0.60 * t          # 1.0 → 0.40
            else:
                t = (soc - mid_soc) / (1.0 - mid_soc)
                soc_frac = 0.40 - (0.40 - self.soc_taper_min_frac) * t  # 0.40 → 0.05
            soc_frac = max(soc_frac, self.soc_taper_min_frac)

        # BMS termal derating
        T = self.battery_temp_c
        if T > 40.0:
            # Yüksek sıcaklık: 40°C → 1.0, 50°C → 0.50
            thermal_frac = max(0.50, 1.0 - 0.05 * (T - 40.0))
        elif T < 5.0:
            # Düşük sıcaklık: 5°C → 1.0, -10°C → 0.20
            thermal_frac = max(0.20, 1.0 - 0.053 * (5.0 - T))
        else:
            thermal_frac = 1.0

        return self.max_dc_power_kw * soc_frac * thermal_frac

    def update_battery_temp(self, ambient_temp_c: float, power_kw: float, dt: float = 1.0):
        """Batarya sıcaklığını güncelle — basit birinci dereceden termal model.

        Isı kaynakları:
          - Joule ısısı: Q_joule [kW] = c_rate² × R_int_norm × kapasite
          - Ortam ile ısı alışverişi: thermal conductance G_th [kW/°C] ile

        Parametreler:
          R_int_norm = 0.02 Ω·kWh  (NMC normalize iç direnç, Schmalstieg 2014)
          G_th       = 0.20 kW/°C  (hava soğutmalı tipik EV paketi)
        → 1C şarjda ΔT_steady ≈ 7-8°C (ölçülen 5-10°C aralığında)

        Kaynak: Schmalstieg et al., J. Power Sources 257 (2014) 325–334.
                Plett, Battery Management Systems Vol.2 (2015), Bölüm 5.
        """
        c_rate = power_kw / max(self.battery_capacity_kwh, 0.1)
        r_int_norm = 0.02    # Ω·kWh — NMC normalize iç direnç
        g_th       = 0.20    # kW/°C — paket termal iletkenlik (hava soğutma)
        heat_kw = c_rate ** 2 * r_int_norm * self.battery_capacity_kwh
        # Steady-state hedef: ortam + Joule / termal iletkenlik
        target_temp = ambient_temp_c + heat_kw / g_th
        # Birinci dereceden ODE (Euler)
        self.battery_temp_c += (dt / self.tau_battery_min) * (target_temp - self.battery_temp_c)

    def _arrhenius(self) -> float:
        """Arrhenius sıcaklık faktörü — NMC aktivasyon enerjisi Ea/R = 6976 K."""
        T_k = self.battery_temp_c + 273.15
        return math.exp(6976.0 * (1.0 / 298.15 - 1.0 / T_k))

    def elec_deg_rate(self) -> float:
        """Elektrolit bozulma hızı — SoC bazlı takvim yaşlanması.

        Yüksek SoC'da elektrolit oksidasyonu hızlanır (anodik çözünme).
        Kaynak: Severson et al., Nature Energy 4 (2019) 383-391.
                Schmalstieg et al., J. Power Sources 257 (2014) 325.
        """
        # SoC > %50'den itibaren üstel artış; zirve %100'de
        soc_factor = math.exp(3.0 * max(0.0, self.current_soc - 0.50))
        return soc_factor * self._arrhenius()

    def mech_stress_rate(self, c_rate: float) -> float:
        """Mekanik stres hızı — C-rate bazlı hacim değişimi (lityum intercalasyon).

        Hızlı şarj → yüksek akım yoğunluğu → katman yapısında gerilim.
        Kaynak: Wang et al., J. Power Sources 196 (2011) 3942.
        """
        return c_rate ** 2 * self._arrhenius()

    def marginal_aging_cost(self, power_kw: float) -> float:
        """Birleşik marjinal batarya yaşlanma maliyeti.

        Üç bileşen:
          1) Elektrolit bozulması (SoC + sıcaklık bazlı takvim yaşlanması)
          2) Mekanik stres (C-rate bazlı döngüsel yaşlanma, hacim genleşmesi)
          3) Yüksek SoC + yüksek C-rate çakışma (SEI büyümesi hızlanması)

        Kaynak:
          - Wang et al., J. Power Sources 196 (2011) 3942 — NMC cycle aging
          - Schmalstieg et al., J. Power Sources 257 (2014) 325 — NMC calendar+cycle
          - Severson et al., Nature Energy 4 (2019) 383 — SoC-dependent calendar
          Arrhenius: Ea/R ≈ 6976 K (aktivasyon enerjisi NMC için), Tref=25°C
        """
        c_rate = power_kw / max(self.battery_capacity_kwh, 0.1)
        k_elec = self.elec_deg_rate()                     # takvim bileşeni
        k_mech = self.mech_stress_rate(c_rate)            # döngüsel bileşen
        # SEI büyümesi: yüksek SoC + yüksek C-rate çakışması
        sei_boost = math.exp(2.0 * max(0.0, self.current_soc - 0.70)) * c_rate
        return 0.50 * k_elec + 0.35 * k_mech + 0.15 * sei_boost * self._arrhenius()

    def apply_power(self, power_kw: float, minute: int) -> float:
        if self.current_soc >= self.target_soc:
            return 0.0
        # CC-CV fiziksel taper: araç fiziksel olarak bu güçten fazlasını alamaz
        effective_power = min(power_kw, self.max_acceptable_power_kw)
        energy_to_absorb = min(effective_power * (1.0 / 60.0), self.energy_needed_kwh)
        self.current_soc = min(
            self.current_soc + energy_to_absorb / self.battery_capacity_kwh,
            self.target_soc
        )
        self.energy_delivered_kwh += energy_to_absorb
        if effective_power > 0.0:
            self.charge_minutes += 1
            # Kümülatif yaşlanma integralleri (dakika bazlı)
            c_rate = effective_power / max(self.battery_capacity_kwh, 0.1)
            self.elec_deg_integral  += self.elec_deg_rate()          / 60.0
            self.mech_stress_integral += self.mech_stress_rate(c_rate) / 60.0
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
    # IEC 60076-7 termal model için ortam sıcaklığı parametreleri
    ambient_temp_min: float = 18.0        # gece min °C
    ambient_temp_max: float = 32.0        # gündüz max °C
    ambient_temp_peak_hour: float = 15.0  # sıcaklık zirvesi saati


@dataclass
class GridConfig:
    trafo_kva: float
    power_factor: float = 0.90
    safety_margin: float = 0.88
    evening_peak_kw: float = 950.0
    peak_start_hour: float = 17.0
    peak_end_hour: float = 22.0
    # IEC 60076-7 termal model parametreleri
    delta_theta_oil_rated: float = 55.0   # K — IEC 60076-7 Table 4, ONAN
    delta_theta_hs_rated: float = 25.0    # K — IEC 60076-7 Table 4, ONAN
    tau_oil_min: float = 150.0            # dk — yağ termal zaman sabiti

    @property
    def trafo_max_kw(self) -> float:
        return round(self.trafo_kva * self.power_factor * self.safety_margin, 1)

    @property
    def peak_start_min(self) -> int:
        return int(self.peak_start_hour * 60)

    @property
    def peak_end_min(self) -> int:
        return int(self.peak_end_hour * 60)

    @property
    def s_rated_kw(self) -> float:
        """Trafo nominal güç (kW) = kVA × pf"""
        return self.trafo_kva * self.power_factor


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

    def to_grid_limit_policy(self, use_dynamic: bool = False,
                                theta_hs_target: float = 98.0) -> GridLimitPolicy:
        if use_dynamic:
            return DynamicGridLimitPolicy(
                trafo_max_kw=self.grid.trafo_max_kw,
                evening_peak_kw=self.grid.evening_peak_kw,
                peak_start_min=self.grid.peak_start_min,
                peak_end_min=self.grid.peak_end_min,
                grid_config=self.grid,
                environment=self.environment,
                theta_hs_target=theta_hs_target,
            )
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
