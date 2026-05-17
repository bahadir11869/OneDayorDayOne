#!/usr/bin/env python3
"""Kontrolcüler - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
from typing import Dict, List, Optional
import copy
import numpy as np

from models import (
    ChargingStation, GridLimitPolicy, DynamicGridLimitPolicy,
    EV, EVState, EVModel,
    SimulationResult, VehicleSession, MetricsSummary
)


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


class DynamicPeakDetector:
    """
    Trafo doluluk yüzdesi ve delta-T'ye bakarak peak'in yaklaştığını dinamik olarak algılar.

    Mantık:
      load_pct   = son_toplam_güç / trafo_max_kw × 100
      delta_kw   = (son_güç - WINDOW dk önceki güç) / WINDOW  [kW/dk]

      Koşul 1 — Yüksek yük   : load_pct ≥ LOAD_HIGH_PCT  → kesin peak mod
      Koşul 2 — Yükselen yük : load_pct ≥ LOAD_MID_PCT AND delta_kw ≥ DELTA_THRESH → peak yaklaşıyor

    Dönen değer:
      (is_peak: bool, load_pct: float, delta_kw: float)
    """

    WINDOW       = 15    # dk — trend hesabı için geriye bakış penceresi
    LOAD_HIGH_PCT = 75.0  # %  — bu eşiği geçerse kesin peak mod
    LOAD_MID_PCT  = 55.0  # %  — bu eşik + yükselen trend → peak mod
    DELTA_THRESH  = 4.0   # kW/dk — yükselme hızı eşiği

    def __init__(self, trafo_max_kw: float):
        self.trafo_max_kw = trafo_max_kw
        self._history: List[float] = []   # dakika bazlı toplam güç geçmişi

    def update(self, total_power_kw: float):
        """Her adımda çağrılır; güç geçmişini tutar."""
        self._history.append(total_power_kw)

    def detect(self, current_bg_kw: float) -> tuple:
        """
        Mevcut yük durumuna göre (load_pct, delta_kw, is_peak) döndürür.
        power_log dolmamışsa bg_load ile çalışır.
        """
        if len(self._history) >= 2:
            recent = self._history[-1]
            window = min(self.WINDOW, len(self._history) - 1)
            past   = self._history[-1 - window]
            delta_kw = (recent - past) / window
        else:
            recent   = current_bg_kw
            delta_kw = 0.0

        load_pct = recent / self.trafo_max_kw * 100.0

        is_high_load   = load_pct >= self.LOAD_HIGH_PCT
        is_rising_fast = (load_pct >= self.LOAD_MID_PCT and delta_kw >= self.DELTA_THRESH)
        is_peak        = is_high_load or is_rising_fast

        return is_peak, load_pct, delta_kw


class AdaptiveSoCController:
    """
    Trafo + Batarya Sağlığı Optimal Kontrolcü.

    Trafo koruması (iki mod):
    A) use_dynamic_grid_limit=False (statik mod — mevcut davranış):
       - Soft peak ramp: pik 30 dk önce lineer düşer.
       - DynamicPeakDetector: doluluk% + delta-T bazlı kısıtlama.
    B) use_dynamic_grid_limit=True (termal mod — IEC 60076-7):
       - DynamicGridLimitPolicy: θ_oil ODE, θ_hs, K_max → dinamik P_max.
       - Termal durum kümülatif — sabah soğuk trafo → yüksek kapasite,
         öğleden sonra ısınan trafo → düşen kapasite.

    Batarya sağlığı:
    - C-rate ≤ 1.5C (endüstri standardı).
    - SoC taper — CC-CV eğrisi (kontrolcü tarafı).
    - Marjinal yaşlanma maliyeti (Wang 2011, Schmalstieg 2014):
        use_aging_cost=True ise skor = w_urgency·mevcut − w_aging·aging_cost
        aging > threshold → güç %50 kısılır.

    Öncelik ve dağıtım:
    - Skor = (bekleme / kalan_enerji) × c_rate_headroom [− w_aging × aging]
    - Orantılı water-filling: taşan bütçe diğer araçlara yeniden dağıtılır.
    """

    RAMP_MINUTES = 30
    LOOKAHEAD    = 5
    SAFETY_KW    = 20.0
    SAFETY_KW_THERMAL = 5.0  # termal modda daha düşük (model kendi marjını içerir)
    MAX_C_RATE   = 1.5

    def __init__(self, stations, limit_policy, bg_load, **kwargs):
        self.stations     = stations
        self.policy       = limit_policy
        self.bg_load      = bg_load
        self.queue        = []
        self.power_log    = []
        self.limit_log    = []
        self.completed    = []
        self.timeline_log = []
        self._peak_det    = DynamicPeakDetector(limit_policy.trafo_max_kw)

        # Yeni parametreler — kwargs ile geriye uyumlu
        self.use_dynamic_grid_limit   = kwargs.get('use_dynamic_grid_limit', True)
        self.use_aging_cost           = kwargs.get('use_aging_cost', True)
        self.w_urgency                = kwargs.get('w_urgency', 1.0)
        self.w_aging                  = kwargs.get('w_aging', 0.5)
        self.aging_throttle_threshold = kwargs.get('aging_throttle_threshold', 3.0)

        # Termal mod aktif mi? (policy tipi ile belirlenir)
        self._thermal_active = (
            self.use_dynamic_grid_limit and
            isinstance(limit_policy, DynamicGridLimitPolicy)
        )

        # Yaşlanma ve batarya termal metrikleri (reporting)
        self._aging_costs: List[float] = []
        self.bat_temp_log: List[float] = []   # dakika bazlı ortalama batarya sıcaklığı

    def compute_limit(self, minute: int) -> float:
        """
        Termal mod:  DynamicGridLimitPolicy.current_limit_kw() → IEC bazlı tavan.
        Statik mod:  3-katmanlı mevcut mantık (peak + ramp + DynamicPeakDetector).
        """
        # ══════════ TERMAL MOD ═══════════════════════════════════════════════
        if self._thermal_active:
            return self.policy.current_limit_kw(minute, self.bg_load[minute])

        # ══════════ STATİK MOD (mevcut davranış — bit-identical) ═════════════
        tod        = minute % 1440
        ramp_start = max(0, self.policy.peak_start_min - self.RAMP_MINUTES)

        # Katman 1: statik peak penceresi
        if self.policy.peak_start_min <= tod < self.policy.peak_end_min:
            return self.policy.evening_peak_kw

        # Katman 2: soft ramp
        if ramp_start <= tod < self.policy.peak_start_min:
            progress = (tod - ramp_start) / self.RAMP_MINUTES
            static_limit = self.policy.trafo_max_kw - progress * (
                self.policy.trafo_max_kw - self.policy.evening_peak_kw
            )
        else:
            static_limit = self.policy.trafo_max_kw

        # Katman 3: dinamik peak tespiti (DynamicPeakDetector)
        is_dyn_peak, load_pct, delta_kw = self._peak_det.detect(self.bg_load[minute])

        if is_dyn_peak:
            det = self._peak_det
            if load_pct >= det.LOAD_HIGH_PCT:
                dyn_limit = self.policy.evening_peak_kw
            else:
                t = (load_pct - det.LOAD_MID_PCT) / (det.LOAD_HIGH_PCT - det.LOAD_MID_PCT)
                t = max(0.0, min(1.0, t))
                dyn_limit = self.policy.trafo_max_kw - t * (
                    self.policy.trafo_max_kw - self.policy.evening_peak_kw
                )
            return min(static_limit, dyn_limit)

        return static_limit

    def soc_tapered_power(self, ev, station_max_kw: float) -> float:
        """C-rate limitli, SoC'a göre kademeli güç tavanı (CC-CV eğrisi)."""
        c_rate_limit = ev.battery_capacity_kwh * self.MAX_C_RATE
        soc = ev.current_soc
        if soc < 0.50:
            taper = 1.0
        elif soc < 0.70:
            taper = 1.0 - 0.30 * ((soc - 0.50) / 0.20)   # 1.0 → 0.70
        else:
            taper = 0.70 - 0.50 * ((soc - 0.70) / 0.10)  # 0.70 → 0.20
        return min(station_max_kw, c_rate_limit, ev.max_dc_power_kw) * max(taper, 0.05)

    def allocate_power(self, minute: int) -> Dict[str, float]:
        limit       = self.compute_limit(minute)
        end_idx     = min(minute + self.LOOKAHEAD, 1439)
        future_base = float(np.max(self.bg_load[minute: end_idx + 1]))

        active = [s for s in self.stations if s.current_ev and not s.current_ev.is_satisfied]
        allocs = {s.station_id: 0.0 for s in self.stations}
        if not active:
            return allocs

        safety = self.SAFETY_KW_THERMAL if self._thermal_active else self.SAFETY_KW
        budget = max(0.0, limit - future_base - safety)
        caps   = {s.station_id: self.soc_tapered_power(s.current_ev, s.max_power_kw) for s in active}

        def score(s):
            ev              = s.current_ev
            wait            = minute - ev.arrival_minute + 1
            energy          = max(ev.energy_needed_kwh, 0.01)
            c_rate_headroom = max(0.1, self.MAX_C_RATE - (
                ev.energy_delivered_kwh / max(ev.battery_capacity_kwh, 0.1)
            ))
            base_score = (wait / energy) * c_rate_headroom

            if not self.use_aging_cost:
                return base_score

            # Yaşlanma maliyeti: proposed_power ≈ cap (max alacağı güç tahmini)
            proposed_power = caps[s.station_id]
            aging = ev.marginal_aging_cost(proposed_power)
            adjusted = self.w_urgency * base_score - self.w_aging * aging
            return max(adjusted, 0.01)  # negatif skor önle

        pending = list(active)
        while budget > 0.01 and pending:
            scores      = {s.station_id: score(s) for s in pending}
            total_score = sum(scores.values())
            if total_score <= 0:
                break
            overflow    = 0.0
            new_pending = []
            for s in pending:
                proportion    = scores[s.station_id] / total_score
                want          = budget * proportion
                remaining_cap = caps[s.station_id] - allocs[s.station_id]
                if want >= remaining_cap:
                    allocs[s.station_id] += remaining_cap
                    overflow += want - remaining_cap
                else:
                    allocs[s.station_id] += want
                    new_pending.append(s)
            if not new_pending or overflow < 0.01:
                break
            budget  = overflow
            pending = new_pending

        # ── Aging throttle: yaşlanma maliyeti eşik üstündeyse gücü %50 kıs ──
        if self.use_aging_cost:
            surplus = 0.0
            throttled_ids = set()
            for s in active:
                sid = s.station_id
                if allocs[sid] > 0.01:
                    aging_cost = s.current_ev.marginal_aging_cost(allocs[sid])
                    self._aging_costs.append(aging_cost)
                    if aging_cost > self.aging_throttle_threshold:
                        cut = allocs[sid] * 0.50
                        surplus += cut
                        allocs[sid] -= cut
                        throttled_ids.add(sid)

            # Surplus'u throttle edilmemiş araçlara water-fill ile dağıt
            if surplus > 0.01:
                eligible = [s for s in active
                            if s.station_id not in throttled_ids
                            and allocs[s.station_id] < caps[s.station_id] - 0.01]
                while surplus > 0.01 and eligible:
                    share = surplus / len(eligible)
                    new_eligible = []
                    for s in eligible:
                        room = caps[s.station_id] - allocs[s.station_id]
                        give = min(share, room)
                        allocs[s.station_id] += give
                        surplus -= give
                        if room - give > 0.01:
                            new_eligible.append(s)
                    eligible = new_eligible

        return allocs

    def step(self, minute: int):
        for s in self.stations:
            if not s.current_ev and self.queue:
                s.current_ev = self.queue.pop(0)
                s.current_ev.charge_start_minute = minute

        limit  = self.compute_limit(minute)
        allocs = self.allocate_power(minute)

        for ev in self.queue:
            self.timeline_log.append({"Dakika": minute, "Araç ID": ev.session_id, "Durum": "Kuyrukta", "İstasyon": "-", "BazGüç (kW)": round(self.bg_load[minute], 1)})
        for s in self.stations:
            if s.current_ev:
                self.timeline_log.append({"Dakika": minute, "Araç ID": s.current_ev.session_id, "Durum": "Şarjda", "İstasyon": s.station_id, "Güç (kW)": round(allocs[s.station_id], 1), "SoC (%)": round(s.current_ev.current_soc * 100, 1), "BazGüç (kW)": round(self.bg_load[minute], 1)})

        # Batarya sıcaklığını güncelle, sonra gücü uygula
        ambient_temp = (
            self.policy._ambient_temp(minute)
            if self._thermal_active
            else 25.0   # termal mod yoksa sabit referans sıcaklık
        )
        for s in self.stations:
            if s.current_ev:
                ev = s.current_ev
                alloc_kw = allocs[s.station_id]
                # Önce batarya sıcaklığını güncelle (apply_power'dan önce)
                ev.update_battery_temp(ambient_temp, alloc_kw)
                ev.apply_power(alloc_kw, minute)
                if ev.is_satisfied:
                    ev.departure_minute = minute
                    self.completed.append(ev)
                    s.current_ev = None

        total_power = self.bg_load[minute] + sum(allocs.values())
        self.power_log.append(total_power)
        self.limit_log.append(limit)

        # Ortalama batarya sıcaklığını logla (şarj eden araçlar)
        charging_evs = [s.current_ev for s in self.stations if s.current_ev]
        if charging_evs:
            self.bat_temp_log.append(float(np.mean([ev.battery_temp_c for ev in charging_evs])))
        else:
            self.bat_temp_log.append(float('nan'))

        # Termal model güncelle ve dinamik limit logla (her dakika 1 kez)
        if self._thermal_active:
            self.policy.update(minute, total_power)
            self.policy.dynamic_limit_log.append(limit)
        # Statik mod fallback — DynamicPeakDetector
        self._peak_det.update(total_power)


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
