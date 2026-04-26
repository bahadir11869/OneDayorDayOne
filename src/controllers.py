#!/usr/bin/env python3
"""Kontrolcüler - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
from typing import Dict, List, Optional
import copy
import numpy as np

from models import (
    ChargingStation, GridLimitPolicy, EV, EVState, EVModel,
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
