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
    trafo_max_kw: float = 600.0
    evening_peak_kw: float = 400.0
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
# Veri Üreticiler
# ==============================================================================
class ArrivalGenerator:
    EV_MODELS = [
        EVModel("Togg T10X", 88.5, 150.0, 0.25),
        EVModel("Tesla Model Y", 75.0, 250.0, 0.20),
        EVModel("Tesla Model Y RWD", 60.0, 170.0, 0.15),
        EVModel("BYD Atto 3", 60.4, 88.0, 0.15),
        EVModel("MG4 Standard", 51.0, 117.0, 0.10),
        EVModel("Renault Megane", 60.0, 130.0, 0.10),
        EVModel("Porsche Taycan", 93.4, 270.0, 0.05),
    ]
    def __init__(self, daily_ev_count: int = 35):
        self.daily_ev_count = daily_ev_count

    def generate_arrivals(self, rng: np.random.Generator) -> Dict[int, List[EV]]:
        n_morn = self.daily_ev_count // 4
        n_lunch = self.daily_ev_count // 4
        n_eve = self.daily_ev_count - n_morn - n_lunch
        arrs = np.concatenate([rng.normal(570, 40, n_morn), rng.normal(780, 35, n_lunch), rng.normal(1110, 75, n_eve)])
        rng.shuffle(arrs)
        schedule = {}
        for i, m in enumerate(np.clip(arrs, 0, 1439)):
            minute_idx = int(m)
            model = rng.choice(self.EV_MODELS, p=[x.probability for x in self.EV_MODELS])
            if minute_idx not in schedule: schedule[minute_idx] = []
            schedule[minute_idx].append(EV(f"EV_{i+1:02d}", model.model_name, model.battery_capacity_kwh, model.max_dc_power_kw, minute_idx, rng.uniform(0.10, 0.40)))
        return schedule

class BackgroundLoadGenerator:
    @staticmethod
    def generate(rng: np.random.Generator) -> np.ndarray:
        hrs = np.arange(1440) / 60.0
        base = 45.0 + 90.0 * np.clip(0.5 * (1 - np.cos(np.pi * np.clip(hrs - 6, 0, 14) / 14)), 0, 1)
        peak_m = 70.0 * np.exp(-0.5 * ((hrs - 9.0) / 1.0) ** 2)
        peak_e = 100.0 * np.exp(-0.5 * ((hrs - 19.0) / 1.2) ** 2)
        return np.clip(base + peak_m + peak_e + rng.normal(0.0, 15.0, 1440), 25.0, 280.0)

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
        for s in occ:
            if is_peak and s.current_ev.target_soc > 0.70:
                s.current_ev.target_soc = 0.70

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
                best = max(self.queue, key=lambda ev: (minute - ev.arrival_minute) * 2.0 - (ev.energy_needed_kwh / ev.max_dc_power_kw * 60))
                self.queue.remove(best)
                s.current_ev = best
                best.charge_start_minute = minute

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

# ==============================================================================
# Grafik ve Ana Akış
# ==============================================================================
class ExecutiveDashboard:
    @staticmethod
    def create(r_u: SimulationResult, r_m: SimulationResult):
        fig = plt.figure(figsize=(16, 24))
        fig.suptitle("EV Şarj Yük Dengeleme — Yönetici Özeti", fontsize=18, fontweight="bold", y=0.995)
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.25)
        axs = [fig.add_subplot(gs[i]) for i in [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3, slice(None))]]

        hrs = np.arange(1440) / 60
        axs[0].fill_between(hrs, 0, r_u.power_timeseries, alpha=0.3, color="red", label="Kontrol Öncesi")
        axs[0].plot(hrs, r_m.power_timeseries, lw=2.5, color="darkgreen", label="Kontrol Sonrası")
        axs[0].plot(hrs, r_m.grid_limit_timeseries, color="red", ls="--", lw=2, label="Trafo Limiti")
        axs[0].axvspan(17, 22, alpha=0.08, color="orange", label="Şebeke Pik (17-22)")
        axs[0].set(xlabel="Gün Saati", ylabel="Yük (kW)", title="Panel 1: Yük Profili", xlim=(0,24))
        axs[0].legend(loc="upper left"); axs[0].grid(True, alpha=0.3)

        w = [s.wait_time_minutes for s in r_m.vehicle_sessions]
        bars = axs[1].bar(["0 dk", "1-15 dk", "15+ dk"], [sum(1 for x in w if x==0), sum(1 for x in w if 0<x<=15), sum(1 for x in w if x>15)], color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black")
        axs[1].set(ylabel="Araç", title="Panel 2: Bekleme Dağılımı")
        for b in bars: axs[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", va="bottom", fontweight="bold")

        mods = sorted(list({s.model_name.split()[0] for s in r_u.vehicle_sessions}))
        g = lambda r: {m: [] for m in mods}
        c_u, c_m, w_u, w_m = g(r_u), g(r_m), g(r_u), g(r_m)
        for s in r_u.vehicle_sessions: m=s.model_name.split()[0]; c_u[m].append(s.charge_time_minutes); w_u[m].append(s.wait_time_minutes)
        for s in r_m.vehicle_sessions: m=s.model_name.split()[0]; c_m[m].append(s.charge_time_minutes); w_m[m].append(s.wait_time_minutes)
        x = np.arange(len(mods)); wd = 0.35

        axs[2].bar(x-wd/2, [np.mean(c_u[m]) for m in mods], wd, color="lightcoral", ec="black", label="Öncesi")
        axs[2].bar(x+wd/2, [np.mean(c_m[m]) for m in mods], wd, color="lightgreen", ec="black", label="Sonrası")
        axs[2].set(ylabel="Şarj (dk)", title="Panel 3: Şarj Süresi", xticks=x, xticklabels=mods); axs[2].legend()

        axs[3].bar(x-wd/2, [np.mean(w_u[m]) for m in mods], wd, color="#f9c784", ec="black", label="Öncesi")
        axs[3].bar(x+wd/2, [np.mean(w_m[m]) for m in mods], wd, color="#74b9ff", ec="black", label="Sonrası")
        axs[3].set(ylabel="Bekleme (dk)", title="Panel 4: Bekleme Süresi", xticks=x, xticklabels=mods); axs[3].legend()

        axs[4].axis("off")
        axs[4].text(0.05, 0.95, f"Kapasite: {r_m.metrics_summary.protected_capacity_percent:.1f}%\nServis: {r_m.metrics_summary.evs_completed} araç\nAlgoritma: İsyan Protokolü + %70 Kısıt", transform=axs[4].transAxes, fontsize=12, va="top", bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.4))

        axs[5].bar(x-wd/2, [np.mean(c_u[m])+np.mean(w_u[m]) for m in mods], wd, color="lightcoral", ec="black")
        axs[5].bar(x+wd/2, [np.mean(c_m[m])+np.mean(w_m[m]) for m in mods], wd, color="lightgreen", ec="black")
        axs[5].set(ylabel="Sistem (dk)", title="Panel 6: Toplam Süre", xticks=x, xticklabels=mods)

        ud = {s.session_id: s for s in r_u.vehicle_sessions}
        d_c, d_w = {m: [] for m in mods}, {m: [] for m in mods}
        for s in r_m.vehicle_sessions:
            if s.session_id in ud:
                m = s.model_name.split()[0]
                d_c[m].append(s.charge_time_minutes - ud[s.session_id].charge_time_minutes)
                d_w[m].append(max(0, s.wait_time_minutes - ud[s.session_id].wait_time_minutes))

        w7 = 0.2
        axs[6].bar(x-1.5*w7, [max(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#e74c3c", ec="black", label="Maks Şarj Artışı")
        axs[6].bar(x-0.5*w7, [min(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#f1948a", ec="black", label="Min Şarj Artışı")
        axs[6].bar(x+0.5*w7, [max(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#2980b9", ec="black", label="Maks Bekleme Artışı")
        axs[6].bar(x+1.5*w7, [min(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#7fb3d5", ec="black", label="Min Bekleme Artışı")
        axs[6].set(xticks=x, xticklabels=mods, ylabel="Delta (dk)", title="Panel 7: Maks/Min Gecikme")
        axs[6].legend(loc="upper left"); axs[6].grid(True, alpha=0.3, axis="y")

        plt.savefig("executive_dashboard_v4.png", dpi=150, bbox_inches="tight")
        print("✓ Grafik: executive_dashboard_v4.png")
        plt.show()

def main(generate_new: bool = False):
    print("EV Yük Dengeleme Simülasyonu başlatılıyor...")
    dataset_file = "dataset.json"

    if not generate_new:
        if not os.path.exists(dataset_file):
            print(f"\nHATA: '{dataset_file}' dosyası yok! Lütfen 'python simulation_v3.py --generate-new' ile oluşturun.\n")
            sys.exit(1)
        
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        schedule = {}
        for v in data['vehicles']:
            m = v['arrival_minute']
            if m not in schedule: schedule[m] = []
            schedule[m].append(EV(v['session_id'], v['model_name'], v['battery_capacity_kwh'], v['max_dc_power_kw'], m, v['initial_soc']))
        bg_load = np.array(data['background_load_profile'])
    else:
        rng = np.random.default_rng(42)
        schedule = ArrivalGenerator(35).generate_arrivals(rng)
        bg_load = BackgroundLoadGenerator.generate(np.random.default_rng(101))
        
        vehicles_data = [{"session_id": ev.session_id, "model_name": ev.model_name, "battery_capacity_kwh": ev.battery_capacity_kwh, "max_dc_power_kw": ev.max_dc_power_kw, "arrival_minute": ev.arrival_minute, "initial_soc": ev.initial_soc} for m in sorted(schedule.keys()) for ev in schedule[m]]
        with open(dataset_file, 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "vehicles": vehicles_data, "background_load_profile": bg_load.tolist()}, f, indent=2)

    st_a = [ChargingStation("S1", StationType.ULTRA_FAST, 200), ChargingStation("S2", StationType.ULTRA_FAST, 200), ChargingStation("S3", StationType.FAST, 180), ChargingStation("S4", StationType.FAST, 180), ChargingStation("S5", StationType.STANDARD, 120)]
    st_b = copy.deepcopy(st_a)
    
    ctrl_a = UnmanagedController(st_a, GridLimitPolicy(), bg_load)
    res_a = Simulation(ctrl_a, copy.deepcopy(schedule)).run()
    
    ctrl_b = ManagedController(st_b, GridLimitPolicy(), bg_load)
    res_b = Simulation(ctrl_b, copy.deepcopy(schedule)).run()
    
    # MATRİS EXCELİNİ YARATAN FONKSİYON
    export_comparative_excel(ctrl_a, ctrl_b)
    
    ExecutiveDashboard.create(res_a, res_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Yük Dengeleme")
    parser.add_argument("--generate-new", action="store_true")
    args = parser.parse_args()
    main(generate_new=args.generate_new)