# streamlit run app.py
#!/usr/bin/env python3
"""
EV Yük Dengeleme — SRPT Simülasyon Arayüzü
Çalıştırma: streamlit run app.py
"""

import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from controllers import AdaptiveSoCController, UnmanagedController
from generators import ArrivalGenerator, BackgroundLoadGenerator, Scenarios
from models import EV, ScenarioConfig, DynamicGridLimitPolicy


# ══════════════════════════════════════════════════════════════════
#  SRPT — 1440 detaylı snapshot + araç logu
# ══════════════════════════════════════════════════════════════════

def build_snapshots(schedule: dict, bg_load: np.ndarray, config: ScenarioConfig):
    """AdaptiveSoC (termal mod) simülasyonu. (snapshots_list, vehicle_log_list) döner."""
    policy   = config.to_grid_limit_policy(use_dynamic=True, theta_hs_target=98.0)
    stations = copy.deepcopy(config.layout.stations)
    ctrl     = AdaptiveSoCController(
        stations, policy, bg_load,
        use_dynamic_grid_limit=True, use_aging_cost=True,
        w_urgency=1.0, w_aging=0.25, aging_throttle_threshold=3.0,
    )
    is_thermal     = isinstance(policy, DynamicGridLimitPolicy)
    snapshots      = []
    overload_total = 0
    vehicle_events = {}

    for minute in range(1440):
        for ev in schedule.get(minute, []):
            ctrl.queue.append(ev)
            vehicle_events[ev.session_id] = {
                "session_id":           ev.session_id,
                "model_name":           ev.model_name,
                "battery_kwh":          ev.battery_capacity_kwh,
                "max_dc_kw":            ev.max_dc_power_kw,
                "arrival_minute":       minute,
                "arrival_time":         _fmt(minute),
                "initial_soc_pct":      round(ev.initial_soc * 100, 1),
                "target_soc_pct":       round(ev.target_soc * 100, 1),
                "charge_start_minute":  None,
                "charge_start_time":    None,
                "departure_minute":     None,
                "departure_time":       None,
                "wait_minutes":         None,
                "charge_minutes":       None,
                "final_soc_pct":        None,
                "energy_delivered_kwh": None,
                "status":               "Kuyrukta",
            }

        for s in ctrl.stations:
            if not s.current_ev and ctrl.queue:
                s.current_ev = ctrl.queue.pop(0)
                s.current_ev.charge_start_minute = minute
                sid = s.current_ev.session_id
                vehicle_events[sid]["charge_start_minute"] = minute
                vehicle_events[sid]["charge_start_time"]   = _fmt(minute)
                vehicle_events[sid]["wait_minutes"]        = minute - vehicle_events[sid]["arrival_minute"]
                vehicle_events[sid]["status"]              = "Şarjda"

        allocs     = ctrl.allocate_power(minute)
        tod        = minute % 1440
        limit      = ctrl.compute_limit(minute)
        is_peak    = policy.peak_start_min <= tod < policy.peak_end_min
        ramp_start = max(0, policy.peak_start_min - AdaptiveSoCController.RAMP_MINUTES)
        is_ramp    = ramp_start <= tod < policy.peak_start_min
        bg         = float(bg_load[minute])
        ev_load    = float(sum(allocs.values()))
        total      = bg + ev_load
        over_kw    = max(0.0, total - limit)
        if over_kw > 0:
            overload_total += 1

        # Batarya sıcaklığını bu dakikanın gücüne göre güncelle (apply_power öncesi)
        ambient_temp = policy._ambient_temp(minute) if is_thermal else 25.0
        for s in ctrl.stations:
            if s.current_ev:
                s.current_ev.update_battery_temp(ambient_temp, allocs[s.station_id])

        # Termal durum anlık
        theta_hs    = float(policy.theta_hs)   if is_thermal else 0.0
        theta_oil   = float(policy.theta_oil)  if is_thermal else 0.0
        aging_rate  = float(policy.aging_rate) if is_thermal else 0.0

        # Adaptif karar listesi
        active_stations    = [s for s in ctrl.stations if s.current_ev and not s.current_ev.is_satisfied]
        adaptive_decisions = []
        for s in active_stations:
            ev         = s.current_ev
            pwr        = round(allocs.get(s.station_id, 0.0), 1)
            wait       = minute - ev.arrival_minute
            energy     = ev.energy_needed_kwh
            taper_cap  = ctrl.soc_tapered_power(ev, s.max_power_kw)
            soc        = ev.current_soc
            c_rate_headroom = max(0.1, AdaptiveSoCController.MAX_C_RATE - (
                ev.energy_delivered_kwh / max(ev.battery_capacity_kwh, 0.1)
            ))
            priority   = round((wait + 1) / max(energy, 0.01) * c_rate_headroom, 2)
            taper_zone = "CC-CV" if soc >= 0.70 else ("Taper" if soc >= 0.50 else "Normal")
            adaptive_decisions.append({
                "station_id":        s.station_id,
                "session_id":        ev.session_id,
                "model_name":        ev.model_name,
                "energy_needed_kwh": round(energy, 2),
                "wait_minutes":      wait,
                "power_kw":          pwr,
                "current_soc_pct":   round(soc * 100, 1),
                "tapered_cap_kw":    round(taper_cap, 1),
                "c_rate_actual":     round(pwr / ev.battery_capacity_kwh, 3) if ev.battery_capacity_kwh > 0 else 0,
                "taper_zone":        taper_zone,
                "is_ramp_active":    is_ramp,
                "priority_score":    priority,
                "no_budget":         pwr < 0.1,
                "battery_temp_c":    round(float(ev.battery_temp_c), 1),
                "aging_cost":        round(float(ev.marginal_aging_cost(pwr)), 4),
            })

        # İstasyon anlık durumu
        station_states = []
        for s in ctrl.stations:
            ev_data = None
            if s.current_ev:
                ev          = s.current_ev
                wait_time   = (ev.charge_start_minute - ev.arrival_minute) if ev.charge_start_minute is not None else (minute - ev.arrival_minute)
                charge_time = (minute - ev.charge_start_minute) if ev.charge_start_minute is not None else 0
                ev_data = {
                    "session_id":           ev.session_id,
                    "model_name":           ev.model_name,
                    "current_soc":          float(ev.current_soc),
                    "initial_soc":          float(ev.initial_soc),
                    "target_soc":           float(ev.target_soc),
                    "energy_delivered_kwh": round(float(ev.energy_delivered_kwh), 2),
                    "energy_needed_kwh":    round(float(ev.energy_needed_kwh), 2),
                    "battery_capacity_kwh": float(ev.battery_capacity_kwh),
                    "max_dc_power_kw":      float(ev.max_dc_power_kw),
                    "wait_time_minutes":    int(wait_time),
                    "charge_time_minutes":  int(charge_time),
                    "arrival_minute":       int(ev.arrival_minute),
                    "is_new":               ev.charge_start_minute == minute,
                    "battery_temp_c":       round(float(ev.battery_temp_c), 1),
                }
            station_states.append({
                "id":           s.station_id,
                "type":         s.station_type.value,
                "max_power_kw": float(s.max_power_kw),
                "occupied":     s.current_ev is not None,
                "ev":           ev_data,
                "power_kw":     round(allocs.get(s.station_id, 0.0), 1),
            })

        # Kuyruk
        queue_state = [{
            "session_id":           ev.session_id,
            "model_name":           ev.model_name,
            "arrival_minute":       int(ev.arrival_minute),
            "wait_minutes":         int(minute - ev.arrival_minute),
            "initial_soc_pct":      round(float(ev.initial_soc) * 100, 1),
            "energy_needed_kwh":    round(float(ev.energy_needed_kwh), 2),
            "battery_capacity_kwh": float(ev.battery_capacity_kwh),
        } for ev in ctrl.queue]

        avg_wait     = float(np.mean([e.wait_time_minutes for e in ctrl.completed])) if ctrl.completed else 0.0
        total_energy = float(sum(e.energy_delivered_kwh for e in ctrl.completed))

        snapshots.append({
            "minute":                     minute,
            "time_str":                   _fmt(minute),
            "stations":                   station_states,
            "queue":                      queue_state,
            "bg_load_kw":                 round(bg, 1),
            "ev_load_kw":                 round(ev_load, 1),
            "total_power_kw":             round(total, 1),
            "grid_limit_kw":              round(limit, 1),
            "is_peak":                    bool(is_peak),
            "is_ramp":                    bool(is_ramp),
            "completed_count":            int(len(ctrl.completed)),
            "avg_wait_minutes":           round(avg_wait, 1),
            "total_energy_delivered_kwh": round(total_energy, 1),
            "overload_kw":                round(over_kw, 1),
            "overload_total_minutes":     int(overload_total),
            "adaptive_decisions":         adaptive_decisions,
            "theta_hs":                   round(theta_hs, 2),
            "theta_oil":                  round(theta_oil, 2),
            "aging_rate":                 round(aging_rate, 4),
            "dynamic_limit_kw":          round(limit, 1) if is_thermal else None,
        })

        for s in ctrl.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    ctrl.completed.append(s.current_ev)
                    sid = s.current_ev.session_id
                    vehicle_events[sid].update({
                        "departure_minute":     minute,
                        "departure_time":       _fmt(minute),
                        "charge_minutes":       minute - (s.current_ev.charge_start_minute or minute),
                        "final_soc_pct":        round(s.current_ev.current_soc * 100, 1),
                        "energy_delivered_kwh": round(s.current_ev.energy_delivered_kwh, 2),
                        "status":               "Tamamlandı",
                    })
                    s.current_ev = None

        ctrl.power_log.append(total)
        ctrl.limit_log.append(limit)

        # Termal modeli dakika sonunda güncelle (step() ile aynı sıra)
        if is_thermal:
            policy.update(minute, total)

    # Hâlâ şarjda / kuyrukta olanları logla
    for s in ctrl.stations:
        if s.current_ev:
            ev  = s.current_ev
            sid = ev.session_id
            vehicle_events[sid].update({
                "final_soc_pct":        round(ev.current_soc * 100, 1),
                "energy_delivered_kwh": round(ev.energy_delivered_kwh, 2),
                "charge_minutes":       1440 - (ev.charge_start_minute or 1440),
                "status":               "Yarım Kaldı",
            })
    for ev in ctrl.queue:
        vehicle_events[ev.session_id]["status"] = "Şarj Edilemedi"

    vehicle_log = sorted(vehicle_events.values(), key=lambda x: x["arrival_minute"])
    return snapshots, vehicle_log


# ══════════════════════════════════════════════════════════════════
#  ALGORİTMASIZ — sadece güç profili + metrikler
# ══════════════════════════════════════════════════════════════════

def run_unmanaged(schedule: dict, bg_load: np.ndarray, config: ScenarioConfig) -> dict:
    """Algoritmasız simülasyon — güç profili + araç logu döner."""
    policy   = config.to_grid_limit_policy()
    stations = copy.deepcopy(config.layout.stations)
    ctrl     = UnmanagedController(stations, policy, bg_load)
    vehicle_events: dict = {}

    for minute in range(1440):
        for ev in schedule.get(minute, []):
            ctrl.queue.append(ev)
            vehicle_events[ev.session_id] = {
                "session_id":           ev.session_id,
                "model_name":           ev.model_name,
                "battery_kwh":          ev.battery_capacity_kwh,
                "arrival_minute":       minute,
                "arrival_time":         _fmt(minute),
                "initial_soc_pct":      round(ev.initial_soc * 100, 1),
                "target_soc_pct":       round(ev.target_soc * 100, 1),
                "charge_start_minute":  None,
                "charge_start_time":    None,
                "wait_minutes":         None,
                "charge_minutes":       None,
                "departure_time":       None,
                "final_soc_pct":        None,
                "energy_delivered_kwh": None,
                "status":               "Kuyrukta",
            }

        # Boş istasyonlara araç ata
        for s in ctrl.stations:
            if not s.current_ev and ctrl.queue:
                s.current_ev = ctrl.queue.pop(0)
                s.current_ev.charge_start_minute = minute
                sid = s.current_ev.session_id
                vehicle_events[sid]["charge_start_minute"] = minute
                vehicle_events[sid]["charge_start_time"]   = _fmt(minute)
                vehicle_events[sid]["wait_minutes"]         = minute - vehicle_events[sid]["arrival_minute"]
                vehicle_events[sid]["status"]               = "Şarjda"

        tod     = minute % 1440
        is_peak = policy.peak_start_min <= tod < policy.peak_end_min
        limit   = policy.evening_peak_kw if is_peak else policy.trafo_max_kw
        bg      = float(bg_load[minute])
        allocs  = {s.station_id: s.effective_max_power_kw() for s in ctrl.stations}

        for s in ctrl.stations:
            if s.current_ev:
                s.current_ev.apply_power(allocs[s.station_id], minute)
                if s.current_ev.is_satisfied:
                    s.current_ev.departure_minute = minute
                    ctrl.completed.append(s.current_ev)
                    sid = s.current_ev.session_id
                    vehicle_events[sid].update({
                        "departure_time":       _fmt(minute),
                        "charge_minutes":       minute - (s.current_ev.charge_start_minute or minute),
                        "final_soc_pct":        round(s.current_ev.current_soc * 100, 1),
                        "energy_delivered_kwh": round(s.current_ev.energy_delivered_kwh, 2),
                        "status":               "Tamamlandı",
                    })
                    s.current_ev = None

        ev_load = float(sum(allocs.values()))
        ctrl.power_log.append(bg + ev_load)
        ctrl.limit_log.append(limit)

    for s in ctrl.stations:
        if s.current_ev:
            ev = s.current_ev
            vehicle_events[ev.session_id].update({
                "final_soc_pct":        round(ev.current_soc * 100, 1),
                "energy_delivered_kwh": round(ev.energy_delivered_kwh, 2),
                "charge_minutes":       1440 - (ev.charge_start_minute or 1440),
                "status":               "Yarım Kaldı",
            })
    for ev in ctrl.queue:
        vehicle_events[ev.session_id]["status"] = "Şarj Edilemedi"

    p    = np.array(ctrl.power_log)
    lim  = np.array(ctrl.limit_log)
    over = p > lim
    return {
        "power_log":        [round(float(v), 1) for v in p],
        "completed_count":  int(len(ctrl.completed)),
        "overload_minutes": int(over.sum()),
        "total_energy":     round(float(sum(e.energy_delivered_kwh for e in ctrl.completed)), 1),
        "avg_wait":         round(float(np.mean([e.wait_time_minutes for e in ctrl.completed])) if ctrl.completed else 0.0, 1),
        "peak_power":       round(float(p.max()), 1),
        "vehicle_log":      sorted(vehicle_events.values(), key=lambda x: x["arrival_minute"]),
    }


def _fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


# ══════════════════════════════════════════════════════════════════
#  VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════

SCENARIO_MAP = {
    "🏪 AVM (Orta)":   "avm_medium",
    "🏢 Ofis (Büyük)": "office_large",
    "🏨 Otel":         "hotel",
    "🏥 Hastane":      "hospital",
    "✈️ Havalimanı":   "airport",
}

DATASET_FILE = os.path.join(os.path.dirname(__file__), "DATASET", "dataset.json")


def load_and_simulate(scenario_key: str, generate_new: bool):
    scenario_fn = getattr(Scenarios, scenario_key)
    config: ScenarioConfig = scenario_fn()

    if not generate_new and os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            data = json.load(f)
        schedule: dict = {}
        for v in data["vehicles"]:
            m = v["arrival_minute"]
            schedule.setdefault(m, []).append(EV(
                v["session_id"], v["model_name"],
                v["battery_capacity_kwh"], v["max_dc_power_kw"],
                m, v["initial_soc"], target_soc=config.fleet.target_soc,
            ))
        bg_load = np.array(data["background_load_profile"])
    else:
        rng      = np.random.default_rng(42)
        schedule = ArrivalGenerator(config.fleet).generate_arrivals(rng)
        bg_load  = BackgroundLoadGenerator.generate(np.random.default_rng(101), config.environment)

    srpt_snaps, srpt_vlog = build_snapshots(copy.deepcopy(schedule), bg_load, config)
    unmanaged              = run_unmanaged(copy.deepcopy(schedule), bg_load, config)
    return srpt_snaps, unmanaged, srpt_vlog, unmanaged["vehicle_log"]


# ══════════════════════════════════════════════════════════════════
#  UI YARDIMCILARI
# ══════════════════════════════════════════════════════════════════

def soc_color(soc: float) -> str:
    if soc >= 0.65: return "#22c55e"
    if soc >= 0.35: return "#f59e0b"
    return "#ef4444"


def power_ratio_color(total: float, limit: float) -> str:
    r = total / limit if limit > 0 else 0
    if r >= 1.0:  return "#ef4444"
    if r >= 0.85: return "#f59e0b"
    return "#22c55e"


def station_type_badge(stype: str) -> str:
    return {"ultra_fast": "⚡ Ultra Hızlı", "fast": "🔋 Hızlı", "standard": "🔌 Standart"}.get(stype, stype)


def render_station_card(station: dict):
    if station["occupied"] and station["ev"]:
        ev     = station["ev"]
        soc    = ev["current_soc"]
        col    = soc_color(soc)
        border = "#3b82f6" if ev.get("is_new") else "#22c55e"
        new_badge = (
            '<span style="background:#1d4ed8;color:#bfdbfe;padding:1px 6px;'
            'border-radius:8px;font-size:0.68em;margin-left:5px;">YENİ</span>'
            if ev.get("is_new") else ""
        )
        html = f"""
        <div style="border:2px solid {border};border-radius:10px;padding:13px;
                    background:#0f2027;color:#e2e8f0;min-height:245px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b style="font-size:1.05em">{station['id']}</b>
            <span style="background:{border};color:#fff;padding:2px 9px;border-radius:12px;font-size:0.72em;">DOLU{new_badge}</span>
          </div>
          <div style="font-size:0.75em;color:#94a3b8;margin-bottom:6px;">
            {station_type_badge(station['type'])} &nbsp;|&nbsp; Max {station['max_power_kw']:.0f} kW
          </div>
          <hr style="border-color:#1e3a5f;margin:6px 0;">
          <div style="font-size:0.85em;margin-bottom:8px;">
            🚗 <b>{ev['model_name']}</b>
            <span style="color:#64748b;font-size:0.8em">&nbsp;#{ev['session_id']}</span>
          </div>
          <div style="margin:8px 0 10px;">
            <div style="display:flex;justify-content:space-between;font-size:0.78em;margin-bottom:3px;">
              <span style="color:#94a3b8">SoC</span>
              <span style="color:{col}"><b>{soc*100:.1f}%</b>
                <span style="color:#64748b"> / {ev['target_soc']*100:.0f}%</span>
              </span>
            </div>
            <div style="background:#1e293b;border-radius:4px;height:10px;overflow:hidden;">
              <div style="width:{min(soc*100,100):.1f}%;height:100%;
                          background:linear-gradient(90deg,{col}99,{col});border-radius:4px;"></div>
            </div>
          </div>
          <div style="font-size:0.78em;color:#cbd5e1;line-height:1.9;">
            ⚡ Atanan güç: <b>{station['power_kw']:.1f} kW</b><br>
            🔋 Verilen: <b>{ev['energy_delivered_kwh']:.2f} kWh</b><br>
            ⏳ Kalan: <b>{ev['energy_needed_kwh']:.2f} kWh</b><br>
            🕐 Şarjda: <b>{ev['charge_time_minutes']} dk</b> &nbsp;|&nbsp; Bekledi: <b>{ev['wait_time_minutes']} dk</b><br>
            🌡 Batarya: <b>{ev.get('battery_temp_c', 25.0):.1f}°C</b>
          </div>
        </div>"""
    else:
        html = f"""
        <div style="border:2px solid #1e293b;border-radius:10px;padding:13px;
                    background:#0a0f1a;color:#475569;min-height:245px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b style="font-size:1.05em;color:#64748b">{station['id']}</b>
            <span style="background:#1e293b;color:#64748b;padding:2px 9px;border-radius:12px;font-size:0.72em;">BOŞ</span>
          </div>
          <div style="font-size:0.75em;margin-bottom:6px;">
            {station_type_badge(station['type'])} &nbsp;|&nbsp; Max {station['max_power_kw']:.0f} kW
          </div>
          <hr style="border-color:#0f172a;margin:6px 0;">
          <div style="text-align:center;margin-top:50px;color:#334155;font-size:1.4em;">
            🔌<br><span style="font-size:0.55em">Araç Bekleniyor</span>
          </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_power_chart(snapshots: list, unmanaged: dict, current_minute: int) -> go.Figure:
    """Tam 24 saat ekseni; o ana kadar SRPT + Algoritmasız karşılaştırması."""
    all_times  = [s["time_str"] for s in snapshots]
    n          = current_minute + 1
    times_done = all_times[:n]

    bg_data     = [s["bg_load_kw"] for s in snapshots[:n]]
    limit_full  = [s["grid_limit_kw"] for s in snapshots]
    unman_total = unmanaged["power_log"][:n]

    # SRPT: sadece EV yükü > 0 olan dakikalarda çiz; öncesi None bırak
    srpt_total = [
        (s["bg_load_kw"] + s["ev_load_kw"]) if s["ev_load_kw"] > 0.01 else None
        for s in snapshots[:n]
    ]

    fig = go.Figure()

    # ── Grid Limiti (tüm gün) ──
    fig.add_trace(go.Scatter(
        x=all_times, y=limit_full,
        name="Grid Limiti",
        line=dict(color="#ef4444", width=2, dash="dash"),
        hovertemplate="%{y:.0f} kW<extra>Grid Limiti</extra>",
    ))

    # ── Algoritmasız toplam ──
    if n > 1:
        fig.add_trace(go.Scatter(
            x=times_done, y=unman_total,
            name="Algoritmasız",
            line=dict(color="#f97316", width=2),
            opacity=0.85,
            hovertemplate="%{y:.1f} kW<extra>Algoritmasız</extra>",
        ))

    # ── Baz yük (dolu alan) ──
    fig.add_trace(go.Scatter(
        x=times_done, y=bg_data,
        fill="tozeroy", name="Baz Yük",
        line=dict(color="#94a3b8", width=2.5),
        fillcolor="rgba(148,163,184,0.35)",
        hovertemplate="%{y:.1f} kW<extra>Baz Yük</extra>",
    ))

    # ── SRPT toplam (baz üstü alan; boşluklar None'dan oluşur) ──
    fig.add_trace(go.Scatter(
        x=times_done, y=srpt_total,
        fill="tonexty", name="Adaptif EV Yükü",
        line=dict(color="#38bdf8", width=2.5),
        fillcolor="rgba(56,189,248,0.40)",
        connectgaps=False,
        hovertemplate="%{y:.1f} kW<extra>Adaptif Toplam</extra>",
    ))

    # ── Anlık saat çizgisi ──
    fig.add_vline(
        x=snapshots[current_minute]["time_str"],
        line_color="#facc15", line_width=1.5, line_dash="dot",
    )

    fig.update_layout(
        uirevision="power_chart",   # zoom state korunur, rerun'da sıfırlanmaz
        height=270,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(
            range=[all_times[0], all_times[-1]],
            showgrid=True, gridcolor="#1e293b",
            tickvals=[all_times[i] for i in range(0, 1440, 60)],
            ticktext=[f"{i:02d}:00" for i in range(24)],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            title=dict(text="kW", font=dict(size=11)),
        ),
        hovermode="x unified",
    )
    return fig


def render_adaptive_decisions(decisions: list):
    st.markdown("### 🧠 Adaptif Kararlar")
    if not decisions:
        st.info("Bu dakikada aktif şarj yok.")
        return

    for d in decisions:
        if d["no_budget"]:
            st.markdown(
                f"<div style='background:#1c1008;border:1px solid #7c2d12;border-radius:8px;"
                f"padding:8px 12px;margin-bottom:8px;font-size:0.82em;color:#fed7aa;'>"
                f"⛔ <b>{d['station_id']}</b> — <b>{d['model_name']}</b> ({d['session_id']})&nbsp; "
                f"SoC: {d['current_soc_pct']:.1f}% | Kalan: {d['energy_needed_kwh']:.2f} kWh | "
                f"Bekleme: {d['wait_minutes']} dk<br>"
                f"<span style='color:#f97316'>Bütçe tükendi — bu dakikada güç ayrılamadı</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            zone_colors = {"Normal": "#22c55e", "Taper": "#f59e0b", "CC-CV": "#f97316"}
            zone_color  = zone_colors.get(d["taper_zone"], "#94a3b8")
            ramp_badge  = (
                '<span style="background:#854d0e;color:#fef3c7;padding:1px 5px;'
                'border-radius:5px;font-size:0.65em;margin-left:4px;">RAMP</span>'
                if d["is_ramp_active"] else ""
            )
            ca, cb = st.columns([1, 2])
            with ca:
                st.markdown(
                    f"<div style='background:#1e293b;border-radius:8px;padding:8px;text-align:center;'>"
                    f"<b style='color:#60a5fa'>{d['station_id']}</b><br>"
                    f"<span style='font-size:1.15em;font-weight:700;color:#22c55e'>{d['power_kw']} kW</span><br>"
                    f"<span style='font-size:0.72em;color:{zone_color};font-weight:600'>{d['taper_zone']}</span>"
                    f"{ramp_badge}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with cb:
                bat_temp = d.get("battery_temp_c", 25.0)
                aging    = d.get("aging_cost", 0.0)
                temp_col = "#ef4444" if bat_temp > 40 else ("#f59e0b" if bat_temp > 35 else "#94a3b8")
                st.markdown(
                    f"🚗 **{d['model_name']}**  \n"
                    f"SoC: **{d['current_soc_pct']:.1f}%** &nbsp;|&nbsp; "
                    f"Tavan: `{d['tapered_cap_kw']} kW` &nbsp;|&nbsp; "
                    f"C-rate: **{d['c_rate_actual']:.2f}C**  \n"
                    f"Kalan: `{d['energy_needed_kwh']:.2f}` kWh &nbsp;|&nbsp; "
                    f"Bekleme: {d['wait_minutes']} dk &nbsp;|&nbsp; "
                    f"Skor: **{d['priority_score']}**  \n"
                    f"<span style='color:{temp_col}'>🌡 {bat_temp:.1f}°C</span> &nbsp;|&nbsp; "
                    f"Yaşlanma: `{aging:.4f}`",
                    unsafe_allow_html=True,
                )
            st.divider()


def _build_live_log(vehicle_log: list, snap: dict, current_minute: int) -> list:
    """Verilen vehicle_log'u current_minute'a göre filtrele ve canlı station verisiyle güncelle."""
    arrived = {v["session_id"]: dict(v) for v in vehicle_log if v["arrival_minute"] <= current_minute}
    for st_data in snap["stations"]:
        if st_data["occupied"] and st_data["ev"]:
            ev  = st_data["ev"]
            sid = ev["session_id"]
            if sid in arrived:
                arrived[sid].update({
                    "final_soc_pct":        round(ev["current_soc"] * 100, 1),
                    "energy_delivered_kwh": ev["energy_delivered_kwh"],
                    "charge_minutes":       ev["charge_time_minutes"],
                    "status":               "Şarjda",
                })
    for qev in snap["queue"]:
        sid = qev["session_id"]
        if sid in arrived:
            arrived[sid]["status"] = "Kuyrukta"
    return sorted(arrived.values(), key=lambda x: x["arrival_minute"])


def _log_to_df(log_list: list, sort_by: str, status_filter: list) -> pd.DataFrame:
    filtered = [v for v in log_list if v["status"] in status_filter] if status_filter else log_list
    if sort_by == "Bekleme (uzundan kısaya)":
        filtered = sorted(filtered, key=lambda x: x.get("wait_minutes") or 0, reverse=True)
    elif sort_by == "Enerji (büyükten küçüğe)":
        filtered = sorted(filtered, key=lambda x: x.get("energy_delivered_kwh") or 0, reverse=True)
    rows = [{
        "Araç ID":          v["session_id"],
        "Model":            v["model_name"],
        "Batarya (kWh)":    v["battery_kwh"],
        "Geliş":            v["arrival_time"],
        "Şarj Başlangıç":   v.get("charge_start_time") or "-",
        "Ayrılış":          v.get("departure_time") or "-",
        "Bekleme (dk)":     v["wait_minutes"] if v.get("wait_minutes") is not None else "-",
        "Şarj Süresi (dk)": v.get("charge_minutes") if v.get("charge_minutes") is not None else "-",
        "İlk SoC (%)":      v["initial_soc_pct"],
        "Son SoC (%)":      v.get("final_soc_pct") or "-",
        "Verilen (kWh)":    v.get("energy_delivered_kwh") or "-",
        "Durum":            v["status"],
    } for v in filtered]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _status_color(val):
    return {
        "Tamamlandı":     "color: #22c55e",
        "Yarım Kaldı":    "color: #f59e0b",
        "Şarj Edilemedi": "color: #ef4444",
        "Şarjda":         "color: #60a5fa",
        "Kuyrukta":       "color: #94a3b8",
    }.get(val, "")


def render_vehicle_log(srpt_log: list, unman_log: list, snap: dict, current_minute: int):
    """SRPT | Algoritmasız | Karşılaştırma tablarıyla araç logunu göster."""
    label = f"📋 Araç Logları  ({current_minute // 60:02d}:{current_minute % 60:02d} itibarıyla)"
    with st.expander(label, expanded=False):
        # Filtre kontrolları (paylaşılan)
        fc1, fc2 = st.columns(2)
        with fc1:
            status_filter = st.multiselect(
                "Durum", ["Tamamlandı", "Yarım Kaldı", "Şarj Edilemedi", "Kuyrukta", "Şarjda"],
                default=["Tamamlandı", "Şarjda", "Kuyrukta", "Şarj Edilemedi"],
                key="log_status_filter",
            )
        with fc2:
            sort_by = st.selectbox(
                "Sırala", ["Geliş Saati", "Bekleme (uzundan kısaya)", "Enerji (büyükten küçüğe)"],
                key="log_sort",
            )

        srpt_live  = _build_live_log(srpt_log,  snap, current_minute)
        unman_live = _build_live_log(unman_log, snap, current_minute)

        tab_srpt, tab_unman, tab_diff = st.tabs(["⚡ Adaptif", "🔴 Algoritmasız", "📊 Karşılaştırma"])

        # ── Tab 1: SRPT ───────────────────────────────────────────
        with tab_srpt:
            df = _log_to_df(srpt_live, sort_by, status_filter)
            if df.empty:
                st.info("Gösterilecek araç yok.")
            else:
                st.dataframe(df.style.map(_status_color, subset=["Durum"]), use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("⬇️ CSV (Adaptif)", csv, file_name="adaptif_arac_log.csv", mime="text/csv")

        # ── Tab 2: Algoritmasız ───────────────────────────────────
        with tab_unman:
            df2 = _log_to_df(unman_live, sort_by, status_filter)
            if df2.empty:
                st.info("Gösterilecek araç yok.")
            else:
                st.dataframe(df2.style.map(_status_color, subset=["Durum"]), use_container_width=True, hide_index=True)
                csv2 = df2.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("⬇️ CSV (Algoritmasız)", csv2, file_name="unmanaged_arac_log.csv", mime="text/csv")

        # ── Tab 3: Karşılaştırma ──────────────────────────────────
        with tab_diff:
            srpt_map  = {v["session_id"]: v for v in srpt_live}
            unman_map = {v["session_id"]: v for v in unman_live}
            all_ids   = sorted(set(srpt_map) | set(unman_map),
                               key=lambda sid: (srpt_map.get(sid) or unman_map.get(sid))["arrival_minute"])

            diff_rows = []
            for sid in all_ids:
                sv = srpt_map.get(sid)
                uv = unman_map.get(sid)
                if not sv or not uv:
                    continue

                def _num(d, key):
                    v = d.get(key)
                    return float(v) if isinstance(v, (int, float)) else None

                sw  = _num(sv, "wait_minutes");    uw  = _num(uv, "wait_minutes")
                sc  = _num(sv, "charge_minutes");  uc  = _num(uv, "charge_minutes")
                se  = _num(sv, "energy_delivered_kwh"); ue = _num(uv, "energy_delivered_kwh")

                def _diff(a, b):
                    if a is None or b is None: return "-"
                    return f"{a - b:+.0f}"

                diff_rows.append({
                    "Araç ID":                sid,
                    "Model":                  sv["model_name"],
                    "Geliş":                  sv["arrival_time"],
                    "Bekleme Adaptif (dk)":      sw if sw is not None else "-",
                    "Bekleme Alg.sız (dk)":      uw if uw is not None else "-",
                    "Bekleme Farkı (dk)":         _diff(sw, uw),
                    "Şarj Süresi Adaptif (dk)":   sc if sc is not None else "-",
                    "Şarj Süresi Alg.sız (dk)":   uc if uc is not None else "-",
                    "Şarj Süresi Farkı (dk)":     _diff(sc, uc),
                    "Enerji Adaptif (kWh)":        se if se is not None else "-",
                    "Enerji Alg.sız (kWh)":        ue if ue is not None else "-",
                    "Enerji Farkı (kWh)":          _diff(se, ue),
                    "Durum Adaptif":              sv["status"],
                    "Durum Alg.sız":              uv["status"],
                })

            if not diff_rows:
                st.info("Karşılaştırılacak araç yok.")
            else:
                df3 = pd.DataFrame(diff_rows)

                def _color_diff(val):
                    if not isinstance(val, str) or not val.startswith(("+", "-")):
                        return ""
                    try:
                        n = float(val)
                        # Bekleme/Şarj farkı: negatif = SRPT daha iyi
                        return "color: #22c55e" if n < 0 else ("color: #ef4444" if n > 0 else "")
                    except Exception:
                        return ""

                def _color_energy_diff(val):
                    if not isinstance(val, str) or not val.startswith(("+", "-")):
                        return ""
                    try:
                        n = float(val)
                        # Enerji farkı: pozitif = SRPT daha fazla enerji verdi = iyi
                        return "color: #22c55e" if n > 0 else ("color: #ef4444" if n < 0 else "")
                    except Exception:
                        return ""

                styled3 = (
                    df3.style
                    .map(_color_diff,        subset=["Bekleme Farkı (dk)", "Şarj Süresi Farkı (dk)"])
                    .map(_color_energy_diff, subset=["Enerji Farkı (kWh)"])
                    .map(_status_color,      subset=["Durum Adaptif", "Durum Alg.sız"])
                )
                st.caption("Fark = Adaptif − Algoritmasız  |  🟢 negatif bekleme/şarj = Adaptif daha hızlı  |  🟢 pozitif enerji = Adaptif daha fazla şarj etti")
                st.dataframe(styled3, use_container_width=True, hide_index=True)
                csv3 = df3.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("⬇️ CSV (Karşılaştırma)", csv3, file_name="karsilastirma.csv", mime="text/csv")


def render_thermal_chart(snapshots: list, current_minute: int) -> go.Figure:
    """θ_hs / θ_oil / V(t) aging rate ve dinamik limit grafiği."""
    all_times = [s["time_str"] for s in snapshots]
    n         = current_minute + 1
    times     = all_times[:n]

    theta_hs_data  = [s["theta_hs"]   for s in snapshots[:n]]
    theta_oil_data = [s["theta_oil"]  for s in snapshots[:n]]
    aging_data     = [s["aging_rate"] for s in snapshots[:n]]
    limit_dyn      = [s["dynamic_limit_kw"] for s in snapshots[:n] if s.get("dynamic_limit_kw") is not None]
    limit_times    = [s["time_str"]         for s in snapshots[:n] if s.get("dynamic_limit_kw") is not None]

    fig = go.Figure()

    # ── Dinamik limit (sağ eksen değil; kW ölçeği ayrı) ──
    # Sol eksen: Sıcaklık (°C), Sağ eksen: V(t) aging
    fig.add_trace(go.Scatter(
        x=times, y=theta_hs_data,
        name="θ_hs (Hot-spot)",
        line=dict(color="#f97316", width=2.5),
        hovertemplate="%{y:.1f}°C<extra>θ_hs</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=theta_oil_data,
        name="θ_oil (Yağ)",
        line=dict(color="#fb923c", width=1.8, dash="dot"),
        hovertemplate="%{y:.1f}°C<extra>θ_oil</extra>",
    ))
    # IEC 98°C hedef çizgisi
    fig.add_hline(y=98.0, line_color="#ef4444", line_dash="dash", line_width=1.5,
                  annotation_text="IEC 98°C", annotation_font_color="#ef4444",
                  annotation_position="top right")
    # V(t) aging — sağ eksen
    fig.add_trace(go.Scatter(
        x=times, y=aging_data,
        name="V(t) Yaşlanma",
        line=dict(color="#a78bfa", width=2),
        yaxis="y2",
        hovertemplate="%{y:.3f}<extra>V(t)</extra>",
    ))
    # V(t)=1.0 referans
    fig.add_hline(y=1.0, line_color="#a78bfa", line_dash="dot", line_width=1,
                  yref="y2")

    fig.update_layout(
        uirevision="thermal_chart",
        height=260,
        margin=dict(l=0, r=60, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(
            range=[all_times[0], all_times[-1]],
            showgrid=True, gridcolor="#1e293b",
            tickvals=[all_times[i] for i in range(0, 1440, 60)],
            ticktext=[f"{i:02d}:00" for i in range(24)],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            title=dict(text="Sıcaklık (°C)", font=dict(size=11)),
        ),
        yaxis2=dict(
            overlaying="y", side="right",
            title=dict(text="V(t) Yaşlanma", font=dict(size=11, color="#a78bfa")),
            showgrid=False,
            tickfont=dict(color="#a78bfa"),
        ),
        hovermode="x unified",
    )
    return fig


def render_battery_temp_chart(snapshots: list, current_minute: int) -> go.Figure:
    """Dinamik trafo limiti ve ortalama batarya sıcaklığı grafiği."""
    all_times = [s["time_str"] for s in snapshots]
    n         = current_minute + 1
    times     = all_times[:n]

    dyn_limit  = [s.get("dynamic_limit_kw") or 0.0 for s in snapshots[:n]]
    total_pwr  = [s["total_power_kw"]              for s in snapshots[:n]]

    # Batarya sıcaklığı: aktif araçların ortalaması (adaptive_decisions'dan)
    bat_temps = []
    for s in snapshots[:n]:
        temps = [d["battery_temp_c"] for d in s["adaptive_decisions"] if "battery_temp_c" in d]
        bat_temps.append(float(np.mean(temps)) if temps else float("nan"))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_times[:n], y=dyn_limit,
        name="Dinamik Limit",
        line=dict(color="#ef4444", width=2, dash="dash"),
        hovertemplate="%{y:.0f} kW<extra>Dinamik Limit</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=total_pwr,
        name="Toplam Güç",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="%{y:.1f} kW<extra>Toplam Güç</extra>",
    ))
    # Batarya sıcaklığı — sağ eksen
    fig.add_trace(go.Scatter(
        x=times, y=bat_temps,
        name="Ort. Bat. Sıcaklık",
        line=dict(color="#facc15", width=2),
        yaxis="y2",
        connectgaps=False,
        hovertemplate="%{y:.1f}°C<extra>Bat. Sıcaklık</extra>",
    ))
    # 40°C BMS derating eşiği
    fig.add_hline(y=40.0, line_color="#f59e0b", line_dash="dot", line_width=1.2,
                  annotation_text="40°C BMS", annotation_font_color="#f59e0b",
                  annotation_position="top right", yref="y2")

    fig.update_layout(
        uirevision="bat_temp_chart",
        height=260,
        margin=dict(l=0, r=60, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(
            range=[all_times[0], all_times[-1]],
            showgrid=True, gridcolor="#1e293b",
            tickvals=[all_times[i] for i in range(0, 1440, 60)],
            ticktext=[f"{i:02d}:00" for i in range(24)],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            title=dict(text="kW", font=dict(size=11)),
        ),
        yaxis2=dict(
            overlaying="y", side="right",
            title=dict(text="Bat. Sıcaklık (°C)", font=dict(size=11, color="#facc15")),
            showgrid=False,
            tickfont=dict(color="#facc15"),
        ),
        hovermode="x unified",
    )
    return fig


def render_adaptive_all_decisions(snapshots: list, current_minute: int):
    """O ana kadar verilen tüm adaptif kararları tablo olarak göster."""
    with st.expander(f"📜 Tüm Adaptif Karar Geçmişi  (0–{_fmt(current_minute)})", expanded=False):
        rows = []
        for s in snapshots[:current_minute + 1]:
            for d in s["adaptive_decisions"]:
                rows.append({
                    "Saat":          s["time_str"],
                    "İstasyon":      d["station_id"],
                    "Araç":          d["session_id"],
                    "Model":         d["model_name"],
                    "Güç (kW)":      d["power_kw"],
                    "SoC (%)":       d["current_soc_pct"],
                    "Kalan (kWh)":   d["energy_needed_kwh"],
                    "Tavan (kW)":    d["tapered_cap_kw"],
                    "C-rate":        d["c_rate_actual"],
                    "Taper Bölge":   d["taper_zone"],
                    "Bekleme (dk)":  d["wait_minutes"],
                    "Skor":          d["priority_score"],
                    "Bat. °C":       d.get("battery_temp_c", "-"),
                    "Yaşlanma":      d.get("aging_cost", "-"),
                    "Ramp":          "✓" if d["is_ramp_active"] else "",
                    "Bütçe Yok":     "⛔" if d["no_budget"] else "",
                })

        if not rows:
            st.info("Henüz karar alınmadı.")
            return

        df = pd.DataFrame(rows)
        st.caption(f"Toplam **{len(rows)}** karar — son 300 gösteriliyor")

        def color_budget(val):
            return "color: #ef4444" if val == "⛔" else ""

        def color_zone(val):
            return {"Normal": "color: #22c55e", "Taper": "color: #f59e0b", "CC-CV": "color: #f97316"}.get(val, "")

        st.dataframe(
            df.tail(300).style
              .map(color_budget, subset=["Bütçe Yok"])
              .map(color_zone,   subset=["Taper Bölge"]),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════
#  ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="EV Şarj — Adaptif", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    html, body, .stApp { background-color: #0d1117 !important; }
    section[data-testid="stSidebar"] { background-color: #0f172a !important; }
    div[data-testid="metric-container"] { background:#1e293b; border-radius:8px; padding:8px 12px; }
    hr { border-color:#1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

    for k, v in [
        ("snapshots", None), ("unmanaged", None),
        ("vehicle_log", None), ("unmanaged_vlog", None),
        ("frame", 0), ("running", False),
        ("replay_start", 0), ("replay_end", 1439),
        ("last_frame_time", None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚡ Adaptif Simülasyon")
        st.divider()

        scenario_label = st.selectbox("Senaryo", list(SCENARIO_MAP.keys()))
        scenario_key   = SCENARIO_MAP[scenario_label]
        generate_new   = st.checkbox("Yeni rastgele veri üret", value=False)

        if st.button("🔄 Simülasyonu Hazırla", use_container_width=True, type="primary"):
            with st.spinner("Simülasyon çalışıyor (Adaptif + Algoritmasız)..."):
                try:
                    snaps, unman, vlog, uvlog = load_and_simulate(scenario_key, generate_new)
                    st.session_state.snapshots      = snaps
                    st.session_state.unmanaged      = unman
                    st.session_state.vehicle_log    = vlog
                    st.session_state.unmanaged_vlog = uvlog
                    st.session_state.frame   = 0
                    st.session_state.running = False
                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback; st.code(traceback.format_exc())
                    return
            st.success("✅ Hazır!")

        st.divider()
        st.markdown("**Oynatma**")
        speed = st.slider("Hız (dk/sn)", 1, 120, 20)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶ Oynat", use_container_width=True):
                if st.session_state.snapshots:
                    st.session_state.frame           = st.session_state.replay_start
                    st.session_state.running         = True
                    st.session_state.last_frame_time = None   # timer sıfırla
        with c2:
            if st.button("⏸ Dur", use_container_width=True):
                st.session_state.running = False
        with c3:
            if st.button("⏮ Sıfır", use_container_width=True):
                st.session_state.frame   = 0
                st.session_state.running = False

        if st.session_state.snapshots:
            pct = st.session_state.frame / 1439
            st.progress(pct, text=f"{_fmt(st.session_state.frame)} ({st.session_state.frame}/1439 dk)")

        st.divider()
        st.markdown("**Aralık Seçimi**")
        ca, cb = st.columns(2)
        with ca:
            rsh = st.number_input("Baş. saat", 0, 23, 0, key="rsh")
            rsm = st.number_input("Baş. dk",   0, 59, 0, key="rsm")
        with cb:
            reh = st.number_input("Bitiş saat", 0, 23, 23, key="reh")
            rem = st.number_input("Bitiş dk",   0, 59, 59, key="rem")

        r_start = int(rsh)*60 + int(rsm)
        r_end   = max(r_start+1, min(int(reh)*60 + int(rem), 1439))

        if st.button("▶ Bu Aralığı Oynat", use_container_width=True):
            if st.session_state.snapshots:
                st.session_state.replay_start    = r_start
                st.session_state.replay_end      = r_end
                st.session_state.frame           = r_start
                st.session_state.running         = True
                st.session_state.last_frame_time = None

        st.divider()
        st.markdown(
            "**Algoritma:** AdaptiveSoC Termal  \n"
            "*Trafo:* IEC 60076-7 θ_hs modeli, dinamik limit  \n"
            "*Batarya:* CC-CV taper, BMS derating, yaşlanma maliyeti  \n"
            "*Öncelik:* (bekleme/enerji) × C-rate + yaşlanma ağırlığı  \n"
            "*Termal:* θ_hs hedef 98°C, V(t) aging integral"
        )

    # ── İçerik yok ───────────────────────────────────────────────
    if st.session_state.snapshots is None:
        st.markdown(
            "<h2 style='color:#60a5fa;text-align:center;margin-top:80px'>⚡ EV Yük Dengeleme — Adaptif</h2>"
            "<p style='color:#64748b;text-align:center'>Sol menüden senaryo seçin ve <b>Simülasyonu Hazırla</b> butonuna basın.</p>",
            unsafe_allow_html=True,
        )
        return

    snap     = st.session_state.snapshots[st.session_state.frame]
    unman    = st.session_state.unmanaged
    vlog     = st.session_state.vehicle_log
    uvlog    = st.session_state.unmanaged_vlog
    last_snp = st.session_state.snapshots[-1]

    # ── Başlık ───────────────────────────────────────────────────
    if snap["is_peak"]:
        peak_html = (
            '<span style="background:#92400e;color:#fef3c7;padding:3px 12px;border-radius:6px;'
            'font-size:0.75em;margin-left:10px;">⚠️ PİK SAAT — Limit Kısıtlı</span>'
        )
    elif snap.get("is_ramp"):
        peak_html = (
            '<span style="background:#44403c;color:#fde68a;padding:3px 12px;border-radius:6px;'
            'font-size:0.75em;margin-left:10px;">📉 SOFT RAMP — Limit Lineer Düşüyor</span>'
        )
    else:
        peak_html = ""
    pct = int(snap["total_power_kw"] / snap["grid_limit_kw"] * 100) if snap["grid_limit_kw"] > 0 else 0

    st.markdown(
        f'<div style="padding:8px 0 6px;border-bottom:1px solid #1e293b;margin-bottom:12px;">'
        f'<span style="font-size:1.4em;font-weight:700;color:#e2e8f0;">⚡ EV Şarj — Adaptif</span>&nbsp;&nbsp;'
        f'<span style="font-size:1.9em;font-weight:900;color:#60a5fa;font-variant-numeric:tabular-nums;">'
        f'🕐 {snap["time_str"]}</span>'
        f'<span style="color:#475569;font-size:0.8em;margin-left:8px;">({snap["minute"]} / 1439 dk)</span>'
        f'{peak_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Metrik barı ───────────────────────────────────────────────
    occupied = len([s for s in snap["stations"] if s["occupied"]])
    ov       = snap["overload_kw"]

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("⚡ Toplam Güç",     f"{snap['total_power_kw']:.0f} kW",  f"%{pct} limit")
    m2.metric("🏭 Baz Yük",        f"{snap['bg_load_kw']:.0f} kW")
    m3.metric("🔋 EV Yükü",        f"{snap['ev_load_kw']:.0f} kW",      f"{occupied} aktif ist.")
    m4.metric("🚗 Kuyruk",         f"{len(snap['queue'])} araç")
    m5.metric("✅ Tamamlanan",      f"{snap['completed_count']} araç",   f"Ort. {snap['avg_wait_minutes']:.0f} dk bkl.")
    m6.metric("🚨 Aşım",
              f"{ov:.0f} kW" if ov > 0 else "✅ Yok",
              f"{snap['overload_total_minutes']} dk toplam aşım")
    m7.metric("🔌 Verilen Enerji",  f"{snap['total_energy_delivered_kwh']:.1f} kWh")

    # ── SRPT vs Algoritmasız karşılaştırma ──────────────────────
    srpt_final  = last_snp["total_energy_delivered_kwh"]
    unman_total = unman["total_energy"]
    diff_energy = round(srpt_final - unman_total, 1)
    diff_over   = unman["overload_minutes"] - last_snp["overload_total_minutes"]
    diff_wait   = round(unman["avg_wait"] - last_snp["avg_wait_minutes"], 1)

    st.markdown(
        f'<div style="background:#0f1f12;border:1px solid #166534;border-radius:8px;'
        f'padding:10px 16px;margin-bottom:8px;font-size:0.85em;color:#bbf7d0;">'
        f'<b>📊 Adaptif vs Algoritmasız (Gün Sonu Tahmini):</b>&nbsp;&nbsp;'
        f'Toplam Enerji: <b>Adaptif {srpt_final:.1f}</b> / Alg.sız {unman_total:.1f} kWh '
        f'<span style="color:{"#22c55e" if diff_energy >= 0 else "#ef4444"}">'
        f'({"+" if diff_energy >= 0 else ""}{diff_energy:.1f} kWh)</span>'
        f'&nbsp;|&nbsp; Aşım Azalması: <b style="color:#22c55e">-{diff_over} dk</b>'
        f'&nbsp;|&nbsp; Ort. Bekleme Azalması: <b style="color:#22c55e">-{diff_wait:.1f} dk</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Termal durum şeridi ──────────────────────────────────────
    ths      = snap["theta_hs"]
    toil     = snap["theta_oil"]
    aging    = snap["aging_rate"]
    peak_ths = max(s["theta_hs"] for s in st.session_state.snapshots[:snap["minute"] + 1])
    ths_col  = "#ef4444" if ths >= 98 else ("#f59e0b" if ths >= 90 else "#22c55e")
    overshoot_badge = (
        '&nbsp; <span style="background:#7f1d1d;color:#fca5a5;padding:1px 7px;border-radius:5px">⚠️ AŞIM</span>'
        if ths >= 98 else ""
    )
    st.markdown(
        f'<div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;'
        f'padding:8px 16px;margin-bottom:12px;font-size:0.82em;color:#94a3b8;">'
        f'🌡 <b>Trafo Termal:</b>&nbsp;&nbsp;'
        f'θ_hs anlık: <b style="color:{ths_col}">{ths:.1f}°C</b> &nbsp;|&nbsp; '
        f'θ_oil: <b>{toil:.1f}°C</b> &nbsp;|&nbsp; '
        f'θ_hs pik (bugün): <b style="color:#f97316">{peak_ths:.1f}°C</b> &nbsp;|&nbsp; '
        f'V(t) yaşlanma: <b style="color:#a78bfa">{aging:.3f}</b>'
        f'{overshoot_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── İstasyon kartları ─────────────────────────────────────────
    st.markdown("### 🔌 Şarj İstasyonları")
    cols = st.columns(len(snap["stations"]))
    for col, station in zip(cols, snap["stations"]):
        with col:
            render_station_card(station)

    st.divider()

    # ── Güç grafiği + SRPT kararları ─────────────────────────────
    col_chart, col_srpt = st.columns([3, 2])
    with col_chart:
        st.markdown("### 📈 Güç Profili — 24 Saat")
        fig = render_power_chart(st.session_state.snapshots, unman, snap["minute"])
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        })
    with col_srpt:
        render_adaptive_decisions(snap["adaptive_decisions"])

    # ── Termal grafikler ──────────────────────────────────────────
    st.divider()
    st.markdown("### 🌡 Termal Durum — Trafo & Batarya")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Trafo Hot-spot & Yaşlanma**")
        fig_th = render_thermal_chart(st.session_state.snapshots, snap["minute"])
        st.plotly_chart(fig_th, use_container_width=True, config={
            "scrollZoom": True, "displayModeBar": False,
        })
    with tc2:
        st.markdown("**Dinamik Limit & Batarya Sıcaklığı**")
        fig_bt = render_battery_temp_chart(st.session_state.snapshots, snap["minute"])
        st.plotly_chart(fig_bt, use_container_width=True, config={
            "scrollZoom": True, "displayModeBar": False,
        })

    # ── Kuyruk ───────────────────────────────────────────────────
    if snap["queue"]:
        st.markdown(f"### 🚗 Kuyruk — {len(snap['queue'])} araç")
        rows = sorted(snap["queue"], key=lambda x: -x["wait_minutes"])
        df = pd.DataFrame([{
            "Araç ID":       r["session_id"],
            "Model":         r["model_name"],
            "Geliş":         _fmt(r["arrival_minute"]),
            "Bekleme (dk)":  r["wait_minutes"],
            "İlk SoC":       f"{r['initial_soc_pct']:.1f}%",
            "Gerekli (kWh)": r["energy_needed_kwh"],
            "Batarya (kWh)": r["battery_capacity_kwh"],
        } for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif snap["minute"] > 0:
        st.markdown("<div style='color:#22c55e;font-size:0.9em;padding:4px 0'>✅ Kuyruk boş</div>", unsafe_allow_html=True)

    # ── Araç logları + SRPT geçmişi ───────────────────────────────
    if vlog and uvlog:
        render_vehicle_log(vlog, uvlog, snap, snap["minute"])
    render_adaptive_all_decisions(st.session_state.snapshots, snap["minute"])

    # ── Animasyon ─────────────────────────────────────────────────
    if st.session_state.running:
        replay_end = st.session_state.get("replay_end", 1439)
        now        = time.time()
        last       = st.session_state.last_frame_time or now
        elapsed    = now - last

        # Geçen sürede kaç frame ilerlememiz gerektiğini hesapla
        to_advance = max(1, int(elapsed * speed))
        new_frame  = min(st.session_state.frame + to_advance, replay_end)

        st.session_state.frame           = new_frame
        st.session_state.last_frame_time = now

        if new_frame >= replay_end:
            st.session_state.running = False
        else:
            time.sleep(0.05)   # ~20 fps sabit, hız frame adımıyla ayarlanır
            st.rerun()


if __name__ == "__main__":
    main()
