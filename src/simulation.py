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

# Script klasör içinden doğrudan çalıştırıldığında import hatalarını önlemek için:
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    EVState, StationType, EVModel, GridLimitPolicy, MetricsSummary,
    VehicleSession, SimulationResult, EV, ChargingStation, EnvironmentProfile,
    GridConfig, ArrivalPattern, FleetProfile, StationLayout, ScenarioConfig
)

from generators import (
    ArrivalGenerator, BackgroundLoadGenerator, Scenarios
)

from controllers import (
    UnmanagedController, ManagedController, SRPTController,
    WaterFillingController, DynamicFairController, Simulation as Sim
)

from export import (
    build_station_matrix, export_comparative_excel, export_multi_controller_excel,
    ExecutiveDashboard
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']


def main(generate_new: bool = False, config: Optional[ScenarioConfig] = None):
    """Ana simülasyon fonksiyonu."""
    if config is None:
        config = Scenarios.avm_medium()
    print(f"EV Yük Dengeleme Simülasyonu başlatılıyor... Senaryo: {config.name}")
    dataset_file = "../DATASET/dataset.json"
    policy = config.to_grid_limit_policy()

    if not generate_new:
        if not os.path.exists(dataset_file):
            print(f"\nHATA: '{dataset_file}' dosyası yok! '--generate-new' ile oluşturun.\n")
            sys.exit(1)
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        schedule = {}
        for v in data['vehicles']:
            m = v['arrival_minute']
            if m not in schedule: schedule[m] = []
            schedule[m].append(EV(v['session_id'], v['model_name'], v['battery_capacity_kwh'],
                                  v['max_dc_power_kw'], m, v['initial_soc'],
                                  target_soc=config.fleet.target_soc))
        bg_load = np.array(data['background_load_profile'])
    else:
        rng = np.random.default_rng(42)
        schedule = ArrivalGenerator(config.fleet).generate_arrivals(rng)
        bg_load = BackgroundLoadGenerator.generate(np.random.default_rng(101), config.environment)
        vehicles_data = [{"session_id": ev.session_id, "model_name": ev.model_name,
                          "battery_capacity_kwh": ev.battery_capacity_kwh,
                          "max_dc_power_kw": ev.max_dc_power_kw,
                          "arrival_minute": ev.arrival_minute, "initial_soc": ev.initial_soc}
                         for m in sorted(schedule.keys()) for ev in schedule[m]]
        with open(dataset_file, 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "scenario": config.name,
                       "vehicles": vehicles_data, "background_load_profile": bg_load.tolist()}, f, indent=2)

    st_a = config.layout.stations
    st_b = copy.deepcopy(st_a)
    st_c = copy.deepcopy(st_a)
    st_d = copy.deepcopy(st_a)
    st_e = copy.deepcopy(st_a)

    ctrl_a = UnmanagedController(st_a, policy, bg_load)
    res_a = Sim(ctrl_a, copy.deepcopy(schedule)).run()

    ctrl_b = ManagedController(st_b, policy, bg_load)
    res_b = Sim(ctrl_b, copy.deepcopy(schedule)).run()

    ctrl_c = SRPTController(st_c, policy, bg_load)
    res_c = Sim(ctrl_c, copy.deepcopy(schedule)).run()

    ctrl_d = WaterFillingController(st_d, policy, bg_load)
    res_d = Sim(ctrl_d, copy.deepcopy(schedule)).run()

    ctrl_e = DynamicFairController(st_e, policy, bg_load)
    res_e = Sim(ctrl_e, copy.deepcopy(schedule)).run()

    # Orijinal ikili karşılaştırma raporunu koru
    export_comparative_excel(ctrl_a, ctrl_b)

    # Tüm 5 kontrolcünün karşılaştırmalı raporu
    all_ctrls = [
        ("Algoritmasiz", ctrl_a),
        ("Yonetimli",    ctrl_b),
        ("SRPT",         ctrl_c),
        ("Su_Doldurma",  ctrl_d),
        ("Dinamik_Adil", ctrl_e),
    ]
    export_multi_controller_excel(all_ctrls)

    # Konsol özeti
    print(f"\n{'Kontrolcü':<20} {'Araç':>6} {'MaksGüç(kW)':>12} {'AşımDk':>8} {'AşımkWh':>9} {'OrtBkl(dk)':>11} {'OrtŞarj(dk)':>12}")
    print("-" * 82)
    for name, ctrl in all_ctrls:
        p = np.array(ctrl.power_log); l = np.array(ctrl.limit_log); over = p > l
        c = ctrl.completed
        avw = np.mean([e.wait_time_minutes for e in c]) if c else 0.0
        avc = np.mean([e.charge_minutes   for e in c]) if c else 0.0
        print(f"{name:<20} {len(c):>6} {p.max():>12.1f} {int(over.sum()):>8} {float(np.where(over,p-l,0).sum()/60):>9.2f} {avw:>11.1f} {avc:>12.1f}")

    ExecutiveDashboard.create(res_a, res_b, ctrl_label="Yönetimli",    filename="../OutputPNG/dashboard_yonetimli.png",    bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_c, ctrl_label="SRPT",         filename="../OutputPNG/dashboard_srpt.png",         bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_d, ctrl_label="Su Doldurma",  filename="../OutputPNG/dashboard_su_doldurma.png",  bg_load=bg_load)
    ExecutiveDashboard.create(res_a, res_e, ctrl_label="Dinamik Adil", filename="../OutputPNG/dashboard_dinamik_adil.png", bg_load=bg_load)

    # Tüm pencerelerin ekranda aynı anda kalması için programı en sonda beklet:
    plt.show()


if __name__ == "__main__":
    _SCENARIO_MAP = {
        "avm_medium":   Scenarios.avm_medium,
        "office_large": Scenarios.office_large,
        "hotel":        Scenarios.hotel,
        "hospital":     Scenarios.hospital,
        "airport":      Scenarios.airport,
    }
    parser = argparse.ArgumentParser(description="EV Yük Dengeleme")
    parser.add_argument("--generate-new", action="store_true")
    parser.add_argument("--scenario", default="avm_medium", choices=list(_SCENARIO_MAP.keys()))
    parser.add_argument("--scenario-file", help="JSON senaryo dosyası")
    args = parser.parse_args()

    if args.scenario_file:
        with open(args.scenario_file, encoding="utf-8") as f:
            cfg = ScenarioConfig.from_dict(json.load(f))
    else:
        cfg = _SCENARIO_MAP[args.scenario]()

    main(generate_new=args.generate_new, config=cfg)
