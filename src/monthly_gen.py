#!/usr/bin/env python3
"""Aylık veri üreticisi — generators.py'den bağımsız."""

from __future__ import annotations
import numpy as np


class MonthlyDataGenerator:
    """30 günlük senaryo verisi üretir — haftaiçi/haftasonu farklı profil.

    Haziran 2026 takvimi baz alınır: 1 Haziran = Pazartesi.
    """

    WEEKDAY_NAMES = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    _START_WEEKDAY = 0  # 1 Haziran 2026 = Pazartesi

    def __init__(self, config, n_days: int = 30):
        self.config = config
        self.n_days = n_days

    def _weekday(self, day_idx: int) -> int:
        return (self._START_WEEKDAY + day_idx) % 7

    def _is_weekend(self, day_idx: int) -> bool:
        return self._weekday(day_idx) >= 5

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

    def generate(self, rng) -> list:
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
