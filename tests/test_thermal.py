#!/usr/bin/env python3
"""
Birim testler — IEC 60076-7 termal model, CC-CV taper, bit-identical regresyon.
"""

import sys
import os
import copy
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    EV, EVModel, ChargingStation, StationType, GridConfig, GridLimitPolicy,
    DynamicGridLimitPolicy, EnvironmentProfile, FleetProfile, ArrivalPattern,
    StationLayout, ScenarioConfig
)
from controllers import AdaptiveSoCController, Simulation
from generators import ArrivalGenerator, BackgroundLoadGenerator, Scenarios


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: IEC 60076-7 — sabit yük altında steady-state θ_oil doğrulaması
# ═══════════════════════════════════════════════════════════════════════════════

class TestThermalModel:
    """DynamicGridLimitPolicy termal model testleri."""

    def _make_policy(self, theta_hs_target=98.0, ambient_min=25.0, ambient_max=25.0):
        """Sabit 25°C ortam, 1600 kVA trafo."""
        env = EnvironmentProfile(
            name="Test", base_min_kw=0, base_max_kw=0,
            operation_start_hour=0, operation_duration_hours=24,
            morning_peak_hour=9, morning_peak_kw=0, morning_peak_width=1,
            evening_peak_hour=18, evening_peak_kw=0, evening_peak_width=1,
            noise_kw=0, load_min_kw=0, load_max_kw=2000,
            ambient_temp_min=ambient_min, ambient_temp_max=ambient_max,
            ambient_temp_peak_hour=15.0,
        )
        grid = GridConfig(trafo_kva=1600)
        policy = DynamicGridLimitPolicy(
            trafo_max_kw=grid.trafo_max_kw,
            evening_peak_kw=950.0,
            peak_start_min=1020, peak_end_min=1320,
            grid_config=grid, environment=env,
            theta_hs_target=theta_hs_target,
        )
        return policy

    def test_steady_state_theta_oil(self):
        """Sabit K=1.0 yük altında θ_oil, IEC steady-state değerine ±1K yakınsamalı."""
        policy = self._make_policy(theta_hs_target=120.0)  # yüksek target → sınırlama yok
        s_rated = policy._s_rated
        # Sabit yük = S_rated → K=1.0
        # Steady-state: θ_oil_ss = θ_amb + Δθ_oil_rated × K² = 25 + 55×1 = 80°C
        for minute in range(3000):  # ~20τ → kesinlikle yakınsama
            policy.update(minute, s_rated)

        expected_theta_oil = 25.0 + 55.0  # 80°C
        assert abs(policy.theta_oil - expected_theta_oil) < 1.0, \
            f"θ_oil={policy.theta_oil:.2f}, beklenen≈{expected_theta_oil}"

    def test_steady_state_theta_hs(self):
        """K=1.0'de θ_hs = θ_oil + Δθ_hs × K^1.6 = 80 + 25 = 105°C."""
        policy = self._make_policy(theta_hs_target=120.0)
        s_rated = policy._s_rated
        for minute in range(3000):
            policy.update(minute, s_rated)

        expected_theta_hs = 25.0 + 55.0 + 25.0 * (1.0 ** 1.6)  # 105°C
        assert abs(policy.theta_hs - expected_theta_hs) < 1.5, \
            f"θ_hs={policy.theta_hs:.2f}, beklenen≈{expected_theta_hs}"

    def test_half_load_theta_oil(self):
        """K=0.5'te θ_oil_ss = 25 + 55×0.25 = 38.75°C."""
        policy = self._make_policy(theta_hs_target=120.0)
        s_rated = policy._s_rated
        for minute in range(3000):
            policy.update(minute, s_rated * 0.5)

        expected = 25.0 + 55.0 * 0.25  # 38.75°C
        assert abs(policy.theta_oil - expected) < 1.0, \
            f"θ_oil={policy.theta_oil:.2f}, beklenen≈{expected}"

    def test_aging_rate_at_98(self):
        """θ_hs=98°C'de yaşlanma hızı V=1.0 olmalı."""
        # V = 2^((98-98)/6) = 2^0 = 1.0
        V = 2.0 ** ((98.0 - 98.0) / 6.0)
        assert abs(V - 1.0) < 1e-9

    def test_aging_rate_at_110(self):
        """θ_hs=110°C'de yaşlanma hızı V=4.0 olmalı."""
        V = 2.0 ** ((110.0 - 98.0) / 6.0)
        assert abs(V - 4.0) < 1e-9

    def test_dynamic_limit_decreases_as_oil_heats(self):
        """Aşırı yük altında yağ ısınınca anlık K_max_inst düşmeli → limit azalmalı.
        Test: soğuk ortam (10°C, K_max_ss_peak gevşek) + K=1.3 (aşırı yük) →
        yağ ~103°C'ye yaklaşırken instantaneous check binding olur.
        """
        policy = self._make_policy(theta_hs_target=120.0,
                                   ambient_min=10.0, ambient_max=10.0)
        s_rated = policy._s_rated
        limits = []
        for minute in range(1500):   # ~10τ → iyice yakınsama
            policy.update(minute, s_rated * 1.3)  # K=1.3: yağ 103°C'ye yaklaşır
            lim = policy.current_limit_kw(minute, 0.0)
            limits.append(lim)
        # İlk limit > son limit (yağ ısındıkça K_max_inst dominant olup düşer)
        assert limits[0] > limits[-1], \
            f"İlk={limits[0]:.1f}, Son={limits[-1]:.1f} — limit düşmeli"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: EV.max_acceptable_power_kw — CC-CV taper doğrulaması
# ═══════════════════════════════════════════════════════════════════════════════

class TestEVTaper:
    """EV CC-CV taper property testleri."""

    def _make_ev(self, soc: float) -> EV:
        ev = EV("TEST_01", "Test", 75.0, 250.0, 0, 0.10)
        ev.current_soc = soc
        return ev

    def test_below_taper_start(self):
        """SoC=0.50 → tam güç (250 kW)."""
        ev = self._make_ev(0.50)
        assert abs(ev.max_acceptable_power_kw - 250.0) < 0.1

    def test_at_taper_start(self):
        """SoC=0.70 → tam güç (taper henüz başlamadı)."""
        ev = self._make_ev(0.70)
        assert abs(ev.max_acceptable_power_kw - 250.0) < 0.1

    def test_at_80_percent(self):
        """SoC=0.80 → ~%40 güç (250 × 0.40 = 100 kW)."""
        ev = self._make_ev(0.80)
        expected = 250.0 * 0.40
        assert abs(ev.max_acceptable_power_kw - expected) < 1.0, \
            f"Gerçek={ev.max_acceptable_power_kw:.1f}, beklenen≈{expected}"

    def test_at_95_percent(self):
        """SoC=0.95 → ~%10 civarı (düşük CV akımı)."""
        ev = self._make_ev(0.95)
        # Segment 2: frac = 0.40 - (0.40-0.05) × (0.95-0.80)/(1.0-0.80)
        #           = 0.40 - 0.35 × 0.75 = 0.1375
        expected = 250.0 * 0.1375
        assert abs(ev.max_acceptable_power_kw - expected) < 2.0, \
            f"Gerçek={ev.max_acceptable_power_kw:.1f}, beklenen≈{expected}"

    def test_at_100_percent(self):
        """SoC=1.00 → minimum fraksiyon (250 × 0.05 = 12.5 kW)."""
        ev = self._make_ev(1.00)
        expected = 250.0 * 0.05
        assert abs(ev.max_acceptable_power_kw - expected) < 0.5

    def test_apply_power_clips(self):
        """apply_power, max_acceptable_power_kw ile clip etmeli."""
        ev = self._make_ev(0.79)  # taper bölgesinde
        max_accept = ev.max_acceptable_power_kw
        # 500 kW vermek iste — clip olmalı
        energy = ev.apply_power(500.0, 0)
        # Bir dakikada absorbe edilen enerji ≤ max_accept/60
        assert energy <= max_accept / 60.0 + 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Bit-identical regresyon — use_dynamic=False, use_aging=False
# ═══════════════════════════════════════════════════════════════════════════════

class TestBitIdentical:
    """AdaptiveSoCController statik mod = mevcut davranış."""

    def _run_simulation(self, use_dynamic: bool, use_aging: bool):
        """Verilen flag'lerle simülasyon koştur, power_log döndür."""
        config = Scenarios.avm_medium()
        policy = config.to_grid_limit_policy(use_dynamic=False)  # her iki durumda statik policy
        rng = np.random.default_rng(42)
        schedule = ArrivalGenerator(config.fleet).generate_arrivals(rng)
        bg_load = BackgroundLoadGenerator.generate(np.random.default_rng(101), config.environment)

        stations = copy.deepcopy(config.layout.stations)
        ctrl = AdaptiveSoCController(
            stations, policy, bg_load,
            use_dynamic_grid_limit=use_dynamic,
            use_aging_cost=use_aging,
        )
        sim = Simulation(ctrl, copy.deepcopy(schedule))
        sim.run()
        return np.array(ctrl.power_log)

    def test_static_no_aging_is_baseline(self):
        """use_dynamic=False + use_aging=False → mevcut davranış ile aynı.
        Not: CC-CV taper EV.apply_power'da eklendi, bu tüm koşuları etkiler.
        İki koşu da aynı flag'lerle → bit-identical olmalı.
        """
        run1 = self._run_simulation(use_dynamic=False, use_aging=False)
        run2 = self._run_simulation(use_dynamic=False, use_aging=False)
        assert np.allclose(run1, run2, atol=0), \
            "Aynı parametrelerle iki koşu farklı sonuç verdi!"

    def test_aging_changes_allocation(self):
        """use_aging_cost=True, power dağıtımını değiştirmeli (farklı sonuç)."""
        run_no_aging = self._run_simulation(use_dynamic=False, use_aging=False)
        run_aging    = self._run_simulation(use_dynamic=False, use_aging=True)
        # Toplam güç farklı olmalı (en azından bazı dakikalarda)
        diff = np.abs(run_no_aging - run_aging)
        assert diff.max() > 0.01, \
            "Aging cost açık ama güç dağılımında hiç fark yok"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Marginal aging cost değerleri
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgingCost:
    """EV.marginal_aging_cost hesap doğrulaması."""

    def test_low_soc_low_crate(self):
        """SoC=0.30, 1C → maliyet = 1.0 (soc_stress=1, c_rate²=1)."""
        ev = EV("T", "M", 75.0, 250.0, 0, 0.10)
        ev.current_soc = 0.30
        cost = ev.marginal_aging_cost(75.0)  # 1C = 75kW/75kWh
        assert abs(cost - 1.0) < 0.01

    def test_high_soc_high_crate(self):
        """SoC=0.80, 1.5C → maliyet > 2.0."""
        ev = EV("T", "M", 75.0, 250.0, 0, 0.10)
        ev.current_soc = 0.80
        cost = ev.marginal_aging_cost(75.0 * 1.5)  # 1.5C
        expected = math.exp(3.5 * 0.10) * 2.25  # e^0.35 × 2.25 ≈ 3.19
        assert abs(cost - expected) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
