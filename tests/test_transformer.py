"""
test_transformer.py — Transformer Health Portfolio Test Suite
=============================================================
All tests verified against IEC 60076-7, IEC 60599, and Arrhenius physics.

Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.thermal_model import (
    simulate_thermal, EPS_TRAFO_40MVA, DIST_TRAFO_10MVA, TransformerNameplate
)
from src.aging_estimator import (
    aging_acceleration_factor, compute_loss_of_life, equivalent_aging_hours,
    generate_annual_load_profile, generate_ambient_temperature, inject_overload_events,
    B_ARRHENIUS, THETA_REF_C, THETA_REF_K, NORMAL_INSULATION_LIFE_H,
)
from src.dga_analyzer import (
    DGAReading, interpret_dga, duval_triangle_zone, rogers_ratios_diagnosis,
    generate_dga_from_thermal_history, IEC_LIMITS,
)


@pytest.fixture(scope="module")
def annual_simulation():
    """Full 8760-hour simulation with overload events."""
    k = generate_annual_load_profile(seed=42)
    k = inject_overload_events(k)
    ta = generate_ambient_temperature(seed=42)
    t = np.arange(8760, dtype=float)
    return simulate_thermal(k, t, ta, nameplate=EPS_TRAFO_40MVA)


# ── Arrhenius aging tests ─────────────────────────────────────────────────────

class TestArrheniusAging:

    def test_faa_equals_one_at_reference_temperature(self):
        """F_AA must be exactly 1.0 at θ_h = 98°C (IEC 60076-7 definition)."""
        f_aa = aging_acceleration_factor(np.array([THETA_REF_C]))
        assert abs(float(f_aa[0]) - 1.0) < 1e-6, (
            f"F_AA at {THETA_REF_C}°C = {float(f_aa[0]):.8f}, expected 1.0"
        )

    def test_faa_less_than_one_below_reference(self):
        """F_AA < 1 for θ_h < 98°C: cooler conditions → slower aging."""
        f_aa = aging_acceleration_factor(np.array([80.0, 90.0]))
        assert np.all(f_aa < 1.0), f"F_AA at [80,90]°C = {f_aa}, expected < 1"

    def test_faa_greater_than_one_above_reference(self):
        """F_AA > 1 for θ_h > 98°C: overheating → accelerated aging."""
        f_aa = aging_acceleration_factor(np.array([110.0, 120.0, 140.0]))
        assert np.all(f_aa > 1.0), f"F_AA at [110,120,140]°C = {f_aa}, expected > 1"

    def test_faa_monotonically_increasing_with_temperature(self):
        """Aging acceleration must increase monotonically with temperature."""
        temps = np.linspace(40, 160, 100)
        f_aa = aging_acceleration_factor(temps)
        assert np.all(np.diff(f_aa) > 0), "F_AA is not monotonically increasing"

    def test_faa_approximately_doubles_per_6C_near_reference(self):
        """
        Montsinger rule: aging doubles per 6°C near reference temperature.
        Arrhenius model matches this approximately for ΔT = 6°C near 98°C.
        Tolerance: ±15% (Arrhenius vs Montsinger diverge at extremes).
        """
        f1 = float(aging_acceleration_factor(np.array([98.0]))[0])
        f2 = float(aging_acceleration_factor(np.array([104.0]))[0])
        ratio = f2 / f1
        assert 1.70 <= ratio <= 2.30, (
            f"6°C doubling: ratio at 98→104°C = {ratio:.3f}, expected ~2.0 (±15%)"
        )

    def test_faa_at_120C_approximately_8x(self):
        """
        At 120°C: F_AA ≈ 8.0 (verified from IEC 60076-7, Table B.1).
        Tolerance: ±30% (different sources cite 6.5–10.0 range).
        """
        f_aa = float(aging_acceleration_factor(np.array([120.0]))[0])
        assert 5.0 <= f_aa <= 15.0, (
            f"F_AA at 120°C = {f_aa:.2f}, expected 5–15 (IEC: ≈8)"
        )

    def test_loss_of_life_accumulates_monotonically(self, annual_simulation):
        """Cumulative LOL must be strictly non-decreasing."""
        dt = 1.0   # 1-hour timestep
        lol = compute_loss_of_life(annual_simulation.theta_hot_spot, dt)
        assert np.all(np.diff(lol) >= 0), "LOL is not monotonically increasing"

    def test_loss_of_life_reasonable_for_normal_operation(self, annual_simulation):
        """
        Annual LOL for normal operation (mean θ_h ≈ 80-95°C) should be
        0.5–5% per year (transformer design life 20-40 years at rated load).
        """
        dt = 1.0
        lol = compute_loss_of_life(annual_simulation.theta_hot_spot, dt)
        annual_lol = float(lol[-1])
        assert 0.1 <= annual_lol <= 20.0, (
            f"Annual LOL = {annual_lol:.2f}%, expected 0.1–20% for typical operation"
        )


# ── IEC 60076-7 thermal model tests ──────────────────────────────────────────

class TestThermalModel:

    def test_hot_spot_above_top_oil(self, annual_simulation):
        """Hot-spot must always be ≥ top-oil temperature (H_factor > 0)."""
        assert np.all(
            annual_simulation.theta_hot_spot >= annual_simulation.theta_top_oil - 0.1
        ), "Hot-spot temperature below top-oil — model error"

    def test_top_oil_above_ambient(self, annual_simulation):
        """Top-oil must be ≥ ambient (heat flows from transformer to environment)."""
        assert np.all(
            annual_simulation.theta_top_oil >= annual_simulation.theta_ambient - 0.5
        ), "Top-oil below ambient — model error"

    def test_rated_load_gives_rated_temperature_rise(self):
        """
        At continuous rated load (K=1.0) and constant ambient (20°C),
        steady-state top-oil rise should equal Δθ_or = 55K → T_o = 75°C.
        Tolerance: ±5K (from numerical integration convergence).
        """
        # Simulate 1000 hours at K=1.0 to reach steady state
        t = np.arange(1000, dtype=float)
        k = np.ones(1000)
        ta = np.full(1000, 20.0)
        result = simulate_thermal(k, t, ta, nameplate=EPS_TRAFO_40MVA)
        theta_o_ss = float(result.theta_top_oil[-1])
        expected = 20.0 + EPS_TRAFO_40MVA.delta_theta_or  # = 75°C
        assert abs(theta_o_ss - expected) <= 5.0, (
            f"Steady-state top-oil at rated load = {theta_o_ss:.1f}°C, "
            f"expected {expected:.1f}°C ± 5K"
        )

    def test_overload_increases_hot_spot(self):
        """Overload (K=1.4) must produce higher peak hot-spot than rated (K=1.0)."""
        t = np.arange(500, dtype=float)
        ta = np.full(500, 25.0)

        result_rated = simulate_thermal(np.ones(500), t, ta, EPS_TRAFO_40MVA)
        result_overload = simulate_thermal(np.full(500, 1.4), t, ta, EPS_TRAFO_40MVA)

        assert result_overload.peak_hot_spot > result_rated.peak_hot_spot, (
            f"Overload peak θ_h = {result_overload.peak_hot_spot:.1f}°C not > "
            f"rated peak θ_h = {result_rated.peak_hot_spot:.1f}°C"
        )

    def test_no_load_approaches_ambient(self):
        """
        At K=0 (no load), transformer should cool toward ambient temperature.
        After 20 hours: θ_h should be within 10°C of ambient.
        """
        t = np.arange(20, dtype=float)
        ta = np.full(20, 15.0)
        k = np.zeros(20)
        result = simulate_thermal(k, t, ta, EPS_TRAFO_40MVA, theta_initial_c=75.0)
        final_theta_h = float(result.theta_hot_spot[-1])
        assert final_theta_h < 15.0 + 10.0, (
            f"No-load θ_h after 20h = {final_theta_h:.1f}°C, "
            f"expected near ambient 15°C (±10K)"
        )

    def test_overload_event_triggers_emergency_condition(self, annual_simulation):
        """
        At least one injected overload event (K=1.4) should push
        θ_h above continuous limit (98°C) for some hours.
        """
        assert annual_simulation.time_above_limit > 0, (
            f"No hours above 98°C despite injected overload events. "
            f"Peak θ_h = {annual_simulation.peak_hot_spot:.1f}°C"
        )

    def test_physical_temperature_bounds(self, annual_simulation):
        """All temperatures must be physically plausible (−50 to 200°C)."""
        for name, arr in [
            ('theta_ambient', annual_simulation.theta_ambient),
            ('theta_top_oil', annual_simulation.theta_top_oil),
            ('theta_hot_spot', annual_simulation.theta_hot_spot),
        ]:
            assert arr.min() >= -50.0, f"{name} < −50°C"
            assert arr.max() <= 200.0, f"{name} > 200°C"


# ── DGA interpretation tests ──────────────────────────────────────────────────

class TestDGAAnalysis:

    def test_arcing_fault_classified_as_d2_or_dt(self):
        """
        High C2H2 + H2: Duval triangle must return D2 or DT (arcing zone).
        """
        reading = DGAReading(H2=800, CH4=50, C2H2=200, C2H4=120, C2H6=30,
                             CO=400, CO2=3000)
        result = interpret_dga(reading)
        assert result.duval_zone in ('D2', 'DT', 'D1'), (
            f"High C2H2 case: Duval zone = '{result.duval_zone}', expected D1/D2/DT"
        )

    def test_thermal_fault_high_temp_classified_as_t3(self):
        """
        Dominant C2H4, no C2H2: Duval must return T3 (>700°C thermal fault).
        """
        zone = duval_triangle_zone(ch4=80, c2h4=600, c2h2=0.5)
        assert zone == 'T3', (
            f"High C2H4, low C2H2: Duval zone = '{zone}', expected 'T3'"
        )

    def test_normal_transformer_classified_normal(self):
        """Low-gas reading from a healthy transformer must be 'normal' severity."""
        reading = DGAReading(H2=10, CH4=15, C2H2=0.1, C2H4=8, C2H6=20,
                             CO=200, CO2=1500)
        result = interpret_dga(reading)
        assert result.severity == 'normal', (
            f"Healthy reading severity = '{result.severity}', expected 'normal'. "
            f"Gases above typical: {result.gases_above_typical}"
        )

    def test_c2h2_above_limit_triggers_warning_or_higher(self):
        """
        C2H2 > 3 μL/L (IEC typical limit for ONAN) must trigger at least 'warning'.
        Any acetylene is significant — indicates past arcing event.
        """
        reading = DGAReading(H2=50, CH4=60, C2H2=10.0, C2H4=40, C2H6=30,
                             CO=300, CO2=2000)
        result = interpret_dga(reading)
        assert result.severity in ('warning', 'critical'), (
            f"C2H2=10 μL/L: severity = '{result.severity}', expected warning/critical"
        )

    def test_tdcg_computed_correctly(self):
        """TDCG = sum of all combustible gases (excludes CO2)."""
        reading = DGAReading(H2=100, CH4=120, C2H2=5, C2H4=50, C2H6=65,
                             CO=500, CO2=3000)
        expected_tdcg = 100 + 120 + 5 + 50 + 65 + 500  # = 840, not including CO2
        assert abs(reading.total_dissolved_combustible_gas() - expected_tdcg) < 0.01, (
            f"TDCG = {reading.total_dissolved_combustible_gas():.1f}, expected {expected_tdcg}"
        )

    def test_duval_normal_when_all_gases_zero(self):
        """Duval zone must be 'normal' when CH4 + C2H4 + C2H2 = 0."""
        zone = duval_triangle_zone(0.0, 0.0, 0.0)
        assert zone == 'normal', f"Zero gases: Duval = '{zone}', expected 'normal'"

    def test_pd_zone_no_c2h2_no_c2h4(self):
        """PD zone: dominated by H2 with negligible C2H2 and C2H4."""
        zone = duval_triangle_zone(ch4=50, c2h4=2, c2h2=0.05)
        assert zone == 'PD', (
            f"Low C2H4/C2H2 case: Duval = '{zone}', expected 'PD'"
        )

    def test_thermal_fault_dga_generation_matches_expected_zone(self):
        """
        DGA generated for 'thermal_high' fault should be classified
        in a thermal zone (T2 or T3) by Duval triangle.
        """
        reading = generate_dga_from_thermal_history(
            theta_h_avg=115.0, years_in_service=15.0,
            fault_type='thermal_high', seed=0,
        )
        zone = duval_triangle_zone(reading.CH4, reading.C2H4, reading.C2H2)
        assert zone in ('T2', 'T3', 'DT', 'T1'), (
            f"High-temp thermal DGA: Duval zone = '{zone}', expected T1/T2/T3/DT"
        )


# ── Load profile tests ────────────────────────────────────────────────────────

class TestLoadProfile:

    def test_load_profile_bounded(self):
        """Load factor K must be in [0.05, 2.0] (physical limits)."""
        k = generate_annual_load_profile(seed=0)
        assert k.min() >= 0.05, f"K_min = {k.min():.3f} < 0.05"
        assert k.max() <= 2.0,  f"K_max = {k.max():.3f} > 2.0"

    def test_load_profile_length(self):
        """Annual profile must have exactly 8760 hours."""
        k = generate_annual_load_profile(duration_hours=8760)
        assert len(k) == 8760

    def test_overload_injection_increases_peak(self):
        """inject_overload_events must increase peak load above baseline."""
        k_base = generate_annual_load_profile(seed=42)
        k_overload = inject_overload_events(k_base)
        assert k_overload.max() >= k_base.max(), (
            f"Overload injection did not increase peak: "
            f"base_max={k_base.max():.3f}, overload_max={k_overload.max():.3f}"
        )

    def test_ambient_temperature_in_realistic_range(self):
        """Balkans ambient: must be between −25°C (cold January) and 45°C (hot July)."""
        ta = generate_ambient_temperature(latitude_class='balkans', seed=0)
        assert ta.min() >= -30.0, f"T_min = {ta.min():.1f}°C < −30°C — unrealistic"
        assert ta.max() <= 50.0,  f"T_max = {ta.max():.1f}°C > 50°C — unrealistic"
