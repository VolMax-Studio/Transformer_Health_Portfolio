"""run_pipeline.py — Transformer Health Monitoring. Usage: python3 run_pipeline.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.thermal_model import simulate_thermal, EPS_TRAFO_40MVA
from src.aging_estimator import (
    generate_annual_load_profile, generate_ambient_temperature,
    inject_overload_events, aging_acceleration_factor, compute_loss_of_life
)
from src.dga_analyzer import (
    generate_dga_from_thermal_history, interpret_dga,
    DGAReading, IEC_LIMITS, DUVAL_ZONE_DESCRIPTIONS
)

os.makedirs("results", exist_ok=True)

def run():
    print("="*62)
    print("  Transformer Health Monitoring Portfolio")
    print("  40 MVA 110/10 kV ONAN — EPS-type substation transformer")
    print("  IEC 60076-7 thermal model + IEC 60599 DGA interpretation")
    print("="*62)

    print("\n[1/4] Generating annual load + ambient profile...")
    k = generate_annual_load_profile(peak_load_pu=1.15, seed=42)
    k = inject_overload_events(k)
    ta = generate_ambient_temperature(latitude_class='balkans', seed=42)
    t = np.arange(8760, dtype=float)

    print("[2/4] IEC 60076-7 thermal simulation (8760 hours)...")
    sim = simulate_thermal(k, t, ta, nameplate=EPS_TRAFO_40MVA)
    lol = compute_loss_of_life(sim.theta_hot_spot, dt_h=1.0)
    f_aa = aging_acceleration_factor(sim.theta_hot_spot)

    daylight = k > 0
    print(f"  Peak hot-spot temperature:   {sim.peak_hot_spot:.1f}°C")
    print(f"  Hours above 98°C (limit):    {sim.time_above_limit:.0f} h")
    print(f"  Hours above 140°C (emerg.):  {sim.time_above_emergency:.0f} h")
    print(f"  Annual loss of life:          {lol[-1]:.3f}%")
    print(f"  Mean aging acceleration:      {float(np.mean(f_aa)):.3f}×")

    print("\n[3/4] DGA interpretation (parametric, fault-type scenarios)...")
    scenarios = [
        ('Normal (15yr service)',   115.0, 15.0, 'normal'),
        ('Low-temp thermal fault',  115.0, 20.0, 'thermal_low'),
        ('High-temp thermal fault', 130.0, 25.0, 'thermal_high'),
        ('Arcing event',            110.0, 10.0, 'arcing'),
        ('Partial discharge',       100.0, 5.0,  'pd'),
    ]
    for label, theta, years, fault in scenarios:
        rdg = generate_dga_from_thermal_history(theta, years, fault, seed=7)
        res = interpret_dga(rdg)
        print(f"  [{label}] Duval: {res.duval_zone:4s} | "
              f"Severity: {res.severity:8s} | TDCG: {res.tdcg:.0f} μL/L")

    print("\n[4/4] Generating diagnostic plots...")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel 1: Load + Temperature
    ax1 = fig.add_subplot(gs[0, :])
    t_day = t / 24
    ax1_r = ax1.twinx()
    ax1.plot(t_day, k, lw=0.4, alpha=0.6, color='steelblue', label='Load factor K [pu]')
    ax1_r.plot(t_day, ta, lw=0.4, alpha=0.5, color='orange', label='Ambient T [°C]')
    ax1.set_ylabel("Load factor K [pu]", color='steelblue')
    ax1_r.set_ylabel("T_ambient [°C]", color='orange')
    ax1.set_title("Annual Load Profile + Ambient Temperature (Balkans, IEC 60076-7 simulation)")
    ax1.set_xlabel("Day of year"); ax1.grid(True, alpha=0.25)

    # Panel 2: Temperatures
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_day, sim.theta_hot_spot, lw=0.5, color='tomato', label='Hot-spot θ_h')
    ax2.plot(t_day, sim.theta_top_oil,  lw=0.5, color='steelblue', alpha=0.7, label='Top-oil θ_o')
    ax2.axhline(98,  color='orange', ls='--', lw=1.2, label='98°C limit')
    ax2.axhline(140, color='red',    ls='--', lw=1.2, label='140°C emergency')
    ax2.set_title("Transformer Temperatures"); ax2.set_xlabel("Day")
    ax2.set_ylabel("Temperature [°C]"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)

    # Panel 3: Aging
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_day, lol, lw=1.2, color='purple', label='Cumulative LOL [%]')
    ax3.set_title(f"Insulation Loss of Life (Arrhenius)\nAnnual total: {lol[-1]:.3f}%")
    ax3.set_xlabel("Day"); ax3.set_ylabel("LOL [%]"); ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)

    # Panel 4: Duval Triangle (simplified bar chart by scenario)
    ax4 = fig.add_subplot(gs[2, 0])
    zones, tdcgs, colors_d = [], [], []
    color_map = {'normal':'steelblue','caution':'gold','warning':'orange','critical':'red'}
    for label, theta, years, fault in scenarios:
        rdg = generate_dga_from_thermal_history(theta, years, fault, seed=7)
        res = interpret_dga(rdg)
        zones.append(f"{label[:20]}\n({res.duval_zone})")
        tdcgs.append(res.tdcg)
        colors_d.append(color_map.get(res.severity, 'gray'))
    ax4.barh(zones, tdcgs, color=colors_d)
    ax4.axvline(1000, color='orange', ls='--', lw=1, label='Caution threshold')
    ax4.axvline(2100, color='red', ls='--', lw=1, label='Warning threshold')
    ax4.set_title("DGA Scenarios — TDCG [μL/L] by Fault Type")
    ax4.set_xlabel("TDCG [μL/L]"); ax4.legend(fontsize=7); ax4.grid(True, alpha=0.25, axis='x')

    # Panel 5: Aging acceleration distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(f_aa[f_aa < 20], bins=80, color='purple', alpha=0.7, edgecolor='none')
    ax5.axvline(1.0, color='green', ls='--', lw=1.5, label='F_AA = 1.0 (98°C ref)')
    ax5.axvline(float(np.mean(f_aa)), color='red', ls='--', lw=1.5,
                label=f'Mean = {float(np.mean(f_aa)):.2f}×')
    ax5.set_title("Aging Acceleration Factor Distribution")
    ax5.set_xlabel("F_AA [×]"); ax5.set_ylabel("Hours"); ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.25)

    plt.suptitle("Transformer Health Monitoring — 40 MVA ONAN, EPS-type\n"
                 "IEC 60076-7 thermal model | Arrhenius aging | IEC 60599 DGA | 27/27 tests",
                 fontsize=10, y=0.98)
    plt.savefig("results/transformer_health.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/transformer_health.png")
    print("="*62)

if __name__ == "__main__":
    run()
