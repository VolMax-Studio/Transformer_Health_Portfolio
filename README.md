# Transformer Health Monitoring Portfolio

Physics-based transformer health assessment combining three IEC standard methods:
thermal modeling (IEC 60076-7), insulation aging (Arrhenius model), and dissolved
gas analysis interpretation (IEC 60599).

**Data: synthetic load profiles and DGA readings based on IEC standard models and published parameter ranges. NOT real field measurements.**  
Target: 40 MVA, 110/10 kV ONAN — representative of EPS (Serbia) substation transformers.

---

## Physical models implemented

### 1. IEC 60076-7 Thermal Model

Two first-order differential equations for top-oil and hot-spot temperature:

```
τ_o · d(Δθ_o)/dt = Δθ_or · ((1 + R·K²)/(1+R))^n − Δθ_o   [top-oil]
τ_w · d(Δθ_h)/dt = Δθ_hr · K^(2m) − Δθ_h                 [winding gradient]

θ_h = θ_a + Δθ_o + H · Δθ_h                               [hot-spot]
```

**40 MVA ONAN nameplate parameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Rated top-oil rise Δθ_or | 55 K | Typical 110 kV class |
| Rated winding rise Δθ_hr | 23 K | |
| Hot-spot factor H | 1.3 | IEC 60076-7, Table A.2 |
| Load/no-load loss ratio R | 8.0 | |
| Oil time constant τ_o | 180 min | |
| Winding time constant τ_w | 10 min | |
| Oil exponent n (ONAN) | 0.9 | IEC 60076-7, Table 4 |
| Winding exponent m (ONAN) | 0.8 | |

### 2. Arrhenius Insulation Aging

```
F_AA = exp(B/Θ_ref − B/(Θ_h + 273.15))
```

B = 15,000 K (Kraft paper activation energy), Θ_ref = 98°C (IEC reference).

| Hot-spot °C | F_AA | Interpretation |
|-------------|------|---------------|
| 80 | 0.13 | Very slow aging (12.5% of reference) |
| 98 | 1.00 | Reference rate |
| 110 | 4.0 | 4× accelerated |
| 120 | ~8 | 8× accelerated |
| 140 | ~32 | 32× accelerated — emergency condition |

Normal insulation life baseline: 150,000 hours at 98°C (IEC 60076-7, Annex A).

### 3. DGA Interpretation (IEC 60599)

**Duval Triangle** — fault classification from CH4, C2H4, C2H2 ratios:  
PD (partial discharge) · D1/D2 (discharge) · T1/T2/T3 (thermal faults) · DT (mixed)

**IEC key gas limits:**

| Gas | Typical [μL/L] | High [μL/L] | Indicates |
|-----|--------------|-------------|-----------|
| H2 | 100 | 300 | PD, corona |
| CH4 | 120 | 400 | Thermal <300°C |
| C2H2 | 3 | 35 | Arcing (any C2H2 = concern) |
| C2H4 | 50 | 200 | Thermal 300–700°C |
| CO | 500 | 1000 | Cellulose degradation |

---

## Annual simulation results

| Metric | Value |
|--------|-------|
| Peak hot-spot temperature | 158.2°C (overload event) |
| Hours above continuous limit (98°C) | 174 h |
| Hours above emergency limit (140°C) | 3 h |
| Annual loss of life (LOL) | 1.20% |
| Mean aging acceleration F_AA | 0.21× |

LOL of 1.20%/year → expected insulation life ≈ 83 years at this loading profile.
The 3 hours above 140°C emergency limit are from the injected 1.4 pu overload event.

---

## DGA Stress-Testing & ML Boundary Analysis (verified by `dga_stress_test.py`)

Using a public DGA transformer dataset containing 589 fault samples (covering partial discharge, arcing, and low/medium/high thermal faults), we compare a traditional physical Duval Triangle classifier against a Machine Learning Classifier (Random Forest).

### Headline Results:
- **Baseline Accuracy:** The Random Forest ML model reaches **74.6%** accuracy, while the physical Duval Triangle baseline reaches only **50.8%**. The lower baseline of the physical model is due to its geometric limitations: it leaves significant portions of the ratio space unclassified (`undefined` or `normal` zone outputs for actual fault readings).
- **Noise Tolerance Limits:** Under increasing sensor noise $\sigma$ (multiplicative Gaussian noise applied to gas inputs), both classifiers degrade gracefully, but the ML model maintains a significant advantage across all noise ranges:

| Sensor Noise $\sigma$ | ML Classifier (RF) | Duval Triangle (Physical) | Accuracy Gain |
|----------------------:|-------------------:|--------------------------:|--------------:|
| 0.00 (No noise)       | **0.746**          | 0.508                     | +0.238        |
| 0.05                  | 0.754              | 0.500                     | +0.254        |
| 0.10                  | 0.737              | 0.475                     | +0.262        |
| 0.20                  | 0.754              | 0.432                     | +0.322        |
| 0.30                  | 0.678              | 0.466                     | +0.212        |
| 0.50 (Severe noise)   | 0.653              | 0.424                     | +0.229        |

The complete boundaries and predictions are plotted in `results/dga_stress_test.png`, showing the physical zones alongside sample distribution and the noise robustness curve.

---

## Quick start

```bash
pip install -r requirements.txt
python3 run_pipeline.py     # generates results/transformer_health.png
python3 dga_stress_test.py  # runs noise stress test, generates results/dga_stress_test.png
pytest tests/ -v            # 27 tests, all pass
```

---

## BPM FiberNetworks / EPS relevance

EPS operates ~220 power transformers at 110/35 kV substations across Serbia.
BPM FiberNetworks provides monitoring infrastructure at these locations.

This analysis maps directly to:
- **Online temperature monitoring** from RTD sensors on transformer tanks
- **DGA sampling** (currently quarterly at most substations — could be continuous with gas-in-oil sensors)
- **Load cycle logging** from SCADA at the substation bus

The NILM principle from [NILM_Disaggregation_Portfolio](https://github.com/VolMax-Studio/NILM_Disaggregation_Portfolio) applies here: derive transformer thermal state from non-intrusive measurements at the bus, without direct sensors on the winding.

## License

MIT
