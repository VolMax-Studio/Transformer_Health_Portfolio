"""
dga_analyzer.py — Dissolved Gas Analysis (DGA) Interpreter
============================================================
Implements three standard DGA fault interpretation methods:
    1. IEC 60599:2022 — Key gas concentration limits and ratios
    2. Duval Triangle method (Michel Duval, 1989)
    3. Rogers ratio method (revised, 4-ratio version)

Physical basis of DGA
---------------------
Thermal and electrical stresses in transformer oil decompose hydrocarbons
and cellulose, generating dissolved gases. The gas composition is a
fingerprint of the fault type and severity:

    H2  (hydrogen):   electrical discharges, corona, overheating of oil
    CH4 (methane):    low-temperature thermal fault in oil (<300°C)
    C2H2 (acetylene): high-energy arcing, electrical discharge (>700°C)
    C2H4 (ethylene):  medium-high temperature thermal fault (300-700°C)
    C2H6 (ethane):    low-temperature thermal fault in oil (<150°C)
    CO  (carbon mon.): thermal fault in paper/cellulose
    CO2 (carbon diox.): thermal fault in paper/cellulose (also normal aging)

Units: all gas concentrations in μL/L (ppm by volume in oil).

Duval Triangle (IEC 60599, Annex A)
-------------------------------------
Uses normalized percentages of CH4, C2H4, C2H2:
    %CH4  = CH4 / (CH4 + C2H4 + C2H2) × 100
    %C2H4 = C2H4 / (CH4 + C2H4 + C2H2) × 100
    %C2H2 = C2H2 / (CH4 + C2H4 + C2H2) × 100

Fault zones (Duval 1989 / IEC 60599:2022):
    PD  — Partial discharge
    D1  — Low energy discharge
    D2  — High energy discharge (arcing)
    T1  — Thermal fault, T < 300°C
    T2  — Thermal fault, 300°C < T < 700°C
    T3  — Thermal fault, T > 700°C
    DT  — Mix of thermal and discharge faults

IEC 60599 typical concentration limits [μL/L]:
    H2:   100 (typical), 300 (high)
    CH4:  120 (typical), 400 (high)
    C2H2: 3 (typical for ONAN — any C2H2 is significant)
    C2H4: 50 (typical), 200 (high)
    C2H6: 65 (typical), 200 (high)
    CO:   500 (typical), 1000 (high — also from cellulose aging)
    CO2:  < 3000 (typical — ratio CO/CO2 > 0.1 indicates cellulose fault)

References
----------
IEC 60599:2022. Mineral oil-filled electrical equipment in service —
Guide to the interpretation of dissolved and free gases analysis.

Duval, M. (1989). Dissolved gas analysis: It can save your transformer.
IEEE Electrical Insulation Magazine, 5(6), 22–27.

Rogers, R.R. (1978). IEEE and IEC codes to interpret incipient faults
in transformers using gas in oil analysis. IEEE Trans. Electr. Insul.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# IEC 60599:2022 typical/high concentration limits [μL/L]
IEC_LIMITS = {
    'H2':   {'typical': 100,  'high': 300},
    'CH4':  {'typical': 120,  'high': 400},
    'C2H2': {'typical': 3,    'high': 35},   # very sensitive: any C2H2 = concern
    'C2H4': {'typical': 50,   'high': 200},
    'C2H6': {'typical': 65,   'high': 200},
    'CO':   {'typical': 500,  'high': 1000},
    'CO2':  {'typical': 3000, 'high': 10000},
}


@dataclass
class DGAReading:
    """Single DGA sample (one oil sampling event)."""
    H2:   float   # Hydrogen [μL/L]
    CH4:  float   # Methane [μL/L]
    C2H2: float   # Acetylene [μL/L]
    C2H4: float   # Ethylene [μL/L]
    C2H6: float   # Ethane [μL/L]
    CO:   float   # Carbon monoxide [μL/L]
    CO2:  float   # Carbon dioxide [μL/L]
    timestamp: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != 'timestamp'}

    def total_dissolved_combustible_gas(self) -> float:
        """TDCG = H2 + CH4 + C2H2 + C2H4 + C2H6 + CO [μL/L]."""
        return self.H2 + self.CH4 + self.C2H2 + self.C2H4 + self.C2H6 + self.CO


@dataclass
class DGAResult:
    """Interpretation result for one DGA reading."""
    reading: DGAReading
    duval_zone: str                  # Duval triangle fault zone
    rogers_diagnosis: str            # Rogers ratio diagnosis
    gases_above_typical: list        # Gas names exceeding typical limit
    gases_above_high: list           # Gas names exceeding high limit
    tdcg: float                      # Total dissolved combustible gas [μL/L]
    severity: str                    # 'normal', 'caution', 'warning', 'critical'
    co_co2_ratio: float              # Cellulose degradation indicator
    c2h2_c2h4_ratio: float          # Arcing indicator
    recommended_action: str


def duval_triangle_zone(ch4: float, c2h4: float, c2h2: float) -> str:
    """
    Identify fault zone in Duval Triangle (IEC 60599, Annex A).

    Parameters
    ----------
    ch4, c2h4, c2h2 : float  Gas concentrations [μL/L].

    Returns
    -------
    str : Zone identifier — 'PD', 'D1', 'D2', 'T1', 'T2', 'T3', 'DT', or 'normal'
    """
    total = ch4 + c2h4 + c2h2
    if total < 1e-6:
        return 'normal'

    # Normalized percentages for Duval triangle
    pct_ch4  = ch4  / total * 100
    pct_c2h4 = c2h4 / total * 100
    pct_c2h2 = c2h2 / total * 100

    # Duval triangle boundary conditions (IEC 60599:2022, Annex A)
    # Zone boundaries derived from Duval 1989 and updated in IEC 60599:2022

    # PD (Partial Discharge): %C2H2 < 0.2 AND %C2H4 < 5
    if pct_c2h2 < 0.2 and pct_c2h4 < 5:
        return 'PD'

    # D1 (Low energy discharge): %C2H2 ≥ 29 OR (%C2H2 ≥ 14 AND %C2H4 < 23)
    if pct_c2h2 >= 29 or (pct_c2h2 >= 14 and pct_c2h4 < 23):
        return 'D1'

    # D2 (High energy discharge / arcing): %C2H2 ≥ 13 AND %C2H4 ≥ 23
    if pct_c2h2 >= 13 and pct_c2h4 >= 23:
        return 'D2'

    # DT (Mix): %C2H2 ≥ 4 AND %C2H4 ≥ 10 AND %C2H4 ≤ 40
    if pct_c2h2 >= 4 and 10 <= pct_c2h4 <= 40:
        return 'DT'

    # T3 (High temperature thermal, >700°C): %C2H2 < 4 AND %C2H4 > 50
    if pct_c2h2 < 4 and pct_c2h4 > 50:
        return 'T3'

    # T2 (Medium temperature thermal, 300-700°C): %C2H2 < 4 AND 20 < %C2H4 ≤ 50
    if pct_c2h2 < 4 and 20 < pct_c2h4 <= 50:
        return 'T2'

    # T1 (Low temperature thermal, <300°C): %C2H2 < 4 AND %C2H4 ≤ 20
    if pct_c2h2 < 4 and pct_c2h4 <= 20:
        return 'T1'

    return 'undefined'


DUVAL_ZONE_DESCRIPTIONS = {
    'PD':        'Partial Discharge (corona)',
    'D1':        'Low Energy Discharge',
    'D2':        'High Energy Discharge (arcing)',
    'T1':        'Thermal Fault < 300°C',
    'T2':        'Thermal Fault 300–700°C',
    'T3':        'Thermal Fault > 700°C',
    'DT':        'Mixed Thermal + Discharge',
    'normal':    'Normal (no fault gases)',
    'undefined': 'Unclassified',
}


def rogers_ratios_diagnosis(h2: float, ch4: float, c2h2: float,
                             c2h4: float, c2h6: float) -> str:
    """
    Rogers ratio method (IEC 60599, revised 4-ratio version).

    Ratios:
        R1 = CH4/H2
        R2 = C2H2/C2H4
        R3 = C2H4/C2H6

    Case codes are applied sequentially.
    """
    eps = 0.01   # avoid division by zero

    R1 = ch4  / max(h2,   eps)
    R2 = c2h2 / max(c2h4, eps)
    R3 = c2h4 / max(c2h6, eps)

    # Rogers case table (simplified from IEEE C57.104)
    if R2 < 0.1 and R1 >= 0.1 and R1 < 1.0 and R3 < 1.0:
        return 'Normal aging'
    if R2 < 0.1 and R1 < 0.1 and R3 < 1.0:
        return 'Partial discharge (corona)'
    if R2 < 0.1 and R1 < 0.1 and R3 >= 1.0 and R3 < 3.0:
        return 'Partial discharge with tracking'
    if 0.1 <= R2 < 3.0 and R1 < 0.1 and R3 >= 3.0:
        return 'High energy discharge'
    if 0.1 <= R2 < 3.0 and 0.1 <= R1 < 1.0 and R3 >= 3.0:
        return 'High energy discharge with thermal'
    if R2 >= 3.0:
        return 'Arcing (high C2H2)'
    if R2 < 0.1 and R1 >= 1.0 and R3 >= 3.0:
        return 'Thermal fault > 700°C'
    if R2 < 0.1 and R1 >= 1.0 and 1.0 <= R3 < 3.0:
        return 'Thermal fault 300–700°C'
    if R2 < 0.1 and R1 >= 0.1 and R1 < 1.0 and R3 >= 1.0:
        return 'Thermal fault < 300°C'

    return 'Indeterminate'


def assess_severity(reading: DGAReading, gases_above_typical: list,
                    gases_above_high: list, duval_zone: str) -> Tuple[str, str]:
    """
    Classify overall severity and recommend action.

    Returns (severity, recommended_action).
    """
    tdcg = reading.total_dissolved_combustible_gas()

    if len(gases_above_high) >= 2 or reading.C2H2 > 35 or tdcg > 4630:
        severity = 'critical'
        action = ('Immediate shutdown recommended. Contact transformer engineer. '
                  'Perform additional confirmatory tests (DGA repeat, furan analysis).')
    elif len(gases_above_high) == 1 or reading.C2H2 > 3 or tdcg > 2100:
        severity = 'warning'
        action = ('Increase DGA sampling frequency to monthly. '
                  'Review recent loading history and thermal records. '
                  'Consider derating if loading > 0.9 pu.')
    elif len(gases_above_typical) >= 2 or tdcg > 1000:
        severity = 'caution'
        action = ('Increase DGA sampling to quarterly. '
                  'Monitor for upward trends in key gases. '
                  'Review cooling system operation.')
    else:
        severity = 'normal'
        action = 'Continue annual DGA sampling. No action required.'

    if duval_zone in ('D2', 'DT') and severity != 'critical':
        severity = 'warning'
        action = ('Discharge detected. ' + action)

    return severity, action


def interpret_dga(reading: DGAReading) -> DGAResult:
    """
    Full DGA interpretation: Duval + Rogers + severity assessment.

    Parameters
    ----------
    reading : DGAReading  Single oil sample.

    Returns
    -------
    DGAResult
    """
    # Duval triangle
    duval_zone = duval_triangle_zone(reading.CH4, reading.C2H4, reading.C2H2)

    # Rogers ratios
    rogers = rogers_ratios_diagnosis(
        reading.H2, reading.CH4, reading.C2H2, reading.C2H4, reading.C2H6
    )

    # IEC limit checks
    gases_above_typical = []
    gases_above_high = []
    for gas, limits in IEC_LIMITS.items():
        val = getattr(reading, gas)
        if val > limits['high']:
            gases_above_high.append(gas)
        elif val > limits['typical']:
            gases_above_typical.append(gas)

    # Ratios
    co_co2 = reading.CO / max(reading.CO2, 1.0)
    c2h2_c2h4 = reading.C2H2 / max(reading.C2H4, 0.01)

    # Severity
    severity, action = assess_severity(reading, gases_above_typical,
                                        gases_above_high, duval_zone)

    return DGAResult(
        reading=reading,
        duval_zone=duval_zone,
        rogers_diagnosis=rogers,
        gases_above_typical=gases_above_typical,
        gases_above_high=gases_above_high,
        tdcg=reading.total_dissolved_combustible_gas(),
        severity=severity,
        co_co2_ratio=co_co2,
        c2h2_c2h4_ratio=c2h2_c2h4,
        recommended_action=action,
    )


def generate_dga_from_thermal_history(
    theta_h_avg: float,
    years_in_service: float,
    fault_type: str = 'normal',
    seed: int = 42,
) -> DGAReading:
    """
    Generate a physically plausible DGA reading based on thermal history.

    This is a simplified parametric model — NOT a true physicochemical
    simulation. Gas generation rates are calibrated to published ranges
    (IEC 60599, Annex B; Cigre TB 771).

    Parameters
    ----------
    theta_h_avg : float  Average hot-spot temperature [°C].
    years_in_service : float  Transformer age [years].
    fault_type : str  'normal', 'thermal_low', 'thermal_high', 'arcing', 'pd'
    seed : int

    Returns
    -------
    DGAReading  Plausible (not guaranteed accurate) DGA concentrations.
    """
    rng = np.random.default_rng(seed)

    # Base gas generation (normal aging) — proportional to temperature and age
    f_aa = float(np.exp(15000 / (98 + 273.15) - 15000 / (theta_h_avg + 273.15)))
    age_factor = min(years_in_service / 30.0, 2.0)  # normalize to 30-year life

    # Base gases from normal aging [μL/L]
    H2   = rng.uniform(5, 30) * f_aa * age_factor
    CH4  = rng.uniform(10, 60) * f_aa * age_factor
    C2H2 = rng.uniform(0, 0.5) * age_factor  # minimal in normal operation
    C2H4 = rng.uniform(5, 30) * f_aa * age_factor
    C2H6 = rng.uniform(10, 40) * f_aa * age_factor
    CO   = rng.uniform(100, 400) * age_factor  # cellulose aging
    CO2  = rng.uniform(2000, 6000) * age_factor  # cellulose aging (mostly)

    # Fault-type injections
    if fault_type == 'thermal_low':
        # T < 300°C: elevated CH4, C2H6
        CH4  += rng.uniform(150, 400)
        C2H6 += rng.uniform(80, 200)
        CO   += rng.uniform(200, 600)  # some cellulose involvement

    elif fault_type == 'thermal_high':
        # T > 700°C: dominant C2H4, some C2H2
        C2H4 += rng.uniform(200, 800)
        C2H2 += rng.uniform(2, 10)
        CH4  += rng.uniform(100, 300)

    elif fault_type == 'arcing':
        # High-energy arcing: dominant C2H2, H2
        C2H2 += rng.uniform(50, 300)
        H2   += rng.uniform(200, 1000)
        C2H4 += rng.uniform(50, 200)

    elif fault_type == 'pd':
        # Partial discharge: dominant H2
        H2   += rng.uniform(200, 600)
        CH4  += rng.uniform(50, 150)

    # Add measurement noise (10-15% for DGA chromatograph)
    noise_frac = 0.12
    def noisy(val):
        return max(0.0, val * (1 + rng.normal(0, noise_frac)))

    return DGAReading(
        H2=noisy(H2), CH4=noisy(CH4), C2H2=noisy(C2H2),
        C2H4=noisy(C2H4), C2H6=noisy(C2H6),
        CO=noisy(CO), CO2=noisy(CO2),
    )
