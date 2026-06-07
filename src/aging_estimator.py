"""
aging_estimator.py — Insulation Aging Model + Load Profile Generation
=======================================================================
Implements the Arrhenius/Montsinger insulation aging model from IEC 60076-7
and generates realistic daily/seasonal transformer load profiles.

Insulation aging model (IEC 60076-7, Annex A)
----------------------------------------------
Transformer insulation life depends on hot-spot temperature.
The Arrhenius equation gives the per-unit aging acceleration factor:

    F_AA = exp(B/Θ_ref − B/(Θ_h + 273.15))

where:
    B      = 15000 K  (empirical activation energy / Boltzmann constant
                       for Kraft paper in mineral oil)
    Θ_ref  = 98°C    (IEC reference temperature at which F_AA = 1.0)
    Θ_h    = hot-spot temperature [°C]

The Montsinger rule approximates this as doubling aging per 6°C rise:
    F_AA ≈ 2^((Θ_h − 98)/6)

IEC 60076-7 uses the Arrhenius form.

Aging acceleration at key temperatures:
    Θ_h = 80°C  → F_AA = 0.125  (aging at 12.5% of reference rate)
    Θ_h = 98°C  → F_AA = 1.000  (reference)
    Θ_h = 110°C → F_AA = 4.0
    Θ_h = 120°C → F_AA = 8.0
    Θ_h = 140°C → F_AA = 32.0

Normal insulation life (Kraft paper): 20–40 years at continuous 98°C.
Typical EPS transformer design life: 30 years.

Loss of life (LOL) over a time period:
    LOL [%] = (1 / L_normal) × ∫F_AA dt × 100

where L_normal = 8.76 × 10^4 hours (10-year equivalent life at 98°C
as a conservative basis — some references use 150,000–200,000 hours).
"""

import numpy as np
import pandas as pd
from typing import Optional


# Arrhenius aging parameters (IEC 60076-7, Annex A)
B_ARRHENIUS     = 15000.0   # K — activation energy / Boltzmann for Kraft paper
THETA_REF_C     = 98.0      # °C — reference temperature (F_AA = 1.0)
THETA_REF_K     = THETA_REF_C + 273.15

# Normal insulation life baseline (hours at continuous 98°C)
# IEC 60076-7 uses 150,000 h (≈17 years); IEEE C57.91 uses 65,000 h
# Using 150,000 h (IEC) as conservative basis
NORMAL_INSULATION_LIFE_H = 150_000.0   # hours


def aging_acceleration_factor(theta_h_c: np.ndarray) -> np.ndarray:
    """
    Compute per-unit aging acceleration factor F_AA (Arrhenius model).

    Parameters
    ----------
    theta_h_c : np.ndarray
        Hot-spot temperature [°C]. Can be scalar or array.

    Returns
    -------
    np.ndarray  F_AA values (1.0 = aging at reference rate, 2.0 = 2× faster).

    Notes
    -----
    Physical limits: F_AA → 0 as θ_h → 0°C (very slow aging).
    F_AA → ∞ as θ_h → ∞ but capped at transformer failure limit (~180°C).
    """
    theta_h_k = np.asarray(theta_h_c, dtype=np.float64) + 273.15
    theta_h_k = np.maximum(theta_h_k, 200.0)  # avoid division by zero below −73°C
    f_aa = np.exp(B_ARRHENIUS / THETA_REF_K - B_ARRHENIUS / theta_h_k)
    return f_aa


def compute_loss_of_life(
    theta_h_c: np.ndarray,
    dt_h: float,
    normal_life_h: float = NORMAL_INSULATION_LIFE_H,
) -> np.ndarray:
    """
    Compute cumulative Loss of Life (LOL) [%] from hot-spot temperature history.

    LOL(t) = (dt / L_normal) × ∑F_AA(θ_h) × 100

    Parameters
    ----------
    theta_h_c : np.ndarray  Hot-spot temperature array [°C].
    dt_h : float            Timestep [hours].
    normal_life_h : float   Design life at reference temperature [hours].

    Returns
    -------
    np.ndarray  Cumulative LOL [%], same length as theta_h_c.
    """
    f_aa = aging_acceleration_factor(theta_h_c)
    aging_per_step = f_aa * dt_h / normal_life_h * 100.0
    return np.cumsum(aging_per_step)


def equivalent_aging_hours(
    theta_h_c: np.ndarray,
    dt_h: float,
) -> float:
    """
    Equivalent aging hours at reference temperature (98°C).
    Used to compare different loading scenarios.
    """
    f_aa = aging_acceleration_factor(theta_h_c)
    return float(np.sum(f_aa) * dt_h)


# ── Load profile generation ──────────────────────────────────────────────────

def generate_annual_load_profile(
    duration_hours: int = 8760,
    peak_load_pu: float = 1.15,      # Annual peak overload (15% above rated)
    base_load_pu: float = 0.40,      # Overnight minimum load
    seasonal_amplitude: float = 0.15, # Winter higher than summer
    noise_std: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a realistic annual transformer load profile [pu].

    Pattern: diurnal cycle (morning/evening peaks) × seasonal variation
    (higher winter load for heating-dominated grid).

    Based on typical EPS distribution substation load shapes (ENTSO-E
    TYNDP load profiles for Balkans region, normalized).

    Parameters
    ----------
    peak_load_pu : float  Maximum load in pu. Default: 1.15 (15% overload
                          at annual peak, e.g. cold January morning).
    base_load_pu : float  Minimum overnight load [pu]. Default: 0.40.
    seasonal_amplitude : float  Winter/summer load difference [pu].
    noise_std : float     Measurement noise [pu].
    seed : int

    Returns
    -------
    np.ndarray  Load factor K [pu], shape (duration_hours,).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(duration_hours)
    hour_of_day = t % 24
    day_of_year = (t / 24).astype(int) % 365

    # Diurnal pattern: morning peak (8h) + evening peak (20h)
    morning_peak = 0.35 * np.exp(-0.5 * ((hour_of_day - 8.0) / 2.5)**2)
    evening_peak = 0.40 * np.exp(-0.5 * ((hour_of_day - 19.0) / 2.5)**2)
    midday_shoulder = 0.15 * np.exp(-0.5 * ((hour_of_day - 13.0) / 3.0)**2)
    diurnal = base_load_pu + morning_peak + evening_peak + midday_shoulder

    # Weekend reduction (15% lower on Saturday/Sunday)
    day_of_week = (t / 24).astype(int) % 7
    weekend = np.where(day_of_week >= 5, 0.85, 1.0)

    # Seasonal variation: highest in winter (Jan=day 0), lowest in summer (July≈day 180)
    seasonal = 1.0 + seasonal_amplitude * np.cos(2 * np.pi * day_of_year / 365)

    k = diurnal * weekend * seasonal

    # Normalize to peak_load_pu
    k = k * (peak_load_pu / np.max(k))

    # Add noise (measurement uncertainty + small random load fluctuations)
    k += rng.normal(0, noise_std, duration_hours)
    k = np.clip(k, 0.05, 2.0)   # physical bounds

    return k


def generate_ambient_temperature(
    duration_hours: int = 8760,
    latitude_class: str = 'balkans',  # 'nordic', 'balkans', 'mediterranean'
    seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic annual ambient temperature profile [°C].

    Uses a cosine seasonal model with diurnal variation and stochastic weather.

    Parameters for latitude classes (approximate):
    - Nordic:       T_mean=5°C, amplitude=15°C, diurnal=8°C
    - Balkans:      T_mean=12°C, amplitude=15°C, diurnal=10°C (Serbia)
    - Mediterranean:T_mean=18°C, amplitude=12°C, diurnal=10°C
    """
    rng = np.random.default_rng(seed)
    t = np.arange(duration_hours)
    day = t / 24.0
    hour = t % 24

    params = {
        'nordic':        {'mean': 5.0,  'seasonal': 15.0, 'diurnal': 8.0},
        'balkans':       {'mean': 12.0, 'seasonal': 15.0, 'diurnal': 10.0},
        'mediterranean': {'mean': 18.0, 'seasonal': 12.0, 'diurnal': 10.0},
    }
    p = params.get(latitude_class, params['balkans'])

    # Seasonal: coldest in Jan (day 15), hottest in July (day 196)
    seasonal = p['seasonal'] * np.cos(2 * np.pi * (day - 196) / 365)
    # Diurnal: coldest at 6h, hottest at 15h
    diurnal = 0.5 * p['diurnal'] * np.sin(2 * np.pi * (hour - 6) / 24)

    # Stochastic weather variation (autocorrelated)
    weather_noise = np.zeros(duration_hours)
    rho = 0.98   # hour-to-hour autocorrelation
    for i in range(1, duration_hours):
        weather_noise[i] = rho * weather_noise[i-1] + rng.normal(0, 1.5)

    theta_a = p['mean'] + seasonal + diurnal + weather_noise
    # Clip to realistic extremes for each latitude class
    extremes = {'nordic': (-40, 38), 'balkans': (-25, 45), 'mediterranean': (-5, 48)}
    lo, hi = extremes.get(latitude_class, (-40, 50))
    return np.clip(theta_a, lo, hi).astype(np.float64)


def inject_overload_events(
    load_factor: np.ndarray,
    events: Optional[list] = None,
) -> np.ndarray:
    """
    Inject known overload events into a baseline load profile.

    Parameters
    ----------
    load_factor : np.ndarray  Baseline load profile [pu].
    events : list of dicts {start_h, duration_h, peak_k, type}
        start_h : int         Hour index where event begins
        duration_h : int      Duration [hours]
        peak_k : float        Peak load factor during event [pu]
        type : str            'ramp', 'step', or 'pulse'

    Returns
    -------
    np.ndarray  Modified load profile with injected overloads.
    """
    if events is None:
        # Default: 2 overload events (summer cooling failure + winter storm)
        events = [
            {'start_h': 3200, 'duration_h': 6,  'peak_k': 1.40, 'type': 'step',
             'label': 'Summer cooling failure (July)'},
            {'start_h': 500,  'duration_h': 12, 'peak_k': 1.35, 'type': 'ramp',
             'label': 'Winter storm overload (January)'},
        ]

    k = load_factor.copy()
    N = len(k)

    for ev in events:
        s = max(0, int(ev['start_h']))
        d = int(ev['duration_h'])
        e = min(N, s + d)
        peak = float(ev['peak_k'])
        ev_type = ev.get('type', 'step')

        if ev_type == 'step':
            k[s:e] = np.maximum(k[s:e], peak)
        elif ev_type == 'ramp':
            ramp = np.linspace(k[s], peak, d // 2 + 1)[:e - s]
            k[s:s + len(ramp)] = np.maximum(k[s:s + len(ramp)], ramp)
        elif ev_type == 'pulse':
            mid = (s + e) // 2
            pulse = peak * np.exp(-0.5 * ((np.arange(s, e) - mid) / (d / 4))**2)
            k[s:e] = np.maximum(k[s:e], pulse)

    return np.clip(k, 0.0, 2.0)
