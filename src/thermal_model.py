"""
thermal_model.py — IEC 60076-7 Transformer Thermal Model
==========================================================
Implements the IEEE/IEC loading guide thermal model for oil-immersed
power transformers (IEC 60076-7:2005, Section 7 — Thermal model).

Physical model
--------------
Two first-order differential equations:

Top-oil temperature rise above ambient:
    τ_o × d(Δθ_o)/dt = Δθ_or × ((1 + R·K²)/(1+R))^n − Δθ_o

Hot-spot-to-top-oil temperature gradient:
    τ_w × d(Δθ_h)/dt = Δθ_hr × K^(2m) − Δθ_h

Hot-spot temperature:
    θ_h = θ_a + Δθ_o + H × Δθ_h

where:
    θ_a    = ambient temperature [°C]
    θ_o    = top-oil temperature [°C]
    θ_h    = hot-spot temperature [°C]
    Δθ_or  = rated top-oil rise [K] — nameplate value
    Δθ_hr  = rated winding (hot-spot) rise over top-oil [K]
    H      = hot-spot factor [−] (accounts for non-uniform winding temperature)
    R      = ratio of load losses to no-load losses at rated conditions
    K      = load factor = I / I_rated  [pu]
    τ_o    = thermal time constant of oil [min]
    τ_w    = winding time constant [min]
    n      = oil exponent (cooling mode dependent)
    m      = winding exponent (cooling mode dependent)

Cooling mode exponents (IEC 60076-7, Table 4):
    ONAN (natural oil, natural air): n=0.9, m=0.8
    ONAF (natural oil, forced air):  n=0.9, m=0.8
    OFAF (forced oil, forced air):   n=1.0, m=1.0
    ODAF (directed forced oil):      n=1.0, m=1.0

Typical nameplate values for 110/10 kV, 40 MVA ONAN transformer
(representative EPS substation transformer, Serbia/Balkans region):
    Δθ_or = 55 K   (IEC limit: 60 K max)
    Δθ_hr = 23 K
    H     = 1.3
    R     = 8.0    (ratio load/no-load losses)
    τ_o   = 180 min
    τ_w   = 10 min

References
----------
IEC 60076-7:2005. Power transformers — Part 7: Loading guide for
oil-immersed power transformers.

IEEE C57.91-2011. Guide for Loading Mineral-Oil-Immersed Transformers.

Susa, D. & Nordman, H. (2009). Practical application of IEC 60076-7
loading guide thermal model for power transformers. IEEE Trans. Power Del.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy.integrate import solve_ivp


@dataclass
class TransformerNameplate:
    """Transformer nameplate thermal parameters."""
    name: str = "40 MVA 110/10 kV ONAN"

    # Rated power and voltages
    mva_rating: float = 40.0      # MVA
    hv_kv: float = 110.0          # HV side voltage [kV]
    lv_kv: float = 10.0           # LV side voltage [kV]

    # Thermal parameters (IEC 60076-7)
    delta_theta_or: float = 55.0  # Rated top-oil rise [K]
    delta_theta_hr: float = 23.0  # Rated winding rise over top-oil [K]
    H_factor: float = 1.3         # Hot-spot factor [-]
    R_ratio: float = 8.0          # Load/no-load loss ratio [-]
    tau_o_min: float = 180.0      # Oil thermal time constant [min]
    tau_w_min: float = 10.0       # Winding time constant [min]

    # Cooling mode exponents (ONAN)
    n_exp: float = 0.9            # Oil exponent
    m_exp: float = 0.8            # Winding exponent
    cooling_mode: str = "ONAN"

    # Insulation limits (IEC 60076-7, Table A.2)
    theta_h_limit_continuous: float = 98.0    # °C — normal aging reference
    theta_h_limit_emergency:  float = 140.0   # °C — emergency max
    theta_o_limit_continuous: float = 90.0    # °C — top-oil limit
    k_factor_limit_continuous: float = 1.0    # pu — continuous rated load
    k_factor_limit_emergency:  float = 1.5    # pu — 4h emergency overload

    # Typical no-load losses (for loss separation)
    no_load_loss_kw: float = 40.0    # kW (approximately 0.1% of rating)
    load_loss_rated_kw: float = 320.0  # kW (0.8% of rating at rated load)


# Standard nameplate: EPS-type 40 MVA substation transformer
EPS_TRAFO_40MVA = TransformerNameplate()

# Smaller distribution transformer (10 MVA)
DIST_TRAFO_10MVA = TransformerNameplate(
    name="10 MVA 35/10 kV ONAN",
    mva_rating=10.0, hv_kv=35.0, lv_kv=10.0,
    delta_theta_or=50.0, delta_theta_hr=20.0,
    H_factor=1.1, R_ratio=6.0,
    tau_o_min=150.0, tau_w_min=7.0,
    no_load_loss_kw=12.0, load_loss_rated_kw=80.0,
)


@dataclass
class ThermalResult:
    """Output of thermal simulation."""
    timestamps_h: np.ndarray       # Time axis [hours]
    theta_ambient: np.ndarray      # Ambient temperature [°C]
    theta_top_oil: np.ndarray      # Top-oil temperature [°C]
    theta_hot_spot: np.ndarray     # Hot-spot temperature [°C]
    delta_theta_oil: np.ndarray    # Oil temp rise above ambient [K]
    delta_theta_winding: np.ndarray  # Winding rise over top-oil [K]
    load_factor_k: np.ndarray      # Load factor K [pu]
    nameplate: TransformerNameplate

    @property
    def peak_hot_spot(self) -> float:
        return float(np.max(self.theta_hot_spot))

    @property
    def time_above_limit(self) -> float:
        """Hours where hot-spot exceeds continuous limit (98°C)."""
        dt = float(np.mean(np.diff(self.timestamps_h)))
        return float(np.sum(self.theta_hot_spot > self.nameplate.theta_h_limit_continuous) * dt)

    @property
    def time_above_emergency(self) -> float:
        """Hours where hot-spot exceeds emergency limit (140°C)."""
        dt = float(np.mean(np.diff(self.timestamps_h)))
        return float(np.sum(self.theta_hot_spot > self.nameplate.theta_h_limit_emergency) * dt)


def simulate_thermal(
    load_factor_k: np.ndarray,
    timestamps_h: np.ndarray,
    theta_ambient: np.ndarray,
    nameplate: TransformerNameplate = EPS_TRAFO_40MVA,
    theta_initial_c: float = 20.0,
) -> ThermalResult:
    """
    Simulate transformer top-oil and hot-spot temperatures.

    Integrates the IEC 60076-7 differential equations using RK45 with
    adaptive step control. Uses time-varying ambient temperature and load.

    Parameters
    ----------
    load_factor_k : np.ndarray
        Per-hour load factor K = I/I_rated [pu]. Values >1.0 = overload.
    timestamps_h : np.ndarray
        Time axis [hours]. Must match length of load_factor_k.
    theta_ambient : np.ndarray
        Ambient temperature at each timestep [°C].
    nameplate : TransformerNameplate
        Transformer thermal parameters.
    theta_initial_c : float
        Initial top-oil temperature [°C]. Default: 20°C (cold start).

    Returns
    -------
    ThermalResult
    """
    np_params = nameplate
    tau_o_h = np_params.tau_o_min / 60.0   # convert to hours
    tau_w_h = np_params.tau_w_min / 60.0

    # Interpolation functions for time-varying inputs
    from scipy.interpolate import interp1d
    k_interp = interp1d(timestamps_h, load_factor_k, kind='linear',
                         fill_value='extrapolate')
    ta_interp = interp1d(timestamps_h, theta_ambient, kind='linear',
                          fill_value='extrapolate')

    def iec_odes(t, y):
        """
        y[0] = Δθ_o  — top-oil rise above ambient [K]
        y[1] = Δθ_h  — hot-spot-to-top-oil gradient [K]
        """
        delta_o, delta_h = y[0], y[1]
        K = float(np.clip(k_interp(t), 0, 2.0))  # clip at 200% for stability

        # Steady-state top-oil rise at current load
        n = np_params.n_exp
        R = np_params.R_ratio
        delta_o_ss = np_params.delta_theta_or * ((1 + R * K**2) / (1 + R)) ** n

        # Steady-state winding gradient at current load
        m = np_params.m_exp
        delta_h_ss = np_params.delta_theta_hr * K ** (2 * m)

        d_delta_o_dt = (delta_o_ss - delta_o) / tau_o_h
        d_delta_h_dt = (delta_h_ss - delta_h) / tau_w_h

        return [d_delta_o_dt, d_delta_h_dt]

    # Initial conditions: cold transformer at ambient
    delta_o_init = theta_initial_c - float(theta_ambient[0])
    delta_o_init = max(0.0, delta_o_init)
    y0 = [delta_o_init, 0.0]

    t_span = (float(timestamps_h[0]), float(timestamps_h[-1]))
    t_eval = timestamps_h

    sol = solve_ivp(
        iec_odes,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-6,
    )

    delta_o = sol.y[0]
    delta_h = sol.y[1]

    theta_o = theta_ambient + delta_o
    theta_h = theta_o + np_params.H_factor * delta_h

    return ThermalResult(
        timestamps_h=timestamps_h,
        theta_ambient=theta_ambient,
        theta_top_oil=theta_o,
        theta_hot_spot=theta_h,
        delta_theta_oil=delta_o,
        delta_theta_winding=delta_h,
        load_factor_k=load_factor_k,
        nameplate=nameplate,
    )
