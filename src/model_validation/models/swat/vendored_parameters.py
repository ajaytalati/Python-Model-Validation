"""
_vendored_models/swat/parameters.py — Default SWAT model parameters.
======================================================================
Date:    26 April 2026
Version: 1.1.0
Source:  Python-Model-Development-Simulation @ main, version_1/models/swat/simulation.py PARAM_SET_A

Re-pulled from upstream on 2026-04-26 to incorporate the re-tuned
beta_Z and c_tilde values flagged in the upstream issue tracker:
  * beta_Z: 2.5 -> 4.0    (closes upstream #5 / #7 — Zt amplitude not
                          reaching deep-sleep threshold; tau_a=3h drains
                          a too fast otherwise)
  * c_tilde: 3.0 -> 2.5   (closes upstream #8 — sleep fraction was ~19%
                          with new beta_Z, lowered threshold restores ~33%)
  * T_Z:     0.01 -> 0.05 (matches upstream PARAM_SET_A noise temperature)

(c_tilde is an observation-channel parameter and does not affect the OT
engine's drift/diffusion. It is recorded here for traceability.)

The OT-Control engine works in **days** as the unit of time. The
upstream SWAT spec uses hours. The conversion is applied here once,
during construction of the parameter dictionary.

Conversion summary
------------------
- timescales tau_W, tau_Z, tau_a, tau_T:      hours -> days       (divide by 24)
- noise temperatures T_W, T_Z, T_a, T_T:      per-hour -> per-day (multiply by 24)
- circadian formula sin(2*pi*t/24 + phi):     becomes sin(2*pi*t + phi) when t in days
"""

from __future__ import annotations

from typing import Dict


# Hours-per-day (used only for unit conversion at parameter-build time).
_HOURS_PER_DAY = 24.0


def default_swat_parameters() -> Dict[str, float]:
    """Return the canonical 'healthy-baseline' SWAT parameter dictionary.

    Values from upstream PARAM_SET_A as of 2026-04-26 (post-fix), with
    all timescales converted from hours to days for engine consistency.

    Returns:
        Dict with keys:
            - kappa, lmbda, gamma_3, beta_Z      (sigmoid couplings)
            - tau_W, tau_Z, tau_a, tau_T         (timescales, in DAYS)
            - A_scale                            (Z rescaling = 6, frozen)
            - phi_0                              (circadian phase = -pi/3)
            - mu_0, mu_E, eta, alpha_T           (Stuart-Landau parameters)
            - T_W, T_Z, T_a, T_T                 (diffusion variances per DAY)
            - c_tilde                            (sleep-detection threshold; obs only)
    """
    import math

    # Block F — fast subsystem couplings.
    p = {
        'kappa':   6.67,
        'lmbda':   32.0,
        'gamma_3': 8.0,
        'beta_Z':  4.0,            # ← was 2.5; upstream re-tune 2026-04-26
        'A_scale': 6.0,            # Frozen scale for tilde Z domain.
        'phi_0':   -math.pi / 3,   # Morning-type baseline.
    }

    # Timescales — hours in spec, days in engine.
    p['tau_W'] = 2.0  / _HOURS_PER_DAY    # 0.0833 days
    p['tau_Z'] = 2.0  / _HOURS_PER_DAY    # 0.0833 days
    p['tau_a'] = 3.0  / _HOURS_PER_DAY    # 0.125  days
    p['tau_T'] = 48.0 / _HOURS_PER_DAY    # 2.0    days

    # Block T — Stuart-Landau testosterone parameters.
    p['mu_0']    = -0.5   # baseline bifurcation parameter
    p['mu_E']    =  1.0   # entrainment coupling
    p['eta']     =  0.5   # cubic saturation
    p['alpha_T'] =  0.3   # T -> u_W loading

    # Diffusion temperatures — per-hour in spec, per-day here.
    p['T_W'] = 0.01   * _HOURS_PER_DAY    # 0.24   / day
    p['T_Z'] = 0.05   * _HOURS_PER_DAY    # 1.20   / day  (← upstream is 0.05/h, was 0.01/h)
    p['T_a'] = 0.01   * _HOURS_PER_DAY    # 0.24   / day
    p['T_T'] = 0.0001 * _HOURS_PER_DAY    # 0.0024 / day

    # Observation-channel parameter (not used in OT drift/diffusion;
    # included for traceability against upstream).  Bumped 2.5 -> 3.0
    # alongside the V_h-anabolic structural fix: under the corrected
    # entrainment formula the healthy-regime W spends more time low
    # (more sleep), so c_tilde=2.5 produced a sleep fraction ~45%
    # (above the 25-40% clinical target band); 3.0 brings it back
    # into range.
    p['c_tilde'] = 3.0

    # ── V_h-anabolic structural fix (upstream PR #11) ─────────────────
    # V_h modulates entrainment amplitude rather than entering u_W
    # directly. New estimable parameters:
    p['lambda_amp_W'] = 5.0   # V_h gain into W-side amplitude
    p['lambda_amp_Z'] = 8.0   # V_h gain into Z-side amplitude
    p['V_n_scale']    = 2.0   # V_n damper scale
    # Clinical constant (not estimable): any |V_c| >= V_c_max collapses
    # the phase factor to 0.
    p['V_c_max']      = 3.0

    return p
