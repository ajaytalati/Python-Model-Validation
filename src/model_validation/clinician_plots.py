"""Clinician-facing diagnostic plots for SWAT trajectories.

Three plot panels per scenario:
  - latent states: W, Z̃, a, T, with circadian C(t) overlay
  - observation channels: HR, sleep stages, steps, stress
  - entrainment: E_dyn (drives μ in SDE), μ(E_dyn), T(t) vs T*=√(μ/η)

Adapted from upstream's `models/swat/sim_plots.py`. Key differences:
  - Operates on the 4-state vector (W, Z̃, a, T) — no carrying
    states for V_h, V_n, V_c (those are now controls in the OT-Control
    framing).
  - Time grid in DAYS (not hours).
  - Both pre-fix and Option-C drift functions can be plotted; the
    entrainment curve uses whichever drift you supplied.
  - Observation channels are computed on-the-fly from the latent
    trajectory + parameters using the spec's deterministic formulas
    plus stochastic sampling (HR Gaussian, sleep ordinal, steps Poisson,
    stress Gaussian).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .runner import (
    ModelInterface, _make_sde_solve, _make_ode_solve,
)


_PHI_MORNING = -math.pi / 3.0


# =========================================================================
# TRAJECTORY PRODUCTION
# =========================================================================

def simulate_for_plotting(
    model: ModelInterface, V_h: float, V_n: float, V_c: float,
    T_0: float, D: float, *, dt: float = 0.005,
    n_realisations: int = 8, rng_seed: int = 42,
    deterministic: bool = False,
) -> dict:
    """Simulate the model and return a dict of arrays for plotting.

    If deterministic=True, runs the diffrax ODE for one trajectory; the
    'trajectories' array has shape (1, n_pts, 4). Otherwise runs the SDE
    over n_realisations and returns shape (n_realisations, n_pts, 4).
    """
    if deterministic:
        # ODE path: one trajectory.
        ode = _make_ode_solve(model, D)
        # ODE is a scalar T_end function; for plotting we need full trajectory.
        # Re-implement here using diffrax directly.
        import diffrax
        u = jnp.array([V_h, V_n, V_c])
        init = jnp.asarray(model.init_state).at[model.amplitude_index].set(T_0)

        def vf(t, y, args):
            return model.drift(t, y, args, model.params)

        n_pts = max(int(round(D / dt)) + 1, 200)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Kvaerno5(),
            t0=0.0, t1=D, dt0=0.01, y0=init, args=u,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, D, n_pts)),
            max_steps=200_000,
        )
        return {
            "t_days": np.asarray(sol.ts),
            "trajectories": np.asarray(sol.ys)[None, :, :],  # (1, n_pts, 4)
            "V_h": V_h, "V_n": V_n, "V_c": V_c, "T_0": T_0, "D": D,
            "params": model.params,
        }

    sde = _make_sde_solve(model, D, dt=dt)
    rng = jax.random.PRNGKey(rng_seed)
    rngs = jax.random.split(rng, n_realisations)
    batched = jax.vmap(sde, in_axes=(None, None, None, None, 0))
    traj = batched(V_h, V_n, V_c, T_0, rngs)        # (M, n_pts, 4)
    n_pts = int(traj.shape[1])
    t_days = np.linspace(0.0, D, n_pts)
    return {
        "t_days": t_days,
        "trajectories": np.asarray(traj),
        "V_h": V_h, "V_n": V_n, "V_c": V_c, "T_0": T_0, "D": D,
        "params": model.params,
    }


# =========================================================================
# OBSERVATION CHANNELS
# =========================================================================

def _sigmoid_np(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# Observation-channel defaults from upstream PARAM_SET_A. The OT-Control
# vendored params dict is minimal (drift only), so we fill these in here.
_OBS_DEFAULTS = {
    "HR_base":     50.0, "alpha_HR":   25.0, "sigma_HR":   8.0,
    "delta_c":     1.5,
    "lambda_base": 0.5,  "lambda_step": 200.0, "W_thresh":  0.6,
    "s_base":      30.0, "alpha_s":    40.0, "beta_s":     10.0,
    "sigma_s":     15.0,
}


def _merge_obs_defaults(p: dict) -> dict:
    out = dict(_OBS_DEFAULTS)
    out.update(p)
    return out


def generate_observations(traj_dict: dict, rng_seed: int = 0) -> dict:
    """Generate the four observation channels from a single trajectory.

    `traj_dict` must contain `t_days` and `trajectories` (M, n_pts, 4).
    Uses the FIRST realisation only (idx=0) for the observation channels.
    Output dict has keys: 'hr', 'sleep_level', 'steps', 'stress' (each
    a 1-D array indexed by t_days), plus the 't_days' grid.
    """
    p = _merge_obs_defaults(traj_dict["params"])
    t_days = traj_dict["t_days"]
    traj = traj_dict["trajectories"][0]              # (n_pts, 4)
    W, Zt, a, T = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    n = len(t_days)
    rng = np.random.default_rng(rng_seed)

    # HR = HR_base + alpha_HR * W + N(0, sigma_HR^2)
    hr_mean = p["HR_base"] + p["alpha_HR"] * W
    hr = hr_mean + rng.normal(0.0, p["sigma_HR"], size=n)

    # 3-level sleep: P(wake) = 1 - σ(Zt - c1); P(deep) = σ(Zt - c2)
    c1 = p["c_tilde"]
    c2 = c1 + p.get("delta_c", 1.5)
    s1 = _sigmoid_np(Zt - c1)
    s2 = _sigmoid_np(Zt - c2)
    draws = rng.random(size=n)
    sleep_level = np.where(draws < 1.0 - s1, 0,
                  np.where(draws < 1.0 - s2, 1, 2)).astype(int)

    # Steps: 15-min bin Poisson with rate r(W) = lambda_base + lambda_step*sigmoid(10*(W-W_thresh))
    bin_hours = 0.25
    dt_days = float(t_days[1] - t_days[0]) if n > 1 else 1.0 / 288.0
    bin_size = max(int(round(bin_hours / 24.0 / dt_days)), 1)
    n_bins = n // bin_size
    if n_bins == 0:
        n_bins = 1
        bin_size = n
    W_bins = W[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    rate = p.get("lambda_base", 0.5) + p.get("lambda_step", 200.0) * \
           _sigmoid_np(10.0 * (W_bins - p.get("W_thresh", 0.6)))
    expected = rate * bin_hours
    step_counts = rng.poisson(expected).astype(int)
    step_t_days = (np.arange(n_bins) * bin_size) * dt_days

    # Stress = s_base + alpha_s*W + beta_s*V_n + N(0, sigma_s^2)
    V_n = traj_dict["V_n"]
    sr_mean = p.get("s_base", 30.0) + p.get("alpha_s", 40.0) * W + \
              p.get("beta_s", 10.0) * V_n
    stress = sr_mean + rng.normal(0.0, p.get("sigma_s", 15.0), size=n)
    stress = np.clip(stress, 0.0, 100.0)

    return {
        "t_days": t_days,
        "hr": hr,
        "hr_mean": hr_mean,
        "sleep_level": sleep_level,
        "sleep_t_days": t_days,
        "step_counts": step_counts,
        "step_t_days": step_t_days,
        "step_rate_per_15min": (rate * bin_hours),
        "stress": stress,
        "stress_mean": sr_mean,
    }


# =========================================================================
# ENTRAINMENT QUALITY (windowed diagnostic, aware of V_c ≠ 0)
# =========================================================================

def _compute_E_obs(t_days: np.ndarray, W: np.ndarray, Zt: np.ndarray,
                    A_scale: float = 6.0) -> np.ndarray:
    """Windowed amp × phase-correlation E (from upstream's sim_plots).

    Phase is correlated against the EXTERNAL light cycle C(t) = sin(2π·t + φ_0)
    so V_c ≠ 0 (subject's internal phase shift) shows up as lost entrainment.
    """
    C_ext = np.sin(2.0 * np.pi * t_days + _PHI_MORNING)
    n = len(t_days)
    if n < 3:
        return np.zeros(n)
    dt_days = float(t_days[1] - t_days[0])
    win = max(int(round(1.0 / dt_days)), 3)        # 24h window in samples
    E = np.zeros(n)
    for i in range(n):
        lo = max(i - win + 1, 0)
        Ww, Zw, Cw = W[lo:i + 1], Zt[lo:i + 1], C_ext[lo:i + 1]
        if len(Ww) < 3:
            continue
        amp_W = (Ww.max() - Ww.min()) / 1.0
        amp_Z = (Zw.max() - Zw.min()) / A_scale
        # Pearson with safety
        sw, sz, sc = Ww.std(), Zw.std(), Cw.std()
        if sw < 1e-12 or sz < 1e-12 or sc < 1e-12:
            continue
        phase_W = max(((Ww - Ww.mean()) * (Cw - Cw.mean())).mean() / (sw * sc), 0.0)
        phase_Z = max(((Zw - Zw.mean()) * (-(Cw - Cw.mean()))).mean() / (sz * sc), 0.0)
        E[i] = (amp_W * phase_W) * (amp_Z * phase_Z)
    return np.clip(E, 0.0, 1.0)


def _compute_E_dyn(traj: np.ndarray, V_h: float, V_n: float, V_c: float,
                    params: Dict[str, float], variant: str = "vendored") -> np.ndarray:
    """Dynamics-side E(t). Uses Option C's amp formulation if variant='option-c'."""
    a, T = traj[:, 2], traj[:, 3]
    alpha_T = params["alpha_T"]
    beta_Z = params["beta_Z"]

    if variant == "option-c":
        lam_amp_W = params["lambda_amp_W"]
        lam_amp_Z = params["lambda_amp_Z"]
        V_n_scale = params.get("V_n_scale", 2.0)
        V_c_max = params.get("V_c_max", 3.0)
        # A = λ_amp · V_h (no +1 offset — V_h=0 gives no entrainment)
        A_W = lam_amp_W * V_h
        A_Z = lam_amp_Z * V_h
        B_W = V_n - a + alpha_T * T
        B_Z = -V_n + beta_Z * a
        amp_W = _sigmoid_np(B_W + A_W) - _sigmoid_np(B_W - A_W)
        amp_Z = _sigmoid_np(B_Z + A_Z) - _sigmoid_np(B_Z - A_Z)
        # Multiplicative V_n dampener (issue #5 / Option D). damp applied
        # at the E level (single multiplier — matches entrainment_quality_option_c).
        damp = np.exp(-V_n / V_n_scale)
        # Phase quality: clamped quarter-period cosine, zero at |V_c| ≥ V_c_max.
        V_c_eff = min(abs(V_c), V_c_max)
        phase = np.cos(np.pi * V_c_eff / (2.0 * V_c_max))
        return damp * amp_W * amp_Z * phase
    else:
        mu_W_slow = V_h + V_n - a + alpha_T * T
        mu_Z_slow = -V_n + beta_Z * a
        sW = _sigmoid_np(mu_W_slow); sZ = _sigmoid_np(mu_Z_slow)
        amp_W = 4.0 * sW * (1.0 - sW)
        amp_Z = 4.0 * sZ * (1.0 - sZ)
        damp = 1.0   # no V_n dampener in vendored variant

    phase = max(np.cos(2.0 * np.pi * V_c / 24.0), 0.0)
    return damp * amp_W * amp_Z * phase


# =========================================================================
# THE THREE PLOT FUNCTIONS
# =========================================================================

def plot_latents(traj_dict: dict, save_path: Path, variant: str = "vendored"):
    """6-panel: W, Z̃, a, T, C_ext, V_h/V_n bars."""
    t = traj_dict["t_days"]
    p = traj_dict["params"]
    traj = traj_dict["trajectories"]                 # (M, n_pts, 4)
    M = traj.shape[0]
    show_traces = min(M, 6)

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.7, show_traces))

    def overlay(ax, comp_idx, label, ylim=None, ref_lines=None):
        for k in range(show_traces):
            ax.plot(t, traj[k, :, comp_idx], lw=0.5, color=colors[k], alpha=0.7)
        ax.plot(t, traj[:, :, comp_idx].mean(0), lw=1.6, color="black", label="mean")
        if ref_lines:
            for y, name, c in ref_lines:
                ax.axhline(y, ls="--", color=c, alpha=0.6, label=name)
        ax.set_ylabel(label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    overlay(axes[0], 0, "W (wakefulness)", ylim=(-0.05, 1.05))
    overlay(axes[1], 1, "Z̃ (sleep depth)",
            ref_lines=[(p["c_tilde"], f"c_tilde = {p['c_tilde']:.2f}", "red")])
    overlay(axes[2], 2, "a (adenosine)")

    # T panel with deterministic ceiling
    mu_max = p["mu_0"] + p["mu_E"]
    refs = [(0.0, "T = 0", "gray")]
    if mu_max > 0:
        T_ceil = math.sqrt(mu_max / p["eta"])
        refs.append((T_ceil, f"T*(E=1) = {T_ceil:.2f}", "green"))
    overlay(axes[3], 3, "T (testosterone)", ref_lines=refs)

    # External light cycle
    C_ext = np.sin(2.0 * np.pi * t + _PHI_MORNING)
    axes[4].plot(t, C_ext, lw=0.6, color="seagreen", label="C_ext (light)")
    axes[4].axhline(0.0, ls=":", color="gray", alpha=0.4)
    axes[4].set_ylabel("C(t)")
    axes[4].set_xlabel("t (days)")
    axes[4].set_ylim(-1.1, 1.1)
    axes[4].legend(loc="upper right", fontsize=7)
    axes[4].grid(alpha=0.3)

    fig.suptitle(
        f"SWAT latents — {variant} — V_h={traj_dict['V_h']:.2f}, "
        f"V_n={traj_dict['V_n']:.2f}, V_c={traj_dict['V_c']:.1f}h, "
        f"T_0={traj_dict['T_0']:.2f}, D={traj_dict['D']:.0f}d  "
        f"(M={M} realisations, {show_traces} shown)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_observations(obs: dict, traj_dict: dict, save_path: Path,
                       variant: str = "vendored"):
    """4-panel: HR, sleep stages, steps, stress."""
    t = obs["t_days"]
    p = _merge_obs_defaults(traj_dict["params"])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 2, 2]})

    # HR
    axes[0].plot(t, obs["hr_mean"], lw=0.6, color="crimson", alpha=0.6,
                  label="HR mean (from W)")
    axes[0].scatter(t, obs["hr"], s=2, alpha=0.35, color="navy",
                     label=f"HR obs (σ={p['sigma_HR']:.1f})")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Sleep stages
    axes[1].fill_between(t, 0, obs["sleep_level"], step="mid",
                         color="midnightblue", alpha=0.7, label="sleep level")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["wake", "light+rem", "deep"])
    axes[1].set_ylim(-0.3, 2.3)
    axes[1].set_ylabel("Sleep stage")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    # Steps
    bar_w = 0.25 / 24.0 * 0.9
    axes[2].bar(obs["step_t_days"], obs["step_counts"], width=bar_w,
                 color="seagreen", alpha=0.7, edgecolor="none",
                 label="steps/15min (Poisson)")
    axes[2].plot(obs["step_t_days"], obs["step_rate_per_15min"],
                  lw=0.6, color="darkgreen", alpha=0.6, label="rate × bin")
    axes[2].set_ylabel("Steps per 15 min")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    # Stress
    axes[3].plot(t, obs["stress_mean"], lw=0.6, color="purple", alpha=0.6,
                  label="stress mean")
    axes[3].scatter(t, obs["stress"], s=1.5, alpha=0.35, color="darkviolet",
                     label=f"stress obs (σ={p.get('sigma_s', 15.0):.1f})")
    axes[3].set_ylabel("Stress (0-100)")
    axes[3].set_xlabel("t (days)")
    axes[3].set_ylim(-5, 105)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"SWAT observations — {variant} — V_h={traj_dict['V_h']:.2f}, "
        f"V_n={traj_dict['V_n']:.2f}, V_c={traj_dict['V_c']:.1f}h",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_entrainment(traj_dict: dict, save_path: Path,
                      variant: str = "vendored"):
    """3-panel: E_dyn (and E_obs), μ(E_dyn), T(t) vs T*."""
    t = traj_dict["t_days"]
    p = traj_dict["params"]
    traj0 = traj_dict["trajectories"][0]              # first realisation
    V_h = traj_dict["V_h"]; V_n = traj_dict["V_n"]; V_c = traj_dict["V_c"]
    A_scale = p["A_scale"]

    E_dyn = _compute_E_dyn(traj0, V_h, V_n, V_c, p, variant=variant)
    E_obs = _compute_E_obs(t, traj0[:, 0], traj0[:, 1], A_scale=A_scale)
    mu = p["mu_0"] + p["mu_E"] * E_dyn
    E_crit = -p["mu_0"] / p["mu_E"]
    T_star = np.where(mu > 0, np.sqrt(np.maximum(mu, 0.0) / p["eta"]), 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, E_dyn, lw=1.0, color="darkviolet",
                  label=f"E_dyn ({variant} formulation)")
    axes[0].plot(t, E_obs, lw=0.8, ls="--", color="darkorange",
                  label="E_obs (24h windowed amp × phase)")
    axes[0].axhline(E_crit, ls=":", color="red", alpha=0.7,
                     label=f"E_crit = {E_crit:.2f}")
    axes[0].set_ylabel("E(t)  (entrainment)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, mu, lw=1.0, color="darkblue", label="μ(E_dyn)")
    axes[1].axhline(0.0, ls=":", color="black", alpha=0.5)
    axes[1].fill_between(t, mu, 0, where=(mu > 0), color="green", alpha=0.15,
                          label="μ > 0 (super-critical)")
    axes[1].fill_between(t, mu, 0, where=(mu < 0), color="red", alpha=0.15,
                          label="μ < 0 (sub-critical)")
    axes[1].set_ylabel("μ = μ_0 + μ_E·E")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, traj0[:, 3], lw=1.0, color="crimson", label="T(t)")
    axes[2].plot(t, T_star, lw=0.8, ls="--", color="green",
                  label="T* = √(μ/η)")
    axes[2].set_ylabel("T")
    axes[2].set_xlabel("t (days)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"SWAT entrainment — {variant} — V_h={V_h:.2f}, V_n={V_n:.2f}, "
        f"V_c={V_c:.1f}h",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# =========================================================================
# CONVENIENCE: full bundle for one (model, scenario) pair
# =========================================================================

def plot_all_for_scenario(
    model: ModelInterface, scenario_label: str,
    V_h: float, V_n: float, V_c: float, T_0: float,
    save_dir: Path, *, D: float = 14.0,
    variant: str = "vendored", n_realisations: int = 8, rng_seed: int = 42,
):
    """Generate the three plots for one model + scenario into save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    traj_dict = simulate_for_plotting(
        model, V_h, V_n, V_c, T_0, D,
        n_realisations=n_realisations, rng_seed=rng_seed,
    )
    obs = generate_observations(traj_dict, rng_seed=rng_seed + 1)

    plot_latents(traj_dict,        save_dir / "latents.png",      variant=variant)
    plot_observations(obs, traj_dict, save_dir / "observations.png", variant=variant)
    plot_entrainment(traj_dict,    save_dir / "entrainment.png",  variant=variant)
    return save_dir
