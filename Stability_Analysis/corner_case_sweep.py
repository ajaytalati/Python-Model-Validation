"""Lyapunov-stability numerical sweep — Option D SWAT model.

Goal
----
Of the 8 binary corners of the control box `(V_h, V_n, V_c) in {low, high}^3`,
**only the healthy corner (V_h=1, V_n=0, V_c=0) should yield T = T_star**. All
7 pathological corners should give T much less than T_star or T -> 0.

If this doesn't hold, the OT-Control optimiser could in principle find
pathological corners that yield high T — and we'd never know.

Method
------
1. **8-corner sweep**: deterministic D=60-day simulation from initial
   state T_0=0.5 at each corner. Plus 32 stochastic trajectories with
   full noise (vmap'd, JIT'd).
2. **IC sweep at corner 4 (healthy)**: 32 random initial conditions
   spanning (W_0, Z_0, a_0, T_0) in [0,1]^3 x [0,1]. Confirms global
   convergence — the healthy attractor's basin is the entire physical
   state space.
3. **Phase-plane plot**: project trajectories to (E_dyn, T) plane.

Outputs
-------
- results/corner_T_end.png       — bar chart of T(D) per corner
- results/T_trajectories.png     — T(t) over 60 days for all 8 corners
- results/phase_plane.png        — (E_dyn, T) phase-plane trajectories
- results/init_cond_sweep.png    — 32 IC convergence at the healthy corner
- results/corner_summary.json    — numerical results
"""
from __future__ import annotations
import json
import sys                                          # noqa: F401  (used in main)
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import diffrax

from model_validation.models.swat.option_c_dynamics import (
    option_c_model, swat_drift_option_c, entrainment_quality_option_c,
)
from model_validation.models.swat.vendored_dynamics import swat_diffusion

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# =========================================================================
# 8-corner setup
# =========================================================================

V_H_LOW, V_H_HIGH = 0.0, 1.0
V_N_LOW, V_N_HIGH = 0.0, 5.0
V_C_LOW, V_C_HIGH = 0.0, 6.0

CORNERS = [
    # (label, V_h, V_n, V_c, reading)
    ("0  (low,  low,  low) ", V_H_LOW,  V_N_LOW,  V_C_LOW,  "depleted, no stress, aligned"),
    ("1  (low,  low,  high)", V_H_LOW,  V_N_LOW,  V_C_HIGH, "depleted, no stress, phase-shift"),
    ("2  (low,  high, low) ", V_H_LOW,  V_N_HIGH, V_C_LOW,  "depleted, severe stress, aligned"),
    ("3  (low,  high, high)", V_H_LOW,  V_N_HIGH, V_C_HIGH, "depleted, severe stress, phase-shift"),
    ("4  (high, low,  low) ", V_H_HIGH, V_N_LOW,  V_C_LOW,  "HEALTHY"),
    ("5  (high, low,  high)", V_H_HIGH, V_N_LOW,  V_C_HIGH, "healthy V_h, no stress, phase-shift"),
    ("6  (high, high, low) ", V_H_HIGH, V_N_HIGH, V_C_LOW,  "healthy V_h, severe stress, aligned"),
    ("7  (high, high, high)", V_H_HIGH, V_N_HIGH, V_C_HIGH, "healthy V_h, severe stress, phase-shift"),
]

D_DAYS = 60.0
INIT_STATE = jnp.array([0.5, 3.5, 0.5, 0.5])      # canonical
N_STOCH_PER_CORNER = 32
SDE_DT = 0.02                                       # coarser dt for speed
N_PTS_DET = 600
N_PTS_STOCH = max(int(round(D_DAYS / SDE_DT)), 32)


# =========================================================================
# JIT-compiled deterministic + stochastic solvers
# =========================================================================

def _build_deterministic_solver(params):
    """Return a JIT'd function (V_h, V_n, V_c, init_state) -> (t_grid, traj)."""
    t_grid = jnp.linspace(0.0, D_DAYS, N_PTS_DET)

    def vf(t, y, args):
        V_h, V_n, V_c = args
        u = jnp.array([V_h, V_n, V_c])
        return swat_drift_option_c(t, y, u, params)

    @jax.jit
    def solve(V_h, V_n, V_c, init_state):
        # Tsit5 (explicit 5th-order RK) is fast and adequate at long horizons.
        # Kvaerno5 (implicit) was failing on a few corners with max_steps
        # exhaustion despite the system not being notably stiffer there.
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(),
            t0=0.0, t1=D_DAYS, dt0=0.001, y0=init_state,
            args=(V_h, V_n, V_c),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(ts=t_grid),
            max_steps=2_000_000,
        )
        return sol.ys                                  # (n_pts, 4)

    return t_grid, solve


def _build_stochastic_solver(params):
    """Return a JIT'd vmap'd function (V_h, V_n, V_c, init_state, rng_array) -> traj_array."""

    def drift_fn(t, y, args):
        V_h, V_n, V_c = args
        u = jnp.array([V_h, V_n, V_c])
        return swat_drift_option_c(t, y, u, params)

    def diffusion_fn(t, y, args):
        return jnp.diag(swat_diffusion(y, params))

    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, D_DAYS, N_PTS_STOCH))

    def solve_one(V_h, V_n, V_c, init_state, rng):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=D_DAYS, tol=SDE_DT / 2.0, shape=(4,), key=rng
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn),
            diffrax.ControlTerm(diffusion_fn, bm),
        )
        sol = diffrax.diffeqsolve(
            terms, diffrax.Euler(),
            t0=0.0, t1=D_DAYS, dt0=SDE_DT, y0=init_state,
            args=(V_h, V_n, V_c),
            saveat=saveat,
            max_steps=int(D_DAYS / SDE_DT) + 100,
        )
        return sol.ys                                   # (n_pts, 4)

    # vmap over rng (per-trajectory parallelism)
    solve_ensemble = jax.vmap(solve_one, in_axes=(None, None, None, None, 0))
    return jax.jit(solve_ensemble)


# =========================================================================
# Analytical E_dyn along trajectory (vectorised)
# =========================================================================

@partial(jax.jit, static_argnums=())
def _E_along(traj, V_h, V_n, V_c, params):
    W, Z, a, T = traj[..., 0], traj[..., 1], traj[..., 2], traj[..., 3]
    return jax.vmap(
        lambda W_, Z_, a_, T_: entrainment_quality_option_c(
            W_, Z_, a_, T_, V_h, V_n, V_c, params
        )
    )(W, Z, a, T)


# =========================================================================
# Main sweep
# =========================================================================

def main():
    print("=" * 72, flush=True)
    print("Lyapunov stability — 8-corner sweep, Option D", flush=True)
    print("=" * 72, flush=True)

    model = option_c_model()
    p = model.params
    print(f"Pinned tau_T = {p['tau_T']:.2f}d, lambda_amp_Z = {p['lambda_amp_Z']:.1f}",
          flush=True)
    print(f"Free: lambda_amp_W={p['lambda_amp_W']:.1f}, "
          f"V_n_scale={p['V_n_scale']:.1f}, c_tilde={p['c_tilde']:.1f}",
          flush=True)
    print(f"D={D_DAYS}d  N_stoch={N_STOCH_PER_CORNER}  SDE_dt={SDE_DT}", flush=True)
    print(flush=True)

    print("Building & JIT-compiling solvers ...", flush=True)
    t_grid_det, det_solve = _build_deterministic_solver(p)
    stoch_solve = _build_stochastic_solver(p)

    # Warm-up JIT once with a single call (so subsequent calls are fast)
    t0 = time.time()
    _ = det_solve(1.0, 0.0, 0.0, INIT_STATE)
    print(f"  Deterministic JIT compiled in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    rng_warmup = jax.random.split(jax.random.PRNGKey(0), N_STOCH_PER_CORNER)
    _ = stoch_solve(1.0, 0.0, 0.0, INIT_STATE, rng_warmup)
    print(f"  Stochastic JIT compiled in {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ===== 8-corner deterministic sweep =====
    rows = []
    deterministic_trajs = []
    for label, V_h, V_n, V_c, reading in CORNERS:
        t0 = time.time()
        traj = np.asarray(det_solve(V_h, V_n, V_c, INIT_STATE))
        E = np.asarray(_E_along(traj, V_h, V_n, V_c, p))

        T_traj = traj[:, 3]
        T_end = float(T_traj[-1])
        T_last_day_mean = float(T_traj[-int(round(N_PTS_DET / D_DAYS)):].mean())
        E_steady = float(E[len(E) // 2:].mean())
        mu_steady = p["mu_0"] + p["mu_E"] * E_steady

        rows.append({
            "label": label, "V_h": V_h, "V_n": V_n, "V_c": V_c,
            "reading": reading,
            "T_end_det": T_end,
            "T_last_day_mean_det": T_last_day_mean,
            "E_steady": E_steady,
            "mu_steady": mu_steady,
        })
        deterministic_trajs.append((label, t_grid_det, traj, E))
        print(f"  Corner {label}  T(D)={T_end:.3f}  E_bar={E_steady:.3f}  "
              f"mu_bar={mu_steady:+.3f}   ({time.time()-t0:.2f}s)",
              flush=True)
    print(flush=True)

    # ===== Stochastic 32-trajectory ensemble per corner (vmap'd) =====
    print(f"Stochastic ensemble ({N_STOCH_PER_CORNER} trajectories per corner) ...",
          flush=True)
    rng_root = jax.random.PRNGKey(42)
    rngs_per_corner = jax.random.split(rng_root, 8 * N_STOCH_PER_CORNER).reshape(
        8, N_STOCH_PER_CORNER, -1
    )
    for i, (label, V_h, V_n, V_c, _) in enumerate(CORNERS):
        t0 = time.time()
        traj_ens = np.asarray(stoch_solve(V_h, V_n, V_c, INIT_STATE,
                                           rngs_per_corner[i]))
        # shape (N_STOCH, n_pts, 4)
        T_ends = traj_ens[:, -1, 3]
        rows[i]["T_end_stoch_mean"] = float(T_ends.mean())
        rows[i]["T_end_stoch_std"] = float(T_ends.std())
        print(f"  {label}: stoch T_end {T_ends.mean():.3f} ± {T_ends.std():.3f}  "
              f"({time.time()-t0:.2f}s)", flush=True)
    print(flush=True)

    # ===== Initial-condition sweep at corner 4 (healthy) =====
    print(f"Initial-condition sweep at healthy corner ({N_STOCH_PER_CORNER} ICs) ...",
          flush=True)
    V_h, V_n, V_c = V_H_HIGH, V_N_LOW, V_C_LOW

    # Generate random ICs
    rng_ic = jax.random.PRNGKey(99)
    keys = jax.random.split(rng_ic, N_STOCH_PER_CORNER)
    W0_arr = jax.random.uniform(keys[0], shape=(N_STOCH_PER_CORNER,),
                                  minval=0.0, maxval=1.0)
    Z0_arr = jax.random.uniform(keys[1], shape=(N_STOCH_PER_CORNER,),
                                  minval=0.0, maxval=6.0)
    a0_arr = jax.random.uniform(keys[2], shape=(N_STOCH_PER_CORNER,),
                                  minval=0.0, maxval=1.0)
    T0_arr = jax.random.uniform(keys[3], shape=(N_STOCH_PER_CORNER,),
                                  minval=0.0, maxval=1.0)
    init_states = jnp.stack([W0_arr, Z0_arr, a0_arr, T0_arr], axis=-1)  # (N, 4)

    # vmap over init_state
    @jax.jit
    def det_solve_at_healthy(init_state):
        return det_solve(V_h, V_n, V_c, init_state)

    t0 = time.time()
    ic_trajs = jax.vmap(det_solve_at_healthy)(init_states)        # (N, n_pts, 4)
    ic_trajs = np.asarray(ic_trajs)
    print(f"  IC sweep: {time.time()-t0:.2f}s", flush=True)
    ic_final_T = ic_trajs[:, -1, 3]
    print(f"  Final T values: min={ic_final_T.min():.3f}, "
          f"max={ic_final_T.max():.3f}, mean={ic_final_T.mean():.3f}, "
          f"std={ic_final_T.std():.3f}", flush=True)
    print(flush=True)

    # ===== Plots =====
    print("Generating plots ...", flush=True)

    # 1. Bar chart of T_end across corners
    fig, ax = plt.subplots(figsize=(13, 6))
    labels_short = [r["label"] for r in rows]
    x = np.arange(8)
    det_T = [r["T_end_det"] for r in rows]
    stoch_mean = [r["T_end_stoch_mean"] for r in rows]
    stoch_std = [r["T_end_stoch_std"] for r in rows]
    bars1 = ax.bar(x - 0.2, det_T, 0.4, color="steelblue",
                    label="Deterministic T(D)")
    ax.bar(x + 0.2, stoch_mean, 0.4, yerr=stoch_std, color="crimson", alpha=0.7,
            capsize=4, label=f"Stochastic mean +/- std (N={N_STOCH_PER_CORNER})")
    ax.axhline(0.0, color="black", lw=0.4)
    ax.axhline(0.3, color="red", ls="--", alpha=0.6,
                label="acceptance: T < 0.3 (pathological)")
    ax.axhline(0.95, color="green", ls="--", alpha=0.6,
                label="acceptance: T > 0.95 (healthy det)")
    bars1[4].set_edgecolor("green"); bars1[4].set_linewidth(3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=30, fontsize=8, ha="right")
    ax.set_ylabel(f"T at terminal (D={D_DAYS:.0f}d)")
    ax.set_title("Option D — 8-corner T(D) — only corner 4 (healthy) reaches T*")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "corner_T_end.png", dpi=130)
    plt.close(fig)

    # 2. T(t) trajectories
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.tab10
    for i, (label, t, traj, E) in enumerate(deterministic_trajs):
        color = cmap(i)
        ls = "-" if i == 4 else "--"
        lw = 2.0 if i == 4 else 1.0
        ax.plot(np.asarray(t), traj[:, 3], color=color, ls=ls, lw=lw, label=label)
    ax.axhline(1.0, color="green", ls=":", alpha=0.5,
                label="T*(E=1) = sqrt((mu_0+mu_E)/eta) = 1.0")
    ax.axhline(0.0, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("T (testosterone amplitude)")
    ax.set_title("T(t) over 60 days for all 8 corners — only healthy reaches T*=1")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "T_trajectories.png", dpi=130)
    plt.close(fig)

    # 3. Phase plane (E_dyn, T)
    fig, ax = plt.subplots(figsize=(11, 8))
    E_crit = -p["mu_0"] / p["mu_E"]
    for i, (label, t, traj, E) in enumerate(deterministic_trajs):
        color = cmap(i)
        ls = "-" if i == 4 else "--"
        lw = 2.0 if i == 4 else 1.0
        ax.plot(E, traj[:, 3], color=color, ls=ls, lw=lw, label=label, alpha=0.8)
        ax.scatter([E[0]], [traj[0, 3]], color=color, marker="o", s=60,
                    edgecolor="black", zorder=5)
        ax.scatter([E[-1]], [traj[-1, 3]], color=color, marker="s", s=80,
                    edgecolor="black", zorder=5)
    ax.axvline(E_crit, color="red", ls=":", lw=2, alpha=0.6,
                label=f"E_crit = {E_crit:.2f}")
    ax.set_xlabel("E_dyn (entrainment quality)")
    ax.set_ylabel("T")
    ax.set_title("(E_dyn, T) phase plane — circles=initial, squares=final\n"
                  "Healthy ends at (E~1, T~1); all others collapse to T=0")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase_plane.png", dpi=130)
    plt.close(fig)

    # 4. IC sweep
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap_v = plt.cm.viridis(np.linspace(0, 1, N_STOCH_PER_CORNER))
    t_axis = np.linspace(0, D_DAYS, ic_trajs.shape[1])
    for k in range(N_STOCH_PER_CORNER):
        ax.plot(t_axis, ic_trajs[k, :, 3], color=cmap_v[k], lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="T*=1.0")
    ax.set_xlabel("t (days)")
    ax.set_ylabel("T")
    ax.set_title(
        f"32 random initial conditions at healthy corner — all converge to T* "
        f"(final T range: {ic_final_T.min():.3f} – {ic_final_T.max():.3f})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "init_cond_sweep.png", dpi=130)
    plt.close(fig)

    # ===== Summary JSON =====
    summary = {
        "n_corners": 8,
        "D_days": D_DAYS,
        "pinned": {"tau_T": float(p["tau_T"]),
                    "lambda_amp_Z": float(p["lambda_amp_Z"])},
        "free_params": {
            "lambda_amp_W": float(p["lambda_amp_W"]),
            "V_n_scale": float(p["V_n_scale"]),
            "c_tilde": float(p["c_tilde"]),
        },
        "E_crit": float(E_crit),
        "T_star_full_entrainment": float(np.sqrt((p["mu_0"] + p["mu_E"]) / p["eta"])),
        "corners": rows,
        "ic_sweep_at_healthy": {
            "n_ics": int(N_STOCH_PER_CORNER),
            "final_T_min": float(ic_final_T.min()),
            "final_T_max": float(ic_final_T.max()),
            "final_T_mean": float(ic_final_T.mean()),
            "final_T_std": float(ic_final_T.std()),
        },
    }
    (OUT / "corner_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote: {OUT / 'corner_T_end.png'}", flush=True)
    print(f"Wrote: {OUT / 'T_trajectories.png'}", flush=True)
    print(f"Wrote: {OUT / 'phase_plane.png'}", flush=True)
    print(f"Wrote: {OUT / 'init_cond_sweep.png'}", flush=True)
    print(f"Wrote: {OUT / 'corner_summary.json'}", flush=True)
    print(flush=True)

    # ===== Acceptance =====
    print("=" * 72, flush=True)
    print("Acceptance criteria", flush=True)
    print("=" * 72, flush=True)
    healthy = rows[4]
    healthy_ok_det = healthy["T_end_det"] > 0.95
    healthy_ok_stoch = healthy["T_end_stoch_mean"] > 0.85
    print(f"  Corner 4 (healthy):", flush=True)
    print(f"    deterministic T(D) = {healthy['T_end_det']:.3f}  "
          f"{'PASS' if healthy_ok_det else 'FAIL'} (target > 0.95)", flush=True)
    print(f"    stochastic   T(D) = {healthy['T_end_stoch_mean']:.3f}  "
          f"{'PASS' if healthy_ok_stoch else 'FAIL'} (target > 0.85)", flush=True)
    pathological_ok = True
    print(f"  Pathological corners (target det T(D) < 0.30):", flush=True)
    for i in [0, 1, 2, 3, 5, 6, 7]:
        T = rows[i]["T_end_det"]
        ok = T < 0.30
        pathological_ok &= ok
        print(f"    Corner {rows[i]['label']}: T = {T:.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(flush=True)
    ic_ok = ic_final_T.min() > 0.85
    print(f"  IC sweep at healthy: all 32 ICs converge to T > 0.85  "
          f"{'PASS' if ic_ok else 'FAIL'}  (got min = {ic_final_T.min():.3f})",
          flush=True)
    print(flush=True)
    overall = healthy_ok_det and healthy_ok_stoch and pathological_ok and ic_ok
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
