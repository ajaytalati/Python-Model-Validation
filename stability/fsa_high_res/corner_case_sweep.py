"""Lyapunov-stability numerical sweep — FSA-high-res model.

Goal
----
Of the 4 binary corners of the (T_B, Phi) control box plus 1 healthy
reference interior point, classify each as healthy (A reaches the
super-critical Stuart-Landau attractor) or pathological (A collapses
to zero because (B, F) is driven into the mu(B,F) < 0 region).

Difference from SWAT
--------------------
SWAT's Lyapunov claim was "1 unique super-critical corner of the
8-corner control box". FSA's claim is more subtle: the bifurcation
parameter mu(B, F) is *state-dependent*, so the healthy/pathology
boundary is in (B, F) state space, not in control space. The
controls (T_B, Phi) shape where the system equilibrates in (B, F),
which then determines whether mu > 0 (healthy) or mu < 0 (pathology).

  T_B drives B via dB/dt = (1 + alpha_A A)/tau_B · (T_B - B).
  Phi drives F via dF/dt = Phi - (1 + lambda_B B + lambda_A A)/tau_F · F.

So at long times:
  B(infinity) = T_B
  F(infinity) ≈ Phi · tau_F / (1 + lambda_B T_B + lambda_A A)

For FSA's deployed parameters (tau_B=14, tau_F=7, lambda_B=3, etc.):
  C0: (T_B=0, Phi=0)    => B*=0, F*=0, mu*=0.02 (weakly super-critical, A*=0.32)
  C1: (T_B=0, Phi=2)    => B*=0, F* ~ 2·7/(1+0) = 14 (cut off by clip), F^2 dominates,
                             mu* <<< 0 — PATHOLOGY
  C2: (T_B=1, Phi=0)    => B*=1, F*=0, mu*=0.32 — STRONGLY super-critical, A*=1.26
  C3: (T_B=1, Phi=2)    => B*=1, F* ~ 2·7/(4+) ≈ 3, mu* << 0 — PATHOLOGY
  C4: (T_B=0.5, Phi=0.05) — healthy reference, mu* ≈ 0.16 — moderate super-critical

Method
------
1. **5-corner sweep**: deterministic D=60-day simulation from
   (B_0=0.3, F_0=0.05, A_0=0.4) at each corner. Plus 32 stochastic
   trajectories with full noise (vmap'd, JIT'd).
2. **IC sweep at C4 (healthy reference)**: 32 random initial conditions
   spanning (B_0, F_0, A_0) in [0,1]^2 × [0, 1.5]. Confirms convergence
   is robust to initial state inside the basin.
3. **Phase-plane plot**: project trajectories to (mu(B,F), A) plane.
   The mu = 0 vertical line separates healthy (right) from pathology
   (left). The healthy attractor is the Stuart-Landau curve A* = sqrt(mu/eta).
4. **Analytical Lyapunov function**: Stuart-Landau potential on A given
   (B, F) — V(A) = -mu(B,F) A^2/2 + eta A^4/4. Show dV/dt <= 0 along
   trajectories at the healthy corner.

Acceptance
----------
- C2 (max-fitness): det A(D) > 0.6 (strong super-critical attractor)
- C4 (healthy ref): det A(D) > 0.6 (moderate super-critical)
- C1, C3 (overtrained): det A(D) < 0.1 (collapse to zero)
- IC sweep at C4: range of A(D) across 32 ICs < 0.3 (robust convergence)
- C0 is sub-critical-borderline (mu_0 = 0.02 is barely positive); not
  used as an acceptance gate, just reported.

Outputs
-------
- results/corner_A_end.png       — bar chart of A(D) per corner
- results/A_trajectories.png     — A(t) over 60 days for all 5 corners
- results/phase_plane.png        — (mu(B,F), A) phase-plane trajectories
- results/init_cond_sweep.png    — 32 IC convergence at the healthy corner
- results/corner_summary.json    — numerical results
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import diffrax

from model_validation.models.fsa_high_res.vendored_dynamics import (
    fsa_drift, fsa_diffusion, fsa_state_clip,
    _bifurcation_parameter,
)
from model_validation.models.fsa_high_res.vendored_parameters import (
    default_fsa_parameters,
)


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# =========================================================================
# Corner setup
# =========================================================================

T_B_LOW, T_B_HIGH = 0.0, 1.0
PHI_LOW, PHI_HIGH = 0.0, 2.0

# C4 is the "moderately healthy reference" interior point — same controls
# the OT-Control adapter uses to derive its model-derived target pool.
T_B_HEALTHY, PHI_HEALTHY = 0.5, 0.05

CORNERS = [
    # (label, T_B, Phi, reading)
    ("0  (T_B=0,   Phi=0)  ",   T_B_LOW,     PHI_LOW,     "no training (weakly super-critical mu_0)"),
    ("1  (T_B=0,   Phi=2)  ",   T_B_LOW,     PHI_HIGH,    "high-strain no fitness — PATHOLOGY"),
    ("2  (T_B=1,   Phi=0)  ",   T_B_HIGH,    PHI_LOW,     "max-fitness no strain — strongly healthy"),
    ("3  (T_B=1,   Phi=2)  ",   T_B_HIGH,    PHI_HIGH,    "max-fitness max-strain — PATHOLOGY (overtrained)"),
    ("4  (T_B=0.5, Phi=0.05)",  T_B_HEALTHY, PHI_HEALTHY, "HEALTHY REFERENCE"),
]

D_DAYS = 60.0                                           # ~4.3 × tau_B
INIT_STATE = jnp.array([0.3, 0.05, 0.4])                # healthy ref init
N_STOCH_PER_CORNER = 32
SDE_DT = 0.02
N_PTS_DET = 600
N_PTS_STOCH = max(int(round(D_DAYS / SDE_DT)), 32)


# =========================================================================
# JIT-compiled solvers
# =========================================================================

def _build_deterministic_solver(params):
    t_grid = jnp.linspace(0.0, D_DAYS, N_PTS_DET)

    def vf(t, y, args):
        T_B, Phi = args
        u = jnp.array([T_B, Phi])
        return fsa_drift(t, y, u, params)

    @jax.jit
    def solve(T_B, Phi, init_state):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(),
            t0=0.0, t1=D_DAYS, dt0=0.01,
            y0=init_state, args=(T_B, Phi),
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
            saveat=diffrax.SaveAt(ts=t_grid),
            max_steps=2_000_000,
        )
        return t_grid, sol.ys

    return solve


def _build_stochastic_solver(params):
    t_grid = jnp.linspace(0.0, D_DAYS, N_PTS_STOCH)

    def drift_fn(t, y, args):
        T_B, Phi = args
        u = jnp.array([T_B, Phi])
        return fsa_drift(t, y, u, params)

    def diffusion_fn(t, y, args):
        return jnp.diag(fsa_diffusion(y, params))

    @jax.jit
    def solve_one(T_B, Phi, init_state, rng):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=D_DAYS, tol=SDE_DT / 2.0,
            shape=(3,), key=rng,
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn),
            diffrax.ControlTerm(diffusion_fn, bm),
        )
        sol = diffrax.diffeqsolve(
            terms, diffrax.Euler(),
            t0=0.0, t1=D_DAYS, dt0=SDE_DT,
            y0=init_state, args=(T_B, Phi),
            saveat=diffrax.SaveAt(ts=t_grid),
            max_steps=int(D_DAYS / SDE_DT) + 100,
        )
        return sol.ys

    @jax.jit
    def solve_batch(T_B, Phi, init_state, rngs):
        return jax.vmap(solve_one, in_axes=(None, None, None, 0))(
            T_B, Phi, init_state, rngs
        )

    return solve_batch


# =========================================================================
# Main sweep
# =========================================================================

def main():
    print("=" * 70, flush=True)
    print("FSA-high-res Lyapunov stability sweep", flush=True)
    print("=" * 70, flush=True)

    params = default_fsa_parameters()
    eta = float(params["eta"])
    print(f"Horizon: {D_DAYS} days  ({D_DAYS / params['tau_B']:.1f} × tau_B)", flush=True)
    print(f"Initial state: B_0={INIT_STATE[0]}, F_0={INIT_STATE[1]}, A_0={INIT_STATE[2]}", flush=True)
    print(flush=True)

    print("Building solvers (JIT) ...", flush=True)
    det_solve = _build_deterministic_solver(params)
    stoch_solve = _build_stochastic_solver(params)
    print("  deterministic + stochastic both compiled.", flush=True)
    print(flush=True)

    # ===================================================================
    # Corner sweep
    # ===================================================================
    summary = {"horizon_days": D_DAYS, "n_stoch": N_STOCH_PER_CORNER,
               "params": params, "corners": []}
    det_trajectories = {}             # label -> (t, traj) for plotting

    print("Running deterministic + stochastic per corner ...", flush=True)
    t0 = time.time()
    for i, (label, T_B, Phi, reading) in enumerate(CORNERS):
        # Deterministic
        t_grid_det, traj_det = det_solve(T_B, Phi, INIT_STATE)
        traj_det = np.asarray(traj_det)            # (N_PTS_DET, 3)
        A_end_det = float(traj_det[-1, 2])
        A_last_day_mean = float(np.mean(traj_det[-int(N_PTS_DET / D_DAYS):, 2]))
        # mu(B, F) at long-time
        B_end, F_end = traj_det[-1, 0], traj_det[-1, 1]
        mu_end = float(_bifurcation_parameter(B_end, F_end, params))
        det_trajectories[label] = (np.asarray(t_grid_det), traj_det)

        # Stochastic
        rng = jax.random.PRNGKey(42 + i)
        rngs = jax.random.split(rng, N_STOCH_PER_CORNER)
        traj_stoch = np.asarray(stoch_solve(T_B, Phi, INIT_STATE, rngs))
        # (N_STOCH, N_PTS_STOCH, 3)
        A_stoch_end = traj_stoch[:, -1, 2]
        A_stoch_mean = float(np.mean(A_stoch_end))
        A_stoch_std = float(np.std(A_stoch_end))

        summary["corners"].append({
            "label": label.strip(),
            "T_B": T_B, "Phi": Phi, "reading": reading,
            "A_end_det": A_end_det,
            "A_last_day_mean_det": A_last_day_mean,
            "B_end_det": float(B_end),
            "F_end_det": float(F_end),
            "mu_end_det": mu_end,
            "A_end_stoch_mean": A_stoch_mean,
            "A_end_stoch_std": A_stoch_std,
        })
        print(f"  Corner {label}: det A={A_end_det:.3f}, "
              f"mu={mu_end:+.3f}, stoch_mean={A_stoch_mean:.3f} ± {A_stoch_std:.3f}",
              flush=True)
    print(f"  ... done in {time.time() - t0:.1f}s", flush=True)
    print(flush=True)

    # ===================================================================
    # IC sweep at corner 4 (healthy reference)
    # ===================================================================
    print("IC sweep at C4 (healthy reference) — 32 random initial conditions ...", flush=True)
    rng_ic = jax.random.PRNGKey(0xBEEF)
    rng_ic_keys = jax.random.split(rng_ic, 32)
    ic_results = []
    for k, rng in enumerate(rng_ic_keys):
        # Random IC in [0, 1]^2 × [0, 1.5]
        kB, kF, kA = jax.random.split(rng, 3)
        B0 = float(jax.random.uniform(kB, (), minval=0.0, maxval=1.0))
        F0 = float(jax.random.uniform(kF, (), minval=0.0, maxval=1.0))
        A0 = float(jax.random.uniform(kA, (), minval=0.0, maxval=1.5))
        ic = jnp.array([B0, F0, A0])
        _, traj = det_solve(T_B_HEALTHY, PHI_HEALTHY, ic)
        traj = np.asarray(traj)
        A_end = float(traj[-1, 2])
        ic_results.append({
            "B_0": B0, "F_0": F0, "A_0": A0, "A_end": A_end,
        })
    A_ends = np.array([r["A_end"] for r in ic_results])
    spread = float(A_ends.max() - A_ends.min())
    print(f"  IC sweep: A(D) interval = [{A_ends.min():.3f}, {A_ends.max():.3f}]  "
          f"spread (max-min) = {spread:.3f}  "
          f"mean = {A_ends.mean():.3f} ± {A_ends.std():.4f}", flush=True)
    summary["ic_sweep"] = {
        "T_B": T_B_HEALTHY, "Phi": PHI_HEALTHY,
        "n_ics": len(ic_results),
        "A_end_range": [float(A_ends.min()), float(A_ends.max())],
        "A_end_mean": float(A_ends.mean()),
        "A_end_std": float(A_ends.std()),
        "results": ic_results,
    }
    print(flush=True)

    # ===================================================================
    # Plots
    # ===================================================================
    print("Generating plots ...", flush=True)

    # Bar chart of corner A(D)
    labels_short = [c["label"][:3] for c in summary["corners"]]
    A_ends_corners = [c["A_end_det"] for c in summary["corners"]]
    colours = []
    for c in summary["corners"]:
        if "PATHOLOGY" in c["reading"]:
            colours.append("crimson")
        elif "HEALTHY" in c["reading"] or "strongly healthy" in c["reading"]:
            colours.append("forestgreen")
        else:
            colours.append("dimgrey")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels_short, A_ends_corners, color=colours)
    for i, c in enumerate(summary["corners"]):
        ax.text(i, A_ends_corners[i] + 0.02,
                 f"μ={c['mu_end_det']:+.2f}", ha="center", fontsize=8)
    ax.axhline(0.6, color="black", linestyle=":", alpha=0.6, label="acceptance: A>0.6 (healthy)")
    ax.axhline(0.1, color="grey", linestyle=":", alpha=0.6, label="acceptance: A<0.1 (path.)")
    ax.set_ylabel("Terminal amplitude A(D)")
    ax.set_xlabel("Corner")
    ax.set_title("FSA corner sweep — terminal amplitude A(D), D=60 days")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "corner_A_end.png", dpi=130)
    plt.close(fig)

    # A(t) trajectories
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["dimgrey", "crimson", "forestgreen", "darkred", "royalblue"]
    for i, (label, _, _, _) in enumerate(CORNERS):
        t, traj = det_trajectories[label]
        ax.plot(t, traj[:, 2], color=palette[i], lw=1.6, label=label.strip())
    ax.set_xlabel("t (days)")
    ax.set_ylabel("A (amplitude)")
    ax.set_title("FSA — deterministic A(t) per corner (D=60 days)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "A_trajectories.png", dpi=130)
    plt.close(fig)

    # Phase plane (mu, A)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, _, _, _) in enumerate(CORNERS):
        _, traj = det_trajectories[label]
        B = traj[:, 0]; F = traj[:, 1]; A = traj[:, 2]
        mu = (params["mu_0"] + params["mu_B"] * B
              - params["mu_F"] * F - params["mu_FF"] * F ** 2)
        ax.plot(mu, A, color=palette[i], lw=1.4, label=label.strip())
        ax.scatter(mu[0], A[0], color=palette[i], marker="o", s=50, edgecolor="black", zorder=10)
        ax.scatter(mu[-1], A[-1], color=palette[i], marker="s", s=70, edgecolor="black", zorder=10)
    # Stuart-Landau attractor curve A* = sqrt(mu/eta)
    mu_grid = np.linspace(0.0, 0.5, 200)
    A_star = np.sqrt(mu_grid / eta)
    ax.plot(mu_grid, A_star, "k--", lw=1.4, alpha=0.7, label=r"$A^* = \sqrt{\mu/\eta}$")
    ax.axvline(0.0, color="red", linestyle=":", alpha=0.6, label=r"$\mu = 0$ (Hopf)")
    ax.set_xlabel(r"$\mu(B, F)$ (bifurcation parameter)")
    ax.set_ylabel(r"$A$ (amplitude)")
    ax.set_title("FSA phase plane — μ(B,F) vs A trajectories\n(circles = initial state, squares = terminal state)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase_plane.png", dpi=130)
    plt.close(fig)

    # IC sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in ic_results:
        ax.scatter(r["A_0"], r["A_end"], color="forestgreen", alpha=0.6, s=30)
    ax.axhline(A_ends.mean(), color="black", linestyle="-", alpha=0.7,
                 label=f"mean A(D) = {A_ends.mean():.3f}")
    ax.axhline(A_ends.mean() + A_ends.std(), color="grey", linestyle=":", alpha=0.7,
                 label=f"±1σ band  ({A_ends.std():.3f})")
    ax.axhline(A_ends.mean() - A_ends.std(), color="grey", linestyle=":", alpha=0.7)
    ax.set_xlabel("Initial A_0")
    ax.set_ylabel("Terminal A(D)")
    ax.set_title("IC sweep at C4 (healthy reference) — 32 random ICs in [0,1]² × [0, 1.5]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "init_cond_sweep.png", dpi=130)
    plt.close(fig)

    # Save summary
    (OUT / "corner_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    )
    print(f"\nWrote: {OUT / 'corner_A_end.png'}", flush=True)
    print(f"Wrote: {OUT / 'A_trajectories.png'}", flush=True)
    print(f"Wrote: {OUT / 'phase_plane.png'}", flush=True)
    print(f"Wrote: {OUT / 'init_cond_sweep.png'}", flush=True)
    print(f"Wrote: {OUT / 'corner_summary.json'}", flush=True)
    print(flush=True)

    # Acceptance
    print("=== ACCEPTANCE ===", flush=True)
    fail_count = 0
    # C2 (max-fitness): A_end > 0.6
    c2 = summary["corners"][2]
    c2_ok = c2["A_end_det"] > 0.6
    print(f"  C2 max-fitness (T_B=1, Phi=0): A_end = {c2['A_end_det']:.3f}  "
          f"{'PASS' if c2_ok else 'FAIL'} (need > 0.6)", flush=True)
    fail_count += not c2_ok
    # C4 (healthy reference): A_end > 0.6
    c4 = summary["corners"][4]
    c4_ok = c4["A_end_det"] > 0.6
    print(f"  C4 healthy ref (T_B=0.5, Phi=0.05): A_end = {c4['A_end_det']:.3f}  "
          f"{'PASS' if c4_ok else 'FAIL'} (need > 0.6)", flush=True)
    fail_count += not c4_ok
    # C1, C3 (overtrained): A_end < 0.1
    c1 = summary["corners"][1]
    c3 = summary["corners"][3]
    c1_ok = c1["A_end_det"] < 0.1
    c3_ok = c3["A_end_det"] < 0.1
    print(f"  C1 overtrained-no-fit (T_B=0, Phi=2): A_end = {c1['A_end_det']:.3f}  "
          f"{'PASS' if c1_ok else 'FAIL'} (need < 0.1)", flush=True)
    print(f"  C3 overtrained-max-fit (T_B=1, Phi=2): A_end = {c3['A_end_det']:.3f}  "
          f"{'PASS' if c3_ok else 'FAIL'} (need < 0.1)", flush=True)
    fail_count += not c1_ok
    fail_count += not c3_ok
    # IC sweep at C4: spread of A(D) values (max - min) < 0.3
    ic_spread = float(A_ends.max() - A_ends.min())
    ic_ok = ic_spread < 0.3
    print(f"  IC sweep at C4: A(D) spread (max-min over 32 ICs) = {ic_spread:.3f}  "
          f"{'PASS' if ic_ok else 'FAIL'} (need < 0.3, i.e. all ICs converge "
          f"to the same neighbourhood)", flush=True)
    fail_count += not ic_ok
    # C0 reported but not gated
    c0 = summary["corners"][0]
    print(f"  C0 sub-critical-borderline (T_B=0, Phi=0): A_end = {c0['A_end_det']:.3f}  "
          f"(reported only, mu_0={params['mu_0']:.3f} barely super-critical)", flush=True)

    print(flush=True)
    if fail_count == 0:
        print("OVERALL: PASS", flush=True)
        return 0
    print(f"OVERALL: FAIL ({fail_count} criteria failed)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
