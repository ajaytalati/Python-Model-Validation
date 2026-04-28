"""Does the model's amplitude obey the bifurcation it claims to obey?

Clinician's question being tested
---------------------------------
"When the model says my patient's bifurcation parameter is positive,
does the simulated endocrine amplitude actually settle at the
predicted Stuart-Landau equilibrium? When the bifurcation parameter
goes negative, does the amplitude actually collapse?"

This is the deepest of the clinical-validity tests. The model claims
that the patient's clinical state — fitness B and strain F — combine
into a single bifurcation parameter mu(B, F):

    mu(B, F)  >  0   =>   healthy attractor at A* = sqrt(mu / eta)
    mu(B, F)  <=  0  =>   collapse to A = 0

Everything the model says about "healthy" vs "overtrained" lives in
this single mu sign. So a clinical-validity gating suite must
explicitly check that the model's amplitude *actually does* land where
the bifurcation parameter says it lands.

What the tests below assert
---------------------------
1. **Super-critical regime matches the analytical fixed point**
   (test_super_critical_amplitude_matches_landau). At the healthy
   reference (T_B = 0.5, Phi = 0.05) the simulated A(D) must match
   sqrt(mu / eta) computed from the terminal (B, F) — within 1%.

2. **Sub-critical regime collapses to zero**
   (test_sub_critical_amplitude_collapses). At an overtrained
   prescription where mu < 0 at the long-time fixed point, the
   simulated A(D) must be essentially zero.

The first test catches "amplitude is healthy-shaped but not at the
predicted level" failures (mis-tuned eta). The second catches "model
doesn't actually collapse despite the bifurcation parameter saying
it should" failures.
"""
import math

from .conftest import bifurcation_at_state, terminal_state


def test_super_critical_amplitude_matches_landau(fsa_model, trajectory_solver):
    """Healthy reference: A(D) lands on sqrt(mu(B, F) / eta).

    Run the healthy reference (T_B = 0.5, Phi = 0.05) for 60 days from
    the healthy initial state. Read the terminal (B, F, A). Compute
    the analytical Stuart-Landau equilibrium A_star = sqrt(mu(B, F) / eta)
    using the model's deployed eta. Check A(D) is within 1% of A_star.
    """
    eta = float(fsa_model.params["eta"])
    B_end, F_end, A_end = terminal_state(trajectory_solver,
                                           T_B=0.5, Phi=0.05)
    mu = bifurcation_at_state(B_end, F_end, fsa_model.params)
    assert mu > 0.0, (
        f"Healthy reference should have super-critical mu. "
        f"Got mu = {mu:.4f} at (B_end, F_end) = ({B_end:.4f}, {F_end:.4f})."
    )
    A_star = math.sqrt(mu / eta)
    rel_error = abs(A_end - A_star) / max(A_star, 1e-9)
    assert rel_error < 0.01, (
        f"Terminal A should match Stuart-Landau equilibrium A* = sqrt(mu/eta). "
        f"Got A(D) = {A_end:.4f}, A* = {A_star:.4f}, "
        f"relative error = {rel_error:.2%}."
    )


def test_sub_critical_amplitude_collapses(fsa_model, trajectory_solver):
    """Overtrained prescription: mu drives negative and amplitude collapses.

    Run an explicitly overtrained prescription (T_B = 0.5, Phi = 2.0)
    for 60 days. Verify two things:
      - the long-time mu(B, F) is negative — i.e. the model genuinely
        believes the patient has been pushed past the cliff;
      - the simulated A(D) is effectively zero.
    """
    B_end, F_end, A_end = terminal_state(trajectory_solver,
                                           T_B=0.5, Phi=2.0)
    mu = bifurcation_at_state(B_end, F_end, fsa_model.params)
    assert mu < 0.0, (
        f"Overtrained prescription should drive mu negative. "
        f"Got mu = {mu:.4f} at (B_end, F_end) = ({B_end:.4f}, {F_end:.4f})."
    )
    assert A_end < 0.01, (
        f"With mu sub-critical, terminal A should collapse to zero. "
        f"Got A(D) = {A_end:.6f}."
    )
