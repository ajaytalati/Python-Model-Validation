"""Shared fixtures for SWAT gating tests.

Two model variants are exposed:
  - `vendored`: the current vendored snapshot (pre-fix, expected to fail
    several tests — the regression target for issue #4).
  - `option_c`: the proposed fix (refined Option C with redefined amp_W
    and a configurable forcing-amplitude baseline).

Test selection:
  pytest tests/swat/                          # runs against vendored snapshot
  pytest --variant option-c tests/swat/        # runs against Option C variant
  pytest --variant option-c --lambda-base 4 tests/swat/   # at smaller-lambda regime
"""
from __future__ import annotations
import pytest

from model_validation.models.swat import vendored_model
from model_validation.runner import with_noise_off, ModelInterface


def pytest_addoption(parser):
    parser.addoption(
        "--variant", default="vendored", choices=["vendored", "option-c"],
        help="Which model variant to test: vendored snapshot, or refined Option C.",
    )
    parser.addoption(
        "--lambda-base", default=None, type=float,
        help="Override the lambda baseline (only meaningful for option-c).",
    )
    parser.addoption(
        "--lambda-z-base", default=None, type=float,
        help="Override the lambda_Z baseline (only meaningful for option-c).",
    )


@pytest.fixture(scope="session")
def variant(request):
    return request.config.getoption("--variant")


@pytest.fixture(scope="session")
def lambda_overrides(request):
    return {
        "lambda": request.config.getoption("--lambda-base"),
        "lambda_Z": request.config.getoption("--lambda-z-base"),
    }


@pytest.fixture(scope="session")
def model(variant, lambda_overrides) -> ModelInterface:
    """The model under test, with noise temperatures forced to zero."""
    if variant == "vendored":
        m = vendored_model()
    elif variant == "option-c":
        from model_validation.models.swat.option_c_dynamics import (
            option_c_model,
        )
        lam = lambda_overrides["lambda"]
        lam_Z = lambda_overrides["lambda_Z"]
        m = option_c_model(lambda_base=lam, lambda_Z_base=lam_Z)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
    return with_noise_off(m)


@pytest.fixture(scope="session")
def model_with_noise(variant, lambda_overrides) -> ModelInterface:
    """The model under test, WITH noise (used for stochastic tests like sleep fraction)."""
    if variant == "vendored":
        return vendored_model()
    elif variant == "option-c":
        from model_validation.models.swat.option_c_dynamics import (
            option_c_model,
        )
        lam = lambda_overrides["lambda"]
        lam_Z = lambda_overrides["lambda_Z"]
        return option_c_model(lambda_base=lam, lambda_Z_base=lam_Z)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
