"""Shared fixtures for SWAT gating tests.

Since the V_h-anabolic structural fix landed upstream (PR #11), the
"vendored" and "option-c" variants now point at the same model — there
is only one SWAT.  The --variant flag is preserved for backward
compatibility but is a no-op: both values return the corrected model.

Test selection:
  pytest tests/swat/                            # runs the SWAT gating suite
  pytest --variant option-c tests/swat/         # equivalent (legacy flag)
  pytest --variant option-c --lambda-base 4 tests/swat/   # override forcing
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


def _build_model(lambda_overrides) -> ModelInterface:
    """Construct the SWAT model, with optional lambda_amp overrides."""
    from model_validation.models.swat.option_c_dynamics import option_c_model
    lam = lambda_overrides["lambda"]
    lam_Z = lambda_overrides["lambda_Z"]
    if lam is None and lam_Z is None:
        return vendored_model()
    return option_c_model(lambda_base=lam, lambda_Z_base=lam_Z)


@pytest.fixture(scope="session")
def model(variant, lambda_overrides) -> ModelInterface:
    """The model under test, with noise temperatures forced to zero.

    --variant is a legacy flag — both 'vendored' and 'option-c' return
    the same corrected SWAT model after upstream PR #11.
    """
    del variant   # both variants point at the same model now
    return with_noise_off(_build_model(lambda_overrides))


@pytest.fixture(scope="session")
def model_with_noise(variant, lambda_overrides) -> ModelInterface:
    """The model under test, WITH noise (for stochastic tests).

    --variant is a legacy flag — both values return the same corrected
    SWAT model after upstream PR #11.
    """
    del variant
    return _build_model(lambda_overrides)
