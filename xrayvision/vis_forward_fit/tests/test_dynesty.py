"""Nested sampling, weighted posterior results, and evidence validation."""

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_dynesty
from xrayvision.vis_forward_fit.forward_fit import sources_to_vis
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities

pytest.importorskip("dynesty")
FLUX_UNIT = apu.ph / (apu.s * apu.cm**2 * apu.keV)


def make_problem(shape="circular"):
    # At zero spatial frequency both models predict their total flux. Other
    # parameters integrate out under their normalised, physically valid priors,
    # leaving an independently integrable one-dimensional evidence problem.
    if shape == "circular":
        source = Source("circular", 2 * FLUX_UNIT, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec)
        limits = [(0.5, 3.5), (-3, 3), (-3, 3), (0.5, 4)]
    else:
        source = Source(
            "elliptical",
            2 * FLUX_UNIT,
            0 * apu.arcsec,
            0 * apu.arcsec,
            2 * apu.arcsec,
            3 * apu.arcsec,
            0 * apu.rad,
        )
        limits = [(0.5, 3.5), (-3, 3), (-3, 3), (0.5, 4), (0.5, 4), (-np.pi / 2, np.pi / 2)]
    sources = SourceList([source])
    values = np.array([2.05 + 0.04j, 1.91 - 0.07j, 2.12 + 0.03j, 1.96 - 0.02j])
    uv = np.zeros(len(values)) / apu.arcsec
    vis = Visibilities(values * FLUX_UNIT, uv, uv, uncertainty=np.full(len(values), 0.2 + 0.3j) * FLUX_UNIT)
    priors = {
        name: ParameterPrior(uniform(lower, upper - lower), unit)
        for name, unit, (lower, upper) in zip(sources.parameter_names, sources.parameter_units(FLUX_UNIT), limits)
    }
    return BayesianVisibilityProblem(vis, sources, priors), values


@pytest.fixture(scope="module")
def evidence_runs():
    return [
        sample_dynesty(make_problem(shape)[0], nlive=100, dlogz=0.05, seed=42 + i)
        for i, shape in enumerate(("circular", "elliptical"))
    ]


def test_evidence_and_weighted_posterior_against_independent_integration(evidence_runs):
    _, values = make_problem()

    def integrand(flux):
        return np.exp(np.sum(norm.logpdf(values.real, flux, 0.2) + norm.logpdf(values.imag, 0, 0.3))) / 3

    evidence = quad(integrand, 0.5, 3.5)[0]
    mean = quad(lambda flux: flux * integrand(flux), 0.5, 3.5)[0] / evidence
    variance = quad(lambda flux: (flux - mean) ** 2 * integrand(flux), 0.5, 3.5)[0] / evidence
    for result in evidence_runs:
        assert np.isfinite(result.log_evidence_error)
        assert result.log_evidence_error > 0
        assert abs(result.log_evidence - np.log(evidence)) < 5 * result.log_evidence_error
        assert_allclose(logsumexp(result.log_weights), result.log_evidence)
        assert_allclose(result.weights.sum(), 1)
        flux = result.parameter_samples("source[0].amp")
        assert flux.unit == FLUX_UNIT
        recovered_mean = np.average(flux.value, weights=result.weights)
        recovered_variance = np.average((flux.value - recovered_mean) ** 2, weights=result.weights)
        assert abs(recovered_mean - mean) < 0.3 * np.sqrt(variance)
        assert_allclose(recovered_variance, variance, rtol=0.3)
        assert result.samples.shape == (len(result.weights), result.problem.ndim)
        assert result.log_likelihood.shape == result.weights.shape
        assert result.parameter_names == result.problem.parameter_names
        assert result.parameter_units == result.problem.parameter_units
        reconstructed = result.sources_from_sample(-1)
        assert_allclose(
            sources_to_vis(reconstructed, result.problem.fit_data.u, result.problem.fit_data.v),
            result.problem.predict(result.samples[-1]),
        )
        assert_allclose(result.log_likelihood[-1], result.problem.log_likelihood(result.samples[-1]))
        with pytest.raises(KeyError, match="Unknown parameter"):
            result.parameter_samples("missing")
    # These data do not constrain morphology. Additional normalised shape
    # priors must integrate to one, giving equal evidences for these models.
    first, second = evidence_runs
    uncertainty = np.hypot(first.log_evidence_error, second.log_evidence_error)
    assert abs(first.log_evidence - second.log_evidence) < 5 * uncertainty


def test_reproducible_without_mutating_global_rng():
    problem, _ = make_problem()
    before = np.random.get_state()
    first = sample_dynesty(problem, nlive=20, dlogz=0.5, sample="rwalk", seed=7)
    second = sample_dynesty(problem, nlive=20, dlogz=0.5, sample="rwalk", seed=7)
    different = sample_dynesty(problem, nlive=20, dlogz=0.5, sample="rwalk", seed=8)
    after = np.random.get_state()
    assert_array_equal(first.samples, second.samples)
    assert_array_equal(first.log_weights, second.log_weights)
    assert first.log_evidence != different.log_evidence
    assert first.diagnostics["sample"] == "rwalk"
    for old, new in zip(before, after):
        assert_array_equal(old, new)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nlive": 8},
        {"nlive": 0},
        {"nlive": -1},
        {"dlogz": 0},
        {"dlogz": -1},
        {"dlogz": np.nan},
        {"dlogz": np.inf},
        {"maxcall": 0},
    ],
)
def test_invalid_settings(kwargs):
    problem, _ = make_problem()
    with pytest.raises(ValueError):
        sample_dynesty(problem, **kwargs)


def test_call_limit_retains_warning_and_settings():
    problem, _ = make_problem()
    with pytest.warns(UserWarning, match="maxiter/maxcall"):
        result = sample_dynesty(problem, nlive=20, dlogz=0.001, maxcall=1, seed=9)
    assert result.diagnostics["maxcall"] == 1
    assert result.diagnostics["ncall"] >= 1
    assert_allclose(result.weights.sum(), 1)


def test_missing_optional_dependency(monkeypatch):
    problem, _ = make_problem()
    monkeypatch.setitem(sys.modules, "dynesty", None)
    with pytest.raises(ImportError, match=r"xrayvisim\[bayes\]"):
        sample_dynesty(problem)
