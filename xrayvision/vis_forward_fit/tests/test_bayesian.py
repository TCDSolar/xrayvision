import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm, randint, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_vis
from xrayvision.vis_forward_fit.sources import InvalidSourceParameters, Source, SourceList
from xrayvision.visibility import Visibilities

PHOTON_FLUX_UNIT = apu.ph / (apu.s * apu.cm**2 * apu.keV)
PHOTON_FLUX_PER_EV_UNIT = apu.ph / (apu.s * apu.cm**2 * apu.eV)


def test_prior_density_normalises_after_unit_conversion():
    prior = ParameterPrior(uniform(loc=1, scale=2), apu.arcmin)
    integral, _ = quad(lambda x: np.exp(prior.logpdf(x, unit=apu.arcsec)), 60, 180)
    assert_allclose(integral, 1)
    assert_allclose(prior.logpdf(120, unit=apu.arcsec), -np.log(120))
    assert prior.logpdf(30, unit=apu.arcsec) == -np.inf


def test_emcee_is_optional(problem_inputs, monkeypatch):
    problem = BayesianVisibilityProblem(*problem_inputs)
    monkeypatch.setitem(sys.modules, "emcee", None)
    with pytest.raises(ImportError, match=r"xrayvisim\[bayes\]"):
        sample_emcee(problem, np.tile(problem.initial_params, (2 * problem.ndim, 1)))


@pytest.mark.parametrize("probability", [0.01, 0.5, 0.99])
def test_prior_inverse_cdf_converts_units(probability):
    distribution = norm(loc=30, scale=5)
    prior = ParameterPrior(distribution, apu.deg)
    value = prior.ppf(probability, unit=apu.rad)
    assert_allclose(distribution.cdf(np.rad2deg(value)), probability)


@pytest.mark.parametrize("angular_factor", [apu.dimensionless_unscaled, apu.arcsec**-2])
def test_equivalent_xray_priors_have_same_density_and_quantiles(angular_factor):
    per_kev_unit = PHOTON_FLUX_UNIT * angular_factor
    per_ev_unit = apu.ph / (apu.s * apu.cm**2 * apu.eV) * angular_factor
    per_kev = ParameterPrior(norm(loc=2, scale=0.1), per_kev_unit)
    per_ev = ParameterPrior(norm(loc=0.002, scale=0.0001), per_ev_unit)
    assert_allclose(per_kev.logpdf(2.05, unit=per_kev_unit), per_ev.logpdf(2.05, unit=per_kev_unit))
    assert_allclose(per_kev.ppf(0.9, unit=per_kev_unit), per_ev.ppf(0.9, unit=per_kev_unit))
    # Values per eV are 1/1000 as large, so their numeric probability density is 1000 times larger.
    assert_allclose(per_kev.logpdf(0.00205, unit=per_ev_unit), per_kev.logpdf(2.05, unit=per_kev_unit) + np.log(1000))


@pytest.mark.parametrize("method, value", [("logpdf", 1), ("ppf", 0.5)])
def test_prior_rejects_incompatible_units(method, value):
    prior = ParameterPrior(norm(), apu.arcsec)
    with pytest.raises(apu.UnitConversionError):
        getattr(prior, method)(value, unit=PHOTON_FLUX_UNIT)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_prior_nonfinite_parameters_have_zero_probability(value):
    assert ParameterPrior(norm(), PHOTON_FLUX_UNIT).logpdf(value, unit=PHOTON_FLUX_UNIT) == -np.inf


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf])
def test_prior_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="between zero and one"):
        ParameterPrior(norm(), PHOTON_FLUX_UNIT).ppf(probability, unit=PHOTON_FLUX_UNIT)


def test_prior_quantile_endpoints():
    prior = ParameterPrior(norm(), PHOTON_FLUX_UNIT)
    assert prior.ppf(0, unit=PHOTON_FLUX_UNIT) == -np.inf
    assert prior.ppf(1, unit=PHOTON_FLUX_UNIT) == np.inf


def test_prior_rejects_discrete_distribution():
    with pytest.raises(TypeError, match="scalar continuous distribution"):
        ParameterPrior(randint(0, 3), PHOTON_FLUX_UNIT)


def test_prior_rejects_invalid_distribution_parameters():
    with pytest.raises(ValueError, match="valid, non-empty scalar support"):
        ParameterPrior(norm(scale=-1), PHOTON_FLUX_UNIT)


@pytest.fixture(params=["circular", "elliptical", "loop", "mixed"])
def problem_inputs(request):
    models = {
        "circular": Source("circular", 2 * PHOTON_FLUX_UNIT, 1 * apu.arcsec, -2 * apu.arcsec, 2 * apu.arcsec),
        "elliptical": Source(
            "elliptical",
            1 * PHOTON_FLUX_UNIT,
            4 * apu.arcsec,
            3 * apu.arcsec,
            1 * apu.arcsec,
            3 * apu.arcsec,
            30 * apu.deg,
        ),
        "loop": Source(
            "loop",
            1.5 * PHOTON_FLUX_UNIT,
            -3 * apu.arcsec,
            1 * apu.arcsec,
            2 * apu.arcsec,
            5 * apu.arcsec,
            20 * apu.deg,
            40 * apu.deg,
        ),
    }
    sources = SourceList(list(models.values()) if request.param == "mixed" else [models[request.param]])
    rng = np.random.default_rng(42)
    u, v = rng.uniform(-0.12, 0.12, (2, 20)) / apu.arcsec
    obs = Visibilities(sources_to_vis(sources, u, v), u, v, uncertainty=np.full(20, 0.03 + 0.06j) * PHOTON_FLUX_UNIT)
    priors = {
        name: ParameterPrior(norm(loc=value, scale=1), unit)
        for name, unit, value in zip(
            sources.parameter_names,
            sources.parameter_units(PHOTON_FLUX_UNIT),
            sources.to_canonical(PHOTON_FLUX_UNIT).params,
        )
    }
    return obs, sources, priors


def test_problem_reuses_existing_predictions_and_quantity_sources(problem_inputs):
    obs, sources, priors = problem_inputs
    problem = BayesianVisibilityProblem(obs, sources, priors)
    assert problem.parameter_names == sources.parameter_names
    assert problem.parameter_units == sources.parameter_units(PHOTON_FLUX_UNIT)
    assert_allclose(problem.predict(problem.initial_params), obs.visibilities)
    reconstructed = problem.sources_from_params(problem.initial_params)
    assert_allclose(sources_to_vis(reconstructed, obs.u, obs.v), obs.visibilities)
    assert reconstructed[0].amp.unit == PHOTON_FLUX_UNIT
    assert reconstructed[0].x0.unit == apu.arcsec
    assert sources[0].amp.unit == PHOTON_FLUX_UNIT  # Constructing the problem leaves caller sources intact.


def test_problem_prior_and_likelihood_are_normalised_and_separate(problem_inputs):
    obs, sources, priors = problem_inputs
    problem = BayesianVisibilityProblem(obs, sources, priors)
    params = problem.initial_params
    expected_prior = -0.5 * problem.ndim * np.log(2 * np.pi)
    expected_likelihood = -20 * np.log(2 * np.pi * 0.03 * 0.06)
    assert_allclose(problem.log_prior(params), expected_prior)
    assert_allclose(problem.log_likelihood(params), expected_likelihood)
    assert_allclose(problem.log_posterior(params), expected_prior + expected_likelihood)

    # A likelihood can be evaluated even where the chosen prior has no support.
    other_priors = {**priors, "source[0].amp": ParameterPrior(uniform(loc=100, scale=10), PHOTON_FLUX_UNIT)}
    other = BayesianVisibilityProblem(obs, sources, other_priors)
    assert other.log_prior(params) == -np.inf
    assert other.log_posterior(params) == -np.inf
    assert_allclose(other.log_likelihood(params), expected_likelihood)


def test_problem_prior_transform_uses_parameter_order(problem_inputs):
    obs, sources, priors = problem_inputs
    problem = BayesianVisibilityProblem(obs, sources, dict(reversed(list(priors.items()))))
    assert_allclose(problem.prior_transform(np.full(problem.ndim, 0.5)), problem.initial_params)


def test_problem_does_not_apply_optimiser_bounds(problem_inputs):
    obs, sources, priors = problem_inputs
    problem = BayesianVisibilityProblem(obs, sources, priors)
    params = problem.initial_params.copy()
    params[1] += 100  # Well outside every source's initial position search window.
    assert params[1] > sources.bounds[1][1]
    assert np.isfinite(problem.log_prior(params))
    assert np.isfinite(problem.log_likelihood(params))


def test_problem_handles_equivalent_prior_and_observation_units(problem_inputs):
    obs, sources, priors = problem_inputs
    original = BayesianVisibilityProblem(obs, sources, priors)
    converted_obs = Visibilities(
        obs.visibilities.to(PHOTON_FLUX_PER_EV_UNIT),
        obs.u.to(apu.deg**-1),
        obs.v.to(apu.deg**-1),
        uncertainty=obs.uncertainty.to(PHOTON_FLUX_PER_EV_UNIT),
    )
    converted = BayesianVisibilityProblem(converted_obs, sources, priors)
    params = converted.initial_params
    n_sources = len(sources)
    assert_allclose(
        converted.predict(params).to_value(PHOTON_FLUX_UNIT), original.predict(original.initial_params).value
    )
    assert_allclose(converted.log_prior(params) - original.log_prior(original.initial_params), n_sources * np.log(1000))
    assert_allclose(
        converted.log_likelihood(params) - original.log_likelihood(original.initial_params), 40 * np.log(1000)
    )
    assert_allclose(converted.prior_transform(np.full(converted.ndim, 0.5)), params)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_problem_nonfinite_proposals_have_zero_probability(problem_inputs, invalid):
    problem = BayesianVisibilityProblem(*problem_inputs)
    params = problem.initial_params.copy()
    params[0] = invalid
    assert problem.log_prior(params) == -np.inf
    assert problem.log_likelihood(params) == -np.inf
    assert problem.log_posterior(params) == -np.inf


@pytest.mark.parametrize("index, value", [(0, -1), (3, 0), (3, -1)])
def test_problem_rejects_invalid_flux_and_widths(problem_inputs, index, value):
    problem = BayesianVisibilityProblem(*problem_inputs)
    params = problem.initial_params.copy()
    params[index] = value
    assert problem.log_likelihood(params) == -np.inf
    with pytest.raises(InvalidSourceParameters):
        problem.predict(params)


def test_problem_reports_configuration_errors(problem_inputs):
    obs, sources, priors = problem_inputs
    with pytest.raises(ValueError, match="At least one source"):
        BayesianVisibilityProblem(obs, SourceList(), {})
    with pytest.raises(ValueError, match=r"missing=.*unknown"):
        BayesianVisibilityProblem(obs, sources, {"typo": next(iter(priors.values()))})
    with pytest.raises(apu.UnitConversionError):
        BayesianVisibilityProblem(obs, sources, {**priors, "source[0].amp": ParameterPrior(norm(), apu.arcsec)})
    with pytest.raises(TypeError, match="must be a ParameterPrior"):
        BayesianVisibilityProblem(obs, sources, {**priors, "source[0].amp": norm()})
    with pytest.raises(ValueError, match="requires known measurement"):
        BayesianVisibilityProblem(obs, sources, priors, weighting="none")


def test_problem_rejects_malformed_vectors(problem_inputs):
    problem = BayesianVisibilityProblem(*problem_inputs)
    for method in (
        problem.predict,
        problem.log_prior,
        problem.log_likelihood,
        problem.log_posterior,
        problem.prior_transform,
    ):
        with pytest.raises(ValueError, match="Expected a parameter vector"):
            method(np.ones(problem.ndim + 1))
        with pytest.raises(ValueError, match="must be real"):
            method(np.ones(problem.ndim, dtype=complex))
        with pytest.raises(TypeError, match="not a Quantity array"):
            method(np.ones(problem.ndim) * PHOTON_FLUX_UNIT)


def test_problem_does_not_hide_prediction_errors(problem_inputs, monkeypatch):
    problem = BayesianVisibilityProblem(*problem_inputs)

    def broken_predict(params):
        raise ValueError("unrelated implementation error")

    monkeypatch.setattr(problem, "predict", broken_predict)
    with pytest.raises(ValueError, match="unrelated implementation error"):
        problem.log_likelihood(problem.initial_params)


def test_problem_skips_prediction_outside_prior_support(problem_inputs, monkeypatch):
    obs, sources, priors = problem_inputs
    priors["source[0].amp"] = ParameterPrior(uniform(loc=100, scale=10), PHOTON_FLUX_UNIT)
    problem = BayesianVisibilityProblem(obs, sources, priors)

    def unexpected_prediction(params):
        pytest.fail("Zero-prior proposals should not evaluate predictions.")

    monkeypatch.setattr(problem, "predict", unexpected_prediction)
    assert problem.log_posterior(problem.initial_params) == -np.inf


def test_problem_loop_geometry_rejection():
    source = Source(
        "loop",
        2 * PHOTON_FLUX_UNIT,
        0 * apu.arcsec,
        0 * apu.arcsec,
        2 * apu.arcsec,
        5 * apu.arcsec,
        20 * apu.deg,
        40 * apu.deg,
    )
    sources = SourceList([source])
    u = np.array([0.01, -0.03]) / apu.arcsec
    v = np.array([0.02, 0.04]) / apu.arcsec
    obs = Visibilities(sources_to_vis(sources, u, v), u, v, uncertainty=np.ones(2) * PHOTON_FLUX_UNIT)
    priors = {
        name: ParameterPrior(norm(), unit)
        for name, unit in zip(sources.parameter_names, sources.parameter_units(PHOTON_FLUX_UNIT))
    }
    problem = BayesianVisibilityProblem(obs, sources, priors)
    params = problem.initial_params.copy()
    params[3:5] = [5, 2]
    assert problem.log_likelihood(params) == -np.inf
    params = problem.initial_params.copy()
    params[6] = 3 * np.pi
    assert problem.log_likelihood(params) == -np.inf
    with pytest.raises(InvalidSourceParameters, match="opening angle"):
        problem.predict(params)
    # Negative curvature is supported by the model, despite the optimiser's beta bounds.
    params[6] = -problem.initial_params[6]
    assert np.isfinite(problem.log_likelihood(params))
