import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

import astropy.units as apu

from xrayvision.vis_forward_fit.fit_data import VisibilityFitData
from xrayvision.vis_forward_fit.forward_fit import fit_statistics, sources_to_vis
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities


def make_observations(**kwargs):
    return Visibilities(
        np.array([2 + 3j, 4 + 5j, 6 + 7j]) * apu.Jy,
        np.array([0.01, 0.02, 0.03]) / apu.arcsec,
        np.array([0.04, 0.05, 0.06]) / apu.arcsec,
        **kwargs,
    )


def test_component_errors_and_population_residuals():
    obs = make_observations(
        uncertainty=np.full(3, 1 + 2j) * apu.Jy,
        amplitude_uncertainty=np.full(3, 9) * apu.Jy,
    )
    data = VisibilityFitData(obs)
    prediction = (obs.visibilities - (1 + 2j) * apu.Jy).to(apu.mJy)
    assert data.weighting == "components"
    assert_allclose(data.residuals(prediction), np.ones(6))
    assert_allclose(data.objective(prediction), 6)
    assert_allclose(data.objective(np.stack([prediction, obs.visibilities])), [6, 0])

    # The IDL convention is an explicit alternative, not an implicit collapse
    # of anisotropic component errors into phase-dependent amplitude errors.
    amplitude_data = VisibilityFitData(obs, weighting="amplitude")
    assert_allclose(amplitude_data.objective(prediction), 15 / 81)


def test_real_uncertainties_apply_to_both_components():
    data = VisibilityFitData(make_observations(uncertainty=np.full(3, 2) * apu.Jy))
    assert_allclose(data.sigma_ri, np.full(6, 2))


def test_statistics_use_real_data_count_and_do_not_reduce_objective():
    data = VisibilityFitData(make_observations(amplitude_uncertainty=np.ones(3) * apu.Jy))
    prediction = data.values - (1 + 2j)
    stats = data.statistics(prediction, n_parameters=4)
    assert stats.objective == 15
    assert stats.chi2 == 15
    assert stats.dof == 2
    assert stats.reduced_chi2 == 7.5
    expected_log_likelihood = multivariate_normal.logpdf(
        data.values_ri, mean=np.concatenate([prediction.real, prediction.imag]), cov=np.eye(6)
    )
    assert_allclose(stats.log_likelihood, expected_log_likelihood)
    assert_allclose(stats.aic, 2 * 4 - 2 * expected_log_likelihood)
    assert_allclose(stats.bic, 4 * np.log(6) - 2 * expected_log_likelihood)
    assert data.objective(prediction) == 15
    assert data.statistics(prediction, n_parameters=6).reduced_chi2 is None


def test_unweighted_objective_is_not_reported_as_chi_square():
    obs = make_observations()
    data = VisibilityFitData(obs)
    stats = data.statistics(data.values - 1, n_parameters=4)
    assert stats.objective == 3
    assert stats.weighting == "none"
    assert stats.chi2 is None
    assert stats.reduced_chi2 is None
    assert stats.log_likelihood is None
    assert stats.aic is None
    assert stats.bic is None
    weighted = make_observations(amplitude_uncertainty=np.full(3, 100) * apu.Jy)
    assert VisibilityFitData(weighted, weighting="none").statistics(data.values - 1, 4) == stats


@pytest.mark.parametrize("error", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("weighting", ["amplitude", "components"])
def test_invalid_standard_deviations_are_rejected(error, weighting):
    kwargs = {"amplitude_uncertainty" if weighting == "amplitude" else "uncertainty": np.full(3, error) * apu.Jy}
    with pytest.raises(ValueError, match="finite and strictly positive"):
        VisibilityFitData(make_observations(**kwargs), weighting=weighting)


@pytest.mark.parametrize("weighting", ["amplitude", "components"])
def test_requested_weighting_requires_uncertainties(weighting):
    with pytest.raises(ValueError, match="requires"):
        VisibilityFitData(make_observations(), weighting=weighting)


def test_incompatible_prediction_units_are_rejected():
    data = VisibilityFitData(make_observations())
    with pytest.raises(apu.UnitConversionError):
        data.objective(np.ones(3) * apu.s)


def test_loop_statistics_count_seven_parameters():
    sources = SourceList(
        [Source("loop", 3, 2 * apu.arcsec, -1 * apu.arcsec, 3 * apu.arcsec, 7 * apu.arcsec, 20 * apu.deg, 45 * apu.deg)]
    )
    u = np.linspace(-0.1, 0.1, 24) / apu.arcsec
    v = np.linspace(0.03, 0.06, 24) / apu.arcsec
    prediction = sources_to_vis(sources, u, v)
    obs = Visibilities((prediction + 1) * apu.Jy, u, v, amplitude_uncertainty=np.ones(24) * apu.Jy)
    stats = fit_statistics(obs, sources)
    assert stats.n_parameters == 7
    assert stats.dof == 41
    assert stats.chi2 == 24
    assert_allclose(stats.reduced_chi2, 24 / 41)


@pytest.mark.parametrize("weighting", ["components", "amplitude"])
def test_log_likelihood_matches_multivariate_gaussian(weighting):
    obs = make_observations(
        uncertainty=np.array([1 + 2j, 3 + 4j, 5 + 6j]) * apu.Jy,
        amplitude_uncertainty=np.array([2, 3, 4]) * apu.Jy,
    )
    data = VisibilityFitData(obs, weighting=weighting)
    prediction = data.values - np.array([0.5 + 1j, -2 + 3j, 4 - 2j])
    sigma = [1, 3, 5, 2, 4, 6] if weighting == "components" else [2, 3, 4, 2, 3, 4]
    expected = multivariate_normal.logpdf(
        np.concatenate([obs.visibilities.value.real, obs.visibilities.value.imag]),
        mean=np.concatenate([prediction.real, prediction.imag]),
        cov=np.diag(np.square(sigma)),
    )
    actual = data.log_likelihood(prediction)
    assert isinstance(actual, float)
    assert_allclose(actual, expected)


def test_log_likelihood_batch_and_prediction_units():
    obs = make_observations(uncertainty=np.full(3, 1 + 2j) * apu.Jy)
    data = VisibilityFitData(obs)
    predictions = np.stack([obs.visibilities, obs.visibilities - (1 + 2j) * apu.Jy]).to(apu.mJy)
    likelihood = data.log_likelihood(predictions)
    assert_allclose(likelihood, [data.log_likelihood(predictions[0]), data.log_likelihood(predictions[1])])
    # Six component residuals of one sigma reduce log likelihood by three.
    assert_allclose(likelihood[1] - likelihood[0], -3)


def test_log_likelihood_observation_unit_jacobian():
    obs = make_observations(uncertainty=np.full(3, 1 + 2j) * apu.Jy)
    converted = Visibilities(obs.visibilities.to(apu.mJy), obs.u, obs.v, uncertainty=obs.uncertainty.to(apu.mJy))
    prediction = obs.visibilities - (1 + 2j) * apu.Jy
    jy = VisibilityFitData(obs).log_likelihood(prediction)
    mjy = VisibilityFitData(converted).log_likelihood(prediction)
    assert_allclose(mjy - jy, -6 * np.log(1000))
    # Absolute scores shift with the density's numeric unit; model differences
    # remain invariant when both fits use the same converted observations.
    jy_data, mjy_data = VisibilityFitData(obs), VisibilityFitData(converted)
    for criterion in ("aic", "bic"):
        jy_scores = [
            getattr(jy_data.statistics(model, k), criterion) for model, k in ((prediction, 2), (obs.visibilities, 4))
        ]
        mjy_scores = [
            getattr(mjy_data.statistics(model, k), criterion) for model, k in ((prediction, 2), (obs.visibilities, 4))
        ]
        assert_allclose(mjy_scores[0] - jy_scores[0], 12 * np.log(1000))
        assert_allclose(mjy_scores[1] - mjy_scores[0], jy_scores[1] - jy_scores[0])


@pytest.mark.parametrize("with_errors", [False, True])
def test_log_likelihood_rejects_unweighted_data(with_errors):
    kwargs = {"uncertainty": np.ones(3) * apu.Jy} if with_errors else {}
    data = VisibilityFitData(make_observations(**kwargs), weighting="none")
    with pytest.raises(ValueError, match="requires known measurement standard deviations"):
        data.log_likelihood(data.values)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e308])
def test_log_likelihood_invalid_prediction_does_not_contaminate_batch(invalid):
    data = VisibilityFitData(make_observations(uncertainty=np.ones(3) * apu.Jy))
    predictions = np.stack([data.values, data.values.copy()])
    predictions[1, 0] = invalid
    actual = data.log_likelihood(predictions)
    assert_allclose(actual[0], data.log_likelihood(data.values))
    assert actual[1] == -np.inf


def test_log_likelihood_validates_prediction_shape_and_units():
    data = VisibilityFitData(make_observations(uncertainty=np.ones(3) * apu.Jy))
    with pytest.raises(ValueError, match="final dimension"):
        data.log_likelihood(np.ones(2, dtype=complex))
    with pytest.raises(apu.UnitConversionError):
        data.log_likelihood(np.ones(3) * apu.s)
