import numpy as np
import pytest
from numpy.testing import assert_allclose

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
