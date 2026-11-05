import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm, randint, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import ParameterPrior

PHOTON_FLUX_UNIT = apu.ph / (apu.s * apu.cm**2 * apu.keV)


def test_prior_density_normalises_after_unit_conversion():
    prior = ParameterPrior(uniform(loc=1, scale=2), apu.arcmin)
    integral, _ = quad(lambda x: np.exp(prior.logpdf(x, unit=apu.arcsec)), 60, 180)
    assert_allclose(integral, 1)
    assert_allclose(prior.logpdf(120, unit=apu.arcsec), -np.log(120))
    assert prior.logpdf(30, unit=apu.arcsec) == -np.inf


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
