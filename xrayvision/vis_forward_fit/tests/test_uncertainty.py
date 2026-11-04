import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as apu

from xrayvision.vis_forward_fit.forward_fit import sources_to_vis, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.vis_forward_fit.uncertainty import _align_sources, estimate_parameter_uncertainty
from xrayvision.visibility import Visibilities


@pytest.fixture
def circular_data():
    rng = np.random.default_rng(18)
    u, v = rng.uniform(-0.12, 0.12, (2, 30)) / apu.arcsec
    sources = SourceList([Source("circular", 2 * apu.Jy, -1.5 * apu.arcsec, 0.8 * apu.arcsec, 2 * apu.arcsec)])
    prediction = sources_to_vis(sources, u, v)
    obs = Visibilities(prediction, u, v, uncertainty=np.full(30, 0.03 + 0.06j) * apu.Jy)
    return obs, sources


def circular_covariance(obs, sources):
    """Independent analytical Fisher matrix for a circular Gaussian."""
    amp, _, _, sigma = sources.to_canonical(apu.Jy).params
    u, v = obs.u.to_value(apu.arcsec**-1), obs.v.to_value(apu.arcsec**-1)
    prediction = sources_to_vis(sources, obs.u, obs.v).to_value(apu.Jy)
    derivative = np.array(
        [
            prediction / amp,
            2j * np.pi * u * prediction,
            2j * np.pi * v * prediction,
            -4 * np.pi**2 * sigma * (u**2 + v**2) * prediction,
        ]
    ).T
    jacobian = np.vstack([derivative.real / 0.03, derivative.imag / 0.06])
    return np.linalg.inv(jacobian.T @ jacobian)


def test_gauss_newton_matches_analytic_covariance(circular_data):
    obs, sources = circular_data
    estimate = estimate_parameter_uncertainty(obs, sources, method="gauss_newton")
    expected = circular_covariance(obs, sources)
    assert estimate.status == "ok"
    assert estimate.diagnostics["rank"] == 4
    assert estimate.diagnostics["rescaled"] is False
    assert_allclose(estimate.covariance, expected, rtol=2e-6, atol=1e-12)
    assert estimate.stderr["source[0].x0"].unit == apu.arcsec
    assert estimate.covariance_between("source[0].amp", "source[0].x0").unit == apu.Jy * apu.arcsec
    centre = sources[0].x0
    assert_allclose(estimate.intervals["source[0].x0"].value, centre + np.array([-1, 1]) * np.sqrt(expected[1, 1]))
    assert estimate.samples is None


def test_gauss_newton_flux_unit_invariance(circular_data):
    obs, sources = circular_data
    converted = Visibilities(
        obs.visibilities.to(apu.mJy),
        obs.u.to(apu.deg**-1),
        obs.v.to(apu.deg**-1),
        uncertainty=obs.uncertainty.to(apu.mJy),
    )
    first = estimate_parameter_uncertainty(obs, sources, method="gauss_newton")
    second = estimate_parameter_uncertainty(converted, sources, method="gauss_newton")
    assert_allclose(second.stderr["source[0].amp"].to_value(apu.Jy), first.stderr["source[0].amp"].value, rtol=1e-7)
    assert_allclose(second.stderr["source[0].sigma"], first.stderr["source[0].sigma"], rtol=1e-7)


def test_monte_carlo_recovers_known_small_noise_errors(circular_data):
    obs, sources = circular_data
    estimate = estimate_parameter_uncertainty(obs, sources, n_samples=96, seed=42)
    expected = np.sqrt(np.diag(circular_covariance(obs, sources)))
    assert estimate.status == "ok"
    assert estimate.diagnostics["n_successful"] == 96
    assert estimate.samples.shape == (96, 4)
    actual = np.array([estimate.stderr[name].value for name in estimate.parameter_names])
    assert_allclose(actual, expected, rtol=0.3)
    assert estimate.intervals["source[0].amp"].unit == apu.Jy
    assert estimate.diagnostics["centre"] == "fitted_model"
    assert_allclose(sources[0].amp.value, 2)


def test_monte_carlo_seed_reproducible_in_parallel(circular_data):
    obs, sources = circular_data
    serial = estimate_parameter_uncertainty(obs, sources, n_samples=4, seed=21)
    parallel = estimate_parameter_uncertainty(obs, sources, n_samples=4, seed=21, n_jobs=2)
    assert_allclose(parallel.samples, serial.samples, rtol=0, atol=0)
    assert_allclose(parallel.covariance, serial.covariance, rtol=0, atol=0)


@pytest.mark.parametrize("uncertainty_method", ["monte_carlo", "gauss_newton"])
@pytest.mark.parametrize("fit_method", ["Nelder-Mead", "PSO"])
def test_public_uncertainty_api(circular_data, uncertainty_method, fit_method):
    obs, sources = circular_data
    _, fitted, result = vis_forward_fit(
        obs,
        sources,
        [17, 17] * apu.pix,
        [1, 1] * apu.arcsec / apu.pix,
        method=fit_method,
        map=False,
        return_result=True,
        pop=24,
        uncertainty=uncertainty_method,
        uncertainty_options={"n_samples": 4, "seed": 7},
    )
    estimate = result.parameter_uncertainty
    assert estimate.method == uncertainty_method
    assert estimate.status == "ok"
    assert estimate.parameter_names == result.parameter_names == fitted.parameter_names
    assert estimate.parameter_units == result.parameter_units
    assert estimate.stderr["source[0].amp"].unit == apu.Jy
    if fit_method == "PSO":
        population = result.pop.get("X")
        assert population.shape == (24, 4)
        assert result.parameter_units[0] == apu.Jy
        if uncertainty_method == "monte_carlo":
            assert estimate.samples.shape == (4, 4)


def test_gauss_newton_does_not_invert_unidentifiable_directions(circular_data):
    _, sources = circular_data
    u = np.zeros(10) / apu.arcsec
    obs = Visibilities(np.full(10, 2 + 0j) * apu.Jy, u, u, uncertainty=np.ones(10) * apu.Jy)
    estimate = estimate_parameter_uncertainty(obs, sources, method="gauss_newton")
    assert estimate.status == "rank_deficient"
    assert estimate.diagnostics["rank"] == 1
    assert estimate.covariance is None
    assert not estimate.stderr
    with pytest.raises(ValueError, match="unavailable"):
        estimate.covariance_between("source[0].amp", "source[0].sigma")


def test_gauss_newton_flags_active_bounds(circular_data):
    obs, sources = circular_data
    bounds = np.array(sources.to_canonical(apu.Jy).bounds)
    bounds[0, 1] = sources[0].x0
    estimate = estimate_parameter_uncertainty(obs, sources, method="gauss_newton", bounds=bounds)
    assert estimate.status == "active_bounds"
    assert estimate.diagnostics["active_bounds"] == ["source[0].x0"]
    assert estimate.covariance is None


def test_circular_ellipse_has_no_identifiable_orientation(circular_data):
    obs, _ = circular_data
    sources = SourceList(
        [
            Source(
                "elliptical",
                2 * apu.Jy,
                -1.5 * apu.arcsec,
                0.8 * apu.arcsec,
                2 * apu.arcsec,
                2 * apu.arcsec,
                30 * apu.deg,
            )
        ]
    )
    estimate = estimate_parameter_uncertainty(obs, sources, method="gauss_newton")
    assert estimate.status == "rank_deficient"
    assert estimate.diagnostics["rank"] == 5
    assert estimate.covariance is None


def test_matching_source_swaps_and_periodic_ellipse_axes():
    first = Source("elliptical", 2, -5 * apu.arcsec, 1 * apu.arcsec, 2 * apu.arcsec, 4 * apu.arcsec, 89 * apu.deg)
    second = Source("elliptical", 3, 5 * apu.arcsec, -2 * apu.arcsec, 3 * apu.arcsec, 5 * apu.arcsec, 30 * apu.deg)
    reference = SourceList([first, second])
    equivalent = Source("elliptical", 2, -5 * apu.arcsec, 1 * apu.arcsec, 4 * apu.arcsec, 2 * apu.arcsec, -1 * apu.deg)
    candidate = SourceList([second, equivalent])
    aligned = _align_sources(candidate, reference, np.ones(12))
    assert_allclose(aligned, reference.params)


@pytest.mark.parametrize("shape", ["elliptical", "loop"])
@pytest.mark.parametrize("method", ["monte_carlo", "gauss_newton"])
def test_uncertainties_for_existing_extended_models(circular_data, shape, method):
    obs, _ = circular_data
    if shape == "elliptical":
        source = Source(
            "elliptical", 2 * apu.Jy, 0 * apu.arcsec, 0 * apu.arcsec, 3 * apu.arcsec, 5 * apu.arcsec, 20 * apu.deg
        )
    else:
        source = Source(
            "loop",
            2 * apu.Jy,
            0 * apu.arcsec,
            0 * apu.arcsec,
            4 * apu.arcsec,
            7 * apu.arcsec,
            20 * apu.deg,
            45 * apu.deg,
        )
    sources = SourceList([source])
    simulated = Visibilities(
        sources_to_vis(sources, obs.u, obs.v),
        obs.u,
        obs.v,
        uncertainty=np.full(obs.u.size, 0.001 + 0.001j) * apu.Jy,
    )
    estimate = estimate_parameter_uncertainty(simulated, sources, method=method, n_samples=4, seed=12)
    assert estimate.status == "ok"
    assert estimate.covariance.shape == (source.n_params, source.n_params)
    angle_name = "source[0].theta" if shape == "elliptical" else "source[0].beta"
    assert estimate.stderr[angle_name].unit == apu.rad


def test_matching_loop_orientation_wrap():
    reference = SourceList(
        [Source("loop", 2, 0 * apu.arcsec, 0 * apu.arcsec, 4 * apu.arcsec, 7 * apu.arcsec, 179 * apu.deg, 45 * apu.deg)]
    )
    candidate = SourceList(
        [
            Source(
                "loop", 2, 0 * apu.arcsec, 0 * apu.arcsec, 4 * apu.arcsec, 7 * apu.arcsec, -179 * apu.deg, 45 * apu.deg
            )
        ]
    )
    aligned = _align_sources(candidate, reference, np.ones(7))
    assert_allclose(aligned[5], np.deg2rad(181))


@pytest.mark.parametrize("n_successful", [0, 1, 2])
def test_monte_carlo_reports_failed_refits(circular_data, monkeypatch, n_successful):
    obs, sources = circular_data
    module = importlib.import_module("xrayvision.vis_forward_fit.uncertainty")
    counter = iter(range(4))

    def fake_refit(task):
        if next(counter) < n_successful:
            return sources.to_canonical(apu.Jy), ""
        return None, "Refit did not converge."

    monkeypatch.setattr(module, "_refit_sample", fake_refit)
    estimate = estimate_parameter_uncertainty(obs, sources, n_samples=4, seed=10)
    assert estimate.diagnostics["n_successful"] == n_successful
    assert estimate.diagnostics["n_failed"] == 4 - n_successful
    assert len(estimate.diagnostics["failures"]) == 4 - n_successful
    assert estimate.status == ("partial" if n_successful >= 2 else "failed")
    assert (estimate.covariance is None) == (n_successful < 2)


def test_refits_reuse_original_bounds(circular_data, monkeypatch):
    obs, sources = circular_data
    forward = importlib.import_module("xrayvision.vis_forward_fit.forward_fit")
    original = forward._vis_forward_fit_minimise
    bounds = np.array(sources.to_canonical(apu.Jy).bounds)
    bounds[:, 0] = [1, 3]
    seen = []

    def record_bounds(*args, **kwargs):
        seen.append(kwargs["bounds"].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(forward, "_vis_forward_fit_minimise", record_bounds)
    estimate = estimate_parameter_uncertainty(obs, sources, n_samples=3, seed=0, bounds=bounds, store_samples=False)
    assert estimate.status == "ok"
    assert estimate.samples is None
    assert len(seen) == 3
    for actual in seen:
        assert_allclose(actual, bounds)


def test_failed_original_fit_has_no_parameter_errors(circular_data, monkeypatch):
    obs, sources = circular_data
    forward = importlib.import_module("xrayvision.vis_forward_fit.forward_fit")
    fitted, result = forward._vis_forward_fit_minimise(obs, sources, "Nelder-Mead")
    result.success = False
    monkeypatch.setattr(forward, "_vis_forward_fit_minimise", lambda *args, **kwargs: (fitted, result))
    _, _, returned = vis_forward_fit(
        obs,
        sources,
        [17, 17] * apu.pix,
        [1, 1] * apu.arcsec / apu.pix,
        map=False,
        return_result=True,
        uncertainty="monte_carlo",
    )
    assert returned.parameter_uncertainty.status == "fit_failed"
    assert returned.parameter_uncertainty.covariance is None


@pytest.mark.parametrize("method", ["monte_carlo", "gauss_newton"])
def test_uncertainty_requires_measurement_errors(circular_data, method):
    obs, sources = circular_data
    unweighted = Visibilities(obs.visibilities, obs.u, obs.v)
    with pytest.raises(ValueError, match="known measurement"):
        estimate_parameter_uncertainty(unweighted, sources, method=method)


def test_uncertainty_option_validation(circular_data):
    obs, sources = circular_data
    with pytest.raises(ValueError, match="return_result"):
        vis_forward_fit(obs, sources, [17, 17] * apu.pix, [1, 1] * apu.arcsec / apu.pix, uncertainty="monte_carlo")
    with pytest.raises(ValueError, match="n_samples"):
        estimate_parameter_uncertainty(obs, sources, n_samples=1)
    with pytest.raises(ValueError, match="confidence_level"):
        estimate_parameter_uncertainty(obs, sources, confidence_level=1)
