import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import astropy.units as apu

from xrayvision.transform import generate_uv
from xrayvision.vis_forward_fit._pso import VisForwardFitProblem
from xrayvision.vis_forward_fit.forward_fit import (
    SourceList,
    _vis_forward_fit_minimise,
    circular_gaussian_vis,
    fit_statistics,
    sources_to_image,
    sources_to_vis,
    vis_forward_fit,
)
from xrayvision.vis_forward_fit.sources import Source
from xrayvision.visibility import Visibilities


def test_simple_fit():
    uu = generate_uv(11 * apu.pixel)[::4]
    u, v = np.meshgrid(uu, uu)
    u = u.flatten().value
    v = v.flatten().value
    vis = circular_gaussian_vis(1, u, v, 0, 0, 2)
    vis_ri = np.hstack([vis.real, vis.imag])

    def objective(x, u, v, vis_ri):
        params = [x[0], u, v, *x[1:]]
        vispred = circular_gaussian_vis(*params)
        vispred_ri = np.hstack([vispred.real, vispred.imag])
        return np.sum(np.abs(vis_ri - vispred_ri) ** 2)

    res = minimize(objective, [0.5, 0.5, 0.5, 1], (u, v, vis_ri), method="Nelder-Mead")
    assert_allclose(res.x, [1, 0, 0, 2], atol=1e-5, rtol=1e-5)


def test_sources_to_map():
    sources = SourceList(
        [
            Source("circular", 2 * apu.arcsec, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 4 * apu.arcsec, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec),
        ]
    )
    image = sources_to_image(sources, [33, 33] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pixel)
    assert_allclose(image.sum().value, 6, rtol=5e-5)
    y, x = 33 // 2 - 4, 33 // 2 - 5
    assert_allclose(image[x, y].value, 2 / (2 * np.pi * 2**2), atol=1e-5, rtol=5e-5)
    y, x = 33 // 2 + 5, 33 // 2 + 4
    assert_allclose(image[x, y].value, 4 / (2 * np.pi * 3**2), atol=1e-5, rtol=5e-5)


def test_sources_to_vis():
    sources = SourceList(
        [
            Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec),
        ]
    )
    uu = generate_uv(33 * apu.pixel).value
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u, v)
    assert_allclose(vis.real.max(), 6, rtol=5e-5)


# Just testing the machinery not if the fitting is robust/good
def test_vis_forward_fit_minimise():
    rng = np.random.default_rng(42)
    sources = SourceList(
        [
            Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("elliptical", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec, 8 * apu.arcsec, 45 * apu.deg),
        ]
    )
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    # Create non-optimal source parameters. sources.params is already a flat list of plain
    # floats in canonical units, so it can be perturbed directly like any numeric array.
    init_souces = SourceList.from_params(sources, rng.standard_normal(len(sources.params)) * 0.1 + sources.params)
    sources_fit, _res = _vis_forward_fit_minimise(visobs, init_souces, method="Nelder-Mead")
    assert_allclose(sources_fit.params, sources.params, atol=1e-4, rtol=1e-5)


# Just testing the machinery not if the fitting is robust/good
def test_vis_forward_fit_minimise_pso():
    rng = np.random.default_rng(42)
    sources = SourceList(
        [
            Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec),
        ]
    )
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    # Create non-optimal source parameters
    init_souces = SourceList.from_params(sources, rng.standard_normal(len(sources.params)) * 0.1 + sources.params)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    sources_fit, _res = _vis_forward_fit_minimise(visobs, init_souces, method="PSO")
    # the sources can swap so sort by x0, y0 before comparison
    sources.sort(key=lambda x: (x.x0, x.y0))
    sources_fit.sort(key=lambda x: (x.x0, x.y0))
    assert_allclose(sources_fit.params, sources.params, atol=1e-4, rtol=5e-5)


def test_vis_forward_fit_weights_by_amplitude_uncertainty():
    sources = SourceList([Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec)])
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)

    unweighted = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    weighted = Visibilities(
        vis.flatten() * apu.ph,
        u.flatten(),
        v.flatten(),
        amplitude_uncertainty=np.full(vis.size, 2.0) * apu.ph,
    )

    # A deliberately off-truth source, so the residual (and hence the objective) at this
    # point is nonzero and the 1/sigma^2 scaling is visible.
    off_source = SourceList([Source("circular", 2, -3 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec)])
    x = np.array([off_source.params])

    problem_unweighted = VisForwardFitProblem(unweighted.u, unweighted.v, unweighted, off_source)
    problem_weighted = VisForwardFitProblem(weighted.u, weighted.v, weighted, off_source)
    out_unweighted: dict = {}
    out_weighted: dict = {}
    problem_unweighted._evaluate(x, out_unweighted)
    problem_weighted._evaluate(x, out_weighted)

    # sigma=2 everywhere -> chi2 should be exactly 1/4 of the unweighted sum of squares.
    assert_allclose(out_weighted["F"], out_unweighted["F"] / 4)


def test_vis_forward_fit_minimise_pso_n_runs():
    rng = np.random.default_rng(3)
    sources = SourceList(
        [
            Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec),
        ]
    )
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    init_souces = SourceList.from_params(sources, rng.standard_normal(len(sources.params)) * 0.1 + sources.params)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())

    _sources_fit, res_single = _vis_forward_fit_minimise(visobs, init_souces, method="PSO", n_runs=1)
    _sources_fit, res_sequential = _vis_forward_fit_minimise(visobs, init_souces, method="PSO", n_runs=3, n_jobs=1)
    _sources_fit, res_parallel = _vis_forward_fit_minimise(visobs, init_souces, method="PSO", n_runs=3, n_jobs=3)

    # More restarts should never find a worse optimum than a single run.
    assert res_sequential.F[0] <= res_single.F[0] + 1e-6
    # The same seeded restarts run in parallel vs sequentially must agree exactly.
    assert_allclose(res_parallel.F, res_sequential.F)


@pytest.mark.parametrize("uv_unit", [apu.arcsec**-1, apu.deg**-1])
@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
def test_vis_forward_fit_single_circular_source(method, uv_unit):
    true_params = [2, -4, -5, 2]
    sources = SourceList([Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec)])
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten().to(uv_unit), v.flatten().to(uv_unit))

    # Quantity-valued initial guess to exercise the public vis_forward_fit() pipeline end-to-end,
    # including the final re-imaging step, which requires Quantity-valued source parameters.
    init_sources = SourceList(
        [Source("circular", 2.1 * apu.ph, -3.9 * apu.arcsec, -4.9 * apu.arcsec, 2.1 * apu.arcsec)]
    )
    _image, sources_fit = vis_forward_fit(
        visobs,
        init_sources,
        shape=[33, 33] * apu.pixel,
        pixel_size=[1, 1] * apu.arcsec / apu.pixel,
        map=False,
        method=method,
    )
    assert_allclose([p.value for p in sources_fit.params], true_params, atol=1e-2, rtol=5e-2)


@pytest.mark.parametrize("include_loop", [False, True])
@pytest.mark.parametrize("uv_unit", [apu.arcsec**-1, apu.deg**-1])
def test_pso_objective_matches_scalar_predictions(include_loop, uv_unit):
    sources = SourceList(
        [
            Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec),
            Source("elliptical", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec, 8 * apu.arcsec, 45 * apu.deg),
        ]
    )
    if include_loop:
        sources.append(
            Source(
                "loop", 3, 2 * apu.arcsec, -1 * apu.arcsec, 3 * apu.arcsec, 7 * apu.arcsec, 20 * apu.deg, 45 * apu.deg
            )
        )
    u = np.array([0.01, 0.03, 0.07]) / apu.arcsec
    v = np.array([0.04, 0.02, -0.05]) / apu.arcsec
    vis = sources_to_vis(sources, u, v)
    visobs = Visibilities(vis * apu.ph, u.to(uv_unit), v.to(uv_unit))
    population = np.array([sources.params, sources.params])
    # Include a nonzero residual to check the objective's scale as well as its minimum.
    population[1, 1] += 1
    expected = []
    for params in population:
        prediction = sources_to_vis(SourceList.from_params(sources, params), u, v)
        expected.append(np.sum(np.abs(vis - prediction) ** 2))

    problem = VisForwardFitProblem(visobs.u, visobs.v, visobs, sources)
    out = {}
    problem._evaluate(population, out)

    assert_allclose(out["F"], expected, atol=1e-12)


def test_vis_forward_fit_elliptical_quantity_theta():
    sources = SourceList(
        [Source("elliptical", 4, 5 * apu.arcsec, 4 * apu.arcsec, 3 * apu.arcsec, 2 * apu.arcsec, 45 * apu.deg)]
    )
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())

    init_sources = SourceList(
        [
            Source(
                "elliptical",
                4.1 * apu.ph,
                4.9 * apu.arcsec,
                4.1 * apu.arcsec,
                3.1 * apu.arcsec,
                2.1 * apu.arcsec,
                40 * apu.deg,
            )
        ]
    )
    _image, sources_fit = vis_forward_fit(
        visobs,
        init_sources,
        shape=[33, 33] * apu.pixel,
        pixel_size=[1, 1] * apu.arcsec / apu.pixel,
        map=False,
        method="Nelder-Mead",
    )
    # theta is always returned in the canonical unit (radians), regardless of what unit
    # it was given in.
    assert sources_fit[0].theta.unit == apu.rad
    # The elliptical model is degenerate under theta -> theta + 180 deg (rotating an ellipse by
    # half a turn leaves it unchanged), so compare reconstructed visibilities, not the raw angle.
    vis_fit = sources_to_vis(sources_fit, u, v)
    assert_allclose(vis_fit.to_value(apu.ph), vis, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
@pytest.mark.parametrize("flux_unit", [apu.Jy, apu.ct / apu.s, apu.ph / (apu.s * apu.cm**2 * apu.keV)])
def test_fit_flux_units_and_reuse(method, flux_unit):
    u, v = np.meshgrid(np.linspace(-0.12, 0.12, 7), np.linspace(-0.1, 0.1, 7))
    u, v = u.ravel() / apu.arcsec, v.ravel() / apu.arcsec
    truth = SourceList([Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec)])
    vis = sources_to_vis(truth, u, v) * flux_unit
    obs = Visibilities(vis, u.to(apu.deg**-1), v.to(apu.deg**-1), uncertainty=np.full(vis.size, 0.1 + 0.2j) * flux_unit)
    # A differently scaled amplitude unit must affect initial values and bounds,
    # not just the unit label attached after optimisation.
    scaled_unit = apu.Unit(1000 * flux_unit)
    initial = SourceList(
        [Source("circular", (2.1 * flux_unit).to(scaled_unit), -3.9 * apu.arcsec, -4.9 * apu.arcsec, 2.1 * apu.arcsec)]
    )
    image, fitted, result = vis_forward_fit(
        obs,
        initial,
        [33, 33] * apu.pix,
        [1, 1] * apu.arcsec / apu.pix,
        map=False,
        method=method,
        return_result=True,
    )
    prediction = sources_to_vis(fitted, obs.u, obs.v)
    assert fitted[0].amp.unit == flux_unit
    assert_allclose(prediction.to_value(flux_unit), vis.value, atol=1e-4)
    assert image.unit == flux_unit
    assert_allclose(initial[0].amp.value, 0.0021)
    assert initial[0].amp.unit == scaled_unit
    assert_allclose(fitted.bounds, fitted.to_canonical().bounds)
    assert_allclose(sources_to_image(fitted, [33, 33] * apu.pix, [1, 1] * apu.arcsec / apu.pix), image)

    # Returned quantity sources work directly as another fit's initial guess.
    refitted, _ = _vis_forward_fit_minimise(obs, fitted, method="Nelder-Mead")
    assert_allclose(sources_to_vis(refitted, u, v), vis.value, atol=1e-4)
    residual = vis.value - prediction.to_value(flux_unit)
    expected = np.sum((residual.real / 0.1) ** 2 + (residual.imag / 0.2) ** 2)
    stats = result.fit_statistics
    assert stats.weighting == "components"
    assert stats.dof == 2 * vis.size - 4
    assert_allclose(stats.objective, expected, atol=1e-12)
    assert_allclose(stats.reduced_chi2, expected / stats.dof, atol=1e-12)
    assert_allclose(fit_statistics(obs, fitted).objective, stats.objective, atol=1e-12)


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
def test_explicit_bounds_recover_source_outside_initial_bounds(method):
    flux_unit = apu.ph / (apu.s * apu.cm**2 * apu.keV)
    u, v = np.meshgrid(np.linspace(-0.06, 0.06, 7), np.linspace(-0.05, 0.05, 7))
    u, v = u.ravel() / apu.arcsec, v.ravel() / apu.arcsec
    truth = SourceList([Source("circular", 2 * flux_unit, -1.5 * apu.arcsec, 0.8 * apu.arcsec, 4 * apu.arcsec)])
    scaled_unit = apu.Unit(1000 * flux_unit)
    initial = SourceList(
        [Source("circular", (1.8 * flux_unit).to(scaled_unit), -1 * apu.arcsec, 0.5 * apu.arcsec, 0.8 * apu.arcsec)]
    )
    vis = sources_to_vis(truth, u, v)
    obs = Visibilities(vis, u, v, uncertainty=np.full(vis.size, 0.03 + 0.06j) * flux_unit)
    default_upper_width = initial.bounds[1][-1]
    bounds = np.array([[0.1, -10, -10, 0.5], [4, 10, 10, 6]])
    source_bounds = bounds.copy()
    source_bounds[:, 0] /= 1000
    initial[0].bounds = source_bounds
    _, fitted, result = vis_forward_fit(
        obs,
        initial,
        [17, 17] * apu.pix,
        [1, 1] * apu.arcsec / apu.pix,
        method=method,
        map=False,
        return_result=True,
        uncertainty="gauss_newton",
    )
    assert fitted[0].sigma.to_value(apu.arcsec) > default_upper_width
    assert_allclose(sources_to_vis(fitted, u, v).to_value(flux_unit), vis.value, atol=1e-4)
    assert_allclose(result.fit_bounds, bounds)
    assert_allclose(fitted[0].bounds, bounds)
    assert_allclose(initial[0].bounds, source_bounds)
    assert result.parameter_uncertainty.covariance is not None
    assert result.parameter_uncertainty.status == "ok"


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
def test_fit_rejects_incompatible_flux_units(method):
    sources = SourceList([Source("circular", 2 * apu.s, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec)])
    obs = Visibilities(np.ones(3) * apu.Jy, np.arange(3) / apu.arcsec, np.arange(3) / apu.arcsec)
    with pytest.raises(apu.UnitConversionError):
        _vis_forward_fit_minimise(obs, sources, method=method)


def test_prediction_sums_differential_flux_in_equivalent_units():
    per_kev = apu.ph / (apu.s * apu.cm**2 * apu.keV)
    per_ev = apu.ph / (apu.s * apu.cm**2 * apu.eV)
    sources = SourceList(
        [
            Source("circular", 1000 * per_kev, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 1 * per_ev, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 1000, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec),
        ]
    )
    uv = np.array([0, 0.05]) / apu.arcsec
    prediction = sources_to_vis(sources, uv, uv)
    expected = circular_gaussian_vis(3000, uv.value, uv.value, 0, 0, 2)
    assert prediction.unit == per_kev
    assert_allclose(prediction.to_value(per_kev), expected)
    assert sources[1].amp.unit == per_ev
