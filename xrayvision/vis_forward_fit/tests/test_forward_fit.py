import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import astropy.units as apu

from xrayvision.transform import generate_uv
from xrayvision.vis_forward_fit.forward_fit import (
    SourceList,
    _vis_forward_fit_minimise,
    circular_gaussian_vis,
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


# Regression test: a single non-Loop source used to crash or get a bogus constraint in the
# PSO path, since VisForwardFitProblem hardcoded a Loop-only sigma_min/sigma_max constraint
# at a fixed parameter offset (x[3] - x[4]) regardless of which sources were being fit.
# Also exercises the full public vis_forward_fit() pipeline (not just _vis_forward_fit_minimise),
# including the final re-imaging step, which requires Quantity-valued source parameters.
@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
def test_vis_forward_fit_single_circular_source(method):
    true_params = [2, -4, -5, 2]
    sources = SourceList([Source("circular", 2, -4 * apu.arcsec, -5 * apu.arcsec, 2 * apu.arcsec)])
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())

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


# Regression test: Elliptical.bounds used to crash with a UnitConversionError when `theta`
# was passed as an angular Quantity, because the +/-22.5 window was applied before stripping units.
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
    fitted_params = [p.value for p in sources_fit.params]
    vis_fit = sources_to_vis(SourceList([type(sources_fit[0])._from_canonical(*fitted_params)]), u.value, v.value)
    assert_allclose(vis_fit, vis, atol=1e-6, rtol=1e-3)
