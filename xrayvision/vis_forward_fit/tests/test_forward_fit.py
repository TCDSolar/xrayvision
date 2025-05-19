import astropy.units as apu
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from xrayvision.transform import generate_uv
from xrayvision.vis_forward_fit.forward_fit import (
    SourceList,
    _vis_forward_fit_minimise,
    circular_gaussian_vis,
    sources_to_image,
    sources_to_vis,
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
    sources = SourceList([Source("circular", 2, -4, -5, 2), Source("circular", 4, 5, 4, 3)])
    image = sources_to_image(sources, [33, 33] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pixel)
    assert_allclose(image.sum(), 6, rtol=5e-5)
    y, x = 33 // 2 - 4, 33 // 2 - 5
    assert_allclose(image[x, y], 2 / (2 * np.pi * 2**2), atol=1e-5, rtol=5e-5)
    y, x = 33 // 2 + 5, 33 // 2 + 4
    assert_allclose(image[x, y], 4 / (2 * np.pi * 3**2), atol=1e-5, rtol=5e-5)


def test_sources_to_vis():
    sources = SourceList([Source("circular", 2, -4, -5, 2), Source("circular", 4, 5, 4, 3)])
    uu = generate_uv(33 * apu.pixel).value
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u, v)
    assert_allclose(vis.real.max(), 6, rtol=5e-5)


# Just testing the machinery not if the fitting is robust/good
def test_vis_forward_fit_minimise():
    sources = SourceList([Source("circular", 2, -4, -5, 2), Source("elliptical", 4, 5, 4, 3, 8, 45)])
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    # Create non-optimal source parameters
    init_souces = SourceList.from_params(sources, np.random.randn(len(sources.params)) * 0.1 + sources.params)
    sources_fit, res = _vis_forward_fit_minimise(visobs, init_souces, method="Nelder-Mead")
    assert_allclose(sources_fit.params, sources.params, atol=1e-4, rtol=1e-5)


# Just testing the machinery not if the fitting is robust/good
def test_vis_forward_fit_minimise_pso():
    sources = SourceList([Source("circular", 2, -4, -5, 2), Source("circular", 4, 5, 4, 3)])
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    # Create non-optimal source parameters
    init_souces = SourceList.from_params(sources, np.random.randn(len(sources.params)) * 0.1 + sources.params)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    sources_fit, res = _vis_forward_fit_minimise(visobs, init_souces, method="PSO")
    assert_allclose(sources_fit.params, sources.params, atol=1e-4, rtol=1e-5)
