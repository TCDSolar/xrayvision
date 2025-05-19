from copy import deepcopy

import astropy.units as apu
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from xrayvision.forward_fit import (
    Source,
    SourceList,
    _vis_forward_fit_minimise,
    circular_gaussian,
    circular_gaussian_vis,
    elliptical_gaussian,
    elliptical_gaussian_vis,
    sources_to_image,
    sources_to_vis,
)
from xrayvision.imaging import image_to_vis, vis_to_image
from xrayvision.transform import generate_uv, generate_xy
from xrayvision.visibility import Visibilities


@pytest.mark.parametrize("size", (65, 79))
def test_circular_ft_equivalence_fft(size):
    # So unless the array is sufficiently large this test fails
    # I think has to do with the fact no taking into account the sampleing and implicit windowing
    # TODO: How does this affect algo where the vis derived from map are compare to the observed?
    sigma = 4 * apu.arcsec
    xx = generate_xy(size * apu.pix)
    x, y = np.meshgrid(xx, xx)
    uu = generate_uv(size * apu.pix)
    u, v = np.meshgrid(uu, uu)
    u = u.flatten()
    v = v.flatten()

    image = circular_gaussian(1, x, y, 1 * apu.arcsec, -1 * apu.arcsec, sigma)

    vis_obs = image_to_vis(image, u=u, v=v)
    vis_func = Visibilities(circular_gaussian_vis(1, u, v, 1 * apu.arcsec, -1 * apu.arcsec, sigma).flatten(), u, v)

    image_func = vis_to_image(vis_func, [size, size] * apu.pixel)
    image_vis = vis_to_image(vis_obs, [size, size] * apu.pixel)

    assert_allclose(vis_func.visibilities, vis_obs.visibilities.value, atol=1e-13)
    assert_allclose(image_func, image_vis.value, atol=1e-13)


@pytest.mark.parametrize("x0", [0, 1, 2, 3])
@pytest.mark.parametrize("y0", [0, -1, 2, -3])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_equivalence_elliptical_to_circular(x0, y0, sigma):
    amp = 1
    x, y = np.meshgrid(np.linspace(-20, 20, 101), np.linspace(-20, 20, 101))
    image_circular = circular_gaussian(amp, x, y, x0, y0, sigma)
    image_elliptical = elliptical_gaussian(amp, x, y, x0, y0, sigma, sigma, 0)
    assert_allclose(image_circular, image_elliptical, atol=1e-13)


@pytest.mark.parametrize("x0", [0, 1, 2, 3])
@pytest.mark.parametrize("y0", [0, -1, 2, -3])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_equivalence_elliptical_to_circular_vis(x0, y0, sigma):
    amp = 1
    u, v = np.meshgrid(np.linspace(-20, 20, 101), np.linspace(-20, 20, 101))
    u = u * 1 / 2.5
    v = v * 1 / 2.5
    vis_circular = circular_gaussian_vis(amp, u, v, x0, y0, sigma)
    vis_elliptical = elliptical_gaussian_vis(amp, u, v, x0, y0, sigma, sigma, 0)
    assert_allclose(vis_circular, vis_elliptical, atol=1e-13)


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
    sources = SourceList([Source("circle", [2, -4, -5, 2]), Source("circle", [4, 5, 4, 3])])
    map = sources_to_image(sources, [33, 33] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pixel)
    assert_allclose(map.sum(), 6, rtol=5e-5)
    y, x = 33 // 2 - 4, 33 // 2 - 5
    assert_allclose(map[x, y], 2 / (2 * np.pi * 2**2), atol=1e-5, rtol=5e-5)
    y, x = 33 // 2 + 5, 33 // 2 + 4
    assert_allclose(map[x, y], 4 / (2 * np.pi * 3**2), atol=1e-5, rtol=5e-5)


def test_sources_to_vis():
    sources = SourceList([Source("circle", [2, -4, -5, 2]), Source("circle", [4, 5, 4, 3])])
    uu = generate_uv(33 * apu.pixel).value
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u, v)
    assert_allclose(vis.real.max(), 6, rtol=5e-5)


def test_vis_forward_fit_minimise():
    sources = SourceList([Source("circle", [2, -4, -5, 2]), Source("circle", [4, 5, 4, 3])])
    sources_orig = deepcopy(sources)
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())

    # Create non-optimal source parameters
    sources.from_list([1, -5, -4, 1, 5, 6, 3, 4])
    sources_fit, res = _vis_forward_fit_minimise(visobs, sources)
    assert_allclose(sources_fit.params, sources_orig.params, atol=1e-4, rtol=1e-5)


def test_vis_forward_fit_minimise_pso():
    sources = SourceList([Source("circle", [2, -4, -5, 2]), Source("circle", [4, 5, 4, 3])])
    sources_orig = deepcopy(sources)
    uu = generate_uv(33 * apu.pixel)
    u, v = np.meshgrid(uu, uu)
    vis = sources_to_vis(sources, u.value, v.value)
    visobs = Visibilities(vis.flatten() * apu.ph, u.flatten(), v.flatten())
    sources_fit, res = _vis_forward_fit_minimise(visobs, sources, method="PSO")

    # the order of the sources isn't necessarily going to match so sort based on x,y location?
    sources_orig = SourceList(sorted(sources_orig.sources, key=lambda s: s.params[1:2]))
    sources_fit = SourceList(sorted(sources_fit.sources, key=lambda s: s.params[1:2]))
    assert_allclose(sources_fit.params, sources_orig.params, atol=1e-4, rtol=1e-5)
