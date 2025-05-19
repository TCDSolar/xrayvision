import astropy.units as apu
import numpy as np
import pytest
from numpy.testing import assert_allclose

from xrayvision.imaging import image_to_vis, vis_to_image
from xrayvision.transform import generate_uv, generate_xy
from xrayvision.vis_forward_fit.sources import (
    Circular,
    Elliptical,
    Source,
    SourceList,
    circular_gaussian,
    circular_gaussian_vis,
    elliptical_gaussian,
    elliptical_gaussian_vis,
)
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


def test_source_factory():
    circular = Source("circular", 1, 2, 3, 4)
    assert isinstance(circular, Circular)
    assert circular.amp == 1
    assert circular.x0 == 2
    assert circular.y0 == 3
    assert circular.sigma == 4

    elliptical = Source("elliptical", 1, 2, 3, 4, 5, 6)
    assert isinstance(elliptical, Elliptical)
    assert elliptical.amp == 1
    assert elliptical.x0 == 2
    assert elliptical.y0 == 3
    assert elliptical.sigmax == 4
    assert elliptical.sigmay == 5
    assert elliptical.theta == 6


def test_source_list():
    orig_sources = SourceList([Source("circular", 1, 2, 3, 4), Source("elliptical", 1, 2, 3, 4, 5, 6)])
    params = orig_sources.params
    assert params == [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]
    new_sources = SourceList.from_params(orig_sources, params)
    assert orig_sources == new_sources
