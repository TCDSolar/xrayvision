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
    loop,
    loop_vis,
)
from xrayvision.visibility import Visibilities


@pytest.mark.parametrize("x0", [0, -10])
@pytest.mark.parametrize("y0", [0, 6])
@pytest.mark.parametrize("sigma", [3])
@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("shape", [63])
def test_circular_ft_equivalence_fft(x0, y0, sigma, size, shape):
    xx = generate_xy(shape * apu.pix, pixel_size=size * apu.arcsec / apu.pixel, phase_center=0 * apu.arcsec)
    yy = generate_xy(shape * apu.pix, pixel_size=size * apu.arcsec / apu.pixel, phase_center=0 * apu.arcsec)
    x, y = np.meshgrid(xx, yy)
    # by definition map center on phase 0,0
    uu = generate_uv(shape * apu.pix, pixel_size=size * apu.arcsec / apu.pixel, phase_center=0 * apu.arcsec)
    vv = generate_uv(shape * apu.pix, pixel_size=size * apu.arcsec / apu.pixel, phase_center=0 * apu.arcsec)
    u, v = np.meshgrid(uu, vv)
    u = u.flatten()
    v = v.flatten()

    image = circular_gaussian(1, x, y, x0 * apu.arcsec, y0 * apu.arcsec, sigma * size * apu.arcsec)

    vis_obs = image_to_vis(image, u=u, v=v, pixel_size=[size, size] * apu.arcsec / apu.pixel)
    # by definition map center on phase 0,0
    vis_func = Visibilities(
        circular_gaussian_vis(1, u, v, x0 * apu.arcsec, y0 * apu.arcsec, sigma * size * apu.arcsec).flatten(), u, v
    )

    image_func = vis_to_image(vis_func, [shape, shape] * apu.pixel, pixel_size=[size, size] * apu.arcsec / apu.pixel)
    image_vis = vis_to_image(vis_obs, [shape, shape] * apu.pixel, pixel_size=[size, size] * apu.arcsec / apu.pixel)

    assert_allclose(vis_obs.visibilities.value, vis_func.visibilities, atol=1e-8)
    assert_allclose(image.value, image_func.value, atol=1e-9)
    assert_allclose(image.value, image_vis.value, atol=1e-9)


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


@pytest.mark.parametrize("size", (65, 79))
def test_loop_ft_equivalence_fft(size):
    # So unless the array is sufficiently large this test fails
    # I think has to do with the fact no taking into account the sampleing and implicit windowing
    # TODO: How does this affect algo where the vis derived from map are compare to the observed?
    # sigma = 4 * apu.arcsec
    xx = generate_xy(size * apu.pix)
    x, y = np.meshgrid(xx, xx)
    uu = generate_uv(size * apu.pix)
    u, v = np.meshgrid(uu, uu)
    u = u.flatten()
    v = v.flatten()

    image = loop(
        80,
        x,
        y,
        0 * apu.arcsec,
        0 * apu.arcsec,
        22.5 * apu.arcsec,
        9.0 * apu.arcsec,
        np.deg2rad(90),
        np.deg2rad(70),
    )

    vis_obs = image_to_vis(image * apu.ph, u=u, v=v)
    vis_func = Visibilities(
        loop_vis(
            80,
            u,
            v,
            0 * apu.arcsec,
            0 * apu.arcsec,
            22.5 * apu.arcsec,
            9.0 * apu.arcsec,
            np.deg2rad(90),
            np.deg2rad(70),
        ).flatten()
        * apu.ph,
        u,
        v,
    )

    image_func = vis_to_image(vis_func, [size, size] * apu.pixel)
    image_vis = vis_to_image(vis_obs, [size, size] * apu.pixel)

    assert_allclose(vis_func.visibilities, vis_obs.visibilities, atol=1e-9)
    assert_allclose(image_func, image_vis, atol=1e-9)


# def test_model_equivalence_fft():
#     uu = generate_uv(65 * apu.pix)
#     u, v = np.meshgrid(uu, uu)
#     u = u.flatten()
#     v = v.flatten()
#
#     image = model_img(np.atleast_2d([1, 2, 0, 0, 10, -90, 0.1]).T, 65, 65, 1)
#     vis_obs = image_to_vis(image * apu.ph, u=u, v=v)
#
#     vis_func = Visibilities(
#         model_vis(np.atleast_2d([1, 2, 0, 0, 10, -90, 0.1]).T, 65, 65, 1).flatten() * apu.ph, u, v,
#     )
#
#     image_func = vis_to_image(vis_func, [65, 65] * apu.pixel)
#     image_vis = vis_to_image(vis_obs, [65, 65] * apu.pixel)
#
#     assert_allclose(vis_func.visibilities, vis_obs.visibilities, atol=1e-9)
#     assert_allclose(image_func, image_vis, atol=1e-9)


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
