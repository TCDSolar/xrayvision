import astropy.units as apu
import numpy as np
import pytest
from astropy.convolution.kernels import Gaussian2DKernel
from numpy.testing import assert_allclose
from sunpy.map import Map

from xrayvision.imaging import image_to_vis, map_to_vis, vis_psf_image, vis_to_image, vis_to_map
from xrayvision.transform import dft_map, generate_uv, idft_map
from xrayvision.visibility import Visibilities


@pytest.fixture
def uv():
    half_log_space = np.logspace(np.log10(0.03030303), np.log10(0.48484848), 10)

    theta = np.linspace(0, 2 * np.pi, 32)
    theta = theta[np.newaxis, :]
    theta = np.repeat(theta, 10, axis=0)

    r = half_log_space
    r = r[:, np.newaxis]
    r = np.repeat(r, 32, axis=1)

    u = r * np.sin(theta)
    v = r * np.cos(theta)

    return u.flatten() / apu.arcsec, v.flatten() / apu.arcsec


@pytest.mark.parametrize("shape", [(10, 10), (5, 5), (2, 2)])
def test_image_to_vis_vis_to_image(shape):
    # Images with the same total flux
    extent = [10, 10] * apu.arcsec
    arr1 = np.full((shape), 1) * apu.ct / np.prod(shape)
    ps1 = extent / (shape * apu.pix)
    img1 = arr1 / (ps1[0] * ps1[1] * apu.pix**2)

    # flux in image is 1
    assert_allclose(img1.sum() * ps1[0] * apu.pix * ps1[1] * apu.pix, 1 * apu.ct)

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv((shape[0]) * apu.pix, pixel_size=ps1[0])
    v = generate_uv((shape[1]) * apu.pix, pixel_size=ps1[1])
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = np.array([u, v]).reshape(2, np.prod(shape)) / apu.arcsec

    vis1 = image_to_vis(img1, u=u, v=v, pixel_size=ps1)
    res1 = vis_to_image(vis1, shape=shape * apu.pix, pixel_size=ps1)
    assert_allclose(res1, img1)


@pytest.mark.skip(reason="WIP")
def test_vis_to_image_conserve():
    shape = (3, 3)
    shape1 = (5, 5)
    shape2 = (7, 7)
    # Images with the same total flux
    extent = [10, 10] * apu.arcsec
    arr1 = np.zeros(shape)
    arr1[shape[0] // 2, shape[1] // 2] = 1 / np.prod(shape)
    ps = extent / (shape * apu.pix)
    img1 = arr1 / (ps[0] * ps[1] * apu.pix**2)

    # flux in image is same
    assert_allclose(img1.sum() * ps[0] * apu.pix * ps[1] * apu.pix, arr1.sum())

    u = generate_uv(shape[0] * apu.pix, pixel_size=ps[0])
    v = generate_uv(shape[1] * apu.pix, pixel_size=ps[1])
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = np.array([u, v]).reshape(2, np.prod(shape)) / apu.arcsec

    vis1 = image_to_vis(img1, u=u, v=v, pixel_size=ps)

    ps2 = extent / (shape1 * apu.pix)
    ps3 = extent / (shape2 * apu.pix)
    res = vis_to_image(vis1, shape=shape * apu.pix, pixel_size=ps)
    res2 = vis_to_image(vis1, shape=shape1 * apu.pix, pixel_size=ps2)
    res3 = vis_to_image(vis1, shape=shape2 * apu.pix, pixel_size=ps3)

    assert_allclose(res, img1)
    assert_allclose(res2, img1)
    assert_allclose(res3, img1)


@pytest.mark.parametrize("pixel_size", [(0.5), (1), (2)])
def test_vis_to_psf(pixel_size, uv):
    u, v = uv
    ps = [pixel_size, pixel_size] * apu.arcsec / apu.pix
    img = np.zeros((65, 65)) * (apu.ph / apu.cm**2)
    img[32, 32] = 1 * (apu.ph / apu.cm**2)  # pixel size of 4 -> 1/4
    obs_vis = dft_map(img, u=u, v=v, pixel_size=ps)
    weights = np.sqrt(u**2 + v**2).value
    weights /= weights.sum()
    psf_calc = idft_map(obs_vis, u=u, v=v, shape=[65, 65] * apu.pix, pixel_size=ps, weights=weights)
    vis = Visibilities(obs_vis, u, v)
    res = vis_psf_image(vis, shape=[65, 65] * apu.pixel, pixel_size=ps, scheme="uniform")
    assert_allclose(res, psf_calc)


def test_vis_to_image_against_idft(uv):
    u, v = uv
    img = np.zeros((65, 65)) * (apu.ph / apu.cm**2)
    img[32, 32] = 1.0 * (apu.ph / apu.cm**2)
    obs_vis = dft_map(img, u=u, v=v, pixel_size=[2.0, 2.0] * apu.arcsec / apu.pix)
    weights = np.sqrt(u**2 + v**2).value
    weights /= weights.sum()
    bp_calc = idft_map(
        obs_vis, u=u, v=v, shape=[65, 65] * apu.pix, pixel_size=[2, 2] * apu.arcsec / apu.pix, weights=weights
    )
    vis = Visibilities(obs_vis, u, v)
    res = vis_to_image(vis, shape=[65, 65] * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix, scheme="uniform")
    assert np.allclose(bp_calc, res)


def test_image_to_vis():
    m = n = 33
    size = m * n
    # Set up empty map
    image = np.zeros((m, n)) * apu.ct / apu.arcsec**2

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix)
    v = generate_uv(n * apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    # For an empty map visibilities should all be zero (0+0j)
    empty_vis = image_to_vis(image, u=v, v=v)
    assert np.array_equal(empty_vis.phase_center, (0.0, 0.0) * apu.arcsec)
    assert np.array_equal(empty_vis.visibilities, np.zeros(n * m, dtype=complex))


def test_image_to_vis_center():
    m = n = 33
    size = m * n
    # Set up empty map
    image = np.zeros((m, n)) * apu.ct / apu.arcsec**2

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix)
    v = generate_uv(n * apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    # For an empty map visibilities should all be zero (0+0j)
    empty_vis = image_to_vis(image, u=u, v=v, phase_center=(2.0, -3.0) * apu.arcsec)
    assert np.array_equal(empty_vis.phase_center, (2.0, -3.0) * apu.arcsec)
    assert np.array_equal(empty_vis.visibilities, np.zeros(n * m, dtype=complex))


@pytest.mark.parametrize(
    "pos,pixel",
    [((0.0, 0.0), (1.0, 1.0)), ((-12.0, 19.0), (2.0, 2.0)), ((12.0, -19.0), (1.0, 5.0)), ((0.0, 0.0), (1.0, 5.0))],
)
def test_map_to_vis(pos, pixel):
    m = n = 33
    size = m * n

    pos = pos * apu.arcsec
    pixel = pixel * apu.arcsec / apu.pix

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix, phase_center=pos[0], pixel_size=pixel[0])
    v = generate_uv(n * apu.pix, phase_center=pos[1], pixel_size=pixel[1])
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    header = {
        "crval1": pos[1].value,
        "crval2": pos[0].value,
        "cdelt1": pixel[1].value,
        "cdelt2": pixel[0].value,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "bunit": "ct / arcsec^2",
    }

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array * apu.ct / apu.arcsec**2
    mp = Map((data, header))
    vis = map_to_vis(mp, u=u, v=v)

    assert np.array_equal(vis.phase_center, pos)

    res = vis_to_image(vis, shape=(m, n) * apu.pixel, pixel_size=pixel)
    assert np.allclose(res, data)


def test_vis_to_image():
    m = n = 30
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix)
    v = generate_uv(n * apu.pix)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array * apu.ct / apu.arcsec**2

    vis = image_to_vis(data, u=u, v=v)
    res = vis_to_image(vis, shape=(m, n) * apu.pixel)
    assert np.allclose(data, res)
    assert res.shape == (m, n)


def test_vis_to_image_single_pixel_size():
    m = n = 32
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix, pixel_size=2.0 * apu.arcsec / apu.pix)
    v = generate_uv(n * apu.pix, pixel_size=2.0 * apu.arcsec / apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv
    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array * apu.ct / apu.arcsec**2

    vis = image_to_vis(data, u=u, v=v, pixel_size=(2.0, 2.0) * apu.arcsec / apu.pix)
    res = vis_to_image(vis, shape=(m, n) * apu.pixel, pixel_size=2.0 * apu.arcsec / apu.pix)
    assert res.shape == (m, n)
    assert np.allclose(data, res)


def test_vis_to_image_invalid_pixel_size():
    m = n = 32
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m * apu.pix)
    v = generate_uv(n * apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array * apu.ct / apu.arcsec**2

    vis = image_to_vis(data, u=u, v=v)
    with pytest.raises(ValueError):
        vis_to_image(vis, shape=(m, n) * apu.pixel, pixel_size=[1, 2, 2] * apu.arcsec / apu.pix)


@pytest.mark.parametrize(
    "m,n,pos,pixel",
    [  # (33, 33, (0, 0), (1.5, 1.5)),
        (33, 33, (10.0, -5.0), (2.0, 3.0)),
        (32, 32, (-12, -19), (1.0, 5.0)),
    ],
)
def test_vis_to_map(m, n, pos, pixel):
    pos = pos * apu.arcsec
    pixel = pixel * apu.arcsec / apu.pix
    u = generate_uv(m * apu.pix, phase_center=pos[0], pixel_size=pixel[0])
    v = generate_uv(n * apu.pix, phase_center=pos[1], pixel_size=pixel[1])
    u, v = np.meshgrid(u, v, indexing="ij")
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {
        "crval1": pos[1].value,
        "crval2": pos[0].value,
        "cdelt1": pixel[1].value,
        "cdelt2": pixel[0].value,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
    }
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)

    res = vis_to_map(vis, shape=(m, n) * apu.pixel, pixel_size=pixel, scheme="natural")
    # TODO: figure out why atol is needed here
    assert_allclose(data, res.data, atol=1e-15)

    assert res.reference_coordinate.Tx == pos[1]
    assert res.reference_coordinate.Ty == pos[0]
    assert res.scale.axis1 == pixel[1]
    assert res.scale.axis2 == pixel[0]
    assert res.dimensions.x == m * apu.pix
    assert res.dimensions.y == n * apu.pix


def test_to_sunpy_single_pixel_size():
    m = n = 32
    u = generate_uv(m * apu.pix, pixel_size=2.0 * apu.arcsec / apu.pix)
    v = generate_uv(m * apu.pix, pixel_size=2.0 * apu.arcsec / apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {"crval1": 0, "crval2": 0, "cdelt1": 2, "cdelt2": 2, "cunit1": "arcsec", "cunit2": "arcsec"}
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)
    res = vis_to_map(vis, shape=(m, n) * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix)
    assert res.meta["cdelt1"] == 2.0
    assert res.meta["cdelt1"] == 2.0
    assert np.allclose(data, res.data)


def test_to_sunpy_map_invalid_pixel_size():
    m = n = 32
    u = generate_uv(m * apu.pix)
    v = generate_uv(n * apu.pix)
    u, v = np.meshgrid(u, v, indexing="ij")
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {"crval1": 0, "crval2": 0, "cdelt1": 1, "cdelt2": 1, "cunit1": "arcsec", "cunit2": "arcsec"}
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)

    with pytest.raises(ValueError):
        vis_to_map(vis, shape=(m, n) * apu.pixel, pixel_size=[1, 2, 3] * apu.arcsec / apu.pix)
