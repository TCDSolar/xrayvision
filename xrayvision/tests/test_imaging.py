import numpy as np
import pytest
from sunpy.map import Map

import astropy.units as apu
from astropy.convolution.kernels import Gaussian2DKernel

from xrayvision.imaging import image_to_vis, map_to_vis, vis_psf_image, vis_to_image, vis_to_map
from xrayvision.transform import dft_map, generate_uv, idft_map
from xrayvision.visibility import Visibility


@pytest.fixture
def uv():
    half_log_space = np.logspace(np.log10(0.03030303), np.log10(0.48484848), 10)

    theta = np.linspace(0, 2*np.pi, 32)
    theta = theta[np.newaxis, :]
    theta = np.repeat(theta, 10, axis=0)

    r = half_log_space
    r = r[:, np.newaxis]
    r = np.repeat(r, 32, axis=1)

    u = r * np.sin(theta)
    v = r * np.cos(theta)

    return u.flatten()/apu.arcsec, v.flatten()/apu.arcsec


@pytest.mark.parametrize('pixel_size', [(0.5), (1), (2)])
def test_psf_to_image(pixel_size, uv):
    u, v = uv
    img = np.zeros((65, 65))*(apu.ph/apu.arcsec**2)
    img[32, 32] = 1.0*(apu.ph/apu.arcsec**2)
    obs_vis = dft_map(img, u=u, v=v, pixel_size=[2., 2.]*apu.arcsec)
    weights = np.sqrt(u**2 + v**2).value
    weights /= weights.sum()
    psf_calc = idft_map(obs_vis, u=u, v=v, shape=[65, 65], pixel_size=[2, 2]*apu.arcsec,
                        weights=weights)
    vis = Visibility(obs_vis, u=u, v=v)
    res = vis_psf_image(vis, shape=[65, 65]*apu.pixel, pixel_size=2*apu.arcsec)
    assert np.allclose(psf_calc, res)


def test_vis_to_image(uv):
    u, v = uv
    img = np.zeros((65, 65)) * (apu.ph / apu.arcsec ** 2)
    img[32, 32] = 1.0 * (apu.ph / apu.arcsec ** 2)
    obs_vis = dft_map(img, u=u, v=v, pixel_size=[2., 2.] * apu.arcsec)
    weights = np.sqrt(u**2 + v**2).value
    weights /= weights.sum()
    bp_calc = idft_map(obs_vis, u=u, v=v, shape=[65, 65], pixel_size=[2, 2] * apu.arcsec,
                       weights=weights)
    vis = Visibility(obs_vis, u=u, v=v)
    res = vis_to_image(vis, shape=[65, 65] * apu.pixel, pixel_size=2 * apu.arcsec)
    assert np.allclose(bp_calc, res)


def test_image_to_vis():
    m = n = 33
    size = m * n
    # Set up empty map
    image = np.zeros((m, n))

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m)
    v = generate_uv(n)
    u, v = np.meshgrid(u, v)
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    # For an empty map visibilities should all be zero (0+0j)
    empty_vis = image_to_vis(image, u=v, v=v)
    assert np.array_equal(empty_vis.center, (0.0, 0.0) * apu.arcsec)
    assert np.array_equal(empty_vis.vis, np.zeros(n*m, dtype=complex))


def test_image_to_vis_center():
    m = n = 33
    size = m * n
    # Set up empty map
    image = np.zeros((m, n))

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m)
    v = generate_uv(n)
    u, v = np.meshgrid(u, v)
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    # For an empty map visibilities should all be zero (0+0j)
    empty_vis = image_to_vis(image, u=u, v=v, center=(2.0, -3.0) * apu.arcsec)
    assert np.array_equal(empty_vis.center,  (2.0, -3.0) * apu.arcsec)
    assert np.array_equal(empty_vis.vis, np.zeros(n * m, dtype=complex))


@pytest.mark.parametrize("pos,pixel", [((0.0, 0.0), (1.0, 1.0)),
                                       ((-12.0, 19.0), (2., 2.)),
                                       ((12.0, -19.0), (1., 5.)),
                                       ((0.0, 0.0), (1.0, 5.0))])
def test_map_to_vis(pos, pixel):
    m = n = 33
    size = m * n

    pos = pos * apu.arcsec
    pixel = pixel * apu.arcsec

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m, pos[0])
    v = generate_uv(n, pos[1])
    u, v = np.meshgrid(u, v)
    u, v = np.array([u, v]).reshape(2, size) / apu.arcsec

    header = {'crval1': pos[0].value, 'crval2': pos[1].value,
              'cdelt1': pixel[0].value, 'cdelt2': pixel[1].value,
              'cunit1': 'arcsec', 'cunit2': 'arcsec'}

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array
    mp = Map((data, header))
    vis = map_to_vis(mp, u=u, v=v)

    assert np.array_equal(vis.center, pos)

    res = vis_to_image(vis, shape=(m, n)*apu.pixel, pixel_size=pixel, natural=False)
    assert np.allclose(res, data)


def test_vis_to_image():
    m = n = 33
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m)
    v = generate_uv(n)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array

    vis = image_to_vis(data, u=u, v=v)
    res = vis_to_image(vis, shape=(m, n)*apu.pixel, natural=False)
    assert np.allclose(data, res)
    assert res.shape == (m, n)


def test_vis_to_image_single_pixel_size():
    m = n = 32
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m, pixel_size=2. * apu.arcsec)
    v = generate_uv(n, pixel_size=2. * apu.arcsec)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv
    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array

    vis = image_to_vis(data, u=u, v=v, pixel_size=(2., 2.) * apu.arcsec)
    res = vis_to_image(vis, shape=(m, n)*apu.pixel, pixel_size=2. * apu.arcsec, natural=False)
    assert res.shape == (m, n)
    assert np.allclose(data, res)


def test_vis_to_image_invalid_pixel_size():
    m = n = 32
    size = m * n

    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    u = generate_uv(m)
    v = generate_uv(n)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, size) / apu.arcsec
    u, v = uv

    # Astropy index order is opposite to that of numpy, is 1st dim is across second down
    data = Gaussian2DKernel(6, x_size=n, y_size=m).array

    vis = image_to_vis(data, u=u, v=v)
    with pytest.raises(ValueError):
        vis_to_image(vis, shape=(m, n)*apu.pixel, pixel_size=[1, 2, 2] * apu.arcsec)


@pytest.mark.parametrize("m,n,pos,pixel", [(33, 33, (10., -5.), (2., 3.)),
                                           (32, 32, (-12, -19), (1., 5.))])
def test_vis_to_map(m, n, pos, pixel):
    pos = pos * apu.arcsec
    pixel = pixel * apu.arcsec
    u = generate_uv(m, pixel[0])
    v = generate_uv(n, pixel[1])
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {'crval1': pos[0].value, 'crval2': pos[1].value,
              'cdelt1': pixel[0].value, 'cdelt2': pixel[1].value,
              'cunit1': 'arcsec', 'cunit2': 'arcsec'}
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)

    res = vis_to_map(vis, shape=(m, n)*apu.pixel, pixel_size=pixel, natural=False)
    # assert np.allclose(res.data, data)

    assert res.reference_coordinate.Tx == pos[0]
    assert res.reference_coordinate.Ty == pos[1]
    assert res.scale.axis1 == pixel[0] / apu.pix
    assert res.scale.axis2 == pixel[1] / apu.pix
    assert res.dimensions.x == m * apu.pix
    assert res.dimensions.y == n * apu.pix


def test_to_sunpy_single_pixel_size():
    m = n = 32
    u = generate_uv(m, pixel_size=2. * apu.arcsec)
    v = generate_uv(m, pixel_size=2. * apu.arcsec)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {'crval1': 0, 'crval2': 0,
              'cdelt1': 2, 'cdelt2': 2,
              'cunit1': 'arcsec', 'cunit2': 'arcsec'}
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)
    res = vis_to_map(vis, shape=(m, n)*apu.pixel, pixel_size=2 * apu.arcsec, natural=False)
    assert res.meta['cdelt1'] == 2.
    assert res.meta['cdelt1'] == 2.
    assert np.allclose(data, res.data)


def test_to_sunpy_map_invalid_pixel_size():
    m = n = 32
    u = generate_uv(m)
    v = generate_uv(m)
    u, v = np.meshgrid(u, v)
    uv = np.array([u, v]).reshape(2, m * n) / apu.arcsec
    u, v = uv

    header = {'crval1': 0, 'crval2': 0,
              'cdelt1': 1, 'cdelt2': 1,
              'cunit1': 'arcsec', 'cunit2': 'arcsec'}
    data = Gaussian2DKernel(2, x_size=n, y_size=m).array
    mp = Map((data, header))

    vis = map_to_vis(mp, u=u, v=v)

    with pytest.raises(ValueError):
        vis_to_map(vis, shape=(m, n)*apu.pixel, pixel_size=[1, 2, 3] * apu.arcsec)
