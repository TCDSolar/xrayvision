import astropy.units as u
import numpy as np
import pytest

from xrayvision.imaging import psf, back_project
from xrayvision.transform import idft_map, dft_map, generate_uv
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

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    sub_uv = np.vstack([x.flatten(), y.flatten()])
    return sub_uv/u.arcsec


@pytest.mark.parametrize('pixel_size', [(0.5), (1), (2)])
def test_psf(pixel_size, uv):
    img = np.zeros((65, 65))*(u.ph/u.arcsec**2)
    img[32, 32] = 1.0*(u.ph/u.arcsec**2)
    obs_vis = dft_map(img, uv, pixel_size=[2., 2.]*u.arcsec)
    weights = np.sqrt(uv[0, :]**2 + uv[1, :]**2).value
    weights /= weights.sum()
    psf_calc = idft_map(uv, obs_vis, shape=[65, 65], pixel_size=[2,2]*u.arcsec,
                        weights=weights)
    vis = Visibility(uv, obs_vis)
    res = psf(vis, shape=[65, 65]*u.pixel, pixel_size=2*u.arcsec, map=False)
    assert np.allclose(psf_calc, res)


def test_back_project(uv):
    img = np.zeros((65, 65)) * (u.ph / u.arcsec ** 2)
    img[32, 32] = 1.0 * (u.ph / u.arcsec ** 2)
    obs_vis = dft_map(img, uv, pixel_size=[2., 2.] * u.arcsec)
    weights = np.sqrt(uv[0, :] ** 2 + uv[1, :] ** 2).value
    weights /= weights.sum()
    bp_calc = idft_map(uv, obs_vis, shape=[65, 65], pixel_size=[2, 2] * u.arcsec,
                        weights=weights)
    vis = Visibility(uv, obs_vis)
    res = back_project(vis, shape=[65, 65] * u.pixel, pixel_size=2 * u.arcsec, map=False)
    assert np.allclose(bp_calc, res)
