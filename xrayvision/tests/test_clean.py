import astropy.units as u
import numpy as np
from astropy.convolution.kernels import Gaussian2DKernel
from numpy.testing import assert_allclose
from scipy import signal

from xrayvision.clean import (
    _component,
    _radial_prolate_sphereoidal,
    _vec_radial_prolate_sphereoidal,
    clean,
    ms_clean,
    vis_clean,
)
from xrayvision.imaging import image_to_vis
from xrayvision.transform import dft_map, idft_map


def test_clean_ideal():
    n = m = 65
    pos1 = [15, 30]
    pos2 = [40, 32]

    clean_map = np.zeros((n, m))
    clean_map[pos1[0], pos1[1]] = 10.0
    clean_map[pos2[0], pos2[1]] = 7.0
    clean_map = clean_map  # << 1 / u.arcsec**2

    dirty_beam = np.zeros((n, m))
    dirty_beam[(n - 1) // 4 : (n - 1) // 4 + (n - 1) // 2, (m - 1) // 2] = 0.75
    dirty_beam[
        (n - 1) // 2,
        (m - 1) // 4 : (m - 1) // 4 + (m - 1) // 2,
    ] = 0.75
    dirty_beam[(n - 1) // 2, (m - 1) // 2] = 0.8
    dirty_beam = np.pad(dirty_beam, (65, 65), "constant")

    dirty_map = signal.convolve(clean_map, dirty_beam, mode="same")

    # Disable convolution of model with gaussian for testing
    out_map, model, resid = clean(dirty_map, dirty_beam, clean_beam_width=None)

    # Within threshold default threshold of 0.1
    assert_allclose(out_map, clean_map, atol=dirty_beam.max() * 1e-13)


def test_component():
    comp = np.zeros((3, 3))
    comp[1, 1] = 1.0

    res = _component(scale=0, shape=(3, 3))
    assert np.array_equal(res, comp)

    res = _component(scale=1, shape=(3, 3))
    assert np.array_equal(res, comp)

    res = _component(scale=2, shape=(6, 6))
    assert np.all(res[0, :] == 0.0)
    assert np.all(res[:, 0] == 0.0)
    assert np.all(res[2:4, 2:4] == res.max())

    res = _component(scale=3, shape=(7, 7))
    assert np.all(res[0, :] == 0.0)
    assert np.all(res[:, 0] == 0.0)
    assert res[3, 3] == 1


def test_radial_prolate_spheroidal():
    amps = [_radial_prolate_sphereoidal(r) for r in [-1.0, 0.0, 0.5, 1.0, 2.0]]
    assert amps[0] == 1.0
    assert amps[1] == 1.0
    assert amps[2] == 0.36106538453111797
    assert amps[3] == 0.0
    assert amps[4] == 0.0


def test_vec_radial_prolate_spheroidal():
    radii = np.linspace(-0.5, 1.5, 1000)
    amps1 = [_radial_prolate_sphereoidal(r) for r in radii]
    amps2 = _vec_radial_prolate_sphereoidal(radii)
    assert np.allclose(amps1, amps2)


def test_ms_clean_ideal():
    n = m = 65
    pos1 = [15, 30]
    pos2 = [40, 32]

    clean_map = np.zeros((n, m))
    clean_map[pos1[0], pos1[1]] = 10.0
    clean_map[pos2[0], pos2[1]] = 7.0

    dirty_beam = np.zeros((n, m))
    dirty_beam[(n - 1) // 4 : (n - 1) // 4 + (n - 1) // 2, (m - 1) // 2] = 0.75
    dirty_beam[
        (n - 1) // 2,
        (m - 1) // 4 : (m - 1) // 4 + (m - 1) // 2,
    ] = 0.75
    dirty_beam[(n - 1) // 2, (m - 1) // 2] = 1.0
    dirty_beam = np.pad(dirty_beam, (65, 65), "constant")

    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    # Disable convolution of model with gaussian for testing
    model, res = ms_clean(
        dirty_map, dirty_beam, pixel_size=[1, 1] * u.arcsec / u.pixel, scales=[1], clean_beam_width=None
    )
    recovered = model + res

    # Within threshold default threshold
    assert np.allclose(clean_map, recovered, atol=dirty_beam.max() * 0.1)


# @pytest.mark.skip(reason="Broken test")
def test_clean_sim():
    n = m = 31
    data = Gaussian2DKernel(3.0, x_size=n, y_size=m).array

    half_log_space = np.logspace(np.log10(0.03030303), np.log10(0.48484848), 10)

    theta = np.linspace(0, 2 * np.pi, 32)
    theta = theta[np.newaxis, :]
    theta = np.repeat(theta, 10, axis=0)

    r = half_log_space
    r = r[:, np.newaxis]
    r = np.repeat(r, 32, axis=1)

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    sub_uv = np.vstack([x.flatten(), y.flatten()])
    sub_uv = np.hstack([sub_uv, np.zeros((2, 1))]) / u.arcsec

    # Factor of 9 is compensate for the factor of  3 * 3 increase in size
    dirty_beam = idft_map(np.ones(321) / 321, u=sub_uv[0, :], v=sub_uv[1, :], shape=(n * 3 + 1, m * 3 + 1) * u.pix)

    vis = dft_map(data, u=sub_uv[0, :], v=sub_uv[1, :])

    dirty_map = idft_map(vis, weights=1 / 321, u=sub_uv[0, :], v=sub_uv[1, :], shape=(n, m) * u.pix)

    clean_map, model, res = clean(
        dirty_map, dirty_beam, pixel_size=[2, 2] * u.arcsec / u.pix, clean_beam_width=0.1 * u.arcsec, niter=500
    )
    assert_allclose(clean_map, data, atol=dirty_beam.max() * 0.1)


def test_vis_clean_sim():
    n = m = 31
    data = Gaussian2DKernel(3.0, x_size=n, y_size=m).array

    half_log_space = np.logspace(np.log10(0.03030303), np.log10(0.48484848), 10)

    theta = np.linspace(0, 2 * np.pi, 32)
    theta = theta[np.newaxis, :]
    theta = np.repeat(theta, 10, axis=0)

    r = half_log_space
    r = r[:, np.newaxis]
    r = np.repeat(r, 32, axis=1)

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    sub_uv = np.vstack([x.flatten(), y.flatten()])
    sub_uv = np.hstack([sub_uv, np.zeros((2, 1))]) / u.arcsec

    vis = image_to_vis(data * u.dimensionless_unscaled, u=sub_uv[0, :], v=sub_uv[1, :])

    clean_map, model, res = vis_clean(
        vis,
        shape=(m, n) * u.pix,
        pixel_size=[2, 2] * u.arcsec / u.pix,
        clean_beam_width=None,
        niter=100,
        scheme="uniform",
    )
    np.allclose(data, clean_map.data, atol=0.1)
