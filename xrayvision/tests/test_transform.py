import numpy as np
import pytest
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from scipy import signal


from ..transform import generate_xy, generate_uv, dft_map, idft_map


@pytest.mark.parametrize("pixel_size", [0.5, 1, 2, 3])
def test_generate_xy_pixel_size(pixel_size):
    # Odd
    pixel_size = pixel_size * u.arcsec
    odd = np.linspace(-16 * pixel_size, 16 * pixel_size, 33)
    assert np.array_equal(odd, generate_xy(33, pixel_size=pixel_size))

    # Even
    even = np.linspace(-15.5 * pixel_size, 15.5 * pixel_size, 32)
    assert np.array_equal(even, generate_xy(32, pixel_size=pixel_size))


@pytest.mark.parametrize("center", [0, +5.5, -5.5])
def test_generate_xy_offset(center):
    # Odd
    center = center * u.arcsec
    odd = np.linspace(-16, 16, 33) * u.arcsec + center
    assert np.array_equal(odd, generate_xy(33, center=center))

    # Even
    even = np.linspace(-15.5, 15.5, 32) * u.arcsec + center
    assert np.array_equal(even, generate_xy(32, center=center))


@pytest.mark.parametrize("center, pixel_size", [(0, (0.5, 1, 2, 3)),
                                                (+5.5, (0.5, 1, 2, 3)),
                                                (-5.5, (0.5, 1, 2, 3))])
def test_generate_xy_offset_size(center, pixel_size):
    center = center * u.arcsec
    pixel_size = pixel_size * u.arcsec
    # No sure if this is a good idea could be hard to debug
    for size in pixel_size:
        # Odd
        odd = np.linspace(-16 * size, 16 * size, 33) + center
        assert np.array_equal(odd, generate_xy(33, center=center, pixel_size=size))

        # Even
        even = np.linspace(-15.5 * size, 15.5 * size, 32) + center
        assert np.array_equal(even, generate_xy(32, center=center, pixel_size=size))


@pytest.mark.parametrize("pixel_size", [0.5, 1, 2, 3])
def test_generate_uv_pixel_size(pixel_size):
    pixel_size = pixel_size * u.arcsec
    m = 33
    n = 32

    # Odd
    odd = np.linspace(-((m-1)/2) * (1/(m * pixel_size)), ((m-1)/2) * (1/(m * pixel_size)), m)

    # Know issue with quantities and isclose/allclose need to add unit to atol default value
    assert np.allclose(odd, generate_uv(m, pixel_size=pixel_size), atol=1e-8/u.arcsec)

    # Even
    even = (np.arange(n) - n / 2 + 0.5) * (1 / (pixel_size * n))
    assert np.allclose(even, generate_uv(32, pixel_size=pixel_size), atol=1e-8 / u.arcsec)


@pytest.mark.parametrize("center", [0.0, -5.5, 5.5])
def test_generate_uv_pixel_offset(center):
    center = center * u.arcsec
    m = 33
    n = 32

    # Odd
    odd = np.linspace(-((m-1)/2) * (1/m), ((m-1)/2) * (1/m), m) / u.arcsec
    if center.value != 0:
        odd += 1/center
    assert np.allclose(odd, generate_uv(m, center=center), atol=1e-8 / u.arcsec)

    # Even
    even = (np.arange(n) - n / 2 + 0.5) * (1 / n) / u.arcsec
    if center.value != 0:
        even += 1/center
    assert np.allclose(even, generate_uv(32, center=center), atol=1e-8 / u.arcsec)


@pytest.mark.parametrize("center, pixel_size", [(0, (0.5, 1, 2, 3)),
                                                (+5.5, (0.5, 1, 2, 3)),
                                                (-5.5, (0.5, 1, 2, 3))])
def test_generate_uv_offset_size(center, pixel_size):
    center = center * u.arcsec
    pixel_size = pixel_size * u.arcsec
    m = 33
    n = 32

    for size in pixel_size:
        # Odd
        odd = np.linspace(-((m - 1) / 2) * (1 / (size * m)), ((m - 1) / 2) * (1 / (size * m)), m)
        if center != 0:
            odd += 1 / center
        assert np.allclose(odd, generate_uv(m, center=center, pixel_size=size),
                           atol=1e-8 / u.arcsec)

        # Even
        even = (np.arange(n) - n / 2 + 0.5) * (1 / (size * n))
        if center != 0:
            even += 1 / center
        assert np.allclose(even, generate_uv(32, center=center, pixel_size=size),
                           atol=1e-8 / u.arcsec)


@pytest.mark.parametrize("shape", [(33, 33), (32, 32), (33, 32), (32, 33)])
def test_dft_idft_map(shape):
    m, n = shape
    size = m * n
    uu = generate_uv(n)
    vv = generate_uv(m)

    uu, vv = np.meshgrid(uu, vv)  # Loses units
    uv = np.array([uu, vv]).reshape(2, size) / u.arcsec

    # All zeros
    zeros = np.zeros(shape)
    vis = dft_map(zeros, uv)
    # All visibilities should be zero
    assert np.array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, shape, uv)
    # Should get back original map after dft(idft())
    assert np.array_equal(zeros, out_map)

    # All ones
    ones = np.ones(shape)
    vis = dft_map(ones, uv)
    out_map = idft_map(vis, shape, uv)
    assert np.allclose(ones, out_map)

    # Delta
    delta = zeros
    delta[m//2, n//2] = 1.0
    vis = dft_map(ones, uv)
    out_map = idft_map(vis, shape, uv)
    assert np.allclose(ones, out_map)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(stddev=5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, uv)
    out_map = idft_map(vis, shape, uv)
    assert np.allclose(gaussian, out_map)


@pytest.mark.parametrize("pixel_size", [(0.5, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.5)])
def test_dft_idft_map_pixel_size(pixel_size):
    pixel_size = pixel_size * u.arcsec
    shape = (33, 33)
    m, n = shape
    size = m * n
    uu = generate_uv(m, pixel_size=pixel_size[0])
    vv = generate_uv(n, pixel_size=pixel_size[1])

    uu, vv = np.meshgrid(uu, vv)  # Losses units
    uv = np.array([uu, vv]).reshape(2, size) / u.arcsec

    # All zeros
    zeros = np.zeros(shape)
    vis = dft_map(zeros, uv, pixel_size=pixel_size)
    # All visibilities should be zero
    assert np.array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    # Should get back original map after dft(idft())
    assert np.array_equal(zeros, out_map)

    # All ones
    ones = np.ones(shape)
    vis = dft_map(ones, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(ones, out_map)

    # Delta
    delta = zeros
    delta[m//2, n//2] = 1.0
    vis = dft_map(ones, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(ones, out_map)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(stddev=5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(gaussian, out_map)


@pytest.mark.parametrize("center", [(0, 0), (2.1, 2.1), (5.4, -5.4), (-5.6, 5.6)])
def test_dft_idft_map_center(center):
    center = center * u.arcsec
    shape = (33, 33)
    m, n = shape
    size = m * n
    uu = generate_uv(n, center=center[1])
    vv = generate_uv(m, center=center[0])

    uu, vv = np.meshgrid(uu, vv)
    uv = np.array([uu, vv]).reshape(2, size) / u.arcsec

    # All zeros
    zeros = np.zeros(shape)
    vis = dft_map(zeros, uv, center=center)
    # All visibilities should be zero
    assert np.array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, shape, uv, center=center)
    # Should get back original map after dft(idft())
    assert np.array_equal(zeros, out_map)

    # All ones
    ones = np.ones(shape)
    vis = dft_map(ones, uv, center=center)
    out_map = idft_map(vis, shape, uv, center=center)
    assert np.allclose(ones, out_map)

    # Delta
    delta = zeros
    delta[m//2, n//2] = 1.0
    vis = dft_map(ones, uv, center=center)
    out_map = idft_map(vis, shape, uv, center=center)
    assert np.allclose(ones, out_map)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(stddev=5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, uv, center=center)
    out_map = idft_map(vis, shape, uv, center=center)
    assert np.allclose(gaussian, out_map)


@pytest.mark.parametrize("shape, pixel_size", [((33, 32), (0.5, 0.5)),
                                               ((33, 32), (1.0, 1.0)),
                                               ((33, 32), (2.0, 2.0)),
                                               ((33, 32), (3.0, 3.0))])
def test_dft_idft_map_shape_pixel_size(shape, pixel_size):
    pixel_size = pixel_size * u.arcsec
    m, n = shape
    size = m * n

    vv = generate_uv(m, pixel_size=pixel_size[0])
    uu = generate_uv(n, pixel_size=pixel_size[1])

    uu, vv = np.meshgrid(uu, vv)
    uv = np.array([uu, vv]).reshape(2, size) / u.arcsec

    # All zeros
    zeros = np.zeros(shape)
    vis = dft_map(zeros, uv, pixel_size=pixel_size)
    # All visibilities should be zero
    assert np.array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    # Should get back original map after dft(idft())
    assert np.array_equal(zeros, out_map)

    # All ones
    ones = np.ones(shape)
    vis = dft_map(ones, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(ones, out_map)

    # Delta
    delta = zeros
    delta[m // 2, n // 2] = 1.0
    vis = dft_map(ones, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(ones, out_map)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(stddev=5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, uv, pixel_size=pixel_size)
    out_map = idft_map(vis, shape, uv, pixel_size=pixel_size)
    assert np.allclose(gaussian, out_map)


def test_equivalence_of_convolve():
    data = np.zeros((33, 33))
    #data[16,16] = 10.0
    data[3:6,3:6] = 5.0

    m, n = (33, 33)
    size = m * n

    vv = generate_uv(m)
    uu = generate_uv(n)

    uu, vv = np.meshgrid(uu, vv)  # Loses units
    uv = np.array([uu, vv]).reshape(2, size) / u.arcsec

    full_vis = dft_map(data, uv)

    sampling = np.random.choice(2, size=(33**2))*1
    sampling.reshape(33,33)[16,16] = 1
    sub_vis = sampling * full_vis

    bp1 = idft_map(full_vis, (33, 33), uv)

    bp2 = idft_map(sub_vis, (33, 33), uv)

    psf1 = idft_map(sampling*9, (33*3, 33*3), uv)

    conv = signal.convolve(data, psf1, mode='same', method='fft')

    non_zero = np.where(sampling != 0)[0]

    psf2 = idft_map(sampling[non_zero], (33, 33), uv[:, non_zero])

    bp3 = idft_map(full_vis[non_zero], (33, 33), uv[:, non_zero])

    # TODO figure out why convolution and inverse don't match perfectly ever where -> padding
    assert np.allclose(bp2, conv)
    assert np.allclose(bp2, bp3)
    assert np.allclose(psf1[33:66,33:66], psf2)
