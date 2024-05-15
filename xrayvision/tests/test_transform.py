import astropy.units as apu
import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy import signal

from xrayvision.transform import dft_map, generate_uv, generate_xy, idft_map


@pytest.mark.parametrize("pixel_size", [0.5, 1, 2, 3])
def test_generate_xy_pixel_size(pixel_size):
    odd = np.linspace(-16 * pixel_size, 16 * pixel_size, 33) * apu.arcsec
    even = np.linspace(-15.5 * pixel_size, 15.5 * pixel_size, 32) * apu.arcsec
    pixel_size = pixel_size * apu.arcsec / apu.pix

    assert np.array_equal(odd, generate_xy(33 * apu.pix, pixel_size=pixel_size))
    assert np.array_equal(even, generate_xy(32 * apu.pix, pixel_size=pixel_size))


@pytest.mark.parametrize("phase_centre", [0, +5.5, -5.5])
def test_generate_xy_offset(phase_centre):
    phase_centre = phase_centre * apu.arcsec
    even = np.linspace(-15.5, 15.5, 32) * apu.arcsec + phase_centre
    odd = np.linspace(-16, 16, 33) * apu.arcsec + phase_centre

    assert np.array_equal(even, generate_xy(32 * apu.pix, phase_centre=phase_centre))
    assert np.array_equal(odd, generate_xy(33 * apu.pix, phase_centre=phase_centre))


@pytest.mark.parametrize(
    "phase_centre, pixel_size", [(0, (0.5, 1, 2, 3)), (+5.5, (0.5, 1, 2, 3)), (-5.5, (0.5, 1, 2, 3))]
)
def test_generate_xy_offset_size(phase_centre, pixel_size):
    phase_centre = phase_centre * apu.arcsec
    pixel_size = pixel_size * apu.arcsec / apu.pix
    # No sure if this is a good idea could be hard to debug
    for size in pixel_size:
        # Odd
        odd = np.linspace(-16 * size, 16 * size, 33) * apu.pix + phase_centre
        assert np.array_equal(odd, generate_xy(33 * apu.pix, phase_centre=phase_centre, pixel_size=size))

        # Even
        even = np.linspace(-15.5 * size, 15.5 * size, 32) * apu.pix + phase_centre
        assert np.array_equal(even, generate_xy(32 * apu.pix, phase_centre=phase_centre, pixel_size=size))


@pytest.mark.parametrize("pixel_size", [0.5, 1, 2, 3])
def test_generate_uv_pixel_size(pixel_size):
    pixel_size = pixel_size * apu.arcsec / apu.pix
    m = 33
    n = 32

    # Odd
    odd = np.linspace(-((m - 1) / 2) * (1 / (m * pixel_size)), ((m - 1) / 2) * (1 / (m * pixel_size)), m) / apu.pix

    # Know issue with quantities and isclose/allclose need to add unit to atol default value
    assert np.allclose(odd, generate_uv(m * apu.pix, pixel_size=pixel_size), atol=1e-8 / apu.arcsec)

    # Even
    even = (np.arange(n) - n / 2 + 0.5) * (1 / (pixel_size * n)) / apu.pix
    assert np.allclose(even, generate_uv(32 * apu.pix, pixel_size=pixel_size), atol=1e-8 / apu.arcsec)


@pytest.mark.parametrize("phase_centre", [0.0, -5.5, 5.5])
def test_generate_uv_pixel_offset(phase_centre):
    phase_centre = phase_centre * apu.arcsec
    m = 33
    n = 32

    # Odd
    odd = np.linspace(-((m - 1) / 2) * (1 / m), ((m - 1) / 2) * (1 / m), m) / apu.arcsec
    if phase_centre.value != 0:
        odd += 1 / phase_centre
    assert np.allclose(odd, generate_uv(m * apu.pix, phase_centre=phase_centre), atol=1e-8 / apu.arcsec)

    # Even
    even = (np.arange(n) - n / 2 + 0.5) * (1 / n) / apu.arcsec
    if phase_centre.value != 0:
        even += 1 / phase_centre
    assert np.allclose(even, generate_uv(n * apu.pix, phase_centre=phase_centre), atol=1e-8 / apu.arcsec)


@pytest.mark.parametrize(
    "phase_centre, pixel_size", [(0, (0.5, 1, 2, 3)), (+5.5, (0.5, 1, 2, 3)), (-5.5, (0.5, 1, 2, 3))]
)
def test_generate_uv_offset_size(phase_centre, pixel_size):
    phase_centre = phase_centre * apu.arcsec
    pixel_size = pixel_size * apu.arcsec / apu.pix
    m = 33
    n = 32

    for size in pixel_size:
        # Odd
        odd = np.linspace(-((m - 1) / 2) * (1 / (size * m)), ((m - 1) / 2) * (1 / (size * m)), m) / apu.pix
        if phase_centre != 0:
            odd += 1 / phase_centre
        assert np.allclose(
            odd, generate_uv(m * apu.pix, phase_centre=phase_centre, pixel_size=size), atol=1e-8 / apu.arcsec
        )

        # Even
        even = (np.arange(n) - n / 2 + 0.5) * (1 / (size * n)) / apu.pix
        if phase_centre != 0:
            even += 1 / phase_centre
        assert np.allclose(
            even, generate_uv(n * apu.pix, phase_centre=phase_centre, pixel_size=size), atol=1e-8 / apu.arcsec
        )


@pytest.mark.parametrize("shape", [(3, 3), (2, 3), (3, 2), (2, 2)])
@pytest.mark.parametrize("pixel_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("center", [(0, 0), (-1, 1), (1, -1), (1, 1)])
def test_dft_idft(shape, pixel_size, center):
    data = np.arange(np.prod(shape)).reshape(shape)
    uu = generate_uv(shape[0] * apu.pix, phase_centre=0 * apu.arcsec, pixel_size=pixel_size[0] * apu.arcsec / apu.pix)
    vv = generate_uv(shape[1] * apu.pix, phase_centre=0 * apu.arcsec, pixel_size=pixel_size[1] * apu.arcsec / apu.pix)
    u, v = np.meshgrid(uu, vv, indexing="ij")
    u = u.flatten()
    v = v.flatten()

    vis = dft_map(data, u=u, v=v, pixel_size=pixel_size * apu.arcsec / apu.pix)
    img = idft_map(vis, u=u, v=v, shape=shape * apu.pix, pixel_size=pixel_size * apu.arcsec / apu.pix)
    img = np.around(img / np.prod(shape))
    assert_allclose(data, img)


@pytest.mark.parametrize("shape", [(33, 33), (32, 32), (33, 32), (32, 33)])
def test_dft_idft_map(shape):
    m, n = shape
    uu = generate_uv(m * apu.pix)
    vv = generate_uv(n * apu.pix)
    u, v = np.meshgrid(uu, vv, indexing="ij")
    u = u.flatten()
    v = v.flatten()

    # All zeros
    zeros = np.zeros((m, n))
    vis = dft_map(zeros, u=u, v=v)
    # All visibilities should be zero
    # assert_array_equal(np.zeros(np.prod((m, n)), dtype=complex), vis)
    out_map = idft_map(vis, u=u, v=v, shape=shape * apu.pix)
    # Should get back the original map after dft(idft())
    assert_allclose(zeros, out_map / np.prod((m, n)))

    # All ones
    ones = np.ones((m, n))
    vis = dft_map(ones, u=u, v=v)
    # All visibilities should be 1
    # assert_allclose(np.ones(np.prod((m, n)), dtype=complex), vis)
    out_map = idft_map(vis, u=u, v=v, shape=shape * apu.pix)
    assert_allclose(ones, out_map / np.prod((m, n)))

    # Delta
    delta = zeros[:, :]
    delta[m // 2, n // 2] = 1.0
    vis = dft_map(delta, u=u, v=v)
    out_map = idft_map(vis, u=u, v=v, shape=shape * apu.pix)
    assert_allclose(out_map / np.prod(shape), delta, atol=1e-14)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, u=u, v=v)
    out_map = idft_map(vis, u=u, v=v, shape=shape * apu.pix)
    assert_allclose(out_map / np.prod(shape), gaussian)


# @pytest.mark.skip('UV coordinate generation off somewhere')
@pytest.mark.parametrize("pixel_size", [(1.0, 1.0), (0.5, 0.5), (1.5, 1.5), (2.0, 0.5), (0.5, 2.0)])
def test_dft_idft_map_pixel_size(pixel_size):
    pixel_size = pixel_size * apu.arcsec / apu.pix
    shape = (32, 32) * apu.pix
    m, n = shape.value.astype(int)
    size = m * n
    uu = generate_uv(m * apu.pix, pixel_size=pixel_size[0])
    vv = generate_uv(n * apu.pix, pixel_size=pixel_size[1])
    u, v = np.meshgrid(uu, vv)
    u = u.flatten()
    v = v.flatten()

    # All zeros
    zeros = np.zeros(shape.value.astype(int))
    vis = dft_map(zeros, u=u, v=v, pixel_size=pixel_size)
    # All visibilities should be zero
    assert_array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    # Should get back the original map after dft(idft())
    assert_array_equal(out_map / np.prod(shape.value), zeros)

    # All ones
    ones = np.ones(shape.value.astype(int))
    vis = dft_map(ones, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod(shape.value), ones)

    # Delta
    delta = zeros[:, :]
    delta[m // 2, n // 2] = 1.0
    vis = dft_map(delta, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod(shape.value), delta, atol=1e-14)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod(shape.value), gaussian)


# @pytest.mark.skip('UV coordinate generation off somewhere')
@pytest.mark.parametrize("phase_centre", [(0, 0), (2.1, 1.1), (5.4, -4.5), (-5.6, 5.6)])
def test_dft_idft_map_center(phase_centre):
    phase_centre = phase_centre * apu.arcsec
    shape = (33, 33)
    m, n = shape
    size = m * n
    shape = shape * apu.pix
    uu = generate_uv(n * apu.pix, phase_centre=phase_centre[0])
    vv = generate_uv(m * apu.pix, phase_centre=phase_centre[1])
    u, v = np.meshgrid(uu, vv)
    u = u.flatten()
    v = v.flatten()

    # All zeros
    zeros = np.zeros((m, n))
    vis = dft_map(zeros, u=u, v=v, phase_centre=phase_centre)
    # All visibilities should be zero
    assert_array_equal(vis, np.zeros(size, dtype=complex))
    out_map = idft_map(vis, u=u, v=v, shape=shape, phase_centre=phase_centre)
    # Should get back the original map after dft(idft())
    assert_array_equal(out_map / np.prod((m, n)), zeros)

    # All ones
    ones = np.ones((m, n))
    vis = dft_map(ones, u=u, v=v, phase_centre=phase_centre)
    out_map = idft_map(vis, u=u, v=v, shape=shape, phase_centre=phase_centre)
    assert_allclose(out_map / np.prod((m, n)), ones)

    # Delta
    delta = zeros[:, :]
    delta[m // 2, n // 2] = 1.0
    vis = dft_map(delta, u=u, v=v, phase_centre=phase_centre)
    out_map = idft_map(vis, u=u, v=v, shape=shape, phase_centre=phase_centre)
    assert_allclose(out_map / np.prod((m, n)), delta, atol=1e-14)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, u=u, v=v, phase_centre=phase_centre)
    out_map = idft_map(vis, u=u, v=v, shape=shape, phase_centre=phase_centre)
    assert_allclose(out_map / np.prod((m, n)), gaussian)


# @pytest.mark.skip('UV coordinate generation off somewhere')
@pytest.mark.parametrize(
    "shape, pixel_size",
    [((11, 10), (0.5, 0.5)), ((11, 10), (1.0, 1.0)), ((11, 10), (2.0, 2.0)), ((11, 10), (3.0, 3.0))],
)
def test_dft_idft_map_shape_pixel_size(shape, pixel_size):
    pixel_size = pixel_size * apu.arcsec / apu.pix
    m, n = shape
    size = m * n
    shape = shape * apu.pix

    uu = generate_uv(m * apu.pix, pixel_size=pixel_size[0])
    vv = generate_uv(n * apu.pix, pixel_size=pixel_size[1])
    u, v = np.meshgrid(uu, vv, indexing="ij")
    u = u.flatten()
    v = v.flatten()

    # All zeros
    zeros = np.zeros((m, n))
    vis = dft_map(zeros, u=u, v=v, pixel_size=pixel_size)
    # All visibilities should be zero
    assert_array_equal(np.zeros(size, dtype=complex), vis)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    # Should get back the original map after dft(idft())
    assert_array_equal(out_map / np.prod((m, n)), zeros)

    # All ones
    ones = np.ones((m, n))
    vis = dft_map(ones, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod((m, n)), ones)

    # Delta
    delta = zeros[:, :]
    delta[m // 2, n // 2] = 1.0
    vis = dft_map(delta, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod((m, n)), delta, atol=1e-14)

    # Gaussian - astropy has axis in reverse order compared to numpy
    gaussian = Gaussian2DKernel(5, x_size=n, y_size=m).array
    vis = dft_map(gaussian, u=u, v=v, pixel_size=pixel_size)
    out_map = idft_map(vis, u=u, v=v, shape=shape, pixel_size=pixel_size)
    assert_allclose(out_map / np.prod((m, n)), gaussian)


def test_equivalence_of_convolve():
    data = np.zeros((33, 33))
    # data[16,16] = 10.0
    data[3:6, 3:6] = 5.0

    m, n = (33, 33)

    vv = generate_uv(m * apu.pix)
    uu = generate_uv(n * apu.pix)
    u, v = np.meshgrid(uu, vv)
    u = u.flatten()
    v = v.flatten()

    full_vis = dft_map(data, u=u, v=v)

    sampling = np.random.choice(2, size=(33**2)) * 1
    sampling.reshape(33, 33)[16, 16] = 1
    sub_vis = sampling * full_vis

    non_zero = np.where(sampling != 0)[0]

    # bp1 = idft_map(full_vis, u=u, v=v, shape=(33, 33))

    bp2 = idft_map(sub_vis[non_zero], u=u[non_zero], v=v[non_zero], shape=(33, 33) * apu.pix)

    # Need to make the psf is large enough to slide over the entire data window
    psf1 = idft_map(sampling[non_zero], u=u[non_zero], v=v[non_zero], shape=(33 * 3, 33 * 3) * apu.pix)

    conv = signal.convolve(data, psf1, mode="same", method="fft")

    psf2 = idft_map(sampling[non_zero], u=u[non_zero], v=v[non_zero], shape=(33, 33) * apu.pix)

    bp3 = idft_map(full_vis[non_zero], u=u[non_zero], v=v[non_zero], shape=(33, 33) * apu.pix)

    assert np.allclose(bp2, conv)
    assert np.allclose(bp2, bp3)
    # Due to the enlarged psf need to only use the centre portion
    assert np.allclose(psf1[33:66, 33:66], psf2)


def test_phase_centre_equivalence():
    # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
    data = np.random.randn(8, 8)
    u = generate_uv(8 * apu.pix, phase_centre=0 * apu.arcsec, pixel_size=1 * apu.arcsec / apu.pix)
    v = generate_uv(8 * apu.pix, phase_centre=0 * apu.arcsec, pixel_size=1 * apu.arcsec / apu.pix)
    u, v = np.meshgrid(u, v)
    u, v = np.array([u, v]).reshape(2, u.size) / apu.arcsec

    # calculate normal 0,0
    vis = dft_map(data, u=u, v=v, pixel_size=[1, 1] * apu.arcsec / apu.pix)
    img1 = idft_map(
        vis, u=u, v=v, weights=1 / u.size, pixel_size=[1, 1] * apu.arcsec / apu.pix, shape=data.shape * apu.pix
    )
    assert_allclose(data, img1)

    # extract phase and amp
    phase = np.arctan2(np.imag(vis), np.real(vis)) * apu.rad
    amp = np.abs(vis)

    # change vis to a phase centre of 5, 5
    phase_shift = 2 * np.pi * (5 * apu.arcsec * u + 5 * apu.arcsec * v) * apu.rad
    vis_shifted = (np.cos(phase + phase_shift) + np.sin(phase + phase_shift) * 1j) * amp

    # make image with centre of 5, 5 with shifted vis
    img2 = idft_map(
        vis_shifted,
        u=u,
        v=v,
        weights=1 / u.size,
        pixel_size=[1, 1] * apu.arcsec / apu.pix,
        shape=data.shape * apu.pix,
        phase_centre=[5, 5] * apu.arcsec,
    )
    assert np.allclose(data, img2)


def test_fft_equivalence():
    # Odd (3, 3) so symmetric and chose shape and pixel size so xy values will run from 0 to 2 the same as in fft
    # TODO: add same kind of test for even for fft2 then A[n/2] has both pos and negative nyquist frequencies
    #  e.g shape (2, 2), (3, 2), (2, 3)
    shape = (3, 3)
    pixel = (1, 1)
    center = (1, 1)

    data = np.arange(np.prod(shape)).reshape(shape)
    uu = generate_uv(
        shape[0] * apu.pix, phase_centre=center[0] * apu.arcsec, pixel_size=pixel[0] * apu.arcsec / apu.pix
    )
    vv = generate_uv(
        shape[1] * apu.pix, phase_centre=center[1] * apu.arcsec, pixel_size=pixel[1] * apu.arcsec / apu.pix
    )
    u, v = np.meshgrid(uu, vv, indexing="ij")
    u = u.flatten()
    v = v.flatten()

    vis = dft_map(data, u=u, v=v, pixel_size=pixel * apu.arcsec / apu.pix, phase_centre=center * apu.arcsec)

    ft = fft2(data)
    fts = fftshift(ft)
    vis = vis.reshape(shape)
    # Convention in xrayvison has the minus sign on the forward transform but numpy has it on reverse
    vis_conj = np.conjugate(vis)
    assert_array_almost_equal(fts, vis_conj)

    vis_ft = ifftshift(vis_conj)
    img = ifft2(vis_ft)
    assert_array_almost_equal(np.real(img), data)
