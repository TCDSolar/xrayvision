import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

import astropy.units as u
from astropy.convolution.kernels import Gaussian2DKernel

from xrayvision.clean import (
    _component,
    _gaussian_basis,
    _group_images,
    _paraboloid_basis,
    _psf_fwhm,
    _radial_prolate_sphereoidal,
    _restore_inputs,
    _vec_radial_prolate_sphereoidal,
    clean,
    ms_clean,
    vis_clean,
    vis_ms_clean,
)
from xrayvision.imaging import image_to_vis, vis_psf_image, vis_to_image
from xrayvision.transform import dft_map, idft_map
from xrayvision.visibility import Visibilities


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
    out_map, _model, _resid = clean(dirty_map, dirty_beam, clean_beam_width=None)

    # Within threshold default threshold of 0.1
    assert_allclose(out_map, clean_map, atol=dirty_beam.max() * 1e-12)


@pytest.mark.parametrize(("n", "m"), [(65, 65), (64, 65), (65, 64), (64, 64)])
def test_clean_even_shape(n, m):
    pos = (20, 30)
    clean_map = np.zeros((n, m))
    clean_map[pos] = 10.0
    on, om = (3 * n) // 2 * 2 + 1, (3 * m) // 2 * 2 + 1
    dirty_beam = np.zeros((on, om))
    dirty_beam[(on - 1) // 2, (om - 1) // 2] = 1.0
    dirty_map = signal.convolve(clean_map, dirty_beam, mode="same")

    _out_map, model, resid = clean(dirty_map, dirty_beam, clean_beam_width=None, niter=200, thres=1e-8)

    assert np.unravel_index(np.argmax(model), model.shape) == pos
    assert_allclose(model.sum(), 10.0, atol=1e-8)
    assert_allclose(np.abs(resid).max(), 0.0, atol=1e-8)


@pytest.fixture
def tapered_cross_beam():
    r"""
    A synthetic dirty beam with a strong (0.75 of peak) but *decaying* cross-shaped sidelobe,
    representative of real dirty beams, whose sidelobe strength falls off with distance from the
    main lobe (confirmed against real STIX back-projected data, which shows near-in sidelobe
    ratios of ~0.75-0.9 at small angular separations from the peak).
    """

    def make_beam(n, m, peak_sidelobe=0.75):
        on, om = (3 * n) // 2 * 2 + 1, (3 * m) // 2 * 2 + 1
        beam = np.zeros((on, om))
        cy, cx = (on - 1) // 2, (om - 1) // 2
        beam[cy, cx] = 1.0
        decay_length = min(n, m) / 8.0
        for d in range(1, min(n, m) // 2):
            amp = peak_sidelobe * np.exp(-d / decay_length)
            beam[cy - d, cx] = amp
            beam[cy + d, cx] = amp
            beam[cy, cx - d] = amp
            beam[cy, cx + d] = amp
        return beam

    return make_beam


@pytest.mark.parametrize(("n", "m"), [(65, 65), (64, 65), (65, 64), (64, 64)])
def test_ms_clean_even_shape(n, m, tapered_cross_beam):
    pos1, pos2 = (15, 30), (40, 32)

    def recover(n, m):
        clean_map = np.zeros((n, m))
        clean_map[pos1] = 10.0
        clean_map[pos2] = 7.0
        dirty_beam = tapered_cross_beam(n, m)
        dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

        model, _res = ms_clean(
            dirty_map,
            dirty_beam,
            pixel_size=[1, 1] * u.arcsec / u.pixel,
            scales=[1, 2, 4],
            clean_beam_width=None,
            niter=3000,
        )
        return model[pos1], model[pos2], model.sum()

    assert_allclose(recover(n, m), recover(65, 65), atol=0.05)


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
    assert res[2, 2] == res.max() == 1.0

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


def test_ms_clean_multiscale_recovers_flux(tapered_cross_beam):
    n = m = 65
    pos1 = [15, 30]
    pos2 = [40, 32]

    clean_map = np.zeros((n, m))
    clean_map[pos1[0], pos1[1]] = 10.0
    clean_map[pos2[0], pos2[1]] = 7.0

    dirty_beam = tapered_cross_beam(n, m)
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    model, _res = ms_clean(
        dirty_map,
        dirty_beam,
        pixel_size=[1, 1] * u.arcsec / u.pixel,
        scales=[1, 2, 4],
        clean_beam_width=None,
        niter=3000,
    )

    # The model should recover close to the true, injected flux at each source and overall.
    assert_allclose(model[pos1[0], pos1[1]], 10.0, atol=0.2)
    assert_allclose(model[pos2[0], pos2[1]], 7.0, atol=0.2)
    assert_allclose(model.sum(), clean_map.sum(), atol=0.5)


def test_ms_clean_non_positive_beam_peak_raises():
    n = m = 65
    with pytest.raises(ValueError, match="positive peak"):
        ms_clean(
            np.ones((n, m)),
            np.zeros((3 * n, 3 * m)),
            pixel_size=[1, 1] * u.arcsec / u.pixel,
            scales=[1, 2],
            clean_beam_width=None,
        )


def test_ms_clean_recovers_extended_source_with_multiple_scales():
    n = m = 65
    clean_map = _component(scale=8, shape=(n, m))
    clean_map /= clean_map.sum()

    dirty_beam = np.zeros((3 * n, 3 * m))
    dirty_beam[(3 * n - 1) // 2, (3 * m - 1) // 2] = 1.0
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    model, _res = ms_clean(
        dirty_map,
        dirty_beam,
        pixel_size=[1, 1] * u.arcsec / u.pixel,
        scales=[1, 2, 4, 8],
        clean_beam_width=None,
        niter=100,
        thres=1e-10,
    )

    # Recover the extended source within a limited number of iterations, even when a point
    # scale is available. Selecting only point components recovers less than 20% of its flux.
    assert_allclose(model.sum(), clean_map.sum(), atol=0.15)
    assert_allclose(model, clean_map, atol=0.004)


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

    clean_map, _model, _res = clean(
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

    clean_map, _model, _res = vis_clean(
        vis,
        shape=(m, n) * u.pix,
        pixel_size=[2, 2] * u.arcsec / u.pix,
        clean_beam_width=None,
        niter=100,
        scheme="uniform",
    )
    np.allclose(data, clean_map.data, atol=0.1)


def test_ms_clean_unsorted_scales_raises(tapered_cross_beam):
    n = m = 65
    dirty_map = np.zeros((n, m))
    dirty_beam = tapered_cross_beam(n, m)

    with pytest.raises(ValueError, match="ascending order"):
        ms_clean(
            dirty_map,
            dirty_beam,
            pixel_size=[1, 1] * u.arcsec / u.pixel,
            scales=[4, 1],
            clean_beam_width=None,
        )


def test_clean_max_iterations(caplog, tapered_cross_beam):
    n = m = 65
    pos = (20, 30)
    clean_map = np.zeros((n, m))
    clean_map[pos] = 10.0
    dirty_beam = tapered_cross_beam(n, m)
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    caplog.set_level("INFO")
    # thres=None and a tiny niter: the loop always has positive residual left to clean, so it
    # runs out the clock rather than converging or hitting the negative-residual guard.
    _out_map, model, _resid = clean(dirty_map, dirty_beam, clean_beam_width=None, niter=3, thres=None)

    assert "Max iterations reached" in caplog.text
    assert model.sum() > 0


def test_clean_convolve_residual(tapered_cross_beam):
    n = m = 65
    pos = (20, 30)
    clean_map = np.zeros((n, m))
    clean_map[pos] = 10.0
    dirty_beam = tapered_cross_beam(n, m)
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    # A handful of iterations, well short of convergence, so a substantial residual remains for
    # convolve_residual to visibly act on.
    out_plain, _model, resid_plain = clean(
        dirty_map, dirty_beam, pixel_size=[1, 1] * u.arcsec / u.pix, clean_beam_width=4.0 * u.arcsec, niter=20
    )
    out_convolved, _model, resid_convolved = clean(
        dirty_map,
        dirty_beam,
        pixel_size=[1, 1] * u.arcsec / u.pix,
        clean_beam_width=4.0 * u.arcsec,
        niter=20,
        convolve_residual=True,
    )

    assert out_plain.shape == out_convolved.shape
    assert np.all(np.isfinite(out_convolved))
    # Convolving the residual with the clean beam should smooth it, not leave it untouched.
    assert not np.allclose(resid_plain, resid_convolved)


def test_ms_clean_clean_beam_width(tapered_cross_beam):
    n = m = 65
    pos1, pos2 = (15, 30), (40, 32)
    clean_map = np.zeros((n, m))
    clean_map[pos1] = 10.0
    clean_map[pos2] = 7.0
    dirty_beam = tapered_cross_beam(n, m)
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    out_map, model, resid = ms_clean(
        dirty_map,
        dirty_beam,
        pixel_size=[1, 1] * u.arcsec / u.pixel,
        scales=[1, 2, 4],
        clean_beam_width=4.0 * u.arcsec,
        niter=3000,
        convolve_residual=True,
    )

    assert out_map.shape == dirty_map.shape
    assert model.shape == dirty_map.shape
    assert resid.shape == dirty_map.shape
    assert np.all(np.isfinite(out_map))


def test_vis_ms_clean_sim():
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

    clean_map, model, residual = vis_ms_clean(
        vis,
        shape=(m, n) * u.pix,
        pixel_size=[2, 2] * u.arcsec / u.pix,
        scales=[1, 2],
        clean_beam_width=0.1 * u.arcsec,
        niter=200,
    )
    assert clean_map.data.shape == data.shape
    assert model.data.shape == data.shape
    assert residual.data.shape == data.shape
    assert_allclose(clean_map.data, data, atol=0.01)


def test_vis_ms_clean_no_restore_no_map():
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

    clean_map, model, residual = vis_ms_clean(
        vis,
        shape=(m, n) * u.pix,
        pixel_size=[2, 2] * u.arcsec / u.pix,
        scales=[1, 2],
        clean_beam_width=None,
        niter=200,
        map=False,
    )
    assert isinstance(clean_map, np.ndarray)
    assert clean_map.shape == data.shape
    assert model.shape == data.shape
    assert residual.shape == data.shape
    assert_allclose(clean_map, model / 4.0 + residual)


def _circles_vis(truth, n_circles=10, per_circle=12, ratio=0.6, first_radius=0.4):
    # u-v points on circles of geometrically decreasing radius, like the grids of a Fourier-based
    # hard x-ray imager, so groups of neighbouring circles have beams of very different width
    radii = first_radius * ratio ** np.arange(n_circles)
    angle = np.linspace(0, np.pi, per_circle, endpoint=False)
    uu = np.concatenate([r * np.cos(angle + 0.3 * i) for i, r in enumerate(radii)]) / u.arcsec
    vv = np.concatenate([r * np.sin(angle + 0.3 * i) for i, r in enumerate(radii)]) / u.arcsec
    return image_to_vis(truth * u.dimensionless_unscaled, u=uu, v=vv, pixel_size=[1, 1] * u.arcsec / u.pix)


def _two_source_truth(n=65):
    y, x = np.indices((n, n))

    def gaussian(cy, cx, sigma, flux):
        kernel = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
        return flux * kernel / kernel.sum()

    return gaussian(30, 28, 1.0, 10.0) + gaussian(34, 40, 6.0, 30.0)


def _groups(n_circles=10, per_circle=12, size=3):
    return [np.arange(i * per_circle, min(i + size, n_circles) * per_circle) for i in range(0, n_circles, size)]


def test_psf_fwhm():
    n = 129
    x = np.arange(n) - (n - 1) // 2
    sigma = 5.0
    gaussian = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    # No negative lobes: the search is bounded by the edges of the beam
    assert abs(_psf_fwhm(gaussian) - 2.3548 * sigma) <= 1

    # Negative lobes either side of the peak, which ring on at twice the width
    ringing = np.cos(np.pi * x / (4 * sigma))[None, :] * gaussian
    assert abs(_psf_fwhm(ringing) - 2 * sigma) <= 1

    # A peak that is a single pixel wide is not narrower than one pixel
    narrow = np.zeros((n, n))
    narrow[(n - 1) // 2, (n - 1) // 2] = 1.0
    narrow[(n - 1) // 2, (n - 1) // 2 + 1] = narrow[(n - 1) // 2, (n - 1) // 2 - 1] = -0.1
    assert _psf_fwhm(narrow) == 1.0


@pytest.mark.parametrize("make_basis", [_gaussian_basis, _paraboloid_basis])
@pytest.mark.parametrize("shape", [(33, 33), (32, 33), (32, 32)])
def test_basis_functions(make_basis, shape):
    fwhm = 6.0
    kernel = make_basis(fwhm, shape)
    assert_allclose(kernel.sum(), 1.0)
    # Centred on the integer pixel `signal.convolve(..., mode="same")` treats as the centre
    centre = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    assert np.unravel_index(np.argmax(kernel), shape) == centre
    # and has the requested full width at half maximum along the row through the peak
    row = kernel[centre[0]]
    assert abs(np.sum(row >= row.max() / 2) - fwhm) <= 1


def test_group_images_sum_to_full():
    n = 33
    vis = _circles_vis(_two_source_truth(n))
    shape, pixel_size = [n, n] * u.pix, [1, 1] * u.arcsec / u.pix
    for scheme in ("natural", "uniform"):
        maps, beams = _group_images(vis, _groups(), shape, pixel_size, scheme)
        assert maps.shape == (4, n, n)
        assert beams.shape == (4, 3 * n, 3 * n)
        assert_allclose(maps.sum(axis=0), vis_to_image(vis, shape=shape, pixel_size=pixel_size, scheme=scheme).value)
        assert_allclose(
            beams.sum(axis=0), vis_psf_image(vis, shape=shape * 3, pixel_size=pixel_size, scheme=scheme).value
        )

    # Visibilities that are in no group are not used
    unused = _groups()[:2]
    maps, _ = _group_images(vis, unused, shape, pixel_size, "natural")
    used = np.concatenate(unused)
    assert_allclose(
        maps.sum(axis=0),
        idft_map(
            vis.visibilities[used],
            u=vis.u[used],
            v=vis.v[used],
            shape=shape,
            weights=np.ones(used.size) / used.size,
            pixel_size=pixel_size,
        ).value,
    )


def test_group_images_invalid_groups():
    n = 33
    vis = _circles_vis(_two_source_truth(n))
    shape, pixel_size = [n, n] * u.pix, [1, 1] * u.arcsec / u.pix
    with pytest.raises(ValueError, match="overlap"):
        _group_images(vis, [[0, 1, 2], [2, 3]], shape, pixel_size, "natural")
    with pytest.raises(ValueError, match="at least one"):
        _group_images(vis, [[0, 1, 2], []], shape, pixel_size, "natural")


@pytest.mark.parametrize("basis", ["gaussian", "paraboloid"])
def test_vis_ms_clean_groups_recovers_flux(basis):
    truth = _two_source_truth()
    vis = _circles_vis(truth)

    # Without a clean beam the result is the model plus the background estimate
    clean_map, model, residual = vis_ms_clean(
        vis,
        shape=[65, 65] * u.pix,
        pixel_size=[1, 1] * u.arcsec / u.pix,
        scales=None,
        groups=_groups(),
        basis=basis,
        map=False,
        niter=3000,
        thres=1e-5,
    )
    assert_allclose(clean_map, model + residual)
    assert_allclose(model.sum(), truth.sum(), rtol=0.1)
    assert np.linalg.norm(clean_map - truth) / np.linalg.norm(truth) < 0.25
    # The compact source is found at its own pixel, not smeared by the extended components
    assert np.unravel_index(np.argmax(model), model.shape) == (30, 28)


def test_vis_ms_clean_groups_map_and_clean_beam():
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)

    clean_map, model, residual = vis_ms_clean(
        vis,
        shape=[33, 33] * u.pix,
        pixel_size=[1, 1] * u.arcsec / u.pix,
        scales=None,
        groups=_groups(),
        clean_beam_width=3.0 * u.arcsec,
        niter=200,
    )
    for result in (clean_map, model, residual):
        assert result.data.shape == (33, 33)
    assert clean_map.meta["naxis1"] == 33


def test_vis_ms_clean_default_clean_beam():
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)
    shape, pixel_size = [33, 33] * u.pix, [1, 1] * u.arcsec / u.pix

    # No groups: a clean beam is used unless told otherwise
    clean_map, model, _residual = vis_ms_clean(vis, shape, pixel_size, [1, 2], map=False, niter=20)
    assert not np.allclose(clean_map, model)


def test_ms_clean_groups_scales_and_biases():
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)
    shape, pixel_size = [33, 33] * u.pix, [1, 1] * u.arcsec / u.pix
    maps, beams = _group_images(vis, _groups(), shape, pixel_size, "natural")

    def run(**kwargs):
        return ms_clean(maps, beams, pixel_size, clean_beam_width=None, niter=1, **kwargs)[0]

    measured = [_psf_fwhm(beam) for beam in beams]
    assert measured == sorted(measured)

    # Measured scales are the defaults, groups given coarse to fine are put in order
    reference = run()
    assert_allclose(ms_clean(maps[::-1], beams[::-1], pixel_size, clean_beam_width=None, niter=1)[0], reference)
    assert_allclose(run(scales=measured), reference)

    # Scale biases only change what is searched for, not the flux that is found
    assert_allclose(run(scale_biases=[1.0, 0.1, 0.1, 0.1]), run(scale_biases=[1.0, 0.5, 0.5, 0.5]))
    # but they do change what is chosen, a heavily favoured coarse scale is used first
    assert not np.allclose(run(scale_biases=[0.01, 0.01, 0.01, 1.0]), reference)
    # the biases are given in the order of the groups passed in
    assert_allclose(
        ms_clean(maps[::-1], beams[::-1], pixel_size, clean_beam_width=None, niter=1, scale_biases=[0.1] * 3 + [1.0])[
            0
        ],
        run(scale_biases=[1.0] + [0.1] * 3),
    )


def test_ms_clean_groups_invalid_input():
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)
    shape, pixel_size = [33, 33] * u.pix, [1, 1] * u.arcsec / u.pix
    maps, beams = _group_images(vis, _groups(), shape, pixel_size, "natural")

    def run(maps=maps, beams=beams, **kwargs):
        return ms_clean(maps, beams, pixel_size, clean_beam_width=None, **kwargs)

    with pytest.raises(ValueError, match="one beam per group"):
        run(beams=beams[:2])
    with pytest.raises(ValueError, match="one beam per group"):
        run(beams=beams[0])
    with pytest.raises(ValueError, match="'gaussian' or 'paraboloid'"):
        run(basis="box")
    with pytest.raises(ValueError, match="ascending order"):
        run(scales=[8, 4, 2, 1])
    with pytest.raises(ValueError, match="one entry per group"):
        run(scales=[1, 2, 4])
    with pytest.raises(ValueError, match="one entry per group"):
        run(scale_biases=[1, 0.5])
    with pytest.raises(ValueError, match="positive peak"):
        run(beams=-beams)


def test_ms_clean_groups_stops():
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)
    shape, pixel_size = [33, 33] * u.pix, [1, 1] * u.arcsec / u.pix
    maps, beams = _group_images(vis, _groups(), shape, pixel_size, "natural")

    # Scales that are only sidelobes or noise are never searched, with none left there is nothing to do
    model, _res = ms_clean(-np.abs(maps), beams, pixel_size, clean_beam_width=None)
    assert not model.any()

    # Threshold on the summed residual
    model, _res = ms_clean(maps, beams, pixel_size, clean_beam_width=None, thres=np.abs(maps.sum(axis=0)).max())
    assert model.any()


def test_ms_clean_groups_diverging(caplog):
    # Two groups with the same delta beams, apart from negative lobes in the second: each
    # component found in the first is subtracted from the second as an *increase*, which is
    # weighted up by the second group's small initial peak. The lobes are small enough for the
    # total beam to still integrate to a positive value.
    n = 33
    peak = (3 * n - 1) // 2
    beams = np.zeros((2, 3 * n, 3 * n))
    beams[:, peak, peak] = 1.0
    beams[1, peak, peak - 3] = beams[1, peak, peak + 3] = -0.6
    maps = np.zeros((2, n, n))
    maps[0, 16, 16] = 1.0
    maps[1, 16, 16] = 0.25

    caplog.set_level("INFO")
    ms_clean(maps, beams, [1, 1] * u.arcsec / u.pix, clean_beam_width=None, gain=1.0)
    assert "Diverging" in caplog.text


def test_ms_clean_groups_negative_largest_scale(caplog):
    n = 33
    peak = (3 * n - 1) // 2
    beams = np.zeros((2, 3 * n, 3 * n))
    beams[:, peak, peak] = 1.0
    beams[1, peak, peak - 3] = beams[1, peak, peak + 3] = 5.0
    maps = np.zeros((2, n, n))
    maps[0, 16, 16] = 1.0
    maps[1, 16, 16] = 0.5

    # Subtracting the lobe of the second group drives it negative, at its own (largest) scale
    caplog.set_level("INFO")
    ms_clean(maps, beams, [1, 1] * u.arcsec / u.pix, clean_beam_width=None, gain=0.5)
    assert "Reached negative on largest scale" in caplog.text


def test_ms_clean_max_iterations_groups(caplog):
    n = 33
    vis = _circles_vis(_two_source_truth(n))
    maps, beams = _group_images(vis, _groups(), [n, n] * u.pix, [1, 1] * u.arcsec / u.pix, "natural")
    caplog.set_level("INFO")
    ms_clean(maps, beams, [1, 1] * u.arcsec / u.pix, clean_beam_width=None, niter=2, thres=0.0)
    assert "Max iterations reached" in caplog.text


def test_ms_clean_scale_bias_does_not_change_flux(tapered_cross_beam):
    n = 65
    clean_map = np.zeros((n, n))
    clean_map[30, 30] = 10.0
    dirty_beam = tapered_cross_beam(n, n)
    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode="same")

    def first_component(biases):
        model, _res = ms_clean(
            dirty_map,
            dirty_beam,
            pixel_size=[1, 1] * u.arcsec / u.pixel,
            scales=[1, 2, 4],
            clean_beam_width=None,
            niter=1,
            scale_biases=biases,
        )
        return model

    assert_allclose(first_component([1.0, 0.5, 0.2]), first_component([1.0, 0.1, 0.01]))
    assert first_component([1.0, 0.5, 0.2]).sum() > 0

    with pytest.raises(ValueError, match="one entry per scale"):
        first_component([1.0, 0.5])


@pytest.mark.parametrize("method", ["clean", "multiscale", "grouped"])
@pytest.mark.parametrize("beam_width", [None, 0.01 * u.arcsec, 3.0 * u.arcsec])
@pytest.mark.parametrize(
    "pixel_size", [[1, 1] * u.arcsec / u.pix, [2, 3] * u.arcsec / u.pix, [2 / 3600, 3 / 3600] * u.deg / u.pix]
)
def test_clean_surface_brightness_and_background(method, beam_width, pixel_size):
    # One half-gain iteration leaves equal flux in the component model and dirty residual.
    # The dirty beam has a broad central peak and a different integral from the restoring beam.
    shape = (17, 19)
    truth = np.zeros(shape)
    truth[8, 9] = 10.0
    beam = Gaussian2DKernel(2.0, 3.0, x_size=57, y_size=51).array
    beam /= beam.max()
    dirty_map = signal.convolve2d(truth, beam, mode="same")
    kwargs = {"pixel_size": pixel_size, "clean_beam_width": beam_width, "gain": 0.5, "niter": 1, "thres": 0.0}
    if method == "clean":
        result = clean(dirty_map, beam, **kwargs)
    elif method == "multiscale":
        result = ms_clean(dirty_map, beam, scales=[1, 2, 4], **kwargs)
    else:
        result = ms_clean(
            np.stack([dirty_map / 2] * 2),
            np.stack([beam / 2] * 2),
            scales=[1, 1],
            basis="paraboloid",
            **kwargs,
        )

    if method != "clean" and beam_width is None:
        model, residual = result
    else:
        image, model, residual = result

    # The raw model stays in flux-per-pixel units for forward projection.
    expected_model = truth / 2
    assert_allclose(model, expected_model, atol=1e-12)
    dy, dx = pixel_size.to_value(u.arcsec / u.pix)
    area = dy * dx
    # Integrate the dirty PSF over this image's field of view, rather than its oversized array
    # or the clean beam. This normalization must also be used when no clean beam is requested.
    beam_window = beam[17:34, 19:38]
    expected_residual = dirty_map / 2 / (beam_window.sum() * area)
    assert_allclose(residual, expected_residual, atol=1e-12)
    assert_allclose(residual.sum() * area, 5.0, atol=1e-10)

    if method == "clean" or beam_width is not None:
        restored = expected_model
        if beam_width is not None:
            y, x = np.indices(beam.shape)
            sigma = beam_width.to_value(u.arcsec) / np.sqrt(8 * np.log(2))
            kernel = np.exp(-(((y - 25) * dy) ** 2 + ((x - 28) * dx) ** 2) / (2 * sigma**2))
            kernel /= kernel.sum()
            restored = signal.convolve2d(expected_model, kernel, mode="same")
        assert_allclose(image, restored / area + expected_residual, atol=1e-12)


@pytest.mark.parametrize("method", ["clean", "multiscale", "grouped"])
def test_vis_clean_no_beam_surface_brightness(method):
    truth = _two_source_truth(33)
    vis = _circles_vis(truth)
    vis = Visibilities(vis.visibilities.value * u.ph, u=vis.u, v=vis.v)
    shape = [33, 33] * u.pix
    pixel_size = [2, 3] * u.arcsec / u.pix
    if method == "clean":
        image, model, residual = vis_clean(vis, shape, pixel_size, clean_beam_width=None, map=False, niter=1)
    else:
        groups = _groups() if method == "grouped" else None
        image, model, residual = vis_ms_clean(
            vis, shape, pixel_size, scales=None, groups=groups, clean_beam_width=None, map=False, niter=1
        )
    assert_allclose(image, model / 6.0 + residual)


@pytest.mark.parametrize("method", ["clean", "multiscale", "grouped"])
def test_clean_background_is_independent_of_restoring_beam(method):
    n = 17
    beam = Gaussian2DKernel(2.0, x_size=3 * n, y_size=3 * n).array
    beam /= beam.max()
    dirty_map = beam[n : 2 * n, n : 2 * n].copy()
    pixel_size = [2, 3] * u.arcsec / u.pix

    def run(width, convolve_residual=False):
        kwargs = {
            "pixel_size": pixel_size,
            "clean_beam_width": width,
            "niter": 0,
            "convolve_residual": convolve_residual,
        }
        if method == "clean":
            return clean(dirty_map, beam, **kwargs)
        if method == "multiscale":
            return ms_clean(dirty_map, beam, scales=[1, 2, 4], **kwargs)
        return ms_clean(dirty_map[None], beam[None], scales=[1], **kwargs)

    background = run(None)[-1]
    for width in [0.01 * u.arcsec, 4.0 * u.arcsec]:
        assert_allclose(run(width)[-1], background, atol=1e-12)
    smoothed = run(4.0 * u.arcsec, convolve_residual=True)[-1]
    assert not np.allclose(smoothed, background)


def _run_clean_variant(method, dirty_map, beam, **kwargs):
    kwargs = {"pixel_size": [1, 1] * u.arcsec / u.pix, "clean_beam_width": None, "niter": 1, **kwargs}
    if method == "clean":
        return clean(dirty_map, beam, **kwargs)
    if method == "multiscale":
        return ms_clean(dirty_map, beam, scales=[1, 2], **kwargs)
    return ms_clean(dirty_map[None], beam[None], scales=[1], **kwargs)


@pytest.mark.parametrize("method", ["clean", "multiscale", "grouped"])
def test_clean_beam_smaller_than_image_raises(method):
    dirty_map = np.ones((17, 19))
    beam = Gaussian2DKernel(2.0, x_size=9, y_size=9).array
    with pytest.raises(ValueError, match="at least as large"):
        _run_clean_variant(method, dirty_map, beam)


@pytest.mark.parametrize("method", ["clean", "multiscale", "grouped"])
def test_clean_beam_with_non_positive_integral_raises(method):
    dirty_map = np.ones((17, 19))
    # A positive peak, but the negative sidelobe means the beam does not integrate to a positive
    # value over the image, so there is no background normalisation.
    beam = np.zeros((51, 57))
    beam[25, 28] = 1.0
    beam[25, 31] = -5.0
    with pytest.raises(ValueError, match="integrate to a positive value"):
        _run_clean_variant(method, dirty_map, beam)


def test_clean_pixel_size_required_with_clean_beam():
    dirty_map = np.ones((17, 19))
    beam = Gaussian2DKernel(2.0, x_size=57, y_size=51).array
    with pytest.raises(ValueError, match="pixel_size is required"):
        clean(dirty_map, beam, niter=1)
    # Without a clean beam the image is only scaled by the pixel area, 1 arcsec^2 if not given
    _image, _model, residual = clean(dirty_map, beam, clean_beam_width=None, niter=1)
    _image, _model, residual_unit = clean(
        dirty_map, beam, pixel_size=[1, 1] * u.arcsec / u.pix, clean_beam_width=None, niter=1
    )
    assert_allclose(residual, residual_unit)


def test_restore_inputs():
    beam = Gaussian2DKernel(2.0, x_size=57, y_size=51).array
    integral, pixel = _restore_inputs(beam, (17, 19), None, None)
    assert_allclose(integral, beam[17:34, 19:38].sum())
    assert_allclose(pixel.to_value(u.arcsec / u.pix), [1.0, 1.0])
    with pytest.raises(ValueError, match="pixel_size is required"):
        _restore_inputs(beam, (17, 19), None, 4.0 * u.arcsec)


def _lobed_groups(lobe, extra_group=False):
    # Two groups with delta beams, apart from lobes at +-3 pixels in the second. With `extra_group`
    # a third, empty group is added so that the lobed one is not the largest scale.
    n = 33
    peak = (3 * n - 1) // 2
    beams = np.zeros((3 if extra_group else 2, 3 * n, 3 * n))
    beams[:, peak, peak] = 1.0
    beams[1, peak, peak - 3] = beams[1, peak, peak + 3] = lobe
    maps = np.zeros((beams.shape[0], n, n))
    maps[0, 16, 16] = 1.0
    maps[1, 16, 16] = 0.25
    return maps, beams


def test_ms_clean_groups_mask_negative_scales(caplog):
    # A positive lobe in the second group's beam drives that scale negative, which by default
    # stops the clean as it is the largest scale. Masked, only that scale is dropped.
    maps, beams = _lobed_groups(0.6)
    # The scales are given so the order of the groups is not measured from the unrealistic beams
    kwargs = {
        "pixel_size": [1, 1] * u.arcsec / u.pix,
        "scales": [1.0, 1.0],
        "clean_beam_width": None,
        "gain": 1.0,
        "niter": 50,
    }
    caplog.set_level("INFO")
    default_model, _ = ms_clean(maps, beams, **kwargs)
    assert "Reached negative on largest scale" in caplog.text

    caplog.clear()
    masked_model, _ = ms_clean(maps, beams, mask_negative_scales=True, **kwargs)
    assert "Reached negative on largest scale" not in caplog.text
    assert "No scale left to clean" in caplog.text
    # The scale dropped after being found negative never adds to the model again, the first scale
    # keeps cleaning
    assert masked_model.sum() != default_model.sum()


def test_ms_clean_groups_mask_negative_scales_has_no_divergence_guard(caplog):
    # The second group's lobes mean the first group's components are subtracted from it as an
    # increase, weighted up by its small initial peak, so its score grows past the guard.
    maps, beams = _lobed_groups(-0.6, extra_group=True)
    kwargs = {
        "pixel_size": [1, 1] * u.arcsec / u.pix,
        "scales": [1.0, 1.0, 1.0],
        "clean_beam_width": None,
        "gain": 1.0,
        "niter": 20,
    }
    caplog.set_level("INFO")
    ms_clean(maps, beams, **kwargs)
    assert "Diverging" in caplog.text

    caplog.clear()
    ms_clean(maps, beams, mask_negative_scales=True, **kwargs)
    assert "Diverging" not in caplog.text


def test_vis_ms_clean_groups_mask_negative_scales(caplog):
    vis = _circles_vis(_two_source_truth(), n_circles=8, ratio=0.7, first_radius=0.5)
    kwargs = {
        "shape": [65, 65] * u.pix,
        "pixel_size": [1, 1] * u.arcsec / u.pix,
        "scales": None,
        "groups": _groups(n_circles=8, size=2),
        "map": False,
        "niter": 3000,
        "thres": 1e-5,
    }
    caplog.set_level("INFO")
    vis_ms_clean(vis, **kwargs)
    assert "Reached negative on largest scale" in caplog.text

    caplog.clear()
    _clean_map, model, _residual = vis_ms_clean(vis, mask_negative_scales=True, **kwargs)
    assert "Reached negative on largest scale" not in caplog.text
    assert np.isfinite(model).all()
