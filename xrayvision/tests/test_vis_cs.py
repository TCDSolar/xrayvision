"""Physical reconstruction checks and regression against actual SolarSoft outputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import erf

import astropy.units as u

from xrayvision.transform import dft_map
from xrayvision.vis_cs import (
    _coordinate_descent,
    _fourier,
    _Gaussians,
    _in_view_volume,
    _random_words,
    _refine,
    _render,
    vis_cs,
)
from xrayvision.visibility import Visibilities


@pytest.fixture
def source_vis():
    theta = np.linspace(0, np.pi, 16, endpoint=False)
    radius = np.array([0.01, 0.02, 0.04, 0.08])
    uu = (radius[:, None] * np.cos(theta)).ravel()
    vv = (radius[:, None] * np.sin(theta)).ravel()
    values = 100 * np.exp(-2 * np.pi**2 * 4**2 * (uu**2 + vv**2)) * np.exp(2j * np.pi * (uu * 5 - vv * 3))
    return Visibilities(values * u.ph, uu / u.arcsec, vv / u.arcsec, amplitude_uncertainty=np.ones(len(uu)) * u.ph)


@pytest.mark.idl
@pytest.mark.parametrize(
    ("case", "shape", "pixel"), [(0, (65, 65), (1.0, 1.0)), (1, (47, 61), (2.0, 1.5))], ids=["square", "rectangular"]
)
def test_vis_cs_idl(case, shape, pixel):
    hissw = pytest.importorskip("hissw")
    # A fixed dictionary, as in validation/vis_cs/compare_idl.py, so both run identical problems.
    ny, nx = shape
    dy, dx = pixel
    rng = np.random.default_rng(21 + case)
    # SSW measures positions from the lower left of the FOV.
    x = rng.uniform(0.2, 0.8, 12) * nx * dx
    y = rng.uniform(0.2, 0.8, 12) * ny * dy
    sx = rng.uniform(2, 10, 12)
    sy = rng.uniform(2, 10, 12)
    phi = rng.uniform(-1, 1, 12)
    words = _Gaussians(x - (nx * dx - 1) / 2, y - (ny * dy - 1) / 2, sx, sy, phi)
    viv = _in_view_volume(words, shape, np.array(pixel))
    theta = np.linspace(0, np.pi, 12, endpoint=False)
    radius = np.array([0.007, 0.015, 0.035, 0.08])
    uu = (radius[:, None] * np.cos(theta)).ravel()
    vv = (radius[:, None] * np.sin(theta)).ravel()
    transform = _fourier(words, uu, vv)
    true_weights = np.zeros(12)
    true_weights[[2, 7]] = [70, 30] / viv[[2, 7]]
    observed = transform @ true_weights + rng.normal(0, 0.05, len(uu)) + 1j * rng.normal(0, 0.05, len(uu))
    sigamp = rng.uniform(0.5, 3, len(uu))

    ssw = hissw.Environment(ssw_packages=["hessi"])
    # force_double_precision as IDL parses plain literals as single precision
    script = """
d = replicate({vis_cs_gaussian, x:0d, y:0d, sx:0d, sy:0d, phi:0d, amp:1d, a:0d, b:0d, c:0d, viv:0d}, {{n}})
d.x = {{x | force_double_precision}}
d.y = {{y | force_double_precision}}
d.sx = {{sx | force_double_precision}}
d.sy = {{sy | force_double_precision}}
d.phi = {{phi | force_double_precision}}
pixel = {{pixel | force_double_precision}}
d = vis_cs_gaussdict_updateabc(d)
d = vis_cs_gaussdict_normalize(d)
d = vis_cs_gaussdict_calcviv(d, {{nx}}, {{ny}}, pixel)
vis = replicate({u:0d, v:0d, obsvis:dcomplex(0d), sigamp:0d}, {{nvis}})
vis.u = {{u | force_double_precision}}
vis.v = {{v | force_double_precision}}
vis.obsvis = dcomplex({{re | force_double_precision}}, {{im | force_double_precision}})
vis.sigamp = {{sigamp | force_double_precision}}
transform = vis_cs_gaussdict_transform(d, {{nx}} * pixel[0], {{ny}} * pixel[1], vis)
weights = vis_cs_coordinate_descent({{nx}} * pixel[0], {{ny}} * pixel[1], d, vis, 100d, 0.5d, 0, 0, {lambda:1000d})
image = vis_cs_gaussdict_render(d, {{nx}}, {{ny}}, weights, pixel)
viv = d.viv
amp = d.amp
    """
    out = ssw.run(
        script=script,
        args={
            "n": len(x),
            "x": x,
            "y": y,
            "sx": sx,
            "sy": sy,
            "phi": phi,
            "nx": nx,
            "ny": ny,
            "pixel": np.array([dx, dy]),  # SSW orders pixel as (dx, dy)
            "nvis": len(uu),
            "u": uu,
            "v": vv,
            "re": observed.real,
            "im": observed.imag,
            "sigamp": sigamp,
        },
        save_vars=["transform", "weights", "image", "viv", "amp"],
    )
    assert_allclose(transform, out["transform"].T, atol=2e-7)
    # SSW QROMB uses EPS=1e-5; scipy quadrature is deliberately tighter.
    assert_allclose(viv, out["viv"], atol=9e-5)
    weights = _coordinate_descent(transform, observed, sigamp, words.amp, viv, 100.0, 0.5, 1000.0, 1000, False)
    assert_allclose(weights, out["weights"], atol=1e-3)
    assert_allclose(_render(words, weights, shape, np.array(pixel)), out["image"], atol=4e-6)
    # Isolate the solver from quadrature and IDL's single-precision PI constant.
    weights = _coordinate_descent(
        out["transform"].T, observed, sigamp, out["amp"], out["viv"], 100.0, 0.5, 1000.0, 1000, False
    )
    assert_allclose(weights, out["weights"], rtol=1e-8, atol=1e-8)
    assert_allclose(weights @ out["viv"], 100.0, rtol=1e-12)


@pytest.mark.parametrize(
    "shape,pixel,center",
    [
        ((65, 65), (1.0, 1.0), (0.0, 0.0)),
        ((41, 51), (2.0, 1.5), (8.0, -6.0)),
    ],
)
def test_photometry_position_and_units(source_vis, shape, pixel, center):
    source_vis.phase_center = center * u.arcsec
    image = vis_cs(
        source_vis,
        shape=shape * u.pix,
        pixel_size=pixel * u.arcsec / u.pix,
        total_flux=100 * u.ph,
        seed=42,
        n_components=500,
        map=False,
    )
    assert image.shape == shape
    assert image.unit == u.ph / u.arcsec**2
    assert np.all(np.isfinite(image))
    assert np.all(image >= 0)
    values = image.value
    flux = values.sum() * np.prod(pixel)
    assert_allclose(flux, 100.0, rtol=0.01)
    yy, xx = np.indices(shape)
    # Visibilities are phased to the phase centre, so positions are relative to it.
    xcent = ((xx - (shape[1] - 1) / 2) * pixel[1] * values).sum() / values.sum()
    ycent = ((yy - (shape[0] - 1) / 2) * pixel[0] * values).sum() / values.sum()
    assert_allclose([xcent, ycent], [5.0, -3.0], atol=0.3)
    predicted = dft_map(
        image * np.prod(pixel) * u.arcsec**2,
        u=source_vis.u,
        v=source_vis.v,
        pixel_size=pixel * u.arcsec / u.pix,
    )
    # Compare to independent analytic source visibilities, including signed phases.
    assert np.linalg.norm(predicted - source_vis.visibilities) / np.linalg.norm(source_vis.visibilities) < 0.06


def test_reproducible_and_detector_flux(source_vis):
    # 63 samples from detector 1 and one from detector 2: detector means are equally weighted.
    n = len(source_vis.u)
    source_vis.meta["isc"] = np.r_[np.ones(n - 1), 2]
    source_vis.meta["totflux"] = np.r_[np.full(n - 1, 80.0), 120.0] * u.ph
    args = {"seed": 17, "n_components": 250, "map": False}
    inferred = vis_cs(source_vis, **args)
    explicit = vis_cs(source_vis, total_flux=100 * u.ph, **args)
    assert_allclose(inferred, explicit, rtol=0, atol=0)
    source_vis.meta.pop("isc")
    ungrouped = vis_cs(source_vis, **args)
    mean_flux = vis_cs(source_vis, total_flux=source_vis.meta["totflux"].mean(), **args)
    assert_allclose(ungrouped, mean_flux, rtol=0, atol=0)


def test_map_output(source_vis):
    source_vis.phase_center = (8, -6) * u.arcsec
    args = {"shape": (41, 51) * u.pix, "pixel_size": (2.0, 1.5) * u.arcsec / u.pix, "total_flux": 100 * u.ph}
    vis_map = vis_cs(source_vis, seed=3, n_components=200, **args)
    image = vis_cs(source_vis, seed=3, n_components=200, map=False, **args)
    assert vis_map.quantity.unit == u.ph / u.arcsec**2
    assert_allclose(vis_map.quantity, image)
    assert_allclose(u.Quantity(vis_map.scale), [1.5, 2.0] * u.arcsec / u.pix)
    center = vis_map.pixel_to_world(*(([51, 41] * u.pix - 1 * u.pix) / 2))
    assert_allclose([center.Tx.to_value(u.arcsec), center.Ty.to_value(u.arcsec)], [-6, 8], atol=1e-9)


def test_dictionary_refinement_and_circular_option():
    rng = np.random.default_rng(9)
    shape, pixel = (55, 65), np.array([2.0, 1.0])
    d = _random_words(rng, 400, shape, pixel, 1.5, 60.0, 1.0)
    assert_allclose(d.sx, d.sy)
    assert np.all(np.abs(d.x) <= 65 / 2)
    assert np.all(np.abs(d.y) <= 55)
    refined = _refine(d[[2, 8, 25]], rng, 400, shape, pixel, 1.5, 60.0, 1.0)
    assert len(refined) == 400
    assert_allclose(refined.sx, refined.sy)
    assert_allclose(refined.x[:3], d.x[[2, 8, 25]])


def test_out_of_view_words_have_zero_volume():
    # Coarse pixels let perturbed copies of an edge word land wholly outside the FOV,
    # which vis_cs drops instead of failing on.
    shape, pixel = (65, 65), np.array([4.0, 4.0])
    edge = _Gaussians(*(np.array([v]) for v in (-128.0, 0.0, 1.5, 1.5, 0.0)))
    viv = _in_view_volume(_refine(edge, np.random.default_rng(0), 400, shape, pixel, 1.5, 60.0, 2.5), shape, pixel)
    assert np.all(np.isfinite(viv))
    assert np.all(viv >= 0)
    assert np.any(viv == 0)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"sparseness": 1.0}, "sparseness"),
        ({"sparseness": -0.1}, "sparseness"),
        ({"n_components": 1}, "n_components"),
        ({"n_components": 2.5}, "n_components"),
        ({"iterations": 0}, "iterations"),
        ({"maxiter": 0}, "maxiter"),
        ({"regularisation": 0}, "regularisation"),
        ({"max_anisotropy": 0.5}, "max_anisotropy"),
        ({"min_sigma": 0 * u.arcsec}, "min_sigma"),
        ({"max_sigma": 1 * u.arcsec}, "min_sigma"),
        ({"shape": (2.5, 3) * u.pix}, "integer"),
        ({"pixel_size": 0 * u.arcsec / u.pix}, "pixel_size"),
        ({"total_flux": 0 * u.ph}, "total_flux"),
        ({"total_flux": [1, 2] * u.ph}, "total_flux"),
    ],
)
def test_invalid_options(source_vis, options, match):
    args = {"total_flux": 100 * u.ph, "n_components": 10}
    args.update(options)
    with pytest.raises(ValueError, match=match):
        vis_cs(source_vis, **args)


def test_missing_flux_and_uncertainty(source_vis):
    with pytest.raises(ValueError, match="total_flux"):
        vis_cs(source_vis)
    bare = Visibilities(source_vis.visibilities, source_vis.u, source_vis.v)
    with pytest.raises(ValueError, match="uncertainties"):
        vis_cs(bare, total_flux=100 * u.ph)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_uncertainties(source_vis, bad):
    error = np.ones(len(source_vis.u))
    error[0] = bad
    vis = Visibilities(source_vis.visibilities, source_vis.u, source_vis.v, amplitude_uncertainty=error * u.ph)
    with pytest.raises(ValueError, match="uncertainties"):
        vis_cs(vis, total_flux=100 * u.ph)


def test_missing_or_bad_detector_flux(source_vis):
    source_vis.meta["totflux"] = [1.0, np.nan]
    with pytest.raises(ValueError, match="totflux"):
        vis_cs(source_vis)
    source_vis.meta["totflux"] = np.ones(len(source_vis.u))
    source_vis.meta["isc"] = [1, 2]
    with pytest.raises(ValueError, match="isc"):
        vis_cs(source_vis)


def test_nonconvergence_is_reported(source_vis):
    with pytest.warns(RuntimeWarning, match="maxiter"):
        vis_cs(source_vis, total_flux=100 * u.ph, n_components=20, maxiter=1, seed=4)


def test_numerical_abort_and_small_dictionary():
    # At these frequencies every Gaussian Fourier coefficient underflows to zero.
    vis = Visibilities(
        np.zeros(2, dtype=complex) * u.ph,
        [1000, 2000] / u.arcsec,
        [1000, 2000] / u.arcsec,
        amplitude_uncertainty=np.ones(2) * u.ph,
    )
    image = vis_cs(vis, total_flux=100 * u.ph, n_components=2, seed=7, abort_numerical_issues=True, map=False)
    assert np.all(image.value == 0)
    partial = vis_cs(vis, total_flux=100 * u.ph, n_components=2, seed=7, abort_numerical_issues=False, map=False)
    assert np.all(np.isfinite(partial))
    assert partial.value.max() > 0


@pytest.mark.parametrize("seed", [0, 10, 20, 23])
def test_sole_active_word_keeps_flux_constraint(seed):
    # All flux in one word leaves it the only active word, at a rounding-level distance from its
    # maximum; SSW's update then moved it (to 16% of the flux for seed 0) and broke the constraint.
    rng = np.random.default_rng(seed)
    n_words = int(rng.integers(2, 6))
    uu, vv = rng.uniform(-0.1, 0.1, (2, 6))
    words = _random_words(rng, n_words, (9, 9), np.ones(2), 1.0, 3.0, 1.5)
    viv = _in_view_volume(words, (9, 9), np.ones(2))
    transform = _fourier(words, uu, vv)
    weights = _coordinate_descent(
        transform, 10 * transform[:, -1], np.ones(6), words.amp, viv, 10.0, 0.5, 1000.0, 50, False
    )
    assert_allclose((weights * viv).sum(), 10, rtol=1e-12)


@pytest.mark.parametrize("abort_numerical_issues", [False, True])
def test_single_surviving_word_keeps_flux_constraint(source_vis, monkeypatch, abort_numerical_issues):
    # Filtering removes the second word, leaving one word to carry all the flux.
    words = _Gaussians(
        x=np.array([5.0, 1000.0]),
        y=np.array([-3.0, 1000.0]),
        sx=np.full(2, 4.0),
        sy=np.full(2, 4.0),
        phi=np.zeros(2),
    )
    monkeypatch.setattr("xrayvision.vis_cs._random_words", lambda *args: words)
    image = vis_cs(
        source_vis,
        total_flux=100 * u.ph,
        n_components=2,
        iterations=1,
        abort_numerical_issues=abort_numerical_issues,
        map=False,
    )
    # The retained source is far from the image boundary and resolved by the pixels.
    assert np.all(np.isfinite(image))
    assert_allclose(image.sum() * u.arcsec**2, 100 * u.ph, rtol=1e-6)


@pytest.mark.parametrize("shape,pixel", [((65, 65), (10.0, 10.0)), ((513, 513), (4.0, 4.0)), ((47, 61), (20.0, 15.0))])
def test_in_view_volume_resolves_narrow_peaks(shape, pixel):
    # Include the reported missed peak, mixed widths, boundaries and wholly outside words.
    ny, nx = shape
    dy, dx = pixel
    left, bottom = -(nx - 1) * dx / 2, -(ny - 1) * dy / 2
    x = np.array([0.0, 5.0, left, left + nx * dx, left - 200.0, 15.0])
    y = np.array([180.0, -40.0, bottom, bottom + ny * dy, bottom - 200.0, 11.0])
    sigma = np.array([1.5, 1.5, 2.0, 2.0, 1.5, 60.0])
    words = _Gaussians(x, y, sigma, sigma, np.zeros(len(x)))
    # Independent analytic rectangle integral for circular Gaussians.
    root = np.sqrt(2) * sigma
    expected = (
        0.25
        * (erf((left + nx * dx - x) / root) - erf((left - x) / root))
        * (erf((bottom + ny * dy - y) / root) - erf((bottom - y) / root))
    )
    assert_allclose(_in_view_volume(words, shape, np.array(pixel)), expected, atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize("ratio", [2.5, 40.0, 10000.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_rotated_gaussian_quadrant_volume(ratio, sign):
    # A corner at the mean gives a known bivariate-normal quadrant probability.
    shape = (129, 129)
    pixel = np.full(2, 1.5 * ratio)
    corner = -(shape[0] - 1) * pixel[0] / 2
    words = _Gaussians(*(np.array([value]) for value in (corner, corner, 1.5, 1.5 * ratio, sign * np.pi / 4)))
    rho = sign * (1 - ratio**2) / (1 + ratio**2)  # SSW's retained cross-term sign
    expected = 0.25 + np.arcsin(rho) / (2 * np.pi)
    assert_allclose(_in_view_volume(words, shape, pixel), expected, atol=1e-10, rtol=1e-7)
