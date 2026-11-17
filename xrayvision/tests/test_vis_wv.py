import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import center_of_mass

import astropy.units as apu

from xrayvision.imaging import vis_psf_image
from xrayvision.simulation.instruments import rhessi_like_uv_coverage
from xrayvision.vis_wv import _fiwt, _fiwt_spectra, _ifiwt, vis_wv
from xrayvision.visibility import Visibilities


@pytest.fixture
def gaussian_vis():
    # Circular Gaussian (flux 100, sigma 4 arcsec) at (x, y) = (10, -5) arcsec seen by RHESSI-like detectors 3-9
    uv = rhessi_like_uv_coverage()
    u, v = uv["u"][2:].flatten(), uv["v"][2:].flatten()
    x0, y0, sigma = 10.0, -5.0, 4.0
    vis_vals = 100 * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2).value)
    vis_vals = vis_vals * np.exp(2j * np.pi * (x0 * u.value + y0 * v.value))
    return Visibilities(visibilities=vis_vals * apu.Unit("ph cm-2 s-1"), u=u, v=v), (x0, y0)


@pytest.mark.parametrize("pixel", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_fiwt_parseval(pixel):
    image = np.random.default_rng(0).random((65, 65))
    psi = _fiwt_spectra(65, 3, pixel)
    assert_allclose((psi**2).sum(axis=0), 1)
    coeffs = _fiwt(image, psi)
    assert coeffs.shape == (4, 65, 65)
    assert_allclose(_ifiwt(coeffs, psi), image)
    assert_allclose((coeffs**2).sum(), (image**2).sum())


def test_vis_wv_bad_inputs(gaussian_vis):
    vis, _ = gaussian_vis
    with pytest.raises(ValueError, match="odd"):
        vis_wv(vis, shape=(64, 64) * apu.pix)
    with pytest.raises(ValueError, match="odd"):
        vis_wv(vis, shape=(65, 63) * apu.pix)
    with pytest.raises(ValueError, match="pixel_size"):
        vis_wv(vis, shape=(65, 65) * apu.pix, pixel_size=(1, 2) * apu.arcsec / apu.pix)


def test_vis_wv(gaussian_vis):
    vis, (x0, y0) = gaussian_vis
    wv_map = vis_wv(vis, shape=(65, 65) * apu.pix, pixel_size=1 * apu.arcsec / apu.pix)
    assert wv_map.unit == apu.Unit("ph cm-2 s-1 arcsec-2")
    assert np.all(wv_map.data >= 0)
    # images are indexed [y, x] centred on pixel 32, the top of the reconstruction is flat so use the centroid
    assert_allclose(center_of_mass(wv_map.data), (32 + y0, 32 + x0), atol=1)
    # flux is constrained to the shortest baseline before negatives are clipped, which adds a little
    dc = vis.visibilities.real[np.argmin(np.hypot(vis.u, vis.v))].value
    assert dc <= wv_map.data.sum() < 1.1 * dc


@pytest.mark.parametrize("include_dc", [True, False], ids=["with-dc", "without-dc"])
def test_vis_wv_fourier_grid(include_dc):
    # Complete Fourier-grid sampling gives an exact point-source reconstruction.
    # Removing the zero-frequency sample makes the PSF sum zero; the flux
    # constraint still supplies the missing information.
    n, flux = 65, 100.0
    frequencies = np.arange(-(n // 2), n // 2 + 1) / n
    u, v = np.meshgrid(frequencies, frequencies)
    selected = np.ones(u.shape, dtype=bool) if include_dc else (u**2 + v**2 > 0)
    u, v = u[selected], v[selected]
    # A point source one pixel right of centre has this analytic visibility.
    unit = apu.Unit("ph cm-2 s-1")
    vis = Visibilities(flux * np.exp(2j * np.pi * u) * unit, u / apu.arcsec, v / apu.arcsec)
    shape = (n, n) * apu.pix
    pixel_size = 1 * apu.arcsec / apu.pix
    if not include_dc:
        psf = vis_psf_image(vis, shape=shape, pixel_size=pixel_size)
        assert_allclose(psf.value.sum(), 0, atol=1e-10)

    image = vis_wv(vis, shape=shape, pixel_size=pixel_size, lam=0, map=False)
    expected = np.zeros((n, n))
    expected[n // 2, n // 2 + 1] = flux
    assert_allclose(image.to_value(unit / apu.arcsec**2), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.idl
@pytest.mark.parametrize(("pixel", "lam"), [(1.0, None), (2.0, None), (1.0, 0.001)])
def test_vis_wv_idl(gaussian_vis, pixel, lam, monkeypatch):
    hissw = pytest.importorskip("hissw")
    # vis_wv.pro centres the PSF with shift(P, 1-center) which leaves its peak one pixel off the FFT origin, so the
    # IDL image is shifted one pixel down in x and y, emulate that here to check the rest of the algorithm matches
    monkeypatch.setattr("xrayvision.vis_wv.vis_psf_image", lambda *a, **k: np.roll(vis_psf_image(*a, **k), 1, (0, 1)))
    vis, _ = gaussian_vis
    ssw = hissw.Environment(ssw_packages=["hessi"])
    # force_double_precision as IDL parses plain literals as single precision
    script = """
obsvis = complex({{re | force_double_precision}}, {{im | force_double_precision}})
vis = replicate({u:0d, v:0d, obsvis:dcomplex(0, 0), totflux:0., sigamp:0., xyoffset:[0., 0.], trange:[0d, 0d]}, n_elements(obsvis))
vis.u = {{u | force_double_precision}}
vis.v = {{v | force_double_precision}}
vis.obsvis = obsvis
image = vis_wv(vis, imsize=[65, 65], pixel=[{{pixel}}, {{pixel}}], autolam={{autolam}}, lam={{lam}})
    """
    out = ssw.run(
        script=script,
        args={
            "u": vis.u.value,
            "v": vis.v.value,
            "re": vis.visibilities.value.real,
            "im": vis.visibilities.value.imag,
            "pixel": pixel,
            "autolam": int(lam is None),
            "lam": 0.05 if lam is None else lam,
        },
    )
    image = vis_wv(vis, shape=(65, 65) * apu.pix, pixel_size=pixel * apu.arcsec / apu.pix, lam=lam, map=False)
    image_idl = out["image"]
    assert_allclose(image.value, image_idl, atol=1e-4 * image_idl.max())
