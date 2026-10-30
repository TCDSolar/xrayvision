import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as apu
from astropy.io import fits

from xrayvision.simulation.instruments import rhessi_like_uv_coverage
from xrayvision.uv_smooth import uv_smooth, uv_smooth_idl
from xrayvision.visibility import Visibilities, VisMeta


@pytest.mark.remote_data
def test_uv_smooth():
    hdul = fits.open(
        "https://umbra.nascom.nasa.gov/rhessi/rhessi_extras/visibility_fits_v2/2002/02/21/hsi_vis_20020221_2357_0054_46tx3e.fits"
    )
    times = np.unique(hdul[-1].data["TRANGE"], axis=0)

    index = np.argwhere(
        (np.all(hdul[3].data["TRANGE"] == times[7], axis=1)) & (np.all(hdul[3].data["ERANGE"] == [12.0, 25.0], axis=1))
    )
    vis_data = hdul[3].data[index.squeeze()]

    ###############################################################################
    # Now lets filter by ISC or detector to remove possibly bad data in this case
    # need to remove ISC 0 and 1.
    vis_data = vis_data[vis_data["isc"] > 2]
    vis_data = vis_data[vis_data["obsvis"] != 0 + 0j]

    # vis_sav = readsav("/Users/sm/hsi_hsi_20020221_0006-0007_12-25.sav")
    # vis_data = vis_sav["vis"]
    ###############################################################################
    # Now we can create the visibility object from the filtered visibilities.
    meta = VisMeta({"isc": vis_data["isc"]})

    vunit = apu.Unit("photon/(cm**2 s)")
    vis = Visibilities(
        visibilities=vis_data["obsvis"] * vunit,
        u=vis_data["u"] / apu.arcsec,
        v=vis_data["v"] / apu.arcsec,
        phase_center=vis_data["xyoffset"][0] * apu.arcsec,
        meta=meta,
        amplitude_uncertainty=vis_data["sigamp"] * vunit,
    )

    image_orig, *_ = uv_smooth_idl(vis)
    image_new, *_ = uv_smooth(vis, shape=128, uv_pixel_size=0.0005)
    assert_allclose(image_new, image_orig, atol=5e-5)


@pytest.fixture
def rhessi_like_gaussian_vis():
    """
    Synthetic RHESSI-like visibilities of a circular Gaussian source at the origin.

    Samples a circular Gaussian (flux=100 ph/cm²/s, sigma=5 arcsec) at u,v points
    arranged on annuli at spatial frequencies corresponding to RHESSI subcollimators
    2-6, with 12 rotation samples per annulus.

    Returns
    -------
    vis : Visibilities
        Synthetic visibilities ready for uv_smooth.
    flux : float
        True source flux (ph/cm^2/s).
    sigma : float
        True source sigma (arcsec).
    """
    uv = rhessi_like_uv_coverage()
    # detectors 3-7 or isc 2-6
    u, v, isc = uv["u"][2:7].flatten(), uv["v"][2:7].flatten(), uv["isc"][2:7].flatten()

    flux = 100.0  # ph/cm^2/s
    sigma = 5.0 * apu.arcsec

    # V(u,v) = flux * exp(-2 pi^2 sigma^2 (u^2 + v^2))  [source at origin, so no phase term]
    vis_vals = flux * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2))

    vis = Visibilities(
        visibilities=vis_vals * apu.Unit("ph cm-2 s-1"),
        u=u,
        v=v,
        meta=VisMeta({"isc": isc}),
    )
    return vis, flux, sigma


def test_uv_smooth_peak_at_origin(rhessi_like_gaussian_vis):
    vis, *_ = rhessi_like_gaussian_vis

    image, *_ = uv_smooth_idl(vis, niter=50)
    # For a source centered at (0, 0) the peak should lie at the image center
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    center = np.array(image.shape) // 2

    assert abs(peak_idx[0] - center[0]) <= 3, f"Peak row {peak_idx[0]} too far from image center {center[0]}"
    assert abs(peak_idx[1] - center[1]) <= 3, f"Peak col {peak_idx[1]} too far from image center {center[1]}"


def test_uv_smooth_matches_gaussian(rhessi_like_gaussian_vis):
    vis, _, sigma = rhessi_like_gaussian_vis
    sigma = sigma.value

    image_orig, *_ = uv_smooth_idl(vis, niter=50)
    image_new, *nw = uv_smooth(vis, shape=128, uv_pixel_size=0.0005, niter=50)
    image_auto, *_ = uv_smooth(vis, shape=128, pixel_size=1.0)

    im_new = image_orig.shape[0]

    # Build the reference Gaussian on the same pixel grid as the uv_smooth output
    pixel_size = nw[1]  # _uv_smooth_pixel_scale()  # arcsec/pixel
    coords = (np.arange(im_new) - im_new // 2) * pixel_size  # arcsec
    xx, yy = np.meshgrid(coords, coords)
    ref_image = (100.0 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    np.testing.assert_allclose(image_orig, ref_image, atol=0.025)
    np.testing.assert_allclose(image_new, ref_image, atol=0.025)
    np.testing.assert_allclose(image_auto, ref_image, atol=0.024)


@pytest.mark.idl
@pytest.mark.parametrize("offset", [(0.0, 0.0), (12.0, 0.0), (0.0, 12.0)], ids=["centre", "x-offset", "y-offset"])
def test_uv_smooth_idl(rhessi_like_gaussian_vis, offset):
    hissw = pytest.importorskip("hissw")
    vis, *_ = rhessi_like_gaussian_vis
    # Offset sources break the transpose symmetry a centred Gaussian hides
    phase = np.exp(2j * np.pi * (offset[0] * vis.u.value + offset[1] * vis.v.value))
    vis = Visibilities(visibilities=vis.visibilities * phase, u=vis.u, v=vis.v, meta=vis.meta)

    ssw = hissw.Environment(ssw_packages=["hessi"])
    # force_double_precision as IDL parses plain literals as single precision
    script = """
obsvis = complex({{re | force_double_precision}}, {{im | force_double_precision}})
visin = {u:{{u | force_double_precision}}, v:{{v | force_double_precision}}, obsvis:obsvis, isc:{{isc | list}}, xyoffset:[0,0], trange:[0,0]}
uv_smooth, visin, map, reconstructed_map_visibilities=visout
    """
    out = ssw.run(
        script=script,
        args={
            "u": vis.u.value,
            "v": vis.v.value,
            "re": vis.visibilities.value.real,
            "im": vis.visibilities.value.imag,
            "isc": vis.meta["isc"].tolist(),
        },
    )
    image_orig, _, pixel_orig = uv_smooth_idl(vis, niter=50)
    image_new, _, pixel_new = uv_smooth(vis, shape=128, uv_pixel_size=0.0005, niter=50)
    image_idl = out["map"]["data"][0]
    # IDL's GRID_TPS differs from the exact thin plate spline by ~4e-5 (of a 0.6 peak) on the 320x320 grid
    assert_allclose(image_orig, image_idl, atol=1e-4)
    assert_allclose(image_new, image_idl, atol=1e-4)
    dx = np.ravel(out["map"]["dx"])[0]
    assert_allclose(pixel_orig, dx, rtol=1e-5)
    assert_allclose(pixel_new, dx, rtol=1e-5)
