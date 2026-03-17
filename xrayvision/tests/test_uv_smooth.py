import astropy.units as apu
import hissw
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import readsav

from xrayvision.uv_smooth import uv_smooth, uv_smooth_alt
from xrayvision.visibility import Visibilities, VisMeta


@pytest.mark.skip("needs local files")
def test_uv_smooth():
    # hdul = fits.open("~/Downloads/hsi_vis_20020221_2357_0054_46tx3e.fits")
    # times = np.unique(hdul[-1].data["TRANGE"], axis=0)
    #
    # index = np.argwhere((np.all(hdul[3].data["TRANGE"] == times[6],axis=1))
    #                     & (np.all(hdul[3].data["ERANGE"] == [12.0, 25.0], axis=1)))
    # vis_data = hdul[3].data[index.squeeze()]
    #
    # ###############################################################################
    # # Now lets filter by ISC or detector to remove possibly bad data in this case
    # # need to remove ISC 0 and 1.
    # vis_data = vis_data[vis_data["isc"] > 1]
    # vis_data = vis_data[vis_data['v'] > 0]
    # # vis_data = vis_data[vis_data["obsvis"] != 0 + 0j]

    vis_sav = readsav("/Users/sm/hsi_hsi_20020221_0006-0007_12-25.sav")
    vis_data = vis_sav["vis"]
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

    image, vis = uv_smooth(vis)
    print("here")


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
    # Approximate RHESSI spatial frequencies (arcsec^-1) for subcollimators 2-6,
    # based on typical RHESSI grid pitches of ~60, 35, 20, 11.5, 7 arcsec.
    isc_ids = np.array([2, 3, 4, 5, 6])
    radii = np.array([0.0083, 0.0143, 0.0250, 0.0435, 0.0714])  # arcsec^-1

    # 12 evenly-spaced rotation angles per detector (simulates spacecraft rotation)
    n_angles = 32
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    u_parts = [r * np.cos(angles) for r in radii]
    v_parts = [r * np.sin(angles) for r in radii]
    isc_parts = [np.full(n_angles, isc, dtype=int) for isc in isc_ids]

    u = np.concatenate(u_parts)
    v = np.concatenate(v_parts)
    isc = np.concatenate(isc_parts)

    flux = 100.0  # ph/cm^2/s
    sigma = 5.0  # arcsec

    # V(u,v) = flux * exp(-2π²σ²(u²+v²))  [source at origin, so no phase term]
    vis_vals = flux * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2))

    vis = Visibilities(
        visibilities=vis_vals * apu.Unit("ph cm-2 s-1"),
        u=u / apu.arcsec,
        v=v / apu.arcsec,
        meta=VisMeta({"isc": isc}),
    )
    return vis, flux, sigma


def _uv_smooth_pixel_scale():
    """
    Return the image pixel size (arcsec) produced by uv_smooth for detmin >= 2.

    Derived from the hardcoded UV grid parameters inside uv_smooth:
      pixel = 0.0005 arcsec^-1, N = 320, Nnew = 1920, downsample = 15.
    The image pixel size follows from the DFT relationship:
      pixel_size = 1 / (im_new * deltaomega).
    """
    pixel_uv = 0.0005  # arcsec^-1
    Nnew = 1920
    Ulimit = (Nnew / 2 - 1) * pixel_uv + pixel_uv / 2
    xpix = -Ulimit + np.arange(Nnew) * pixel_uv
    im_new = Nnew // 15
    xpixnew = xpix[np.arange(im_new) * 15 + 7]
    deltaomega = (xpixnew[-1] - xpixnew[0]) / im_new
    return 1.0 / (im_new * deltaomega)  # arcsec/pixel


def test_uv_smooth_peak_at_origin(rhessi_like_gaussian_vis):
    vis, flux, sigma = rhessi_like_gaussian_vis

    image, _ = uv_smooth(vis, niter=50)

    # For a source centered at (0, 0) the peak should lie at the image center
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    center = np.array(image.shape) // 2

    assert abs(peak_idx[0] - center[0]) <= 3, f"Peak row {peak_idx[0]} too far from image center {center[0]}"
    assert abs(peak_idx[1] - center[1]) <= 3, f"Peak col {peak_idx[1]} too far from image center {center[1]}"


def test_uv_smooth_matches_gaussian(rhessi_like_gaussian_vis):
    vis, flux, sigma = rhessi_like_gaussian_vis

    image, _ = uv_smooth(vis, niter=50)
    image_alt, _ = uv_smooth_alt(vis)
    im_new = image.shape[0]

    # Build the reference Gaussian on the same pixel grid as the uv_smooth output
    pixel_size = 1.0  # _uv_smooth_pixel_scale()  # arcsec/pixel
    coords = (np.arange(im_new) - im_new // 2) * pixel_size  # arcsec
    xx, yy = np.meshgrid(coords, coords)
    ref_image = (100.0 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    np.testing.assert_allclose(image, ref_image, atol=0.05)


def test_uv_smooth_idl(rhessi_like_gaussian_vis):
    vis, flux, sigma = rhessi_like_gaussian_vis
    # vis.u
    # vis.v
    # vis.obsvis
    # vis.isc

    ssw = hissw.Environment(ssw_packages=["hessi"])
    script = """
visin = {u:{{u | list }}, v:{{v | list }}, obsvis:{{obsvis | list}}, isc:{{isc|list}}, xyoffset:[0,0], trange:[0,0]}
uv_smooth, visin, map, reconstructed_map_visibilities=visout
    """
    out = ssw.run(
        script=script,
        args={
            "u": vis.u.value.tolist(),
            "v": vis.v.value.tolist(),
            "obsvis": vis.visibilities.value.tolist(),
            "isc": vis.meta["isc"].tolist(),
        },
    )
    image, _ = uv_smooth(vis, niter=50)
    image_idl = out["map"]["data"][0]
    assert_allclose(image, image_idl)
