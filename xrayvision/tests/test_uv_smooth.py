import astropy.units as apu
from scipy.io import readsav

from xrayvision.uv_smooth import uv_smooth
from xrayvision.visibility import Visibilities, VisMeta


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

    vis_sav = readsav("/Users/sm/hsi_hsi_20020221_0006-0007_12-25.save")
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
