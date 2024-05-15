"""
======================================
RHESSI Visibility Imaging
======================================

Create images from RHESSI visibility data
"""

import astropy.units as apu
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_psf_map, vis_to_map
from xrayvision.mem import mem
from xrayvision.visibility import Visibility

###############################################################################
# We will use `astropy.io.fits` to download and open the RHESSI visibility fits
# file

hdul = fits.open(
    "https://hesperia.gsfc.nasa.gov/rhessi_extras/visibility_fits_v2/"
    "2002/02/21/hsi_vis_20020221_2357_0054_46tx3e.fits"
)

###############################################################################
# No lets extract the visibility data the first thing we will do is filter the
# visibilities by time. We will extract the 3rd integration time range.

times = np.unique(hdul[-1].data["TRANGE"], axis=0)
time_index, _ = np.where(hdul[-1].data["TRANGE"] == times[6])
vis_data = hdul[3].data[time_index]


###############################################################################
# Next lets filter by energy range in this case 12 - 25 keV

energy_index, _ = np.where(vis_data["ERANGE"] == [12.0, 25.0])
vis_data = vis_data[energy_index]

###############################################################################
# Now lets filter by ISC or detector to remove possibly bad data in this case
# need to remove ISC 0 and 1.

vis_data = vis_data[vis_data["isc"] > 1]
vis_data = vis_data[vis_data["obsvis"] != 0 + 0j]

###############################################################################
# Now we can create the visibility object from the filtered visibilities.

vunit = apu.Unit("photon/(cm**2 s)")
vis = Visibility(
    vis=vis_data["obsvis"] * vunit,
    u=vis_data["u"] / apu.arcsec,
    v=vis_data["v"] / apu.arcsec,
    offset=vis_data["xyoffset"][0] * apu.arcsec,
)
setattr(vis, "amplitude_error", vis_data["sigamp"] * vunit)
setattr(vis, "isc", vis_data["isc"])


###############################################################################
# Lets have a look at the point spread function (PSF) or dirty beam

psf_map = vis_psf_map(vis, shape=(101, 101) * apu.pixel, pixel_size=1.5 * apu.arcsec / apu.pixel, scheme="uniform")

###############################################################################
# We can now make an image using the back projection algorithm essentially and
# inverse Fourier transform of the visibilities.

backproj_map = vis_to_map(vis, shape=[101, 101] * apu.pixel, pixel_size=1.5 * apu.arcsec / apu.pix)

###############################################################################
# Back projection contain many artifact due to the incomplete sampling of the u-v
# plane as a result various algorithms have been developed to remove or deconvolve
# this effect. CLEAN is one of the oldest and simplest, a CLEAN image can be made.

# vis_data_59 = vis_data[vis_data['isc'] > 3]
#
# vis_59 = Visibility(vis=vis_data_59['obsvis']*apu.Unit('ph/cm*s'), u=vis_data_59['u']/apu.arcsec,
#                     v=vis_data_59['v']/apu.arcsec, offset=vis_data_59['xyoffset'][0]*apu.arcsec)

clean_map, model_map, residual_map = vis_clean(
    vis,
    shape=[101, 101] * apu.pixel,
    pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix,
    clean_beam_width=10 * apu.arcsec,
    niter=100,
)

###############################################################################
# MEM

mem_map = mem(vis, shape=[129, 129] * apu.pixel, pixel=[2, 2] * apu.arcsec / apu.pix)
mem_map.plot()


###############################################################################
# Comparison
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(221, projection=psf_map)
fig.add_subplot(222, projection=backproj_map)
fig.add_subplot(223, projection=clean_map)
fig.add_subplot(224, projection=mem_map)
axs = fig.get_axes()
psf_map.plot(axes=axs[0])
axs[0].set_title("PSF")
backproj_map.plot(axes=axs[1])
axs[1].set_title("Back Projection")
clean_map.plot(axes=axs[2])
axs[2].set_title("Clean")
mem_map.plot(axes=axs[3])
axs[3].set_title("MEM")
plt.show()
