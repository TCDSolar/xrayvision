"""
======================================
RHESSI Visibility Imaging
======================================

Create images from RHESSI visibility data
"""
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as apu
from astropy.io import fits

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image, vis_to_map
from xrayvision.visibility import Visibility

###############################################################################
# We will use `astropy.io.fits` to download and open the RHESSI visibility fits
# file
hdul = fits.open('https://hesperia.gsfc.nasa.gov/rhessi_extras/visibility_fits'
                 '/2002/02/20/hsi_vis_20020220_2102_2125_17tx4e.fits')

###############################################################################
# No lets extract the visibility data the first thing we will do is filter the
# visibilities by time. We will extract the 3rd integration time range.

times = np.unique(hdul[-1].data['TRANGE'], axis=0)
time_index, _ = np.where(hdul[-1].data['TRANGE'] == times[2])
vis_data = hdul[3].data[time_index]

###############################################################################
# Next lets filter by energy range in this case 12 - 25 keV

energy_index, _ = np.where(vis_data['ERANGE'] == [12.0, 25.0])
vis_data = vis_data[energy_index]

###############################################################################
# Now lets filter by ISC or detector to remove possibly bad data in this case
# need to remove ISC 0 and 1.

vis_data = vis_data[vis_data['isc'] > 1]

###############################################################################
# Now we can create the visibility object from the filtered visibilities.

vis = Visibility(vis=vis_data['obsvis'], u=vis_data['u']/apu.arcsec, v=vis_data['v']/apu.arcsec,
                 offset=vis_data['xyoffset'][0]*apu.arcsec)

###############################################################################
# We can no make an image using the back projection algorithm essentially and
# inverse Fourier transform of the visibilities.

rhessi_image = vis_to_image(vis, shape=[101, 101]*apu.pixel,
                            pixel_size=1.5*apu.arcsec, natural=False)

plt.imshow(rhessi_image, origin='lower')  # IDL indices start or bottom left

###############################################################################
# We can no make an image using the back projection algorithm essentially and
# inverse Fourier transform of the visibilities.

rhessi_map = vis_to_map(vis, shape=[101, 101]*apu.pixel,
                        pixel_size=1.5*apu.arcsec, natural=False)

rhessi_map.peek()

###############################################################################
# Back projection contain many artifact due to the incomplete sampling of the u-v
# plane as a result various algorithms have been developed to remove or deconvolve
# this effect. CLEAN is one of the oldest and simplest, a CLEAN image can be made.

vis_data_59 = vis_data[vis_data['isc'] > 3]

vis_59 = Visibility(vis=vis_data_59['obsvis']*apu.Unit('ph/cm*s'), u=vis_data_59['u']/apu.arcsec,
                    v=vis_data_59['v']/apu.arcsec, offset=vis_data_59['xyoffset'][0]*apu.arcsec)

clean_map, model_map, residual_map = vis_clean(vis_59, shape=[101, 101]*apu.pixel,
                                               pixel=[1.5, 1.5]*apu.arcsec,
                                               clean_beam_width=10*apu.arcsec,
                                               niter=100,  natural=False)

clean_map.peek()
