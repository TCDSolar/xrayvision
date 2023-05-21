"""
======================================
Solar Orbiter/STIX Visibility Imaging
======================================

Create images from STIX visibility data
"""
import pickle

import matplotlib.pyplot as plt

import astropy.units as apu

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_psf_map, vis_to_map
from xrayvision.mem import mem

###############################################################################
# The STIX data has already been prepared and stored in python pickle format
# the variables can be simply restored.

with open('/Users/shane/Projects/stixpy/examples/stix_vis.pkl', 'rb') as f:
    stix_data = pickle.load(f)

time_range, energy_range, offset, stix_vis = stix_data
stix_vis.offset = offset

###############################################################################
# Lets have a look at the point spread function (PSF) or dirty beam

psf_map = vis_psf_map(stix_vis, shape=(129, 129)*apu.pixel,
                      pixel_size=2*apu.arcsec,
                      natural=False)
psf_map.plot()

###############################################################################
# Back projection

backproj_map = vis_to_map(stix_vis, shape=(129, 129)*apu.pixel,
                          pixel_size=2*apu.arcsec, natural=False)
backproj_map.plot()
###############################################################################
# Clean

clean_map, model_map, resid_map = vis_clean(stix_vis, shape=[129, 129]*apu.pixel,
                                            pixel=[2, 2]*apu.arcsec, niter=100,
                                            clean_beam_width=20*apu.arcsec)
clean_map.plot()
###############################################################################
# MEM

mem_map = mem(stix_vis, shape=[129, 129]*apu.pixel, pixel=[2, 2]*apu.arcsec)
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
backproj_map.plot(axes=axs[1])
clean_map.plot(axes=axs[2])
mem_map.plot(axes=axs[3])
print(1)
