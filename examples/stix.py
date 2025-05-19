"""
======================================
Solar Orbiter/STIX Visibility Imaging
======================================

Imports

"""

import pickle

import astropy.units as apu
import matplotlib.pyplot as plt
import numpy as np

from xrayvision.clean import vis_clean
from xrayvision.forward_fit import vis_forward_fit
from xrayvision.imaging import vis_psf_map, vis_to_map
from xrayvision.mem import mem, resistant_mean

###############################################################################
# Create images from STIX visibility data.
#
# The STIX data has already been prepared and stored in python pickle format
# the variables can be simply restored.

with open("./stix_vis.pkl", "rb") as file:
    stix_data = pickle.load(file)

time_range = stix_data["time_range"]
energy_range = stix_data["energy_range"]
stix_vis = stix_data["stix_visibilities"]

###############################################################################
# Lets have a look at the point spread function (PSF) or dirty beam

psf_map = vis_psf_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=1 * apu.arcsec / apu.pix, scheme="uniform")
psf_map.plot()

###############################################################################
# Back projection

backproj_map = vis_to_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=1 * apu.arcsec / apu.pix, scheme="uniform")
backproj_map.plot()

###############################################################################
# Clean

clean_map, model_map, resid_map = vis_clean(
    stix_vis,
    shape=[129, 129] * apu.pixel,
    pixel_size=[1, 1] * apu.arcsec / apu.pix,
    clean_beam_width=20 * apu.arcsec,
    niter=100,
)
clean_map.plot()

###############################################################################
# MEM

# Compute percent_lambda
snr_value, _ = resistant_mean((np.abs(stix_vis.visibilities) / stix_vis.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90)

mem_map = mem(
    stix_vis, shape=[129, 129] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pix, percent_lambda=percent_lambda
)

###############################################################################
# VIS_FWD_FIT

vis_fwd_map = vis_forward_fit(stix_vis, shape=[129, 129] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pix)

vis_fwd_pso_map = vis_forward_fit(
    stix_vis, method="PSO", shape=[129, 129] * apu.pixel, pixel_size=[1, 1] * apu.arcsec / apu.pix
)

###############################################################################
# Comparison
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(231, projection=psf_map)
fig.add_subplot(232, projection=backproj_map)
fig.add_subplot(233, projection=clean_map)
fig.add_subplot(234, projection=mem_map)
fig.add_subplot(235, projection=mem_map)
fig.add_subplot(236, projection=mem_map)

axs = fig.get_axes()
psf_map.plot(axes=axs[0])
axs[0].set_title("PSF")
backproj_map.plot(axes=axs[1])
axs[1].set_title("Back Projection")
clean_map.plot(axes=axs[2])
axs[2].set_title("Clean")
mem_map.plot(axes=axs[3])
axs[3].set_title("MEM")
vis_fwd_map.plot(axes=axs[4])
axs[4].set_title("VIS_FWRDFIT")
vis_fwd_pso_map.plot(axes=axs[5])
axs[5].set_title("VIS_FWRDFIT_PSO")

plt.show()
