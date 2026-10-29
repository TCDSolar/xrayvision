"""
======================================
Spectral Component Imaging
======================================

Separate the visibilities of two spectral components (e.g. thermal and
non-thermal emission) from a set of visibilities measured in multiple energy
bins using `~xrayvision.spectral.vis_spectral_components`, then image each
component separately.

This example uses a synthetic, two-component scene so that it is fully
self-contained. It is based on the method described in Stiefel et al. 2025
(A&A, 704, A316) for real STIX data see the `stixpy
<https://stixpy.readthedocs.io>`__ documentation.
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as apu

from xrayvision.clean import vis_clean
from xrayvision.imaging import image_to_vis
from xrayvision.spectral import vis_spectral_components
from xrayvision.visibility import Visibilities

rng = np.random.default_rng(3)

VUNIT = apu.ph / apu.cm**2

###############################################################################
# Set up a synthetic, two-component scene with a single, centrally located
# "thermal" source and two "non-thermal" footpoint sources with an
# energy-independent morphology.

shape = (65, 65) * apu.pixel
pixel_size = [1, 1] * apu.arcsec / apu.pix
n, m = shape.value.astype(int)
yy, xx = np.mgrid[0:n, 0:m]


def gaussian_source(center, sigma):
    cy, cx = center
    source = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return source / source.sum()


thermal_image = gaussian_source((32, 26), 3)
footpoints_image = gaussian_source((32, 40), 2) + gaussian_source((26, 40), 2)
footpoints_image /= footpoints_image.sum()

###############################################################################
# Define the energy bins, the fractional contribution of each component to
# the total flux in each bin, and the total flux itself. In a real analysis
# these would come from an independent spectral fit. Here a simple model is
# used where the thermal component dominates at low energies and the
# non-thermal component dominates at high energies.

energies = np.array([6, 10, 16, 25, 40, 60])
frac_thermal = 1 / (1 + (energies / 20) ** 4)
frac_nonthermal = 1 - frac_thermal
fractions = np.stack([frac_thermal, frac_nonthermal], axis=1)

total_flux = 1e4 * (energies / 6) ** -4 * VUNIT

###############################################################################
# Use a sparse, spiral (u, v) coverage typical of a Fourier-imaging
# instrument like STIX or RHESSI.

radii = np.logspace(np.log10(0.03), np.log10(0.45), 10)
theta = np.linspace(0, 2 * np.pi, 32, endpoint=False) + np.pi / 32
r, th = np.meshgrid(radii, theta, indexing="ij")
u = (r * np.cos(th)).flatten() / apu.arcsec
v = (r * np.sin(th)).flatten() / apu.arcsec

###############################################################################
# Create the visibilities for each energy bin by combining the two
# components according to their fractional contributions, scaling by the
# total flux, and adding representative noise.

sigma_frac = 0.02  # 2% amplitude uncertainty
vis_per_energy = []
for i in range(len(energies)):
    image = (fractions[i, 0] * thermal_image + fractions[i, 1] * footpoints_image) * total_flux[i]
    true_vis = image_to_vis(image, u=u, v=v, pixel_size=pixel_size)

    sigma = sigma_frac * total_flux[i]
    noise = (rng.normal(scale=sigma.value, size=u.shape) + 1j * rng.normal(scale=sigma.value, size=u.shape)) * VUNIT

    vis_per_energy.append(
        Visibilities(
            true_vis.visibilities + noise,
            u=u,
            v=v,
            amplitude_uncertainty=np.full(u.shape, sigma.value) * VUNIT,
        )
    )

###############################################################################
# Decompose the per-energy visibilities into visibilities of the thermal and
# non-thermal spectral components.

vis_thermal, vis_nonthermal = vis_spectral_components(vis_per_energy, fractions, normalization=total_flux)

###############################################################################
# Image each spectral component separately. As in Stiefel et al. 2025 (Sect.
# 2.4), any standard reconstruction method can be applied independently to
# the visibilities of each component; CLEAN is used here, but MEM
# (`~xrayvision.mem.mem`) works equally well and was used for the images
# published in that paper, with no significant differences found between the
# two.

clean_kwargs = dict(shape=shape, pixel_size=pixel_size, niter=100, clean_beam_width=6 * apu.arcsec)
clean_thermal, _, _ = vis_clean(vis_thermal, **clean_kwargs)
clean_nonthermal, _, _ = vis_clean(vis_nonthermal, **clean_kwargs)

###############################################################################
# For comparison, also image one of the energy bins directly, without
# spectral component imaging. Because both sources contribute in this bin,
# the standard energy-range image blends them together, whereas spectral
# component imaging (below) cleanly separates the two.

mixed_idx = np.argmin(np.abs(frac_thermal - 0.5))
clean_combined, _, _ = vis_clean(vis_per_energy[mixed_idx], **clean_kwargs)

###############################################################################
# Compare the combined, blended energy-range image (with the two components
# overlaid as contours) to the separated thermal and non-thermal component
# images.

fig = plt.figure(figsize=(14, 5))
ax0 = fig.add_subplot(131, projection=clean_combined)
ax1 = fig.add_subplot(132, projection=clean_thermal)
ax2 = fig.add_subplot(133, projection=clean_nonthermal)

clean_combined.plot(axes=ax0)
ax0.set_title(f"Combined ({energies[mixed_idx]} keV bin)")
levels = np.arange(20, 100, 20) * apu.percent
clean_thermal.draw_contours(levels=levels, colors="red", linewidths=0.8, axes=ax0)
clean_nonthermal.draw_contours(levels=levels, colors="blue", linewidths=0.8, axes=ax0)

clean_thermal.plot(axes=ax1)
ax1.set_title("Thermal component")

clean_nonthermal.plot(axes=ax2)
ax2.set_title("Non-thermal component")
plt.show()
