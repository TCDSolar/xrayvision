"""
===============================================
VIS-CS reconstruction of two Gaussian sources
===============================================

Reconstruct sparse synthetic visibilities with known total flux and amplitude uncertainties using an adaptive Gaussian dictionary.
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from xrayvision.vis_cs import vis_cs
from xrayvision.visibility import Visibilities

###############################################################################
# Create synthetic image and source visibilities with sample several half-plane Fourier annuli,
# with independent analytic source data.

angles = np.linspace(0, np.pi, 24, endpoint=False)
radii = np.array([0.007, 0.015, 0.03, 0.06, 0.12])
uu = (radii[:, None] * np.cos(angles)).ravel()
vv = (radii[:, None] * np.sin(angles)).ravel()
values = np.zeros(len(uu), dtype=complex)
shape = (81, 81)
coords = np.arange(81) - 40
xx, yy = np.meshgrid(coords, coords)
truth = np.zeros(shape)
for flux, x, y, sigma in [(80.0, -11.0, -3.0, 4.0), (20.0, 11.0, 4.0, 4.0)]:
    values += flux * np.exp(-2 * np.pi**2 * sigma**2 * (uu**2 + vv**2)) * np.exp(2j * np.pi * (uu * x + vv * y))
    truth += flux / (2 * np.pi * sigma**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

rng = np.random.default_rng(17)
sigma = 0.3
values += sigma * (rng.standard_normal(len(uu)) + 1j * rng.standard_normal(len(uu)))
flux_unit = u.ph / (u.cm**2 * u.s)

###############################################################################
# Create visibility object

vis = Visibilities(
    values * flux_unit, uu / u.arcsec, vv / u.arcsec, amplitude_uncertainty=np.full(len(uu), sigma) * flux_unit
)

###############################################################################
# Run vis-cs
#

vis_map = vis_cs(vis, shape=shape * u.pix, pixel_size=1 * u.arcsec / u.pix, total_flux=100 * flux_unit, seed=42)


###############################################################################
# Plot true and reconstructed images

fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, data, title in zip(axes, [truth, vis_map.data], ["Truth", "VIS-CS"], strict=True):
    artist = ax.imshow(
        data,
        origin="lower",
        extent=(-40.5, 40.5, -40.5, 40.5),
        vmin=0,
        vmax=max(truth.max(), vis_map.data.max()),
        cmap="inferno",
    )
    ax.set(xlabel="X [arcsec]", ylabel="Y [arcsec]", title=title)
fig.colorbar(artist, ax=axes, label="Surface brightness [ph cm⁻² s⁻¹ arcsec⁻²]")
plt.show()
