"""
======================================
RHESSI Visibility Imaging
======================================

Create images from RHESSI visibility data

"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

import astropy.units as apu
from astropy.io import fits

from sunpy.map import Map

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_psf_map, vis_to_map
from xrayvision.mem import mem, resistant_mean
from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_image, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities, VisMeta

###############################################################################
# We will use `astropy.io.fits` to download and open the RHESSI visibility fits
# file

hdul = fits.open(
    "https://hesperia.gsfc.nasa.gov/rhessi_extras/visibility_fits_v2/2002/02/21/hsi_vis_20020221_2357_0054_46tx3e.fits"
)

###############################################################################
# No lets extract the visibility data the first thing we will do is filter the
# visibilities by time. We select one complete integration range. Matching both
# endpoints together avoids including each selected row twice.

times = np.unique(hdul[-1].data["TRANGE"], axis=0)
time_index = np.all(hdul[-1].data["TRANGE"] == times[6], axis=1)
vis_data = hdul[3].data[time_index]


###############################################################################
# Next lets filter by energy range in this case 12 - 25 keV

energy_index = np.all(vis_data["ERANGE"] == [12.0, 25.0], axis=1)
vis_data = vis_data[energy_index]

###############################################################################
# Now lets filter by ISC or detector to remove possibly bad data in this case
# need to remove ISC 0 and 1.

vis_data = vis_data[vis_data["isc"] > 1]
vis_data = vis_data[vis_data["obsvis"] != 0 + 0j]

###############################################################################
# Now we can create the visibility object from the filtered visibilities.

meta = VisMeta({"vis_labels": vis_data["isc"]})

vunit = apu.Unit("photon/(cm**2 s)")
vis = Visibilities(
    visibilities=vis_data["obsvis"] * vunit,
    u=vis_data["u"] / apu.arcsec,
    v=vis_data["v"] / apu.arcsec,
    phase_center=vis_data["xyoffset"][0] * apu.arcsec,
    meta=meta,
    amplitude_uncertainty=vis_data["sigamp"] * vunit,
)


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

# Compute percent_lambda
# Loop through ISCs starting with 6-9, but if we don't have at least 2 vis, lower isc_min to include next one down, etc.
isc_min = 6
nbig = 0

while isc_min >= 0 and nbig < 2:
    ibig = np.argwhere(vis.meta.vis_labels >= isc_min)
    nbig = len(ibig)
    isc_min = isc_min - 1

# If still don't have at least 2 vis, return -1, otherwise calculate mean (but reject points > sigma away from mean)
if nbig < 2:
    snr_value = -1
else:
    snr_value, _ = resistant_mean((np.abs(vis.visibilities[ibig]) / vis.amplitude_uncertainty[ibig]).flatten(), 3)

percent_lambda = 11.0 / (snr_value**2 + 383.0)

mem_map = mem(vis, shape=[101, 101] * apu.pixel, pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix)
plt.figure()
mem_map.plot()

###############################################################################
# VIS_FWD_FIT
#
# Forward fitting assumes the source is a parametric shape (here a single
# elliptical Gaussian) and fits its parameters directly to the visibilities.
# Source position is relative to the visibility's own phase center.

sources = SourceList(
    [
        Source(
            "elliptical",
            1000 * vunit,
            0 * apu.arcsec,
            0 * apu.arcsec,
            15 * apu.arcsec,
            15 * apu.arcsec,
            0 * apu.rad,
        )
    ]
)

# Use explicit broad bounds instead of the defaults derived from the initial
# guess. Columns are amp, x0, y0, sigmax, sigmay, theta in observation flux
# units, arcseconds, and radians. A full pi interval covers every orientation.
sources[0].bounds = np.array([[0, -75, -75, 0.5, 0.5, -np.pi / 2], [5000, 75, 75, 50, 50, np.pi / 2]])
vis_fwd_map = vis_forward_fit(
    vis,
    sources,
    shape=[101, 101] * apu.pixel,
    pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix,
    weighting="amplitude",
)

vis_fwd_pso_map, fitted_sources, pso_result = vis_forward_fit(
    vis,
    sources,
    method="PSO",
    n_runs=3,
    shape=[101, 101] * apu.pixel,
    pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix,
    weighting="amplitude",
    return_result=True,
)

###############################################################################
# Bayesian visibility fitting
# ---------------------------
#
# Priors describe a source inside the displayed field, with positive flux and
# widths. For this comparison their physical support matches the optimisation
# bounds, but they remain explicit scientific choices and should be revisited.
# Amplitude uncertainties are used for both complex components, as in the
# preceding PSO fit; this assumes independent real and imaginary errors with
# equal standard deviations.
#
# Ellipse axes can be exchanged while adding pi/2 to the angle. Use a single
# equivalent orientation interval [-pi/4, pi/4), with identical priors on both
# widths, to avoid duplicate axis labels without excluding source orientations.

flux_unit = vis.visibilities.unit
initial_sources = fitted_sources.to_canonical(flux_unit)
ellipse = initial_sources[0]
ellipse.theta = (ellipse.theta + np.pi / 4) % np.pi - np.pi / 4
if ellipse.theta >= np.pi / 4:
    ellipse.sigmax, ellipse.sigmay = ellipse.sigmay, ellipse.sigmax
    ellipse.theta -= np.pi / 2
priors = {
    "source[0].amp": ParameterPrior(uniform(loc=0, scale=5000), flux_unit),
    "source[0].x0": ParameterPrior(uniform(loc=-75, scale=150), apu.arcsec),
    "source[0].y0": ParameterPrior(uniform(loc=-75, scale=150), apu.arcsec),
    "source[0].sigmax": ParameterPrior(uniform(loc=0.5, scale=49.5), apu.arcsec),
    "source[0].sigmay": ParameterPrior(uniform(loc=0.5, scale=49.5), apu.arcsec),
    "source[0].theta": ParameterPrior(uniform(loc=-np.pi / 4, scale=np.pi / 2), apu.rad),
}
problem = BayesianVisibilityProblem(vis, initial_sources, priors, weighting="amplitude")
rng = np.random.default_rng(43)
centre = problem.initial_params
spread = np.array([0.02 * centre[0], 1, 1, 0.05 * centre[3], 0.05 * centre[4], 0.02])
positions = []
for _ in range(2400):
    candidate = centre + rng.normal(size=problem.ndim) * spread
    if np.isfinite(problem.log_posterior(candidate)):
        positions.append(candidate)
    if len(positions) == 24:
        break
if len(positions) != 24:
    raise RuntimeError("Could not initialise 24 valid walkers; inspect the fit and priors.")
posterior = sample_emcee(problem, np.asarray(positions), n_steps=3000, burn_in=500, seed=43)
print("RHESSI reduced chi-square (initial PSO fit):", pso_result.fit_statistics.reduced_chi2)
print("RHESSI autocorrelation status:", posterior.autocorrelation_status)
print("Autocorrelation times:", posterior.autocorrelation_time)
print("Mean production acceptance fraction:", posterior.acceptance_fraction.mean())
print(posterior.diagnostics["autocorrelation_message"])
for name in posterior.parameter_names:
    print(name, "95% credible interval:", posterior.credible_interval(name, level=0.95))

# A complete sampled parameter vector provides a representative image. This is
# the largest sampled posterior density, rather than a proven global maximum.
best_step, best_walker = np.unravel_index(np.argmax(posterior.log_prob), posterior.log_prob.shape)
sampled_sources = posterior.sources_from_sample(int(best_step), int(best_walker))
bayesian_image = sources_to_image(sampled_sources, [101, 101] * apu.pixel, [1.5, 1.5] * apu.arcsec / apu.pix)
bayesian_map = Map((bayesian_image, vis_fwd_pso_map.meta))

###############################################################################
# Comparison on shared map axes
# -----------------------------
#
# Every map has the same pixel grid and WCS. Show coordinate labels and ticks
# only on the lower-left panel, so repeated solar-coordinate labels do not
# overlap. The Bayesian panel is the highest-posterior production draw.

comparison_maps = (
    (psf_map, "PSF"),
    (backproj_map, "Back projection"),
    (clean_map, "CLEAN"),
    (mem_map, "MEM"),
    (vis_fwd_map, "SciPy forward fit"),
    (vis_fwd_pso_map, "PSO forward fit"),
    (bayesian_map, "Bayesian highest-posterior draw"),
)
fig = plt.figure(figsize=(15, 8), layout="constrained")
axs = []
for panel, (amap, title) in enumerate(comparison_maps):
    reference_axis = axs[0] if axs else None
    axis = fig.add_subplot(2, 4, panel + 1, projection=amap, sharex=reference_axis, sharey=reference_axis)
    axs.append(axis)
    amap.plot(axes=axis)
    axis.set_title(title, fontsize=11)
    for coordinate in axis.coords:
        coordinate.set_auto_axislabel(False)
        coordinate.set_axislabel("")
        coordinate.set_ticks_visible(panel == 4)
        coordinate.set_ticklabel_visible(panel == 4)
axs[4].coords[0].set_axislabel("Solar X [arcsec]")
axs[4].coords[1].set_axislabel("Solar Y [arcsec]")
for coordinate in axs[4].coords:
    coordinate.set_format_unit(apu.arcsec)
    coordinate.set_ticks(number=3)
plt.show()

###############################################################################
# Posterior corner plot
# ---------------------
#
# Real observations have no injected truth markers. Inspect sampling diagnostics
# before interpreting the intervals; the example run length is not a convergence
# guarantee. For display, express every ellipse using its major and minor
# widths and the major-axis angle modulo pi. This reconciles the apparent
# width/angle lobes where the original angle interval swaps axis labels.
# Stored parameter samples are unchanged. This transformation does not merge
# genuinely distinct source shapes or establish sampling convergence. Near
# circular shapes still have poorly defined orientation; angles crossing the
# display boundary at +/- pi/2 need circular summaries instead of percentiles.

display_samples = posterior.samples.copy()
swap_axes = display_samples[:, 3] < display_samples[:, 4]
display_samples[swap_axes, 3:5] = display_samples[swap_axes, 3:5][:, ::-1]
display_samples[swap_axes, 5] += np.pi / 2
display_samples[:, 5] = (display_samples[:, 5] + np.pi / 2) % np.pi - np.pi / 2
names = ["amp", "x0", "y0", "sigma_major", "sigma_minor", "theta_major"]
labels = [f"{name}\n{unit.to_string('latex_inline')}" for name, unit in zip(names, posterior.parameter_units)]
spread = display_samples.std(axis=0, ddof=1)
plot_ranges = list(zip(display_samples.min(axis=0) - spread, display_samples.max(axis=0) + spread))
corner.corner(
    display_samples,
    labels=labels,
    titles=names,
    range=plot_ranges,
    fig=plt.figure(figsize=(12, 12)),
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
    title_fmt=".2f",
    title_kwargs={"fontsize": 9},
    label_kwargs={"fontsize": 10},
    bins=30,
    smooth=0.8,
    levels=[0.68, 0.95],
    plot_datapoints=False,
    fill_contours=True,
)
plt.show()
