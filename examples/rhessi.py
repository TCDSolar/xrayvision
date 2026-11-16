"""
======================================
RHESSI Visibility Imaging
======================================

We select a time interval, energy band, and detector subset from RHESSI
visibilities, then reconstruct the source using back projection, CLEAN, maximum
entropy, and an elliptical Gaussian model. Comparing the images shows how
different reconstruction assumptions affect the result. Bayesian fitting then
lets us explore uncertainty in the ellipse's parameters.

Read the Docs uses short runs to illustrate the workflow. Set
``XRAYVISION_QUICK_EXAMPLES=1`` to preview these locally, or ``0`` to use the
longer settings. For scientific use, extend sampling and check the diagnostics
and stability across independent runs before interpreting the uncertainties.

"""

import os

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

quick_run = os.environ.get("XRAYVISION_QUICK_EXAMPLES", os.environ.get("READTHEDOCS", "False")).lower() in {"1", "true"}

###############################################################################
# Open the RHESSI visibility data
# -------------------------------
#
# We use :func:`~astropy.io.fits.open` to open a visibility FITS file from the
# RHESSI archive. The file contains measurements for several time and energy
# ranges, so we first select the observation we want to reconstruct.

hdul = fits.open(
    "https://hesperia.gsfc.nasa.gov/rhessi_extras/visibility_fits_v2/2002/02/21/hsi_vis_20020221_2357_0054_46tx3e.fits"
)

###############################################################################
# Select one time interval
# ------------------------
#
# We select one complete integration range so that all measurements describe
# the same interval. Matching both endpoints keeps each selected visibility
# row once.

times = np.unique(hdul[-1].data["TRANGE"], axis=0)
time_index = np.all(hdul[-1].data["TRANGE"] == times[6], axis=1)
vis_data = hdul[3].data[time_index]


###############################################################################
# Select the energy band
# ----------------------
#
# We restrict the observation to 12-25 keV. The reconstructed source will
# therefore describe emission in this energy band and the selected time interval.

energy_index = np.all(vis_data["ERANGE"] == [12.0, 25.0], axis=1)
vis_data = vis_data[energy_index]

###############################################################################
# Choose the detector subset
# --------------------------
#
# For this observation, we exclude ISC 0 and 1 and discard zero-valued entries
# before reconstruction. The retained detectors determine which spatial
# frequencies contribute to the images below.

vis_data = vis_data[vis_data["isc"] > 1]
vis_data = vis_data[vis_data["obsvis"] != 0 + 0j]

###############################################################################
# Build the visibility object
# ---------------------------
#
# :class:`~xrayvision.visibility.Visibilities` keeps the selected complex
# measurements, spatial frequencies, phase centre, and uncertainties together.
# We retain the detector labels as metadata and use the file's amplitude
# uncertainties for the weighted forward fits later in the example.

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
# Inspect the response to a point source
# --------------------------------------
#
# :func:`~xrayvision.imaging.vis_psf_map` calculates the point spread function,
# or dirty beam, from the sampled spatial frequencies. Its sidelobes help us
# recognise structure that the sampling pattern can introduce into an image.

psf_map = vis_psf_map(vis, shape=(101, 101) * apu.pixel, pixel_size=1.5 * apu.arcsec / apu.pixel, scheme="uniform")

###############################################################################
# Make a first image with back projection
# ---------------------------------------
#
# :func:`~xrayvision.imaging.vis_to_map` forms an image directly from the measured
# Fourier components. This gives a useful first view of the source. The limited
# frequency coverage also introduces sidelobes, which motivate the reconstruction
# methods below.

backproj_map = vis_to_map(vis, shape=[101, 101] * apu.pixel, pixel_size=1.5 * apu.arcsec / apu.pix)

###############################################################################
# Reduce sidelobes with CLEAN
# ---------------------------
#
# :func:`~xrayvision.clean.vis_clean` identifies bright components and iteratively
# removes their dirty-beam response. It restores the component model with a
# chosen clean beam and adds the residual image. Here we use a 10-arcsecond
# restoring beam and allow up to 100 iterations.

clean_map, model_map, residual_map = vis_clean(
    vis,
    shape=[101, 101] * apu.pixel,
    pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix,
    clean_beam_width=10 * apu.arcsec,
    niter=100,
)

###############################################################################
# Reconstruct a maximum-entropy image
# -----------------------------------
#
# :func:`~xrayvision.mem.mem` balances agreement with the visibilities against
# an entropy term, allowing us to reconstruct an extended source without choosing
# a parametric shape. We estimate a signal-to-noise ratio from the selected
# detectors, which can inform a regularisation choice. The call below uses
# MEM's default regularisation; the calculated ``percent_lambda`` is not passed
# to it.

# Start with the higher ISC labels and expand the subset until it contains
# at least two measurements.
isc_min = 6
nbig = 0

while isc_min >= 0 and nbig < 2:
    ibig = np.argwhere(vis.meta.vis_labels >= isc_min)
    nbig = len(ibig)
    isc_min = isc_min - 1

# Use a sentinel if no suitable subset exists; otherwise estimate a robust
# mean signal-to-noise ratio.
if nbig < 2:
    snr_value = -1
else:
    snr_value, _ = resistant_mean((np.abs(vis.visibilities[ibig]) / vis.amplitude_uncertainty[ibig]).flatten(), 3)

percent_lambda = 11.0 / (snr_value**2 + 383.0)

mem_map = mem(vis, shape=[101, 101] * apu.pixel, pixel_size=[1.5, 1.5] * apu.arcsec / apu.pix)
plt.figure()
mem_map.plot()

###############################################################################
# Fit an elliptical source with Nelder-Mead
# -----------------------------------------
#
# Forward fitting describes the image with a small set of source parameters.
# We use :obj:`~xrayvision.vis_forward_fit.sources.Source` to define an elliptical
# Gaussian and :func:`~xrayvision.vis_forward_fit.forward_fit.vis_forward_fit`
# to fit its flux, position, widths, and orientation directly to the visibilities.
# The default Nelder-Mead method starts from the initial guess below. Source
# positions are relative to the observation's phase centre.

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

###############################################################################
# Search more broadly with particle swarm optimisation
# ----------------------------------------------------
#
# The same :func:`~xrayvision.vis_forward_fit.forward_fit.vis_forward_fit` API
# also offers particle swarm optimisation (PSO). A population of candidate
# ellipses explores the bounds more broadly than a single local fit. We run
# three independent searches and retain the solution with the smallest objective.
# Both methods use the amplitude uncertainties for weighting, so their fit
# objectives can be compared directly.

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
# Describe plausible ellipses with explicit priors
# ------------------------------------------------
#
# The PSO fit gives one solution. We now ask which range of ellipses is supported
# by the measurements and our prior assumptions. We keep the same source model
# and error convention, and specify broad uniform priors with
# :class:`~xrayvision.vis_forward_fit.bayesian.ParameterPrior`. These are explicit
# scientific assumptions, separate from the optimisation bounds.
#
# We first express the fitted ellipse in the same orientation interval used by
# the priors, swapping its axis labels where necessary. Then
# :class:`~xrayvision.vis_forward_fit.bayesian.BayesianVisibilityProblem` combines
# the source, priors, and complex-visibility likelihood.
#
# See :ref:`sphx_glr_generated_gallery_bayesian_visibility_fit.py` for a
# walkthrough using a known synthetic source, including posterior visibility
# predictions and comparison of alternative source models.

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

###############################################################################
# Explore parameter uncertainty with emcee
# ----------------------------------------
#
# We initialise 24 walkers near the PSO solution and use
# :func:`~xrayvision.vis_forward_fit.bayesian.sample_emcee` to explore the posterior.
# After the initial burn-in, the samples describe plausible parameter values
# and their correlations. The short documentation run illustrates this process;
# longer runs are needed to assess the uncertainty reliably.

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
if quick_run:
    print("Quick documentation run: illustrative posterior; extend sampling and check convergence for scientific use.")
posterior = sample_emcee(
    problem, np.asarray(positions), n_steps=600 if quick_run else 10000, burn_in=200 if quick_run else 1000, seed=43
)

###############################################################################
# Check the fit and sampling diagnostics
# --------------------------------------
#
# The initial PSO reduced chi-square describes its agreement with the data under
# the chosen error convention. Acceptance fractions and autocorrelation estimates
# help us assess the subsequent sampling. In particular, ``chain_too_short``
# calls for a longer run before interpreting the reported credible intervals.
# Each 95% interval describes one parameter separately.

print("RHESSI reduced chi-square (initial PSO fit):", pso_result.fit_statistics.reduced_chi2)
print("RHESSI autocorrelation status:", posterior.autocorrelation_status)
print("Autocorrelation times:", posterior.autocorrelation_time)
print("Mean production acceptance fraction:", posterior.acceptance_fraction.mean())
print(posterior.diagnostics["autocorrelation_message"])
for name in posterior.parameter_names:
    print(name, "95% credible interval:", posterior.credible_interval(name, level=0.95))

###############################################################################
# Reconstruct a representative Bayesian image
# -------------------------------------------
#
# We select the sampled ellipse with the highest posterior density and render
# it with :func:`~xrayvision.vis_forward_fit.forward_fit.sources_to_image`. Using
# a complete sample preserves the relationships between its parameters. This
# gives one representative image; the full posterior describes its uncertainty.

best_step, best_walker = np.unravel_index(np.argmax(posterior.log_prob), posterior.log_prob.shape)
sampled_sources = posterior.sources_from_sample(int(best_step), int(best_walker))
bayesian_image = sources_to_image(sampled_sources, [101, 101] * apu.pixel, [1.5, 1.5] * apu.arcsec / apu.pix)
bayesian_map = Map((bayesian_image, vis_fwd_pso_map.meta))

###############################################################################
# Compare the reconstructed images
# --------------------------------
#
# We bring the results together on the same coordinate grid to compare source
# position and extent. The PSF provides a reference for the sampling response;
# CLEAN and MEM reconstruct an image without assuming an elliptical source,
# while the forward fits use that specific shape. The Bayesian panel shows
# the highest-posterior sample selected above.

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
# Examine parameter uncertainty and correlations
# ----------------------------------------------
#
# The corner plot shows the marginal distributions of the ellipse's parameters
# and how they vary together. Titles give marginal medians and 68% intervals;
# green lines and squares identify the same complete sample used for the image.
#
# An ellipse can be represented by exchanging its widths and rotating it by
# 90 degrees. For this plot, we express every sample with major/minor widths
# and the major-axis angle so that equivalent axis labels do not split the
# distributions. The stored samples retain the original source parameters.

display_samples = posterior.samples.copy()
swap_axes = display_samples[:, 3] < display_samples[:, 4]
display_samples[swap_axes, 3:5] = display_samples[swap_axes, 3:5][:, ::-1]
display_samples[swap_axes, 5] += np.pi / 2
display_samples[:, 5] = (display_samples[:, 5] + np.pi / 2) % np.pi - np.pi / 2
names = ["amp", "x0", "y0", "sigma_major", "sigma_minor", "theta_major"]
labels = [f"{name}\n{unit.to_string('latex_inline')}" for name, unit in zip(names, posterior.parameter_units)]
spread = display_samples.std(axis=0, ddof=1)
plot_ranges = list(zip(display_samples.min(axis=0) - spread, display_samples.max(axis=0) + spread))
corner_figure = corner.corner(
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
# Flattening log_prob uses the same step/walker order as display_samples, so
# this also applies the plot's major/minor conversion to the selected draw.
best_display = display_samples[np.argmax(posterior.log_prob)]
corner.overplot_lines(corner_figure, best_display, color="C2")
corner.overplot_points(corner_figure, best_display[None, :], color="C2", marker="s", markersize=4)
corner_figure.text(
    0.62,
    0.84,
    "Green: highest-posterior draw\nTitles: marginal median and 68% interval",
    fontsize=12,
    ha="center",
)
plt.show()
