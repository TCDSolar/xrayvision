"""
======================================
Solar Orbiter/STIX Visibility Imaging
======================================

We reconstruct a solar X-ray source from STIX visibilities using back projection,
CLEAN, maximum entropy, and an elliptical Gaussian source model. Comparing the
images shows how each method handles the limited spatial-frequency coverage.
We then use Bayesian fitting to explore the uncertainty in the ellipse's flux,
position, size, and orientation.

Read the Docs uses short runs to illustrate the workflow. Set
``XRAYVISION_QUICK_EXAMPLES=1`` to preview these locally, or ``0`` to use the
longer settings. For scientific use, extend sampling and check the diagnostics
and stability across independent runs before interpreting the uncertainties.

"""

import os
import pickle
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

import astropy.units as apu

from sunpy.map import Map

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_psf_map, vis_to_map
from xrayvision.mem import mem, resistant_mean
from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_image, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList

quick_run = os.environ.get("XRAYVISION_QUICK_EXAMPLES", os.environ.get("READTHEDOCS", "False")).lower() in {"1", "true"}

###############################################################################
# Load the prepared STIX observations
# -----------------------------------
#
# The example dataset contains prepared measurements for a selected
# time and energy range. We load its
# :class:`~xrayvision.visibility.Visibilities` object so we can focus on image
# reconstruction. It keeps the complex measurements, spatial frequencies,
# phase centre, and uncertainties together.

# Sphinx-Gallery runs in the example directory without defining __file__.
example_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
with (example_dir / "stix_vis.pkl").open("rb") as file:
    stix_data = pickle.load(file)

time_range = stix_data["time_range"]
energy_range = stix_data["energy_range"]
stix_vis = stix_data["stix_visibilities"]

###############################################################################
# Inspect the response to a point source
# --------------------------------------
#
# Before reconstructing the source, we examine the point spread function using
# :func:`~xrayvision.imaging.vis_psf_map`. This dirty beam shows the response of
# the sampled spatial frequencies to a point source. Its sidelobes help us
# recognise structure that the sampling pattern can introduce into an image.

psf_map = vis_psf_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix, scheme="uniform")
plt.figure()
psf_map.plot()

###############################################################################
# Make a first image with back projection
# ---------------------------------------
#
# :func:`~xrayvision.imaging.vis_to_map` forms an image directly from the measured
# Fourier components. This gives a useful first view of the source. The limited
# frequency coverage also introduces sidelobes, which motivate the reconstruction
# methods below.

backproj_map = vis_to_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix, scheme="uniform")
plt.figure()
backproj_map.plot()

###############################################################################
# Reduce sidelobes with CLEAN
# ---------------------------
#
# :func:`~xrayvision.clean.vis_clean` identifies bright components and iteratively
# removes their dirty-beam response. It restores the component model with a
# chosen clean beam and adds the residual image. Here we use a 20-arcsecond
# restoring beam and allow up to 100 iterations.

clean_map, model_map, resid_map = vis_clean(
    stix_vis,
    shape=[129, 129] * apu.pixel,
    pixel_size=[2, 2] * apu.arcsec / apu.pix,
    clean_beam_width=20 * apu.arcsec,
    niter=100,
)
plt.figure()
clean_map.plot()

###############################################################################
# Reconstruct a maximum-entropy image
# -----------------------------------
#
# :func:`~xrayvision.mem.mem` balances agreement with the visibilities against
# an entropy term, allowing us to reconstruct an extended source without choosing
# a parametric shape. We estimate the signal-to-noise ratio and use it to choose
# ``percent_lambda``, which sets the strength of the regularisation.

snr_value, _ = resistant_mean((np.abs(stix_vis.visibilities) / stix_vis.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90) * apu.percent

mem_map = mem(
    stix_vis, shape=[129, 129] * apu.pixel, pixel_size=[2, 2] * apu.arcsec / apu.pix, percent_lambda=percent_lambda
)

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
            15 * stix_vis.visibilities.unit,
            1 * apu.arcsec,
            2 * apu.arcsec,
            5 * apu.arcsec,
            2 * apu.arcsec,
            1 * apu.rad,
        )
    ]
)

# Use explicit broad bounds instead of the defaults derived from the initial
# guess. Columns are amp, x0, y0, sigmax, sigmay, theta in observation flux
# units, arcseconds, and radians. A full pi interval covers every orientation.
sources[0].bounds = np.array([[0, -128, -128, 0.5, 0.5, -np.pi / 2], [100, 128, 128, 64, 64, np.pi / 2]])
vis_fwd_map = vis_forward_fit(
    stix_vis,
    sources,
    shape=[129, 129] * apu.pixel,
    pixel_size=[2, 2] * apu.arcsec / apu.pix,
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
    stix_vis,
    sources,
    method="PSO",
    n_runs=3,
    shape=[129, 129] * apu.pixel,
    pixel_size=[2, 2] * apu.arcsec / apu.pix,
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

flux_unit = stix_vis.visibilities.unit
initial_sources = fitted_sources.to_canonical(flux_unit)
ellipse = initial_sources[0]
ellipse.theta = (ellipse.theta + np.pi / 2) % np.pi - np.pi / 2
if ellipse.theta >= 0:
    ellipse.sigmax, ellipse.sigmay = ellipse.sigmay, ellipse.sigmax
    ellipse.theta -= np.pi / 2
priors = {
    "source[0].amp": ParameterPrior(uniform(loc=0, scale=100), flux_unit),
    "source[0].x0": ParameterPrior(uniform(loc=-128, scale=256), apu.arcsec),
    "source[0].y0": ParameterPrior(uniform(loc=-128, scale=256), apu.arcsec),
    "source[0].sigmax": ParameterPrior(uniform(loc=0.5, scale=63.5), apu.arcsec),
    "source[0].sigmay": ParameterPrior(uniform(loc=0.5, scale=63.5), apu.arcsec),
    "source[0].theta": ParameterPrior(uniform(loc=-np.pi / 2, scale=np.pi / 2), apu.rad),
}
problem = BayesianVisibilityProblem(stix_vis, initial_sources, priors, weighting="amplitude")

###############################################################################
# Explore parameter uncertainty with emcee
# ----------------------------------------
#
# We initialise 24 walkers near the PSO solution and use
# :func:`~xrayvision.vis_forward_fit.bayesian.sample_emcee` to explore the posterior.
# After the initial burn-in, the samples describe plausible parameter values
# and their correlations. The short documentation run illustrates this process;
# longer runs are needed to assess the uncertainty reliably.

rng = np.random.default_rng(42)
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
    problem, np.asarray(positions), n_steps=600 if quick_run else 10000, burn_in=200 if quick_run else 1000, seed=42
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

print("STIX reduced chi-square (initial PSO fit):", pso_result.fit_statistics.reduced_chi2)
print("STIX autocorrelation status:", posterior.autocorrelation_status)
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
bayesian_image = sources_to_image(sampled_sources, [129, 129] * apu.pixel, [2, 2] * apu.arcsec / apu.pix)
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
