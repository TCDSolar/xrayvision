"""
======================================
Solar Orbiter/STIX Visibility Imaging
======================================

Imports

"""

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

###############################################################################
# Create images from STIX visibility data.
#
# The STIX data has already been prepared and stored in python pickle format
# the variables can be simply restored.

# Sphinx-Gallery runs in the example directory without defining __file__.
example_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
with (example_dir / "stix_vis.pkl").open("rb") as file:
    stix_data = pickle.load(file)

time_range = stix_data["time_range"]
energy_range = stix_data["energy_range"]
stix_vis = stix_data["stix_visibilities"]

###############################################################################
# Lets have a look at the point spread function (PSF) or dirty beam

psf_map = vis_psf_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix, scheme="uniform")
plt.figure()
psf_map.plot()

###############################################################################
# Back projection

backproj_map = vis_to_map(stix_vis, shape=(129, 129) * apu.pixel, pixel_size=2 * apu.arcsec / apu.pix, scheme="uniform")
plt.figure()
backproj_map.plot()

###############################################################################
# Clean

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
# MEM

# Compute percent_lambda
snr_value, _ = resistant_mean((np.abs(stix_vis.visibilities) / stix_vis.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90) * apu.percent

mem_map = mem(
    stix_vis, shape=[129, 129] * apu.pixel, pixel_size=[2, 2] * apu.arcsec / apu.pix, percent_lambda=percent_lambda
)

###############################################################################
# VIS_FWD_FIT

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
# equivalent orientation interval [-pi/2, 0), with identical priors on both
# widths, to avoid duplicate axis labels without excluding source orientations.
# Its cut is away from this source's fitted orientation, around -40 degrees.
# Shifting this interval does not change the physical prior when the widths
# have identical priors, but avoids a discontinuity through the sampled region.

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
posterior = sample_emcee(problem, np.asarray(positions), n_steps=3000, burn_in=500, seed=42)
print("STIX reduced chi-square (initial PSO fit):", pso_result.fit_statistics.reduced_chi2)
print("STIX autocorrelation status:", posterior.autocorrelation_status)
print("Autocorrelation times:", posterior.autocorrelation_time)
print("Mean production acceptance fraction:", posterior.acceptance_fraction.mean())
print(posterior.diagnostics["autocorrelation_message"])
for name in posterior.parameter_names:
    print(name, "95% credible interval:", posterior.credible_interval(name, level=0.95))

# A complete sampled parameter vector provides a representative image. This is
# the largest sampled posterior density, rather than a proven global maximum.
best_step, best_walker = np.unravel_index(np.argmax(posterior.log_prob), posterior.log_prob.shape)
sampled_sources = posterior.sources_from_sample(int(best_step), int(best_walker))
bayesian_image = sources_to_image(sampled_sources, [129, 129] * apu.pixel, [2, 2] * apu.arcsec / apu.pix)
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
