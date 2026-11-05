"""
=======================================
Bayesian recovery of a synthetic source
=======================================

Fit noisy complex X-ray visibilities with the existing circular Gaussian model,
inspect emcee traces and visibility predictions, and finish with a corner plot.
The example uses the public fitting API; no separate source implementation is
needed. Install the documentation dependencies to run the plotting example,
including ``corner`` (``pip install 'xrayvisim[docs]'``).
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_image, sources_to_vis, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities

###############################################################################
# Simulate independent complex visibility measurements
# ----------------------------------------------------
#
# Source amplitude is differential photon flux integrated over angular area,
# rather than surface brightness. Real and imaginary noise have separate known
# standard deviations. Each sampled baseline appears once: conjugate copies
# would not provide independent measurements.

flux_unit = apu.ph / (apu.s * apu.cm**2 * apu.keV)
truth = SourceList(
    [
        Source("circular", 2 * flux_unit, -1.5 * apu.arcsec, 0.8 * apu.arcsec, 2 * apu.arcsec),
    ]
)
rng = np.random.default_rng(74)
u, v = rng.uniform(-0.12, 0.12, (2, 30)) / apu.arcsec
true_visibilities = sources_to_vis(truth, u, v)
sigma_real, sigma_imag = 0.03, 0.06
noise = (rng.normal(0, sigma_real, 30) + 1j * rng.normal(0, sigma_imag, 30)) * flux_unit
observed = Visibilities(
    true_visibilities + noise,
    u,
    v,
    uncertainty=np.full(30, sigma_real + 1j * sigma_imag) * flux_unit,
)

###############################################################################
# Initialise with the existing SciPy fit
# --------------------------------------
#
# The local fit and its Gauss-Newton covariance place the walkers near a useful
# starting point. This covariance is only used for initialisation; the posterior
# uncertainty below comes from sampling. Optimisation bounds do not become
# priors.

guess = SourceList(
    [
        Source("circular", 1.8 * flux_unit, -1 * apu.arcsec, 0.5 * apu.arcsec, 2.2 * apu.arcsec),
    ]
)
image_shape = [65, 65] * apu.pix
pixel_size = [0.5, 0.5] * apu.arcsec / apu.pix
fitted_image, fitted, fit_result = vis_forward_fit(
    observed,
    guess,
    image_shape,
    pixel_size,
    map=False,
    return_result=True,
    uncertainty="gauss_newton",
)
initial_covariance = fit_result.parameter_uncertainty.covariance
if initial_covariance is None:
    raise RuntimeError("The local fit did not provide a usable initialisation covariance.")

###############################################################################
# Compare the injected and initially recovered images
# ---------------------------------------------------
#
# Render both source lists on the same angular grid and share a colour scale.
# Analytical images sample a surface-brightness density in canonical arcsecond
# coordinates, so the displayed intensity is photon flux per square arcsecond.
# This image comparison uses the initial SciPy fit, before posterior sampling.

input_image = sources_to_image(truth, image_shape, pixel_size)
field_of_view = (image_shape * pixel_size).to_value(apu.arcsec)
extent = [-field_of_view[1] / 2, field_of_view[1] / 2, -field_of_view[0] / 2, field_of_view[0] / 2]
brightness_unit = flux_unit / apu.arcsec**2
vmax = max(input_image.value.max(), fitted_image.value.max())
image_figure, image_axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True, layout="constrained")
for panel, (axis, image, title) in enumerate(
    zip(image_axes, (input_image, fitted_image), ("Injected source", "Initial SciPy fit"))
):
    artist = axis.imshow(image.value, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    axis.set(xlim=(-10, 10), ylim=(-10, 10), title=title)
    axis.tick_params(bottom=panel == 0, left=panel == 0, labelbottom=panel == 0, labelleft=panel == 0)
image_axes[0].set(xlabel="x [arcsec]", ylabel="y [arcsec]")
image_figure.colorbar(artist, ax=image_axes, label=brightness_unit.to_string("latex_inline"))
plt.show()

###############################################################################
# Specify scientific priors explicitly
# ------------------------------------
#
# Flux and width have positive, bounded priors; positions have broad normal
# priors in arcseconds. Prior names match the source parameter names exactly.
# Units can differ from the observation units when they are convertible.

priors = {
    "source[0].amp": ParameterPrior(uniform(loc=0.1, scale=3.9), flux_unit),
    "source[0].x0": ParameterPrior(norm(loc=0, scale=5), apu.arcsec),
    "source[0].y0": ParameterPrior(norm(loc=0, scale=5), apu.arcsec),
    "source[0].sigma": ParameterPrior(uniform(loc=0.5, scale=5.5), apu.arcsec),
}
problem = BayesianVisibilityProblem(observed, fitted, priors)
positions = rng.multivariate_normal(problem.initial_params, initial_covariance, size=16)
posterior = sample_emcee(problem, positions, n_steps=5000, burn_in=500, seed=24)

print("Autocorrelation status:", posterior.autocorrelation_status)
print("Autocorrelation times:", posterior.autocorrelation_time)
print("Mean production acceptance fraction:", posterior.acceptance_fraction.mean())
print(posterior.diagnostics["autocorrelation_message"])
truth_params = np.asarray(truth.to_canonical(flux_unit).params)
coverage = {}
for name, true_value, unit in zip(posterior.parameter_names, truth_params, posterior.parameter_units):
    interval = posterior.credible_interval(name, level=0.95)
    inside = bool(interval[0].value <= true_value <= interval[1].value)
    coverage[name] = inside
    print(name, "injected:", true_value * unit, "95% credible interval:", interval, "contains injected value:", inside)

# These are individual, marginal intervals, not a joint 95% region. A correctly
# calibrated interval can miss the true value in any particular noisy dataset.
print("All injected parameters inside their marginal 95% intervals:", all(coverage.values()))

###############################################################################
# Compare the injected, initial, and sampled source images
# --------------------------------------------------------
#
# Select the actual production draw with the largest sampled log posterior.
# This is an approximation to the posterior mode, not an optimisation proving
# the global maximum. Using a complete draw preserves the joint parameter
# configuration. The credible intervals and full sample distribution still
# describe uncertainty; this single image is only a representative point result.

best_step, best_walker = np.unravel_index(np.argmax(posterior.log_prob), posterior.log_prob.shape)
sampled_sources = posterior.sources_from_sample(int(best_step), int(best_walker))
posterior_image = sources_to_image(sampled_sources, image_shape, pixel_size)
comparison_images = (input_image, fitted_image, posterior_image)
comparison_titles = ("Injected source", "Initial SciPy fit", "Highest-posterior draw")
vmax = max(image.value.max() for image in comparison_images)
posterior_image_figure, posterior_image_axes = plt.subplots(
    1, 3, figsize=(13, 4.5), sharex=True, sharey=True, layout="constrained"
)
for panel, (axis, image, title) in enumerate(zip(posterior_image_axes, comparison_images, comparison_titles)):
    artist = axis.imshow(image.value, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    axis.set(xlim=(-10, 10), ylim=(-10, 10), title=title)
    axis.tick_params(bottom=panel == 0, left=panel == 0, labelbottom=panel == 0, labelleft=panel == 0)
posterior_image_axes[0].set(xlabel="x [arcsec]", ylabel="y [arcsec]")
posterior_image_figure.colorbar(artist, ax=posterior_image_axes, label=brightness_unit.to_string("latex_inline"))
plt.show()

###############################################################################
# Inspect the production traces
# -----------------------------
#
# Burn-in is excluded from ``chain``. Walkers within this ensemble are dependent;
# they are not independent chains for convergence statistics. An autocorrelation
# estimate and apparently stable traces do not prove that all modes were found.
# For this simple single-source example the local fit provides a good start.

symbols = [r"$F$", r"$x_0$", r"$y_0$", r"$\sigma$"]
labels = [f"{symbol}\n{unit.to_string('latex_inline')}" for symbol, unit in zip(symbols, posterior.parameter_units)]
trace_figure, trace_axes = plt.subplots(problem.ndim, 1, figsize=(10, 8), sharex=True, layout="constrained")
for column, axis in enumerate(trace_axes):
    axis.plot(posterior.chain[:, :, column], color="C0", alpha=0.15, linewidth=0.4)
    axis.axhline(truth_params[column], color="C3", linestyle="--", linewidth=1)
    axis.set_ylabel(labels[column])
trace_axes[-1].set_xlabel("Production step")
trace_figure.suptitle("Parameter traces; dashed lines show injected values")
plt.show()

###############################################################################
# Compare posterior visibility predictions with the observations
# --------------------------------------------------------------
#
# Predict visibilities from individual source draws. This propagates the joint
# parameter distribution through the original analytical visibility function.
# The intervals describe uncertainty in the noiseless model prediction; they do
# not include a second draw of measurement noise. Each baseline is a separate
# point, ordered by spatial frequency for display.

draw_indices = rng.choice(len(posterior.samples), size=200, replace=False)
predictions = np.array([problem.predict(posterior.samples[i]).to_value(flux_unit) for i in draw_indices])
order = np.argsort(np.hypot(problem.fit_data.u, problem.fit_data.v))
baseline = np.arange(1, len(order) + 1)
visibility_figure, visibility_axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
for axis, component, sigma, label in zip(
    visibility_axes,
    (np.real, np.imag),
    (sigma_real, sigma_imag),
    ("Real", "Imaginary"),
):
    low, median, high = np.quantile(component(predictions), [0.025, 0.5, 0.975], axis=0)[:, order]
    axis.errorbar(
        baseline,
        component(observed.visibilities.value)[order],
        yerr=sigma,
        fmt="o",
        markersize=3,
        color="C0",
        alpha=0.7,
        label="Observed (1 standard deviation)",
    )
    axis.errorbar(
        baseline,
        median,
        yerr=[median - low, high - median],
        fmt="s",
        markersize=4,
        markerfacecolor="none",
        color="C1",
        capsize=2,
        label="Posterior model (95% interval)",
    )
    axis.plot(baseline, component(true_visibilities.value)[order], "x", color="C3", label="Injected model")
    axis.set_ylabel(f"{label} visibility\n{flux_unit.to_string('latex_inline')}")
visibility_axes[0].legend(fontsize=8, ncols=3)
visibility_axes[-1].set_xlabel("Visibility sample (ordered by spatial frequency)")
plt.show()

###############################################################################
# Finish with a corner plot
# -------------------------
#
# ``samples`` already contains the flattened production draws in the order of
# ``parameter_names`` and ``parameter_units``. No sampler internals are needed.
# The diagonal panels show marginal distributions and the 16th, 50th, and 84th
# percentiles; the lower panels show joint distributions with 68% and 95%
# probability contours. Red markers indicate the injected parameters. A corner
# plot displays parameter uncertainty and correlation, not sampling convergence.
# Added noise can shift the posterior peak away from an injected value; compare
# the marginal credible intervals reported above rather than expecting equality.
# With repeated source types or periodic angles, label switching or angular
# wrapping would need attention before interpreting these marginal summaries.

# Pad the sampled range and injected value by one posterior standard deviation
# on each side. These display limits do not change the credible intervals.
posterior_spread = posterior.samples.std(axis=0, ddof=1)
lower_limits = np.minimum(posterior.samples.min(axis=0), truth_params) - posterior_spread
upper_limits = np.maximum(posterior.samples.max(axis=0), truth_params) + posterior_spread
plot_ranges = list(zip(lower_limits, upper_limits))

corner_figure = corner.corner(
    posterior.samples,
    range=plot_ranges,
    fig=plt.figure(figsize=(9, 9)),
    labels=labels,
    titles=symbols,
    label_kwargs={"fontsize": 11},
    title_kwargs={"fontsize": 10},
    truths=truth_params,
    truth_color="C3",
    color="C0",
    bins=35,
    smooth=0.8,
    levels=[0.68, 0.95],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    plot_datapoints=False,
    fill_contours=True,
)
corner_figure.text(0.63, 0.85, "Circular Gaussian posterior\nRed: injected parameters", fontsize=12, ha="center")
plt.show()
