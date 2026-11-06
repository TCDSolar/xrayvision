"""
=======================================
Bayesian recovery of a synthetic source
=======================================

How accurately can we recover a source's flux, position, and size from noisy
visibilities? We start with a circular Gaussian whose parameters are known,
make an initial fit, and use emcee to explore the range of source parameters
consistent with the measurements and our prior assumptions.

We then ask whether the data can distinguish two overlapping circular sources
from a single elliptical source. Dynesty estimates the evidence for each model,
and we compare their reconstructed images, parameter uncertainties, AIC, and BIC.

Read the Docs uses short runs to illustrate the workflow. Set
``XRAYVISION_QUICK_EXAMPLES=1`` to preview these locally, or ``0`` to use the
longer settings. For scientific use, extend sampling until the diagnostics and
independent runs support reliable parameter uncertainties and model comparisons.
"""

import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, triang, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_dynesty, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_image, sources_to_vis, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities

quick_run = os.environ.get("XRAYVISION_QUICK_EXAMPLES", os.environ.get("READTHEDOCS", "False")).lower() in {"1", "true"}
if quick_run:
    print("Quick documentation run: short chains and coarse evidence estimates, extend sampling for scientific use.")

###############################################################################
# Create noisy observations of a known source
# -------------------------------------------
#
# We first define a circular Gaussian and calculate its complex visibilities at
# 30 spatial frequencies. Adding Gaussian noise produces a synthetic observation,
# with separate known noise levels for the real and imaginary components.
# Knowing the original source lets us check how well the fitting recovers it.

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
# Make an initial circular-source fit
# ------------------------------------
#
# A first fit of a circular source is made with the default fitting method,
# Nelder-Mead. This provides a reference image and a starting point for the
# Bayesian fit. We also request a local Gauss-Newton covariance estimate, which
# will set the initial spread and correlations of the emcee walkers. The final
# parameter uncertainties will come from the posterior samples.

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
# Compare the original source with the initial fit
# ------------------------------------------------
#
# Before sampling, we compare the initial reconstruction with the source used
# to create the data. The common image grid and colour scale let us judge its
# position, size, and brightness directly.

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
# Describe plausible source parameters
# -------------------------------------
#
# Bayesian fitting combines how well a source explains the data with our prior
# assumptions about its parameters. Here we allow a broad range of positive
# fluxes and sizes, and use broad normal distributions for position. These
# priors are chosen explicitly, separately from the bounds used by the first fit.

priors = {
    "source[0].amp": ParameterPrior(uniform(loc=0.1, scale=3.9), flux_unit),
    "source[0].x0": ParameterPrior(norm(loc=0, scale=5), apu.arcsec),
    "source[0].y0": ParameterPrior(norm(loc=0, scale=5), apu.arcsec),
    "source[0].sigma": ParameterPrior(uniform(loc=0.5, scale=5.5), apu.arcsec),
}

###############################################################################
# Explore parameter uncertainty with emcee
# ----------------------------------------
#
# We combine the observations, circular-source model, and priors in a Bayesian
# fitting problem. Sixteen walkers start near the first fit, spread according
# to its covariance, and explore the posterior distribution. After discarding
# the initial burn-in steps, their samples describe which parameter values are
# plausible and how those parameters vary together.

problem = BayesianVisibilityProblem(observed, fitted, priors)
positions = rng.multivariate_normal(problem.initial_params, initial_covariance, size=16)
posterior = sample_emcee(
    problem, positions, n_steps=600 if quick_run else 5000, burn_in=200 if quick_run else 500, seed=24
)

###############################################################################
# Check the sampling and recovered parameters
# -------------------------------------------
#
# We inspect the sampling diagnostics before interpreting the uncertainties.
# In particular, ``chain_too_short`` indicates that the run needs extending,
# even an estimated autocorrelation time does not establish convergence by itself.
#
# Because this is a simulation, we can also check whether each injected
# parameter falls within its marginal 95% credible interval. Noise can shift
# the recovered values away from the input, so exact agreement is not expected.

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

# These intervals describe each parameter separately. Even a well-calibrated
# interval can miss the injected value in a particular noisy observation.
print("All injected parameters inside their marginal 95% intervals:", all(coverage.values()))

###############################################################################
# Reconstruct an image from the posterior
# ---------------------------------------
#
# To show one representative reconstruction, we choose the sampled source with
# the highest posterior density. Using a complete sample keeps its parameters
# together, including their correlations. We compare this image with the input
# source and first fit. The full posterior, rather than this one image, describes
# the uncertainty in the reconstruction.

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
# Follow the walkers through the run
# ----------------------------------
#
# Each line below follows a walker's parameter values after burn-in. The traces
# help us see whether the walkers explore a stable region or continue to drift.
# Dashed lines mark the injected values. We interpret these plots alongside the
# diagnostics above, since the walkers are correlated and stable-looking traces
# can still miss another region of plausible solutions.

symbols = [r"$F$", r"$x_0$", r"$y_0$", r"$\sigma$"]
labels = [f"{symbol}\n{unit.to_string('latex_inline')}" for symbol, unit in zip(symbols, posterior.parameter_units)]
trace_figure, trace_axes = plt.subplots(problem.ndim, 1, figsize=(10, 8), sharex=True, layout="constrained")
for column, axis in enumerate(trace_axes):
    axis.plot(posterior.chain[:, :, column], color="C0", alpha=0.15, linewidth=0.4)
    axis.axhline(truth_params[column], color="C3", linestyle="--", linewidth=1)
    axis.set_ylabel(labels[column])
trace_axes[-1].set_xlabel("Production step")
trace_figure.suptitle("Parameter traces, dashed lines show injected values")
plt.show()

###############################################################################
# Compare posterior visibility predictions with the observations
# --------------------------------------------------------------
#
# We also check the reconstruction against the measurements it was fitted to.
# Predicting visibilities from 200 posterior samples gives a range of plausible
# model values at each measured spatial frequency. The orange intervals show
# uncertainty in that prediction, while the blue error bars show measurement
# noise. Red crosses mark the noiseless source used to make the observation.
# The points are ordered by spatial frequency to make the comparison easier.

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
# Plot the emcee posterior with corner
# ------------------------------------
#
# A corner plot brings the parameter uncertainties and correlations together.
# The diagonal panels show each parameter's distribution, with its median and
# 16th/84th percentiles. The panels below show pairs of parameters with 68% and
# 95% probability contours, tilted contours reveal correlations.
#
# Red markers show the injected parameters. Green markers show the same complete
# sample used for the reconstructed image. Their separation reflects both the
# measurement noise and the spread of plausible sources in the posterior.

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
best_params = posterior.samples[np.argmax(posterior.log_prob)]
corner.overplot_lines(corner_figure, best_params, color="C2")
corner.overplot_points(corner_figure, best_params[None, :], color="C2", marker="s", markersize=4)
corner_figure.text(
    0.63,
    0.85,
    "Circular Gaussian posterior\nRed: injected parameters\nGreen: highest-posterior draw",
    fontsize=12,
    ha="center",
)
plt.show()

###############################################################################
# Can one ellipse explain two overlapping sources?
# ------------------------------------------------
#
# So far, we have assumed the source is a circular Gaussian and asked how well
# its parameters can be measured. We now ask which source model the data support.
# Two overlapping circles can resemble a single ellipse in an image. Can the
# visibilities distinguish them, and do they justify the two additional
# parameters needed for two circles?
#
# We create a second observation from two overlapping circular Gaussians. Both
# candidate models will be fitted to these same visibilities, with the same
# spatial-frequency coverage and noise levels.

blend_truth = SourceList(
    [
        Source("circular", 1.1 * flux_unit, -1.6 * apu.arcsec, -0.5 * apu.arcsec, 1.7 * apu.arcsec),
        Source("circular", 0.9 * flux_unit, 1.6 * apu.arcsec, 0.5 * apu.arcsec, 1.7 * apu.arcsec),
    ]
)
comparison_rng = np.random.default_rng(85)
blend_visibilities = sources_to_vis(blend_truth, u, v)
comparison_noise = (
    comparison_rng.normal(0, sigma_real, len(u)) + 1j * comparison_rng.normal(0, sigma_imag, len(u))
) * flux_unit
comparison_observed = Visibilities(
    blend_visibilities + comparison_noise,
    u,
    v,
    uncertainty=np.full(len(u), sigma_real + 1j * sigma_imag) * flux_unit,
)

###############################################################################
# State the assumptions for each model
# ------------------------------------
#
# Model comparison depends on the full prior assumptions, including the range
# of sources each model allows. We give both models the same prior on total
# flux: two independent uniform component fluxes add to a triangular
# distribution, which we use for the ellipse's flux. We also use broad position
# and width priors. The models still imply different priors on quantities such
# as source area and centroid, so the comparison remains conditional on these
# illustrative choices.
#
# The circle labels are arbitrary, so both components receive identical priors
# and can exchange labels during sampling. We will call them left and right
# only when plotting. For the ellipse, we use a 90-degree angle range with
# unordered widths, covering each non-circular ellipse once without counting
# equivalent axis swaps twice.

candidate_sources = {
    "Two circles": SourceList(
        [
            Source("circular", 1 * flux_unit, -1 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec),
            Source("circular", 1 * flux_unit, 1 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec),
        ]
    ),
    "One ellipse": SourceList(
        [
            Source(
                "elliptical", 2 * flux_unit, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec, 2 * apu.arcsec, 0 * apu.rad
            )
        ]
    ),
}
comparison_problems = {}
for model_name, source_list in candidate_sources.items():
    model_priors = {}
    if model_name == "Two circles":
        for source_index in range(2):
            prefix = f"source[{source_index}]."
            model_priors.update(
                {
                    prefix + "amp": ParameterPrior(uniform(0.1, 1.9), flux_unit),
                    prefix + "x0": ParameterPrior(norm(0, 5), apu.arcsec),
                    prefix + "y0": ParameterPrior(norm(0, 5), apu.arcsec),
                    prefix + "sigma": ParameterPrior(uniform(0.5, 5.5), apu.arcsec),
                }
            )
    else:
        model_priors.update(
            {
                "source[0].amp": ParameterPrior(triang(0.5, loc=0.2, scale=3.8), flux_unit),
                "source[0].x0": ParameterPrior(norm(0, 5), apu.arcsec),
                "source[0].y0": ParameterPrior(norm(0, 5), apu.arcsec),
                "source[0].sigmax": ParameterPrior(uniform(0.5, 5.5), apu.arcsec),
                "source[0].sigmay": ParameterPrior(uniform(0.5, 5.5), apu.arcsec),
                "source[0].theta": ParameterPrior(uniform(-np.pi / 4, np.pi / 2), apu.rad),
            }
        )
    comparison_problems[model_name] = BayesianVisibilityProblem(comparison_observed, source_list, model_priors)

###############################################################################
# Compare the evidence with dynesty
# ---------------------------------
#
# Dynesty explores each model starting from its priors and estimates its
# evidence: the likelihood averaged over all the sources allowed by those
# priors. This asks whether the extra flexibility improves the explanation of
# the data enough to outweigh the additional range of possible sources.
#
# In the longer mode, we repeat with an independent seed and more live points
# to see how stable the evidence estimates are. Quick mode runs each model once
# for illustration. We report the estimated numerical errors and, where
# available, the change between runs. The ``dlogz`` setting controls when sampling
# stops, the reported evidence error is a separate estimate of numerical precision.
#
# For two circles, we also inspect how much posterior weight falls in each
# label ordering. Symmetry implies equal probability for the two orderings.
# A large imbalance can indicate that sampling needs extending.

comparison_runs = []
live_point_counts = (50,) if quick_run else (200, 400)
for run_index, nlive in enumerate(live_point_counts):
    run_results = {}
    for model_index, (model_name, model_problem) in enumerate(comparison_problems.items()):
        result = sample_dynesty(
            model_problem,
            nlive=nlive,
            dlogz=1.0 if quick_run else 0.2,
            sample="rwalk",
            seed=81 + 10 * run_index + model_index,
        )
        run_results[model_name] = result
        print(
            f"{model_name}, {nlive} live points: log Z = {result.log_evidence:.2f} +/- {result.log_evidence_error:.2f}"
        )
        if model_name == "Two circles":
            print(
                "Posterior weight with source[0] left of source[1]:",
                result.weights[result.samples[:, 1] < result.samples[:, 5]].sum(),
            )
    comparison_runs.append(run_results)

log_bayes_factors = []
log_bayes_errors = []
for run_results in comparison_runs:
    circular_result = run_results["Two circles"]
    elliptical_result = run_results["One ellipse"]
    log_bayes_factors.append(circular_result.log_evidence - elliptical_result.log_evidence)
    # The two model runs use independent random seeds.
    log_bayes_errors.append(np.hypot(elliptical_result.log_evidence_error, circular_result.log_evidence_error))
    print(f"log Bayes factor (two circles / one ellipse): {log_bayes_factors[-1]:.2f} +/- {log_bayes_errors[-1]:.2f}")
if len(comparison_runs) > 1:
    print("Change in log Bayes factor between runs:", log_bayes_factors[1] - log_bayes_factors[0])
    print("Estimated error on that change:", np.hypot(*log_bayes_errors))

###############################################################################
# Compare AIC and BIC at refined fits
# -----------------------------------
#
# AIC and BIC provide another way to compare the models, balancing the best
# likelihood against a penalty for the number of fitted parameters. Two circles
# have eight parameters, while one ellipse has six. Smaller criteria favour
# a model. Here the real and imaginary measurements are independent, giving
# ``n = 2 * N_vis`` - the known noise levels add no fitted parameters.
#
# These criteria require maximum-likelihood estimates. We take each model's
# highest-likelihood nested sample and refine it with SciPy's L-BFGS-B method,
# using broad physical bounds. This is a local refinement, so finding the global
# maximum still depends on exploring the possible solutions. AIC and BIC are
# approximations, especially for mixtures with coincident or vanishing components.

nested_results = comparison_runs[-1]
maximum_likelihood_statistics = {}
for model_name, result in nested_results.items():
    start = result.sources_from_sample(int(np.argmax(result.log_likelihood)))
    for source in start:
        if model_name == "Two circles":
            source.bounds = [[0, -15, -15, 0.1], [4, 15, 15, 10]]
        else:
            source.bounds = [[0, -15, -15, 0.1, 0.1, -np.pi / 4], [4, 15, 15, 10, 10, np.pi / 4]]
    _, _, refined = vis_forward_fit(
        comparison_observed,
        start,
        image_shape,
        pixel_size,
        method="L-BFGS-B",
        map=False,
        return_result=True,
    )
    if not refined.success:
        raise RuntimeError(f"Maximum-likelihood refinement failed for {model_name}: {refined.message}")
    stats = refined.fit_statistics
    maximum_likelihood_statistics[model_name] = stats
    print(
        f"{model_name}: k={stats.n_parameters}, max log L={stats.log_likelihood:.2f}, "
        f"AIC={stats.aic:.2f}, BIC={stats.bic:.2f}"
    )

###############################################################################
# Bring the images and model comparisons together
# -----------------------------------------------
#
# We can now compare the images alongside the evidence and information criteria.
# Higher evidence favours a model - the log Bayes factor is positive when it
# favours two circles and negative when it favours one ellipse. With equal
# prior model odds, the Bayes factor gives the posterior odds between these
# two hypotheses, conditional on their priors and the measurement model.
#
# As before, each reconstructed image uses the sampled source with the highest
# posterior density. AIC and BIC use the separately refined likelihood fits.
# All plots below use the final nested-sampling run, which has more live points
# in the longer mode.

comparison_images = [sources_to_image(blend_truth, image_shape, pixel_size)]
comparison_best = {}
for model_name, result in nested_results.items():
    log_posterior = result.log_likelihood + np.array([result.problem.log_prior(sample) for sample in result.samples])
    best_index = int(np.argmax(log_posterior))
    comparison_best[model_name] = result.samples[best_index]
    comparison_images.append(sources_to_image(result.sources_from_sample(best_index), image_shape, pixel_size))

model_figure = plt.figure(figsize=(14, 8), layout="constrained")
if quick_run:
    model_figure.suptitle("Quick documentation run: illustrative sampling")
grid = model_figure.add_gridspec(2, 6)
model_axes = [model_figure.add_subplot(grid[0, :2])]
model_axes.extend(
    model_figure.add_subplot(grid[0, i : i + 2], sharex=model_axes[0], sharey=model_axes[0]) for i in (2, 4)
)
vmax = max(image.value.max() for image in comparison_images)
for panel, (axis, image, title) in enumerate(
    zip(model_axes, comparison_images, ("Injected overlapping circles", "Two-circle model", "One-ellipse model"))
):
    artist = axis.imshow(image.value, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    axis.set(xlim=(-10, 10), ylim=(-10, 10), title=title)
    axis.tick_params(bottom=panel == 0, left=panel == 0, labelbottom=panel == 0, labelleft=panel == 0)
model_axes[0].set(xlabel="x [arcsec]", ylabel="y [arcsec]")
model_figure.colorbar(artist, ax=model_axes, label=brightness_unit.to_string("latex_inline"))
evidence_axis = model_figure.add_subplot(grid[1, :3])
for run_index, (run_results, nlive) in enumerate(zip(comparison_runs, live_point_counts)):
    evidence_axis.errorbar(
        np.arange(2) + (run_index - (len(comparison_runs) - 1) / 2) * 0.12,
        [result.log_evidence for result in run_results.values()],
        yerr=[result.log_evidence_error for result in run_results.values()],
        fmt="o",
        capsize=4,
        label=f"{nlive} live points",
    )
evidence_axis.set(xticks=[0, 1], xticklabels=list(nested_results), ylabel="Log evidence (larger is favoured)")
evidence_axis.set_title(
    rf"Two circles / one ellipse: $\log B = {log_bayes_factors[-1]:.1f} \pm {log_bayes_errors[-1]:.1f}$"
)
evidence_axis.legend()
criteria_axis = model_figure.add_subplot(grid[1, 3:])
criteria_axis.axis("off")
criteria_axis.set_title("Information criteria (smaller is favoured)")
criteria_table = criteria_axis.table(
    cellText=[
        [name, stats.n_parameters, f"{stats.log_likelihood:.1f}", f"{stats.aic:.1f}", f"{stats.bic:.1f}"]
        for name, stats in maximum_likelihood_statistics.items()
    ],
    colLabels=["Model", "k", "max log L", "AIC", "BIC"],
    cellLoc="center",
    loc="center",
)
criteria_table.auto_set_font_size(False)
criteria_table.set_fontsize(10)
criteria_table.scale(1, 2)
criteria_axis.text(
    0.5,
    0.2,
    f"Refined maximum-likelihood fits: n = {2 * len(u)} real measurements\n"
    "Images above show highest-posterior sampled draws.",
    ha="center",
    transform=criteria_axis.transAxes,
)
plt.show()

###############################################################################
# Explore the uncertainty within each model
# -----------------------------------------
#
# Finally, we examine the parameters of each candidate model. For overlapping
# circles, one component's flux or position can trade off against the other's.
# The corner plot makes these correlations visible. We assign left/right labels
# for display so that interchangeable source labels do not obscure the result.
# Red markers show the injected circular components. There is no injected ellipse
# parameter vector, so its plot has no red markers.
#
# Dynesty samples carry different posterior weights, which must be included in
# the histograms and quantiles. These plots describe uncertainty within each
# model. We use the evidence and information criteria above to compare models.

blend_parameters = np.asarray(blend_truth.to_canonical(flux_unit).params)
for model_name, result in nested_results.items():
    display_samples = result.samples.copy()
    best = comparison_best[model_name].copy()
    if model_name == "Two circles":
        components = display_samples.reshape(-1, 2, 4)
        order = np.argsort(components[:, :, 1], axis=1)
        display_samples = np.take_along_axis(components, order[:, :, None], axis=1).reshape(-1, 8)
        best = best.reshape(2, 4)[np.argsort(best.reshape(2, 4)[:, 1])].ravel()
        model_symbols = [rf"${symbol}_{{{side}}}$" for side in ("L", "R") for symbol in ("F", "x", "y", r"\sigma")]
        model_truth = list(blend_parameters)
    else:
        model_symbols = [r"$F$", r"$x_0$", r"$y_0$", r"$\sigma_x$", r"$\sigma_y$", r"$\theta$"]
        model_truth = [None] * result.problem.ndim
    model_labels = [
        f"{symbol}\n{unit.to_string('latex_inline')}" for symbol, unit in zip(model_symbols, result.parameter_units)
    ]
    weights = result.weights
    mean = np.average(display_samples, axis=0, weights=weights)
    spread = np.sqrt(np.average((display_samples - mean) ** 2, axis=0, weights=weights))
    model_ranges = []
    for column, injected in enumerate(model_truth):
        low, high = corner.quantile(display_samples[:, column], [0.001, 0.999], weights=weights)
        if injected is not None:
            low, high = min(low, injected), max(high, injected)
        model_ranges.append((low - spread[column], high + spread[column]))
    figure = corner.corner(
        display_samples,
        weights=weights,
        range=model_ranges,
        fig=plt.figure(figsize=(16, 16) if model_name == "Two circles" else (13, 13)),
        labels=model_labels,
        titles=model_symbols,
        truths=model_truth,
        truth_color="C3",
        color="C0",
        bins=30,
        smooth=0.8,
        levels=[0.68, 0.95],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        plot_datapoints=False,
        fill_contours=True,
        label_kwargs={"fontsize": 10},
        title_kwargs={"fontsize": 9},
        max_n_ticks=3,
    )
    corner.overplot_lines(figure, best, color="C2")
    corner.overplot_points(figure, best[None, :], color="C2", marker="s", markersize=4)
    figure.text(
        0.65,
        0.85,
        f"{model_name} model: weighted posterior\n"
        + (
            "Red: injected left/right components\n"
            if model_name == "Two circles"
            else "No injected ellipse parameter vector\n"
        )
        + "Green: highest-posterior sampled draw",
        ha="center",
        fontsize=11,
    )
    plt.show()
