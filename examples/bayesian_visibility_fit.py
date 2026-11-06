"""
=======================================
Bayesian recovery of a synthetic source
=======================================

Fit noisy complex X-ray visibilities with the existing circular Gaussian model,
inspect emcee traces and visibility predictions, and plot the posterior with
corner. Then compare two overlapping circular Gaussians with one elliptical
Gaussian using dynesty evidence, AIC, and BIC, showing how parameter estimation
differs from model comparison.
The example uses the public fitting API; no separate source implementation is
needed. Install the documentation dependencies to run the plotting example,
including ``corner`` (``pip install 'xrayvisim[docs]'``).
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, triang, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_dynesty, sample_emcee
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
# Plot the emcee posterior with corner
# ------------------------------------
#
# ``samples`` already contains the flattened production draws in the order of
# ``parameter_names`` and ``parameter_units``. No sampler internals are needed.
# The diagonal panels show marginal distributions and the 16th, 50th, and 84th
# percentiles; the lower panels show joint distributions with 68% and 95%
# probability contours. Red markers indicate the injected parameters and green
# markers show the highest-posterior draw used for the sampled image. A corner
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
# Compare source models with dynesty
# ----------------------------------
#
# Emcee estimates parameters within a chosen model. Nested sampling also
# estimates the evidence, the likelihood averaged over the model's entire prior.
# A flexible model can fit better but is supported only when this improvement
# outweighs the prior volume giving poor fits. Evidence depends on the priors;
# it is not just a comparison of the best likelihoods or parameter counts.
#
# Use a new noisy dataset from two overlapping circular sources, keeping the
# same spatial frequencies, flux units, and noise convention for both models.
# No SciPy fit or initial ensemble is needed for nested sampling.

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
# Keep model priors explicit and comparable
# -----------------------------------------
#
# Each circular component has a uniform flux prior [0.1, 2]. Their sum has a
# triangular distribution on [0.2, 4], which is used for the ellipse's total
# flux, giving both models the same total-flux prior. Every position has a
# normal prior with mean zero and standard deviation 5 arcsec, and each width
# is uniform on [0.5, 6] arcsec. These choices do not give identical priors on
# centroid or source area. Evidence remains conditional on these assumptions.
#
# The two circles have identical, independent priors. Both label permutations
# belong to the sampled model: no ordering constraint or factorial correction
# is applied to the evidence. Left/right labels are assigned only for plotting.
# For the ellipse, theta spans 90 degrees and the two widths remain unordered.
# Swapping the axes and rotating by 90 degrees gives an equivalent ellipse,
# so this chart covers each non-circular physical ellipse once. It avoids
# imposing an ordering constraint on independent width priors. A circular
# ellipse still has no identifiable orientation.

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
# Estimate evidence and check repeatability
# -----------------------------------------
#
# Repeat with an independent seed and more live points. Report dynesty's
# estimated errors and the change between runs, rather than claiming that a
# stopping criterion proves convergence. Here dlogz limits the estimated
# remaining evidence contribution; it is not the error bar on log evidence.
# The example is modest in size; difficult or multimodal data can require many
# more live points and further checks.
# Random-walk proposals help explore the correlated blend parameters. The
# Symmetry implies equal posterior probability for the two label orderings.
# The printed weights assess this finite-run approximation; left/right
# relabelling cannot repair a missed mode.

comparison_runs = []
live_point_counts = (200, 400)
for run_index, nlive in enumerate(live_point_counts):
    run_results = {}
    for model_index, (model_name, model_problem) in enumerate(comparison_problems.items()):
        result = sample_dynesty(
            model_problem, nlive=nlive, dlogz=0.2, sample="rwalk", seed=81 + 10 * run_index + model_index
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
print("Change in log Bayes factor between runs:", log_bayes_factors[1] - log_bayes_factors[0])
print("Estimated error on that change:", np.hypot(*log_bayes_errors))

###############################################################################
# Refine maximum likelihood and report AIC and BIC
# -------------------------------------------------
#
# AIC = 2*k - 2*log(L_max), BIC = k*log(n) - 2*log(L_max), with k=8 for two
# circles and k=6 for one ellipse. Here n=2*N_vis counts the independent real
# and imaginary measurements; the supplied noise scales are not fitted.
# These use maximum likelihood, not maximum posterior density or evidence.
# Start the existing SciPy fitter at the best likelihood draw and refine it.
# Broad physical bounds are optimisation safeguards, not Bayesian priors.
# This local refinement does not guarantee that a global maximum was found.
#
# Smaller AIC/BIC values favour a model. They are asymptotic approximations,
# particularly fragile for mixtures whose components coincide or disappear;
# they should not be treated as exact substitutes for the evidence.

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
# See the competing images, evidence, and information criteria
# ------------------------------------------------------------
#
# A positive log Bayes factor favours two circles under these priors. Equal
# prior model odds would give posterior model odds equal to the Bayes factor;
# these are not probabilities that either model is the true source.
#
# Use an actual high-posterior draw for each image, selected using likelihood
# plus prior density. The largest nested-sampling weight is not the posterior
# mode: weights also reflect the prior volume represented by each sample.
# Use the second, larger run for the images and corner plots below.

comparison_images = [sources_to_image(blend_truth, image_shape, pixel_size)]
comparison_best = {}
for model_name, result in nested_results.items():
    log_posterior = result.log_likelihood + np.array([result.problem.log_prior(sample) for sample in result.samples])
    best_index = int(np.argmax(log_posterior))
    comparison_best[model_name] = result.samples[best_index]
    comparison_images.append(sources_to_image(result.sources_from_sample(best_index), image_shape, pixel_size))

model_figure = plt.figure(figsize=(14, 8), layout="constrained")
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
        np.arange(2) + (run_index - 0.5) * 0.12,
        [result.log_evidence for result in run_results.values()],
        yerr=[result.log_evidence_error for result in run_results.values()],
        fmt="o",
        capsize=4,
        label=f"{nlive} live points, independent seeds",
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
    f"Refined maximum-likelihood fits; n = {2 * len(u)} real measurements\n"
    "Images above show highest-posterior sampled draws.",
    ha="center",
    transform=criteria_axis.transAxes,
)
plt.show()

###############################################################################
# Inspect weighted parameter distributions for both models
# ---------------------------------------------------------
#
# Nested samples require weights in both histograms and marginal quantiles.
# Display the weighted central 99.8 percent range, padded by a posterior standard
# deviation and extended to include injected values where the parameter exists.
# Sort each two-circle draw by x position for display, including the selected
# highest-posterior draw, while preserving the original samples and weights.
# An ellipse has no injected parameter vector for a two-circle source, so no
# red truth markers are shown for that model. A precise posterior can accompany
# the wrong morphology; uncertainty within one model does not assess that model.

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
