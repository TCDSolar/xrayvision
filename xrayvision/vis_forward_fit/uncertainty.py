"""Parameter uncertainty estimates shared by visibility fitting backends."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from operator import index
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

import astropy.units as apu
from astropy.units import Quantity

from xrayvision.vis_forward_fit.fit_data import VisibilityFitData
from xrayvision.vis_forward_fit.sources import Elliptical, GenericSource, Loop, SourceList
from xrayvision.visibility import Visibilities

__all__ = ["ParameterUncertainty", "estimate_parameter_uncertainty"]


@dataclass
class ParameterUncertainty:
    """A backend-independent frequentist uncertainty estimate.

    ``stderr`` contains one-standard-deviation errors as quantities. ``intervals``
    contains pairs of lower/upper endpoints at ``confidence_level``. Covariance
    and samples are numeric in the ordered ``parameter_names``/``parameter_units``;
    covariance entry (i, j) has the product of units i and j. Use
    `covariance_between` to retrieve a quantity. Angular samples and intervals
    are unwrapped around the fitted source. These are not posterior samples or
    Bayesian credible intervals.

    ``status`` is ``'ok'``, ``'partial'`` (some refits failed), ``'bounds_hit'``,
    ``'ill_conditioned'``, or an unavailable-estimate state such as
    ``'failed'``, ``'fit_failed'``, ``'rank_deficient'``, or ``'active_bounds'``.
    Unavailable estimates have no covariance or parameter summaries; details
    are retained in ``diagnostics``. Raw PSO populations remain on the native
    result and are not included in these statistical samples.
    """

    method: str
    parameter_names: tuple[str, ...]
    parameter_units: tuple[Any, ...]
    confidence_level: float
    status: str = "ok"
    stderr: dict[str, Quantity] = field(default_factory=dict)
    intervals: dict[str, Quantity] = field(default_factory=dict)
    covariance: NDArray[np.float64] | None = None
    samples: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def covariance_between(self, first: str, second: str) -> Quantity:
        """Return one covariance entry with its product unit."""
        if self.covariance is None:
            raise ValueError(f"Covariance is unavailable: {self.status}.")
        i, j = self.parameter_names.index(first), self.parameter_names.index(second)
        return self.covariance[i, j] * self.parameter_units[i] * self.parameter_units[j]


def _fit_succeeded(result: Any) -> bool:
    """Require SciPy success, or a finite feasible native PSO result."""
    if getattr(result, "success", None) is not None and not result.success:
        return False
    values = getattr(result, "x", None)
    if values is None:
        values = getattr(result, "X", None)
    if values is None or not np.all(np.isfinite(values)):
        return False
    objective = getattr(result, "fun", None)
    if objective is None:
        objective = getattr(result, "F", None)
    violation = getattr(result, "CV", None)
    return bool(
        objective is not None
        and np.all(np.isfinite(objective))
        and (violation is None or np.all(np.asarray(violation) <= 0))
    )


def _parameter_scales(sources: SourceList, bounds: NDArray[np.float64]) -> NDArray[np.float64]:
    """Physical parameter scales for finite differences and source matching."""
    x = np.asarray(sources.params, dtype=float)
    span = bounds[1] - bounds[0]
    scales = np.where(np.isfinite(span) & (span > 0), span, np.maximum(np.abs(x), 1))
    return np.asarray(scales, dtype=float)


def _bound_hits(x: NDArray[np.float64], bounds: NDArray[np.float64], scales: NDArray[np.float64]) -> NDArray[np.bool_]:
    tolerance = 1e-6 * scales
    return np.asarray((x - bounds[0] <= tolerance) | (bounds[1] - x <= tolerance), dtype=bool)


def _equivalent_parameters(
    candidate: GenericSource, reference: GenericSource, scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Choose equivalent ellipse axes and unwrap orientation about a reference."""
    target = np.asarray(reference.param_list, dtype=float)
    values = np.asarray(candidate.param_list, dtype=float).copy()
    alternatives = [values]
    if isinstance(candidate, Elliptical):
        swapped = values.copy()
        swapped[3:5] = values[3:5][::-1]
        swapped[5] += np.pi / 2
        alternatives.append(swapped)
    angle_index = 5 if isinstance(candidate, (Elliptical, Loop)) else None
    period = np.pi if isinstance(candidate, Elliptical) else 2 * np.pi
    if angle_index is not None:
        for alternative in alternatives:
            delta = alternative[angle_index] - target[angle_index]
            alternative[angle_index] = target[angle_index] + (delta + period / 2) % period - period / 2
    return min(alternatives, key=lambda alternative: float(np.sum(((alternative - target) / scales) ** 2)))


def _align_sources(candidate: SourceList, reference: SourceList, scales: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match interchangeable source types before covariance/interval summaries."""
    offsets = np.cumsum([0, *[source.n_params for source in reference]])
    aligned = np.empty(len(reference.params))
    for source_type in {type(source) for source in reference}:
        ref_indices = [i for i, source in enumerate(reference) if type(source) is source_type]
        candidates = [source for source in candidate if type(source) is source_type]
        if len(candidates) != len(ref_indices):
            raise ValueError("Refitted source types do not match the original model.")
        cost = np.empty((len(ref_indices), len(candidates)))
        alternatives = {}
        for row, i in enumerate(ref_indices):
            scale = scales[offsets[i] : offsets[i + 1]]
            target = np.asarray(reference[i].param_list, dtype=float)
            for col, source in enumerate(candidates):
                values = _equivalent_parameters(source, reference[i], scale)
                alternatives[row, col] = values
                cost[row, col] = np.sum(((values - target) / scale) ** 2)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            i = ref_indices[row]
            aligned[offsets[i] : offsets[i + 1]] = alternatives[int(row), int(col)]
    return aligned


def _refit_sample(task: tuple[Any, ...]) -> tuple[SourceList | None, str]:
    """One picklable refit, without recursively requesting uncertainty estimates."""
    from xrayvision.vis_forward_fit.forward_fit import _vis_forward_fit_minimise  # pyright: ignore[reportPrivateUsage]

    vis, sources, fit_method, weighting, bounds, n_runs, pop = task
    try:
        fitted, result = _vis_forward_fit_minimise(
            vis,
            sources,
            fit_method,
            weighting=weighting,
            bounds=bounds,
            n_runs=n_runs,
            pop=pop,
            n_jobs=1,
        )
        if not _fit_succeeded(result):
            return None, str(getattr(result, "message", "Refit returned no finite feasible solution."))
        return fitted, ""
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return None, f"{type(error).__name__}: {error}"


def _set_summaries(
    estimate: ParameterUncertainty, covariance: NDArray[np.float64], endpoints: NDArray[np.float64]
) -> None:
    estimate.covariance = covariance
    for i, (name, unit) in enumerate(zip(estimate.parameter_names, estimate.parameter_units)):
        estimate.stderr[name] = np.sqrt(max(covariance[i, i], 0)) * unit
        estimate.intervals[name] = endpoints[:, i] * unit


def _monte_carlo(
    vis: Visibilities,
    sources: SourceList,
    data: VisibilityFitData,
    estimate: ParameterUncertainty,
    bounds: NDArray[np.float64],
    *,
    fit_method: str,
    n_samples: int,
    seed: int | None,
    n_jobs: int,
    store_samples: bool,
    n_runs: int,
    pop: int,
) -> ParameterUncertainty:
    from xrayvision.vis_forward_fit.forward_fit import sources_to_vis

    n_samples, n_jobs = index(n_samples), index(n_jobs)
    if n_samples < 2 or n_jobs < 1:
        raise ValueError("Monte Carlo requires n_samples >= 2 and n_jobs >= 1.")
    rng = np.random.default_rng(seed)
    mean = np.asarray(sources_to_vis(sources, data.u, data.v))
    # Both estimators require known standard deviations; validated by the entry point.
    assert data.sigma_ri is not None
    sigma_r, sigma_i = np.split(data.sigma_ri, 2)
    noise = rng.normal(size=(n_samples, 2, mean.size))
    draws = mean + noise[:, 0] * sigma_r + 1j * noise[:, 1] * sigma_i
    tasks = []
    for draw in draws:
        errors = {"uncertainty": (sigma_r + 1j * sigma_i) * data.flux_unit}
        if data.weighting == "amplitude":
            errors = {"amplitude_uncertainty": sigma_r * data.flux_unit}
        synthetic = Visibilities(
            draw * data.flux_unit,
            data.u / apu.arcsec,
            data.v / apu.arcsec,
            phase_center=vis.phase_center,
            meta=vis.meta,
            **errors,
        )
        tasks.append((synthetic, sources, fit_method, data.weighting, bounds, n_runs, pop))
    if n_jobs == 1:
        results = [_refit_sample(task) for task in tasks]
    else:
        # Draw noise in the parent first: a seed gives the same data in serial and
        # parallel. PSO restarts inside each independent refit remain sequential.
        with ProcessPoolExecutor(max_workers=min(n_jobs, n_samples)) as executor:
            results = list(executor.map(_refit_sample, tasks))
    scales = _parameter_scales(sources, bounds)
    samples = []
    hits = np.zeros(len(sources.params), dtype=int)
    failures = []
    for i, (fitted, message) in enumerate(results):
        if fitted is None:
            failures.append({"sample": i, "message": message})
            continue
        # Bound diagnostics refer to the original parameter labels and boxes;
        # summaries use matched labels and equivalent angular representations.
        hits += _bound_hits(np.asarray(fitted.params), bounds, scales)
        samples.append(_align_sources(fitted, sources, scales))
    estimate.diagnostics.update(
        n_requested=n_samples,
        n_successful=len(samples),
        n_failed=len(failures),
        failures=failures,
        seed=seed,
        centre="fitted_model",
        interval="percentile",
        fit_method=fit_method,
        n_jobs=n_jobs,
        bound_hits=dict(zip(estimate.parameter_names, hits.tolist())),
    )
    sample_array = np.asarray(samples, dtype=float).reshape(-1, len(sources.params))
    if store_samples:
        estimate.samples = sample_array
    if len(samples) < 2:
        estimate.status = "failed"
        estimate.diagnostics["message"] = "Fewer than two successful refits; uncertainty is unavailable."
        return estimate
    alpha = (1 - estimate.confidence_level) / 2
    _set_summaries(
        estimate,
        np.atleast_2d(np.cov(sample_array, rowvar=False, ddof=1)),
        np.quantile(sample_array, [alpha, 1 - alpha], axis=0),
    )
    estimate.status = "partial" if failures else "bounds_hit" if np.any(hits) else "ok"
    return estimate


def _gauss_newton(
    sources: SourceList,
    data: VisibilityFitData,
    estimate: ParameterUncertainty,
    bounds: NDArray[np.float64],
    *,
    relative_step: float,
    rcond: float | None,
) -> ParameterUncertainty:
    from xrayvision.vis_forward_fit.forward_fit import sources_to_vis

    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative_step must be finite and positive.")
    if rcond is not None and (not np.isfinite(rcond) or not 0 < rcond < 1):
        raise ValueError("rcond must lie strictly between zero and one.")
    x = np.asarray(sources.params, dtype=float)
    scales = _parameter_scales(sources, bounds)
    active = _bound_hits(x, bounds, scales)
    # A sigma ordering constraint can also be active without touching a box.
    offset = 0
    for source in sources:
        if isinstance(source, Loop) and source.sigma_max - source.sigma_min <= 1e-6 * scales[offset + 4]:
            active[offset + 3 : offset + 5] = True
        offset += source.n_params
    estimate.diagnostics["active_bounds"] = [name for name, hit in zip(estimate.parameter_names, active) if hit]
    if np.any(active):
        estimate.status = "active_bounds"
        estimate.diagnostics["message"] = (
            "A symmetric unconstrained Gaussian covariance is inappropriate at an active bound."
        )
        return estimate

    def residual(params: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = sources_to_vis(SourceList.from_params(sources, params), data.u, data.v)
        result = data.residuals(prediction)
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite residuals while differentiating the visibility model.")
        return result

    # Differentiate in scaled coordinates to avoid comparing Jy, arcseconds and
    # radians by their raw magnitudes. Neither chi-square nor J is reduced by dof.
    jacobian = np.empty((2 * data.values.size, x.size))
    for i in range(x.size):
        step = min(relative_step * scales[i], (x[i] - bounds[0, i]) / 2, (bounds[1, i] - x[i]) / 2)
        if not np.isfinite(step) or step <= 0 or x[i] + step == x[i] or x[i] - step == x[i]:
            estimate.status = "failed"
            estimate.diagnostics["message"] = f"No usable finite-difference step for {estimate.parameter_names[i]}."
            return estimate
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise"):
                jacobian[:, i] = (residual(plus) - residual(minus)) / (2 * step / scales[i])
        except (ValueError, FloatingPointError) as error:
            estimate.status = "failed"
            estimate.diagnostics["message"] = str(error)
            return estimate
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError as error:
        estimate.status = "failed"
        estimate.diagnostics["message"] = f"Jacobian decomposition failed: {error}"
        return estimate
    # A numerical Jacobian has finite-difference noise above machine precision.
    # Do not interpret roundoff in a truly flat direction as information.
    rank_tolerance = (
        rcond
        if rcond is not None
        else max(
            np.finfo(float).eps * max(jacobian.shape),
            10 * np.finfo(float).eps / relative_step,
        )
    )
    threshold = rank_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    estimate.diagnostics.update(
        rank=rank,
        singular_values=singular_values,
        relative_step=relative_step,
        rcond=rank_tolerance,
        rescaled=False,
    )
    if rank < x.size:
        estimate.status = "rank_deficient"
        estimate.diagnostics["message"] = (
            "The residual Jacobian does not identify every parameter; no finite covariance is reported."
        )
        return estimate
    condition = float(singular_values[0] / singular_values[-1])
    estimate.diagnostics["condition_number"] = condition
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = vt.T / singular_values
        covariance_scaled = inverse @ inverse.T
        covariance = scales[:, None] * covariance_scaled * scales[None, :]
    if not np.all(np.isfinite(covariance)):
        estimate.status = "failed"
        estimate.diagnostics["message"] = "Covariance cannot be represented with finite numeric values."
        return estimate
    z = norm.ppf((1 + estimate.confidence_level) / 2)
    half_width = z * np.sqrt(np.maximum(np.diag(covariance), 0))
    endpoints = np.stack([x - half_width, x + half_width])
    _set_summaries(estimate, covariance, endpoints)
    crossing = (endpoints[0] < bounds[0]) | (endpoints[1] > bounds[1])
    estimate.diagnostics["interval_crosses_bounds"] = [
        name for name, crosses in zip(estimate.parameter_names, crossing) if crosses
    ]
    estimate.status = "ill_conditioned" if condition > 1e8 else "bounds_hit" if np.any(crossing) else "ok"
    return estimate


def estimate_parameter_uncertainty(
    vis: Visibilities,
    sources: SourceList,
    *,
    method: str = "monte_carlo",
    fit_method: str = "Nelder-Mead",
    weighting: str = "auto",
    bounds: NDArray[np.float64] | None = None,
    n_samples: int = 200,
    seed: int | None = None,
    n_jobs: int = 1,
    store_samples: bool = True,
    confidence_level: float = 0.6826894921370859,
    n_runs: int = 1,
    pop: int = 100,
    relative_step: float = float(np.cbrt(np.finfo(float).eps)),
    rcond: float | None = None,
) -> ParameterUncertainty:
    """Estimate parameter errors for an already fitted `SourceList`.

    ``method='monte_carlo'`` simulates Gaussian real/imaginary noise about the
    fitted model, refits with ``fit_method``, and reports empirical standard
    deviations and percentile intervals. The same numeric ``bounds`` are used
    for every refit. Supply the original ``result.fit_bounds`` to preserve the
    original search domain; otherwise the source's default bounds are used.
    ``seed`` reproduces the synthetic data in serial and parallel runs.

    ``method='gauss_newton'`` differentiates the shared weighted residuals and
    computes a local covariance with an SVD. ``relative_step`` controls the
    scaled central differences and ``rcond`` the numerical rank threshold.
    Supplied measurement errors are treated as absolute; covariance is never
    multiplied by reduced chi-square. Both methods require known errors.

    ``confidence_level`` sets the central interval probability. MC-specific
    controls are ``n_samples``, ``seed``, ``n_jobs``, ``store_samples``, ``n_runs``
    (PSO restarts per refit) and ``pop`` (PSO particles). Matching identical source
    types and equivalent ellipse axes occurs before summarising MC samples.
    This matching does not establish uniqueness of a multimodal solution.
    """
    if method not in {"monte_carlo", "gauss_newton"}:
        raise ValueError("Uncertainty method must be 'monte_carlo' or 'gauss_newton'.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if not sources:
        raise ValueError("At least one fitted source is required.")
    data = VisibilityFitData(vis, weighting=weighting)
    if data.sigma_ri is None:
        raise ValueError("Parameter uncertainty estimation requires known measurement standard deviations.")
    sources = sources.to_canonical(data.flux_unit)
    fit_bounds = np.asarray(sources.bounds if bounds is None else bounds, dtype=float)
    x = np.asarray(sources.params, dtype=float)
    if fit_bounds.shape != (2, x.size) or np.any(np.isnan(fit_bounds)) or np.any(fit_bounds[0] >= fit_bounds[1]):
        raise ValueError("bounds must have shape (2, n_params) with each lower bound below its upper bound.")
    if not np.all(np.isfinite(x)) or np.any(x < fit_bounds[0]) or np.any(x > fit_bounds[1]):
        raise ValueError("Fitted parameters must be finite and inside the supplied bounds.")
    estimate = ParameterUncertainty(
        method, sources.parameter_names, sources.parameter_units(data.flux_unit), confidence_level
    )
    estimate.diagnostics["weighting"] = data.weighting
    if method == "monte_carlo":
        return _monte_carlo(
            vis,
            sources,
            data,
            estimate,
            fit_bounds,
            fit_method=fit_method,
            n_samples=n_samples,
            seed=seed,
            n_jobs=n_jobs,
            store_samples=store_samples,
            n_runs=n_runs,
            pop=pop,
        )
    return _gauss_newton(sources, data, estimate, fit_bounds, relative_step=relative_step, rcond=rcond)
