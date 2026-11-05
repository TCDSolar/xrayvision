from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

import astropy.units as apu
from astropy.units import Quantity, quantity_input

from sunpy.map import GenericMap, Map

from xrayvision.imaging import generate_header
from xrayvision.transform import generate_xy
from xrayvision.vis_forward_fit.fit_data import FitStatistics, VisibilityFitData
from xrayvision.vis_forward_fit.sources import (
    Circular,
    Elliptical,
    Loop,
    SourceList,
    circular_gaussian_img,
    circular_gaussian_vis,
    elliptical_gaussian_img,
    elliptical_gaussian_vis,
    loop_img,
    loop_vis,
)
from xrayvision.vis_forward_fit.uncertainty import (
    ParameterUncertainty,
    _fit_succeeded,  # pyright: ignore[reportPrivateUsage]
    estimate_parameter_uncertainty,
)
from xrayvision.visibility import Visibilities

if TYPE_CHECKING:
    from pymoo.core.result import Result

__all__ = [
    "SOURCE_TO_IMAGE",
    "SOURCE_TO_VIS",
    "fit_statistics",
    "sources_to_image",
    "sources_to_vis",
    "vis_forward_fit",
]

#: Mapping of sources to image generation functions
SOURCE_TO_IMAGE: dict[str, Callable[..., Any]] = {
    Circular.__name__: circular_gaussian_img,
    Elliptical.__name__: elliptical_gaussian_img,
    Loop.__name__: loop_img,
}

#: Mapping of sources to visibility generation functions
SOURCE_TO_VIS: dict[str, Callable[..., Any]] = {
    Circular.__name__: circular_gaussian_vis,
    Elliptical.__name__: elliptical_gaussian_vis,
    Loop.__name__: loop_vis,
}


def sources_to_image(
    source_list: SourceList,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    center: Quantity[apu.arcsec] = (0, 0) * apu.arcsec,
) -> Quantity | NDArray[np.float64]:
    r"""
    Create an image from a list of sources.

    Parameters
    ----------
    source_list :
        List of sources and their parameters
    shape :
        Shape of the image create
    pixel_size :
        Size

    Returns
    -------

    """
    if not source_list:
        raise ValueError("At least one source is required.")
    flux_unit = source_list.flux_unit
    source_list = source_list.to_canonical(flux_unit)
    image = None
    # Source params (other than amp) are plain floats in canonical arcsec/radian units
    # (see xrayvision.vis_forward_fit.sources), so the grid they're evaluated on must be too.
    x = generate_xy(shape[1], pixel_size=pixel_size[1], phase_center=center[1]).to_value(apu.arcsec)
    y = generate_xy(shape[0], pixel_size=pixel_size[0], phase_center=center[0]).to_value(apu.arcsec)
    x, y = np.meshgrid(x, y)
    for source in source_list:
        try:
            if image is None:
                image = SOURCE_TO_IMAGE[source.__class__.__name__](
                    *[source.param_list[0], x, y, *source.param_list[1:]]
                )
            else:
                image += SOURCE_TO_IMAGE[source.__class__.__name__](
                    *[source.param_list[0], x, y, *source.param_list[1:]]
                )
        except KeyError:
            raise KeyError(f"Unknown source type: {source.__class__.__name__}")
    return image if flux_unit is None else image * flux_unit


def sources_to_vis(
    source_list: SourceList, u: Quantity | NDArray[np.float64], v: Quantity | NDArray[np.float64]
) -> Quantity | NDArray[np.complex128]:
    r"""
    Create visibilities from a list of sources.

    Parameters
    ----------
    source_list
    u :
        u coordinates to evaluate sources
    v :
        u coordinates to evaluate sources

    Returns
    -------
    vis :
        Complex visibilities
    """
    # Source params (other than amp) are plain floats in canonical arcsec/radian units
    # (see xrayvision.vis_forward_fit.sources), so u, v must be converted to match (1/arcsec).
    # When they're already plain floats (e.g. the internal fit loop, which always works in
    # canonical units), this is a no-op and nothing is converted.
    if isinstance(u, Quantity):
        u = u.to_value(1 / apu.arcsec)  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(v, Quantity):
        v = v.to_value(1 / apu.arcsec)  # pyright: ignore[reportAttributeAccessIssue]
    flux_unit = source_list.flux_unit
    source_list = source_list.to_canonical(flux_unit)
    vis = np.zeros(u.shape, dtype=np.complex128)
    for source in source_list:
        try:
            vis += SOURCE_TO_VIS[source.__class__.__name__](*[source.param_list[0], u, v, *source.param_list[1:]])
        except KeyError:
            raise KeyError(f"Unknown source type: {source.__class__.__name__}")
    return vis if flux_unit is None else vis * flux_unit


def fit_statistics(vis: Visibilities, sources: SourceList, *, weighting: str = "auto") -> FitStatistics:
    """Calculate fit diagnostics using the same residuals as SciPy and PSO.

    All source parameters are counted as free. Reduced chi-square uses
    ``2 * n_visibilities - n_parameters`` and is unavailable without errors
    or with non-positive degrees of freedom. See `VisibilityFitData` for
    the error conventions selected by ``weighting``.
    """
    data = VisibilityFitData(vis, weighting=weighting)
    numeric_sources = sources.to_canonical(data.flux_unit)
    prediction = sources_to_vis(numeric_sources, data.u, data.v)
    return data.statistics(prediction, sum(source.n_params for source in sources))


def _run_one_pso(
    visobs: Visibilities,
    sources: SourceList,
    seed: int,
    pop: int,
    weighting: str,
    bounds: NDArray[np.float64],
) -> "Result":
    r"""Run a single PSO optimisation. A module-level function so it can be pickled and
    sent to worker processes by `_vis_forward_fit_minimise`'s multi-restart path.
    """
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.optimize import minimize as moo_minimize

    from xrayvision.vis_forward_fit._pso import VisForwardFitProblem

    problem = VisForwardFitProblem(visobs.u, visobs.v, visobs, sources, weighting=weighting, bounds=bounds)
    algo = PSO(pop_size=pop)
    return moo_minimize(problem, algo, seed=seed)


def _vis_forward_fit_minimise(
    visobs: Visibilities,
    sources: SourceList,
    method: str,
    n_runs: int = 1,
    n_jobs: int = 1,
    pop: int = 100,
    *,
    weighting: str = "auto",
    bounds: NDArray[np.float64] | None = None,
) -> tuple[SourceList, Any]:
    r"""
    Internal minimisation function

    Parameters
    ----------
    visobs :
        Input Visibilities
    sources :
        List of sources
    method :
        Method to use for the minimisation
    n_runs :
        PSO only. Number of independent PSO optimisations to run (each with fresh
        random initialisation), keeping the lowest-chi2 result. All particles in a
        single PSO run share one global best, so if that run's global best locks onto
        a local rather than global optimum, more particles in that same run do not fix
        it; independent restarts are a standard guard against that failure mode for
        multimodal problems (e.g. multiple sources, where swapping source labels gives
        equivalent solutions). Ignored for scipy methods. Default 1 (a single run).
    n_jobs :
        PSO only, with n_runs > 1. Number of restarts to run in parallel via
        `concurrent.futures.ProcessPoolExecutor`; each restart is fully independent.
        Default 1 (sequential).
    pop :
        PSO only. Number of particles per run. Default 100.
    weighting :
        Error convention; see `VisibilityFitData`.
    bounds :
        Optional numeric lower/upper bounds in canonical units, shape (2, n_params).
        Reused unchanged during Monte Carlo refitting.

    """
    if not sources:
        raise ValueError("At least one source is required.")
    data = VisibilityFitData(visobs, weighting=weighting)
    # The observation chooses the numerical amplitude scale, not a fixed flux unit.
    sources = sources.to_canonical(data.flux_unit)
    fit_bounds = np.asarray(sources.bounds if bounds is None else bounds, dtype=float).copy()
    if fit_bounds.shape != (2, len(sources.params)) or np.any(np.isnan(fit_bounds)):
        raise ValueError("bounds must have shape (2, n_params) and contain no NaNs.")
    if np.any(fit_bounds[0] >= fit_bounds[1]):
        raise ValueError("Every lower bound must be strictly below its upper bound.")
    if method.casefold() == "pso":
        if n_runs < 1 or n_jobs < 1 or pop < 1:
            raise ValueError("n_runs, n_jobs, and pop must be positive.")
        results: list[Result]
        if n_runs == 1 or n_jobs == 1:
            results = [_run_one_pso(visobs, sources, seed, pop, weighting, fit_bounds) for seed in range(n_runs)]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Executor.map's typeshed overloads only cover up to 5 positional iterables;
                # with 6 here it falls back to an untyped Iterator[Any], hence the annotation above.
                results = list(
                    executor.map(
                        _run_one_pso,
                        [visobs] * n_runs,
                        [sources] * n_runs,
                        range(n_runs),
                        [pop] * n_runs,
                        [weighting] * n_runs,
                        [fit_bounds] * n_runs,
                    )
                )
        # pymoo's Result has no type stubs
        res = min(results, key=lambda r: r.F[0])  # pyright: ignore[reportUnknownLambdaType]
        sources_fit = sources.from_params(sources, res.X)
    else:

        def objective(x: NDArray[np.float64]) -> float | NDArray[np.float64]:
            cur_sources = sources.from_params(sources, x)
            vispred = sources_to_vis(cur_sources, data.u, data.v)
            return data.objective(vispred)

        res = minimize(
            objective,
            sources.params,
            method=method,
            bounds=[(x, y) for x, y in zip(*fit_bounds)],
        )
        sources_fit = sources.from_params(sources, res.x)
    prediction = sources_to_vis(sources_fit, data.u, data.v)
    res.fit_statistics = data.statistics(prediction, sum(source.n_params for source in sources))
    res.fit_bounds = fit_bounds
    res.parameter_names = sources.parameter_names
    res.parameter_units = sources.parameter_units(data.flux_unit)
    res.parameter_uncertainty = None
    return sources_fit, res


@quantity_input()  # type: ignore[untyped-decorator]
def vis_forward_fit(
    vis: Visibilities,
    sources: SourceList,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    map: bool | None = True,  # noqa: A002
    method: str | None = "Nelder-Mead",
    n_runs: int = 1,
    n_jobs: int = 1,
    pop: int = 100,
    *,
    weighting: str = "auto",
    return_result: bool = False,
    uncertainty: str | None = None,
    uncertainty_options: dict[str, Any] | None = None,
) -> GenericMap | tuple[Quantity | NDArray[np.float64], SourceList] | tuple[Any, SourceList, Any]:
    r"""
    Visibility forward fit method.

    Parameters
    ----------
    vis :
        Input visibilities
    sources :
        List of sources and their initial parameters
    shape :
        Shape of the image create
    pixel_size :
        Pixel size
    map :
        If `True` (default) return a `~sunpy.map.GenericMap` of the fitted image. If `False`
        return a ``(image, sources_out)`` tuple instead, where ``sources_out`` is the fitted
        `SourceList`.
    method :
        Method to use any of those supported methods by `scipy.optimize.minimize` or 'PSO' for particle swarm optimization
    n_runs :
        ``method='PSO'`` only. Number of independent PSO restarts to run, keeping the
        lowest-chi2 result - a standard guard against a single run's shared global-best
        converging to a local rather than global optimum. Default 1.
    n_jobs :
        ``method='PSO'`` only, with ``n_runs`` > 1. Number of restarts to run in
        parallel. Default 1 (sequential).
    pop :
        ``method='PSO'`` only. Number of particles per run. Default 100.
    weighting :
        ``'auto'`` (default), ``'components'``, ``'amplitude'``, or ``'none'``.
        See `VisibilityFitData` for the uncertainty conventions. The objective
        is always an unreduced sum of squared residuals.
    return_result :
        If `True`, return ``(image_or_map, sources_out, result)``. The native
        optimiser result carries a `FitStatistics` object as ``fit_statistics``.
        Otherwise, preserve the normal map or ``(image, sources_out)`` return.
    uncertainty :
        Optional ``'monte_carlo'`` or ``'gauss_newton'`` parameter errors. Requires
        known observation errors and ``return_result=True``. The common estimate
        is attached as ``result.parameter_uncertainty``; by default it is `None`.
    uncertainty_options :
        Keyword options for `estimate_parameter_uncertainty`, e.g.
        ``{'n_samples': 200, 'seed': 42, 'n_jobs': 2}`` for Monte Carlo or
        ``{'relative_step': 1e-5}`` for Gauss-Newton. Refits keep the original
        fitting bounds, weighting convention, and optimiser method.
    """
    if uncertainty not in {None, "monte_carlo", "gauss_newton"}:
        raise ValueError("uncertainty must be None, 'monte_carlo', or 'gauss_newton'.")
    if uncertainty is None and uncertainty_options is not None:
        raise ValueError("uncertainty_options requires an uncertainty method.")
    if uncertainty is not None:
        if not return_result:
            raise ValueError("Set return_result=True to retrieve parameter uncertainties.")
        if VisibilityFitData(vis, weighting=weighting).sigma_ri is None:
            raise ValueError("Parameter uncertainty estimation requires known measurement standard deviations.")
    if method is None:
        method = "Nelder-Mead"
    sources_out, res = _vis_forward_fit_minimise(
        vis, sources, method=method, n_runs=n_runs, n_jobs=n_jobs, pop=pop, weighting=weighting
    )
    if uncertainty is not None:
        options: dict[str, Any] = {"n_runs": n_runs, "pop": pop, **(uncertainty_options or {})}
        if _fit_succeeded(res):
            res.parameter_uncertainty = estimate_parameter_uncertainty(
                vis,
                sources_out,
                method=uncertainty,
                fit_method=method,
                weighting=weighting,
                bounds=res.fit_bounds,
                **options,
            )
        else:
            res.parameter_uncertainty = ParameterUncertainty(
                uncertainty,
                res.parameter_names,
                res.parameter_units,
                options.get("confidence_level", 0.6826894921370859),
                status="fit_failed",
                diagnostics={"message": "The original optimisation did not return a successful finite feasible fit."},
            )
    # Optimisation amplitudes are numeric in the observation's unit. Restore that
    # unit rather than the possibly differently scaled initial-guess amplitude unit.
    for source in sources_out:
        source.amp = source.amp * vis.visibilities.unit
    image = sources_to_image(sources_out, shape, pixel_size)
    output_sources = SourceList([source.to_quantities() for source in sources_out])
    if map:
        header = generate_header(vis, shape=shape, pixel_size=pixel_size)
        image = Map((image, header))
    if return_result:
        return image, output_sources, res
    if map:
        return image
    return image, output_sources
