"""Priors and sampler-independent problems for Bayesian visibility fitting."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

import astropy.units as apu
from astropy.units import Quantity, UnitBase

from xrayvision.vis_forward_fit.fit_data import VisibilityFitData
from xrayvision.vis_forward_fit.forward_fit import SOURCE_TO_VIS, sources_to_vis
from xrayvision.vis_forward_fit.sources import Circular, Elliptical, InvalidSourceParameters, Loop, SourceList
from xrayvision.visibility import Visibilities

if TYPE_CHECKING:
    from scipy.stats import rv_continuous
    from scipy.stats._distn_infrastructure import rv_continuous_frozen

    from astropy.units.typing import UnitLike

    type _ScalarDistribution = rv_continuous_frozen[rv_continuous, float | np.float64]

__all__ = [
    "BayesianFitResult",
    "BayesianVisibilityProblem",
    "NestedFitResult",
    "ParameterPrior",
    "sample_dynesty",
    "sample_emcee",
]


class ParameterPrior:
    """A scalar continuous prior distribution with an explicit parameter unit.

    ``distribution`` is a frozen ``scipy.stats`` distribution. Its
    numerical parameters describe values in ``unit``. Both ``logpdf`` and
    ``ppf`` take the unit of the fitter's numeric parameter, so changing flux
    units or supplying an angular prior in degrees preserves the same prior.

    Use these objects in a mapping keyed by ``SourceList.parameter_names``.
    Priors are explicitly supplied scientific choices; optimiser bounds are
    never used to create them automatically. Only independent scalar priors
    are represented here, not correlated or conditional distributions.
    """

    distribution: "_ScalarDistribution"
    unit: UnitBase

    def __init__(self, distribution: "_ScalarDistribution", unit: "UnitLike") -> None:
        self.distribution = distribution
        self.unit = apu.Unit(unit)
        if not all(callable(getattr(self.distribution, name, None)) for name in ("logpdf", "ppf", "support")):
            raise TypeError("A scalar continuous distribution with logpdf, ppf, and support methods is required.")
        lower, upper = self.distribution.support()
        if np.ndim(lower) != 0 or np.ndim(upper) != 0 or not float(lower) < float(upper):
            raise ValueError("Prior distribution must have valid, non-empty scalar support.")

    def logpdf(self, value: float, *, unit: "UnitLike") -> float:
        """Log density in the numeric parameter's unit, including its Jacobian.

        ``value`` is a bare number in the supplied ``unit``. Non-finite values
        have zero prior probability. Incompatible units raise a units error.
        """
        factor = float(apu.Unit(unit).to(self.unit))
        if not np.isfinite(value):
            return -np.inf
        density = float(self.distribution.logpdf(value * factor))
        if np.isnan(density):
            raise ValueError("Prior distribution returned a NaN log density.")
        return float(density + np.log(factor))

    def ppf(self, probability: float, *, unit: "UnitLike") -> float:
        """Inverse CDF as a bare numeric parameter in the supplied unit.

        Also know as the percent point or quantile function
        Accepts probabilities in [0, 1]. Endpoints may produce infinite values
        for unbounded distributions, following SciPy's convention.
        """
        factor = float(apu.Unit(unit).to(self.unit))
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Prior probability must be finite and between zero and one.")
        value = float(self.distribution.ppf(probability))
        if np.isnan(value):
            raise ValueError("Prior distribution returned a NaN quantile.")
        return value / factor


class BayesianVisibilityProblem:
    """Connect explicit independent priors to existing visibility source models.

    Parameters
    ----------
    vis : Visibilities
        Observations with known measurement standard deviations.
    sources : SourceList
        Source types and reference parameters. No optimiser bounds are applied.
    priors : mapping
        One `ParameterPrior` for every name in ``sources.parameter_names``.
        Insertion order does not affect the flat parameter ordering.
    weighting : str
        Error convention understood by `VisibilityFitData`; default ``'auto'``.

    All vector arguments are numeric in ``parameter_units``: arcseconds,
    radians, and the observation's flux unit. Methods operate on one parameter
    vector at a time. Likelihood and priors remain separate from any sampler.

    Negative fluxes, non-positive widths, reversed loop widths, unsupported
    loop opening angles, and non-finite predictions have zero likelihood.
    ``log_prior`` and ``prior_transform`` describe exactly the supplied product
    prior. Rejecting invalid geometry does not turn it into a normalised prior
    conditional on valid geometry; this distinction matters for future evidence
    calculations. Choose priors describing the intended model domain explicitly.
    """

    def __init__(
        self,
        vis: Visibilities,
        sources: SourceList,
        priors: Mapping[str, ParameterPrior],
        *,
        weighting: str = "auto",
    ) -> None:
        if not sources:
            raise ValueError("At least one source is required.")
        self.fit_data = VisibilityFitData(vis, weighting=weighting)
        if self.fit_data.sigma_ri is None:
            raise ValueError("Bayesian fitting requires known measurement standard deviations.")
        self.sources = sources.to_canonical(self.fit_data.flux_unit)
        for source in self.sources:
            if type(source).__name__ not in SOURCE_TO_VIS:
                raise ValueError(f"Unknown source type: {type(source).__name__}.")
        self.parameter_names = self.sources.parameter_names
        self.parameter_units = self.sources.parameter_units(self.fit_data.flux_unit)
        self.ndim = len(self.parameter_names)
        self.initial_params = np.asarray(self.sources.params, dtype=np.float64)
        missing = set(self.parameter_names) - set(priors)
        unknown = set(priors) - set(self.parameter_names)
        if missing or unknown:
            raise ValueError(
                f"Prior names must match parameters exactly; missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        self.priors = dict(priors)
        for name, unit in zip(self.parameter_names, self.parameter_units):
            prior = self.priors[name]
            if not isinstance(prior, ParameterPrior):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"Prior for {name} must be a ParameterPrior.")
            prior.unit.to(unit)

    def _parameter_vector(self, params: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        if isinstance(params, Quantity):
            raise TypeError("Pass a numeric vector in parameter_units, not a Quantity array.")
        if np.iscomplexobj(params):
            raise ValueError("Parameters must be real.")
        values = np.asarray(params, dtype=np.float64)
        if values.shape != (self.ndim,):
            raise ValueError(f"Expected a parameter vector of shape ({self.ndim},), got {values.shape}.")
        return values

    def _canonical_sources(self, params: Sequence[float] | NDArray[np.float64]) -> SourceList:
        values = self._parameter_vector(params)
        if not np.all(np.isfinite(values)):
            raise InvalidSourceParameters("Source parameters must be finite.")
        sources = SourceList.from_params(self.sources, values)
        offset = 0
        for source in sources:
            parameters = values[offset : offset + source.n_params]
            if parameters[0] < 0:
                raise InvalidSourceParameters("Source flux must be non-negative.")
            if isinstance(source, Circular) and parameters[3] <= 0:
                raise InvalidSourceParameters("Circular sigma must be positive.")
            if isinstance(source, (Elliptical, Loop)) and np.any(parameters[3:5] <= 0):
                raise InvalidSourceParameters("Source widths must be positive.")
            if isinstance(source, Loop) and parameters[3] > parameters[4]:
                raise InvalidSourceParameters("Loop sigma_min must not exceed sigma_max.")
            offset += source.n_params
        return sources

    def sources_from_params(self, params: Sequence[float] | NDArray[np.float64]) -> SourceList:
        """Reconstruct sources with quantities in the observation's flux unit.

        Validates finite values, fluxes, and widths. Opening-angle validation
        remains in the existing loop visibility function used by ``predict``.
        """
        sources = self._canonical_sources(params)
        for source in sources:
            source.amp = source.amp * self.fit_data.flux_unit
        return SourceList([source.to_quantities() for source in sources])

    def predict(self, params: Sequence[float] | NDArray[np.float64]) -> Quantity:
        """Predict complex visibilities; invalid source parameters raise an error."""
        sources = self._canonical_sources(params)
        prediction = sources_to_vis(sources, self.fit_data.u, self.fit_data.v)
        return prediction * self.fit_data.flux_unit

    def log_prior(self, params: Sequence[float] | NDArray[np.float64]) -> float:
        """Return the supplied product-prior log density, without fit bounds."""
        values = self._parameter_vector(params)
        if not np.all(np.isfinite(values)):
            return -np.inf
        densities = [
            self.priors[name].logpdf(value, unit=unit)
            for name, unit, value in zip(self.parameter_names, self.parameter_units, values)
        ]
        if any(density == -np.inf for density in densities):
            return -np.inf
        return float(np.sum(densities))

    def log_likelihood(self, params: Sequence[float] | NDArray[np.float64]) -> float:
        """Return the normalised likelihood, independently of the supplied priors."""
        try:
            prediction = self.predict(params)
        except InvalidSourceParameters:
            return -np.inf
        return float(self.fit_data.log_likelihood(prediction))

    def log_posterior(self, params: Sequence[float] | NDArray[np.float64]) -> float:
        """Return log prior plus log likelihood, skipping zero-prior proposals."""
        prior = self.log_prior(params)
        if prior == -np.inf:
            return -np.inf
        likelihood = self.log_likelihood(params)
        return -np.inf if likelihood == -np.inf else prior + likelihood

    def prior_transform(self, probabilities: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a unit-cube vector through the supplied independent prior CDFs.

        Does not clip, reorder, or reject draws to enforce source geometry,
        since doing so would change the declared prior distribution.
        """
        values = self._parameter_vector(probabilities)
        return np.asarray(
            [
                self.priors[name].ppf(probability, unit=unit)
                for name, unit, probability in zip(self.parameter_names, self.parameter_units, values)
            ],
            dtype=np.float64,
        )


@dataclass
class BayesianFitResult:
    """Production samples and diagnostics from Bayesian visibility fitting.

    ``chain`` has shape ``(n_steps, n_walkers, n_parameters)`` and contains no
    burn-in steps. ``log_prob`` has shape ``(n_steps, n_walkers)``. Numeric
    parameter ordering and units follow the supplied problem. Raw source
    labels and angles are retained; no source matching or unwrapping occurs.

    Autocorrelation status is ``'estimated'``, ``'chain_too_short'``, or
    ``'unavailable'``. Estimates retained for short chains are provisional.
    Even ``'estimated'`` does not establish convergence or exploration of all
    posterior modes. Walkers in one ensemble are not independent chains.
    """

    problem: BayesianVisibilityProblem
    chain: NDArray[np.float64]
    log_prob: NDArray[np.float64]
    acceptance_fraction: NDArray[np.float64]
    autocorrelation_time: NDArray[np.float64] | None
    autocorrelation_status: str
    diagnostics: dict[str, object]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter labels in the order used by ``chain``."""
        return self.problem.parameter_names

    @property
    def parameter_units(self) -> tuple[UnitBase, ...]:
        """Parameter units, including the observation's flux unit."""
        return self.problem.parameter_units

    @property
    def samples(self) -> NDArray[np.float64]:
        """A flat view of all production draws; rows are correlated samples."""
        return self.chain.reshape(-1, self.chain.shape[-1])

    def parameter_samples(self, name: str) -> Quantity:
        """Return a named sample column with its parameter unit."""
        try:
            column = self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown parameter: {name}.") from error
        return self.samples[:, column] * self.parameter_units[column]

    def credible_interval(self, name: str, *, level: float = 0.6826894921370859) -> Quantity:
        """Return a marginal equal-tailed credible interval, with units.

        Raw labels and angles are used. For angular wrapping or source-label
        ambiguity these marginal percentiles may not be useful summaries.
        """
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("Credible interval level must be finite and strictly between zero and one.")
        samples = self.parameter_samples(name)
        tail = (1 - level) / 2
        return np.quantile(samples.value, [tail, 1 - tail]) * samples.unit

    def sources_from_sample(self, step: int, walker: int) -> SourceList:
        """Reconstruct one actual production draw as quantity-valued sources."""
        return self.problem.sources_from_params(self.chain[step, walker])


def sample_emcee(
    problem: BayesianVisibilityProblem,
    initial_positions: NDArray[np.float64],
    *,
    n_steps: int = 1000,
    burn_in: int = 0,
    seed: int | None = None,
    progress: bool = False,
) -> BayesianFitResult:
    """Sample a visibility posterior with emcee's default stretch move.

    Requires the optional ``xrayvisim[bayes]`` extra. Initial positions must be
    a real numeric array of shape ``(n_walkers, problem.ndim)`` in the problem's
    parameter units. Every walker must have a finite log posterior. At least
    ``2 * ndim`` walkers and an independent initial ensemble are required.
    No optimiser bounds, automatic initialisation, or source relabelling are
    applied. Emcee checks the initial ensemble's numerical conditioning.

    ``burn_in`` steps are run and discarded before ``n_steps`` production
    steps. Acceptance fractions and autocorrelation diagnostics describe the
    production run. A seed reproduces the chain for the same initial positions
    and software versions, without changing NumPy's global random state.
    """
    try:
        import emcee
    except ModuleNotFoundError as error:
        if error.name != "emcee":
            raise
        raise ImportError("Emcee sampling requires the optional dependency: pip install 'xrayvisim[bayes]'.") from error

    n_steps, burn_in = index(n_steps), index(burn_in)
    if n_steps < 1 or burn_in < 0:
        raise ValueError("n_steps must be positive and burn_in must be non-negative.")
    if isinstance(initial_positions, Quantity):
        raise TypeError("Initial positions must be numeric in parameter_units, not a Quantity array.")
    if np.iscomplexobj(initial_positions):
        raise ValueError("Initial positions must be real.")
    positions = np.array(initial_positions, dtype=np.float64, copy=True)
    if positions.ndim != 2 or positions.shape[1] != problem.ndim:
        raise ValueError(f"Initial positions must have shape (n_walkers, {problem.ndim}).")
    n_walkers = positions.shape[0]
    if n_walkers < 2 * problem.ndim:
        raise ValueError("The stretch move requires at least twice as many walkers as parameters.")
    initial_log_prob = np.array([problem.log_posterior(position) for position in positions])
    if not np.all(np.isfinite(initial_log_prob)):
        raise ValueError("Each initial walker must have a finite log posterior.")

    sampler = emcee.EnsembleSampler(n_walkers, problem.ndim, problem.log_posterior)
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = emcee.State(positions, log_prob=initial_log_prob)
    if burn_in:
        state = sampler.run_mcmc(state, burn_in, progress=progress)
        sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=progress)

    tau: NDArray[np.float64] | None = None
    status = "unavailable"
    message = "At least two production steps are needed for autocorrelation estimation."
    if n_steps >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                tau = np.asarray(sampler.get_autocorr_time(), dtype=np.float64)
                status = "estimated"
                message = "Emcee's chain-length criterion is met; convergence is not established."
            except emcee.autocorr.AutocorrError as error:
                tau = np.asarray(error.tau, dtype=np.float64)
                status = "chain_too_short"
                message = str(error)
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            tau = None
            status = "unavailable"
            message = "Autocorrelation estimates are not finite and strictly positive."

    acceptance = np.asarray(sampler.acceptance_fraction, dtype=np.float64)
    return BayesianFitResult(
        problem=problem,
        chain=np.array(sampler.get_chain(), dtype=np.float64, copy=True),
        log_prob=np.array(sampler.get_log_prob(), dtype=np.float64, copy=True),
        acceptance_fraction=acceptance.copy(),
        autocorrelation_time=tau,
        autocorrelation_status=status,
        diagnostics={
            "n_steps": n_steps,
            "n_walkers": n_walkers,
            "burn_in": burn_in,
            "seed": seed,
            "autocorrelation_message": message,
            "zero_acceptance_walkers": np.flatnonzero(acceptance == 0).tolist(),
        },
    )


@dataclass
class NestedFitResult:
    """Weighted posterior samples and evidence from static nested sampling.

    Rows of ``samples`` are not equally weighted posterior draws. Use
    ``weights`` for summaries or pass them to a plotting function accepting
    weights. ``log_weights`` are dynesty's unnormalised log evidence weights.
    ``log_evidence_error`` is its estimated standard error on log evidence;
    it does not establish that every mode was explored.

    Evidence comparisons require the same observations, error convention,
    and numeric flux unit, with explicitly normalised model priors. The log
    Bayes factor favouring A over B is ``A.log_evidence - B.log_evidence``;
    posterior model odds also depend on prior model odds. Invalid geometry
    retains zero likelihood under the declared prior, as in the problem.
    """

    problem: BayesianVisibilityProblem
    samples: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    log_weights: NDArray[np.float64]
    log_evidence: float
    log_evidence_error: float
    diagnostics: dict[str, object]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter labels in sample-column order."""
        return self.problem.parameter_names

    @property
    def parameter_units(self) -> tuple[UnitBase, ...]:
        """Parameter units, including the observation's flux unit."""
        return self.problem.parameter_units

    @property
    def weights(self) -> NDArray[np.float64]:
        """Normalised posterior weights, summing to one."""
        return np.exp(self.log_weights - logsumexp(self.log_weights))

    def parameter_samples(self, name: str) -> Quantity:
        """Return a named weighted sample column with its parameter unit."""
        try:
            column = self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown parameter: {name}.") from error
        return self.samples[:, column] * self.parameter_units[column]

    def sources_from_sample(self, sample: int) -> SourceList:
        """Reconstruct one weighted sample as quantity-valued sources."""
        return self.problem.sources_from_params(self.samples[sample])


def sample_dynesty(
    problem: BayesianVisibilityProblem,
    *,
    nlive: int = 500,
    dlogz: float = 0.01,
    maxcall: int | None = None,
    sample: str = "auto",
    seed: int | None = None,
    progress: bool = False,
) -> NestedFitResult:
    """Estimate posterior weights and model evidence with static dynesty.

    Requires the optional ``xrayvisim[bayes]`` extra. Uses the existing
    normalised likelihood and independent prior transform, without optimiser
    bounds or initial positions. Uses dynesty's default bounding method with
    ``nlive`` live points. ``sample`` selects its proposal method, default
    ``'auto'``; ``'rwalk'`` can help with correlated parameter distributions.

    ``dlogz`` controls the estimated remaining contribution to log evidence;
    it is not a target error on log evidence. ``maxcall`` optionally limits
    likelihood calls (approximately, following dynesty). A run stopped at
    this limit may have unreliable evidence; dynesty's warnings are retained.
    Neither termination nor the reported error guarantees mode exploration.
    Repeat with different seeds and more live points before interpreting
    model comparisons. A seed leaves NumPy's global random state unchanged.
    """
    try:
        import dynesty
    except ModuleNotFoundError as error:
        if error.name != "dynesty":
            raise
        raise ImportError(
            "Dynesty sampling requires the optional dependency: pip install 'xrayvisim[bayes]'."
        ) from error

    nlive = index(nlive)
    if nlive <= 2 * problem.ndim:
        raise ValueError("nlive must be greater than twice the number of parameters.")
    if not np.isfinite(dlogz) or dlogz <= 0:
        raise ValueError("dlogz must be finite and positive.")
    if maxcall is not None:
        maxcall = index(maxcall)
        if maxcall < 1:
            raise ValueError("maxcall must be positive.")

    sampler = dynesty.NestedSampler(
        problem.log_likelihood,
        problem.prior_transform,
        problem.ndim,
        nlive=nlive,
        sample=sample,
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=progress)
    results = sampler.results
    return NestedFitResult(
        problem=problem,
        samples=np.array(results.samples, dtype=np.float64, copy=True),
        log_likelihood=np.array(results.logl, dtype=np.float64, copy=True),
        log_weights=np.array(results.logwt, dtype=np.float64, copy=True),
        log_evidence=float(results.logz[-1]),
        log_evidence_error=float(results.logzerr[-1]),
        diagnostics={
            "nlive": nlive,
            "dlogz": dlogz,
            "maxcall": maxcall,
            "sample": sample,
            "seed": seed,
            "niter": int(results.niter),
            "ncall": int(np.sum(results.ncall)),
            "efficiency": float(results.eff),
        },
    )
