"""Priors and sampler-independent problems for Bayesian visibility fitting."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import astropy.units as apu
from astropy.units import Quantity, UnitBase
from astropy.units.typing import UnitLike

from xrayvision.vis_forward_fit.fit_data import VisibilityFitData
from xrayvision.vis_forward_fit.forward_fit import SOURCE_TO_VIS, sources_to_vis
from xrayvision.vis_forward_fit.sources import Circular, Elliptical, InvalidSourceParameters, Loop, SourceList
from xrayvision.visibility import Visibilities

if TYPE_CHECKING:
    from scipy.stats import rv_continuous
    from scipy.stats._distn_infrastructure import rv_continuous_frozen

    type _ScalarDistribution = rv_continuous_frozen[rv_continuous, float | np.float64]

__all__ = ["BayesianVisibilityProblem", "ParameterPrior"]


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

    def __init__(self, distribution: "_ScalarDistribution", unit: UnitLike) -> None:
        self.distribution = distribution
        self.unit = apu.Unit(unit)
        if not all(callable(getattr(self.distribution, name, None)) for name in ("logpdf", "ppf", "support")):
            raise TypeError("A scalar continuous distribution with logpdf, ppf, and support methods is required.")
        lower, upper = self.distribution.support()
        if np.ndim(lower) != 0 or np.ndim(upper) != 0 or not float(lower) < float(upper):
            raise ValueError("Prior distribution must have valid, non-empty scalar support.")

    def logpdf(self, value: float, *, unit: UnitLike) -> float:
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

    def ppf(self, probability: float, *, unit: UnitLike) -> float:
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
