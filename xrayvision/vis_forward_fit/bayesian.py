"""Unit-aware priors for Bayesian visibility fitting."""

from typing import TYPE_CHECKING

import numpy as np

import astropy.units as apu
from astropy.units import UnitBase
from astropy.units.typing import UnitLike

if TYPE_CHECKING:
    from scipy.stats import rv_continuous
    from scipy.stats._distn_infrastructure import rv_continuous_frozen

    type _ScalarDistribution = rv_continuous_frozen[rv_continuous, float | np.float64]

__all__ = ["ParameterPrior"]


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
