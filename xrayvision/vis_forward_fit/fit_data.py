"""Shared observation preparation and statistics for complex visibility fits."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import astropy.units as apu
from astropy.units import Quantity

from xrayvision.visibility import Visibilities

__all__ = ["FitStatistics", "VisibilityFitData"]


@dataclass(frozen=True)
class FitStatistics:
    """Fit diagnostics, separate from the unreduced optimisation objective.

    ``chi2`` and ``reduced_chi2`` are ``None`` for unweighted fits. ``dof`` is
    ``2 * n_visibilities - n_parameters``; ``reduced_chi2`` is also ``None``
    when this is non-positive. Each complex visibility supplies two independent
    real measurements under the diagonal Gaussian error model used here.
    """

    objective: float
    n_visibilities: int
    n_parameters: int
    dof: int
    weighting: str
    chi2: float | None
    reduced_chi2: float | None


class VisibilityFitData:
    """Prepare one set of observations for any visibility fitting backend.

    Spatial frequencies are numeric inverse arcseconds. Visibility values and
    errors are numeric in ``flux_unit``, taken from the observations without
    imposing a particular physical flux unit. Bare predictions and source
    amplitudes used by a fitter are interpreted in this unit.

    ``weighting='auto'`` uses component uncertainties when ``vis.uncertainty``
    is present, otherwise amplitude uncertainties, otherwise no weighting.
    ``'components'`` uses the real/imaginary parts of complex uncertainties as
    separate standard deviations; real uncertainties apply equally to both.
    ``'amplitude'`` duplicates ``vis.amplitude_uncertainty`` for both components,
    matching the IDL sigamp convention. ``'none'`` requests unweighted fitting.
    Correlations between components or baselines are not modelled here.
    """

    def __init__(self, vis: Visibilities, *, weighting: str = "auto") -> None:
        if weighting not in {"auto", "components", "amplitude", "none"}:
            raise ValueError("weighting must be 'auto', 'components', 'amplitude', or 'none'.")
        self.flux_unit = vis.visibilities.unit
        self.u: NDArray[np.float64] = np.array(vis.u.to_value(apu.arcsec**-1), dtype=np.float64, copy=True)
        self.v: NDArray[np.float64] = np.array(vis.v.to_value(apu.arcsec**-1), dtype=np.float64, copy=True)
        self.values: NDArray[np.complex128] = np.array(
            vis.visibilities.to_value(self.flux_unit), dtype=np.complex128, copy=True
        )
        if self.values.ndim != 1 or self.values.size == 0:
            raise ValueError("Fitting requires a non-empty one-dimensional visibility array.")
        if self.u.shape != self.values.shape or self.v.shape != self.values.shape:
            raise ValueError("u, v, and visibility values must have matching shapes.")
        if not all(np.all(np.isfinite(array)) for array in (self.u, self.v, self.values)):
            raise ValueError("Visibility values and spatial frequencies must be finite.")

        uncertainty = vis.uncertainty
        if weighting == "auto":
            if uncertainty is not None:
                weighting = "components"
            elif vis.amplitude_uncertainty is not None:
                weighting = "amplitude"
            else:
                weighting = "none"
        self.weighting = weighting
        self.sigma_ri: NDArray[np.float64] | None = None
        if weighting == "components":
            if uncertainty is None:
                raise ValueError("Component weighting requires visibility uncertainties.")
            errors = uncertainty.to_value(self.flux_unit)
            sigma_real = np.real(errors)
            sigma_imag = np.imag(errors) if np.iscomplexobj(errors) else sigma_real
            self.sigma_ri = np.concatenate([sigma_real, sigma_imag])
        elif weighting == "amplitude":
            amplitude_uncertainty = vis.amplitude_uncertainty
            if amplitude_uncertainty is None:
                raise ValueError("Amplitude weighting requires amplitude uncertainties.")
            errors = amplitude_uncertainty.to_value(self.flux_unit)
            if np.iscomplexobj(errors):
                raise ValueError("Amplitude uncertainties must be real standard deviations.")
            self.sigma_ri = np.concatenate([errors, errors])
        if self.sigma_ri is not None:
            if not np.all(np.isfinite(self.sigma_ri)) or np.any(self.sigma_ri <= 0):
                raise ValueError("Uncertainty standard deviations must be finite and strictly positive.")
        self.values_ri = np.concatenate([self.values.real, self.values.imag])

    def residuals(self, prediction: Quantity | NDArray[np.complex128]) -> NDArray[np.float64]:
        """Return stacked real/imaginary residuals, optionally divided by errors.

        A leading population dimension is allowed, so scalar and batched
        optimisers use exactly the same residual definition.
        """
        if isinstance(prediction, Quantity):
            prediction = prediction.to_value(self.flux_unit)  # pyright: ignore[reportAttributeAccessIssue]
        prediction = np.asarray(prediction)
        if prediction.ndim == 0 or prediction.shape[-1] != self.values.size:
            raise ValueError("Prediction's final dimension must match the number of visibilities.")
        prediction_ri = np.concatenate([prediction.real, prediction.imag], axis=-1)
        residual = self.values_ri - prediction_ri
        if self.sigma_ri is not None:
            residual = residual / self.sigma_ri
        return np.asarray(residual, dtype=np.float64)

    def objective(self, prediction: Quantity | NDArray[np.complex128]) -> float | NDArray[np.float64]:
        """
        Return the unreduced sum of squared residuals for each prediction.
        """
        result = np.sum(self.residuals(prediction) ** 2, axis=-1)
        return float(result) if result.ndim == 0 else result

    def log_likelihood(self, prediction: Quantity | NDArray[np.complex128]) -> float | NDArray[np.float64]:
        r"""
        Return the normalised independent-component Gaussian log likelihood.

        Uses the same error convention and unreduced objective as the optimisers:
        ``-0.5 * (objective + 2 * sum(log(sigma_ri)) + 2 * N_v * log(2*pi))``.
        Known measurement standard deviations are required; unweighted fitting
        does not implicitly assume unit noise. Correlations are not included.

        The density is defined with respect to real/imaginary numerical values
        in ``flux_unit``. Converting the observation unit changes its absolute
        log density by the corresponding Jacobian, comparisons between models
        must use the same observation unit. Quantity predictions are converted
        to this unit before evaluation.

        Leading batch dimensions are preserved. Non-finite predictions or
        overflowing residual sums return negative infinity for the affected
        predictions. Shape and unit errors still raise exceptions.
        """
        if self.sigma_ri is None:
            raise ValueError("Log likelihood requires known measurement standard deviations.")
        normalisation = 2 * np.sum(np.log(self.sigma_ri)) + self.values_ri.size * np.log(2 * np.pi)
        with np.errstate(over="ignore", invalid="ignore"):
            result = -0.5 * (self.objective(prediction) + normalisation)
        result = np.where(np.isfinite(result), result, -np.inf)
        return float(result) if result.ndim == 0 else result

    def statistics(self, prediction: Quantity | NDArray[np.complex128], n_parameters: int) -> FitStatistics:
        """Report nominal degrees of freedom and, with errors, reduced chi-square.

        ``n_parameters`` must count only freely fitted parameters. The current
        fitters vary every source parameter. Reducing the objective is a reporting
        operation and must not be used to scale a Bayesian likelihood.
        """
        if n_parameters < 0:
            raise ValueError("n_parameters must be non-negative.")
        objective = self.objective(prediction)
        if isinstance(objective, np.ndarray):
            raise ValueError("Statistics require a single prediction, not a population.")
        objective = float(objective)
        dof = 2 * self.values.size - n_parameters
        chi2 = objective if self.sigma_ri is not None else None
        reduced_chi2 = chi2 / dof if chi2 is not None and dof > 0 else None
        return FitStatistics(objective, self.values.size, n_parameters, dof, self.weighting, chi2, reduced_chi2)
