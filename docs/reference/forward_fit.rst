.. vis_forward_fit:

Vis Forward Fit ('xrayvision.vis_forward_fit')
**********************************************

The ``vis_forward_fit`` submodule contains visibility forward fitting methods
for parametric source imaging. See :cite:t:`Aschwanden2002_forward_fit` for
RHESSI forward fitting and :cite:t:`Volpara2022_forward_fit` for STIX forward
fitting with particle swarm optimisation.

Units and reusable sources
==========================

Public source constructors require angular quantities for coordinates, sizes,
and orientations. Numerical evaluation uses arcseconds, radians, and inverse
arcseconds. Sources returned by a fit can be passed directly to
``sources_to_vis``, ``sources_to_image``, or another fit; their bounds and
``SourceList.from_params`` also accept the returned quantity-valued parameters.

There is no fixed flux unit. Each fit uses the unit of the observed complex
visibilities. Source amplitudes expressed as quantities are converted to that
unit before initial values and bounds are calculated. Bare amplitudes are
interpreted in the observation unit, and incompatible quantities raise an
Astropy units error. Fitted amplitudes are returned in the observation unit.
For example, differential photon flux in photons per second, square centimetre,
and keV is supported just as counts per second or another compatible visibility
unit is. Converting a source amplitude to a unit expressed per eV changes its
numeric value appropriately; fitting does not simply remove the unit label.

Outside a fit, source predictions retain an explicit amplitude unit when one
is supplied. A mixed source list uses its first explicit amplitude unit for
the sum; bare amplitudes in that list are interpreted in the same unit.

Errors and fit statistics
=========================

Both SciPy and PSO use the same prepared observations and stack residuals in
the order ``[real residuals, imaginary residuals]``. ``weighting`` selects the
error convention:

* ``"auto"`` (default) uses ``Visibilities.uncertainty`` if present, otherwise
  ``amplitude_uncertainty``, otherwise an unweighted fit.
* ``"components"`` requires ``uncertainty``. For complex uncertainty values,
  the real and imaginary parts specify the respective component standard
  deviations. A real uncertainty supplies the same standard deviation to both.
* ``"amplitude"`` uses the amplitude standard deviation for both components,
  matching the IDL ``sigamp`` weighting convention. Select this explicitly
  when reproducing that convention for data with component uncertainties.
* ``"none"`` always uses unweighted residuals.

Standard deviations must be finite and strictly positive. The present error
model assumes independent real and imaginary components and independent
baselines; it does not include covariance. An amplitude standard deviation is
not generally interchangeable with two different component standard deviations.

Optimisation always minimises the unreduced sum of squared residuals. When
errors are supplied this is chi-square. The reduced statistic is reported
separately:

.. math::

   \chi^2 = \sum_{l=1}^{N_v}
       \left[\frac{\operatorname{Re}(V_l-V_l^{\mathrm{model}})^2}{\sigma_{R,l}^2}
       + \frac{\operatorname{Im}(V_l-V_l^{\mathrm{model}})^2}{\sigma_{I,l}^2}\right],
   \qquad
   \chi^2_{\mathrm{red}} = \frac{\chi^2}{2N_v-N_{\mathrm{free}}}.

Here ``N_v`` counts independent complex visibilities and ``N_free`` counts
freely fitted real parameters. All parameters of the current source models
are free: a circle has four, an ellipse six, and a loop seven. Conjugate copies
of measured visibilities do not supply independent observations and should not
be added to fitting input. The denominator follows the stacked real/imaginary
data count used by IDL, rather than ``N_v - N_free`` printed in Equation 4 of
:cite:t:`Volpara2022_forward_fit`.
The IDL PSO loop parameter-count discrepancy is not reproduced.

Set ``return_result=True`` to access statistics and the native optimiser result:

.. code-block:: python

   from xrayvision.vis_forward_fit.forward_fit import vis_forward_fit, fit_statistics

   image, fitted_sources, result = vis_forward_fit(
       vis, initial_sources, shape, pixel_size,
       map=False, weighting="auto", return_result=True,
   )
   stats = result.fit_statistics
   print(stats.objective, stats.chi2, stats.dof, stats.reduced_chi2)

   # Re-evaluate diagnostics later, using the same weighting convention.
   stats = fit_statistics(vis, fitted_sources, weighting="auto")

With ``return_result=True`` the return is always
``(image_or_map, fitted_sources, result)``. Without it the existing return
convention is preserved. ``chi2`` and ``reduced_chi2`` are ``None`` for
unweighted fits, which still report ``objective``. ``reduced_chi2`` is also
``None`` for non-positive degrees of freedom. Initial-guess-dependent
optimisation bounds are not implicitly Bayesian priors.

``VisibilityFitData.log_likelihood(prediction)`` evaluates the normalised
Gaussian log likelihood using the same independent component errors:

.. math::

   \log L = -\frac{1}{2}\left[\chi^2
       + 2\sum_{j=1}^{2N_v}\log\sigma_j + 2N_v\log(2\pi)\right].

It requires known measurement standard deviations and rejects unweighted data;
the existing optimiser objective is unchanged. Predictions may be quantities
or numeric arrays, including batches. Non-finite predictions receive negative
infinity, while invalid shapes and incompatible units raise exceptions.

The likelihood is a density in the numerical real and imaginary visibility
values expressed in the observation's flux unit. Changing that unit changes
the absolute log density by the unit-conversion Jacobian. Model comparisons
must use the same observation unit and noise convention for every model.

Explicit prior specifications
=============================

``ParameterPrior`` wraps a scalar continuous SciPy distribution with an
explicit unit. Distribution parameters such as ``loc`` and ``scale`` are
numerical values in that unit. Priors can be named using the keys returned by
``SourceList.parameter_names``:

.. code-block:: python

   import astropy.units as u
   from scipy.stats import norm, uniform
   from xrayvision.vis_forward_fit.bayesian import ParameterPrior

   photon_flux_unit = u.ph / (u.s * u.cm**2 * u.keV)
   priors = {
       "source[0].amp": ParameterPrior(uniform(loc=1, scale=3), photon_flux_unit),
       "source[0].x0": ParameterPrior(norm(loc=0, scale=0.1), u.arcmin),
   }
   log_density = priors["source[0].x0"].logpdf(2, unit=u.arcsec)
   position = priors["source[0].x0"].ppf(0.75, unit=u.arcsec)

Here the source amplitude is differential photon flux integrated over the
source's angular extent. Surface brightness would additionally be per unit
angular area, for example photons per second, square centimetre, keV, and
square arcsecond. No particular flux unit is imposed by ``ParameterPrior``.

``logpdf`` includes the Jacobian for conversion to the numeric parameter's
unit, and ``ppf`` returns values in that unit. These are prior distributions,
separate from optimisation bounds and measurement errors. SciPy supplies the
distributions; a posterior sampler consumes their log densities together with
the visibility log likelihood. Correlated and conditional priors are not
represented by this scalar wrapper.

Sampler-independent Bayesian problem
-----------------------------------

``BayesianVisibilityProblem`` connects observations, an existing ``SourceList``,
and a complete set of named priors. It requires measurement errors, validates
prior names and units, and uses ``sources_to_vis`` for predictions. For a single
circular source and observations ``vis`` in the photon flux unit above:

.. code-block:: python

   from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem
   from xrayvision.vis_forward_fit.sources import Source, SourceList

   sources = SourceList([
       Source("circular", 2 * photon_flux_unit, 0 * u.arcsec, 0 * u.arcsec, 2 * u.arcsec),
   ])
   priors.update({
       "source[0].y0": ParameterPrior(norm(loc=0, scale=0.1), u.arcmin),
       "source[0].sigma": ParameterPrior(uniform(loc=0.5, scale=4.5), u.arcsec),
   })
   problem = BayesianVisibilityProblem(vis, sources, priors)
   params = problem.initial_params
   log_posterior = problem.log_posterior(params)
   prediction = problem.predict(params)
   reconstructed = problem.sources_from_params(params)

Parameter vectors are numerical values ordered by ``parameter_names``, in
``parameter_units``. Reconstructed sources carry quantities, including the
observation's amplitude unit. The problem supports circles, ellipses, loops,
and mixed source lists. It does not impose initial-guess-dependent optimiser
bounds, including their position and orientation windows.

``log_prior`` evaluates exactly the supplied independent priors.
``log_likelihood`` evaluates the normalised measurement model independently of
those priors. ``log_posterior`` adds the two and skips prediction when prior
probability is zero. ``prior_transform`` maps a vector of probabilities through
the same independent inverse CDFs, without changing their distribution.

Invalid fluxes, widths, loop ordering, or loop opening angles receive negative
infinity from the likelihood. Direct predictions raise
``InvalidSourceParameters``, a ``ValueError`` subclass; unrelated implementation
errors are not suppressed. The existing loop geometry, including its minimum
component width and zero-angle approximation, is retained.

Geometry rejection does not normalise a prior conditional on valid geometry.
For future evidence calculations, the prior must explicitly represent the
intended model domain; it is not enough to discard invalid draws and assume
that the remaining distribution is normalised. Source labels and orientations
are retained as supplied, without bootstrap-style relabelling.

Emcee posterior sampling
------------------------

Install the optional sampler with ``pip install 'xrayvisim[bayes]'``.
``sample_emcee`` takes an existing ``BayesianVisibilityProblem`` and an initial
ensemble of numeric parameter vectors. Positions have shape
``(n_walkers, problem.ndim)`` and use ``problem.parameter_units``. Each walker
must have a finite posterior density, and the default stretch move requires
at least twice as many walkers as parameters. Emcee also checks the numerical
independence of the initial ensemble.

For the single circular-source example above, the columns are flux, x, y,
and sigma. Choose an initial spread appropriate to those units:

.. code-block:: python

   import numpy as np
   from xrayvision.vis_forward_fit.bayesian import sample_emcee

   rng = np.random.default_rng(42)
   spread = np.array([0.05, 0.2, 0.2, 0.1])
   positions = problem.initial_params + rng.normal(size=(32, problem.ndim)) * spread
   posterior = sample_emcee(
       problem, positions, n_steps=2000, burn_in=500, seed=42,
   )
   flux_samples = posterior.parameter_samples("source[0].amp")
   x_interval = posterior.credible_interval("source[0].x0", level=0.95)
   draw_sources = posterior.sources_from_sample(step=-1, walker=0)
   print(posterior.autocorrelation_status, posterior.diagnostics)

``BayesianFitResult.chain`` contains production draws only, with shape
``(n_steps, n_walkers, n_parameters)``; ``samples`` is a flattened view of
these draws. ``log_prob`` stores the matching log posterior values.
Acceptance fractions describe production sampling after the burn-in reset.
A seed reproduces the chain for the same starting positions and software
versions, without resetting NumPy's global random state.

``parameter_samples`` returns quantity-valued columns, and
``credible_interval`` returns marginal equal-tailed intervals. Reconstruct
individual draws to obtain source lists for visibility or image predictions.
No image is generated automatically. Source labels and angles remain as
sampled, so label switching and angular wrapping require consideration before
interpreting marginal intervals. These are Bayesian credible intervals,
separate from the frequentist parameter-error API below.

Autocorrelation status is ``"estimated"``, ``"chain_too_short"``, or
``"unavailable"``. Short-chain estimates are retained as provisional values;
invalid estimates are returned as ``None``. The ``"estimated"`` status means
that emcee's chain-length criterion is met, not that convergence or exploration
of every mode has been demonstrated. Inspect traces, acceptance fractions, and
the diagnostic message, then choose burn-in and production lengths for the
problem. Walkers within one ensemble are dependent and must not be treated as
independent chains for convergence statistics. See
`emcee's autocorrelation guide <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_.

Parameter uncertainties
=======================

Parameter errors can be requested for either SciPy or PSO through the same
optional interface. Both estimators use the prepared measurement errors and
require ``return_result=True``. Fits without an uncertainty request keep their
existing return convention and have ``result.parameter_uncertainty = None``.

.. code-block:: python

   image, fitted_sources, result = vis_forward_fit(
       vis, initial_sources, shape, pixel_size,
       method="PSO", map=False, return_result=True,
       uncertainty="monte_carlo",
       uncertainty_options={"n_samples": 200, "seed": 42, "n_jobs": 2},
   )
   errors = result.parameter_uncertainty
   print(errors.status, errors.diagnostics)
   print(errors.stderr["source[0].x0"])       # Standard error, in arcseconds
   print(errors.stderr["source[0].amp"])      # In the observation's flux unit
   print(errors.intervals["source[0].x0"])    # Lower/upper interval endpoints
   cross_covariance = errors.covariance_between("source[0].amp", "source[0].x0")

``ParameterUncertainty`` records the method, parameter names, units, standard
errors, interval endpoints, covariance, and estimator diagnostics. Its default
central interval level is approximately 68.27 percent. ``confidence_level``
can be supplied in ``uncertainty_options``. These estimates are frequentist;
the samples are not draws from a Bayesian posterior.

Monte Carlo refitting
---------------------

``uncertainty="monte_carlo"`` performs a parametric bootstrap. Independent
Gaussian noise is added to the real and imaginary components of the **fitted
model**, using the standard deviations selected by ``weighting``. Each synthetic
dataset is refitted with the same optimiser, weighting, and original numeric
bounds. Bounds are not re-estimated around each noisy result. This differs from
the legacy IDL procedure that adds noise to the observed visibilities.

The sample standard deviation supplies ``stderr`` and empirical central
percentiles supply ``intervals``. ``n_samples`` is the number of noisy datasets,
not the number of PSO particles or restarts. PSO's ``n_runs`` and ``pop`` settings
apply to each refit and may be overridden in ``uncertainty_options``. Setting
``n_jobs`` there parallelises noisy refits; restarts within those refits remain
sequential to avoid nested process pools. A seed gives the same synthetic data
and fitted samples in serial and parallel execution. Optimiser restarts use
their existing deterministic seed sequence so noise generation is reproducible.

Equivalent source types are matched to the fitted reference before summaries
are calculated. Ellipse axis exchanges and periodic orientations are aligned
as well. Angular samples and interval endpoints are unwrapped around the fitted
reference, and may extend beyond a conventional angular range. This alignment
does not prove that a multimodal source configuration is uniquely identified.

``diagnostics`` records requested, successful, and failed refits, failure
messages, and bound-hit counts. Failed refits are excluded from summaries but
are never hidden: ``status="partial"`` signals their presence. Fewer than two
successful refits produces ``status="failed"`` and no covariance or error
summaries. Bound hits are counted against the original parameter labels and
boxes before matching. Too few samples, failed fits, or frequent bound hits
limit how much can be inferred from the summary errors. Inspect these
diagnostics when assessing an estimate.

A successful SciPy refit must have a successful native convergence flag and
finite parameters and objective. For PSO, success here means a finite feasible
returned solution; it does not establish convergence to the global minimum.

By default, ``samples`` retains numeric fitted parameters with columns ordered
by ``parameter_names`` and expressed in ``parameter_units``. Set
``store_samples=False`` to retain only summaries and diagnostics. Covariance
is also numeric with this ordering: its (i, j) entry has the product of parameter
units i and j, which ``covariance_between`` restores.

Local Gauss-Newton approximation
--------------------------------

``uncertainty="gauss_newton"`` calculates the Jacobian of the same weighted
real/imaginary residuals at the best fit, independently of which optimiser was
used. It uses scaled central finite differences and an SVD to calculate

.. math::

   \operatorname{Cov}(\hat{\theta}) \simeq (J^{\mathsf{T}}J)^{-1}.

Measurement errors are treated as absolute; covariance is not multiplied by
reduced chi-square. Normal intervals centred on the fitted parameters use the
requested confidence level. ``relative_step`` controls the finite differences.
``rcond`` can override the relative singular-value cutoff; its default accounts
for both machine precision and finite-difference roundoff. This is a local
Gaussian approximation, which may be inaccurate for nonlinear or multimodal
fits even when the numerical Jacobian has full rank.

Rank-deficient directions produce ``status="rank_deficient"`` and no covariance
rather than misleading finite errors from a pseudoinverse. A fit on an active
bound or loop sigma-ordering constraint produces ``status="active_bounds"``
and no symmetric Gaussian estimate. ``"ill_conditioned"`` flags a large scaled
Jacobian condition number; ``"bounds_hit"`` flags intervals crossing fitting
bounds. Details are available in ``diagnostics``. A failed original optimisation
produces ``status="fit_failed"`` without attempting either estimator.

An existing fitted source list can be analysed without repeating the main fit:

.. code-block:: python

   from xrayvision.vis_forward_fit.uncertainty import estimate_parameter_uncertainty

   local_errors = estimate_parameter_uncertainty(
       vis, fitted_sources, method="gauss_newton",
       weighting="auto", bounds=result.fit_bounds,
   )

When calling this function directly, pass the original bounds and the same
weighting convention as the fit. Without explicit bounds the source defaults
are used, which may change the refitting domain and active-bound diagnostics.

PSO populations
---------------

The native PSO result retains ``result.pop``. Its canonical numeric columns
are described by ``result.parameter_names`` and ``result.parameter_units``:

.. code-block:: python

   population = result.pop.get("X")
   names = result.parameter_names
   units = result.parameter_units
   population_amplitudes = population[:, 0] * units[0]

The population remains useful for convergence diagnostics and inspecting
alternative solutions. It follows the optimiser's search dynamics, so its
spread is not a measurement-driven parameter error and it is not combined
with Monte Carlo samples or used as a posterior. ``pop`` controls the actual
number of particles through pymoo's ``pop_size`` option.

.. automodapi:: xrayvision.vis_forward_fit

.. automodapi:: xrayvision.vis_forward_fit.forward_fit
   :include-all-objects:

.. automodapi:: xrayvision.vis_forward_fit.sources
   :include-all-objects:

.. automodapi:: xrayvision.vis_forward_fit.fit_data

.. automodapi:: xrayvision.vis_forward_fit.bayesian

.. automodapi:: xrayvision.vis_forward_fit.uncertainty
