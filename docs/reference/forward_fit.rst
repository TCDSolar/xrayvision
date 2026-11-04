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
``None`` for non-positive degrees of freedom. A future Gaussian log likelihood
must use the unreduced chi-square and its noise normalization, without dividing
by degrees of freedom. Initial-guess-dependent optimisation bounds are not
implicitly Bayesian priors.

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

.. automodapi:: xrayvision.vis_forward_fit.uncertainty
