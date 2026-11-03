.. vis_forward_fit:

Vis Forward Fit ('xrayvision.vis_forward_fit')
**********************************************

The ``vis_forward_fit`` submodule contains the visibility forward fitting methods

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
`Volpara et al. (2022) <https://doi.org/10.1051/0004-6361/202243907>`_.
The installed legacy PSO loop parameter-count discrepancy is not reproduced.

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

.. automodapi:: xrayvision.vis_forward_fit

.. automodapi:: xrayvision.vis_forward_fit.forward_fit
   :include-all-objects:

.. automodapi:: xrayvision.vis_forward_fit.sources
   :include-all-objects:

.. automodapi:: xrayvision.vis_forward_fit.fit_data
