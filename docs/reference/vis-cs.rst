.. _vis-cs:

VIS-CS (``xrayvision.vis_cs``)
******************************

VIS-CS reconstructs an image as a sparse positive sum of Gaussian sources. The
implementation follows :cite:t:`Felix2017_vis_cs`.

Use it with a visibility object that includes amplitude uncertainties::

    import astropy.units as u
    from xrayvision.vis_cs import vis_cs

    vis_map = vis_cs(
        vis,
        shape=(65, 65) * u.pix,
        pixel_size=(1, 1) * u.arcsec / u.pix,
        total_flux=100 * vis.visibilities.unit,
        seed=42,
    )

The result is a SunPy Map (or a Quantity with ``map=False``) of surface
brightness, in visibility units per square arcsecond, centred on
``vis.phase_center``. Visibilities are taken as phased to ``vis.phase_center``, as
``image_to_vis`` produces and ``vis_to_image`` assumes (and as RHESSI and STIX provide
them), the phase centre only places the map. Shape, pixel size and phase center
follow NumPy ``(y, x)`` order. Multiply by pixel area before using
``dft_map``, which sums flux per pixel. A total-flux constraint is required; when
``total_flux`` is omitted, supply ``vis.meta['totflux']`` per visibility. With
``vis.meta['isc']`` labels, total flux is the equally weighted mean of detector
means, as in SSW. Invalid observations and nonpositive uncertainties raise an error.

The default dictionary has ``200000 // nvis`` words (at least two). Two generations
retain the strongest words, perturb them, and fill the remainder with new random
Gaussians. ``seed`` makes Python runs reproducible. ``n_components`` and
``iterations`` can be changed for experiments. Each generation uses eight
regularisation stages, ending at the requested ``sparseness`` (default 0.5).

Notes
=====

* Gaussian sigma limits of 1.5--60 arcsec, the paper describes an
  upper limit of 40 arcsec. Set ``max_sigma=40*u.arcsec`` for that limit.
* The port retains SSW's square-root-of-SIGAMP weighting, numerical regularisation
  scale, continuous in-view flux constraint and integration convention. Use the
  same numerical flux units when comparing runs; the SSW penalty is not invariant
  to a change of input flux units.
* The in-view integral uses shifted bounds beginning at the first pixel center
  and the opposite Gaussian cross-term sign to its renderer. These conventions
  are preserved for compatibility, including for rotated sources near the edge.
  The constraint is applied before pruning. Therefore the rendered, sampled image
  is not guaranteed to sum to the requested flux, especially at FOV boundaries.
* Refined words perturbed wholly outside the field of view have zero in-view
  volume and are dropped - SSW keeps them.
* When a single word remains active, the flux constraint holds it at the full
  in-view flux and SSW divides by its rounding-level distance from that limit.
  Whether SSW then moves it (breaking the flux constraint) or treats it as singular
  (aborting, with the numerical-issue option) depends on rounding; here it is held fixed.
* At most 200 words above the SSW threshold are rendered. Unlike SSW's indexing of
  the 201st word, dictionaries with fewer than 201 words are supported safely.
* NumPy and IDL random streams differ. Equal seeds across languages do not produce
  identical dictionaries. Python integrates more tightly and renders in float64;
  SSW uses QROMB with EPS=1e-5, a single-precision PI constant and float32 images.
  Integration uses each Gaussian's own width to resolve narrow peaks in large
  fields of view, while retaining vectorised quadrature for the default axis ratios.
* ``maxiter`` bounds solver sweeps per stage and warns on nonconvergence.

.. automodapi:: xrayvision.vis_cs
