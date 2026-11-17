.. _vis-wv:

VIS-WV (``xrayvision.vis_wv``)
******************************

VIS-WV reconstructs an image by compressed sensing, using sparsity in a finite
isotropic wavelet transform built from the Meyer wavelet. The implementation
follows :cite:t:`DuvalPoo2018_vis_wv`.

Use it with any visibility object::

    import astropy.units as u
    from xrayvision.vis_wv import vis_wv

    wv_map = vis_wv(
        vis,
        shape=(65, 65) * u.pix,
        pixel_size=(1, 1) * u.arcsec / u.pix,
    )

The result is a SunPy Map (or a Quantity with ``map=False``) of surface
brightness, in visibility units per square arcsecond, centred on
``vis.phase_center``. Shape must be square with an odd number of pixels and
pixels must be square. Multiply by pixel area before using ``dft_map``, which
sums flux per pixel.

The dirty map is deconvolved by the PSF with FISTA in the image domain, soft
thresholding the wavelet coefficients each iteration. The total flux is held at the
real part of the visibility with the shortest baseline, and the result is clipped to
be positive. When ``lam`` is omitted the regularisation parameter is set after the
first iteration by the Miller criterion, capped so it cannot give a null solution.
Iteration stops after ``maxiter`` iterations (default 200), or once the objective
function increases after the first ten.

Notes
=====

* SSW centres the PSF with ``shift(P, 1-center)``, leaving its peak one pixel off
  the FFT origin, so SSW images are shifted one pixel down in x and y. Here the PSF
  is centred correctly.
* SSW's finest wavelet band falls to zero above ~\ :math:`\pi/4` arcsec\ :sup:`-1`,
  so for pixels smaller than ~0.9 arcsec the transform is no longer a Parseval
  frame. Here that band is kept flat beyond its peak, which only changes those
  frequencies.
* When the zero-frequency visibility is missing the PSF sum can vanish. In that
  case the PSF and dirty map share a normalisation by the PSF peak rather than
  their sums, to avoid amplifying cancellation error.
* Otherwise, the results match SSW to within 1e-4 of the image peak.

.. automodapi:: xrayvision.vis_wv
