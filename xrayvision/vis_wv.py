"""VIS_WV reconstruction using compressed sensing with a finite isotropic wavelet transform.

Based on :cite:t:`DuvalPoo2018_vis_wv` and SolarSoft implementation.

The image is found by FISTA deconvolution of the dirty map by the PSF, with an L1 penalty on the coefficients of a
finite isotropic transform built from the Meyer wavelet.
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from numpy.typing import NDArray

import astropy.units as apu
from astropy.units import Quantity

from sunpy.map import GenericMap, Map

from xrayvision.imaging import generate_header, validate_and_expand_kwarg, vis_psf_image, vis_to_image
from xrayvision.utils import get_logger
from xrayvision.visibility import Visibilities

__all__ = ["vis_wv"]

logger = get_logger(__name__, "DEBUG")


def _meyer_aux(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Evaluate the polynomial transition function used by the Meyer spectra.

    Parameters
    ----------
    x : numpy.ndarray
        Transition coordinates, with any array shape. Values outside the
        interval [0, 1] are mapped to the corresponding endpoint value.

    Returns
    -------
    numpy.ndarray
        Transition values with the same shape as ``x``. The result is zero
        below zero, one above one, and follows the seventh-degree polynomial
        on [0, 1].

    Notes
    -----
    The transition polynomial is

    .. math::

        \nu(x) = 35x^4 - 84x^5 + 70x^6 - 20x^7.

    See Also
    --------
    _meyer_scaling, _meyer_wavelet
    """
    y = 35 * x**4 - 84 * x**5 + 70 * x**6 - 20 * x**7
    return np.where(x > 1, 1.0, np.where(x >= 0, y, 0.0))


def _meyer_scaling(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Evaluate the real, even Meyer scaling spectrum.

    Parameters
    ----------
    omega : numpy.ndarray
        Frequency coordinates in the Meyer frequency convention, with any
        array shape. The response depends only on their absolute values.

    Returns
    -------
    numpy.ndarray
        Scaling-spectrum amplitudes with the same shape as ``omega``. The
        response is one for :math:`|\omega| \leq \pi/4`, tapers to zero at
        :math:`|\omega| = \pi/2`, and vanishes beyond that frequency.

    Notes
    -----
    The scaling spectrum is

    .. math::

        \widehat{\phi}(\omega) =
        \begin{cases}
            1, & |\omega| < \pi/4, \\
            \cos\!\left[\dfrac{\pi}{2}
                \nu\!\left(\dfrac{4|\omega|}{\pi} - 1\right)\right],
                & \pi/4 \leq |\omega| \leq \pi/2, \\
            0, & |\omega| > \pi/2.
        \end{cases}

    Here :math:`\nu` is the transition function implemented by `_meyer_aux`.

    See Also
    --------
    _meyer_aux, _meyer_wavelet, _fiwt_spectra
    """
    x = np.abs(omega)
    fall = np.cos(np.pi / 2 * _meyer_aux(4 * x / np.pi - 1))
    return np.where(x < np.pi / 4, 1.0, np.where(x <= np.pi / 2, fall, 0.0))


def _meyer_wavelet(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Evaluate the real, even Meyer mother-wavelet spectrum.

    Parameters
    ----------
    omega : numpy.ndarray
        Frequency coordinates in the Meyer frequency convention, with any
        array shape. The response depends only on their absolute values.

    Returns
    -------
    numpy.ndarray
        Wavelet-spectrum amplitudes with the same shape as ``omega``. The
        response vanishes for :math:`|\omega| \leq \pi/4` and
        :math:`|\omega| > \pi`, rising to one at :math:`|\omega| = \pi/2`
        before tapering back to zero.

    See Also
    --------
    _meyer_aux, _meyer_scaling, _fiwt_spectra
    """
    x = np.abs(omega)
    rise = np.sin(np.pi / 2 * _meyer_aux(4 * x / np.pi - 1))
    fall = np.cos(np.pi / 2 * _meyer_aux(2 * x / np.pi - 1))
    return np.where(x <= np.pi / 4, 0.0, np.where(x <= np.pi / 2, rise, np.where(x <= np.pi, fall, 0.0)))


def _fiwt_spectra(n: int, nscales: int, pixel_size: float) -> NDArray[np.float64]:
    r"""
    Construct the spectra for a finite isotropic Meyer wavelet transform.

    Parameters
    ----------
    n : int
        Odd side length of the square image, in pixels.
    nscales : int
        Positive number of wavelet scales. A separate scaling-function band
        is included in addition to these wavelet bands.
    pixel_size : float
        Positive width of a square pixel, in arcseconds per pixel.

    Returns
    -------
    numpy.ndarray
        Real spectra with shape ``(nscales + 1, n, n)``. The first plane is
        the scaling spectrum; the remaining planes are wavelet spectra at
        scales :math:`a_j = 2^{-j}` for ``j`` from zero to ``nscales - 1``.
        The frequency axes are in unshifted FFT order, with zero frequency
        at index ``(0, 0)``.

    Notes
    -----
    The spectra form a Parseval frame:

    .. math::

        \phi^2 + \sum_j \psi_j^2 = 1.

    The finest band completes the squared amplitudes of all preceding
    bands. This retains the frame property when small pixels extend the
    sampled frequencies beyond the support of the original SSW finest band.
    Applying ``ifftshift`` to the radial frequency grid allows the returned
    spectra to multiply unshifted FFTs directly.

    See Also
    --------
    _fiwt, _ifiwt
    """
    # Omega fixes the bands in arcsec^-1 independent of the pixel size
    omega = 2.0 ** (nscales - 1) / pixel_size
    xi = np.linspace(-omega, omega, n)
    dist = ifftshift(np.hypot(*np.meshgrid(xi, xi)))
    psi = np.stack([_meyer_scaling(dist)] + [_meyer_wavelet(2.0**-j * dist) for j in range(nscales)])
    # SSW lets the finest band fall to 0 above ~pi/4 arcsec^-1, losing the frame for pixels below ~0.9 arcsec,
    # keep it flat at 1 instead; this only changes frequencies past its peak
    psi[-1] = np.sqrt(np.clip(1 - (psi[:-1] ** 2).sum(axis=0), 0, None))
    return psi


def _fiwt(image: NDArray[np.float64], psi: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Compute the finite isotropic wavelet coefficients of a real image.

    Parameters
    ----------
    image : numpy.ndarray
        Real image with shape ``(n, n)``.
    psi : numpy.ndarray
        Real, even spectra with shape ``(nscales + 1, n, n)``, in unshifted
        FFT order, as returned by `_fiwt_spectra`.

    Returns
    -------
    numpy.ndarray
        Real coefficients with shape ``(nscales + 1, n, n)``. The first
        plane contains scaling coefficients and the remaining planes contain
        wavelet coefficients. Coefficients retain the numerical units of
        ``image``.

    Notes
    -----
    Each spectrum filters the image FFT independently. Its inverse FFT is
    real because the input image is real and the spectra are real and even.
    With spectra from `_fiwt_spectra`, the transform preserves the sum of
    squared amplitudes, and `_ifiwt` reconstructs the image from unmodified
    coefficients to floating-point precision.

    See Also
    --------
    _fiwt_spectra, _ifiwt
    """
    return ifft2(psi * fft2(image)).real


def _ifiwt(coeffs: NDArray[np.float64], psi: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Synthesize a real image from finite isotropic wavelet coefficients.

    Parameters
    ----------
    coeffs : numpy.ndarray
        Real scaling and wavelet coefficients with shape
        ``(nscales + 1, n, n)``, ordered as in `_fiwt`.
    psi : numpy.ndarray
        Real, even spectra with the same shape as ``coeffs``, in unshifted
        FFT order, as returned by `_fiwt_spectra`.

    Returns
    -------
    numpy.ndarray
        Reconstructed real image with shape ``(n, n)`` and the same numerical
        units as ``coeffs``.

    Notes
    -----
    The coefficient FFTs are multiplied by their corresponding spectra and
    summed over the band axis before taking the inverse FFT. For the
    Parseval frame returned by `_fiwt_spectra`, this is the adjoint of
    `_fiwt` and exactly reconstructs the image from unmodified coefficients
    up to floating-point precision.

    See Also
    --------
    _fiwt_spectra, _fiwt
    """
    return ifft2((psi * fft2(coeffs)).sum(axis=0)).real


@apu.quantity_input  # type: ignore[untyped-decorator]
def vis_wv(
    vis: Visibilities,
    shape: Quantity[apu.pix] = (129, 129) * apu.pix,
    pixel_size: Quantity[apu.arcsec / apu.pix] = (1, 1) * apu.arcsec / apu.pix,
    *,
    nscales: int = 3,
    lam: float | None = None,
    maxiter: int = 200,
    map: bool = True,  # noqa: A002
) -> GenericMap | Quantity:
    r"""
    Finite isotropic wavelet transform compressed sensing (VIS_WV) image reconstruction.

    Solves

    .. math::

        \min_f \| H \mathbf{F} f - v \|_2^2 + \lambda \| W f \|_1

    with FISTA in the image domain (as a deconvolution of the dirty map by the PSF), where :math:`W` is the finite
    isotropic wavelet transform built from the Meyer wavelet. The total flux is constrained to the real part of the
    visibility with the shortest baseline and the result is clipped to be positive. A port of ``vis_wv.pro`` in SSW.

    Parameters
    ----------
    vis :
        Input visibilities.
    shape :
        Image shape, must be square and odd.
    pixel_size :
        Pixel size, must be square.
    nscales :
        Number of wavelet scales.
    lam :
        Regularisation parameter, if `None` estimated from the first unregularised iteration (Miller criterion).
    maxiter :
        Maximum number of FISTA iterations.
    map :
        Return a sunpy map or bare array.

    Returns
    -------
    :
        Reconstructed image in units of the visibilities per arcsec squared.

    Notes
    -----
    ``vis_wv.pro`` centres the PSF one pixel off the FFT origin, so its images are shifted one pixel down in x and y;
    here the PSF is centred correctly. Its finest wavelet band also falls to zero above ~ :math:`\pi/4` arcsec\
    :sup:`-1`, so for pixels below ~0.9 arcsec the transform is no longer a frame; here that band is kept flat.
    Otherwise, the results agree, except when the PSF sum is close to zero. In that case, the PSF and dirty map
    share a normalisation by the PSF peak to avoid amplifying cancellation error in their sums.

    References
    ----------
    * :cite:t:`DuvalPoo2018_vis_wv`
    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    n, m = shape.to_value(apu.pix).astype(int)
    if n != m or n % 2 == 0:
        raise ValueError(f"shape must be square and odd, got {shape}.")
    if pixel_size[0] != pixel_size[1]:
        raise ValueError(f"pixel_size must be square, got {pixel_size}.")
    pixel = pixel_size[0].to_value(apu.arcsec / apu.pix)

    psi = _fiwt_spectra(n, nscales, pixel)
    dirty_map: NDArray[np.float64] = vis_to_image(vis, shape=shape, pixel_size=pixel_size).value
    psf: NDArray[np.float64] = vis_psf_image(vis, shape=shape, pixel_size=pixel_size).value

    # Flux constraint from the shortest baseline
    radius = np.hypot(vis.u, vis.v)
    dc = vis.visibilities[radius == radius.min()].real.max()
    unit = dc.unit
    dc = dc.value

    psf_sum = psf.sum()
    if abs(psf_sum) <= 1e-12 * np.abs(psf).sum():
        # Missing zero-frequency data can make the signed PSF sum vanish.
        # Use the same stable scale for the PSF and dirty map in that case.
        scale = np.max(np.abs(psf))
        psf = psf / scale
        dirty_map = dirty_map / scale
    else:
        psf = psf / psf_sum
        dirty_map *= dc / dirty_map.sum()
    psf_ft = fft2(ifftshift(psf))
    dirty_ft = fft2(dirty_map)
    lipschitz = 2 * np.max(np.abs(psf_ft) ** 2)

    autolam = lam is None
    lam = 0.0 if lam is None else lam
    x = dirty_map
    y = x
    t = 1.0
    old_total = 0.0
    for i in range(1, maxiter + 1):
        x_old, t_old = x, t

        # Gradient step then soft threshold the wavelet coefficients
        y = y - 2 / lipschitz * ifft2(np.conj(psf_ft) * (psf_ft * fft2(y) - dirty_ft)).real
        wy = _fiwt(y, psi)
        wy = np.sign(wy) * np.clip(np.abs(wy) - lam / lipschitz, 0, None)
        x = _ifiwt(wy, psi)
        x = x - (x.sum() - dc) / n**2

        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        y = x + (t_old - 1) / t * (x - x_old)

        residual = dirty_map - ifft2(psf_ft * fft2(x)).real
        likelihood = np.sum(residual**2)
        sparsity = np.abs(_fiwt(x, psi)).sum()

        if i == 1 and autolam:
            # Capped so lambda can't be large enough to give a null solution
            lam = min(likelihood / sparsity, np.abs(ifft2(np.conj(psf_ft) * dirty_ft)).max())

        total = likelihood + lam * sparsity
        if i % 25 == 0:
            logger.debug(f"Iter: {i}, Obj function: {total}, likelihood: {likelihood}")
        if i > 9 and old_total <= total:
            break
        old_total = total

    logger.debug(f"Lambda: {lam}")
    image = np.clip(x, 0, None) / pixel**2 * unit / apu.arcsec**2

    if map:
        header = generate_header(vis, shape=shape, pixel_size=pixel_size)
        header["bunit"] = image.unit.to_string("fits")
        return Map((image.value, header))
    return image
