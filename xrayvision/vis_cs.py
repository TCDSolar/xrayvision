"""VIS-CS reconstruction using a sparse, adaptive Gaussian dictionary.

Based on :cite:t:`Felix2017_vis_cs` and SolarSoft implementation.

Dictionary Gaussians ("words") have unit integrated flux and are positioned in arcsec
about the image centre, so pixel centres are the ``generate_xy`` coordinates.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad, quad_vec
from scipy.special import erf

import astropy.units as u

from sunpy.map import GenericMap, Map

from xrayvision.imaging import generate_header, validate_and_expand_kwarg
from xrayvision.transform import generate_xy
from xrayvision.visibility import Visibilities

__all__ = ["vis_cs"]

_Array = npt.NDArray[Any]

# Constants
_STAGES = 8  # regularisation stages, from sparseness 0.999 down to the requested value
_MAX_WORDS = 200  # strongest words kept from each generation
_MIN_WEIGHT = 1e-8
_MAX_FLUX_FRACTION = 0.999  # largest share of the total flux one word may take
_MOVE_TOLERANCE = 1e-3  # smallest update, as a fraction of a word's maximum, that counts as a move
_BATCH = 64  # candidate updates evaluated together
_INTEGRATION_LIMIT = 8.0  # Gaussian tails beyond this many scaled widths contribute less than 3e-29


@dataclass
class _Gaussians:
    r"""
    A dictionary of unit-flux elliptical Gaussians ("words").

    Each word :math:`k` has integrated flux one and surface brightness

    .. math::

        g_k(x, y) = \frac{1}{2\pi\sigma_x\sigma_y}
            \exp\left(-\frac{p^2}{2\sigma_x^2} - \frac{q^2}{2\sigma_y^2}\right),

    where :math:`(p, q)` are the offsets from the word's centre rotated by
    :math:`\phi` (see `principal`).

    Attributes
    ----------
    x, y : numpy.ndarray
        Centres in arcsec, relative to the image centre.
    sx, sy : numpy.ndarray
        Standard deviations along the word's principal axes, in arcsec.
    phi : numpy.ndarray
        Rotation angles of the principal axes, in radians.
    """

    x: _Array
    y: _Array
    sx: _Array
    sy: _Array
    phi: _Array

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: Any) -> "_Gaussians":
        return _Gaussians(*(getattr(self, f.name)[index] for f in fields(self)))

    @classmethod
    def concatenate(cls, parts: list["_Gaussians"]) -> "_Gaussians":
        """
        Join several dictionaries into one, preserving their order.

        Parameters
        ----------
        parts : list of `_Gaussians`
            Dictionaries to join.

        Returns
        -------
        `_Gaussians`
            The words of ``parts`` in sequence.
        """
        return cls(*(np.concatenate([getattr(part, f.name) for part in parts]) for f in fields(cls)))

    @property
    def amp(self) -> _Array:
        r"""Peak surface brightness :math:`1 / (2\pi\sigma_x\sigma_y)` of each unit-flux word."""
        return np.asarray(1 / (2 * np.pi * self.sx * self.sy))

    def principal(self, x: _Array, y: _Array) -> tuple[_Array, _Array]:
        r"""
        Rotate offsets (or frequencies) into each word's principal axes.

        Parameters
        ----------
        x, y : numpy.ndarray
            Offsets in arcsec, or frequencies in 1/arcsec, broadcastable against the
            words.

        Returns
        -------
        p, q : numpy.ndarray
            Components along the ``sx`` and ``sy`` axes,
            :math:`p = x\cos\phi - y\sin\phi` and :math:`q = x\sin\phi + y\cos\phi`.
        """
        cos, sin = np.cos(self.phi), np.sin(self.phi)
        return x * cos - y * sin, x * sin + y * cos


def _random_words(
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    pixel: _Array,
    min_sigma: float,
    max_sigma: float,
    anisotropy: float,
) -> _Gaussians:
    r"""
    Create random words.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    count : int
        Number of words to draw.
    shape : tuple of int
        Image shape ``(ny, nx)`` in pixels.
    pixel : numpy.ndarray
        Pixel size ``(dy, dx)`` in arcsec.
    min_sigma, max_sigma : float
        Limits of the standard deviations, in arcsec.
    anisotropy : float
        Largest allowed ratio between the two standard deviations of a word.

    Returns
    -------
    `_Gaussians`
        ``count`` words with centres uniform over the field of view.

    Notes
    -----
    With :math:`r` uniform on :math:`[0, 1)`, the major width is
    :math:`\sigma_x = \sigma_{\min} + r^2(\sigma_{\max} - \sigma_{\min})`, which favours
    compact words. The other width is :math:`\sigma_x(1 + 0.25 z)` for a standard normal
    :math:`z`, clipped to the anisotropy and sigma limits, and the angle is uniform on
    :math:`[0, \pi/2)`.

    See Also
    --------
    _refine
    """
    (ny, nx), (dy, dx) = shape, pixel
    x, y, size, phi = rng.random((4, count))
    sx = min_sigma + size**2 * (max_sigma - min_sigma)
    sy = np.clip(sx * (1 + 0.25 * rng.standard_normal(count)), sx / anisotropy, sx * anisotropy)
    return _Gaussians((x - 0.5) * nx * dx, (y - 0.5) * ny * dy, sx, np.clip(sy, min_sigma, max_sigma), phi * np.pi / 2)


def _refine(
    words: _Gaussians,
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    pixel: _Array,
    min_sigma: float,
    max_sigma: float,
    anisotropy: float,
) -> _Gaussians:
    r"""
    Build the next generation's dictionary from the strongest words.

    Parameters
    ----------
    words : `_Gaussians`
        Strongest words of the previous generation, kept unchanged.
    rng : numpy.random.Generator
        Source of randomness.
    count : int
        Size of the new dictionary, at least ``len(words)``.
    shape : tuple of int
        Image shape ``(ny, nx)`` in pixels.
    pixel : numpy.ndarray
        Pixel size ``(dy, dx)`` in arcsec.
    min_sigma, max_sigma : float
        Limits of the standard deviations, in arcsec.
    anisotropy : float
        Largest allowed ratio between the two standard deviations of a word.

    Returns
    -------
    `_Gaussians`
        ``count`` words: ``words``, then perturbed copies of them filling up to about
        half the dictionary, then new random words.

    Notes
    -----
    Each copy is shifted by two pixels times a standard normal in each direction, has
    each width scaled by ``1 + 0.4`` times a standard normal (then clipped to the
    anisotropy and sigma limits), and is rotated by :math:`\pi/8` times a standard normal.

    See Also
    --------
    _random_words
    """
    copies = max(0, (count // 2 - len(words)) // len(words))
    old = words[np.tile(np.arange(len(words)), copies)]
    shift_x, shift_y, grow_x, grow_y, turn = rng.standard_normal((5, len(old)))
    sx = np.clip(old.sx * (1 + 0.4 * grow_x), old.sy / anisotropy, old.sy * anisotropy)
    sy = np.clip(old.sy * (1 + 0.4 * grow_y), old.sx / anisotropy, old.sx * anisotropy)
    variations = _Gaussians(
        old.x + 2 * pixel[1] * shift_x,
        old.y + 2 * pixel[0] * shift_y,
        np.clip(sx, min_sigma, max_sigma),
        np.clip(sy, min_sigma, max_sigma),
        old.phi + np.pi / 8 * turn,
    )
    fill = _random_words(rng, count - len(words) - len(variations), shape, pixel, min_sigma, max_sigma, anisotropy)
    return _Gaussians.concatenate([words, variations, fill])


def _in_view_volume(words: _Gaussians, shape: tuple[int, int], pixel: _Array) -> _Array:
    r"""
    Integrate each unit-flux word over the field of view.

    Parameters
    ----------
    words : `_Gaussians`
        The dictionary.
    shape : tuple of int
        Image shape ``(ny, nx)`` in pixels.
    pixel : numpy.ndarray
        Pixel size ``(dy, dx)`` in arcsec.

    Returns
    -------
    numpy.ndarray
        In-view volume of each word: its flux inside the field of view, about one for
        words well inside it and zero for words wholly outside.

    Notes
    -----
    Writing the word's exponent as :math:`-(aX^2 + 2bXY + cY^2)`, the integral over
    :math:`X` is analytic (an error-function difference). The remaining integral is
    scaled by each word's marginal Y width, so narrow peaks cannot disappear between
    quadrature samples. Negligible Gaussian tails are excluded before integrating
    all words together with `scipy.integrate.quad_vec`. Extremely sharp conditional
    X transitions are integrated separately with explicit breakpoints.

    SSW integrates over one field-of-view width starting at the first pixel centre,
    not the pixel edge, and uses the opposite sign of :math:`b` to its renderer. Both
    conventions are kept so the flux constraint matches SSW.

    See Also
    --------
    _coordinate_descent
    """
    (ny, nx), (dy, dx) = shape, pixel
    x0 = -(nx - 1) / 2 * dx - words.x
    x1 = x0 + nx * dx
    y0 = -(ny - 1) / 2 * dy - words.y
    height = ny * dy
    sx, sy, phi = words.sx, words.sy, words.phi
    cos, sin = np.cos(phi), np.sin(phi)
    # Keep SSW's opposite cross-term sign to the rendered Gaussian.
    root_a = np.sqrt(cos**2 / (2 * sx**2) + sin**2 / (2 * sy**2))
    b = sin * cos * (1 / sy**2 - 1 / sx**2) / 2
    # t = Y / (sqrt(2) * marginal_sigma_y). This form avoids cancellation in c-b^2/a.
    y_scale = 1 / np.sqrt(2 * ((sx * sin) ** 2 + (sy * cos) ** 2))
    cutoff = _INTEGRATION_LIMIT
    lower = np.clip(y0 * y_scale, -cutoff, cutoff)
    upper = np.clip((y0 + height) * y_scale, -cutoff, cutoff)
    cross = b / (root_a * y_scale)
    left, right = root_a * x0, root_a * x1
    # Also bound the conditional X probability: its support can be much narrower
    # than the marginal Y Gaussian for highly elongated, rotated words.
    bound0 = np.divide(left - cutoff, cross, out=np.full_like(cross, -np.inf), where=cross != 0)
    bound1 = np.divide(right + cutoff, cross, out=np.full_like(cross, np.inf), where=cross != 0)
    lower = np.clip(np.maximum(lower, np.minimum(bound0, bound1)), -cutoff, cutoff)
    upper = np.clip(np.minimum(upper, np.maximum(bound0, bound1)), -cutoff, cutoff)
    width = np.maximum(upper - lower, 0)
    scale = width / (2 * np.sqrt(np.pi))

    def integrand(s: float) -> _Array:
        t = lower + s * width
        inner = erf(right - cross * t) - erf(left - cross * t)
        return np.asarray(scale * np.exp(-(t**2)) * inner)

    volume = np.asarray(quad_vec(integrand, 0, 1, epsabs=1e-10, epsrel=1e-7, norm="max")[0])
    # The default axis-ratio bound never takes this path. At extreme anisotropy,
    # an erf transition can otherwise hide in a very thin boundary layer.
    for i in np.flatnonzero(np.abs(cross) * width > 32):
        transitions = (np.array([left[i], right[i]])[:, None] + np.array([-cutoff, -1, 0, 1, cutoff])) / cross[i]
        points = (transitions.ravel() - lower[i]) / width[i]
        points = points[(points > 0) & (points < 1)]

        def scalar_integrand(s: float, i: int = int(i)) -> float:
            t = lower[i] + s * width[i]
            return float(scale[i] * np.exp(-(t**2)) * (erf(right[i] - cross[i] * t) - erf(left[i] - cross[i] * t)))

        volume[i] = quad(scalar_integrand, 0, 1, points=points, epsabs=1e-10, epsrel=1e-7)[0]
    return volume


def _fourier(words: _Gaussians, uu: _Array, vv: _Array) -> _Array:
    r"""
    Evaluate each word's visibilities analytically.

    Parameters
    ----------
    words : `_Gaussians`
        The dictionary.
    uu, vv : numpy.ndarray
        Spatial frequencies in 1/arcsec.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n_vis, n_words)``,

        .. math::

            V_{jk} = \exp\left(-2\pi^2(\sigma_x^2 p_{jk}^2 + \sigma_y^2 q_{jk}^2)
                + 2\pi i (u_j x_k + v_j y_k)\right),

        with :math:`(p, q)` the frequencies rotated into each word's principal axes.
        The positive sign and the image-centre origin match `~xrayvision.transform.dft_map`.
    """
    uu, vv = np.asarray(uu)[:, None], np.asarray(vv)[:, None]
    p, q = words.principal(uu, vv)
    decay = -2 * np.pi**2 * (words.sx**2 * p**2 + words.sy**2 * q**2)
    phase = 2 * np.pi * (uu * words.x + vv * words.y)
    return np.asarray(np.exp(decay + 1j * phase))


def _render(words: _Gaussians, weights: _Array, shape: tuple[int, int], pixel: _Array) -> _Array:
    """
    Sample the weighted sum of words at the pixel centres.

    Parameters
    ----------
    words : `_Gaussians`
        Words to render.
    weights : numpy.ndarray
        Integrated flux of each word.
    shape : tuple of int
        Image shape ``(ny, nx)`` in pixels.
    pixel : numpy.ndarray
        Pixel size ``(dy, dx)`` in arcsec.

    Returns
    -------
    numpy.ndarray
        Surface brightness (flux per square arcsec) of shape ``shape``, in NumPy
        ``(y, x)`` order, at the `~xrayvision.transform.generate_xy` pixel centres.
    """
    y, x = (generate_xy(n * u.pix, pixel_size=d * u.arcsec / u.pix).to_value(u.arcsec) for n, d in zip(shape, pixel))
    xx, yy = np.meshgrid(x, y)
    image = np.zeros(shape)
    for i, weight in enumerate(weights):
        word = words[i]
        p, q = word.principal(xx - word.x, yy - word.y)
        image += weight * word.amp * np.exp(-(p**2 / (2 * word.sx**2) + q**2 / (2 * word.sy**2)))
    return image


def _coordinate_descent(
    transform: _Array,
    observed: _Array,
    sigamp: _Array,
    amp: _Array,
    viv: _Array,
    total_flux: float,
    sparseness: float,
    regularisation: float,
    maxiter: int,
    abort_numerical_issues: bool,
) -> _Array:
    r"""
    Fit word weights by active-set coordinate descent.

    Parameters
    ----------
    transform : numpy.ndarray
        Complex visibilities of each unit-flux word, shape ``(n_vis, n_words)``.
    observed : numpy.ndarray
        Complex observed visibilities, shape ``(n_vis,)``.
    sigamp : numpy.ndarray
        Amplitude uncertainties, shape ``(n_vis,)``.
    amp : numpy.ndarray
        Peak surface brightness of each unit-flux word.
    viv : numpy.ndarray
        In-view volume of each word (see `_in_view_volume`), all positive.
    total_flux : float
        In-view flux the solution must carry.
    sparseness : float
        Final regularisation fraction in [0, 1).
    regularisation : float
        Penalty scale.
    maxiter : int
        Maximum sweeps per regularisation stage.
    abort_numerical_issues : bool
        Return zero weights on a singular or zero-curvature update in the final stage.

    Returns
    -------
    numpy.ndarray
        Integrated flux of each word -  mostly zeros.

    Raises
    ------
    ValueError
        If ``total_flux`` is too small for finite penalties.
    FloatingPointError
        If an update reaches a singular flux constraint and ``abort_numerical_issues``
        is false.

    Warns
    -----
    RuntimeWarning
        If a stage does not converge within ``maxiter`` sweeps.

    Notes
    -----
    The unknowns are peak brightnesses :math:`x_k = w_k A_k`. Each stage minimises

    .. math::

        \left\| \sum_k x_k b_k - t \right\|^2 + \lambda \sum_k x_k,
        \qquad \text{subject to} \quad \sum_k x_k / x_k^{\max} = 1,
        \quad x_k \geq 0,

    where :math:`b_k` and :math:`t` are the real and imaginary parts of the word and
    observed visibilities divided by :math:`\sqrt{\sigma}` (SSW weights by the square
    root of SIGAMP) and :math:`b_k` also by :math:`A_k`, and
    :math:`x_k^{\max} = F A_k / V_k` puts the whole in-view flux :math:`F` in word
    :math:`k`. A Newton step updates one word at a time while rescaling the others to
    keep the constraint, and no word may exceed 99.9% of its maximum.

    The eight stages use :math:`\lambda = \text{regularisation}\, e^{1/F} f/(1 - f)`
    for fractions :math:`f` from 0.999 down to ``sparseness``. Each stage alternates
    full sweeps with sweeps over the active words until a full sweep moves nothing.
    """
    sigma = np.sqrt(sigamp)  # SSW intentionally weights by sqrt(SIGAMP), not SIGAMP
    model = transform / sigma[:, None] / amp
    basis = np.ascontiguousarray(np.concatenate((model.real, model.imag)).T)  # (word, 2 * nvis)
    target = np.concatenate((np.real(observed), np.imag(observed))) / np.tile(sigma, 2)
    curvature = -2 * np.sum(basis**2, axis=1)
    x_max = total_flux * amp / viv
    x = np.zeros(len(amp))
    n_active = min(2, len(amp))
    # Share the full flux between up to two words; filtering may leave only one.
    if n_active:
        x[:n_active] = x_max[:n_active] / n_active
    residual = x @ basis - target
    xsum = x.sum()
    active = x > 0

    fractions = np.linspace(0.999, sparseness, _STAGES)
    with np.errstate(over="ignore", divide="ignore"):
        penalties = regularisation * np.exp(1 / np.float64(total_flux)) * fractions / (1 - fractions)
    if not np.all(np.isfinite(penalties)):
        raise ValueError("Total flux is too small for the regularisation scale.")

    def sweep(indices: _Array, penalty: float, abort: bool) -> bool | None:
        """Update ``indices`` in order, return whether any moved, or None to abort with zero weights.

        Most candidate updates are rejected and leave the state unchanged, so candidates are
        evaluated in batches up to the first accepted (or singular) one, which is applied
        before evaluating the words after it.
        """
        nonlocal x, residual, xsum, n_active
        moved, start = False, 0
        for _ in range(len(indices)):  # each pass advances start by at least one
            if start >= len(indices):
                break
            batch = indices[start : start + _BATCH]
            b = basis[batch]
            xi, limit = x[batch], x_max[batch]
            gap = limit - xi
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                alpha = 1 / gap
                epsilon = (residual - b * xi[:, None] + target) * alpha[:, None]
                divisor = curvature[batch] + np.einsum("ij,ij->i", epsilon, 4 * b - 2 * epsilon)
                gradient = 2 * (b - epsilon) @ residual + penalty * (1 - alpha * (xsum - xi))
                newx = np.clip(gradient / divisor + xi, 0, _MAX_FLUX_FRACTION * limit)
            singular = gap <= 0
            # The flux constraint holds a sole active word at its maximum, where gap is only
            # rounding: SSW then moves it by chance and breaks the constraint. Keep it fixed.
            pinned = (xi > 0) & (n_active < 2)
            solvable = ~singular & (divisor != 0) & np.isfinite(divisor) & np.isfinite(newx)
            newx = np.where(solvable, newx, 0.0)
            delta = newx - xi
            moves = ~singular & ~pinned & (((newx == 0) != (xi == 0)) | (np.abs(delta) > _MOVE_TOLERANCE * limit))
            stops = (singular | (divisor == 0) if abort else singular) & ~pinned
            events = np.flatnonzero(moves | stops)
            if not len(events):
                start += len(batch)
                continue
            k = events[0]
            if stops[k]:
                if abort:
                    return None
                raise FloatingPointError("VIS-CS reached a singular flux constraint.")
            i = batch[k]
            n_active += int(newx[k] > 0) - int(xi[k] > 0)
            active[i] = newx[k] > 0
            residual += delta[k] * (b[k] - epsilon[k])
            rescale = 1 - delta[k] * alpha[k]  # rescale the other words to keep the flux constraint
            x *= rescale
            xsum = rescale * xsum - x[i] + newx[k]
            x[i] = newx[k]
            moved, start = True, start + k + 1
        return moved

    for stage, penalty in enumerate(penalties):
        abort = abort_numerical_issues and stage == _STAGES - 1
        full_sweep = True
        for _ in range(maxiter):
            moved = sweep(np.arange(len(x)) if full_sweep else np.flatnonzero(active), penalty, abort)
            if moved is None:
                return np.zeros_like(x)
            if full_sweep and not moved:
                break
            full_sweep = not moved  # after a productive sweep, revisit only the active words
        else:
            warnings.warn("VIS-CS coordinate descent reached maxiter before convergence.", RuntimeWarning, stacklevel=3)
    return np.asarray(x / amp)


def _strongest(weights: _Array) -> _Array:
    """
    Select the words kept from a generation.

    Parameters
    ----------
    weights : numpy.ndarray
        Integrated flux of each word.

    Returns
    -------
    numpy.ndarray
        Indices, in dictionary order, of weights above ``1e-8`` and above the 201st
        largest weight, so at most 200 words.
    """
    limit = _MIN_WEIGHT
    if len(weights) > _MAX_WORDS:
        limit = max(limit, np.partition(weights, -_MAX_WORDS - 1)[-_MAX_WORDS - 1])
    return np.flatnonzero(weights > limit)


def _total_flux(vis: Visibilities, unit: u.UnitBase, nvis: int) -> float:
    """
    Estimate the total flux from per-visibility values in the metadata.

    Parameters
    ----------
    vis : `~xrayvision.visibility.Visibilities`
        Visibilities whose ``meta['totflux']`` holds one flux per visibility, with
        optional ``meta['isc']`` detector labels.
    unit : astropy.units.UnitBase
        Visibility unit, quantities in ``meta['totflux']`` are converted to it.
    nvis : int
        Number of visibilities.

    Returns
    -------
    float
        The mean flux of each detector, averaged equally over detectors, or the plain
        mean without detector labels.

    Raises
    ------
    ValueError
        If ``meta['totflux']`` is missing, has the wrong shape or holds nonpositive or
        non-finite values, or ``meta['isc']`` has the wrong shape.
    """
    meta = vis.meta
    if not isinstance(meta, Mapping) or "totflux" not in meta:
        raise ValueError("Provide total_flux or vis.meta['totflux'], vis-cs requires a total-flux constraint.")
    fluxes = meta["totflux"]
    fluxes = np.asarray(fluxes.to_value(unit) if isinstance(fluxes, u.Quantity) else fluxes)
    if fluxes.shape != (nvis,) or not np.all(np.isfinite(fluxes) & (fluxes > 0)):
        raise ValueError("meta['totflux'] must contain one positive finite flux per visibility.")
    if "isc" not in meta:
        return float(fluxes.mean())
    labels = np.asarray(meta["isc"])
    if labels.shape != (nvis,):
        raise ValueError("meta['isc'] must have one detector label per visibility.")
    return float(np.mean([fluxes[labels == label].mean() for label in np.unique(labels)]))


def vis_cs(
    vis: Visibilities,
    shape: u.Quantity = (65, 65) * u.pix,
    pixel_size: u.Quantity = (1, 1) * u.arcsec / u.pix,
    *,
    total_flux: u.Quantity | None = None,
    sparseness: float = 0.5,
    seed: int | np.random.Generator | None = None,
    n_components: int | None = None,
    iterations: int = 2,
    min_sigma: u.Quantity = 1.5 * u.arcsec,
    max_sigma: u.Quantity = 60 * u.arcsec,
    max_anisotropy: float = 2.5,
    regularisation: float = 1000.0,
    maxiter: int = 1000,
    abort_numerical_issues: bool = False,
    map: bool = True,  # noqa: A002
) -> u.Quantity | GenericMap:
    """Reconstruct visibilities with the a gaussian dictionary method.

    Parameters
    ----------
    vis : `~xrayvision.visibility.Visibilities`
        One-dimensional visibilities with finite positive amplitude uncertainties.
        UV coordinates must have inverse angular units.
    shape : `~astropy.units.Quantity`, optional
        Output shape in pixels, in NumPy ``(ny, nx)`` order.
    pixel_size : `~astropy.units.Quantity`, optional
        Scalar or ``(dy, dx)`` angular pixel sizes, matching ``dft_map``.
    total_flux : `~astropy.units.Quantity`, optional
        Positive total flux in visibility units. If omitted, use ``vis.meta['totflux']``
        (a Quantity or numerical array in visibility units), averaging within each
        ``vis.meta['isc']`` detector first, then equally across detectors. Without
        detector labels the arithmetic mean of the flux array is used.
    sparseness : float, optional
        Regularisation fraction in [0, 1). Larger values favor fewer Gaussians.
    seed : int or `~numpy.random.Generator`, optional
        Random state for repeatable dictionaries. NumPy and IDL random streams differ.
    n_components : int, optional
        Number of dictionary words=, defaults to ``max(2, 200000 // nvis)``.
    iterations : int, optional
        Dictionary generations, including the initial dictionary (default two).
    min_sigma, max_sigma : `~astropy.units.Quantity`, optional
        Gaussian sigma limits. Defaults 1.5 and 60 arcsec.
    max_anisotropy : float, optional
        Axis-ratio bound (default 2.5), one gives circular Gaussians.
    regularisation : float, optional
        Penalty scale (default 1000), applied to the numerical flux values in
        the input visibility unit. Use consistent flux units across comparisons.
    maxiter : int, optional
        Maximum full/active-set sweeps per each of eight regularisation stages.
        Emits a RuntimeWarning if a stage does not converge.
    abort_numerical_issues : bool, optional
        Return zero coefficients for a singular final-stage update.
    map : bool, optional
        Return a `sunpy.map.Map` (default) or a bare Quantity.

    Returns
    -------
    image : `~sunpy.map.GenericMap` or `~astropy.units.Quantity`
        Nonnegative surface brightness in visibility units per square arcsecond.
        Multiply by pixel area before using ``dft_map``. Visibilities are taken as
        phased to ``vis.phase_center``, as ``image_to_vis`` produces and
        ``vis_to_image`` assumes (and as RHESSI/STIX provide them), so the phase
        centre only sets the map centre. Uses the positive Fourier sign of ``dft_map``.

    Notes
    -----
    This follows the matches the convention of the vis-cs version in SSW, including
    sqrt(SIGAMP) weighting, eight sparsity stages, continuous in-view total-flux
    constraint, active-set descent, dictionary refinement and up-to-200-word pruning.
    The paper describes sigma up to 40 arcsec but SSW uses 60 arcsec. NumPy uses a
    different random generator and float64 rendering (SSW renders float32).
    Flux is constrained before pruning and continuous integration differs from
    summing pixel samples, so the returned pixel sum need not equal total_flux.

    References
    ----------
    * :cite:t:`Felix2017_vis_cs`
    """
    shape = validate_and_expand_kwarg(u.Quantity(shape), "shape")
    dims = shape.to_value(u.pix)
    if not (np.all(np.isfinite(dims)) and np.all(dims >= 1) and np.all(dims == np.round(dims))):
        raise ValueError("shape must contain one or two positive integer pixel counts.")
    image_shape = (int(dims[0]), int(dims[1]))
    pixel_size = validate_and_expand_kwarg(u.Quantity(pixel_size), "pixel_size")
    pixel = pixel_size.to_value(u.arcsec / u.pix)
    if not np.all(np.isfinite(pixel) & (pixel > 0)):
        raise ValueError("pixel_size must contain one or two positive finite values.")
    lo = u.Quantity(min_sigma).to_value(u.arcsec)
    hi = u.Quantity(max_sigma).to_value(u.arcsec)
    if np.ndim(lo) or np.ndim(hi) or not (0 < lo <= hi < np.inf):
        raise ValueError("Require finite 0 < min_sigma <= max_sigma.")
    if not np.isfinite(sparseness) or not 0 <= sparseness < 1:
        raise ValueError("sparseness must be in [0, 1).")
    if not np.isfinite(max_anisotropy) or max_anisotropy < 1:
        raise ValueError("max_anisotropy must be finite and at least one.")
    if not np.isfinite(regularisation) or regularisation <= 0:
        raise ValueError("regularisation must be finite and positive.")
    for name, value in (("iterations", iterations), ("maxiter", maxiter)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    unit = vis.visibilities.unit
    observed = np.asarray(vis.visibilities.to_value(unit), dtype=complex)
    uu, vv = vis.u.to_value(1 / u.arcsec), vis.v.to_value(1 / u.arcsec)
    if vis.amplitude_uncertainty is None:
        raise ValueError("VIS-CS requires amplitude uncertainties.")
    sigamp = np.asarray(vis.amplitude_uncertainty.to_value(unit))
    if observed.ndim != 1 or len(observed) < 2 or uu.shape != observed.shape or vv.shape != observed.shape:
        raise ValueError("VIS-CS requires at least two one-dimensional visibility samples.")
    if (
        sigamp.shape != observed.shape
        or np.iscomplexobj(sigamp)
        or np.any(sigamp <= 0)
        or not all(np.all(np.isfinite(t)) for t in (observed, uu, vv, sigamp))
    ):
        raise ValueError(
            "Visibilities and UV coordinates must be finite, with positive finite amplitude uncertainties."
        )
    flux = _total_flux(vis, unit, len(observed)) if total_flux is None else u.Quantity(total_flux).to_value(unit)
    if np.ndim(flux) or not np.isfinite(flux) or flux <= 0:
        raise ValueError("total_flux must be a positive finite scalar.")
    if n_components is None:
        n_components = max(2, 200000 // len(observed))
    if isinstance(n_components, bool) or not isinstance(n_components, Integral) or n_components < 2:
        raise ValueError("n_components must be an integer of at least two.")

    rng = np.random.default_rng(seed)
    words = _random_words(rng, n_components, image_shape, pixel, lo, hi, max_anisotropy)
    image = np.zeros(image_shape)
    for generation in range(iterations):
        viv = _in_view_volume(words, image_shape, pixel)
        # Perturbed copies can drift wholly out of view; they cannot carry in-view flux.
        in_view = np.isfinite(viv) & (viv > 0)
        words, viv = words[in_view], viv[in_view]
        weights = _coordinate_descent(
            _fourier(words, uu, vv),
            observed,
            sigamp,
            words.amp,
            viv,
            float(flux),
            sparseness,
            regularisation,
            maxiter,
            abort_numerical_issues,
        )
        best = _strongest(weights)
        if not len(best):
            break
        if generation == iterations - 1:
            image = _render(words[best], weights[best], image_shape, pixel)
            break
        words = _refine(words[best], rng, n_components, image_shape, pixel, lo, hi, max_anisotropy)

    brightness = unit / u.arcsec**2
    if not map:
        return image * brightness
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)
    header["bunit"] = brightness.to_string("fits")
    return Map((image, header))
