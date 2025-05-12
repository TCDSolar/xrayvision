from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import chain
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import binom, factorial

import astropy.units as apu
from astropy.units import Quantity

#: A source parameter value, either a plain float or an astropy `Quantity`.
type FloatOrQuantity = float | Quantity

#: Canonical internal units: every length-like source parameter is stored as a plain
#: float in arcsec, and every angle-like parameter as a plain float in radians. Users
#: must supply a `Quantity` for these at construction time; it is converted once, here,
#: at the boundary - internal code (e.g. the fit loop) only ever sees already-canonical
#: plain floats and never has to guess what unit a bare number is in.
DISTANCE_UNIT = apu.arcsec
ANGLE_UNIT = apu.rad

__all__ = [
    "Circular",
    "Elliptical",
    "GenericSource",
    "Source",
    "SourceFactory",
    "SourceList",
    "circular_gaussian_img",
    "circular_gaussian_vis",
    "elliptical_gaussian_img",
    "elliptical_gaussian_vis",
]


def circular_gaussian_img(
    amp: float, x: NDArray[np.float64], y: NDArray[np.float64], x0: float, y0: float, sigma: float
) -> NDArray[np.float64]:
    r"""
    Circular gaussian function sampled at x, y.

    .. math::

        F(x, y) = A \exp{\left(-\frac{(x0-x)^2 + (y0 - y)^2}{2\sigma^2}\right)}


    Parameters
    ----------
    amp :
        Amplitude
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma :
        Sigma

    See Also
    --------
    circular_gaussian_vis
    """
    return amp / (2 * np.pi * sigma**2) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def circular_gaussian_vis(
    amp: float, u: NDArray[np.float64], v: NDArray[np.float64], x0: float, y0: float, sigma: float
) -> NDArray[np.complex128]:
    r"""
    Circular gaussian in Fourier space sampled at u, v.

    .. math::

        F(u, v) = A \exp{\left( -2\pi^2 \sigma^2 (u^2 +v^2 \right)}) \exp( 2\pi i(x0u + y0v))


    Parameters
    ----------
    amp :
        Amplitude
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma :
        Sigma

    See Also
    --------
    circular_gaussian
    """
    return cast(
        NDArray[np.complex128],
        amp * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2)) * np.exp(2j * np.pi * (x0 * u + y0 * v)),
    )


def elliptical_gaussian_img(
    amp: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x0: float,
    y0: float,
    sigmax: float,
    sigmay: float,
    theta: float,
) -> NDArray[np.float64]:
    r"""
    Elliptical gaussian sampled at x, y.

    .. math::

        x' &= ((x0 - x) \cos(\theta) + ((y0 - y) \sin(\theta)) \\
        y' &= -((x0 - x) \sin(\theta) + ((y0 - y) \cos(\theta)) \\
        F(x, y) &= \frac{A}{(2 \pi \sigma_x \sigma_y)} \exp \left( \frac{x'^2}{2\sigma_x^2} + \frac{y'^2}{\sigma_y^2} \right)


    Parameters
    ----------
    amp :
        Amplitude
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigmax :
        Sigma in x direction
    sigmay :
        Sigma in y direction
    theta :
        Rotation angle in anticlockwise

    See Also
    --------
    elliptical_gaussian_vis
    """
    sint = np.sin(theta)
    cost = np.cos(theta)
    xp = ((x0 - x) * cost) + ((y0 - y) * sint)
    yp = -((x0 - x) * sint) + ((y0 - y) * cost)
    return cast(
        NDArray[np.float64],
        amp / (2 * np.pi * sigmax * sigmay) * np.exp(-((xp**2 / (2 * sigmax**2)) + (yp**2 / (2 * sigmay**2)))),
    )


def elliptical_gaussian_vis(
    amp: float,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    x0: float,
    y0: float,
    sigmax: float,
    sigmay: float,
    theta: float,
) -> NDArray[np.complex128]:
    r"""
    Elliptical gaussian in Fourier space sampled at u, v.

    .. math::

        x' &= u\cos(\theta) +v \sin(\theta) \\
        y' &= -u \sin(\theta) + v \cos(\theta) \\
        F(x, y) &= A \exp \left( -2\pi^2 ((u'^2\sigma_x^2) + (v'^2\sigma_y^2) \right) \exp( 2\pi i(x0u + y0v))

    Parameters
    ----------
    amp :
        Amplitude
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigmax :
        Sigma in x direction
    sigmay :
        Sigma in y direction
    theta :
        Rotation angle in anticlockwise

    See Also
    --------
    elliptical_gaussian
    """
    sint = np.sin(theta)
    cost = np.cos(theta)
    up = cost * u + sint * v
    vp = -sint * u + cost * v
    return cast(
        NDArray[np.complex128],
        amp
        * np.exp(-2 * np.pi**2 * ((up**2 * sigmax**2) + (vp**2 * sigmay**2)))
        * np.exp(2j * np.pi * (x0 * u + y0 * v)),
    )


def loop_img_old(
    amp: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x0: float,
    y0: float,
    fwhm_min: float,
    fwhm_max: float,
    rotation: float,
    loopwidth: float,
    max_comps: int = 21,
) -> NDArray[np.float64]:
    r"""
    Loop source in image space sampled at x, y.

    The loop source is approximated with a series of equispaced circular Gaussians.

    Parameters
    ----------
    amp :
        Total flux
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    fwhm_min :
        FWHM of the semiminor axis
    fwhm_max :
        FWHM of the semimajor axis
    rotation :
        Position angle of the loop major axis in radians
    loopwidth :
        Arc extent parameter (related to opening angle)
    max_comps : int
        Upper limit on the number of equispaced circular Gaussians used to approximate the loop

    Returns
    -------
    image : ndarray
        2D image of the loop source

    See Also
    --------
    loop_img
    """
    sig2fwhm = np.sqrt(8 * np.log(2))

    # Calculate the relative strengths of the  sources to reproduce a gaussian and their collective stddev.
    iseq0 = np.arange(max_comps)
    relflux0 = factorial(max_comps - 1) / (factorial(iseq0) * factorial(max_comps - 1 - iseq0)) / 2 ** (max_comps - 1)
    ok = np.flatnonzero(relflux0 > 0.01)  # Just keep; circles that contain; at least 1 % of flux
    ncirc = ok.size
    relflux = relflux0[ok] / relflux0[ok].sum()
    iseq = np.arange(ncirc)
    reltheta = iseq / (ncirc - 1.0) - 0.5  # locations of circles for arclength=1
    factor = np.sqrt((reltheta**2 * relflux).sum()) * sig2fwhm  # FWHM of binomial distribution for arclength=1

    loopangle = loopwidth / factor
    if np.abs(loopangle) >= 2 * np.pi:
        raise ValueError(f"Internal parameterization error - Loop arc {loopangle} exceeds 2 pi.")

    if loopangle == 0.0:
        loopangle = 0.01  # Avoid problems if loopangle = 0

    theta = np.abs(loopangle) * (iseq / (ncirc - 1.0) - 0.5)  # equispaced between + - loopangle / 2
    xloop = np.sin(theta)  # for unit radius of curvature, R
    yloop = np.cos(theta)  # relative to center of curvature

    if loopangle < 0:
        yloop = -yloop  # Sign of loopangle determines sense of loop curvature

    # Determine the size and location of the equivalent separated components in a coord system where x is an axis
    # parallel to the line joining the footpoints. Note that there are combinations of loop angle, sigminor and
    # sigmajor that cannot occur with radius > 1arcsec. In such a case circle radius is set to 1. Such cases will lead
    # to bad solutions and be flagged as such at the end.

    sigminor = fwhm_min / sig2fwhm
    sigmajor = fwhm_max / sig2fwhm
    fsumx2 = (xloop**2 * relflux).sum()  # scale - free factors describing loop moments for endpoint separation=1
    fsumy = (yloop * relflux).sum()
    fsumy2 = (yloop**2 * relflux).sum()
    loopradius = np.sqrt((sigmajor**2 - sigminor**2) / (fsumx2 - fsumy2 + fsumy**2))
    sgm_unti = getattr(sigmajor, "unit", 1)
    term = max((sigmajor**2 - loopradius**2 * fsumx2), 1 * sgm_unti**2)  # > 0 condition avoids problems in next step.
    circfwhm = max(sig2fwhm * np.sqrt(term), 1 * sgm_unti)  # Set minimum to avoid display problems

    cgshift = loopradius * fsumy  # will enable emission centroid location to be unchanged
    relx = xloop * loopradius  # x is axis joining 'footpoints'
    rely = yloop * loopradius - cgshift

    # Calculate source structures for each circle.
    pasep = rotation
    sinus = np.sin(pasep)
    cosinus = np.cos(pasep)

    image = None
    pixel = [1, 1]
    for i in range(iseq.size):
        flux_new = amp * relflux[i]  #  Split the flux between components.

        x_loc_new = x0 - relx[i] * sinus + rely[i] * cosinus
        y_loc_new = y0 + relx[i] * cosinus + rely[i] * sinus

        x_tmp = ((x - x_loc_new) * cosinus) + ((y - y_loc_new) * sinus)
        y_tmp = -((x - x_loc_new) * sinus) + ((y - y_loc_new) * cosinus)
        x_tmp = 2.0 * np.sqrt(2.0 * np.log(2.0)) * x_tmp / circfwhm
        y_tmp = 2.0 * np.sqrt(2.0 * np.log(2.0)) * y_tmp / circfwhm
        im_tmp = np.exp(-(x_tmp**2.0 + y_tmp**2.0) / 2.0)
        # Accumulate via None rather than np.zeros_like: amp/x,y may be Quantity, and a
        # zeros array started from a Quantity's shape would wrongly inherit its unit.
        if image is None:
            image = im_tmp / (im_tmp.sum() * pixel[0] * pixel[1]) * flux_new
        else:
            image += im_tmp / (im_tmp.sum() * pixel[0] * pixel[1]) * flux_new

    return cast(NDArray[np.float64], image)


def loop_vis_old(
    amp: float,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    x0: float,
    y0: float,
    fwhm_minor: float,
    fwhm_major: float,
    rotation: float,
    loopwidth: float,
    max_comps: int = 21,
) -> NDArray[np.complex128]:
    r"""
    Loop source in Fourier space sampled at u, v.

    Parameters
    ----------
    amp :
        Total flux
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    fwhm_minor :
        FWHM of the semiminor axis
    fwhm_major :
        FWHM of the semimajor axis
    rotation :
        Position angle of the loop major axis in radians
    loopwidth :
        Arc extent parameter (related to opening angle)
    max_comps : int
        Upper limit on the number of equispaced circular Gaussians used to approximate the loop

    Returns
    -------
    vis : ndarray (complex128)
        Complex visibilities evaluated at (u, v)

    See Also
    --------
    loop_vis
    """

    sig2fwhm = np.sqrt(8 * np.log(2.0))

    # Calculate the relative strengths of the sources to reproduce a gaussian and their collective stddev.
    iseq0 = np.arange(max_comps)
    relflux0 = (
        factorial(max_comps - 1) / (factorial(iseq0) * factorial(max_comps - 1 - iseq0)) / 2 ** (max_comps - 1)
    )  # TOTAL(relflux)=1
    ok = np.flatnonzero(relflux0 > 0.01)  # Just keep circles that contain at least 1% of flux
    ncirc = ok.size
    relflux = relflux0[ok] / (relflux0[ok]).sum()
    iseq = np.arange(ncirc)
    reltheta = iseq / (ncirc - 1.0) - 0.5  # locations of circles for arclength=1
    factor = np.sqrt((reltheta**2 * relflux).sum()) * sig2fwhm  # FWHM of binomial distribution for arclength=1

    loopangle = loopwidth / factor
    if np.abs(loopangle).sum() >= 2 * np.pi:
        raise ValueError(f"Internal parameterization error - Loop arc {loopangle} exceeds 2pi.")

    if loopangle == 0:
        loopangle = 0.01  # Avoids problems if loopangle = 0

    theta = np.abs(loopangle) * (iseq / (ncirc - 1.0) - 0.5)  # equispaced between +- loopangle/2
    xloop = np.sin(theta)  # for unit radius of curvature, R
    yloop = np.cos(theta)  # relative to center of curvature

    if loopangle < 0:
        # Sign of loopangle determines sense of loop curvature # Sign of loopangle determines sense of loop curvature
        yloop = -yloop

    # Determine the size and location of the equivalent separated components in a coord system where...
    # x is an axis parallel to the line joining the footpoints
    # Note that there are combinations of loop angle, sigminor and sigmajor that cannot occur with radius>1arcsec.
    # In such a case circle radius is set to 1.  Such cases will lead to bad solutions and be flagged as such at the end.

    # eccen = np.sqrt(1 - (sigma_min**2 / sigma_max**2))
    # sigminor = sigma_min * (1 - eccen ** 2) ** 0.25 / sig2fwhm
    # sigmajor = sigma_max / (1 - eccen ** 2) ** 0.25 / sig2fwhm

    sigminor = fwhm_minor / sig2fwhm
    sigmajor = fwhm_major / sig2fwhm
    fsumx2 = (xloop**2 * relflux).sum()  # scale-free factors describing loop moments for endpoint separation=1
    fsumy = (yloop * relflux).sum()
    fsumy2 = (yloop**2 * relflux).sum()
    loopradius = np.sqrt((sigmajor**2 - sigminor**2) / (fsumx2 - fsumy2 + fsumy**2))
    sgm_unti = getattr(sigmajor, "unit", 1)
    term = max((sigmajor**2 - loopradius**2 * fsumx2), 1 * sgm_unti**2)  # >0 condition avoids problems in next step.
    circfwhm = max(sig2fwhm * np.sqrt(term), 1 * sgm_unti)  # Set minimum to avoid display problems

    cgshift = loopradius * fsumy
    relx = xloop * loopradius  # x is axis joining 'footpoints'
    rely = yloop * loopradius - cgshift  # will enable emission centroid location to be unchanged

    # Calculate source structures for each circle.
    pasep = rotation  # position angle of line joining arc endpoints
    x_loc_new = x0 - relx * np.sin(pasep) + rely * np.cos(pasep)
    y_loc_new = y0 + relx * np.cos(pasep) + rely * np.sin(pasep)

    flux_new = amp * relflux  # Split the flux between components.

    arg = (-(np.pi**2) * circfwhm**2) / (4 * np.log(2)) * (u**2 + v**2)
    relvis = np.exp(arg)

    # Accumulate via None rather than np.zeros_like: u/amp may be Quantity, and a zeros
    # array started from a Quantity's shape would wrongly inherit its unit.
    vis = None
    for j in range(ncirc):
        if vis is None:
            vis = flux_new[j] * relvis * np.exp(2j * np.pi * (x_loc_new[j] * u + y_loc_new[j] * v))
        else:
            vis += flux_new[j] * relvis * np.exp(2j * np.pi * (x_loc_new[j] * u + y_loc_new[j] * v))
    return cast(NDArray[np.complex128], vis)


def loop_img(
    flux: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x0: float,
    y0: float,
    sigma_minor: float,
    sigma_major: float,
    rotation: float,
    loopwidth: float,
    min_fraction: float = 0.01,
    max_comps: int = 21,
) -> NDArray[np.float64]:
    r"""
    Loop source in image space sampled at x, y.

    The loop is approximated as a series of circular Gaussians with binomially-weighted
    fluxes arranged along a circular arc.

    Parameters
    ----------
    flux :
        Total integrated flux
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma_minor :
        Standard deviation of the loop width perpendicular to the arc
    sigma_major :
        Standard deviation of the loop extent along the arc
    rotation :
        Position angle of the loop major axis in radians
    loopwidth :
        Arc extent parameter (related to opening angle)
    min_fraction : float
        Minimum relative flux to retain a component
    max_comps : int
        Upper limit on the number of Gaussian components

    Returns
    -------
    image : ndarray
        Loop brightness distribution evaluated at (x, y)

    See Also
    --------
    loop_vis, loop_img_old
    """
    component_fluxes, n_components = _compute_binomial_weights(max_comps, min_fraction)

    loop_params = _compute_loop_geometry(sigma_minor, sigma_major, loopwidth, component_fluxes, n_components)

    x_components, y_components = _transform_to_image_coords(
        loop_params["rel_x"], loop_params["rel_y"], x0, y0, rotation
    )

    data = _evaluate_gaussians_on_grid(
        x, y, x_components, y_components, loop_params["component_sigma"], flux * component_fluxes, rotation
    )

    return data


def loop_vis(
    flux: float,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    x0: float,
    y0: float,
    sigma_minor: float,
    sigma_major: float,
    rotation: float,
    loopwidth: float,
    min_fraction: float = 0.01,
    max_comps: int = 21,
) -> NDArray[np.complex128]:
    r"""
    Loop source in Fourier space sampled at u, v.

    Parameters
    ----------
    flux :
        Total integrated flux
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma_minor :
        Standard deviation of the loop width perpendicular to the arc
    sigma_major :
        Standard deviation of the loop extent along the arc
    rotation :
        Position angle of the loop major axis in radians
    loopwidth :
        Arc extent parameter (related to opening angle)
    min_fraction : float
        Minimum relative flux to retain a component
    max_comps : int
        Upper limit on the number of Gaussian components

    Returns
    -------
    vis : ndarray (complex128)
        Complex visibilities evaluated at (u, v)

    See Also
    --------
    loop_img, loop_vis_old
    """
    component_fluxes, n_components = _compute_binomial_weights(max_comps, min_fraction)

    loop_params = _compute_loop_geometry(sigma_minor, sigma_major, loopwidth, component_fluxes, n_components)

    x_components, y_components = _transform_to_image_coords(
        loop_params["rel_x"], loop_params["rel_y"], x0, y0, rotation
    )

    vis = _evaluate_visibility_analytical(
        u, v, x_components, y_components, loop_params["component_sigma"], flux * component_fluxes
    )

    return vis


def _compute_binomial_weights(max_comps: int, min_fraction: float = 0.01) -> tuple[NDArray[np.float64], int]:
    """
    Compute normalized binomial distribution weights for loop Gaussian components.

    Parameters
    ----------
    max_comps : int
        Maximum number of components to consider
    min_fraction : float
        Minimum relative flux to retain a component

    Returns
    -------
    weights : ndarray
        Normalized flux weights summing to 1
    n_kept : int
        Number of components retained
    """
    indices = np.arange(max_comps)

    # Binomial coefficients: C(n-1, k) / 2^(n-1)
    # Using scipy.special.binom is more efficient and numerically stable than factorial
    binomial_coeffs = binom(max_comps - 1, indices) / 2.0 ** (max_comps - 1)

    # Keep only significant components
    significant = binomial_coeffs > min_fraction
    weights_kept = binomial_coeffs[significant]

    # Normalize
    weights_normalized = weights_kept / weights_kept.sum()

    return weights_normalized, len(weights_normalized)


def _compute_loop_geometry(
    sigma_minor: float, sigma_major: float, arc_param: float, flux_weights: NDArray[np.float64], n_comps: int
) -> dict[str, Any]:
    """
    Compute loop geometry: component positions and size.

    Parameters
    ----------
    sigma_minor :
        Minor axis standard deviation (perpendicular to arc)
    sigma_major :
        Major axis standard deviation (along arc)
    arc_param :
        Arc extent parameter
    flux_weights : ndarray
        Normalized flux weights for each component
    n_comps : int
        Number of components

    Returns
    -------
    params : dict
        Dictionary with keys ``rel_x``, ``rel_y``, ``component_sigma``, ``radius``
    """
    # Component positions (normalized to [-0.5, +0.5])
    comp_indices = np.arange(n_comps)
    normalized_positions = comp_indices / (n_comps - 1.0) - 0.5

    # Binomial spatial distribution factor
    # This relates the arc extent to the actual opening angle
    binomial_width = np.sqrt((normalized_positions**2 * flux_weights).sum()) * 2 * np.sqrt(2 * np.log(2))

    # Opening angle
    loop_angle = arc_param / binomial_width

    # Validate opening angle
    if np.abs(loop_angle) >= 2.0 * np.pi:
        raise ValueError(
            f"Loop arc parameter {arc_param} produces opening angle {loop_angle:.3f} rad "
            f"(>= 2π). Reduce arc_param or adjust sigma_max."
        )

    # Handle zero angle
    if loop_angle == 0.0:
        loop_angle = 0.01

    # Angular positions along arc
    theta = np.abs(loop_angle) * (comp_indices / (n_comps - 1.0) - 0.5)

    # Positions on unit circle (radius = 1)
    x_unit = np.sin(theta)
    y_unit = np.cos(theta)

    # Flip y for negative angles (curvature sense)
    if loop_angle < 0:
        y_unit = -y_unit

    # Compute loop radius from ellipse parameters
    # Statistical moments of the distribution
    moment_x2 = (x_unit**2 * flux_weights).sum()
    moment_y = (y_unit * flux_weights).sum()
    moment_y2 = (y_unit**2 * flux_weights).sum()

    # Radius of curvature
    denominator = moment_x2 - moment_y2 + moment_y**2
    if denominator <= 0:
        # Degenerate case - use fallback
        radius = sigma_major
    else:
        radius = np.sqrt((sigma_major**2 - sigma_minor**2) / denominator)

    # Component sigma (circular Gaussians)
    # Need to account for the spread along the arc
    variance_residual = sigma_major**2 - radius**2 * moment_x2

    # Ensure non-negative variance
    variance_residual = max(variance_residual, 0.0)
    component_sigma = np.sqrt(variance_residual)
    # Set minimum size to avoid numerical issues
    component_sigma = max(component_sigma, 1.0)

    # Center-of-gravity shift (keeps loop centered at x0, y0)
    cg_shift = radius * moment_y

    # Relative positions (before rotation)
    rel_x = x_unit * radius
    rel_y = y_unit * radius - cg_shift

    return {"rel_x": rel_x, "rel_y": rel_y, "component_sigma": component_sigma, "radius": radius}


def _transform_to_image_coords(
    rel_x: NDArray[np.float64], rel_y: NDArray[np.float64], x0: float, y0: float, rotation_angle: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Transform relative component positions to absolute image coordinates.

    Parameters
    ----------
    rel_x : ndarray
        Relative x positions in the loop-aligned frame
    rel_y : ndarray
        Relative y positions in the loop-aligned frame
    x0 :
        Loop center x coordinate
    y0 :
        Loop center y coordinate
    rotation_angle :
        Rotation angle in radians

    Returns
    -------
    x_abs : ndarray
        Absolute x component positions
    y_abs : ndarray
        Absolute y component positions
    """
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)

    # Rotation matrix application
    # Note: negative rel_x term because of coordinate convention
    x_abs = x0 - rel_x * sin_angle + rel_y * cos_angle
    y_abs = y0 + rel_x * cos_angle + rel_y * sin_angle

    return x_abs, y_abs


def _evaluate_gaussians_on_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_centers: NDArray[np.float64],
    y_centers: NDArray[np.float64],
    sigma: float,
    fluxes: NDArray[np.float64],
    rotation: float,
) -> NDArray[np.float64]:
    """
    Evaluate the sum of circular Gaussians on an image grid.

    Parameters
    ----------
    x : ndarray
        x coordinate grid
    y : ndarray
        y coordinate grid
    x_centers : ndarray
        x positions of Gaussian centers
    y_centers : ndarray
        y positions of Gaussian centers
    sigma :
        Standard deviation of each Gaussian component
    fluxes : ndarray
        Flux of each component
    rotation :
        Rotation angle in radians

    Returns
    -------
    image : ndarray
        Sum of all Gaussian components
    """
    # Pre-compute constants
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)

    # Initialize output
    image = np.zeros(x.shape, like=fluxes)

    # Sum Gaussians
    for x_c, y_c, flux in zip(x_centers, y_centers, fluxes):
        # Shift to component center
        dx_grid = x - x_c
        dy_grid = y - y_c

        # Rotate to component frame (optional - components are circular)
        # This rotation isn't strictly needed for circular Gaussians but matches original
        x_rot = dx_grid * cos_rot + dy_grid * sin_rot
        y_rot = -dx_grid * sin_rot + dy_grid * cos_rot

        # Gaussian profile: exp(-r^2/(2*sigma^2))
        gaussian = np.exp(-0.5 * ((x_rot / sigma) ** 2 + (y_rot / sigma) ** 2))

        # Normalize using flux density (continuous) normalization
        # This makes it consistent with circular_gaussian_img()
        gaussian_normalized = gaussian / (gaussian.sum())

        # Add contribution
        image += gaussian_normalized * flux

    return image


def _evaluate_visibility_analytical(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    x_centers: NDArray[np.float64],
    y_centers: NDArray[np.float64],
    sigma: float,
    fluxes: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""
    Analytical Fourier transform of a multi-component circular Gaussian loop.

    .. math::

        V(u,v) = \sum_i F_i \exp\left(-2\pi^2\sigma^2(u^2+v^2)\right)
                 \exp\left(2\pi i(u x_i + v y_i)\right)

    Parameters
    ----------
    u : ndarray
        u coordinates
    v : ndarray
        v coordinates
    x_centers : ndarray
        x positions of Gaussian components
    y_centers : ndarray
        y positions of Gaussian components
    sigma :
        Standard deviation of each Gaussian component
    fluxes : ndarray
        Flux of each component

    Returns
    -------
    vis : ndarray (complex128)
        Complex visibilities evaluated at (u, v)
    """
    # Gaussian envelope in Fourier space
    # FT{exp(-r^2/(2*sigma^2))} ~ exp(-2*pi^2*sigma^2*k^2)
    uv_squared = u**2 + v**2
    envelope = np.exp(-2 * np.pi**2 * sigma**2 * uv_squared)

    # Initialize visibility
    vis = np.zeros_like(u, dtype=np.complex128)

    # Sum over components
    for x_c, y_c, flux in zip(x_centers, y_centers, fluxes):
        # Phase from component position
        phase = 2.0 * np.pi * (x_c * u + y_c * v)

        # Add component contribution
        vis += flux * envelope * np.exp(1j * phase)

    return vis


def _strip_unit(value: FloatOrQuantity) -> float:
    r"""Return the bare numeric value, stripping units from a `Quantity` if present.

    Only used for ``amp``, which has no single canonical unit (it depends on the data
    being fit), so it is never converted - just unwrapped if given as a `Quantity`.
    """
    if isinstance(value, Quantity):
        return value.value  # type: ignore[no-any-return]  # pyright: ignore[reportAttributeAccessIssue]
    return value


class GenericSource(ABC):
    r"""
    Abstract source class defining the properties and methods.

    Subclasses declare a `_param_units` tuple, one entry per dataclass field in
    declaration order, giving the canonical unit each parameter is converted to at
    construction time (or `None` for a parameter that is not unit-constrained, e.g.
    ``amp``). The public constructor (``__init__``, and so also `SourceFactory`)
    requires a `~astropy.units.Quantity` for every unit-constrained parameter and
    converts it immediately, so every such attribute is a plain float in its
    canonical unit from then on - internal code never has to guess what unit a bare
    number is in. `_from_canonical` bypasses this validation for code (the
    optimizer's fit loop, via `SourceList.from_params`) that already has plain
    floats in the canonical units.
    """

    _registry: dict[str, "type[GenericSource]"] = {}
    _param_units: ClassVar[tuple[Any, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        key = cls.__name__.lower()
        GenericSource._registry[key] = cls

    @classmethod
    def _canonicalize(cls, *values: FloatOrQuantity) -> tuple[float, ...]:
        r"""Validate and convert constructor arguments to their canonical units."""
        names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
        canonical = []
        for name, unit, value in zip(names, cls._param_units, values):
            if unit is None:
                canonical.append(value)
            elif isinstance(value, Quantity):
                canonical.append(value.to_value(unit))  # pyright: ignore[reportAttributeAccessIssue]
            else:
                raise TypeError(
                    f"{cls.__name__}.{name} must be given as an astropy Quantity with units "
                    f"convertible to '{unit}', not a bare number: {value!r}. "
                    f"Pass e.g. {value} * astropy.units.{unit}."
                )
        return tuple(canonical)

    @classmethod
    def _from_canonical(cls, *values: float) -> "GenericSource":
        r"""Construct directly from already-canonical plain floats, bypassing validation."""
        obj = cls.__new__(cls)
        for name, value in zip((f.name for f in fields(cls)), values):  # type: ignore[arg-type]
            setattr(obj, name, value)
        return obj

    def to_quantities(self) -> "GenericSource":
        r"""Return a copy with every unit-constrained field as a `Quantity` in its canonical unit.

        Useful for presenting a result (e.g. a fitted source) to a caller: internal code
        works with plain floats, but a value handed back across the public API should
        carry its unit rather than leaving the caller to guess it. The copy bypasses
        `__init__` (which would immediately strip the unit straight back off again), so
        it is for display/inspection only - do not feed it back into fitting machinery
        such as `SourceList.from_params` or `.bounds`.
        """
        obj = self.__class__.__new__(self.__class__)
        for name, unit in zip((f.name for f in fields(self)), self._param_units):  # type: ignore[arg-type]
            value = getattr(self, name)
            setattr(obj, name, value if unit is None else value * unit)
        return obj

    @property
    def n_params(self) -> int:
        r"""The number of parameters"""
        return len(self.__dict__.keys())

    @property
    @abstractmethod
    def bounds(self) -> list[list[float]]:
        r"""Return the lower and upper bounds of the source."""
        pass

    @property
    @abstractmethod
    def param_list(self) -> list[float]:
        """Return list of parameters if fixed order"""
        pass

    @abstractmethod
    def estimate_bounds(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        """Return estimated bounds"""
        pass


@dataclass()
class Circular(GenericSource):
    amp: FloatOrQuantity
    x0: float
    y0: float
    sigma: float

    _param_units: ClassVar[tuple[Any, ...]] = (None, DISTANCE_UNIT, DISTANCE_UNIT, DISTANCE_UNIT)

    def __init__(self, amp: FloatOrQuantity, x0: Quantity, y0: Quantity, sigma: Quantity) -> None:
        r"""
        Circular gaussian source parameters.

        Parameters
        ----------
        amp :
            Amplitude
        x0 :
            Center x coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        y0 :
            Center y coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        sigma :
            Standard deviation, as a `~astropy.units.Quantity` convertible to arcsec
        """
        self.amp, self.x0, self.y0, self.sigma = self._canonicalize(amp, x0, y0, sigma)

    @property
    def bounds(self) -> list[list[float]]:
        amp = _strip_unit(self.amp)
        return [
            [amp / 4, self.x0 - 5 * np.abs(self.sigma), self.y0 - 5 * np.abs(self.sigma), self.sigma / 4],
            [amp * 4, self.x0 + 5 * np.abs(self.sigma), self.y0 + 5 * np.abs(self.sigma), self.sigma * 4],
        ]

    @property
    def param_list(self) -> list[float]:
        return [self.amp, self.x0, self.y0, self.sigma]

    def estimate_bounds(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        raise NotImplementedError()


@dataclass
class Elliptical(GenericSource):
    amp: FloatOrQuantity
    x0: float
    y0: float
    sigmax: float
    sigmay: float
    theta: float

    _param_units: ClassVar[tuple[Any, ...]] = (
        None,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        ANGLE_UNIT,
    )
    #: Half-width of the default theta search window (see `bounds`), in radians.
    _THETA_WINDOW = np.deg2rad(22.5)

    def __init__(
        self,
        amp: FloatOrQuantity,
        x0: Quantity,
        y0: Quantity,
        sigmax: Quantity,
        sigmay: Quantity,
        theta: Quantity,
    ) -> None:
        r"""
        Elliptical gaussian source parameters.

        Parameters
        ----------
        amp :
            Amplitude
        x0 :
            Center x coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        y0 :
            Center y coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        sigmax :
            Standard deviation in x direction, as a `~astropy.units.Quantity` convertible to arcsec
        sigmay :
            Standard deviation in y direction, as a `~astropy.units.Quantity` convertible to arcsec
        theta :
            Rotation angle in anticlockwise, as a `~astropy.units.Quantity` convertible to radians (e.g. degrees)
        """
        self.amp, self.x0, self.y0, self.sigmax, self.sigmay, self.theta = self._canonicalize(
            amp, x0, y0, sigmax, sigmay, theta
        )

    @property
    def bounds(self) -> list[list[float]]:
        amp = _strip_unit(self.amp)
        return [
            [
                amp / 4,
                self.x0 - (5 * np.abs(self.sigmax)),
                self.y0 - (5 * np.abs(self.sigmay)),
                self.sigmax / 4,
                self.sigmay / 4,
                self.theta - self._THETA_WINDOW,
            ],
            [
                amp * 4,
                self.x0 + (5 * np.abs(self.sigmax)),
                self.y0 + (5 * np.abs(self.sigmay)),
                self.sigmax * 4,
                self.sigmay * 4,
                self.theta + self._THETA_WINDOW,
            ],
        ]

    @property
    def param_list(self) -> list[float]:
        return [self.amp, self.x0, self.y0, self.sigmax, self.sigmay, self.theta]

    def estimate_bounds(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        return self.bounds


@dataclass
class Loop(GenericSource):
    amp: FloatOrQuantity
    x0: float
    y0: float
    sigma_min: float
    sigma_max: float
    alpha: float
    beta: float

    _param_units: ClassVar[tuple[Any, ...]] = (
        None,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        DISTANCE_UNIT,
        ANGLE_UNIT,
        ANGLE_UNIT,
    )

    def __init__(
        self,
        amp: FloatOrQuantity,
        x0: Quantity,
        y0: Quantity,
        sigma_min: Quantity,
        sigma_max: Quantity,
        alpha: Quantity,
        beta: Quantity,
    ) -> None:
        r"""
        Loop source parameters.

        Parameters
        ----------
        amp :
            Total flux
        x0 :
            Center x coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        y0 :
            Center y coordinate, as a `~astropy.units.Quantity` convertible to arcsec
        sigma_min :
            Standard deviation of the loop width perpendicular to the arc, as a `~astropy.units.Quantity`
            convertible to arcsec
        sigma_max :
            Standard deviation of the loop extent along the arc, as a `~astropy.units.Quantity`
            convertible to arcsec
        alpha :
            Position angle of the loop major axis, as a `~astropy.units.Quantity` convertible to radians
        beta :
            Arc extent parameter, as a `~astropy.units.Quantity` convertible to radians
        """
        self.amp, self.x0, self.y0, self.sigma_min, self.sigma_max, self.alpha, self.beta = self._canonicalize(
            amp, x0, y0, sigma_min, sigma_max, alpha, beta
        )

    @property
    def bounds(self) -> list[list[float]]:
        amp = _strip_unit(self.amp)
        return [
            [
                amp / 2,
                self.x0 - (2 * np.abs(self.sigma_max)),
                self.y0 - (2 * np.abs(self.sigma_max)),
                self.sigma_min / 2,
                self.sigma_max / 2,
                -np.pi / 2,
                0,
            ],
            [
                amp * 2,
                self.x0 + (2 * np.abs(self.sigma_max)),
                self.y0 + (2 * np.abs(self.sigma_max)),
                self.sigma_min * 2,
                self.sigma_max * 2,
                np.pi / 2,
                np.pi,
            ],
        ]

    @property
    def param_list(self) -> list[float]:
        return [self.amp, self.x0, self.y0, self.sigma_min, self.sigma_max, self.alpha, self.beta]

    def estimate_bounds(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        return self.bounds


class SourceList(UserList[GenericSource]):
    r"""
    List of Sources
    """

    def __init__(self, sources: list[GenericSource] | None = None) -> None:
        r"""
        List of Sources

        Parameters
        ----------
        sources :
            Sources
        """
        super().__init__(sources)

    @property
    def params(self) -> list[float]:
        r"""Flat list of all parameters for all sources"""
        return list(chain.from_iterable([source.param_list for source in self.data]))

    @property
    def bounds(self) -> list[list[float]]:
        r"""Flat list of upper and lower bounds for all sources"""
        return np.hstack([s.bounds for s in self.data]).tolist()  # type: ignore[no-any-return]

    @classmethod
    def from_params(cls, sources: "SourceList", params: Sequence[float] | NDArray[np.float64]) -> "SourceList":
        r"""
        Create a source list from given parameters and sources.

        Parameters
        ----------
        sources :
            List of sources
        params
            Flat list of all parameters for all sources.
        """
        j = 0
        new_sources = cls()
        for source in sources:
            n_params = source.n_params
            # Bypasses unit validation: these params are already plain floats in
            # canonical units, produced by a previous call to `.params`/`.bounds` or by
            # the optimiser, so there is nothing left to convert.
            new_sources.append(
                type(source)._from_canonical(*list(params[j : j + n_params]))  # pyright: ignore[reportPrivateUsage]
            )
            j += n_params

        return new_sources


class SourceFactory:
    r"""
    Source Factory
    """

    def __init__(self, registry: "dict[str, type[GenericSource]]") -> None:
        self._registry: dict[str, type[GenericSource]] = registry

    def __call__(self, shape_type: str, *args: Any, **kwargs: Any) -> GenericSource:
        shape_type = shape_type.lower()
        cls = self._registry.get(shape_type)
        if not cls:
            raise ValueError(f"Unknown shape type: {shape_type}")
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"Error creating '{shape_type}': {e}")


#: Instance of SourceFactory
Source = SourceFactory(registry=GenericSource._registry)  # pyright: ignore[reportPrivateUsage]
