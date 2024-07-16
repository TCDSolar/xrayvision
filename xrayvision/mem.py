r"""
Implementation of Maximum Entropy Method
"""

from types import SimpleNamespace
from typing import Union, Optional

import astropy.units as apu
import numpy as np
from astropy.units import Quantity
from numpy.linalg import norm
from numpy.typing import NDArray
from sunpy.map import Map

from xrayvision.imaging import generate_header
from xrayvision.transform import generate_xy
from xrayvision.utils import get_logger
from xrayvision.visibility import Visibilities

__all__ = [
    "_get_entropy",
    "_get_fourier_matrix",
    "_estimate_flux",
    "_get_mean_visibilities",
    "_proximal_entropy",
    "_proximal_operator",
    "_optimise_fb",
    "mem",
]


logger = get_logger(__name__, "DEBUG")


def _get_entropy(image, flux):
    r"""
    Return the entropy of an image.

    The entropy is defined as

    .. math::

        H(x) = sum_i x_i * log(x_i/(m e))

    where :math:`x` is an image and :math:`m` is the total flux divided by the number of pixels of
    in the image

    Parameters
    ----------
    image :
        Input image array
    flux :
        Total flux divided by the number of pixels of the image

    Returns
    -------

    """
    return np.sum(image * np.log(image / (flux * np.e)))


def _get_fourier_matrix(vis, shape=(64, 64) * apu.pix, pixel_size=(4.0312500, 4.0312500) * apu.arcsec):
    r"""
    Return the complex Fourier matrix used to compute the value of the visibilities.

    Generate the complex Fourier matrix :math:`Hv` used to compute the value of the
    visibilities. If :math:`\vec{x}` is the vectorised image, then
    :math:`v = \mathbf{\mathit{Hv}} \vec{x}` is the vector containing the complex visibilities

    Parameters
    ----------
    vis :
        Visibly object containing u, v sampling
    shape :
        Shape of the images
    pixel_size
        Size of the pixels

    Returns
    -------
    The complex Fourier matrix
    """
    m, n = shape.to("pix")
    y = generate_xy(m, phase_center=0 * apu.arcsec, pixel_size=pixel_size[1])
    x = generate_xy(n, phase_center=0 * apu.arcsec, pixel_size=pixel_size[0])
    x, y = np.meshgrid(x, y, indexing="ij")
    uv = np.vstack([vis.u, vis.v])
    # Check apu are correct for exp need to be dimensionless and then remove apu for speed
    if (vis.u * x[0, 0]).unit == apu.dimensionless_unscaled and (vis.v * y[0, 0]).unit == apu.dimensionless_unscaled:
        uv = uv.value
        x = x.value
        y = y.value

        Hv = np.exp(
            1j * 2 * np.pi * (x[..., np.newaxis] * uv[np.newaxis, 0, :] + y[..., np.newaxis] * uv[np.newaxis, 1, :])
        )

        return Hv * pixel_size[0].value * pixel_size[1].value


def _estimate_flux(vis, shape, pixel, maxiter=1000, tol=1e-3):
    r"""
    Estimate the total flux in the image by solving an optimisation problem.

    This function estimates the total flux of an event by solving the problem

    .. math::

        \underset{\chi}{\mathrm{argmin}}(f) = \sum \left (\Re V(f) - \Re v))^2
        + ( \Im V(f) - \Im v)^2 ) / \sigma^2 \right )

    subject to :math:`f >= 0`,

    the algorithm finds a positive image :math:`f` that minimizes the :math:`\chi` square function.
    The estimation of the total flux is then obtained by computing the total fux of :math:`f`.
    The method implemented :math:`f` the minimization is projected Landweber.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of image
    pixel :
        Size of pixels
    maxiter : int
        Maximum number of iterations
    tol :
        Tolerance at which to stop

    Returns
    -------
    Estimated total flux

    """

    Hv, Lip, Visib = _prepare_for_optimise(pixel, shape, vis)

    # PROJECTED LANDWEBER
    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value("pix").astype(int)).flatten()

    for i in range(maxiter):
        x_old = x[:]

        # GRADIENT STEP
        grad = 2.0 * np.matmul((np.matmul(Hv, x).value - Visib).T, Hv)
        y = x - 1.0 / Lip * grad

        # PROJECTION ON THE POSITIVE ORTHANT
        x = y.clip(min=0.0)
        tmp = x[:]
        Hvx = np.matmul(Hv, tmp)

        diff_V = Hvx - Visib
        chi2 = (diff_V**2.0).sum()

        if i % 25 == 0:
            logger.info(f"Iter: {i}, Chi2: {chi2}")

        if np.sqrt(((x - x_old) ** 2.0).sum()) < tol * np.sqrt((x_old**2.0).sum()):
            break

    return x.sum() * pixel[0] * pixel[1]


def _prepare_for_optimise(pixel, shape, vis):
    r"""
    Return matrices and vectors in format for optimisation

    For complex values create new matrix concatenating the real and imaginary components and
    normalise by the standard error

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of image
    pixel :
        Size of pixels


    Returns
    -------

    """
    Hv = _get_fourier_matrix(vis, shape, pixel)
    # Division of real and imaginary part of the matrix 'Hv'
    ReHv = np.real(Hv)
    ImHv = np.imag(Hv)
    # 'Hv' is composed by the union of its real and imaginary part
    Hv = np.concatenate([ReHv, ImHv], axis=-1)
    # Division of real and imaginary part of the visibilities
    ReV = np.real(vis.visibilities)
    ImV = np.imag(vis.visibilities)
    # 'Visib' is composed by the real and imaginary part of the visibilities
    Visib = np.concatenate([ReV, ImV], axis=-1)
    # Standard deviation of the real and imaginary part of the visibilities
    if vis.amplitude_uncertainty is None:
        sigma_Re = np.ones_like(ReV)
        sigma_Im = np.ones_like(ImV)
    else:
        sigma_Re = vis.amplitude_uncertainty
        sigma_Im = vis.amplitude_uncertainty
    # 'sigma': standard deviation of the data contained in 'Visib'
    sigma = np.concatenate([sigma_Re, sigma_Im], axis=-1)
    # RESCALING of 'Hv' AND 'Visib'(NEEDED FOR COMPUTING THE VALUE OF THE \chi ** 2; FUNCTION)
    # The vector 'Visib' and every column of 'Hv' are divided by 'sigma'
    Visib = Visib / sigma
    ones = np.ones(shape.to_value("pix").astype(int))
    sigma1 = sigma * ones[..., np.newaxis]
    Hv = Hv / sigma1
    # COMPUTATION OF THE LIPSCHITZ CONSTANT; 'Lip' OF THE GRADIENT OF  THE \chi ** 2
    # FUNCTION (NEEDED TO GUARANTEE THE CONVERGENCE OF THE ALGORITHM)
    Hv = Hv.transpose(2, 0, 1).reshape(Hv.shape[2], -1)
    HvHvT = np.matmul(Hv, Hv.T)
    # TODO magic number
    Lip = 2.1 * norm(HvHvT, 2)
    return Hv, Lip, Visib


def _get_mean_visibilities(vis, shape, pixel):
    r"""
    Return the mean visibilities sampling the same call in the discretisation of the (u,v) plane.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of image
    pixel :
        Size of pixels

    Returns
    -------
    Mean Visibilities
    """

    if vis.amplitude_uncertainty is None:
        amplitude_uncertainty = np.ones_like(vis.visibilities)
    else:
        amplitude_uncertainty = vis.amplitude_uncertainty

    imsize2 = shape[0] / 2
    pixel_size = 1 / (shape[0] * pixel[0])

    iu = vis.u / pixel_size
    iv = vis.v / pixel_size
    ru = np.around(iu)
    rv = np.around(iv)

    # index of the u coordinates of the sampling frequencies in the discretisation of the u axis
    ru = ru * apu.pix + imsize2.to(apu.pix)
    # index of the v coordinates of the sampling frequencies in the discretisation of the v axis
    rv = rv * apu.pix + imsize2.to(apu.pix)

    # matrix that represents the discretization of the (u,v)-plane
    iuarr = np.zeros(shape.to_value("pixel").astype(int))

    count = 0
    n_vis = vis.u.shape[0]
    u = np.zeros(n_vis) * (1 / apu.arcsec)
    v = np.zeros(n_vis) * (1 / apu.arcsec)
    den = np.zeros(n_vis)
    weights = np.ones_like(vis.visibilities**2)
    visib = np.zeros_like(vis.visibilities)
    for ip in range(n_vis):
        # what about 0.5 pix offset
        i = ru[ip].to_value("pix").astype(int)
        j = rv[ip].to_value("pix").astype(int)
        # (i, j) is the position of the spatial frequency in
        # the discretization of the (u,v)-plane 'iuarr'
        if iuarr[i, j] == 0.0:
            u[count] = vis.u[ip]
            v[count] = vis.v[ip]
            # we save in 'u' and 'v' the u and v coordinates of the first frequency that corresponds
            # to the position (i, j) of the discretization of the (u,v)-plane 'iuarr'

            visib[count] = vis.visibilities[ip]
            weights[count] = amplitude_uncertainty[ip] ** 2.0
            den[count] = 1.0
            iuarr[i, j] = count

            count += 1
        else:
            # Save the sum of the visibilities that correspond to the same position (i, j)
            visib[iuarr[i, j].astype(int)] += vis.visibilities[ip]
            # Save the number of the visibilities that correspond to the same position (i, j)
            den[iuarr[i, j].astype(int)] += 1.0
            # Save the sum of the variances of the amplitudes of the visibilities that
            # correspond to the same position (i, j)
            weights[iuarr[i, j].astype(int)] += amplitude_uncertainty[ip] ** 2

    u = u[:count]
    v = v[:count]
    visib = visib[:count]
    den = den[:count]

    # computation of the mean value of the visibilities that correspond to the same
    # position in the discretization of the (u,v)-plane
    visib = visib / den
    # computation of the mean value of the standard deviation of the visibilities that
    # correspond to the same position in the discretization of the (u,v)-plane
    weights = np.sqrt(weights[:count]) / den

    return SimpleNamespace(u=u, v=v, visibilities=visib, amplitude_uncertainty=weights)


def _proximal_entropy(y, m, lamba, Lip, tol=10**-10):
    r"""
    This function computes the value of the proximity operator of the entropy function subject to
    positivity constraint, i.e. it solves the problem

                 argmin_x 1/2*|| y-x ||^2 + \lambda/Lip * H(x)
                 subject to x >= 0

    Actually, this problem can be reduced to finding the zero of the gradient of the objective
    function and it is therefore solved by means of a bisection method.

    Parameters
    ----------
    y
    m
    lamba
    Lip
    tol

    Returns
    -------

    """
    # INITIALIZATION OF THE BISECTION METHOD
    # TODO where does this number come from
    a = np.full_like(y, 1e-24 * y.unit)
    b = np.where(y > m, y, m)

    while np.max(b - a) > tol * y.unit:
        c = (a + b) / 2
        f_c = c - y + lamba / Lip * np.log(c / m)

        tmp1 = f_c <= 0
        tmp2 = f_c >= 0

        a[tmp1] = c[tmp1]
        b[tmp2] = c[tmp2]

    c = (a + b) / 2
    return c


def _proximal_operator(z, f, m, lamb, Lip, niter=250):
    r"""
    Computes the value of the proximity operator of the entropy function subject to
    positivity constraint and flux constraint by means of a Dykstra-like proximal algorithm
    (see Combettes, Pesquet, "Proximal Splitting Methods in Signal Processing", (2011)).
    The problem to solve is:

                       argmin_x 1/2*|| x - y ||^2 + \lambda/Lip * H(x)

    subject to positivity constraint and flux constraint.

    Parameters
    ----------
    z
    f
    m
    lamb
    Lip
    niter

    Returns
    -------

    """

    # INITIALIZATION OF THE DYKSTRA - LIKE SPLITTING
    x = z[:]
    p = np.zeros_like(x)
    q = np.zeros_like(x)

    for i in range(niter):
        tmp = x + p
        # Projection on the hyperplane that represents the flux constraint
        y = tmp + (f - tmp.sum()) / tmp.size
        p = x + p - y

        x = _proximal_entropy(y + q, m, lamb, Lip)

        if np.abs(x.sum() - f) <= 0.01 * f:
            break

        q = y + q - x

    return x, i


def _optimise_fb(Hv, Visib, Lip, flux, lambd, shape, pixel, maxiter, tol):
    r"""
    Solve the optimization problem using a forward-backward splitting algorithm

    .. math::

        \underset{x}{\mathrm{argmin}} \quad \chi^{2}(x) + \lambda H(x)

    subject to positivity constraint and flux constraint. Where :math:`x` is the image to
    reconstruct, :math:`\lambda` is the regularization parameter and :math:`H(x)` is the entropy of
    the image.

    The algorithm implemented is a forward-backward splitting algorithm
    (see Combettes,  Pesquet, "Proximal Splitting Methods in Signal Processing" (2011) and
    Beck, Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising
    and Deblurring Problems" (2009)).

    Parameters
    ----------
    Hv :
        Fourier matrix used to calculate the visibilities of the photon flux
        (actually, Hv = [Re(F); Im(F)] where F is the complex Fourier matrix)
    Visib :

    Lip :
        Lipschitz constant of the gradient of the \chi^2 function
    flux :
        total flux of the image
    lambd :
        regularization parameter
    shape :
        Image size
    pixel :
        Pixel size
    maxiter :
        Maximum number of iterations
    tol :
        Tolerance value used in the stopping rule ( || x - x_old || <= tol || x_old ||)
    Returns
    -------
    MEM Image
    """

    # 'f': value of the total flux of the image (taking into account the area of the pixel)
    f = flux / (pixel[0] * pixel[1])
    # 'm': total flux divided by the number of pixels of the image
    m = f / np.prod(shape.to_value("pix"))

    # INITIALIZATION

    # 'x': constant image with total flux equal to 'f'
    x = np.ones(shape.to_value("pix").astype(int)) + 1.0
    x = x / x.sum() * f
    z = x
    t = 1.0

    # COMPUTATION OF THE OBJECTIVE FUNCTION 'J'

    tmp = x.flatten()[:]
    Hvx = np.matmul(Hv, tmp)
    f_R = _get_entropy(x, m)

    diff_V = Hvx - Visib
    f_0 = (diff_V**2).sum()
    J = f_0 + lambd * f_R

    n_iterations = 0  # number of iterations done in the proximal steps to update the minimizer
    for i in range(maxiter):
        J_old = np.copy(J)
        x_old = np.copy(x)
        t_old = np.copy(t)

        # GRADIENT STEP
        grad = 2 * np.matmul((np.matmul(Hv, z.flatten()) - Visib), Hv).reshape(*shape.to_value("pix").astype(int))
        y = z - 1 / Lip * grad

        # PROXIMAL STEP
        p, pi = _proximal_operator(y, f, m, lambd, Lip)

        # COMPUTATION OF THE OBJECTIVE FUNCTION 'Jp' IN 'p'
        tmp = p.flatten()
        Hvp = np.matmul(Hv, tmp)
        f_Rp = _get_entropy(p, m)

        diff_Vp = Hvp - Visib
        f_0 = (diff_Vp**2).sum()
        Jp = f_0 + lambd * f_Rp

        # CHECK OF THE MONOTONICITY
        # we update 'x' only if 'Jp' is less than or equal to 'J_old'
        check = True
        if Jp > J_old:
            x[:] = x_old
            J = J_old
            check = False
            n_iterations += pi
        else:
            x[:] = p
            J = Jp
            n_iterations = 0

        if n_iterations >= 500:
            break  # if the number of iterations done to update 'x' is too big, then break

        # ACCELERATION

        t = (1 + np.sqrt(1.0 + 4.0 * t_old**2.0)) / 2.0
        tau = (t_old - 1.0) / t
        z = x + tau * (x - x_old) + (t_old / t) * (p - x)

        if i % 25 == 0:
            logger.info(f"Iter: {i}, Obj function: {J}")

        if check and (np.sqrt(((x - x_old) ** 2.0).sum()) < tol * np.sqrt((x_old**2).sum())):
            break

    return x_old


def resistant_mean(data, sigma_cut):
    r"""
    Return a resistant mean

    Remove outliers using the median and the median absolute deviation. An approximation formula is
    used to correct for the truncation caused by removing outliers

    Parameters
    ----------
    data : `numpy.ndarray`
        Data
    sigma_cut : float
        Cutoff interms of sigma

    Returns
    -------
    `tuple`
        Resistant mean and standard deviation
    """
    mad_scale = 0.6745
    mad_scale2 = 0.8
    mad_lim = 1e-24
    sig_coeff = [-0.15405, +0.90723, -0.23584, +0.020142]
    median = np.median(data)
    abs_deviation = np.abs(data - median)
    median_abs_deviation = np.median(abs_deviation) / mad_scale
    if median_abs_deviation < mad_lim:
        median_abs_deviation = np.mean(abs_deviation) / mad_scale2

    cutoff = sigma_cut * median_abs_deviation
    good_index = np.where(abs_deviation <= cutoff)
    if not good_index[0].size > 0:
        raise ValueError("Unable to compute mean")

    good_points = data[good_index]
    mean = np.mean(good_points)
    sigma = np.sqrt((((good_points - mean) ** 2).sum()) / good_points.size)

    # Compensate Sigma for truncation (formula by HF):
    if sigma_cut <= 4.50:
        sigma = sigma / np.polyval(sig_coeff[::-1], sigma_cut)

    cutoff = sigma_cut * sigma

    good_index = np.where(abs_deviation <= cutoff)
    good_points = data[good_index]

    mean = np.mean(good_points)
    sigma = np.sqrt((((good_points - mean) ** 2).sum()) / good_points.size)

    if sigma_cut <= 4.50:
        sigma = sigma / np.polyval(sig_coeff[::-1], sigma_cut)

    # Now the standard deviation of the mean:
    sigma = sigma / np.sqrt(good_points.size - 1)

    return mean, sigma


@apu.quantity_input
def mem(
    vis: Visibilities,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    *,
    percent_lambda: Optional[Quantity[apu.percent]] = 0.02 * apu.percent,
    maxiter: int = 1000,
    tol: float = 1e-3,
    map: bool = True,
) -> Union[Quantity, NDArray[np.float64]]:
    r"""
    Maximum Entropy Method visibility based image reconstruction

    Parameters
    ----------
    vis :
        Input Visibilities
    shape :
        Image size
    pixel_size :
        Pixel size
    percent_lambda
        Value used to compute the regularization parameter as a percentage of a maximum value
        automatically overestimated by the algorithm. Must be in the range [0.0001,0.2]
    maxiter :
        Maximum number of iterations of the optimisation loop
    tol : float
        tolerance value used in the stopping rule ( || x - x_old || <= tol || x_old ||)
    map :
        Return a sunpy map or bare array
    Returns
    -------

    """
    total_flux = _estimate_flux(vis, shape, pixel_size)

    mean_vis = _get_mean_visibilities(vis, shape, pixel_size)
    Hv, Lip, Visib = _prepare_for_optimise(pixel_size, shape, mean_vis)

    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value("pix").astype(int)).flatten()

    # COMPUTATION OF THE; OBJECTIVE; FUNCTION; 'chi2'
    x = x + total_flux / (shape[0] * shape[1] * pixel_size[0] * pixel_size[1]).value
    Hvx = np.matmul(Hv, x)

    lambd = 2 * np.abs(np.matmul((Hvx.value - Visib), Hv)).max() * percent_lambda

    im = _optimise_fb(Hv, Visib, Lip, total_flux, lambd, shape, pixel_size, maxiter, tol)

    # This is needed to match IDL output - prob array vs cartesian indexing
    im = im.T

    if map:
        header = generate_header(vis, shape=shape, pixel_size=pixel_size)
        return Map((im, header))
    return im
