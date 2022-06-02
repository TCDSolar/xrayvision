r"""
Implementation of Maximum Entropy Method
"""

from types import SimpleNamespace

import numpy as np
import astropy.units as apu
from numpy.linalg import norm

from xrayvision.transform import generate_xy
from xrayvision.utils import get_logger

__all__ = ['get_entropy', 'get_fourier_matrix', 'estimate_flux', 'get_mean_visibilities',
           'proximal_entropy', 'proximal_operator', 'optimise_fb', 'mem']


logger = get_logger(__name__, 'DEBUG')


def get_entropy(image, flux):
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
    return np.sum(image*np.log(image/(flux*np.e)))


def get_fourier_matrix(vis, shape=[64, 64]*apu.pix, pixel_size=[4.0312500, 4.0312500]*apu.arcsec):
    r"""
    Return the complex Fourier matrix used to compute the value of the visibilities.

    Generate the complex Fourier matrix :math:`Hv` used to compute the value of the
    visibilities. If :math:`\vec{x}` is the vectorised image, then
    :math:`v = \mathbf{\mathit{Hv}} \vec{x}` is the vector containing the complex visibilities

    Parameters
    ----------
    vis :
        Visibly object containg u, v sampling
    shape :
        Shape of the images
    pixel_size
        Size of the pixels

    Returns
    -------
    The complex Fourier matrix
    """
    m, n = shape.to_value('pix')
    y = generate_xy(m, 0 * apu.arcsec, pixel_size[1])
    x = generate_xy(n, 0 * apu.arcsec, pixel_size[0])
    x, y = np.meshgrid(x, y)
    # Check apu are correct for exp need to be dimensionless and then remove apu for speed
    if (vis.uv[0, :] * x[0, 0]).unit == apu.dimensionless_unscaled and \
            (vis.uv[1, :] * y[0, 0]).unit == apu.dimensionless_unscaled:
        uv = vis.uv.value
        x = x.value
        y = y.value

        Hv = np.exp(1j * 2 * np.pi * (x[..., np.newaxis] * uv[np.newaxis, 0, :]
                                      + y[..., np.newaxis] * uv[np.newaxis, 1, :]))

        return Hv * pixel_size[0] * pixel_size[1]


def estimate_flux(vis, shape, pixel, maxiter=1000, tol=1e-3):
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
    Esimated total flux

    """

    Hv, Lip, Visib = _prepare_for_optimise(pixel, shape, vis)

    # PROJECTED LANDWEBER
    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value('pix').astype(int)).flatten()

    for i in range(maxiter):
        x_old = x[:]

        # GRADIENT STEP
        grad = 2. * np.matmul((np.matmul(Hv,  x).value - Visib).T, Hv)
        y = x - 1. / Lip * grad

        # PROJECTION ON THE POSITIVE ORTHANT
        x = y.clip(min=0.0)
        tmp = x[:]
        Hvx = np.matmul(Hv, tmp)

        diff_V = Hvx - Visib
        chi2 = (diff_V ** 2.).sum()

        logger.info(f'Iter: {i}, Chi2: {chi2}')
        if np.sqrt(((x - x_old) ** 2.).sum()) < tol * np.sqrt((x_old ** 2.).sum()):
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
    Hv = get_fourier_matrix(vis, shape, pixel)
    # Division of real and imaginary part of the matrix 'Hv'
    ReHv = np.real(Hv)
    ImHv = np.imag(Hv)
    # 'Hv' is composed by the union of its real and imaginary part
    Hv = np.concatenate([ReHv, ImHv], axis=-1)
    # Division of real and imaginary part of the visibilities
    ReV = np.real(vis.vis)
    ImV = np.imag(vis.vis)
    # 'Visib' is composed by the real and imaginary part of the visibilities
    Visib = np.concatenate([ReV, ImV], axis=-1)
    # Standard deviation of the real and imaginary part of the visibilities
    sigma_Re = vis.amplitude_error
    sigma_Im = vis.amplitude_error
    # 'sigma': standard deviation of the data contained in 'Visib'
    sigma = np.concatenate([sigma_Re, sigma_Im], axis=-1)
    # RESCALING of 'Hv' AND 'Visib'(NEEDED FOR COMPUTING THE VALUE OF THE \chi ** 2; FUNCTION)
    # The vector 'Visib' and every column of 'Hv' are divided by 'sigma'
    Visib = Visib / sigma
    ones = np.ones(shape.to_value('pix').astype(int))
    sigma1 = sigma * ones[..., np.newaxis]
    Hv = Hv / sigma1
    # COMPUTATION OF THE LIPSCHITZ CONSTANT; 'Lip' OF THE GRADIENT OF  THE \chi ** 2
    # FUNCTION (NEEDED TO GUARANTEE THE CONVERGENCE OF THE ALGORITHM)
    Hv = Hv.transpose(2, 0, 1).reshape(Hv.shape[2], -1)
    HvHvT = np.matmul(Hv, Hv.T)
    # TODO magic number
    Lip = 2.1 * norm(HvHvT, 2)
    return Hv, Lip, Visib


def get_mean_visibilities(vis, shape, pixel):
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

    imsize2 = shape[0] / 2
    pixel_size = 1/(shape[0]*pixel[0])

    iu = vis.uv[0, :] / pixel_size
    iv = vis.uv[1, :] / pixel_size
    ru = np.around(iu)
    rv = np.around(iv)

    # index of the u coordinates of the sampling frequencies in the discretization of the u axis
    ru = ru + imsize2
    # index of the v coordinates of the sampling frequencies in the discretization of the v axis
    rv = rv + imsize2

    # matrix that represents the discretization of the (u,v)-plane
    iuarr = np.zeros(shape.to_value('pixel').astype(int))

    count = 0
    n_vis = vis.uv.shape[1]
    u = np.zeros(n_vis) * (1 / apu.arcsec)
    v = np.zeros(n_vis) * (1 / apu.arcsec)
    den = np.zeros(n_vis)
    weights = np.zeros_like(vis.amplitude_error**2)
    visib = np.zeros_like(vis.vis)
    for ip in range(n_vis):
        # what about 0.5 pix offset
        i = ru[ip].to_value('pix').astype(int)
        j = rv[ip].to_value('pix').astype(int)
        # (i, j) is the position of the spatial frequency in
        # the discretization of the (u,v)-plane 'iuarr'
        if iuarr[i, j] == 0.:
            u[count] = vis.uv[0, ip]
            v[count] = vis.uv[1, ip]
            # we save in 'u' and 'v' the u and v coordinates of the first frequency that corresponds
            # to the position (i, j) of the discretization of the (u,v)-plane 'iuarr'

            visib[count] = vis.vis[ip]
            weights[count] = vis.amplitude_error[ip]**2.
            den[count] = 1.
            iuarr[i, j] = count

            count += 1
        else:
            # Save the sum of the visibilities that correspond to the same position (i, j)
            visib[iuarr[i, j].astype(int)] += vis.vis[ip]
            # Save the number of the visibilities that correspond to the same position (i, j)
            den[iuarr[i, j].astype(int)] += 1.
            # Save the sum of the variances of the amplitudes of the visibilities that
            # correspond to the same position (i, j)
            weights[iuarr[i, j].astype(int)] += vis.amplitude_error[ip]**2

    u = u[:count]
    v = v[:count]
    visib = visib[:count]
    den = den[:count]

    # computation of the mean value of the visibilities that correspond to the same
    # position in the discretization of the (u,v)-plane
    visib = visib/den
    # computation of the mean value of the standard deviation of the visibilities that
    # correspond to the same position in the discretization of the (u,v)-plane
    weights = np.sqrt(weights[:count])/den

    return SimpleNamespace(uv=np.vstack((u, v)), vis=visib, amplitude_error=weights)


def proximal_entropy(y, m, lamba, Lip, tol=10**-10):
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
    a = np.full_like(y, 1e-24*y.unit)
    b = np.where(y > m, y, m)

    while np.max(b - a) > tol*y.unit:
        c = (a + b) / 2
        f_c = c - y + lamba / Lip * np.log(c / m)

        tmp1 = f_c <= 0
        tmp2 = f_c >= 0

        a[tmp1] = c[tmp1]
        b[tmp2] = c[tmp2]

    c = (a + b) / 2
    return c


def proximal_operator(z, f, m, lamb, Lip, niter=250):
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

        x = proximal_entropy(y + q, m, lamb, Lip)

        if np.abs(x.sum() - f) <= 0.01 * f:
            break

        q = y + q - x

    return x, i


def optimise_fb(Hv, Visib, Lip, flux, lambd, shape, pixel, maxiter, tol):
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
    m = f / np.prod(shape.to_value('pix'))

    # INITIALIZATION

    # 'x': constant image with total flux equal to 'f'
    x = np.ones(shape.to_value('pix').astype(int)) + 1.
    x = x / x.sum() * f
    z = x
    t = 1.0

    # COMPUTATION OF THE OBJECTIVE FUNCTION 'J'

    tmp = x.flatten()[:]
    Hvx = np.matmul(Hv, tmp)
    f_R = get_entropy(x, m)

    diff_V = Hvx - Visib
    f_0 = (diff_V**2).sum()
    J = f_0 + lambd * f_R

    n_iterations = 0  # number of iterations done in the proximal steps to update the minimizer
    for i in range(maxiter):

        J_old = np.copy(J)
        x_old = np.copy(x)
        t_old = np.copy(t)

        # GRADIENT STEP
        grad = 2 * np.matmul((np.matmul(Hv, z.flatten()) - Visib),
                             Hv).reshape(*shape.to_value('pix').astype(int))
        y = z - 1 / Lip * grad

        # PROXIMAL STEP
        p, pi = proximal_operator(y, f, m, lambd, Lip)

        # COMPUTATION OF THE OBJECTIVE FUNCTION 'Jp' IN 'p'
        tmp = p.flatten()
        Hvp = np.matmul(Hv, tmp)
        f_Rp = get_entropy(p, m)

        diff_Vp = Hvp - Visib
        f_0 = (diff_Vp**2).sum()
        Jp = f_0 + lambd * f_Rp

        # CHEK OF THE MONOTONICITY
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

        t = (1 + np.sqrt(1. + 4. * t_old ** 2.)) / 2.
        tau = (t_old - 1.) / t
        z = x + tau * (x - x_old) + (t_old / t) * (p - x)

        logger.info(f'Iter: {i}, Obj function: {J}')

        if check and (np.sqrt(((x - x_old)**2.).sum()) < tol * np.sqrt((x_old**2).sum())):
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
        raise ValueError('Unable to compute mean')

    good_points = data[good_index]
    mean = np.mean(good_points)
    sigma = np.sqrt((((good_points - mean)**2).sum()) / good_points.size)

    # Compensate Sigma for truncation (formula by HF):
    if cutoff.max() <= 4.50:
        sigma = sigma / np.poly(cutoff.max(), sig_coeff)

    cutoff = sigma_cut * sigma

    good_index = np.where(abs_deviation <= cutoff)
    good_points = data[good_index]

    mean = np.mean(good_points)
    sigma = np.sqrt((((good_points - mean)**2).sum()) / good_points.size)
    # Now the standard deviation of the mean:
    sigma = sigma / np.sqrt(good_points.size - 1)

    return mean, sigma


def get_percent_lambda(vis):
    r"""
    Return 'percent_lambda' use with MEM

    Calculate the signal-to-noise ratio (SNR) for the given visibility bag by trying to use the
    coarser sub-collimators adding finer ones until there are at least 2 visiblities, then use
    resistant mean of of abs(obsvis) / sigamp

    Parameters
    ----------
    vis

    Returns
    -------

    """
    # Loop through ISCs starting with 3-10, but if we don't have at least 2 vis, lower isc_min to
    # include next one down, etc.

    # TODO this start at 3 not 10?
    isc_min = 3
    nbig = 0
    isc_sizes = np.array([float(s[:-1]) for s in vis.label])
    while isc_min >= 0 and nbig < 2:
        ibig = np.argwhere(isc_sizes >= isc_min)
        isc_min = isc_min - 1

    # If still don't have at least 2 vis, return -1, otherwise calculate mean
    # (but reject points > sigma away from mean)
    if ibig.size < 2:
        snr_value = -1
    else:
        snr_value, _ = resistant_mean(
            (np.abs(vis.vis[ibig])/vis.amplitude_error[ibig]).flatten(), 3)

    # TODO magic numbers
    percent_lambda = 2 / (snr_value**2 + 90)

    return percent_lambda


def mem(vis, percent_lambda=None, shape=None, pixel=None, maxiter=1000, tol=1e-3):
    r"""
    Maximum Entropy Method for visibility based image reconstruction

    Parameters
    ----------
    vis :
        Input Visibilities
    percent_lambda
        value used to compute the regularization parameter as a percentage of a maximum value
        automatically overestimated by the algorithm. Must be in the range [0.0001,0.2]
    shape
        Image size
    pixel
        Pixel size
    maxiter : int
        Maximum number of iterations of the optimization loop
    tol : float
        tolerance value used in the stopping rule ( || x - x_old || <= tol || x_old ||)

    Returns
    -------

    """
    total_flux = estimate_flux(vis, (64, 64) * apu.pix, [4.0312500, 4.0312500] * apu.arcsec)
    if percent_lambda is None:
        percent_lambda = get_percent_lambda(vis)

    mean_vis = get_mean_visibilities(vis, shape, pixel)
    Hv, Lip, Visib = _prepare_for_optimise(pixel, shape, mean_vis)

    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value('pix').astype(int)).flatten()

    # COMPUTATION OF THE; OBJECTIVE; FUNCTION; 'chi2'
    x = x + total_flux/(shape[0]*shape[1]*pixel[0]*pixel[1]).value
    Hvx = np.matmul(Hv, x)

    lambd = 2 * np.abs((np.matmul((Hvx.value - Visib), Hv))).max()*percent_lambda

    im = optimise_fb(Hv, Visib, Lip, total_flux, lambd, shape, pixel, maxiter, tol)
    return im
