import logging
from types import SimpleNamespace

import numpy as np
import astropy.units as apu
from numpy.linalg import norm

from xrayvision.transform import generate_xy


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def get_entropy(image, flux):
    return np.sum(image*np.log(image/(flux*np.e)))


def get_fourier_matrix(vis, shape=[64, 64]*apu.pix, pixel_size=[4.0312500, 4.0312500]*apu.arcsec):

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
    sigma1 = sigma*ones[...,np.newaxis]
    Hv = Hv / sigma1

    # COMPUTATION OF THE LIPSCHITZ CONSTANT; 'Lip' OF THE GRADIENT OF  THE \chi ** 2
    # FUNCTION (NEEDED TO GUARANTEE THE CONVERGENCE OF THE ALGORITHM)
    Hv = Hv.transpose(2, 0, 1).reshape(Hv.shape[2], -1)

    HvHvT = np.matmul(Hv, Hv.T)
    # TODO magic number
    Lip = 2.1 * norm(HvHvT, 2)

    # PROJECTED LANDWEBER

    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value('pix').astype(int)).flatten()

    # COMPUTATION OF THE; OBJECTIVE; FUNCTION; 'chi2'
    tmp = x[:]
    Hvx = np.matmul(Hv, tmp)

    diff_V = Hvx.value - Visib.value
    chi2 = np.sum(diff_V ** 2.)

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

        logging.info(f'Iter: {i}, Chi2: {chi2}')
        if np.sqrt(((x - x_old) ** 2.).sum()) < tol * np.sqrt((x_old ** 2.).sum()):
            break

    return x.sum() * pixel[0] * pixel[1]


def get_mean_visib(vis, shape, pixel):

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
    weights= np.zeros_like(vis.amplitude_error**2)
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
            # in 'visib' we save the sum of the visibilities that correspond to the same position (i, j)
            visib[iuarr[i, j].astype(int)] += vis.vis[ip]
            # in 'den' we save the number of the visibilities that correspond to the same position (i, j)
            den[iuarr[i, j].astype(int)] += 1.
            # in 'wgtarr' we save the sum of the variances of the amplitudes of the visibilities that
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

    return SimpleNamespace(uv=np.vstack((u, v)), vis=visib, weight=weights)


def proximal_entropy(y, m, lamba, Lip, tol=10**-10):
    # INITIALIZATION OF THE BISECTION METHOD
    # TODO where does this number come from
    a = np.full_like(y, 1e-24*y.unit)
    b = np.where(y > m, y, m)

    while np.max(b - a) > tol*y.unit:
        c = (a + b) / 2
        f_c = c - y + lamba /Lip * np.log(c / m)

        tmp1 = f_c <= 0
        tmp2 = f_c >= 0

        a[tmp1] = c[tmp1]
        b[tmp2] = c[tmp2]

    c = (a + b) / 2
    return c


def proximal_operator(z, f, m, lamb, Lip, niter=250):
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


def mem(vis, percent_lambda=0.1, shape=None, pixel=None, maxiter=1000, tol=1e-3):
    mean_vis = get_mean_visib(vis, shape, pixel)

    total_flux = estimate_flux(vis, (64, 64) * apu.pix, [4.0312500, 4.0312500] * apu.arcsec)

    Hv = get_fourier_matrix(mean_vis, shape, pixel)
    # Division of real and imaginary part of the matrix 'Hv'
    ReHv = np.real(Hv)
    ImHv = np.imag(Hv)
    # 'Hv' is composed by the union of its real and imaginary part
    Hv = np.concatenate([ReHv, ImHv], axis=-1)

    # Division of real and imaginary part of the visibilities
    ReV = np.real(mean_vis.vis)
    ImV = np.imag(mean_vis.vis)
    # 'Visib' is composed by the real and imaginary part of the visibilities
    Visib = np.concatenate([ReV, ImV], axis=-1)

    # Standard deviation of the real and imaginary part of the visibilities
    sigma_Re = mean_vis.weight
    sigma_Im = mean_vis.weight

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

    # should have same apu as image ct/ keV cm s arcsec**2
    x = np.zeros(shape.to_value('pix').astype(int)).flatten()

    # COMPUTATION OF THE; OBJECTIVE; FUNCTION; 'chi2'
    x = x + total_flux/(shape[0]*shape[1]*pixel[0]*pixel[1]).value
    Hvx = np.matmul(Hv, x)

    # TODO cacluate base on signal to noise e.g default, percent_lambda, stx_mem_ge_percent_lambda(stx_vis_get_snr(vis))
    lambd = 2 * np.abs((np.matmul((Hvx.value - Visib), Hv))).max()*0.010658418

    im = optimise_fb(Hv, Visib, Lip, total_flux, lambd, shape, pixel, maxiter, tol)
    return im
