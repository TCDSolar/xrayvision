import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import RBFInterpolator

from xrayvision.visibility import Visibilities


def uv_smooth(vis: Visibilities, niter: int = 50):
    pixel = 0.0005
    detmin = min(vis.meta["isc"])
    if detmin == 0:
        # fov = 0.45
        pixel = pixel * 2.0
        N = 450

    if detmin == 1:
        # fov = 0.26
        pixel = pixel * 2.0
        N = 260

    if detmin >= 2:
        # fov = 0.16
        N = 320

    r = np.sqrt(vis.u**2 + vis.v**2).value

    # Construct new u`, v` grid to interpolate
    Ulimit = (N / 2 - 1) * pixel + pixel / 2
    usampl = -Ulimit + np.arange(N) * pixel
    vsampl = usampl
    uu, vv = np.meshgrid(usampl, vsampl)

    # Interpolate real and imag components onto new grid
    uv_obs = np.vstack([vis.u, vis.v]).T
    uv_samp = np.vstack([uu.flatten(), vv.flatten()]).T
    interpolator = RBFInterpolator(uv_obs, vis.visibilities.real / (4 * np.pi**2))
    real_interp = interpolator(uv_samp)
    interpolator = RBFInterpolator(uv_obs, vis.visibilities.imag / (4 * np.pi**2))
    imag_interp = interpolator(uv_samp)
    vis_interp = (real_interp + 1j * imag_interp).reshape((N, N))

    # Set any component outside the original uv sampling to 0
    vis_interp[np.sqrt(uu**2 + vv**2) > r.max()] = 0j

    # fov = 0.96
    # Define new grid to zero pad the visibilities
    Nnew = 1920
    Ulimit = (Nnew / 2 - 1) * pixel + pixel / 2
    xpix = -Ulimit + np.arange(Nnew) * pixel
    ypix = xpix

    # use np.pad ?
    intzpadd = np.zeros((Nnew, Nnew), dtype=np.complex128)
    intzpadd[(Nnew - N) // 2 : (Nnew - N) // 2 + N, (Nnew - N) // 2 : (Nnew - N) // 2 + N] = vis_interp

    im_new = Nnew // 15
    intznew = np.zeros((im_new, im_new), dtype=np.complex128)
    xpixnew = np.zeros(im_new)
    ypixnew = np.zeros(im_new)

    for i in range(im_new):
        xpixnew[i] = xpix[15 * i + 7]
        ypixnew[i] = ypix[15 * i + 7]
        for j in range(im_new):
            intznew[i, j] = intzpadd[15 * i + 7, 15 * j + 7]

    # xx, yy = np.meshgrid(np.arange(im_new), np.arange(im_new), indexing="xy")
    # intznew = intzpadd[(15 * xx) + 7, (15 * yy) + 7]
    # xpixnew = xpix[(15 * xx[0,:]) + 7]
    # ypixnew = ypix[(15 * yy[:,0]) + 7]

    # OMEGA = (xpixnew[im_new - 1] - xpixnew[0]) / 2
    # X = im_new // (4 * OMEGA)
    deltaomega = (xpixnew[im_new - 1] - xpixnew[0]) / im_new
    # deltax = 2 * X / im_new

    g = intznew[:, :]
    # fftInverse = 4 * np.pi**2 * deltaomega * deltaomega * fft2(ifftshift(intznew)).real
    # fftInverse = fftshift(fftInverse)

    # Characteristic function of disk
    chi = np.zeros((im_new, im_new), dtype=np.complex128)
    chi[np.sqrt(xpixnew.reshape(-1, 1) ** 2 + ypixnew.reshape(1, -1) ** 2) < r.max()] = 1 + 0j

    # Landweber iterations
    map_actual = np.zeros((im_new, im_new), dtype=np.complex128)
    map_iteration = np.zeros((niter, im_new, im_new))
    map_solution = np.zeros((im_new, im_new))

    # relaxation parameter
    tau = 0.2

    descent = np.zeros(niter - 1)
    normAf_g = np.zeros(niter)

    # iteration 0: Inverse Fourier Transform of the initial solution
    map_shifted = ifftshift(map_actual)
    F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi**2 * deltaomega * deltaomega)
    F_Trasf = fftshift(F_Trasf_shifted)

    for iter in range(niter):
        # Update rule
        F_Trasf_up = F_Trasf + tau * (g - chi * F_Trasf)
        F_Trasf = F_Trasf_up

        # FFT of the updated soln
        F_Trasf_shifted = ifftshift(F_Trasf)
        map_shifted = fft2(F_Trasf_shifted) / (im_new**2) * 4 * np.pi**2 * deltaomega * deltaomega
        map_actual = fftshift(map_shifted)

        # Project solution onto subset of positive solutions (positivity constraint)
        map_actual[map_actual < 0] = 0 + 0j
        map_iteration[iter, :, :] = map_actual.real

        map_shifted = ifftshift(map_actual)
        F_Trasf_shifted = ifft2(map_shifted) * (im_new**2) / (4 * np.pi**2 * deltaomega * deltaomega)
        F_Trasf = fftshift(F_Trasf_shifted)

        # Stopping criterion based on the descent of ||Af - g||
        Af_g = chi * F_Trasf - g
        normAf_g[iter] = np.sqrt(np.sum(np.abs(Af_g) * np.abs(Af_g)))
        if iter >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                descent[iter - 1] = (normAf_g[iter - 1] - normAf_g[iter]) / normAf_g[iter - 1]
            if descent[iter - 1] < 0.02:
                break

    # output
    F_Trasf = 4 * np.pi**2 * F_Trasf

    if iter == niter:
        map_solution[:, :] = map_iteration[14, :, :].real
    else:
        map_solution = map_actual.real

    map_solution = map_solution * im_new**2

    return map_solution, F_Trasf
