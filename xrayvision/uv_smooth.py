import logging

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import RBFInterpolator

from xrayvision.visibility import Visibilities

__all__ = ["uv_smooth"]

logger = logging.getLogger(__name__)


def uv_smooth(vis: Visibilities, niter: int = 50):
    r"""
    UV-smoothing imaging algorithm.

    Parameters
    ----------
    vis :
        Input visibilities.
    niter :
        Maximum number of iterations.

    Returns
    -------

    """

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
        map_shifted = fft2(F_Trasf_shifted) * 4 * np.pi**2 * deltaomega * deltaomega
        map_actual = fftshift(map_shifted)

        # Project solution onto subset of positive solutions (positivity constraint)
        map_actual[map_actual.real < 0] = 0 + 0j
        map_iteration[iter, :, :] = map_actual.real

        map_shifted = ifftshift(map_actual)
        F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi**2 * deltaomega * deltaomega)
        F_Trasf = fftshift(F_Trasf_shifted)

        # Stopping criterion based on the descent of ||Af - g||
        Af_g = chi * F_Trasf - g
        normAf_g[iter] = np.sqrt(np.sum(np.abs(Af_g) * np.abs(Af_g)))
        if iter >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                descent[iter - 1] = (normAf_g[iter - 1] - normAf_g[iter]) / normAf_g[iter - 1]
            if descent[iter - 1] < 0.02:
                logger.info("Converge at iteration %d", iter)
                break
    else:
        logger.info("Max iterations reached %d", iter)

    # output
    F_Trasf = 4 * np.pi**2 * F_Trasf

    if iter == niter - 1:
        map_solution[:, :] = map_iteration[14, :, :].real
    else:
        map_solution = map_actual.real

    return map_solution, F_Trasf


def uv_smooth_flexible(vis: Visibilities, niter: int = 50, pixel_size: float = 1, image_dim: int = 128):
    """
    UV smooth with user-specified image dimensions and pixel size.

    Parameters
    ----------
    vis : Visibilities
        Input visibilities
    niter : int
        Number of Landweber iterations
    pixel_size : float, optional
        Desired pixel size in arcseconds for the output image.
        If None, automatically determined.
    image_dim : int
        Desired output image dimension in pixels (default: 128)

    Returns
    -------
    map_solution : ndarray
        Reconstructed image of size (image_dim, image_dim)
    pixel_size_arcsec : float
        Actual pixel size in arcseconds
    """

    # Calculate maximum UV radius
    # r = np.sqrt(vis.u**2 + vis.v**2).value
    # r_max = r.max()

    # Determine UV pixel size
    if pixel_size is not None:
        # Convert arcsec to arcsec^-1 for UV space
        # For an image of size L arcsec, the UV sampling is 1/L arcsec^-1
        fov_arcsec = pixel_size * image_dim
        uv_pixel = 1.0 / fov_arcsec
    else:
        # Use default based on detector
        uv_pixel = 0.0005
        detmin = min(vis.meta["isc"])
        if detmin <= 1:
            uv_pixel = uv_pixel * 2.0

    # Call the main function with appropriate parameters
    map_solution, F_Trasf, xpix, ypix = uv_smooth_new(vis, niter=niter, pixel_size=uv_pixel, shape=image_dim)

    # Calculate actual pixel size
    fov_uv = xpix[-1] - xpix[0]
    pixel_size_arcsec = 1.0 / fov_uv

    return map_solution, pixel_size_arcsec


def uv_smooth_new(
    vis: Visibilities, niter: int = 50, pixel_size: float = 1, shape: int = 128, natural_weighting: bool = True
):
    """
    UV smooth algorithm with flexible image dimensions.

    Parameters
    ----------
    vis : Visibilities
        Input visibilities
    niter : int
        Number of Landweber iterations (default: 50)
    pixel_size : float, optional
        Pixel size in UV space (arcsec^-1). If None, automatically determined
        from detector configuration.
    shape : int, optional
        Final image dimension in pixels. If None, automatically determined
        from detector configuration.
    natural_weighting : bool
        Apply natural weighting based on UV coverage (default: True)

    Returns
    -------
    map_solution : ndarray
        Reconstructed image
    F_Trasf : ndarray
        Final Fourier transform
    """

    # Determine pixel size if not provided
    if pixel_size is None:
        pixel_size = 0.0005
        detmin = min(vis.meta["isc"])
        if detmin <= 1:
            pixel_size = pixel_size * 2.0

    # Calculate maximum UV radius from the data
    r = np.sqrt(vis.u**2 + vis.v**2).value
    r_max = r.max()

    # Determine appropriate grid size for interpolation
    # This should be large enough to capture the UV coverage
    # Rule of thumb: make sure pixel * N/2 > r_max
    if shape is None:
        # Auto-determine based on detector configuration
        detmin = min(vis.meta["isc"])
        if detmin == 0:
            N = 450
        elif detmin == 1:
            N = 260
        else:
            N = 320
    else:
        # Calculate N to ensure we cover the UV space properly
        # N should be at least 2 * r_max / pixel
        N = int(np.ceil(2 * r_max / pixel_size))
        # Make it even for FFT efficiency
        if N % 2 == 1:
            N += 1
        # Ensure minimum size
        N = max(N, 64)

    # Construct new u, v grid to interpolate
    # CRITICAL: Ulimit must accommodate r_max
    Ulimit = (N / 2 - 1) * pixel_size + pixel_size / 2

    # Check if our grid is large enough
    if Ulimit < r_max:
        # Increase N to accommodate the data
        N = int(np.ceil(2 * r_max / pixel_size)) + 2
        if N % 2 == 1:
            N += 1
        Ulimit = (N / 2 - 1) * pixel_size + pixel_size / 2
        print(f"Warning: Grid size increased to N={N} to accommodate r_max={r_max:.6f}")

    usampl = -Ulimit + np.arange(N) * pixel_size
    vsampl = usampl
    uu, vv = np.meshgrid(usampl, vsampl)

    # Interpolate real and imag components onto new grid
    uv_obs = np.vstack([vis.u, vis.v]).T
    uv_samp = np.vstack([uu.flatten(), vv.flatten()]).T

    # Normalize by 4π² for consistency with IDL code
    interpolator = RBFInterpolator(uv_obs, vis.visibilities.real / (4 * np.pi**2))
    real_interp = interpolator(uv_samp)
    interpolator = RBFInterpolator(uv_obs, vis.visibilities.imag / (4 * np.pi**2))
    imag_interp = interpolator(uv_samp)
    vis_interp = (real_interp + 1j * imag_interp).reshape((N, N))

    # CRITICAL: Set any component outside the original uv sampling to 0
    # This prevents sampling regions where we have no data
    uv_radius = np.sqrt(uu**2 + vv**2)
    vis_interp[uv_radius > r_max] = 0j

    # Define new grid to zero pad the visibilities
    # The zero-padding increases the sampling in image space
    # Choose Nnew based on desired oversampling
    oversample_factor = 6  # This gives ~15x downsampling later
    Nnew = N * oversample_factor

    # Make sure Nnew is even
    if Nnew % 2 == 1:
        Nnew += 1

    Ulimit_new = (Nnew / 2 - 1) * pixel_size + pixel_size / 2
    xpix = -Ulimit_new + np.arange(Nnew) * pixel_size
    ypix = xpix

    # Zero pad the visibilities
    intzpadd = np.zeros((Nnew, Nnew), dtype=np.complex128)
    start_idx = (Nnew - N) // 2
    intzpadd[start_idx : start_idx + N, start_idx : start_idx + N] = vis_interp

    # Downsample to get final image size
    # This reduces the pixel size in image space
    downsample_factor = 15
    im_new = Nnew // downsample_factor

    # Ensure im_new is reasonable
    if im_new < 32:
        im_new = 32
        downsample_factor = Nnew // im_new

    intznew = np.zeros((im_new, im_new), dtype=np.complex128)
    xpixnew = np.zeros(im_new)
    ypixnew = np.zeros(im_new)

    for i in range(im_new):
        idx = downsample_factor * i + downsample_factor // 2
        if idx >= Nnew:
            idx = Nnew - 1
        xpixnew[i] = xpix[idx]
        ypixnew[i] = ypix[idx]
        for j in range(im_new):
            jdx = downsample_factor * j + downsample_factor // 2
            if jdx >= Nnew:
                jdx = Nnew - 1
            intznew[i, j] = intzpadd[idx, jdx]

    deltaomega = (xpixnew[im_new - 1] - xpixnew[0]) / im_new

    g = intznew[:, :]

    # Characteristic function of disk
    # CRITICAL: chi should only be 1 where we have UV data coverage
    # This constraint ensures we don't extrapolate beyond measured frequencies
    chi = np.zeros((im_new, im_new), dtype=np.complex128)
    uv_grid_radius = np.sqrt(xpixnew.reshape(-1, 1) ** 2 + ypixnew.reshape(1, -1) ** 2)

    # IMPORTANT: Only set chi=1 where we actually have data
    # Use a slightly smaller radius to avoid edge effects
    chi[uv_grid_radius < (r_max * 0.95)] = 1 + 0j

    if natural_weighting:
        # Apply natural weighting based on density of UV coverage
        # This down-weights regions with sparse sampling
        from scipy.ndimage import gaussian_filter

        # Create a density map of UV coverage
        uv_density = np.zeros((im_new, im_new))
        for u_obs, v_obs in zip(vis.u.value, vis.v.value):
            # Find nearest grid point
            i = np.argmin(np.abs(xpixnew - u_obs))
            j = np.argmin(np.abs(ypixnew - v_obs))
            if i < im_new and j < im_new:
                uv_density[i, j] += 1

        # Smooth the density map
        uv_density_smooth = gaussian_filter(uv_density, sigma=1.0)

        # Normalize and use as weight
        if uv_density_smooth.max() > 0:
            weights = uv_density_smooth / uv_density_smooth.max()
            # Avoid complete zero weighting
            weights = np.clip(weights, 0.1, 1.0)
            chi = chi * weights

    # Landweber iterations
    map_actual = np.zeros((im_new, im_new), dtype=np.complex128)
    map_iteration = np.zeros((niter, im_new, im_new))

    # Relaxation parameter (can be tuned)
    tau = 0.2

    descent = np.zeros(niter - 1)
    normAf_g = np.zeros(niter)

    # Iteration 0: Inverse Fourier Transform of the initial solution
    map_shifted = ifftshift(map_actual)
    F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi**2 * deltaomega * deltaomega)
    F_Trasf = fftshift(F_Trasf_shifted)

    for iter in range(niter):
        # Update rule
        F_Trasf_up = F_Trasf + tau * (g - chi * F_Trasf)
        F_Trasf = F_Trasf_up

        # FFT of the updated solution
        F_Trasf_shifted = ifftshift(F_Trasf)
        map_shifted = fft2(F_Trasf_shifted) / (im_new**2) * 4 * np.pi**2 * deltaomega * deltaomega
        map_actual = fftshift(map_shifted)

        # Project solution onto subset of positive solutions (positivity constraint)
        map_actual[map_actual.real < 0] = 0 + 0j
        map_iteration[iter, :, :] = map_actual.real

        # Back to Fourier space
        map_shifted = ifftshift(map_actual)
        F_Trasf_shifted = ifft2(map_shifted) * (im_new**2) / (4 * np.pi**2 * deltaomega * deltaomega)
        F_Trasf = fftshift(F_Trasf_shifted)

        # Stopping criterion based on the descent of ||Af - g||
        Af_g = chi * F_Trasf - g
        normAf_g[iter] = np.sqrt(np.sum(np.abs(Af_g) * np.abs(Af_g)))

        if iter >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                descent[iter - 1] = (normAf_g[iter - 1] - normAf_g[iter]) / normAf_g[iter - 1]

            logger.debug("Iteration %d: %f", iter, descent[iter - 1])

            if descent[iter - 1] < 0.02:
                print(f"Converged at iteration {iter}")
                break

    # Output
    F_Trasf = 4 * np.pi**2 * F_Trasf

    if iter == niter - 1:
        # If didn't converge, use iteration 14 (arbitrary choice from original)
        if niter > 14:
            map_solution = map_iteration[14, :, :].real
        else:
            map_solution = map_iteration[niter - 1, :, :].real
    else:
        map_solution = map_actual.real

    map_solution = map_solution * im_new**2

    return map_solution, F_Trasf, xpixnew, ypixnew
