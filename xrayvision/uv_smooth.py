import logging

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import RBFInterpolator

from xrayvision.visibility import Visibilities

__all__ = ["uv_smooth", "uv_smooth_new"]

logger = logging.getLogger(__name__)


def uv_smooth(vis: Visibilities, niter: int = 50):
    r"""
    uv_smooth image reconstruction method.

    This method reconstructs images from sparse Fourier-domain visibilities using an iterative Landweber scheme. It
    interpolates visibilities onto a regular uv grid, applies smoothing to stabilize the reconstruction, and
    iteratively refines the image while enforcing positivity.

    Parameters
    ----------
    vis :
        Input visibilities.
    niter :
        Maximum number of iterations.

    References
    ----------
    * :cite:t:`Massone2009_uv_smooth`

    Returns
    -------
        Reconstructed 2D image of the X-ray source.

    Notes
    -----
    UV_smooth solves the ill-posed image reconstruction problem in the uv-plane
    through the following steps:

    1. **Visibility Gridding**: Sparse visibilities `V(u_i, v_i)` are interpolated onto a regular uv grid `V_grid(u,v)`
    using a smoothing kernel `K(u,v)`: `V_grid(u,v) = sum_i V(u_i,v_i) * K(u-u_i, v-v_i)`

    2. **Smoothing / Regularization**: The kernel enforces smoothness in the uv-plane to mitigate noise and compensate
    for sparse coverage.

    3. **Iterative Landweber Update**: The image `I_n(x,y)` at iteration n is updated using the Landweber scheme:
    `I_{n+1} = I_n + λ * F^{-1}[ V_grid - F[I_n] ]`
    where `F` and `F^{-1}` are the Fourier and inverse Fourier transforms, and λ is a relaxation parameter. After
    each iteration, positivity is enforced: `I_{n+1} = max(0, I_{n+1})`

    4. **Stopping Criterion**: Iteration continues until either:
    `|| F[I_{n+1}] - V_grid || < tolerance`
    or the maximum number of iterations `max_iter` is reached. This residual norm ensures that updates stop when the
    reconstruction is consistent with the measured visibilities.
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

    xx, yy = np.meshgrid(np.arange(im_new), np.arange(im_new), indexing="xy")
    intznew = intzpadd[(15 * xx) + 7, (15 * yy) + 7]
    xpixnew = xpix[(15 * xx[0, :]) + 7]
    ypixnew = ypix[(15 * yy[:, 0]) + 7]

    OMEGA = (xpixnew[im_new - 1] - xpixnew[0]) / 2
    X = im_new // (4 * OMEGA)
    deltaomega = (xpixnew[im_new - 1] - xpixnew[0]) / im_new
    deltax = 2 * X / im_new

    g = intznew[:, :]

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
                logger.info("Converged at iteration %d", iter)
                break
    else:
        logger.info("Max iterations reached %d", iter)

    # output
    F_Trasf = 4 * np.pi**2 * F_Trasf

    if iter == niter - 1:
        map_solution[:, :] = map_iteration[14, :, :].real
    else:
        map_solution = map_actual.real

    return map_solution, F_Trasf, deltax


def uv_smooth_new(
    vis: Visibilities, shape=128, pixel_size: float = 1.0, uv_pixel_size: float | None = None, niter: int = 50, **kwargs
):
    r"""
    uv_smooth image reconstruction method.

    This method reconstructs images from sparse Fourier-domain visibilities using an iterative Landweber scheme. It
    interpolates visibilities onto a regular uv grid, applies smoothing to stabilize the reconstruction, and
    iteratively refines the image while enforcing positivity.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of output image
    pixel_size :
        Size of output pixels
    uv_pixel_size :
        Grid spacing in uv-plane (arcsec^-1) **Use with caution a holdover to allow comparison to old RHESSI code**
    niter :
        Maximum number of iterations.

    Returns
    -------
    final_image :
        Reconstructed 2D image of the X-ray source.
    fourier_transform :
        The final visibilities
    pixel_size :
        The output pixel size

    References
    ----------
    * :cite:t:`Massone2009_uv_smooth`

    Notes
    -----
    UV_smooth solves the ill-posed image reconstruction problem from sparse Fourier-domain
    measurements through a four-stage pipeline:

    **Stage 1: Grid Configuration**

    Determines uv pixel size and grid dimensions based on the desired output image shape, pixel size as well as the
    padding and downsampling factors.

    **Stage 2: Visibility Gridding**

    Sparse visibility measurements :math:`V(u_i, v_i)` at irregular UV coordinates are interpolated onto a regular Cartesian
    grid using Radial Basis Function (RBF) interpolation:

    .. math::
        V_{grid}(u,v) = \sum_i V(u_i, v_i) \cdot \phi(\|(u,v) - (u_i,v_i)\|)

    where :math:`\phi` is the RBF kernel. Real and imaginary components are interpolated separately. Points outside the
    maximum observed spatial extent are masked to zero to preserve the UV coverage constraint.

    **Stage 3: Sinc Interpolation to Final Grid**

    The gridded visibilities are resampled to the output reconstruction grid using sinc interpolation, the theoretically
    optimal method for band-limited signals. This is implemented efficiently via:

    1. **Zero-padding** grid in Fourier space
    2. **Centered downsampling** by factor obtain output grid

    This combination is mathematically equivalent to sinc interpolation:

    .. math::
        V_{final}(u) = \sum_n V_{grid}[n] \cdot \text{sinc}\left(\frac{u - u_n}{\Delta u}\right)

    where :math:`\text{sinc}(x) = \sin(\pi x) / (\pi x)`. The Whittaker-Shannon interpolation
    theorem guarantees this preserves all information within the signal's bandwidth, making it
    ideal for band-limited signal as with Fourier based X-ray imaging.

    The downsampling uses centered sampling with offset :math:`(factor - 1) / 2` to preserve spatial
    symmetry around the origin, which is critical for maintaining Fourier transform properties.

    **Stage 4: Iterative Landweber Reconstruction**

    The image :math:`I(x,y)` is reconstructed through iterative refinement using the Landweber scheme
    with positivity constraint:

    .. math::
        F_{n+1} &= F_n + \tau (V_{final} - \chi \cdot F_n) \\
        I_{n+1} &= \mathcal{F}^{-1}[F_{n+1}] \\
        I_{n+1} &= \max(0, I_{n+1})

    where:

    - :math:`F_n` is the Fourier transform of the image at iteration n
    - :math:`\tau = 0.2` is the relaxation parameter controlling step size
    - :math:`\chi` is the characteristic function (binary mask) defining the observed UV coverage region
    - :math:`\mathcal{F}^{-1}` denotes the inverse Fourier transform
    - The max operation enforces non-negativity (physical images have positive intensities)

    After each positivity projection, the image is transformed back to the Fourier domain to
    compute the residual for the next iteration.

    **Convergence Criterion**

    Iteration continues until either:

    1. The relative change in residual norm falls below a given threshold.:

       .. math::
           \frac{\|F_{n-1} - V_{final}\| - \|F_n - V_{final}\|}{\|F_{n-1} - V_{final}\|} < threshold

    2. The maximum number of iterations `niter` is reached

    This stopping criterion ensures the reconstruction is consistent with the observed
    visibilities while avoiding over-iteration that could amplify noise.

    **Mathematical Foundation**

    The reconstruction leverages several key principles:

    - **Band-limited signals**: Fourier based X-ray image data is naturally band-limited by the
      maximum baseline, making sinc interpolation theoretically optimal
    - **Regularization through iteration**: The Landweber scheme with positivity acts as an
      implicit regularizer, stabilizing the ill-posed inverse problem
    - **UV coverage constraint**: Restricting reconstruction to the observed Fourier domain
      (via χ mask) prevents artifacts from unmeasured regions
    - **Fourier symmetry**: Centered sampling and symmetric grids preserve the mathematical
      properties required for accurate Fourier transforms

    """
    # Zero-padding and downsampling constants
    # PADDED_GRID_SIZE = 1920
    DOWNSAMPLE_FACTOR = 15
    # DOWNSAMPLE_OFFSET = 7
    PADDING_FACTOR = 6

    r = np.sqrt(vis.u**2 + vis.v**2).value
    r_max = r.max()

    # for comparison to old RHESSI code
    if uv_pixel_size is not None:
        padded_uv_grid_size = shape * DOWNSAMPLE_FACTOR
        uv_grid_size = int(padded_uv_grid_size / PADDING_FACTOR)
    else:
        padded_uv_grid_size, uv_grid_size, uv_pixel_size = determine_grid_parameters(
            shape, pixel_size, PADDING_FACTOR, DOWNSAMPLE_FACTOR
        )

    # Construct regular UV grid for interpolation
    uv_limit = (uv_grid_size / 2 - 1) * uv_pixel_size + uv_pixel_size / 2
    uv_sample_coords = -uv_limit + np.arange(uv_grid_size) * uv_pixel_size
    uu_grid, vv_grid = np.meshgrid(uv_sample_coords, uv_sample_coords)

    # Interpolate visibilities onto regular grid using RBF
    vis_gridded = interpolate_visibilities_to_grid(
        u=vis.u, v=vis.v, vis=vis.visibilities, ugrid=uu_grid, vgrid=vv_grid, **kwargs
    )

    # Mask values outside the original UV sampling coverage
    uv_grid_radius = np.sqrt(uu_grid**2 + vv_grid**2)
    vis_gridded[uv_grid_radius > r_max] = 0j

    vis_downsampled, x_coords_downsampled, y_coords_downsampled, delta_omega, pixel_size = (
        _sinc_interpolate_to_final_grid(
            vis_gridded=vis_gridded,
            uv_grid_size=uv_grid_size,
            uv_pixel_size=uv_pixel_size,
            padded_uv_grid_size=padded_uv_grid_size,
        )
    )

    # Target visibilities for reconstruction
    target_visibilities = vis_downsampled[:, :]

    # Create characteristic function (mask) for UV coverage
    x_grid, y_grid = np.meshgrid(x_coords_downsampled, y_coords_downsampled)
    uv_coverage_mask = np.zeros(vis_downsampled.shape, dtype=np.complex128)
    uv_coverage_mask[np.sqrt(x_grid**2 + y_grid**2) < r_max] = 1 + 0j

    final_image, fourier_transform = landweber_iteration(
        target_visibilities=target_visibilities,
        downsampled_size=vis_downsampled.shape[0],
        uv_coverage_mask=uv_coverage_mask,
        delta_omega=delta_omega,
        niter=niter,
    )

    return final_image, fourier_transform, pixel_size


def determine_grid_parameters(
    shape: int, pixel_size: float, PADDING_FACTOR: int, DOWNSAMPLE_FACTOR: int
) -> tuple[int, int, float]:
    r"""
    Determine Fourier pixel and grid sizes based on output image shape and pixel size.

    Parameters
    ----------
    shape :
        Output image shape.
    pixel_size :
        Size of output pixels.
    PADDING_FACTOR :
        Padding factor.
    DOWNSAMPLE_FACTOR
        Downsampling factor.

    Returns
    -------
    padded_uv_grid_size :
        Size of the padded u, v grid
    uv_grid_size :
        Size of the u, v grid
    uv_pixel_size :
        Size of the u, v pixels
    """

    uv_extent = 1 / pixel_size
    output_uv_pixel_size = uv_extent / (shape)
    uv_pixel_size = output_uv_pixel_size / DOWNSAMPLE_FACTOR
    padded_uv_grid_size = shape * DOWNSAMPLE_FACTOR
    uv_grid_size = int(padded_uv_grid_size / PADDING_FACTOR)

    return padded_uv_grid_size, uv_grid_size, uv_pixel_size


def _sinc_interpolate_to_final_grid(
    vis_gridded: np.ndarray,
    uv_grid_size: int,
    uv_pixel_size: float,
    padded_uv_grid_size: int,
    downsample_factor: int = 15,
    downsample_offset: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample visibility grid using sinc interpolation (zero-pad + downsample).

    This function implements sinc interpolation in the Fourier domain by:
    1. Zero-padding the input grid to a larger size (increases frequency resolution)
    2. Downsampling by selecting centered samples (implements the sinc kernel)

    The combination of zero-padding in Fourier space followed by decimation is
    mathematically equivalent to sinc interpolation, which is the ideal band-limited
    interpolation method.

    Parameters
    ----------
    vis_gridded : np.ndarray
        Input gridded visibilities, shape (grid_size, grid_size).
    grid_size : int
        Original grid size.
    pixel_size : float
        Pixel spacing (same in padded and final grids).
    padded_size : int, optional
        Intermediate padded grid size (default: 1920).
    downsample_factor : int, optional
        Downsampling factor for final grid (default: 15).

    Returns
    -------
    vis_final : np.ndarray
        Sinc-interpolated visibilities on final grid.
    x_coords_final : np.ndarray
        X coordinates of final grid.
    y_coords_final : np.ndarray
        Y coordinates of final grid.
    delta_omega : float
        Final pixel size in Fourier domain
    pixel_size : float
        Finale pixel size in spatial domain
    Notes
    -----
    The mathematical relationship:
    - Zero-padding in Fourier domain → sinc interpolation in spatial domain
    - Centered downsampling → preserves spatial symmetry
    - Combined: ideal band-limited resampling

    The downsample offset uses centered sampling:
        offset = (downsample_factor - 1) // 2
    This ensures the center of each sampling block is selected, preserving
    the spatial symmetry required for accurate Fourier transforms.
    """
    # Zero-pad visibilities to larger grid
    padded_uv_limit = (padded_uv_grid_size / 2 - 1) * uv_pixel_size + uv_pixel_size / 2
    padded_x_coords = -padded_uv_limit + np.arange(padded_uv_grid_size) * uv_pixel_size
    padded_y_coords = padded_x_coords

    vis_zero_padded = np.zeros((padded_uv_grid_size, padded_uv_grid_size), dtype=np.complex128)
    pad_start = (padded_uv_grid_size - uv_grid_size) // 2
    pad_end = pad_start + uv_grid_size
    vis_zero_padded[pad_start:pad_end, pad_start:pad_end] = vis_gridded

    # pad_width = (padded_uv_grid_size - uv_grid_size) // 2
    # vis_zero_padded_new = np.pad(vis_gridded, pad_width, mode='constant', constant_values=0j)

    # Downsample to final image grid
    downsampled_size = padded_uv_grid_size // downsample_factor
    vis_downsampled = np.zeros((downsampled_size, downsampled_size), dtype=np.complex128)
    x_coords_downsampled = np.zeros(downsampled_size)
    y_coords_downsampled = np.zeros(downsampled_size)

    for i in range(downsampled_size):
        x_coords_downsampled[i] = padded_x_coords[downsample_factor * i + downsample_offset]
        y_coords_downsampled[i] = padded_y_coords[downsample_factor * i + downsample_offset]
        for j in range(downsampled_size):
            vis_downsampled[i, j] = vis_zero_padded[
                downsample_factor * i + downsample_offset, downsample_factor * j + downsample_offset
            ]

    # Calculate grid spacing in frequency domain
    total_extent = x_coords_downsampled[-1] - x_coords_downsampled[0]
    delta_omega = total_extent / downsampled_size  # UV pixel spacing
    pixel_size = 1 / total_extent  # Image spacing
    return vis_downsampled, x_coords_downsampled, y_coords_downsampled, delta_omega, pixel_size


def landweber_iteration(
    target_visibilities, downsampled_size, uv_coverage_mask, delta_omega, niter=50, tau=0.2, convergence_threshold=0.02
):
    r"""
    Perform Landweber iterative reconstruction with positivity constraint.

    Parameters
    ----------
    target_visibilities
        Gridded visibility data
    downsampled_size
        Output size
    uv_coverage_mask
        Mask for observed u, v coverage
    delta_omega :
        Pixel size in Fourier space (assumes square pixel)
    niter :
        Maximum number of iterations
    tau :
        Relaxation parameter
    convergence_threshold :
        Threshold for convergence

    Returns
    -------

    """
    FOURIER_NORMALIZATION = 4 * np.pi**2
    # Initialize Landweber iteration
    current_image = np.zeros((downsampled_size, downsampled_size), dtype=np.complex128)
    iteration_history = np.zeros((niter, downsampled_size, downsampled_size))
    final_image = np.zeros((downsampled_size, downsampled_size))

    residual_norms = np.zeros(niter)
    descent_rates = np.zeros(niter - 1)

    # Initial Fourier transform (of zero image)
    fourier_transform = fftshift(ifft2(ifftshift(current_image)) / (FOURIER_NORMALIZATION * delta_omega * delta_omega))

    # Landweber iterations
    for iteration in range(niter):
        # Update Fourier transform using Landweber scheme
        fourier_transform = fourier_transform + tau * (target_visibilities - uv_coverage_mask * fourier_transform)

        # Transform back to image domain
        current_image = fftshift(fft2(ifftshift(fourier_transform)) * FOURIER_NORMALIZATION * delta_omega * delta_omega)

        # Enforce positivity constraint
        current_image[current_image.real < 0] = 0 + 0j
        iteration_history[iteration, :, :] = current_image.real

        # Transform updated image back to Fourier domain for next iteration
        fourier_transform = fftshift(
            ifft2(ifftshift(current_image)) / (FOURIER_NORMALIZATION * delta_omega * delta_omega)
        )

        # Calculate residual for convergence check
        residual = uv_coverage_mask * fourier_transform - target_visibilities
        residual_norms[iteration] = np.sqrt(np.sum(np.abs(residual) * np.abs(residual)))

        # Check convergence criterion
        if iteration >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                descent_rates[iteration - 1] = (
                    residual_norms[iteration - 1] - residual_norms[iteration]
                ) / residual_norms[iteration - 1]
            if descent_rates[iteration - 1] < convergence_threshold:
                logger.info("Converged at iteration %d", iteration)
                break
    else:
        logger.info("Max iterations reached %d", iteration)

    # Prepare final output
    fourier_transform = FOURIER_NORMALIZATION * fourier_transform

    if iteration == niter - 1:
        # Use iteration 14 if max iterations reached (maintains original behavior)
        final_image[:, :] = iteration_history[14, :, :].real
    else:
        final_image = current_image.real

    return final_image, fourier_transform


def interpolate_visibilities_to_grid(u, v, vis, ugrid, vgrid, **kwargs):
    r"""
    Interpolate sparse visibilities to regular grid using `~RBFInterpolator`.

    Parameters
    ----------
    u :
        Sparse ``v`` coordinates.
    v :
        Sparse ``u`` coordinates.
    vis :
        Sparse complex visibilities corresponding to ``u`` and ``v``.
    ugrid :
        Regular grid of ``v``
    vgrid :
        Regular grid of ``u``

    Returns
    -------
    vis_grid : ndarray
        Interpolated visibilities on regular grid (complex)
    """
    norm = 4 * np.pi**2
    uv_observed = np.vstack([u, v]).T
    uv_grid_points = np.vstack([ugrid.flatten(), vgrid.flatten()]).T

    # Interpolate real and imaginary components separately
    real_interpolator = RBFInterpolator(uv_observed, vis.real / norm, **kwargs)
    real_interpolated = real_interpolator(uv_grid_points)

    imag_interpolator = RBFInterpolator(uv_observed, vis.imag / norm, **kwargs)
    imag_interpolated = imag_interpolator(uv_grid_points)

    vis_gridded = (real_interpolated + 1j * imag_interpolated).reshape(ugrid.shape)

    return vis_gridded
