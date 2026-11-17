import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import RBFInterpolator

from xrayvision.visibility import Visibilities

__all__ = ["uv_smooth", "uv_smooth_idl"]

logger = logging.getLogger(__name__)


def uv_smooth_idl(vis: Visibilities, niter: int = 50) -> tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
    r"""
    uv_smooth image reconstruction method, a direct port of the RHESSI IDL ``uv_smooth``.

    This method reconstructs images from sparse Fourier-domain visibilities using an iterative Landweber scheme. It
    interpolates visibilities onto a regular uv grid, applies smoothing to stabilize the reconstruction, and
    iteratively refines the image while enforcing positivity.

    The grid and image size are hard coded from the lowest subcollimator (``isc``) present, as in the IDL code, use
    `~xrayvision.uv_smooth.uv_smooth` for a general version.

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

    detmin = min(vis.meta["isc"])  # type: ignore[index]
    # Pixel scale factor and grid size set by the lowest subcollimator: fov = 0.45, 0.26 and 0.16 respectively
    pixel_scale, N = {0: (2.0, 450), 1: (2.0, 260)}.get(int(detmin), (1.0, 320))
    pixel = 0.0005 * pixel_scale

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

    intzpadd = np.zeros((Nnew, Nnew), dtype=np.complex128)
    intzpadd[(Nnew - N) // 2 : (Nnew - N) // 2 + N, (Nnew - N) // 2 : (Nnew - N) // 2 + N] = vis_interp

    im_new = Nnew // 15

    idx = 15 * np.arange(im_new) + 7
    intznew = intzpadd[np.ix_(idx, idx)]
    xpixnew = xpix[idx]
    ypixnew = ypix[idx]

    OMEGA = (xpixnew[im_new - 1] - xpixnew[0]) / 2
    X = im_new / (4 * OMEGA)
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

    converged = False
    for iteration in range(niter):
        # Update rule
        F_Trasf_up = F_Trasf + tau * (g - chi * F_Trasf)
        F_Trasf = F_Trasf_up

        # FFT of the updated soln
        F_Trasf_shifted = ifftshift(F_Trasf)
        map_shifted = fft2(F_Trasf_shifted) * 4 * np.pi**2 * deltaomega * deltaomega
        map_actual = fftshift(map_shifted)

        # Project solution onto subset of positive solutions (positivity constraint)
        map_actual[map_actual.real < 0] = 0 + 0j
        map_iteration[iteration, :, :] = map_actual.real

        map_shifted = ifftshift(map_actual)
        F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi**2 * deltaomega * deltaomega)
        F_Trasf = fftshift(F_Trasf_shifted)

        # Stopping criterion based on the descent of ||Af - g||
        Af_g = chi * F_Trasf - g
        normAf_g[iteration] = np.sqrt(np.sum(np.abs(Af_g) * np.abs(Af_g)))
        if iteration >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                descent[iteration - 1] = (normAf_g[iteration - 1] - normAf_g[iteration]) / normAf_g[iteration - 1]
            if descent[iteration - 1] < 0.02:
                logger.info("Converged at iteration %d", iteration)
                converged = True
                break
    if not converged:
        logger.info("Max iterations reached %d", niter)

    # output
    F_Trasf = 4 * np.pi**2 * F_Trasf

    if converged:
        map_solution = map_actual.real
    else:
        map_solution[:, :] = map_iteration[14, :, :].real

    return map_solution, F_Trasf, deltax


def uv_smooth(
    vis: Visibilities,
    shape: int = 128,
    pixel_size: float = 1.0,
    uv_pixel_size: float | None = None,
    niter: int = 50,
    **kwargs: Any,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
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
    2. **Downsampling** by a fixed factor about zero frequency to obtain the output grid

    This combination is mathematically equivalent to sinc interpolation:

    .. math::
        V_{final}(u) = \sum_n V_{grid}[n] \cdot \text{sinc}\left(\frac{u - u_n}{\Delta u}\right)

    where :math:`\text{sinc}(x) = \sin(\pi x) / (\pi x)`. The Whittaker-Shannon interpolation
    theorem guarantees this preserves all information within the signal's bandwidth, making it
    ideal for band-limited signal as with Fourier based X-ray imaging.

    The downsampled grid is aligned so zero frequency sits at the centre index the FFT expects. Any misalignment
    multiplies the image by a phase ramp that damps and shifts sources away from the map centre.

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
    DOWNSAMPLE_FACTOR = 15
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

    # Construct regular UV grid for interpolation, zero frequency at index uv_grid_size // 2 (FFT layout)
    uv_sample_coords = (np.arange(uv_grid_size) - uv_grid_size // 2) * uv_pixel_size
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
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], float, float]:
    """
    Resample visibility grid using sinc interpolation (zero-pad + downsample).

    This function implements sinc interpolation in the Fourier domain by:
    1. Zero-padding the input grid to a larger size (increases frequency resolution)
    2. Downsampling by selecting every ``downsample_factor`` sample about zero frequency

    The combination of zero-padding in Fourier space followed by decimation is
    mathematically equivalent to sinc interpolation, which is the ideal band-limited
    interpolation method.

    Parameters
    ----------
    vis_gridded : np.ndarray
        Input gridded visibilities, shape (uv_grid_size, uv_grid_size).
    uv_grid_size : int
        Original grid size.
    uv_pixel_size : float
        Pixel spacing (same in padded and final grids).
    padded_uv_grid_size : int
        Intermediate padded grid size.
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
    - Downsampling about zero frequency → matches the FFT frequency layout
    - Combined: ideal band-limited resampling

    The samples are taken at ``padded_uv_grid_size // 2 + downsample_factor * (i - downsampled_size // 2)`` so zero
    frequency lands at index ``downsampled_size // 2``, as `~numpy.fft.fftshift` assumes. The RHESSI IDL code samples
    at ``downsample_factor * i + 7`` instead (see `~xrayvision.uv_smooth.uv_smooth_idl`), which is half a downsampled
    cell off and multiplies the image by a phase ramp, damping and shifting off-centre sources.
    """
    # Zero-pad visibilities to larger grid, keeping zero frequency at index padded_uv_grid_size // 2
    vis_zero_padded = np.zeros((padded_uv_grid_size, padded_uv_grid_size), dtype=np.complex128)
    pad_start = padded_uv_grid_size // 2 - uv_grid_size // 2
    pad_end = pad_start + uv_grid_size
    vis_zero_padded[pad_start:pad_end, pad_start:pad_end] = vis_gridded

    # Downsample so that zero frequency lands at index downsampled_size // 2, the layout fftshift assumes. Any other
    # alignment multiplies the image by a phase ramp which damps and shifts off-centre sources.
    downsampled_size = padded_uv_grid_size // downsample_factor
    offsets = downsample_factor * (np.arange(downsampled_size) - downsampled_size // 2)
    idx = padded_uv_grid_size // 2 + offsets
    vis_downsampled = vis_zero_padded[np.ix_(idx, idx)]
    x_coords_downsampled = offsets * uv_pixel_size
    y_coords_downsampled = x_coords_downsampled

    # Calculate grid spacing in frequency domain
    total_extent = x_coords_downsampled[-1] - x_coords_downsampled[0]
    delta_omega = total_extent / downsampled_size  # UV pixel spacing
    pixel_size = 1 / total_extent  # Image spacing
    return vis_downsampled, x_coords_downsampled, y_coords_downsampled, delta_omega, pixel_size


def landweber_iteration(
    target_visibilities: npt.NDArray[Any],
    downsampled_size: int,
    uv_coverage_mask: npt.NDArray[Any],
    delta_omega: float,
    niter: int = 50,
    tau: float = 0.2,
    convergence_threshold: float = 0.02,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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
    final_image :
        The reconstructed image, if the maximum number of iterations was reached without converging this is the image
        from iteration 15 (index 14).
    fourier_transform :
        The Fourier transform of the final image.
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
    converged = False
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
                converged = True
                break
    if not converged:
        logger.info("Max iterations reached %d", niter)

    # Prepare final output
    fourier_transform = FOURIER_NORMALIZATION * fourier_transform

    if converged:
        final_image = current_image.real
    else:
        # Use iteration 14 if max iterations reached (maintains original behavior)
        final_image[:, :] = iteration_history[14, :, :].real

    return final_image, fourier_transform


def interpolate_visibilities_to_grid(
    u: Any, v: Any, vis: Any, ugrid: npt.NDArray[Any], vgrid: npt.NDArray[Any], **kwargs: Any
) -> npt.NDArray[Any]:
    r"""
    Interpolate sparse visibilities to regular grid using `~RBFInterpolator`.

    Parameters
    ----------
    u :
        Sparse ``u`` coordinates.
    v :
        Sparse ``v`` coordinates.
    vis :
        Sparse complex visibilities corresponding to ``u`` and ``v``.
    ugrid :
        Regular grid of ``u``
    vgrid :
        Regular grid of ``v``

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

    vis_gridded: npt.NDArray[Any] = (real_interpolated + 1j * imag_interpolated).reshape(ugrid.shape)

    return vis_gridded
