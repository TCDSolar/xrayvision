"""
CLEAN algorithms.

The CLEAN algorithm solves the deconvolution problem by assuming a model for the true sky intensity
which is a collection of point sources or in the case of multiscale clean a collection of
appropriate component shapes at different scales.

"""
import logging

import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
from sunpy.map.map_factory import Map

from astropy.convolution import Gaussian2DKernel

from xrayvision.imaging import vis_psf_image

__all__ = ['clean', 'vis_clean', 'ms_clean', 'vis_ms_clean']

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

__common_clean_doc__ = r"""
    clean_beam_width : `float`
        The width of the gaussian to convolve the model with. If set to 0.0 \
        the gaussian to convolution is disabled
    gain : `float`
        The gain per loop or loop gain
    thres : `float`
        Terminates clean when ``residual.max() <= thres``
    niter : `int`
        Maximum number of iterations to perform

    Returns
    -------
    `numpy.ndarray`
        The CLEAN image 2D

    Notes
    -----
    The CLEAN algorithm can be summarised in pesudo code as follows:

    .. math::
       & \textrm{CLEAN} \left (I^{D}(l, m),\ B(l,m),\ \gamma,\ f_{Thresh},\ N \right ) \\
       & I^{Res} = I^{D},\ M = \{\},\ i=0 \\
       & \textbf{while} \ \operatorname{max} I^{Res} > f_{Thresh} \ \textrm{and} \ i \lt N \
       \textbf{do:} \\
       & \qquad l_{max}, m_{max} = \underset{l,m}{\operatorname{argmax}} I^{Res}(l,m) \\
       & \qquad f_{max} = I^{Res}(l_{max}, m_{max}) \\
       & \qquad I^{Res} = I^{Res} - \alpha \cdot f_{max} \cdot \operatorname{shift} \left
       ( B(l,m), l_{max}, m_{max} \right ) \\
       & \qquad M = M + \{ l_{max}, m_{max}: \alpha \cdot f_{max} \} \\
       & \qquad i = i + 1 \\
       & \textbf{done} \\
       & \textbf{return}\  M,\ I^{Res}

    """


def clean(dirty_map, dirty_beam, pixel=None, clean_beam_width=4.0,
          gain=0.1, thres=0.01, niter=5000):
    r"""
    Clean the image using Hogbom's original method.

    CLEAN iteratively subtracts the PSF or dirty beam from the dirty map to create the residual.
    At each iteration the location of the maximum residual is found and a shifted dirty beam is
    subtracted that location updating the residual. This process continues until either `niter`
    iterations is reached or the maximum residual <= `thres`.

    Parameters
    ----------
    dirty_map : `numpy.ndarray`
        The dirty map to be cleaned 2D
    dirty_beam : `numpy.ndarray`
        The dirty beam or point spread function (PSF) 2D must
    pixel :
        Size of a pixel
    """
    # Ensure both beam and map are even/odd on same axes
    if not [x % 2 == 0 for x in dirty_map.shape] == [x % 2 == 0 for x in dirty_beam.shape]:
        raise ValueError('')
    pad = [0 if x % 2 == 0 else 1 for x in dirty_map.shape]

    # Assume beam, map center is in middle
    beam_center = (dirty_beam.shape[0] - 1)/2.0, (dirty_beam.shape[1] - 1)/2.0
    map_center = (dirty_map.shape[0] - 1)/2.0, (dirty_map.shape[1] - 1)/2.0

    # Work out size of map for slicing over-sized dirty beam
    shape = dirty_map.shape
    height = shape[0] // 2
    width = shape[1] // 2

    # max_beam = dirty_beam.max()

    # Model for sources
    model = np.zeros(dirty_map.shape)
    componets = []
    for i in range(niter):
        # Find max in dirty map and save to point source
        mx, my = np.unravel_index(dirty_map.argmax(), dirty_map.shape)
        imax = dirty_map[mx, my]
        # TODO check if correct and how to undo
        # imax = imax * max_beam
        model[mx, my] += gain * imax

        logger.info(f"Iter: {i}, strength: {imax}, location: {mx, my}")

        offset = map_center[0] - mx, map_center[1] - my
        shifted_beam_center = int(beam_center[0] + offset[0]), int(beam_center[1] + offset[1])
        xr = slice(shifted_beam_center[0] - height, shifted_beam_center[0] + height + pad[0])
        yr = slice(shifted_beam_center[1] - width, shifted_beam_center[1] + width + pad[0])

        shifted = dirty_beam[xr, yr]

        comp = imax * gain * shifted

        componets.append((mx, my, comp[mx, my]))

        dirty_map = np.subtract(dirty_map, comp)

        # if dirty_map.max() <= thres:
        #      logger.info("Threshold reached")
        #      break
        # # el
        # if np.abs(dirty_map.min()) > dirty_map.max():
        #     logger.info("Largest residual negative")
        #     break

    else:
        print("Max iterations reached")

    if clean_beam_width != 0.0:
        # Convert from FWHM to StDev
        x_stdev = (clean_beam_width/pixel[0] / (2.0 * np.sqrt(2.0 * np.log(2.0)))).value
        y_stdev = (clean_beam_width/pixel[1] / (2.0 * np.sqrt(2.0 * np.log(2.0)))).value
        clean_beam = Gaussian2DKernel(x_stdev, y_stdev, x_size=dirty_beam.shape[1],
                                      y_size=dirty_beam.shape[0]).array
        # Normalise beam
        clean_beam = clean_beam / clean_beam.max()

        # Convolve clean beam with model and scale
        clean_map = (signal.convolve2d(model, clean_beam/clean_beam.sum(), mode='same')
                     / (pixel[0]*pixel[1]))

        # Scale residual map with model and scale
        dirty_map = dirty_map / clean_beam.sum() / (pixel[0] * pixel[1])
        return clean_map+dirty_map, model, dirty_map

    return model+dirty_map, model, dirty_map


clean.__doc__ += __common_clean_doc__


def vis_clean(vis, shape, pixel, clean_beam_width=4.0, niter=100, map=True, **kwargs):
    r"""
    Clean the visibilities using Hogbom's original method.

    A wrapper around lower level `clean` which calculates the dirty map and psf

    Parameters
    ----------
    vis : `xrayvision.visibilty.Visibly`
        The visibilities to clean
    shape :
        Size of map
    pixel :
        Size of pixel
    map : `boolean` optional
        Return an `sunpy.map.Map` by default or data only if `False`
    """

    dirty_map = vis_to_map(vis, shape=shape, pixel_size=pixel)
    dirty_beam = vis_psf_image(vis, shape=shape*3, pixel_size=pixel, map=False)
    clean_map, model, residual = clean(dirty_map.data, dirty_beam.value, pixel=pixel,
                                       clean_beam_width=clean_beam_width, niter=niter, **kwargs)
    if not map:
        return clean_map, model, residual

    return [Map((data, dirty_map.meta)) for data in (clean_map, model, residual)]


vis_clean.__doc__ += __common_clean_doc__


__common_ms_clean_doc__ = r"""
    scales : array-like, optional, optional
        The scales to use eg ``[1, 2, 4, 8]``
    clean_beam_width : `float`
        The width of the gaussian to convolve the model with. If set to 0.0 the gaussian \
        convolution is disabled
    gain : `float`
        The gain per loop or loop gain
    thres : `float`
        Terminates clean when `residuals.max() <= thres``
    niter : `int`
        Maximum number of iterations to perform

    Returns
    -------
    `numpy.ndarray`
        Cleaned image

    Notes
    -----
    This is an implementation of the multiscale clean algorithm as outlined in [R1]_ adapted for \
    x-ray Fourier observations.

    It is based on the on the implementation in the CASA software which can be found here_.

    .. _here: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc#L956 #noqa

    References
    ----------
    .. [R1] Cornwell, T. J., "Multiscale CLEAN Deconvolution of Radio Synthesis Images", IEEE Journal of Selected Topics in Signal Processing, vol 2, p793-801, Paper_ #noqa

    .. _Paper: https://ieeexplore.ieee.org/document/4703304/
    """


def ms_clean(dirty_map, dirty_beam, pixel, scales=None,
             clean_beam_width=4.0, gain=0.1, thres=0.01, niter=5000):
    r"""
    Clean the map using a multiscale clean algorithm.

    Parameters
    ----------
    dirty_map : `numpy.ndarray`
        The 2D dirty map to be cleaned
    dirty_beam : `numpy.ndarray`
        The 2D dirty beam should have the same dimensions as `dirty_map`
    """
    # Compute the number of dyadic scales, their sizes and scale biases
    number_of_scales = np.floor(np.log2(min(dirty_map.shape))).astype(int)
    scale_sizes = 2**np.arange(number_of_scales)

    if scales:
        scales = np.array(scales)
        number_of_scales = len(scales)
        scale_sizes = scales

    scale_sizes = np.where(scale_sizes == 0, 1, scale_sizes)

    scale_biases = 1 - 0.6 * scale_sizes / scale_sizes.max()

    model = np.zeros(dirty_map.shape)

    map_center = (dirty_map.shape[0] - 1)/2.0, (dirty_map.shape[1] - 1)/2.0
    height = dirty_map.shape[0] // 2
    width = dirty_map.shape[1] // 2
    pad = [0 if x % 2 == 0 else 1 for x in dirty_map.shape]

    # Pre-compute scales, residual maps and dirty beams at each scale and dirty beam cross terms
    scales = np.zeros((dirty_map.shape[0], dirty_map.shape[1], number_of_scales))
    scaled_residuals = np.zeros((dirty_map.shape[0], dirty_map.shape[1], number_of_scales))
    scaled_dirty_beams = np.zeros((dirty_beam.shape[0], dirty_beam.shape[1], number_of_scales))
    max_scaled_dirty_beams = np.zeros(number_of_scales)
    cross_terms = {}

    for i, scale in enumerate(scale_sizes):
        scales[:, :, i] = component(scale=scale, shape=dirty_map.shape)
        scaled_residuals[:, :, i] = signal.convolve(dirty_map, scales[:, :, i], mode='same')
        scaled_dirty_beams[:, :, i] = signal.convolve(dirty_beam, scales[:, :, i], mode='same')
        max_scaled_dirty_beams[i] = scaled_dirty_beams[:, :, i].max()
        for j in range(i, number_of_scales):
            cross_terms[(i, j)] = signal.convolve(
                signal.convolve(dirty_beam, scales[:, :, i], mode='same'),
                scales[:, :, j], mode='same')

    # Clean loop
    for i in range(niter):
        # print(f'Clean loop {i}')
        # For each scale find the strength and location of max residual
        # Chose scale with has maximum strength
        max_index = np.argmax(scaled_residuals)
        max_x, max_y, max_scale = np.unravel_index(max_index, scaled_residuals.shape)

        strength = scaled_residuals[max_x, max_y, max_scale]

        # Adjust for the max of scaled beam
        strength = strength / max_scaled_dirty_beams[max_scale]

        logger.info(f"Iter: {i}, max scale: {max_scale}, strength: {strength}")

        # Loop gain and scale dependent bias
        strength = strength * scale_biases[max_scale] * gain

        beam_center = [(scaled_dirty_beams[:, :, max_scale].shape[0] - 1) / 2.0,
                       (scaled_dirty_beams[:, :, max_scale].shape[1] - 1) / 2.0]

        offset = map_center[0] - max_x, map_center[1] - max_y
        shifted_beam_center = int(beam_center[0] + offset[0]), int(beam_center[1] + offset[1])
        xr = slice(shifted_beam_center[0] - height, shifted_beam_center[0] + height + pad[0])
        yr = slice(shifted_beam_center[1] - width, shifted_beam_center[1] + width + pad[0])

        # shifted = dirty_beam[xr, yr]

        comp = strength * shift(scales[:, :, max_scale],
                                (max_x - map_center[0], max_y - map_center[1]), order=0)

        # comp = strength * scales[xr, yr]

        # Add this component to current model
        model = np.add(model, comp)

        # Update all images using precomputed terms
        for j, _ in enumerate(scale_sizes):
            if j > max_scale:
                cross_term = cross_terms[(max_scale, j)]
            else:
                cross_term = cross_terms[(j, max_scale)]

            # comp = strength * shift(cross_term[xr, yr],
            #                         (max_x - beam_center[0], max_y - beam_center[1]), order=0)

            comp = strength * cross_term[xr, yr]

            scaled_residuals[:, :, j] = np.subtract(scaled_residuals[:, :, j], comp)

        # End max(res(a)) or niter
        if scaled_residuals[:, :, max_scale].max() <= thres:
            logger.info("Threshold reached")
            # break

        # Largest scales largest residual is negative
        if np.abs(scaled_residuals[:, :, 0].min()) > scaled_residuals[:, :, 0].max():
            logger.info("Max scale residual negative")
            break

    else:
        logger.info("Max iterations reached")

    # Convolve model with clean beam  B_G * I^M
    if clean_beam_width != 0.0:
        x_stdev = (clean_beam_width/pixel[0] / (2.0 * np.sqrt(2.0 * np.log(2.0)))).value
        y_stdev = (clean_beam_width/pixel[1] / (2.0 * np.sqrt(2.0 * np.log(2.0)))).value
        clean_beam = Gaussian2DKernel(x_stdev, y_stdev, x_size=dirty_beam.shape[1],
                                      y_size=dirty_beam.shape[0]).array

        # Normalise beam
        clean_beam = clean_beam / clean_beam.max()

        clean_map = signal.convolve2d(model, clean_beam, mode='same') / (pixel[0]*pixel[1])

        # Scale residual map with model and scale
        dirty_map = (scaled_residuals / clean_beam.sum() / (pixel[0] * pixel[1])).sum(axis=2)

        return clean_map+dirty_map, model, dirty_map
    # Add residuals B_G * I^M + I^R
    return model, scaled_residuals.sum(axis=2)


ms_clean.__doc__ += __common_ms_clean_doc__


def vis_ms_clean(vis, shape, pixel, scales=None, clean_beam_width=4.0,
                 gain=0.1, thres=0.01, niter=5000, map=True):
    r"""
    Clean the visibilities using a multiscale clean method.

    A wrapper around `ms_clean` which calculates the dirty map and psf.

    Parameters
    ----------
    vis : `xrayvision.visibilty.Visibly`
        The visibilities to clean
    shape :
        Size of map
    pixel :
        Size of pixel

    """
    dirty_map = vis_to_map(vis, shape=shape, pixel_size=pixel)
    dirty_beam = vis_psf_image(vis, shape=shape * 3, pixel_size=pixel, map=False)
    clean_map, model, residual = ms_clean(dirty_map.data, dirty_beam, scales=scales,
                                          clean_beam_width=clean_beam_width, gain=gain,
                                          thres=thres, niter=niter)
    if not map:
        return clean_map, model, residual

    return [Map((data, dirty_map.meta)) for data in (clean_map, model, residual)]


vis_ms_clean.__doc__ += __common_ms_clean_doc__


def radial_prolate_sphereoidal(nu):
    r"""
    Calculate prolate spheroidal wave function approximation.

    Parameters
    ----------
    nu : `float`
        The radial value to evaluate the function at

    Returns
    -------
    `float`
        The amplitude of the the prolate spheroid function at `nu`

    Notes
    -----
    Note this is a direct translation of the on the implementation the CASA code reference by [1] \
    and can be found here Link_

    .. _Link: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc#L956 #noqa

    """
    if nu <= 0:
        return 1.0
    elif nu >= 1.0:
        return 0.0
    else:
        n_p = 5
        n_q = 3

        p = np.zeros((n_p, 2))
        q = np.zeros((n_q, 2))

        p[0, 0] = 8.203343e-2
        p[1, 0] = -3.644705e-1
        p[2, 0] = 6.278660e-1
        p[3, 0] = -5.335581e-1
        p[4, 0] = 2.312756e-1
        p[0, 1] = 4.028559e-3
        p[1, 1] = -3.697768e-2
        p[2, 1] = 1.021332e-1
        p[3, 1] = -1.201436e-1
        p[4, 1] = 6.412774e-2

        q[0, 0] = 1.0000000e0
        q[1, 0] = 8.212018e-1
        q[2, 0] = 2.078043e-1
        q[0, 1] = 1.0000000e0
        q[1, 1] = 9.599102e-1
        q[2, 1] = 2.918724e-1

        part = 0
        nuend = 0.0

        if 0.0 <= nu < 0.75:
            part = 0
            nuend = 0.75
        elif 0.75 <= nu <= 1.00:
            part = 1
            nuend = 1.0

        top = p[0, part]
        delnusq = np.power(nu, 2.0) - np.power(nuend, 2.0)

        for k in range(1, n_p):
            top += p[k, part] * np.power(delnusq, k)

        bot = q[0, part]
        for k in range(1, n_q):
            bot += q[k, part] * np.power(delnusq, k)

        if bot != 0.0:
            return top/bot
        else:
            return 0


def vec_radial_prolate_sphereoidal(nu):
    r"""
    Calculate prolate spheroidal wave function approximation.

    Parameters
    ----------
    nu : `float` array
        The radial value to evaluate the function at

    Returns
    -------
    `float`
        The amplitude of the the prolate spheroid function at `nu`

    Notes
    -----
    Note this is based on the implementation the CASA code reference by [1] and can be found here
    Link_

    .. _Link: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc#L956 #noqa

    """
    nu = np.array(nu)

    n_p = 5
    n_q = 3

    p = np.zeros((n_p, 2))
    q = np.zeros((n_q, 2))

    p[0, 0] = 8.203343e-2
    p[1, 0] = -3.644705e-1
    p[2, 0] = 6.278660e-1
    p[3, 0] = -5.335581e-1
    p[4, 0] = 2.312756e-1
    p[0, 1] = 4.028559e-3
    p[1, 1] = -3.697768e-2
    p[2, 1] = 1.021332e-1
    p[3, 1] = -1.201436e-1
    p[4, 1] = 6.412774e-2

    q[0, 0] = 1.0000000e0
    q[1, 0] = 8.212018e-1
    q[2, 0] = 2.078043e-1
    q[0, 1] = 1.0000000e0
    q[1, 1] = 9.599102e-1
    q[2, 1] = 2.918724e-1

    lower = np.where((nu >= 0.0) & (nu < 0.75))  # part = 0, nuend = 0.75
    upper = np.where((nu >= 0.75) & (nu <= 1.00))  # part = 1, nuend = 1.0

    delnusq = np.zeros_like(nu)
    delnusq[lower] = np.power(nu[lower], 2.0) - np.power(0.75, 2.0)
    delnusq[upper] = np.power(nu[upper], 2.0) - np.power(1.00, 2.0)

    top = np.zeros_like(nu, dtype=float)
    top[lower] = p[0, 0]
    top[upper] = p[0, 1]

    k = np.arange(1, n_p)
    top[lower] += np.sum(p[k, 0, np.newaxis] * np.power(delnusq[lower], k[..., np.newaxis]), axis=0)
    top[upper] += np.sum(p[k, 1, np.newaxis] * np.power(delnusq[upper], k[..., np.newaxis]), axis=0)

    bot = np.zeros_like(nu, dtype=float)
    bot[lower] = q[0, 0]
    bot[upper] = q[0, 1]

    j = np.arange(1, n_q)
    bot[lower] += np.sum(q[j, 0, np.newaxis] * np.power(delnusq[lower], j[..., np.newaxis]), axis=0)
    bot[upper] += np.sum(q[j, 1, np.newaxis] * np.power(delnusq[upper], j[..., np.newaxis]), axis=0)

    out = np.zeros(nu.shape)
    out[bot != 0] = top[bot != 0]/bot[bot != 0]
    out = np.where(nu <= 0, 1.0, out)
    out = np.where(nu >= 1, 0.0, out)

    return out


def component(scale, shape):
    r"""

    Parameters
    ----------
    scale

    Returns
    -------

    """
    # if scale == 0.0:
    #     out = np.zeros((3, 3))
    #     out[1,1] = 1.0
    #     return out
    # elif scale % 2 == 0:  # Even so keep output even
    #     shape = np.array((2 * scale + 2, 2 * scale + 2), dtype=int)
    # else:  # Odd so keep odd
    #     shape = np.array((2 * scale + 1, 2 * scale + 1), dtype=int)

    refx, refy = (np.array(shape) - 1) / 2.0

    if scale == 0.0:
        wave_amp = np.zeros(shape)
        wave_amp[int(refx), int(refy)] = 1
        return wave_amp

    xy = np.mgrid[0:shape[0]:1, 0:shape[1]:1]
    radii_squared = ((xy[0, :, :] - refx) / scale)**2 + ((xy[1, :, :] - refy) / scale)**2

    rad_zeros_indices = radii_squared <= 0.0
    amp_zero_indices = radii_squared >= 1.0

    wave_amp = vec_radial_prolate_sphereoidal(np.sqrt(radii_squared.reshape(radii_squared.size)))
    wave_amp = wave_amp.reshape(shape)
    wave_amp[rad_zeros_indices] = vec_radial_prolate_sphereoidal([0])[0]

    wave_amp = wave_amp * (1 - radii_squared)

    wave_amp[amp_zero_indices] = 0.0

    return wave_amp
