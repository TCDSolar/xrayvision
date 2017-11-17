"""
CLEAN algorithms.

adsf
"""


import numpy as np
from astropy.convolution import Gaussian2DKernel
from scipy import signal
from scipy.ndimage.interpolation import shift

__all__ = ['clean']


def clean(dirty_map, dirty_beam, clean_beam_width=4.0, gain=0.1, thres=0.01, niter=1000):
    r"""
    Clean the image using Hogbom's orginal method.

    Will stop when either `thres` is reached or `niter`
    iteration have been completed.

    Parameters
    ----------
    dirty_map : `numpy.ndarray`
        The dirty map to be cleaned 2D
    dirty_beam : `numpy.ndarray`
        The dirty beam or point spread function (PSF) 2D
    clean_beam_width : `float`
        The width of the gaussian to convolve the model with. If set to 0.0 \
        the gaussian to convolution is disabled
    gain : `float`
        The gain per loop or loop gain
    thres : `float`
        A threshold val at which to stop
    niter : `int`
        Maximum number of iterations to perform

    Returns
    -------
    `numpy.ndarray`
        The CLEAN image 2D

    """
    # Assume bear center is in middle
    beam_center = (dirty_beam.shape[0] - 1)/2.0, (dirty_beam.shape[1] - 1)/2.0

    # Model for sources
    model = np.zeros(dirty_map.shape)
    for _i in range(niter):
        # Find max in dirty map and save to point source
        mx, my = np.unravel_index(dirty_map.argmax(), dirty_map.shape)
        Imax = dirty_map[mx, my]
        model[mx, my] += gain*Imax

        comp = Imax * gain * shift(dirty_beam, (mx - beam_center[0], my - beam_center[1]), order=0)

        dirty_map = np.subtract(dirty_map, comp)

        if dirty_map.max() <= thres:
            break

    if clean_beam_width != 0.0:
        clean_beam = Gaussian2DKernel(stddev=clean_beam_width, x_size=dirty_beam.shape[1],
                                      y_size=dirty_beam.shape[0]).array

        model = signal.convolve2d(model, clean_beam, mode='same')  # noqa

        clean_beam = clean_beam * (1/clean_beam.max())
        dirty_map = dirty_map / clean_beam.sum()

    return model + dirty_map
