"""
Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT) related functions.

There are two implementations one a standard DFT `dft` and IDFT `idft` in terms of pixel space, i.e.
the input has no positional information other than an arbitary 0 origin and a length. The second
takes inputs which have positional information `dft_map` and the inverse `idft_map`

"""
import numpy as np

import astropy.units as apu
from astropy.units.core import UnitsError


@apu.quantity_input(center='angle', pixel_size='angle')
def generate_xy(number_pixels, center=0.0 * apu.arcsec, pixel_size=1.0 * apu.arcsec):
    """
    Generate the x or y coordinates given the number of pixels, center and pixel size.

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    center : `float`, optional
        Center coordinates
    pixel_size : `float`, optional
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    `numpy.array`
        The generated x, y coordinates

    See Also
    --------
    `generate_uv` : Generates corresponding coordinates but un u, v space

    Examples
    --------
    >>> generate_xy(9)
    <Quantity [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.] arcsec>

    >>> generate_xy(9, pixel_size=2.5 * apu.arcsec)
    <Quantity [-10. , -7.5, -5. , -2.5,  0. ,  2.5,  5. ,  7.5, 10. ] arcsec>

    >>> generate_xy(9, center=10 * apu.arcsec, pixel_size=2.5 * apu.arcsec)
    <Quantity [ 0. ,  2.5,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. ] arcsec>

    """
    x = (np.arange(number_pixels) - number_pixels / 2 + 0.5) * pixel_size + center
    return x


@apu.quantity_input(center='angle', pixel_size='angle')
def generate_uv(number_pixels, center=0.0 * apu.arcsec, pixel_size=1.0 * apu.arcsec):
    """
    Generate the u or v  coordinates given the number of pixels, center and pixel size.

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    center : `float`, optional
        Center coordinates
    pixel_size : `float`, optional
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    `numpy.array`
        The generated u, v coordinates

    See Also
    --------
    `generate_xy` : Generates corresponding coordinate but un x, y space

    Examples
    --------
    >>> generate_uv(9)
    <Quantity [-0.44444444, -0.33333333, -0.22222222, -0.11111111,  0.        ,
                0.11111111,  0.22222222,  0.33333333,  0.44444444] 1 / arcsec>

    >>> generate_uv(9, pixel_size=2.5 * apu.arcsec)
    <Quantity [-0.17777778, -0.13333333, -0.08888889, -0.04444444,  0.        ,
                0.04444444,  0.08888889,  0.13333333,  0.17777778] 1 / arcsec>

    >>> generate_uv(9, center=10 * apu.arcsec, pixel_size=2.5 * apu.arcsec)
    <Quantity [-0.07777778, -0.03333333,  0.01111111,  0.05555556,  0.1       ,
                0.14444444,  0.18888889,  0.23333333,  0.27777778] 1 / arcsec>

    """
    x = (np.arange(number_pixels) - number_pixels / 2 + 0.5) * (1 / (pixel_size * number_pixels))
    if center.value != 0.0:
        x += 1 / center
    return x


@apu.quantity_input(center='angle', pixel_size='angle')
def dft_map(input_array, *, u, v, center=(0.0, 0.0) * apu.arcsec, pixel_size=(1.0, 1.0) * apu.arcsec):
    r"""
    Discrete Fourier transform in terms of coordinates returning 1-D array complex visibilities.

    Parameters
    ----------
    input_array : `numpy.ndarray`
        Input array to be transformed should be 2D (m, n)
    uv : `numpy.array`
        Array of 2xN u, v coordinates where the visibilities will be evaluated
    center : `float` (x, y), optional
        Coordinates of the center of the map e.g. ``(0,0)`` or ``[5.0, -2.0]``
    pixel_size : `float` (dx, dy), optional
        The pixel size in x and y directions, need not be square e.g. ``(1, 3)``

    Returns
    -------
    `numpy.ndarray`
        Array of N `complex` visibilities evaluated at the u, v coordinates given bu `uv`

    """
    m, n = input_array.shape

    y = generate_xy(m, center[1], pixel_size[1])
    x = generate_xy(n, center[0], pixel_size[0])

    x, y = np.meshgrid(x, y)
    uv = np.vstack([u, v])
    # Check units are correct for exp need to be dimensionless and then remove units for speed
    if (uv[0, :] * x[0, 0]).unit == apu.dimensionless_unscaled and \
            (uv[1, :] * y[0, 0]).unit == apu.dimensionless_unscaled:

        uv = uv.value
        x = x.value
        y = y.value

        vis = np.sum(input_array[..., np.newaxis] * np.exp(-2j * np.pi * (
            x[..., np.newaxis] * uv[np.newaxis, 0, :] + y[..., np.newaxis] * uv[np.newaxis, 1, :])),
            axis=(0, 1))

        return vis
    else:
        raise UnitsError("Incompatible units on uv {uv.unit} should cancel with xy "
                         "to leave a dimensionless quantity")


@apu.quantity_input(center='angle', pixel_size='angle')
def idft_map(input_vis, *, u, v, shape, weights=None, center=(0.0, 0.0) * apu.arcsec,
             pixel_size=(1.0, 1.0) * apu.arcsec):
    r"""
    Inverse discrete Fourier transform in terms of coordinates returning a 2D real array or image.

    Parameters
    ----------
    uv : `numpy.ndarray`
        Array of 2xN u, v coordinates corresponding to the input visibilities in `input_vis`
    input_vis : `numpy.ndarray`
        Array of N `complex` input visibilities
    shape : `float` (m,n)
        The shape of the output array to create
    weights : `numpy.ndarray`
        Array of weights for visibilities
    center : `float` (x, y), optional
        Coordinates of the center of the map e.g. ``(0,0)`` or ``[5.0, -2.0]``
    pixel_size : `float` (dx, dy), optional
        The pixel size in x and y directions, need not be square e.g. ``(1, 3)``

    Returns
    -------
    `numpy.ndarray`
        The complex visibilities evaluated at the u, v coordinates

    """
    m, n = shape
    y = generate_xy(m, center[1], pixel_size[1])
    x = generate_xy(n, center[0], pixel_size[0])
    x, y = np.meshgrid(x, y)

    if weights is None:
        weights = np.ones(input_vis.shape)
    uv = np.vstack([u, v])
    # Check units are correct for exp need to be dimensionless and then remove units for speed
    if (uv[0, :] * x[0, 0]).unit == apu.dimensionless_unscaled and \
            (uv[1, :] * y[0, 0]).unit == apu.dimensionless_unscaled:

        uv = uv.value
        x = x.value
        y = y.value

        image = np.sum(input_vis * weights * np.exp(2j * np.pi * (
            x[..., np.newaxis] * uv[np.newaxis, 0, :] + y[..., np.newaxis] * uv[np.newaxis, 1, :])),
            axis=2)

        return np.real(image)
    else:
        raise UnitsError("Incompatible units on uv {uv.unit} should cancel with xy "
                         "to leave a dimensionless quantity")

# def dft(im, uv):
#     """
#     Discrete Fourier transform of the input array or image calculated at the given u, v
#     coordinates
#
#     Loops over a list of u, v coordinates rather than looping over u and v separately
#
#     Parameters
#     ----------
#     im :  `numpy.ndarray`
#         Input array
#
#     uv : `numpy.ndarray`
#         Array of u, v coordinates where visibilities will be calculated
#
#     Returns
#     -------
#     vis : `numpy.ndarray`
#         The complex visibilities evaluated at the u, v coordinates given bu `uv`
#
#     """
#     m, n = im.shape
#     size = im.size
#     vis = np.zeros(size, dtype=complex)
#     xy = np.mgrid[0:m, 0:n].reshape(2, size)
#     for i in range(uv.shape[1]):
#         vis[i] = np.sum(
#             im.reshape(size) * np.exp(
#                 -2j * np.pi * (uv[0, i] * xy[0, :] / m + uv[1, i] * xy[1, :] / n)))
#
#     return vis


# def idft(vis, shape, uv):
#     """
#     Inverse discrete Fourier transform of the input array or image calculated at the given u, v \
#     coordinates
#
#     Loops over a list of x, y pixels rather than looping over x and y separately
#
#     Parameters
#     ----------
#     vis: `numpy.array`
#         The input visibilities to use
#
#     shape :  `tuple` (x, y)
#         Size of image to create
#
#     uv : `numpy.ndarray`
#         Array of u, v coordinates corresponding to the visibilities in `vis`
#
#     Returns
#     -------
#     `numpy.ndarray`
#         The inverse transform or back projection
#
#     """
#     m, n = shape
#     size = m * n
#     out = np.zeros(m * n)
#     xy = np.mgrid[0:m, 0:n].reshape(2, size)
#     for i in range(size):
#         out[i] = (1 / vis.size) * np.sum(
#             vis * np.exp(
#                 2j * np.pi * (uv[0, :] * xy[0, i] / m + uv[1, :] * xy[1, i] / n)))
#
#     return out.reshape(m, n)
