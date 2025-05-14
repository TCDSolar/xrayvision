"""
Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT) related functions.

There are two implementations one a standard DFT `dft` and IDFT `idft` in terms of pixel space, i.e.
the input has no positional information other than an arbitrary 0 origin and a length. The second
takes inputs which have positional information `dft_map` and the inverse `idft_map`

"""

import astropy.units as apu
import numpy as np
import numpy.typing as npt
from astropy.units import Quantity
from astropy.units.core import UnitsError

__all__ = ["generate_xy", "generate_uv", "dft_map", "idft_map"]


@apu.quantity_input()
def generate_xy(
    number_pixels: Quantity[apu.pix],
    *,
    phase_center: Quantity[apu.arcsec] | None = 0.0 * apu.arcsec,
    pixel_size: Quantity[apu.arcsec / apu.pix] | None = 1.0 * apu.arcsec / apu.pix,
) -> Quantity[apu.arcsec]:
    """
    Generate the x or y coordinates given the number of pixels, phase_center and pixel size.

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    phase_center : `float`, optional
        Center coordinates
    pixel_size : `float`, optional
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    :
        The generated x, y coordinates

    See Also
    --------
    generate_uv:
        Generates corresponding coordinates but in Fourier or u, v space.

    Examples
    --------
    >>> import astropy.units as apu
    >>> generate_xy(9*apu.pix)
    <Quantity [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.] arcsec>

    >>> generate_xy(9*apu.pix, pixel_size=2.5 * apu.arcsec/apu.pix)
    <Quantity [-10. ,  -7.5,  -5. ,  -2.5,   0. ,   2.5,   5. ,   7.5,  10. ] arcsec>

    >>> generate_xy(9*apu.pix, phase_center=10 * apu.arcsec, pixel_size=2.5 * apu.arcsec/apu.pix)
    <Quantity [ 0. ,  2.5,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. ] arcsec>

    """
    x = (
        np.arange(number_pixels.to_value(apu.pixel)) - (number_pixels.to_value(apu.pix) / 2) + 0.5
    ) * apu.pix * pixel_size + phase_center
    return x


@apu.quantity_input()
def generate_uv(
    number_pixels: Quantity[apu.pix],
    *,
    phase_center: Quantity[apu.arcsec] | None = 0.0 * apu.arcsec,
    pixel_size: Quantity[apu.arcsec / apu.pix] | None = 1.0 * apu.arcsec / apu.pix,
) -> Quantity[1 / apu.arcsec]:
    """
    Generate the u or v coordinates given the number of pixels, phase_center and pixel size.

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    phase_center : `float`, optional
        Center coordinates
    pixel_size : `float`, optional
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    :
        The generated u, v coordinates.

    See Also
    --------
    generate_xy:
        Generates corresponding coordinates but in Fourier or u, v space.

    Examples
    --------
    >>> import astropy.units as apu
    >>> generate_uv(9*apu.pix)
    <Quantity [-0.44444444, -0.33333333, -0.22222222, -0.11111111,  0.        ,
                0.11111111,  0.22222222,  0.33333333,  0.44444444] 1 / arcsec>

    >>> generate_uv(9*apu.pix, pixel_size=2.5 * apu.arcsec/apu.pix)
    <Quantity [-0.17777778, -0.13333333, -0.08888889, -0.04444444,  0.        ,
                0.04444444,  0.08888889,  0.13333333,  0.17777778] 1 / arcsec>

    >>> generate_uv(9*apu.pix, phase_center=10 * apu.arcsec, pixel_size=2.5 * apu.arcsec/apu.pix)
    <Quantity [-0.07777778, -0.03333333,  0.01111111,  0.05555556,  0.1       ,
                0.14444444,  0.18888889,  0.23333333,  0.27777778] 1 / arcsec>

    """
    # x = (np.arange(number_pixels) - number_pixels / 2 + 0.5) / (pixel_size * number_pixels)

    x = (np.arange(number_pixels.to_value(apu.pixel)) - (number_pixels.to_value(apu.pix) / 2) + 0.5) / (
        pixel_size * number_pixels
    )

    if phase_center.value != 0.0:  # type: ignore
        x += 1 / phase_center  # type: ignore
    return x


@apu.quantity_input()
def dft_map(
    input_array: Quantity | npt.NDArray,
    *,
    u: Quantity[1 / apu.arcsec],
    v: Quantity[1 / apu.arcsec],
    phase_center: Quantity[apu.arcsec] = (0.0, 0.0) * apu.arcsec,
    pixel_size: Quantity[apu.arcsec / apu.pix] = (1.0, 1.0) * apu.arcsec / apu.pix,
) -> Quantity | npt.NDArray:
    r"""
    Discrete Fourier transform in terms of coordinates returning 1-D array complex visibilities.

    Parameters
    ----------
    input_array :
        Input array to be transformed should be 2D (m, n)
    u :
        Array of 2xN u coordinates where the visibilities are evaluated.
    v :
        Array of 2xN v coordinates where the visibilities are evaluated.
    phase_center :
        Coordinates of the phase_center of the map e.g. ``(0,0)`` or ``[5.0, -2.0]``.
    pixel_size : `float` (dx, dy), optional
        The pixel size need not be square e.g. ``(1, 3)``.

    Returns
    -------
    :
        Array of N `complex` visibilities evaluated at the given `u`, `v` coordinates.

    """
    m, n = input_array.shape * apu.pix
    # python array index in row, column hence y, x
    y = generate_xy(m, phase_center=phase_center[0], pixel_size=pixel_size[0])  # type: ignore
    x = generate_xy(n, phase_center=phase_center[1], pixel_size=pixel_size[1])  # type: ignore

    x, y = np.meshgrid(x, y)
    uv = np.vstack([u, v])
    # Check units are correct for exp need to be dimensionless and then remove units for speed
    if (uv[0, :] * x[0, 0]).unit == apu.dimensionless_unscaled and (
        uv[1, :] * y[0, 0]
    ).unit == apu.dimensionless_unscaled:
        uv = uv.value  # type: ignore
        x = x.value
        y = y.value

        vis = np.sum(
            input_array[..., np.newaxis]
            * np.exp(
                2j * np.pi * (x[..., np.newaxis] * uv[np.newaxis, 0, :] + y[..., np.newaxis] * uv[np.newaxis, 1, :])
            ),
            axis=(0, 1),
        )

        return vis
    else:
        raise UnitsError("Incompatible units on uv {uv.unit} should cancel with xy to leave a dimensionless quantity")


@apu.quantity_input
def idft_map(
    input_vis: Quantity | npt.NDArray,
    *,
    u: Quantity[1 / apu.arcsec],
    v: Quantity[1 / apu.arcsec],
    shape: Quantity[apu.pix],
    weights: npt.NDArray | None = None,
    phase_center: Quantity[apu.arcsec] = (0.0, 0.0) * apu.arcsec,
    pixel_size: Quantity[apu.arcsec / apu.pix] = (1.0, 1.0) * apu.arcsec / apu.pix,
) -> Quantity | npt.NDArray:
    r"""
    Inverse discrete Fourier transform in terms of coordinates returning a 2D real array or image.

    Parameters
    ----------
    input_vis :
        Input array of N complex visibilities to be transformed to a 2D array.
    u :
        Array of N u coordinates corresponding to the input visibilities in `input_vis`
    v :
        Array of N v coordinates corresponding to the input visibilities in `input_vis`
    shape :
        The shape of the output array to create.
    weights :
        Array of weights for visibilities.
    phase_center :
        Coordinates of the phase_center of the map e.g. ``(0,0)`` or ``[5.0, -2.0]``.
    pixel_size :
        The pixel size this need not be square e.g. ``(1, 3)``.

    Returns
    -------
    :
        2D image obtained from the visibilities evaluated at the given `u`, `v` coordinates.

    """
    m, n = shape
    # python array index in row, column hence y, x
    y = generate_xy(m, phase_center=phase_center[0], pixel_size=pixel_size[0])  # type: ignore
    x = generate_xy(n, phase_center=phase_center[1], pixel_size=pixel_size[1])  # type: ignore

    x, y = np.meshgrid(x, y)

    if weights is None:
        weights = np.ones(input_vis.shape)
    uv = np.vstack([u, v])
    # Check units are correct for exp need to be dimensionless and then remove units for speed
    if (uv[0, :] * x[0, 0]).unit == apu.dimensionless_unscaled and (
        uv[1, :] * y[0, 0]
    ).unit == apu.dimensionless_unscaled:
        uv = uv.value  # type: ignore
        x = x.value
        y = y.value

        image = np.sum(
            input_vis
            * weights
            * np.exp(
                -2j * np.pi * (x[..., np.newaxis] * uv[np.newaxis, 0, :] + y[..., np.newaxis] * uv[np.newaxis, 1, :])
            ),
            axis=2,
        )

        return np.real(image)
    else:
        raise UnitsError("Incompatible units on uv {uv.unit} should cancel with xy to leave a dimensionless quantity")


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
#         The complex visibilities evaluated at the u, v coordinates given by `uv`
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
