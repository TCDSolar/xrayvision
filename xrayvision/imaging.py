import astropy.units as u
import numpy as np

from xrayvision.transform import idft_map


def get_weights(vis, natural=True, norm=True):
    r"""
    Return natural spatial frequency weight factor for each visibility.

    Defaults to use natural weighting scheme given by $(vis.u^2 + vis.v^2)^{1/2}$

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    natural : `boolean` optional
        Use natural or uniform weighting scheme
    norm : `boolean`
        Normalise the weighs before returning

    Returns
    -------
    `weights`

    """
    if natural:
        weights = np.sqrt(vis.uv[0, :] ** 2 + vis.uv[1, :] ** 2).value
    else:
        weights = np.ones_like(vis.vis, dtype=float)

    if norm:
        weights /= weights.sum()

    return weights


def psf(vis, shape=(65, 65), pixel_size=2*u.arcsec, natural=True, map=True):
    """
    Create the point spread function for given u, v point of the visibilities.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `Tuple[~astropy.units.Quantity]`, optional
        Shape of the image, if only one value is given assume square.
    pixel_size : `Tuple[~astropy.units.Quantity]`, optional
        Size of pixels, if only one value is given assume square pixels.
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`
    map : `boolean` optional
        Return an `sunpy.map.Map` by default or data only if `False`

    Returns
    -------
    `~astropy.units.Quantity`
        Point spread function

    """
    if shape.size == 1:
        shape = shape.repeat(2)

    if pixel_size.size == 1:
        pixel_size = pixel_size.repeat(2)

    shape = shape.to_value(u.pixel)
    weights = get_weights(vis, natural=natural)

    # Make sure psf is always odd so power is in exactly one pixel
    m, n = [s//2 * 2 + 1 for s in shape]
    psf_arr = idft_map(vis.uv, np.ones(vis.vis.shape)*vis.vis.unit, shape=(m, n),
                       weights=weights, pixel_size=pixel_size)

    if not map:
        return psf_arr

    psf_map = vis.to_map(shape=shape, pixel_size=pixel_size)
    psf_map.data[:] = psf_arr
    return psf_map


def back_project(vis, shape=(65, 65)*u.pixel, pixel_size=2*u.arcsec, natural=True, map=True):
    """
    Create an image by 'back projecting' the given visibilities onto the sky.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `~astropy.units.Quantity`
        Shape of the image, if only one value is given assume square.
    pixel_size : `~astropy.units.Quantity`
        Size of pixels, if only one value is given assume square pixels.
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`
    map : `boolean` optional
        Return an `sunpy.map.Map` by default or data only if `False`

    Returns
    -------
    `~astropy.units.Quantity`
        Back projection image

    """
    if shape.size == 1:
        shape = shape.repeat(2)

    if pixel_size.size == 1:
        pixel_size = pixel_size.repeat(2)

    shape = shape.to_value(u.pixel)
    weights = get_weights(vis, natural=natural)
    bp_arr = idft_map(vis.uv, vis.vis, shape=shape, weights=weights, pixel_size=pixel_size)

    if not map:
        return bp_arr

    psf_map = vis.to_map(shape=shape, pixel_size=pixel_size)
    psf_map.data[:] = bp_arr
    return psf_map
