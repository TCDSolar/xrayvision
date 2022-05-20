import astropy.units as u
import numpy as np

from xrayvision.transform import idft_map, dft_map


def psf(vis, shape=(65, 65), pixel_size=2*u.arcsec):
    """
    Create the point spread function of the given visibilities

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `Tuple[~astropy.units.Quantity]`, optional
        Shape of the image, if only one value is given assume square.
    pixel_size : `Tuple[~astropy.units.Quantity]`, optional
        Size of pixels, if only one value is given assume square pixels.

    Returns
    -------
    `~astropy.units.Quantity`
        Point spread function

    """
    if  shape.size == 1:
        shape = shape.repeat(2)

    if pixel_size.size == 1:
        pixel_size = pixel_size.repeat(2)

    shape = shape.to_value(u.pixel)

    # Make sure psf is aways odd so power is in exactly one pixel
    m, n = [s//2 * 2 +1 for s in shape]
    psf = idft_map(vis.uv, np.ones(vis.vis.shape)*vis.vis.unit, shape=(m, n),
                   weights=np.ones(vis.vis.shape) / vis.vis.shape[0], pixel_size=pixel_size)
    return psf


def back_project(vis, shape=(65, 65)*u.pixel, pixel_size=2*u.arcsec):
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
    bp = idft_map(vis.uv, vis.vis, shape=shape,
                  weights=np.ones(vis.vis.shape) / vis.vis.shape[0], pixel_size=pixel_size)
    return bp
