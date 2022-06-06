import astropy.units as apu
import numpy as np
from sunpy.map import Map

from xrayvision.transform import idft_map, dft_map
from xrayvision.visibility import Visibility


def get_weights(vis, natural=True, norm=True):
    r"""
    Return natural spatial frequency weight factor for each visibility.

    Defaults to use natural weighting scheme given by `(vis.u**2 + vis.v**2)^{1/2}`

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
    weights = np.sqrt(vis.u**2 + vis.v**2).value
    if not natural:
        weights = np.ones_like(vis.vis, dtype=float)

    if norm:
        weights /= weights.sum()

    return weights


def vis_psf_image(vis, shape=(65, 65), pixel_size=2*apu.arcsec, natural=True):
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

    Returns
    -------
    `~astropy.units.Quantity`
        Point spread function

    """
    if shape.size == 1:
        shape = shape.repeat(2)

    if pixel_size.size == 1:
        pixel_size = pixel_size.repeat(2)

    shape = shape.to_value(apu.pixel)
    weights = get_weights(vis, natural=natural)

    # Make sure psf is always odd so power is in exactly one pixel
    m, n = [s//2 * 2 + 1 for s in shape]
    psf_arr = idft_map(np.ones(vis.vis.shape)*vis.vis.unit, u=vis.u, v=vis.v,
                       shape=(m, n), weights=weights, pixel_size=pixel_size)
    return psf_arr


def vis_psf_map(vis, shape=(65, 65), pixel_size=2*apu.arcsec, natural=True):
    r"""
    Create a map of the point spread function for given the visibilities.

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

    Returns
    -------

    """
    header = generate_header(vis.center, pixel_size, shape, vis)
    psf = vis_psf_image(vis, shape=shape, pixel_size=pixel_size, natural=natural)
    return Map((psf, header))


def vis_to_image(vis, shape=(33, 33)*apu.pixel, pixel_size=1*apu.arcsec, natural=True):
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

    Returns
    -------
    `~astropy.units.Quantity`
        Back projection image

    """
    if shape.size == 1:
        shape = shape.repeat(2)

    if pixel_size.size == 1:
        pixel_size = pixel_size.repeat(2)

    shape = shape.to_value(apu.pixel)
    weights = get_weights(vis, natural=natural)
    bp_arr = idft_map(vis.vis, u=vis.u, v=vis.v, shape=shape,
                      weights=weights, pixel_size=pixel_size, center=vis.center)

    return bp_arr


def vis_to_map(vis, shape=(65, 65)*apu.pixel, pixel_size=2*apu.arcsec, natural=True):
    r"""
    Create a map by performing a back projection of inverse transform on the visibilities.

    Parameters
    ----------
    shape : `int` (m, n)
        Shape of the output map in pixels
    center : `float` (x, y)
        Coordinates of the map center if given will override `self.xyoffset`
    pixel_size : `float` (dx, dy), optional
        Size of the pixels in x, y if only one give assumed same in both directions

    Returns
    -------
    `sunpy.map.Map`
        Map object with the map created from the visibilities and the meta data will contain the
        offset and the pixel size

    """
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)

    image = vis_to_image(vis, shape=shape, pixel_size=pixel_size, natural=natural)
    return Map((image, header))


def generate_header(vis, *, pixel_size, shape):
    header = {'crval1': vis.center[0, 0].value if vis.center.ndim == 2 else vis.center[0].value,
              'crval2': vis.center[0, 1].value if vis.center.ndim == 2 else vis.center[1].value,
              'cdelt1': pixel_size[0].value,
              'cdelt2': pixel_size[1].value,
              'ctype1': 'HPLN-TAN',
              'ctype2': 'HPLT-TAN',
              'naxis': 2,
              'naxis1': shape[0].value,
              'naxis2': shape[1].value,
              'cunit1': 'arcsec', 'cunit2': 'arcsec'}
    if vis.center:
        header['crval1'] = vis.center[0].value
        header['crval2'] = vis.center[1].value
    if pixel_size:
        if pixel_size.ndim == 0:
            header['cdelt1'] = pixel_size.value
            header['cdelt2'] = pixel_size.value
        elif pixel_size.ndim == 1 and pixel_size.size == 2:
            header['cdelt1'] = pixel_size[0].value
            header['cdelt2'] = pixel_size[1].value
        else:
            raise ValueError(f"pixel_size can have a length of 1 or 2 not {pixel_size.shape}")
    return header


def image_to_vis(image, *, u, v, center=(0.0, 0.0) * apu.arcsec, pixel_size=(1.0, 1.0) * apu.arcsec):
    r"""
    Return a Visibility created from the image and u, v sampling.

    Parameters
    ----------
    image : `numpy.ndarray`
        The 2D input image
    uv : `numpy.ndarray`
        Array of 2xN u, v coordinates where the visibilities will be evaluated
    center : `float` (x, y)
        The coordinates of the center of the image
    pixel_size : `float` (dx, dy)
        The pixel size in  x and y directions

    Returns
    -------
    `Visibility`

        The new visibility object

    """
    vis = dft_map(image, u=u, v=v, center=center, pixel_size=pixel_size)
    return Visibility(vis, u=u, v=v, center=center)


def map_to_vis(map, *, u, v):
    r"""
    Return a Visibility object created from the map and u, v sampling.

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map
    u : `numpy.ndarray`
        Array of u coordinates where the visibilities will be evaluated
    v : `numpy.ndarray`
        Array of v coordinates where the visibilities will be evaluated

    Returns
    -------
    `Visibility`
        The new visibility object

    """
    meta = map.meta
    new_pos = np.array([0., 0.])
    if "crval1" in meta:
        new_pos[0] = float(meta["crval1"])
    if "crval2" in meta:
        new_pos[1] = float(meta["crval2"])

    new_psize = np.array([1., 1.])
    if "cdelt1" in meta:
        new_psize[0] = float(meta["cdelt1"])
    if "cdelt2" in meta:
        new_psize[1] = float(meta["cdelt2"])

    return image_to_vis(map.data, u=u, v=v, center=new_pos * apu.arcsec,
                        pixel_size=new_psize * apu.arcsec)
