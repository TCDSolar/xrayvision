import numpy as np
from sunpy.map import Map

import astropy.units as apu

from xrayvision.transform import dft_map, idft_map
from xrayvision.visibility import Visibility

__all__ = ['get_weights', 'validate_and_expand_kwarg', 'vis_psf_image', 'vis_psf_map',
           'vis_to_image', 'vis_to_map', 'generate_header', 'image_to_vis', 'map_to_vis']

ANGLE = apu.get_physical_type('angle')


def get_weights(vis, natural=True, norm=True):
    r"""
    Return natural spatial frequency weight factor for each visibility.

    Defaults to use natural weighting scheme given by `(vis.u**2 + vis.v**2)^{1/2}`.

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


def validate_and_expand_kwarg(q, name=''):
    r"""
    Expand a scalar or array of size one to size two by repeating.

    Parameters
    ----------
    q : `astropy.units.quantity.Quantity`
        Input value
    name : `str`
        Name of the keyword

    Examples
    --------
    >>> import astropy.units as u
    >>> validate_and_expand_kwarg(1*u.cm)
        <Quantity [1., 1.] cm>
    >>> validate_and_expand_kwarg([1]*u.cm)
        <Quantity [1., 1.] cm>
    >>> validate_and_expand_kwarg([1,1]*u.cm)
        <Quantity [1., 1.] cm>
    >>> validate_and_expand_kwarg([1, 2, 3]*u.cm)  #doctest: +SKIP
        Traceback (most recent call last):
        ValueError:  argument must be scalar or an 1D array of size 1 or 2.

    """
    q = np.atleast_1d(q)
    if q.shape == (1,):
        q = np.repeat(q, 2)

    if q.shape != (2,):
        raise ValueError(f'{name} argument must be scalar or an 1D array of size 1 or 2.')

    return q


@apu.quantity_input(shape=apu.pixel, pixel_size='angle')
def vis_psf_image(vis, *, shape=(65, 65)*apu.pixel, pixel_size=2*apu.arcsec, natural=True):
    """
    Create the point spread function for given u, v point of the visibilities.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `astropy.units.quantity.Quantity`, optional
        Shape of the image, if only one value is given assume square (repeating the value).
    pixel_size : `astropy.units.quantity.Quantity`, optional
        Size of pixels, if only one value is given assume square pixels (repeating the value).
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`

    Returns
    -------
    `astropy.units.quantity.Quantity`
        Point spread function

    """
    shape = validate_and_expand_kwarg(shape, 'shape')
    pixel_size = validate_and_expand_kwarg(pixel_size, 'pixel_size')
    shape = shape.to_value(apu.pixel)
    weights = get_weights(vis, natural=natural)

    # Make sure psf is always odd so power is in exactly one pixel
    m, n = [s//2 * 2 + 1 for s in shape]
    psf_arr = idft_map(np.ones(vis.vis.shape)*vis.vis.unit, u=vis.u, v=vis.v,
                       shape=(m, n), weights=weights, pixel_size=pixel_size)
    return psf_arr


@apu.quantity_input(shape=apu.pixel, pixel_size='angle')
def vis_psf_map(vis, *, shape=(65, 65)*apu.pixel, pixel_size=2*apu.arcsec, natural=True):
    r"""
    Create a map of the point spread function for given the visibilities.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `astropy.units.quantity.Quantity`, optional
        Shape of the image, if only one value is given assume square (repeating the value).
    pixel_size : `Tuple[~astropy.units.Quantity]`, optional
        Size of pixels, if only one value is given assume square pixels (repeating the value).
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`

    Returns
    -------

    """
    shape = validate_and_expand_kwarg(shape, 'shape')
    pixel_size = validate_and_expand_kwarg(pixel_size, 'pixel_size')
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)
    psf = vis_psf_image(vis, shape=shape, pixel_size=pixel_size, natural=natural)
    return Map((psf, header))


@apu.quantity_input(shape=apu.pixel, pixel_size='angle')
def vis_to_image(vis, shape=(65, 65)*apu.pixel, pixel_size=2*apu.arcsec, natural=True):
    """
    Create an image by 'back projecting' the given visibilities onto the sky.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `~astropy.units.Quantity`
        Shape of the image, if only one value is given assume square (repeating the value).
    pixel_size : `~astropy.units.Quantity`
        Size of pixels, if only one value is given assume square pixels (repeating the value).
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`

    Returns
    -------
    `~astropy.units.Quantity`
        Back projection image

    """
    shape = validate_and_expand_kwarg(shape, 'shape')
    pixel_size = validate_and_expand_kwarg(pixel_size, 'pixel_size')
    shape = shape.to_value(apu.pixel)
    weights = get_weights(vis, natural=natural)
    bp_arr = idft_map(vis.vis, u=vis.u, v=vis.v, shape=shape,
                      weights=weights, pixel_size=pixel_size, center=vis.center)

    return bp_arr


@apu.quantity_input(shape=apu.pixel, pixel_size='angle')
def vis_to_map(vis, shape=(65, 65)*apu.pixel, pixel_size=2*apu.arcsec, natural=True):
    r"""
    Create a map by performing a back projection of inverse transform on the visibilities.

    Parameters
    ----------
    vis : `xrayvision.visibility.Visibility`
        Input visibilities
    shape : `int` (m, n)
        Shape of the image, if only one value is given assume square (repeating the value).
    pixel_size : `float` (dx, dy), optional
        Size of pixels, if only one value is given assume square pixels (repeating the value).
    natural : `boolean` optional
        Weight scheme use natural by default, uniform if `False`.

    Returns
    -------
    `sunpy.map.Map`
        Map object with the map created from the visibilities and the metadata will contain the
        offset and the pixel size

    """
    shape = validate_and_expand_kwarg(shape, 'shape')
    pixel_size = validate_and_expand_kwarg(pixel_size, 'pixel_size')
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)

    image = vis_to_image(vis, shape=shape, pixel_size=pixel_size, natural=natural)
    return Map((image, header))


def generate_header(vis, *, shape, pixel_size):
    r"""
    Generate a map head given the visibilities, pixel size and shape

    Parameters
    ----------
    vis :
    shape : `~astropy.units.Quantity`
        Shape of the image, if only one value is given assume square (repeating the value).
    pixel_size : `~astropy.units.Quantity`
        Size of pixels, if only one value is given assume square pixels (repeating the value)
    Returns
    -------

    """
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


@apu.quantity_input(center='angle', pixel_size='angle')
def image_to_vis(image, *, u, v, center=(0.0, 0.0) * apu.arcsec, pixel_size=2.0*apu.arcsec):
    r"""
    Return a Visibility created from the image and u, v sampling.

    Parameters
    ----------
    image : `numpy.ndarray`
        The 2D input image
    u : `astropy.units.Quantity`
        Array of u coordinates where the visibilities will be evaluated
    v : `astropy.units.Quantity`
        Array of v coordinates where the visibilities will be evaluated
    center : `~astropy.units.Quantity` (x, y)
        The coordinates of the center of the image.
    pixel_size : `~astropy.units.Quantity`
        Size of pixels, if only one value is given assume square pixels (repeating the value).

    Returns
    -------
    `Visibility`

        The new visibility object

    """
    pixel_size = validate_and_expand_kwarg(pixel_size, 'pixel_size')
    if not apu.get_physical_type(1/u) == ANGLE and apu.get_physical_type(1 / v) == ANGLE:
        raise ValueError('u and v must be inverse angle (e.g. 1/deg or 1/arcsec')
    vis = dft_map(image, u=u, v=v, center=center, pixel_size=pixel_size)
    return Visibility(vis, u=u, v=v, center=center)


def map_to_vis(amap, *, u, v):
    r"""
    Return a Visibility object created from the map and u, v sampling.

    Parameters
    ----------
    amap : `sunpy.map.Map`
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
    if not apu.get_physical_type(1/u) == ANGLE and apu.get_physical_type(1 / v) == ANGLE:
        raise ValueError('u and v must be inverse angle (e.g. 1/deg or 1/arcsec')
    meta = amap.meta
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

    return image_to_vis(amap.data, u=u, v=v, center=new_pos * apu.arcsec,
                        pixel_size=new_psize * apu.arcsec)
