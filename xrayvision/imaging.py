from typing import Optional

import astropy.units as apu
import numpy as np
from astropy.units import Quantity
from sunpy.map import GenericMap, Map

from xrayvision.transform import dft_map, idft_map
from xrayvision.visibility import Visibilities

__all__ = [
    "get_weights",
    "validate_and_expand_kwarg",
    "vis_psf_image",
    "vis_psf_map",
    "vis_to_image",
    "vis_to_map",
    "generate_header",
    "image_to_vis",
    "map_to_vis",
]

ANGLE = apu.get_physical_type(apu.deg)
WEIGHT_SCHEMES = ("natural", "uniform")


def get_weights(vis: Visibilities, scheme: str = "natural", norm: bool = True) -> np.ndarray:
    r"""
    Return spatial frequency weight factors for each visibility.

    Defaults to use natural weighting scheme given by `(vis.u**2 + vis.v**2)^{1/2}`.

    Parameters
    ----------
    vis :
        Input visibilities
    scheme :
        Weighting scheme to use, defaults to natural
    norm :
        Normalise the weighs before returning, defaults to True.

    Returns
    -------

    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"Invalid weighting scheme {scheme}, must be one of: {WEIGHT_SCHEMES}")
    weights = np.sqrt(vis.u**2 + vis.v**2).value
    if scheme == "natural":
        weights = np.ones_like(vis.visibilities, dtype=float)

    if norm:
        weights /= weights.sum()

    return weights


@apu.quantity_input()
def validate_and_expand_kwarg(q: Quantity, name: Optional[str] = "") -> Quantity:
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
        raise ValueError(f"{name} argument must be scalar or an 1D array of size 1 or 2.")

    return q


@apu.quantity_input
def image_to_vis(
    image: Quantity,
    *,
    u: Quantity[apu.arcsec**-1],
    v: Quantity[apu.arcsec**-1],
    phase_center: Optional[Quantity[apu.arcsec]] = (0.0, 0.0) * apu.arcsec,
    pixel_size: Optional[Quantity[apu.arcsec / apu.pix]] = 1.0 * apu.arcsec / apu.pix,
) -> Visibilities:
    r"""
    Return a Visibilities object created from the image and u, v sampling.

    Parameters
    ----------
    image :
        The 2D input image
    u :
        Array of u coordinates where the visibilities will be evaluated
    v :
        Array of v coordinates where the visibilities will be evaluated
    phase_center :
        The coordinates the phase_center.
    pixel_size :
        Size of pixels, if only one value is passed, assume square pixels (repeating the value).

    Returns
    -------
    :
        The new Visibilities object

    """
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    if not (apu.get_physical_type((1 / u).unit) == ANGLE and apu.get_physical_type((1 / v).unit) == ANGLE):
        raise ValueError("u and v must be inverse angle (e.g. 1/deg or 1/arcsec")
    vis = dft_map(
        image, u=u, v=v, phase_center=[0.0, 0.0] * apu.arcsec, pixel_size=pixel_size
    )  # TODO: adapt to generic map center
    return Visibilities(vis, u=u, v=v, phase_center=phase_center)


@apu.quantity_input()
def vis_to_image(
    vis: Visibilities,
    shape: Quantity[apu.pix] = (65, 65) * apu.pixel,
    pixel_size: Optional[Quantity[apu.arcsec / apu.pix]] = 1 * apu.arcsec / apu.pix,
    scheme: str = "natural",
) -> Quantity:
    """
    Create an image by 'back projecting' the given visibilities onto the sky.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of the image, if only one value is passed assume square (repeating the value).
    pixel_size :
        Size of pixels, if only one value is passed assume square pixels (repeating the value).
    scheme :
        Weight scheme natural by default.

    Returns
    -------
    `~astropy.units.Quantity`
        Back projection image

    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    shape = shape.to(apu.pixel)
    weights = get_weights(vis, scheme=scheme)
    bp_arr = idft_map(
        vis.visibilities,
        u=vis.u,
        v=vis.v,
        shape=shape,
        weights=weights,
        pixel_size=pixel_size,
        phase_center=[0.0, 0.0] * apu.arcsec,  # TODO update to have generic image center
    )

    return bp_arr


@apu.quantity_input
def vis_psf_map(
    vis: Visibilities,
    *,
    shape: Quantity[apu.pix] = (65, 65) * apu.pixel,
    pixel_size: Optional[Quantity[apu.arcsec / apu.pix]] = 1 * apu.arcsec / apu.pix,
    scheme: Optional[str] = "natural",
) -> GenericMap:
    r"""
    Create a map of the point spread function for given the visibilities.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of the image, if only one value is passed, assume square (repeating the value).
    pixel_size :
        Size of pixels, if only one value is passed, assume square pixels (repeating the value).
    scheme :
        Weight scheme to use natural by default.

    Returns
    -------
    :
        Map of the point spread function
    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)
    psf = vis_psf_image(vis, shape=shape, pixel_size=pixel_size, scheme=scheme)
    return Map((psf, header))


@apu.quantity_input()
def vis_psf_image(
    vis: Visibilities,
    *,
    shape: Quantity[apu.pix] = (65, 65) * apu.pixel,
    pixel_size: Quantity[apu.arcsec / apu.pix] = 1 * apu.arcsec / apu.pix,
    scheme: str = "natural",
) -> Quantity:
    """
    Create the point spread function for given u, v point of the visibilities.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of the image, if only one value passed, assume square (repeating the value).
    pixel_size :
        Size of pixels, if only one value is passed, assume square pixels (repeating the value).
    scheme :
        Weight scheme, natural by default.

    Returns
    -------
    :
        Point spread function

    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    shape = shape.to(apu.pixel)
    weights = get_weights(vis, scheme=scheme)

    # Make sure psf is always odd so power is in exactly one pixel
    shape = [s // 2 * 2 + 1 for s in shape.to_value(apu.pix)] * shape.unit
    psf_arr = idft_map(
        np.ones(vis.visibilities.shape) * vis.visibilities.unit,
        u=vis.u,
        v=vis.v,
        shape=shape,
        weights=weights,
        pixel_size=pixel_size,
    )
    return psf_arr


@apu.quantity_input()
def vis_to_map(
    vis: Visibilities,
    shape: Quantity[apu.pix] = (65, 65) * apu.pixel,
    pixel_size: Optional[Quantity[apu.arcsec / apu.pix]] = 1 * apu.arcsec / apu.pixel,
    scheme: Optional[str] = "natural",
) -> GenericMap:
    r"""
    Create a map by performing a back projection of inverse transform on the visibilities.

    Parameters
    ----------
    vis :
        Input visibilities
    shape :
        Shape of the image, if only one value is passed, assume square (repeating the value).
    pixel_size :
        Size of pixels, if only one value is passed, assume square pixels (repeating the value).
    scheme :
        Weight scheme natural by default.

    Returns
    -------
    :
        Map object with the map created from the visibilities and the metadata will contain the offset and the pixel size

    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)

    image = vis_to_image(vis, shape=shape, pixel_size=pixel_size, scheme=scheme)
    return Map((image, header))


@apu.quantity_input()
def generate_header(vis: Visibilities, *, shape: Quantity[apu.pix], pixel_size: Quantity[apu.arcsec / apu.pix]) -> dict:
    r"""
    Generate a map head given the visibilities, pixel size and shape

    Parameters
    ----------
    vis :
        Input visibilities.
    shape :
        Shape of the image, if only one value is passed assume square (repeating the value).
    pixel_size :
        Size of pixels, if only one value is passed assume square pixels (repeating the value)

    Returns
    -------
    :
    """
    header = {
        "crval1": (vis.phase_center[1]).to_value(apu.arcsec),
        "crval2": (vis.phase_center[0]).to_value(apu.arcsec),
        "cdelt1": (pixel_size[1] * apu.pix).to_value(apu.arcsec),
        "cdelt2": (pixel_size[0] * apu.pix).to_value(apu.arcsec),
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "naxis": 2,
        "naxis1": shape[1].value,
        "naxis2": shape[0].value,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
    }
    return header


@apu.quantity_input()
def map_to_vis(amap: GenericMap, *, u: Quantity[1 / apu.arcsec], v: Quantity[1 / apu.arcsec]) -> Visibilities:
    r"""
    Return a Visibilities object created from the map, sampling it at give `u`, `v` coordinates.

    Parameters
    ----------
    amap :
        Input map.
    u :
        Array of u coordinates where the visibilities will be evaluated
    v :
        Array of v coordinates where the visibilities will be evaluated

    Returns
    -------
    :
        The new Visibilities object

    """
    if not apu.get_physical_type(1 / u) == ANGLE and apu.get_physical_type(1 / v) == ANGLE:
        raise ValueError("u and v must be inverse angle (e.g. 1/deg or 1/arcsec")
    meta = amap.meta
    new_pos = np.array([0.0, 0.0])
    if "crval1" in meta:
        new_pos[1] = float(meta["crval1"])
    if "crval2" in meta:
        new_pos[0] = float(meta["crval2"])

    new_psize = np.array([1.0, 1.0])
    if "cdelt1" in meta:
        new_psize[1] = float(meta["cdelt1"])
    if "cdelt2" in meta:
        new_psize[0] = float(meta["cdelt2"])

    vis = image_to_vis(
        amap.quantity, u=u, v=v, pixel_size=new_psize * apu.arcsec / apu.pix, phase_center=new_pos * apu.arcsec
    )

    return vis
