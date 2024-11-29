import astropy.coordinates
import astropy.units as u
from astropy.wcs import WCS
from sunpy.coordinates.frameattributes import ObserverCoordinateAttribute
from sunpy.coordinates.frames import HeliographicStonyhurst, SunPyBaseCoordinateFrame
from sunpy.sun.constants import radius as _RSUN

__all__ = ["Projective"]


X_CTYPE = "PJLN"
Y_CTYPE = "PJLT"


class Projective(SunPyBaseCoordinateFrame):
    """A generic projective coordinate frame for an image taken by an arbitrary imager."""

    observer = ObserverCoordinateAttribute(HeliographicStonyhurst)
    rsun = astropy.coordinates.QuantityAttribute(default=_RSUN, unit=u.km)
    frame_specific_representation_info = {
        astropy.coordinates.SphericalRepresentation: [
            astropy.coordinates.RepresentationMapping("lon", u.arcsec),
            astropy.coordinates.RepresentationMapping("lat", u.arcsec),
            astropy.coordinates.RepresentationMapping("distance", "distance"),
        ],
        astropy.coordinates.UnitSphericalRepresentation: [
            astropy.coordinates.RepresentationMapping("lon", u.arcsec),
            astropy.coordinates.RepresentationMapping("lat", u.arcsec),
        ],
    }


def projective_wcs_to_frame(wcs):
    r"""
    This function registers the coordinate frames to their FITS-WCS coordinate
    type values in the `astropy.wcs.utils.wcs_to_celestial_frame` registry.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`

    Returns
    -------
    : `Projective`
    """
    if hasattr(wcs, "coordinate_frame"):
        return wcs.coordinate_frame

    # Not a lat,lon coordinate system bail out early
    if set(wcs.wcs.ctype) != {X_CTYPE, Y_CTYPE}:
        return None

    dateavg = wcs.wcs.dateobs

    rsun = wcs.wcs.aux.rsun_ref
    if rsun is not None:
        rsun *= u.m

    hgs_longitude = wcs.wcs.aux.hgln_obs
    hgs_latitude = wcs.wcs.aux.hglt_obs
    hgs_distance = wcs.wcs.aux.dsun_obs

    observer = HeliographicStonyhurst(
        lat=hgs_latitude * u.deg, lon=hgs_longitude * u.deg, radius=hgs_distance * u.m, obstime=dateavg, rsun=rsun
    )

    frame_args = {"obstime": dateavg, "observer": observer, "rsun": rsun}

    return Projective(**frame_args)


def projective_frame_to_wcs(frame, projection="TAN"):
    r"""
    For a given frame, this function returns the corresponding WCS object.

    It registers the WCS coordinates types from their associated frame in the
    `astropy.wcs.utils.celestial_frame_to_wcs` registry.

    Parameters
    ----------
    frame : `Projective`
    projection : `str`, optional

    Returns
    -------
    `astropy.wcs.WCS`
    """
    # Bail out early if not STIXImaging frame
    if not isinstance(frame, Projective):
        return None

    wcs = WCS(naxis=2)
    wcs.wcs.aux.rsun_ref = frame.rsun.to_value(u.m)

    # Sometimes obs_coord can be a SkyCoord, so convert down to a frame
    obs_frame = frame.observer
    if hasattr(obs_frame, "frame"):
        obs_frame = frame.observer.frame

    wcs.wcs.aux.hgln_obs = obs_frame.lon.to_value(u.deg)
    wcs.wcs.aux.hglt_obs = obs_frame.lat.to_value(u.deg)
    wcs.wcs.aux.dsun_obs = obs_frame.radius.to_value(u.m)

    wcs.wcs.dateobs = frame.obstime.utc.iso
    wcs.wcs.cunit = ["arcsec", "arcsec"]
    wcs.wcs.ctype = [X_CTYPE, Y_CTYPE]

    return wcs


# Remove once min version of sunpy has https://github.com/sunpy/sunpy/pull/7594
astropy.wcs.utils.WCS_FRAME_MAPPINGS.insert(1, [projective_wcs_to_frame])
astropy.wcs.utils.FRAME_WCS_MAPPINGS.insert(1, [projective_frame_to_wcs])

PROJECTIVE_CTYPE_TO_UCD1 = {
    "PJLT": "custom:pos.projective.lat",
    "PJLN": "custom:pos.projective.lon",
    "PJRZ": "custom:pos.projective.z",
}
astropy.wcs.wcsapi.fitswcs.CTYPE_TO_UCD1.update(PROJECTIVE_CTYPE_TO_UCD1)
